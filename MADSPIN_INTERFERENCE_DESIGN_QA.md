# Pure-interference mode: three design questions

Base: `bdf383554`, tip of `claude/ms-pure-interference` (PR #351).
Companion documents: `MADSPIN_SEQUENTIAL_PLAN.md` section 13 (the design record)
and `MadSpin/validation/interference_closure/RESULTS.md` on
`claude/ms-interference-closure` (the closure validation, where issues 1 and 2
were raised).

**Nothing here is implemented.** This is an evidence-backed answer to three
questions plus a recommendation for each.

---

## Question 1 -- the normalisation and the banner

### 1.1 What is written to the LHE `<event>` weight today

The chain is short and there is no other factor anywhere:

* `interface_madspin.py:3895` -- `br = self.branching_ratio * wsign if pure_interference else self.branching_ratio`
* `interface_madspin.py:3902-3905` -- `full_evt.wgt *= br` and `wgts[key] *= br`
  for every entry of the multi-weight `<rwgt>` block
* `wsign` is set at `interface_madspin.py:3824` from the sign of the
  convolution, `wsign = -1.0 if (wgt*jac) < 0 else 1.0`

`full_evt.wgt` before that line is the *parent* event's weight, untouched.

**The parent weight is the total cross-section, not sigma/N.** MG5 writes LHE
files with `IDWTUP = -4`, the convention in which the cross-section is the
*average* of the event weights. Measured directly on three sample files in the
tree:

| file | `<init>` XSECUP sum | XWGTUP per event | N |
|---|---|---|---|
| `tests/input_files/ttbar.lhe.gz` | 504.328 | 504.328 (all identical) | 25 |
| `tests/input_files/wj_zj.lhe.gz` | 2324.35 + 389.595 = 2713.945 | 2713.945 | 5000 |
| `tests/input_files/hj_heft.lhe.gz` | 13.987564 | 13.987564 | 25 |

So for an unweighted MadSpin sample every written event carries

    |w| = sigma_parent * BR        (the same number for every event)

which is exactly the 0.79865722 the plan's section 13.12 measured, and exactly
the *unpolarised* `sigma * BR` of the same run (relative difference 0.000e+00).
**The previous measurement is confirmed**, and the reason it is that number and
not `sigma*BR/N` is the `IDWTUP=-4` mean convention, not anything MadSpin does.

One correction to a premise that has been circulating: in a normal MG5
unweighted sample it is **not** true that `sigma = sum(w)`. It is
`sigma = mean(w) = sum(w)/N`. That matters below.

### 1.2 Can a user recover the size of the interference from the LHE file alone?

**No -- and the reason is sharper than "sum(w) is zero".**

Under the mean convention the sample's own cross-section is `mean(w) = 0`, which
is *correct*: this sample really has no rate. The physical content of the sample
is not its rate but its contribution to differential distributions, and that
contribution is carried by the **acceptance rate against the bound**, not by the
weights.

The algebra. For a production event `p`, write `W(Omega)` for the signed
convolution `wgt*jac` at decay configuration `Omega`, and

    c = <W>_Omega        (the mean over the decay phase space of the
                          *unrestricted* convolution -- a decay-side constant,
                          the same for every production event; that constancy
                          is what makes ordinary redraw-until-accept unbiased,
                          plan section 13.7b)

*Ordinary mode* (redraw until accept): one output event per production event,
distributed in `Omega` proportionally to `W`, weight `sigma*BR`. For any
observable `O`,

    S_ord(O) = N_read * sigma*BR * <W O> / c                       (1)

*Interference mode* (`interface_madspin.py:3839`, `3872-3877`): one draw,
accepted with probability `|W|/maxwgt`, nothing written on rejection, weight
`+- sigma*BR`. Then

    S_int(O) = sum_kept sign(W) sigma*BR O
             = sigma*BR * sum_p E[ sign(W) O |W|/maxwgt ]
             = N_read * sigma*BR * <W O> / maxwgt                  (2)

Dividing (1) by (2): the two differ by the single factor

    **maxwgt / c**

and `c = eff_joint * maxwgt`, where `eff_joint` is the unweighting efficiency of
an *ordinary* joint run on the same decay chain -- so the factor is simply
`1 / eff_joint`. That is precisely the recipe the closure validation used, and
it is in its analysis script verbatim
(`MadSpin/validation/interference_closure/analyse_interference.py:329`):

    m['unit'] = (m['ref'] / m['n_read']) * (m['maxwgt'] / c_meas)   # interference
    m['unit'] = m['xsec'] / m['n_written']                          # diagonal

with `c_meas = maxwgt_unpol * n_written/n_trials` (line 319-320) and
`w = unit * sign(w_LHE)` (line 341) -- **the LHE weight's magnitude is thrown
away and only its sign is used.** Numbers from `data/meta.json`:
`eff = 0.323725`, `maxwgt_unpol = 6.937097e-10`, `c = 2.245713e-10`; the `x_t`
sample has `maxwgt = 3.758035e-10`, and
`(23.763645/50000) * (3.758035e-10/2.245713e-10) = 7.953e-4` = the stored
`unit = 7.9497e-4`. The algebra above reproduces the validated normalisation
exactly.

**Consequence, stated plainly:** the sample's physical normalisation depends on
`max_weight`, an internal accept/reject bound that depends on `nb_sigma`,
`Nevents_for_max_weight` and the process. Today that number is written nowhere
in the output (the closure test had to add a `logger.info` to
`get_maxwgt_for_onshell` to see it, `RESULTS.md` section 4), and neither is `c`.
So **the file as written today does not determine its own normalisation to
within an arbitrary factor.** The validation's "max_weight belongs in the
banner" is right, and it is actually understated: `c` is missing too.

`c` is universal in practice: the closure measured it on three different
production processes whose maximum weights differ by a factor two and got the
same value to 1% (`RESULTS.md` section 4). It is a *decay-side* constant because
the production density matrix cancels between the restricted contraction and its
normalising trace.

Side note worth one check before anything is built: `c = 2.2457e-10 +- 0.45%`
and `1/(m_t Gamma_t)^4 = 2.256e-10` for the default SM card (`m_t = 173.0`,
`Gamma_t = 1.4915`) -- i.e. `c` looks like exactly `1/prod_denominators`, the
Breit-Wigner constant `calculate_matrix_element_from_density` divides by
(`interface_madspin.py:8000-8001`, `D = complex(0, mass*width); prod_denominators *= D*D.conjugate()`).
Agreement is 0.45%, one sigma of the measurement. If that holds, `c` is
*analytic* and free. It should be verified, not assumed -- see the robust route
in 1.3.

### 1.3 Is the user's proposal workable?

> the lhef file should have events with weights +max_weight and -max_weight so
> it does not need a dedicated banner block. Maybe this is not max_weight but
> the weight value should give the normalization.

**The instinct is right; the literal value is wrong.** `max_weight` is a
matrix-element ratio in internal units (`6.94e-10` for the `t t~` run) -- it is
not a cross-section and writing it into `XWGTUP` would be wrong by ten orders of
magnitude and dimensionally meaningless.

There are two self-consistent correct values, and they differ by which event
count the consumer divides by.

**(A) Per-read-event, constant magnitude** -- keeps the current accept/reject:

    w = +- sigma_parent * BR * maxwgt / c        ( = +- sigma*BR / eff_joint )

* Constant per run: `maxwgt` is a single scalar for the joint scheme
  (`get_maxwgt_for_onshell`, `interface_madspin.py:4664-4720`) and `c` is a
  decay-side constant. Every event carries the same magnitude, only the sign
  varies -- exactly the shape of the user's proposal.
* Known *before* the event loop, so it is fully streaming-compatible.
* But the consumer must divide by `N_read`, not by the number of events in the
  file (3-9% of it). So the file is still not self-describing: `N_read` has to
  come from somewhere. It is already in the banner note
  (`interface_madspin.py:4088`, `'Events written / read : %d / %d'`).

**(B) Fully weighted, no accept/reject at all** -- the cleaner option:

    keep every production event, write  w = sigma_parent * BR * W / c

* `mean(w) = sigma*BR*<W>_int/c = 0` -- the sample's cross-section, correct, and
  consistent with `XSECUP = 0`.
* `sum_bin(w)/N_file` is the interference contribution to that bin, in pb, with
  `N_file = N_read`. **The file is genuinely self-normalising in MG5's own
  `IDWTUP=-4` convention, and needs no banner block at all** for normalisation.
* `max_weight` disappears from the normalisation entirely -- which dissolves
  issue 1 at the root rather than papering over it. No overweight events, no
  `nb_pi_overflow` bias channel, and the `z` test loses its most likely failure
  mode.
* Uses all `N` events instead of 3-9%, so the statistics per production event
  are strictly better (the plan's 13.7b concern was that redraw-until-accept
  *normalises away* the local interference size `<|W|>`; carrying it in the
  weight preserves it just as well as carrying it in the keep rate, with less
  variance).
* Cost: the output is a weighted sample. Tools that assume unit weights break --
  but they break on a signed zero-cross-section sample anyway.

**How to get `c` at run time.** Both options need it. The robust route needs no
new physics and no new probe: `_polarization_ratios`
(`interface_madspin.py:6150-6166`) already demonstrates the trick -- save
`density_prod.hel_restriction`, swap in another restriction, contract the *same*
matrices again, restore. Doing that once per trial inside the existing maximum
weight probe (`_joint_maxwgt_range`, `interface_madspin.py:4812-4866`) with the
restriction set to the *trace* restriction gives `<W_full>` = `c` for free, to
the probe's own statistics (`Nevents_for_max_weight * max_weight_ps_point`
trials, typically 75 x 500 = 37500, i.e. sub-percent). The analytic
`1/prod_denominators` shortcut, if it checks out, is a cross-check on that, not
a substitute.

**What breaks downstream, item by item**

| item | effect |
|---|---|
| `<init>` XSECUP / XERRUP / XMAXUP | already zeroed today (`_report_pure_interference` -> `_rewrite_lhe_banner_cross(base_out, 0.0, ...)`, `interface_madspin.py:4093`). Unchanged by either option. |
| Pythia / showering | already impossible (`XSECUP = 0`); the code warns about it (`interface_madspin.py:5816-5826`). Neither option makes it worse; neither fixes it. The existing `interference_init_cross measured\|zero\|reference` idea from plan 13.7c is the fix for that, orthogonal to this. |
| `sum(w)`-based tooling | `sum(w) ~ 0` either way. Today's magnitude is an active decoy: it reads `sigma*BR`, the *unpolarised* cross-section, which is wrong by `maxwgt/c` (3.09x under (A)) or by `<|W|>/c` (~4x the other way under (B)) and the factor is run-dependent. Both options make the magnitude mean something. |
| multi-weight `<rwgt>` entries | scaled by the same `br` (`interface_madspin.py:3904-3905`), so they follow automatically under either option. |
| `keep_weight_for_polarization_*` | **already broken in this mode, independently of this question.** `_polarization_ratios` computes `restricted/full` with `full = me` = the *interference* contraction (`interface_madspin.py:8009` and `8017`), while the numerators are ordinary symmetric diagonal blocks. The denominator is a signed quantity that passes through zero, so the ratios can be arbitrarily large and sign-flipping, and `_add_polarization_weights` then writes `evt.wgt * ratio` (`interface_madspin.py:6293`). These weights do not mean "the C-polarised fraction of this event" in this mode. Nothing validates the combination. Either refuse the two options together, or define the denominator to be the unrestricted contraction. This is a third issue, smaller than the two known ones but real. |

### 1.4 Recommendation for Question 1

**(c) both, with option (B) for the weight.**

1. Write a *physically meaningful* event weight. Prefer **(B)**, the fully
   weighted sample `w = sigma*BR*W/c`: it makes the LHE self-normalising under
   MG5's own convention, removes `max_weight` from the normalisation completely,
   and improves the statistics. If keeping the unweighted-magnitude character of
   the sample matters more, **(A)** with `w = +- sigma*BR*maxwgt/c` is the
   correct constant, and the user's proposal is then exactly right in form.
2. **Keep the banner block anyway.** Not for the normalisation -- (B) does not
   need it -- but because `XSECUP = 0` deletes the reference cross-section from
   the file, and because the diagnostics (`S`, `sqrt(sum w^2)`, `z`, kept/read,
   overweight count) have nowhere else to live. It costs nothing; it already
   exists; the closure's analysis script already reads `ref` and `n_read` from
   it (`analyse_interference.py:305`, via `read_pi_block`).
3. Add `max_weight` and the measured `c` to that block regardless of which
   weight convention is chosen. They are the two numbers that let a user audit
   the normalisation, and they are the two numbers the closure test had to
   reconstruct by hand.

The one thing **not** to do is leave the weight at `+- sigma*BR`: it is neither
the sample's normalisation nor a harmless placeholder, it is a number that looks
like a cross-section and is wrong by a run-dependent factor of a few.

---

## Question 2 -- the syntax

### 2.1 What the syntax is today

A MadSpin-card option, declared at `interface_madspin.py:163-167`:

    set pure_interference t = 0 T

particle (name or pdg), `=` (or `:`), the **production-side** helicity set, then
the **decay-side** one, both drawn from `{0, +, R, -, L, T}`
(`_POL_TOKENS` / `_parse_pol_side`, `interface_madspin.py:5660-5672`). Several
particles are separated by `;` (`_pure_interference`, `interface_madspin.py:5705`).
A complete working card, from the closure test:

    set spinmode onshell
    set BW_cut 15
    set pure_interference t = + -
    decay t > b w+, w+ > l+ vl
    decay t~ > b~ w-, w- > l- vl~

Parsing produces `pdg -> (P, D)`; validation (`_validate_pure_interference`,
`interface_madspin.py:5760-5826`) requires a density spinmode, disjoint `P`/`D`,
a pdg that something actually decays, and a production process not braced away
from either side. Setting the option is what turns the mode on; there is no
separate boolean.

### 2.2 Why it ended up a card option -- the claim verified

The design record (plan 13.6) says `{0}{T}` is not grammar `extract_process`
accepts. **Verified, and it is a hard rejection today**, at
`madgraph_interface.py:5126` / `5151-5152`:

    part_name, pol = part_name.split('{',1)
    pol, rest = pol.split('}',1)
    ...
    if rest:
        raise self.InvalidCmd('A space is required after the "}" symbol to separate particles')

`t{0}{T}` leaves `rest = '{T}'` and raises -- with a misleading message about
spaces.

Is it *genuinely* blocked or merely inconvenient? The parser change itself is
small (loop over brace groups). The blast radius is not: a leg carries a single
flat `Leg['polarization']` list of ints, `ProcessDefinition.check_polarization`
(`base_objects.py:3876-3906`) reasons about that flat list, and `polarization`
appears 221 times across 15 modules under `madgraph/` -- diagram generation,
`helas_objects`, `export_v4`, `group_subprocs`, `fks_common`,
`loop_base_objects`, `banner`, `systematics`, `hepmc_parser`. A second,
*semantically different* brace on the same leg has no representation in that
data model. So: **not a grammar accident that a two-line fix would clear -- a
data-model change in shared code.** The design record's "rejected" is justified.

There is a second, independent reason, which is the one that actually settles
it: see 2.3.

### 2.3 The physics point the user's mental model is missing

> I thought it was production using `{L}` and decay using `{R}` in which case
> `t t~` is easily reachable

**This mode requires the production sample to be unpolarised, so a `{L}` on the
production process is the opposite of what it needs.** Verified in the code and
in the physics:

* `p p > t{L} t~` generates only the L amplitude. Its events are distributed as
  `|M_L|^2`. The interference between L and R is simply **absent** from the
  sample -- `rho_prod` has only the `(L,L)` entry -- so there is nothing to
  restrict to. Plan 13.5 states this; `_validate_pure_interference` enforces it
  (`interface_madspin.py:5798-5820`): if the banner's braces on that pdg do not
  contain `P u D`, the run is refused with an explicit "the events carry no
  amplitude for helicity ..., regenerate the production process without the
  brace on that leg".
* This is *why* both sets have to come from the MadSpin card:
  "`_production_polarization` cannot supply `P`, there is no production brace to
  inherit" (`_pure_interference` docstring, `interface_madspin.py:5680-5684`).
  The request that started this feature -- "should allow to specify
  production/decay polarization" -- is precisely that observation.
* An existing production brace is still *allowed on the other legs*, and the
  closure exploits it: `p p > t t~{+}` with
  `set pure_interference t = + -` gives the `(I, D+)` block
  (`RESULTS.md` section 3). What is refused is a brace on the leg the cross
  restriction is asked for.

**If the user means `{L}`/`{R}` as MadSpin-card notation** -- naming which
interference term to keep, never passed to MG5 generation -- then it is purely a
spelling question and the physics objection disappears. In that reading, a
`decay`-line spelling like

    decay t{-}{+} > b w+

is implementable *in MadSpin only*: `do_decay` (`interface_madspin.py:982-997`)
would have to stop refusing braces in density mode and route them into
`_pure_interference` instead of into the decay process definition, and
`self.decay.reorder_branch(decaybranch)` (line 1006) would have to be handed the
brace-free string. Feasible, and the plan already argues the carve-out would be
*principled* (the decay matrix element and the BR stay fully inclusive -- the
brace restricts one index of the convolution, not the decay ME). But the plan's
counter-argument stands and I agree with it: the same characters would then mean
"project the decay ME" under `spinmode=none`/`madspin_v1` (where `do_decay`
currently only warns, `interface_madspin.py:999-1003`) and "select an
interference block" under a density spinmode. Two meanings, one spelling,
selected by a mode flag elsewhere in the card.

A MadSpin-side *production* line (e.g. `production t{-} t~` in the MadSpin card)
would be new grammar with no existing consumer, and would look exactly like the
generation-level brace it is not. I would not.

### 2.4 The `;` truncation -- confirmed, with one correction

Confirmed by running it (`MadSpinInterface().exec_cmd(..., precmd=True)`):

    set pure_interference t = + - ; t~ = + -
      -> options['pure_interference'] == 't = + -'          # the t~ half is gone
      -> stderr: 'Command "t~" not recognized, please try again'

`extended_cmd.Cmd.precmd` (`extended_cmd.py:1032-1042`) splits **every** card
line on `;` and dispatches the pieces as separate commands; the MadSpin card is
read through `import_command_file` -> `exec_cmd(line, precmd=True)`
(`extended_cmd.py:1712-1718`, reached from
`common_run_interface.py:4359`), so the split is unavoidable.

One correction to `RESULTS.md`: it is not *completely* silent -- `Cmd.default`
(`extended_cmd.py:1614-1620`) logs `Command "t~" not recognized, please try
again`. It is a warning in a long log, the run proceeds, and the resulting
sample is a valid-looking single-particle interference sample. The severity
assessment in `RESULTS.md` stands; only the word "silently" needs softening to
"with nothing but a generic unrecognised-command warning".

Which spellings avoid it:

| spelling | works? | why |
|---|---|---|
| a different separator, e.g. `,` | **yes** | `precmd` only splits on `;`. But `_parse_pol_side` (`interface_madspin.py:5662`) already does `text.replace(',', ' ')`, so `,` is currently *inside* the polarisation vocabulary; using it as the entry separator too would need care. `|` or `&` are unambiguous. |
| quoting: `set pure_interference "t = + - ; t~ = + -"` | **no** | `precmd` does a bare `line.split(';')`, quote-unaware. |
| repeated `set pure_interference` lines | **no, as the code stands** | `do_set` does `self.options[args[0]] = ' '.join(args[1:])` (`interface_madspin.py:1065`) -- it *overwrites*. Accumulating instead is a small, local change and would be the most natural card idiom. |
| a brace spelling on `decay` lines | **yes** | one `decay` line per particle already, no separator needed at all. |

### 2.5 Recommendation for Question 2

**Keep the card option; fix the separator; do not go to braces.**

1. Make repeated `set pure_interference` lines **accumulate** rather than
   overwrite, so the multi-particle case is written as

       set pure_interference t  = + -
       set pure_interference t~ = + -

   That is the most idiomatic MadSpin-card spelling, it needs no new separator
   vocabulary, it is immune to `precmd`, and it makes the `(I,I)` block directly
   reachable. (Special-casing one option inside `do_set` is a little ugly; the
   alternative is to have `_pure_interference` merge on parse and `do_set`
   append with a separator it chooses itself.)
2. **Also** make `_pure_interference` reject an entry containing an unconsumed
   `;`-shaped fragment, or better: have `_validate_pure_interference` fail loudly
   if the option string still looks truncated. The silent-wrong-sample failure
   mode is the actual bug, and accumulating `set` lines removes only the common
   case of it.
3. Do **not** invest in a brace-based spelling. It requires either a shared
   `Leg['polarization']` data-model change (production side, 221 call sites) or
   an overload of the same characters with a second meaning (decay side). The
   card option is already validated end to end and reads unambiguously.

---

## Question 3 -- why 9 blocks and not 10?

### 3.1 The user's count is right, and so is the code's

The typo reading is confirmed: `(++;-+)` appears twice in the user's list and
the intended set is the six distinct off-diagonal hermitian pairs

    (++;+-) (++;-+) (++;--) (+-;-+) (+-;--) (-+;--)

4 diagonal + 6 pairs = **10 hermitian terms** = 4 + 2*6 = **16 matrix entries**.
The 9-block decomposition covers the same 16 entries as
`4*1 + 4*2 + 1*4` (`RESULTS.md` section 1). Both counts are correct; they count
different things.

### 3.2 The mapping (verified against `_restriction_rows`)

`DensityMatrix._restriction_rows` (`MadSpin/decay.py:5083-5106`) builds the mask
as a strict **product of per-particle conditions**:

    mask = np.ones(...)
    for k, allowed in enumerate(restriction):
        bra, ket = h[:, 2*k], h[:, 2*k+1]
        if _is_cross_restriction(allowed):
            mask &= ((isin(bra, P) & isin(ket, D)) | (isin(bra, D) & isin(ket, P)))
        else:
            mask &= isin(bra, allowed); mask &= isin(ket, allowed)

One loop iteration per decaying particle `k`, each touching only that particle's
own `(bra, ket)` columns, combined with `&=`. `normalize_hel_restriction`
(`decay.py:4934-4985`) enforces the matching shape: a sequence with **one entry
per particle**, each entry `None`, a flat set, or a `(P, D)` pair.

On a two-state basis the transposition-closed per-particle forms are exactly
`D+ = {+1}`, `D- = {-1}` and `I = ({+1},{-1})`; the mapping of the user's terms
onto the product blocks (label = `(bra1,bra2 ; ket1,ket2)`):

| user term | particle 1 | particle 2 | block |
|---|---|---|---|
| `(++;+-)` | `(+,+)` diagonal | `(+,-)` flip | `(D+, I)` |
| `(++;-+)` | `(+,-)` flip | `(+,+)` diagonal | `(I, D+)` |
| `(+-;--)` | `(+,-)` flip | `(-,-)` diagonal | `(I, D-)` |
| `(-+;--)` | `(-,-)` diagonal | `(+,-)` flip | `(D-, I)` |
| `(++;--)` | `(+,-)` flip | `(+,-)` flip | `(I, I)` |
| `(+-;-+)` | `(+,-)` flip | `(-,+)` flip | `(I, I)` |

**The coordinator's reading is correct.** Four mixed blocks carry one user term
(two matrix entries) each; `(I,I)` bundles two user terms (four entries):
`4 + 4*2 + 4 = 16`. The difference between 9 and 10 is *only* whether the two
both-flip terms are separated.

### 3.3 Can the 10 be recovered from the 9? No.

The `(I,I)` mask is `(bra1,ket1) in {(+,-),(-,+)}` **and**
`(bra2,ket2) in {(+,-),(-,+)}` -- all four combinations of the two particles'
flip directions:

    (+,-)&(+,-) -> (++;--)          (-,+)&(-,+) -> its transpose
    (+,-)&(-,+) -> (+-;-+)          (-,+)&(+,-) -> its transpose

Keeping only `(++;--)` needs `((+,-)&(+,-)) OR ((-,+)&(-,+))` -- a **union of
two products**, not a product of unions. A product mask that contains both
`(+,-)&(+,-)` and `(-,+)&(-,+)` must have `{(+,-),(-,+)}` available on *each*
particle and therefore also contains `(+,-)&(-,+)`. So:

* **no per-particle restriction can express it**, by the structure of the loop
  above;
* **no linear combination of runs can recover it either.** The nine blocks are
  the atoms of the lattice of legal (transposition-closed, hence real) product
  masks, they are pairwise disjoint and they tile the 16 entries (asserted by
  `check_blocks.py`); `(++;--)` is not a union of them, so it is not any signed
  sum of them. The subtraction trick that yields `(I,I)` --
  `(I,I) = x_t - (I,D+) - (I,D-)` -- has no analogue one level down.

What *would* make it possible: let a restriction be a **list of alternatives**
OR'd together, each alternative a per-particle product, with the requirement
that the union be closed under global transposition. Then

    (++;--) block = [ (({+},{-}), ({+},{-})) ]  u  its transpose
    (+-;-+) block = [ (({+},{-}), ({-},{+})) ]  u  its transpose

The `_restriction_rows` change is contained -- OR the per-alternative masks, keep
the `(basis_id, restriction)` cache key (a tuple of tuples is still hashable),
`trace()` still vanishes (neither alternative has a diagonal entry),
`_combine_restrictions` untouched. What is *not* contained is everything above
it: a card syntax that names the correlation between two particles' flip
directions, validation for it, and `tensor_product`'s per-particle concatenation
(`decay.py:5224`) would need to distribute over the alternatives. The plan
already refused a cousin of this ("the global form ... would need a
union-of-two-product-masks in `_restriction_row_mask`, breaking its structure",
section 13.9).

### 3.4 Are the two both-flip terms individually meaningful for `t t~`?

Yes, but not cleanly -- and this is what decides whether the bundling matters.

Writing the production spin density matrix as
`rho = 1/4 [ 1 x 1 + B+.sigma x 1 + 1 x B-.sigma + C_ij sigma_i x sigma_j ]`
in the helicity basis with axes `(x,y,z) = (r, n, k)`, a direct numerical
evaluation gives

    4 Re rho(++;--) = C_rr - C_nn
    4 Re rho(+-;-+) = C_rr + C_nn
    4 Im rho(++;--) = -(C_rn + C_nr)
    4 Im rho(+-;-+) =   C_nr - C_rn

so

    2 Re rho(+-;-+) - 2 Re rho(++;--) = C_nn      <- exactly RESULTS.md 6b
    2 Re rho(+-;-+) + 2 Re rho(++;--) = C_rr

**Neither both-flip term is a single observable**: each one is the half-sum
`(C_rr +- C_nn)/2`. `C_nn` is their *difference* and `C_rr` is their *sum*. So
even with the union-of-products generalisation you would not get a "`C_nn`-only"
sample from one run -- you would get two runs whose difference is `C_nn`.

**In practice the bundling costs nothing.** The `(I,I)` sample contracts the
whole block against the decay density matrix, which supplies the angular
structure that separates `C_nn` from `C_rr` at the *observable* level. Both
coefficients are therefore measurable from the single `(I,I)` sample, and the
closure measured both: the interference contributes `+0.03626 +- 0.00090` to
`<C_nn>` and `+0.00247 +- 0.00091` to `<C_rr>` -- from the same events
(`RESULTS.md` section 6). The whole `C_nn` effect sits in `(I,I)` and the four
singly-interfering blocks are flat at zero in it (`RESULTS.md` 6b, and
`plots/blocks_cnn.pdf`). What the bundling costs is only the ability to attribute
a given *event* to one of the two terms.

### 3.5 Recommendation for Question 3

**Answer the counting question; do not split `(I,I)`.**

* The two counts are both right: 9 factorised blocks, 10 hermitian terms, 16
  entries. The whole difference is that `(I,I)` bundles `(++;--)` and `(+-;-+)`.
* The bundling is forced by the design, not an oversight: `hel_restriction` is a
  per-particle sequence and `_restriction_rows` ANDs per-particle conditions.
  That structure is what makes the mask cacheable, `tensor_product`-composable
  and automatically transposition-closed (hence real) for any mixture of `None`,
  symmetric and cross entries.
* Splitting is possible only via a union-of-products restriction, and buys a
  distinction between `(C_rr - C_nn)/2` and `(C_rr + C_nn)/2` -- neither of which
  is an observable on its own, and both of which are already measurable from the
  single `(I,I)` sample. **Not worth it**, unless a use case appears that needs
  per-event attribution rather than per-observable separation.
* If it is ever wanted, it is a *normalised form* change in `decay.py` plus new
  card syntax naming the flip-direction correlation -- a separate feature, not a
  tweak, exactly as plan 13.9 says of its "global" cousin.

---

## Summary of what is actually broken (for a fix PR, when one is wanted)

1. **`;` truncation** (`RESULTS.md` section 2). Multi-particle
   `pure_interference` is unreachable from a card; the failure mode is a
   different, valid-looking sample with only a generic
   `Command "t~" not recognized` warning. Fix: accumulate repeated `set` lines.
2. **The normalisation is not in the file** (`RESULTS.md` section 4, and 1.2
   above). The written weight `+- sigma*BR` is not the sample's normalisation;
   the normalisation is `sigma*BR*maxwgt/c` per *read* event, and neither
   `maxwgt` nor `c` is recorded anywhere. Fix: correct the weight (preferably by
   dropping the accept/reject and writing `sigma*BR*W/c` per event), and record
   `maxwgt`, `c`, `N_read` in `<MGPureInterference>`.
3. **`keep_weight_for_polarization_*` is meaningless in this mode** (1.3 above,
   new here). `_polarization_ratios` divides diagonal blocks by the interference
   contraction, a signed quantity that passes through zero. Fix: refuse the
   combination, or use the unrestricted contraction as the denominator.
