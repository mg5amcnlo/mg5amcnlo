---
title: "Process Grouping"
nav_order: 3.5
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Process grouping

A multiprocess request such as `p p > j j j j` expands into many subprocesses,
but many of them have the same matrix element once their legs are relabelled.
In the built-in Standard Model with five massless flavours, the 486
subprocesses of `p p > j j j j` fall into 23 classes. Within a class, all
subprocesses are related by a permutation of quark flavours, and possibly by
charge conjugation.
pyAmpliCol records such relations in the artifact, can skip generating the
redundant subprocesses, can find relations numerically, and exports the
skipped subprocesses to UMAMI through their representative.

| Mechanism | What it does | Status of the relation | Generated subprocesses |
| --- | --- | --- | --- |
| Every generation | Writes `process-groups.json`; each subprocess is its own group | Identity | All |
| `--flavor-blind` | Keeps one representative per flavour class, records the others as its members | **Assumed**, with a warning when the model may break it | Representatives only |
| `--analyze-process-groups numerical`, `pyamplicol group-processes` | Merges generated subprocesses whose resolved components agree up to a rational factor | Tested numerically at three points | All, unchanged |
| UFO conserved-charge filter | Drops expansion candidates that cannot have a tree amplitude | Exact | Not applicable |

The sections below define the relation, explain how candidate subprocesses
and leg mappings are found, and describe each mechanism, the file format and
the UMAMI export. A [worked example](#worked-example-p-p--j-j-j-j) with
`p p > j j j j` closes the page.

## The relation between two subprocesses

A group has one **representative**, which is always a generated subprocess,
and any number of **members**. Each member carries a relation to the
representative:

```text
ME_member(p[0], ..., p[n-1]) = factor * ME_representative(q[0], ..., q[n-1])
q[r] = p[permutation[r]]
```

- Legs are counted from 0 in the public order of the process expression,
  incoming legs first. Momenta are four-vectors in that order.
- `permutation[r]` is the member leg that plays the role of representative leg
  `r`. This is the convention of `ProcessAlias.particle_permutation` and of
  the UMAMI momentum permutation.
- A permutation never moves a leg across `>`. It may exchange the two incoming
  legs, as between `d g > d g` and `g u > u g`.
- `factor` is a signed rational number whose numerator and denominator are at
  most 99 in absolute value. `--flavor-blind` records factor 1; a
  flavour-blind member gets another factor only when its group is later merged
  numerically. A member that was not generated is evaluated through its
  representative and must have a positive factor; a map that would give one a
  non-positive factor is rejected when it is written.
- `conjugated` marks a relation that includes charge conjugation (see
  [Charge conjugation](#charge-conjugation)).
- Helicities follow the legs: the member's helicity on leg `permutation[r]`
  equals the representative's helicity on leg `r`. This also holds for
  conjugated relations.

For example, the relation between `u u~ > d u d~ u~` and its representative
`d d~ > d u d~ u~` has permutation `[0, 1, 3, 2, 5, 4]`:

```text
ME(u u~ > d u d~ u~)(p0, p1, p2, p3, p4, p5)
    = ME(d d~ > d u d~ u~)(p0, p1, p3, p2, p5, p4)
```

Representative leg 2 (the outgoing `d`) takes the momentum of member leg 3
(the outgoing `u`), so the relabelling is `u <-> d`, applied to every leg.

## Particle types

A relabelling may only exchange particles that the dynamics cannot tell
apart. pyAmpliCol summarises everything a relabelling must preserve in a
**particle type**:

| Component | Meaning |
| --- | --- |
| spin | `2S+1`: 1 for scalars, 2 for fermions, 3 for vectors |
| colour | dimension of the SU(3) representation: 1, 3 or 8 (triplet and antitriplet are not distinguished) |
| mass | the name of the mass parameter, or `ZERO` |
| width | the name of the width parameter, or `ZERO` |
| self-conjugate | whether the particle is its own antiparticle |

A particle and its antiparticle share one type. Masses and widths are compared
by **parameter name**, after compile-time resolution. A parameter that the
model or its restriction fixes to zero at compile time counts as `ZERO`. Two
different parameters with equal numerical values are never identified.

Examples:

- Built-in SM: `d`, `u`, `s`, `c` and `b` share one type (massless fermion
  triplet), and the top quark has its own (`mass_6`, `width_6`).
- Shipped `sm.json`, default restriction: `d`, `u`, `s` and `c` share one
  type. The `b` quark (`MB`) and the top (`MT`, `WT`) each have their own.
- In `sm.json`, the massless leptons `e-` and `mu-` and the three neutrinos all
  have the same type (massless colour-singlet fermion), while `ta-` (`MTA`)
  has its own. `W+` and `W-` share a type. The photon and the `Z` do not,
  because the `Z` mass and width differ from `ZERO`.

The sorted list of particle types of each side is the **coarse key** of a
subprocess. Only subprocesses with equal coarse keys are ever compared.

## Flavour profiles

Equal types are necessary but not sufficient. The relabelling must also be
**consistent**: a flavour must be renamed the same way on every leg where it
appears. pyAmpliCol describes each flavour `f` of a subprocess by its
**profile**, the four counts

```text
profile(f) = (initial particles, initial antiparticles,
              final particles,   final antiparticles)
```

Self-conjugate particles are counted as particles. Two subprocesses are
related by a type-preserving flavour relabelling exactly when they have the
same number of incoming legs and the same multiset of `(type, profile)` pairs.

### Why sorting by PDG code fails

**A related pair that a positional pairing breaks.** Compare
`u u~ > d u d~ u~` with `d d~ > d u d~ u~`:

| Subprocess | profile of `u` | profile of `d` |
| --- | --- | --- |
| `u u~ > d u d~ u~` | (1, 1, 1, 1) | (0, 0, 1, 1) |
| `d d~ > d u d~ u~` | (0, 0, 1, 1) | (1, 1, 1, 1) |

The `(type, profile)` multisets agree, so the relabelling `u <-> d` relates
the two. Ordering each side by type, then profile, then flavour, and pairing
the ordered legs gives the permutation `[0, 1, 3, 2, 5, 4]` shown above.
Sorting each side by signed PDG code instead pairs the incoming `u` with the
incoming `d`, but the outgoing `u` with the outgoing `u`. No relabelling does
that, and evaluating the representative at those momenta computes a different
function.

**An unrelated pair that a type-only comparison merges.** `d d~ > u u~` and
`d d~ > d d~` have the same particle types on each side. The first has only
the s-channel gluon exchange. The second also has the t-channel exchange and
the interference between the two. Their profiles differ: in `d d~ > u u~`,
`d` has `(1, 1, 0, 0)` and `u` has `(0, 0, 1, 1)`, while in `d d~ > d d~`, `d`
has `(1, 1, 1, 1)`. `--flavor-blind` therefore never puts them in one class.

When several flavours of a subprocess share a type and a profile (`s` and `c`
in `d d~ > s c s~ c~`), each pairing of them is a valid leg mapping. The first
mapping pairs them by increasing `|PDG|`. The numerical analysis tries at most
24 such mappings, and at most 24 conjugated ones.

## Charge conjugation

Charge conjugation replaces every particle by its antiparticle, which
exchanges the particle and antiparticle entries of each profile. With
`--consider-charge-conjugation` (the default), two subprocesses are also
related when one matches the conjugate of the other. Examples are
`u~ g > u~ g g g` and `d g > d g g g`, or `d u~ > d u u~ u~` and
`d u~ > d d d~ u~`. The relation is stored with `"conjugated": true`, and it
means:

- helicity values move with the legs **unchanged**: no helicity is flipped;
- leading-colour (LC) flow words are relabelled and **reversed**, because
  every quark line runs in the opposite direction.

Charge conjugation with unchanged helicities is a symmetry of vector-like
interactions such as QCD and QED. It is not a symmetry of chiral couplings
such as those of the `W` and `Z`. Relations that also need a helicity flip
(CP-type relations) are not pursued. For example, `u d~ > W+ a` and
`d u~ > W- a` have equal helicity-summed matrix elements by CP, but their
helicity components differ, so the numerical analysis never groups them.
`--flavor-blind` does put them in one class, with a warning that a chiral
electroweak vertex breaks charge conjugation.

`--no-consider-charge-conjugation` (`process.consider_charge_conjugation =
false`) disables conjugated relations. The setting only matters together with
`--flavor-blind` or the numerical analysis.

## Flavour-blind generation

`--flavor-blind` (`process.flavor_blind = true`) states an **assumption**:
every subprocess in a flavour class has the same matrix element as the class
representative. A flavour class contains the subprocesses related by a
type-preserving flavour relabelling and, unless
`--no-consider-charge-conjugation`, by charge conjugation.

Generation then proceeds as follows:

1. The request is expanded and checked exactly as without the flag.
   Subprocesses are named at this stage, so a generated subprocess has the
   same process ID with or without `--flavor-blind`.
2. The expanded subprocesses are partitioned into flavour classes. The
   representative is the canonical member: it has the smallest `|PDG|`
   codes, and particles come before antiparticles. For example, `d d~ > g g`
   represents `u u~ > g g`, and `d g > d g` represents `u~ g > u~ g`.
3. Only representatives are generated. Each other member is recorded in
   `process-groups.json` with factor 1, basis `flavor-blind`,
   `"generated": false`, and the leg permutation of the first flavour mapping.
   A non-conjugated mapping is preferred; a conjugated one is used only when
   no non-conjugated mapping exists.

A subprocess with an explicit `ProcessAlias` keeps its own artifact entry and
is not folded into a class.

Generation logs the outcome of step 3 in one line, such as
`Flavor-blind grouping: 486 subprocesses -> 23 generated representatives; 463
routed members (140 by charge conjugation); 0 relabelling warnings`, and the
summary table printed at the end has a `process grouping` row; see the
[worked example](#worked-example-p-p--j-j-j-j).

Members stay selectable. Every SDK (`Runtime.load(..., process=...)`,
`pyamplicol evaluate --process`, `rusticol_runtime_load` and the C++, Fortran
and Rust wrappers) resolves a member ID or expression, in any side order,
through `process-groups.json` to its representative. The representative is
evaluated at the permuted momenta and every output is multiplied by the
recorded factor. Helicity and colour-flow IDs are reported in the member's leg
order; for a conjugated member each LC flow word is reversed, and colour-flow
selectors given in member IDs are translated back. See
[Process-group members](process-selection-and-permutations.md#process-group-members).

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_4j_blind", process="d u~ > d u u~ u~")
runtime.representative_process_key  # the generated representative
runtime.process_group_factor  # Fraction(1, 1)
runtime.charge_conjugated  # True for this member
values = runtime.evaluate(points)  # points in the member's leg order
```

The UMAMI export applies this mapping automatically (see
[UMAMI export](#umami-export-of-flavour-blind-members)).

### Warnings

The assumption is not verified. Generation checks whether each relation is a
**symmetry of the model vertices that the representative is generated with**.
A vertex is allowed when its coupling orders fit `max_coupling_orders` and,
under the default `minimal` coupling-order policy, the representative's
minimal coupling-order envelope. The check works as follows:

- The relabelling is completed to a permutation of the model's particles.
  Flavours that the relabelling moves but that occur in only one of the two
  subprocesses are paired by particle type. Every
  allowed vertex that the permutation moves must have an allowed image with
  the same particles in the same order and the same structure and coupling.
  For compiled (UFO or JSON) models, structure and coupling mean the Lorentz
  and colour expressions and the resolved coupling. For the built-in SM, they
  mean the vertex kind and its coupling.
- A conjugated relation also requires that the conjugate of every allowed
  vertex is allowed, and that no allowed vertex couples fermions chirally. A
  Lorentz structure is chiral when exchanging its left- and right-handed
  projectors changes it, or when it contains `gamma5`. A built-in electroweak
  current is chiral when its left- and right-handed couplings differ.

A member that fails the check gets a warning, stored as its `warning` field in
`process-groups.json`. The warnings are also logged once on the
`pyamplicol.generation` logger, with their count and up to three examples.
For `pyamplicol generate "p p > z j" OUT --model built-in-sm --flavor-blind`:

```text
'u u~ > g z' assumed equal to 'd d~ > g z', but relabelling d->u, u->d changes
the allowed vertex [d a d] (coupling orders: QCD<=1, QED<=1)

'u~ g > u~ z' assumed equal to 'd g > d z', but relabelling d->u, u->d changes
the allowed vertex [d a d] and charge conjugation is broken by the chiral
allowed vertex [d z d] (coupling orders: QCD<=1, QED<=1)
```

The check inspects vertices, not diagrams, and completes the relabelling in
one fixed way, so it can warn about a relation that holds. In the same run,
`s s~ > g z` is equal to `d d~ > g z` at tree level, but it still warns:
`d <-> s` alone moves the allowed `W` vertex `[d w+ u]`, although also
exchanging `u <-> c` would restore it.

### Strict flavour-blind generation

`--flavor-blind-strict` (`process.flavor_blind_strict = true`, which implies
`flavor_blind`) routes only the relations that pass the check above. Within a
flavour class, each member tries every flavour pairing, conjugated ones last,
against each representative kept so far, and joins the first one without a
warning. A member that matches none is generated and becomes a representative
of its own, which later members may join. No member of the written map has a
warning. For `p p > j j` with explicit electroweak orders, `u u~ > g g` is then
generated next to `d d~ > g g`, while `s c > s c` still joins `d u > d u`.

The check is local and conservative, so strict generation may generate a
member whose relation holds; it never routes a member the check rejects.

### When to use it

Equal matrix elements within a class are **your responsibility**. The flag is
safe when the interactions that can appear are blind to the relabelled
flavours. The typical case is pure-QCD jets with massless quarks under the
default `minimal` coupling-order policy: `p p > j j`, `p p > j j j` and
`p p > j j j j` produce no warnings.

It is not safe when photons, `Z` or `W` bosons can couple to the relabelled
flavours:

- `p p > z j`, `p p > a j`, `p p > e+ e- j`: up- and down-type quarks have
  different electric charges and `Z` couplings, and the chiral `Z` and `W`
  couplings break charge conjugation;
- jets with electroweak orders allowed explicitly, for example
  `--coupling-order-policy explicit --max-coupling-order QED=2`;
- requests that mix charged leptons and neutrinos: in massless-lepton models
  they share a type, so `p p > e+ e-` and `p p > ve ve~` would fall into one
  class.

To check an assumption, generate the subprocesses in question without
`--flavor-blind` and run the
[numerical analysis](#numerical-analysis): it merges exactly the relations
that hold. For example, with `u u~ > a a` and `d d~ > a a` it finds
`ME(u u~ > a a) = 16 ME(d d~ > a a)`, the ratio `(Q_u / Q_d)^4` of the
quark charges. `--flavor-blind` would have assumed factor 1.

## Conserved charges in UFO expansion

When generating with a UFO or other compiled model, a multiparticle request is
expanded in four steps:

1. Each side of `>` is filled as a **multiset** of particles, so two
   candidates never differ only by an ordering within one side.
2. Candidates that violate an **additive charge conserved by every vertex**
   are dropped.
3. The colour-readiness probe runs on the remaining candidates.
4. Tree amplitudes are constructed. With `--flavor-blind`, the flavour classes
   are formed just before this step.

A conserved charge assigns a rational value to every particle, the opposite
value to its antiparticle and zero to self-conjugate particles. The values on
every vertex term sum to zero. The conserved charges are a basis of the
rational nullspace of the vertex/particle incidence matrix, computed from the
model's vertex list alone. No quantum numbers are hard-coded.

Every vertex conserves these charges, so every tree graph does too. A
candidate whose incoming and outgoing totals differ can have no tree
amplitude, whatever the coupling orders. The filter therefore removes nothing
that amplitude construction would have kept; it only drops early the
candidates that would have failed later. A request with no candidate left is
rejected with an error. For the shipped `sm.json`
with its default restriction, the basis has nine charges, spanning electric
charge, one quark number per generation (the CKM matrix is diagonal), one
lepton number per generation and two ghost numbers. With a non-diagonal CKM
matrix, the three quark numbers reduce to baryon number.

The built-in SM keeps its own catalogue prefilters (`flavor_scheme`,
`max_quark_lines`). `pyamplicol model processes` applies the same charge
filter, but not the colour and amplitude checks. See
[Models and Processes](models-and-processes.md#multiparticles)
for the expansion syntax.

## Numerical analysis

`--analyze-process-groups numerical` (`process.analyze_process_groups =
"numerical"`) runs after the artifact is written. `pyamplicol group-processes`
runs the same analysis on an existing artifact. The analysis looks for
relations among the **generated group representatives** and merges the
groups it relates. Flavour-blind members are not analysed: they have no
runtime process of their own, and loading one only evaluates its
representative.

### Procedure

1. Representatives are visited in canonical order. Each one is compared with
   the earlier representatives that kept their own group and have the same
   coarse key and the same colour accuracy.
2. The candidate leg mappings are, in order: the flavour-profile mappings,
   then (with charge conjugation) the conjugated profile mappings, then a
   fallback that pairs the legs of each side by type and signed PDG code.
3. Three **phase-space points** are used, in f64. At each point, the model
   inputs are varied independently and deterministically. Each mutable
   parameter of kind `external` or `normalization`, other than a particle
   mass or width, is multiplied by a factor in `[0.9, 1.1]`. The factor
   depends only on the parameter name, the point and the seed, so both
   processes of a comparison always see the same values. Couplings and other
   derived parameters follow from the varied inputs. Masses and widths are
   never varied. A derived mass, such as `MW` computed from `Gf`, `aEWM1` and
   `MZ` in `sm.json`, is read back from the runtime after the variation.
4. Each point is **on shell** for the resolved masses. It is generated in the
   centre-of-mass frame with the beams along the z axis (massive beams are
   supported), at `sqrt(s) = max(1000, 2 * threshold)` in the model's energy
   unit. The threshold is the larger of the sum of the beam masses and the sum
   of the final-state masses. The final state is a massive RAMBO point. A
   `2 > 1` process uses `sqrt(s)` equal to the final mass, and a decay is
   evaluated at rest.
5. The representative is evaluated at its points, and the candidate member at
   the permuted points `p[permutation[r]] = q[r]`. All surviving mappings are
   evaluated in one batch per point.
6. Every resolved component is compared through the mapping. At LC, a
   component is a helicity configuration times a colour flow. Helicity values
   are permuted. Flow words are relabelled, and reversed for a conjugated
   mapping; they are compared as sets of open colour lines, or as one adjoint
   trace up to cyclic rotation. At NLC and full colour, a component is a
   helicity configuration times a contracted colour component.

A mapping is accepted when one common ratio `member / representative`
describes every component at all three points:

| Quantity | Rule |
| --- | --- |
| zero components | below `1e-12` times the largest component at that point; a zero on one side must be a zero on the other |
| agreement of a non-zero component | relative `1e-9`, never tighter than `1e-9 * 1e-4` times the largest component |
| common ratio across points | relative `1e-9` |
| a point that vanishes entirely on either side | no relation |
| imaginary part of the ratio | at most `1e-9` of its modulus |
| recorded factor | rational within `1e-9` relative, numerator and denominator at most 99 in absolute value |
| `--only-unit-multiplier` | only factor 1 is accepted |
| seed | `20260917`, recorded in the map |

The search stops at the first earlier group and mapping that pass. The
subprocess's group is then merged into that group with basis `numerical`.

A representative is **skipped**, and stays its own group, in any of these
cases:

- it was generated with selected helicities (incomplete helicity coverage);
- it was generated with selected colour flows or sectors;
- its masses leave no on-shell phase space;
- its runtime raises an error.

Each skip is logged as a warning and listed in the summary.

## The `group-processes` command

```console
pyamplicol group-processes ARTIFACT \
  [--analyze-process-groups {none,numerical}] \
  [--only-unit-multiplier] \
  [--consider-charge-conjugation | --no-consider-charge-conjugation]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--analyze-process-groups` | `numerical` | `none` only writes the trivial map when the artifact has no `process-groups.json` |
| `--only-unit-multiplier` | off | accept only new relations with factor 1 |
| `--[no-]consider-charge-conjugation` | on | also try conjugated mappings |

The `numerical` default belongs to the subcommand, and it takes precedence
over a `--card` value. Pass `--analyze-process-groups none` or
`--set process.analyze_process_groups=none` to change it. The schema default
of `process.analyze_process_groups`, used by generation, remains `none`.

The command reads the stored map, analyses its representatives, merges the
new relations, and rewrites `process-groups.json` in place. The rewrite runs
under the lock that also serializes generation and appends. The file's
payload record in
`artifact.json` is updated, but the artifact ID is unchanged, because the
`process-groups` role is not part of the runtime identity. UMAMI files are
never touched. The command prints a summary; add `--json` for
machine-readable output. For the five subprocesses of the
[file example](#the-process-groupsjson-file), generated without analysis:

```console
pyamplicol group-processes artifacts/grouping_demo --json
```

```json
{
  "analysis": "numerical",
  "artifact": "/home/user/artifacts/grouping_demo",
  "groups_after": 2,
  "groups_before": 5,
  "new_relations": [
    {"conjugated": false, "factor": "16", "permutation": [0, 1, 2, 3],
     "process": "u u~ > a a", "process_id": "u_ubar_to_a_a",
     "representative": "d_dbar_to_a_a"},
    {"conjugated": true, "factor": "1", "permutation": [0, 1, 2, 3],
     "process": "d~ g > d~ g", "process_id": "dbar_g_to_dbar_g",
     "representative": "d_g_to_d_g"},
    {"conjugated": false, "factor": "1", "permutation": [1, 0, 2, 3],
     "process": "g u > u g", "process_id": "g_u_to_u_g",
     "representative": "d_g_to_d_g"}
  ],
  "processes_analysed": 5,
  "skipped": []
}
```

`representative` is the group the subprocess joined. A second run reports
`groups_before: 2`, `processes_analysed: 2` and no new relation.

When two groups merge, the canonically smaller representative stays. Every
member of the absorbed group, including its flavour-blind members, is
re-expressed through the new link:

- factors multiply;
- permutations compose;
- the `conjugated` flags combine (two conjugations cancel);
- the basis becomes the least verified of the two links, in the order
  `identity`, `numerical`, `flavor-blind`;
- a warning is kept.

The analysis settings are appended to `analyses` unless an identical entry
is already there. Rerunning with the same settings is deterministic and adds
nothing. A later run with relaxed settings, for example without
`--only-unit-multiplier`, can add relations.

The Python equivalent is `pyamplicol.group_processes` (also available as
`pyamplicol.api.group_processes`). It takes the same options as keyword
arguments: `analysis="numerical"`, `only_unit_multiplier=False` and
`consider_charge_conjugation=True`. See
[Python API](python-api.md#group-equivalent-subprocesses).

## The `process-groups.json` file

Every generation lane (compiled, eager, recurrence, on-the-fly) writes
`process-groups.json` at the artifact root, declared in `artifact.json` with
payload role `process-groups` and media type `application/json`. Without
`--flavor-blind` and without numerical analysis, each generated subprocess is
its own group with factor 1. An append keeps the stored groups and adds the
new subprocesses. A newly generated subprocess whose process ID matches an
ungenerated flavour-blind member replaces that member and becomes its own
group.

The example below comes from

```console
pyamplicol generate "d d~ > a a" artifacts/grouping_demo --model built-in-sm \
  --process "u u~ > a a" --process "d g > d g" --process "d~ g > d~ g" \
  --process "g u > u g" --analyze-process-groups numerical
```

(lists are folded onto one line for readability):

```json
{
  "kind": "pyamplicol-process-groups",
  "schema_version": 1,
  "analyses": [
    {"variant": "numerical", "consider_charge_conjugation": true,
     "only_unit_multiplier": false, "points": 3, "parameter_variation": 0.1,
     "tolerance": 1e-09, "seed": 20260917}
  ],
  "groups": [
    {
      "representative": {"process_id": "d_dbar_to_a_a", "process": "d d~ > a a"},
      "members": [
        {"process_id": "d_dbar_to_a_a", "process": "d d~ > a a",
         "pdgs": [1, -1, 22, 22], "initial_count": 2,
         "factor": {"numerator": 1, "denominator": 1},
         "permutation": [0, 1, 2, 3], "conjugated": false,
         "basis": "identity", "generated": true},
        {"process_id": "u_ubar_to_a_a", "process": "u u~ > a a",
         "pdgs": [2, -2, 22, 22], "initial_count": 2,
         "factor": {"numerator": 16, "denominator": 1},
         "permutation": [0, 1, 2, 3], "conjugated": false,
         "basis": "numerical", "generated": true}
      ]
    },
    {
      "representative": {"process_id": "d_g_to_d_g", "process": "d g > d g"},
      "members": [
        {"process_id": "d_g_to_d_g", "process": "d g > d g",
         "pdgs": [1, 21, 1, 21], "initial_count": 2,
         "factor": {"numerator": 1, "denominator": 1},
         "permutation": [0, 1, 2, 3], "conjugated": false,
         "basis": "identity", "generated": true},
        {"process_id": "dbar_g_to_dbar_g", "process": "d~ g > d~ g",
         "pdgs": [-1, 21, -1, 21], "initial_count": 2,
         "factor": {"numerator": 1, "denominator": 1},
         "permutation": [0, 1, 2, 3], "conjugated": true,
         "basis": "numerical", "generated": true},
        {"process_id": "g_u_to_u_g", "process": "g u > u g",
         "pdgs": [21, 2, 2, 21], "initial_count": 2,
         "factor": {"numerator": 1, "denominator": 1},
         "permutation": [1, 0, 2, 3], "conjugated": false,
         "basis": "numerical", "generated": true}
      ]
    }
  ]
}
```

Reading it:

- `u u~ > a a` is 16 times `d d~ > a a` at equal momenta.
- `d~ g > d~ g` is the charge conjugate of `d g > d g`, leg by leg.
- `g u > u g` at `(p0, p1, p2, p3)` equals `d g > d g` at
  `(p1, p0, p2, p3)`: representative leg 0, the incoming quark, takes the
  momentum of member leg 1.
- All five subprocesses remain generated: numerical grouping records
  relations but removes nothing.

| Field | Meaning |
| --- | --- |
| `kind`, `schema_version` | `pyamplicol-process-groups`, version 1 |
| `analyses` | one entry per distinct numerical-analysis setting applied |
| `groups` | sorted by representative process ID |
| `representative` | process ID and expression of the group's generated representative |
| `members` | the representative first (identity relation), then every member |
| `pdgs`, `initial_count` | signed PDG codes in public leg order, number of incoming legs |
| `factor` | `ME_member = factor * ME_representative`, as a reduced fraction |
| `permutation` | `q[r] = p[permutation[r]]`, representative leg to member leg |
| `conjugated` | the relation includes charge conjugation |
| `basis` | `identity`, `flavor-blind` (assumed) or `numerical` (tested) |
| `generated` | whether the member has its own runtime process in the artifact |
| `warning` | present only for a flavour-blind relation that may not be a model symmetry |

Every subprocess of the artifact appears exactly once as a generated member,
and a representative is always generated.

`pyamplicol inspect` shows the map in its `Process groups` section (colours
omitted here). For the five subprocesses above:

```text
Process groups

+--------+--------------+-----------+--------+------------+-------------+-------------+-----------+----------+
| groups | subprocesses | generated | routed | conjugated | factor != 1 | bases       | analyses  | warnings |
+--------+--------------+-----------+--------+------------+-------------+-------------+-----------+----------+
|      2 |            5 |         5 |      0 |          1 |           1 | numerical 3 | numerical |        0 |
+--------+--------------+-----------+--------+------------+-------------+-------------+-----------+----------+

+---------------------+-------------+----------------+--------+---+-------------+-----------+
| process ID          | process     | role           | factor | C | permutation | basis     |
+---------------------+-------------+----------------+--------+---+-------------+-----------+
| d_dbar_to_a_a       | d d~ > a a  | representative |        |   |             |           |
| `- u_ubar_to_a_a    | u u~ > a a  | generated      |     16 |   |             | numerical |
+---------------------+-------------+----------------+--------+---+-------------+-----------+
| d_g_to_d_g          | d g > d g   | representative |        |   |             |           |
| |- g_u_to_u_g       | g u > u g   | generated      |      1 |   | [1 0 2 3]   | numerical |
| `- dbar_g_to_dbar_g | d~ g > d~ g | generated      |      1 | C |             | numerical |
+---------------------+-------------+----------------+--------+---+-------------+-----------+

C: charge-conjugated relation; factor F: ME = F x ME(representative)
permutation p: representative leg r takes the momentum of member leg p[r]
numerical analysis: 3 points, tolerance 1e-09
```

A `generated` member has its own runtime process, and a `routed` member is
evaluated through its representative. The permutation column is empty for the
identity. The factor, `C` and permutation columns appear only when some member
needs them, and a `note` column with `[n]` references to the warnings printed
below the table appears when a member has a warning. Maps with more than 80
subprocesses list each group's members as a grid of process strings instead;
see [Inspect and select a process](command-line-interface.md#inspect-and-select-a-process).
An artifact without grouping prints one line, `Process groups: none; each of
the N processes is its own group`.

The summary printed by this `generate` command reports the same map in its
`process grouping` row, `5 subprocesses -> 2 groups: 5 generated, 0 routed`,
followed by `3 numerical relations (1 charge-conjugated, 1 with factor != 1);
0 warnings` and the analysis settings. Its `Generated Processes` table lists
`u u~ > a a` as `member of d_dbar_to_a_a (x16)` and `d~ g > d~ g` as
`member of d_g_to_d_g (C)`.

## UMAMI export of flavour-blind members

The UMAMI provider exports every flavour-blind member, although it has no
runtime process:

- A member adds flavour selectors that evaluate its **representative's
  runtime**, with no `runtime_processes` row of its own. Its
  `members[].runtime` names the representative.
- The momentum permutation is the composition of the recorded relation with
  the runtime ordering. It may exchange the incoming legs, but the selector
  keeps the member's physical incoming PDG codes for the PDF weights, so this
  is not a beam-exchange assumption.
- A conjugated member uses the representative's colour flows reversed. Its
  LHE colour tags and helicity labels are those of the member.
- The member's factor (1 for flavour-blind relations) multiplies the runtime
  factor.
- Every provider exporting such members records an additional
  `grouping.assumptions` entry:

  ```text
  flavor-blind matrix elements: ungenerated process-group members are
  evaluated through their relabelled representative (assumed, not proven)
  ```

A member whose relabelling moves unequal masses between legs, or maps a
restricted helicity selection onto a different set, is exported by a separate
provider that loads the representative's runtime. Warnings stay in
`process-groups.json`; they are not copied into the UMAMI metadata.
Numerically grouped subprocesses are generated and keep their ordinary
export; `group-processes` does not change the UMAMI files. See
[UMAMI and MadSpace](umami-api.md#flavour-blind-members) for the selector
layout.

## Worked example: `p p > j j j j`

Preview the flavour classes with a dry run, which prints the subprocesses
that would be generated:

```console
pyamplicol generate "p p > j j j j" artifacts/pp_4j_blind \
  --model built-in-sm --flavor-blind --dry-run
```

With the built-in SM at the default `flavor_scheme = 5`, the request has:

| Setting | Generated subprocesses |
| --- | --- |
| no `--flavor-blind` | 486 |
| `--flavor-blind --no-consider-charge-conjugation` | 33 |
| `--flavor-blind` | 23 |

Charge conjugation merges, for example, the `d~ d~` and `d~ g` initial
states into the `d d` and `d g` classes, and `d u~ > d u u~ u~` into the
class of `d u~ > d d d~ u~`. Generate the artifact and analyse the
representatives in one step:

```console
pyamplicol generate "p p > j j j j" artifacts/pp_4j_blind \
  --model built-in-sm --flavor-blind --analyze-process-groups numerical
```

Before generating, the command logs the flavour classes it found:

```text
INFO pyamplicol.generation: Flavor-blind grouping: 486 subprocesses -> 23 generated representatives; 463 routed members (140 by charge conjugation); 0 relabelling warnings
```

and its final summary reads the written map back (abridged, colours omitted):

```text
| processes              | 23 requests                                                                              |
| process grouping       | 486 subprocesses -> 23 groups: 23 generated, 463 routed                                  |
|                        | 463 flavor-blind relations (140 charge-conjugated); 0 warnings                           |
|                        | numerical analysis: 3 points, tolerance 1e-09                                            |
...
+--------------------+------------------+-----------------------------+
| name               | process          | group                       |
+--------------------+------------------+-----------------------------+
| p_p_to_j_j_j_j_1   | d d > d d d d~   | 10 subprocesses (9 routed)  |
| p_p_to_j_j_j_j_2   | d d > d d u u~   | 40 subprocesses (39 routed) |
...
| p_p_to_j_j_j_j_486 | g g > g g g g    |                             |
+--------------------+------------------+-----------------------------+
```

The minimal coupling-order envelope is pure QCD, so no relation warns. The
artifact contains 23 runtime processes, and `process-groups.json` has 23
groups covering all 486 subprocesses: 463 flavour-blind members, 140 of them
conjugated. The numerical analysis finds no further relation among the 23
representatives. Some of the classes:

| Representative | Process ID | Class size | Conjugated members | Incoming legs exchanged |
| --- | --- | --- | --- | --- |
| `d d~ > d u d~ u~` | `p_p_to_j_j_j_j_32` | 20 | 0 | 0 |
| `d u > d u s s~` | `p_p_to_j_j_j_j_9` | 60 | 30 | 0 |
| `d u~ > d d d~ u~` | `p_p_to_j_j_j_j_52` | 40 | 20 | 20 |
| `d g > d g g g` | `p_p_to_j_j_j_j_81` | 10 | 5 | 0 |
| `d d~ > g g g g` | `p_p_to_j_j_j_j_51` | 5 | 0 | 0 |
| `g g > g g g g` | `p_p_to_j_j_j_j_486` | 1 | 0 | 0 |

Three members illustrate the relations:

| Member | Representative | `permutation` | `conjugated` |
| --- | --- | --- | --- |
| `u u~ > d u d~ u~` | `d d~ > d u d~ u~` | `[0, 1, 3, 2, 5, 4]` | false |
| `d u~ > d u u~ u~` | `d u~ > d d d~ u~` | `[1, 0, 4, 5, 3, 2]` | true |
| `u~ g > u~ g g g` | `d g > d g g g` | `[0, 1, 2, 3, 4, 5]` | true |

The second relation exchanges the incoming legs and conjugates. Its
representative leg 0, the incoming `d`, takes the momentum of member leg 1,
the incoming `u~`, which conjugation and the relabelling `u <-> d` turn into a
`d`. Loaded by its own expression, each of these three members agrees with a
separately generated copy of the same subprocess to about `1e-14`.

`pyamplicol inspect artifacts/pp_4j_blind` summarises the 23 classes and lists
the members of each, grouped into non-conjugated and conjugated blocks, with
`g g > g g g g` listed as `ungrouped`.
`--process p_p_to_j_j_j_j_52` lists one class member by member, with
permutations, and `--process 'd u~ > d u u~ u~' --full-physics` adds a
`Process Group` table with that member's representative, factor, conjugation
and permutation.

The UMAMI provider in `API/umami` has 23 `runtime_processes` rows. Its
`members` cover all 486 subprocess IDs, and its `grouping.assumptions` list
contains the flavour-blind entry.

To check the assumption itself, generate all 486 subprocesses without
`--flavor-blind` and let the numerical analysis group them:

```console
pyamplicol generate "p p > j j j j" artifacts/pp_4j \
  --model built-in-sm --analyze-process-groups numerical
```

This generates all 486 runtime processes, so it takes much longer. The
analysis then finds the same 23 groups, with the same representatives,
permutations and conjugation flags, all with factor 1, now with basis
`numerical`.

## Limitations

- `--flavor-blind` is an assumption. Its warning comes from a per-vertex check
  and can be raised for a relation that holds (see [Warnings](#warnings)).
- A flavour-blind representative that later turns out to have no tree-level
  amplitude is skipped as unsupported, and its members disappear with it. The
  skip warning gives only their number, not their names.
- A routed member uses its representative's model parameters, so its value
  follows the representative's flavours. Changing a parameter that
  distinguishes their flavours (a mass, width or Yukawa coupling) invalidates
  the relation; generate the process explicitly if it must follow independent
  parameter values.
- Correlated evaluation through a member is supported only when the composed
  permutation (relation and requested order) is the identity and the relation
  is not conjugated; the factor is then applied. Other selections fail with an
  error naming the member and its representative.
- For conjugated members with several open quark lines, the reversed LC flow
  IDs may list the open lines in another order than a separate generation of
  the same subprocess would; values and flow selection agree.
- The numerical analysis is a test at three f64 points, not a proof: two
  matrix elements that agree at those points within `1e-9` are grouped.
- The search covers only relations with a rational factor whose numerator
  and denominator are at most 99, a leg permutation within each side, and
  optional charge conjugation with unchanged helicities. It does not cover
  CP-type relations or crossing between the initial and final states.
- Subprocesses generated with selected helicities or colour flows are never
  grouped numerically.
- Relations are only searched between subprocesses with equal particle types
  per side, so two different mass parameters with equal values are never
  identified.
- Numerical relations do not shrink the artifact or the UMAMI export: both
  subprocesses stay generated.
