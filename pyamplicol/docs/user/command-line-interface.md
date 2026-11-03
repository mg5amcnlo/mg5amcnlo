---
title: "Command-Line Interface"
nav_order: 5
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Command-Line Interface

The `pyamplicol` command covers the complete user workflow: inspect or compile a
model, plan or generate process artifacts, inspect their physics axes, evaluate
phase-space points, and profile the optimized runtime. The same schema also
drives the typed [Python API](python-api.md).

> [!TIP]
> Start with the four-card walkthrough in [Quick Start](quick-start.md). This page
> is a command map, not a requirement to configure every available switch.

## Command map

| Command | Purpose | Typical use |
| --- | --- | --- |
| `generate` | Plan or write a process artifact | `pyamplicol generate "d d~ > z g" artifacts/builtin_ddbar_to_zg --model built-in-sm` |
| `evaluate` | Evaluate totals or resolved components | `pyamplicol evaluate artifacts/builtin_ddbar_to_zg --momenta point.json` |
| `profile` | Calibrate and measure the optimized total path | `pyamplicol profile artifacts/builtin_ddbar_to_zg --target-runtime 1` |
| `benchmark` | Compatibility alias for `profile` | Existing cards use `action = "benchmark"` |
| `inspect` | Show artifact and process metadata without running it | `pyamplicol inspect artifacts/builtin_ddbar_to_zg` |
| `group-processes` | Find generated subprocesses with equal matrix elements | `pyamplicol group-processes artifacts/pp_jj` |
| `model inspect` | Summarize a built-in, JSON, UFO, or compiled model | `pyamplicol model inspect models/json/sm/sm.json` |
| `model compile` | Compile portable model IR or a prepared kernel bundle | See [Models and Processes](models-and-processes.md) |
| `model processes` | Expand and inspect a process request without generating it | `pyamplicol model processes "p p > all all" --model built-in-sm` |
| `config template` | Write the exhaustive schema-v1 TOML template | `pyamplicol config template run.toml` |
| `config resolve` | Show requested and effective settings | `pyamplicol config resolve run.toml` |
| `examples list` | List shipped card names, actions, and descriptions | `pyamplicol examples list` |
| `examples copy` | Create an editable, self-contained example workspace | `pyamplicol examples copy ./pyamplicol-examples --force` |
| `examples run` | Run one named packaged card in a private cache workspace | `pyamplicol examples run evaluate_total` |
| `doctor` | Diagnose the installed runtime and SDK | `pyamplicol doctor` |
| `self-test` | Run the installed-package smoke test | `pyamplicol self-test` |

License-request helpers are documented in
[Symbolica and Licensing](symbolica-and-licensing.md).

Run `pyamplicol COMMAND --help` for the options accepted by a command. For the
model family, place `--help` after the second verb, for example
`pyamplicol model compile --help`.

## Cards, direct options, and overrides

There are three equivalent ways to supply configuration:

1. a schema-v1 TOML card;
2. dedicated command-line options;
3. repeated `--set PATH=VALUE` overrides.

Configuration precedence is:

```text
defaults < TOML card < dedicated CLI options < repeated --set overrides
```

Any license or resource clamp is applied last and reported in the effective
configuration. Unknown fields are errors rather than silently ignored.

Run a card directly:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

Use an explicit action when the same card supplies shared settings:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml
pyamplicol profile --card qq_z6g_recurrence_jit_o2.toml
```

Override one leaf without editing the card:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --set generation.workers=2 \
  --set generation.mode=replace
```

`--set` is deliberately repeatable and order-sensitive. For commonly used
fields, prefer the readable dedicated form—for example `--workers 2`,
`--execution-mode eager`, or `--color-accuracy nlc`.

For the complete field reference, see [Configuration](configuration.md) or create
an annotated local template:

```console
pyamplicol config template pyamplicol.toml
```

## Generate

The shortest built-in-model command is:

```console
pyamplicol generate "d d~ > z g" artifacts/builtin_ddbar_to_zg \
  --model built-in-sm
```

Useful generation options include:

- `--process EXPRESSION` to add another request;
- `--name NAME` to name a request;
- `--multiparticle 'p=d,d~,g'` to define a multiparticle label;
- `--flavor-scheme N` and `--max-quark-lines N` to constrain expansion;
- `--flavor-blind` to generate one subprocess per flavour-relabelling class,
  or `--flavor-blind-strict` to route only the relations that are symmetries
  of the allowed vertices
  (see [Group equivalent subprocesses](#group-equivalent-subprocesses));
- `--analyze-process-groups {none,numerical}`, `--only-unit-multiplier`, and
  `--[no-]consider-charge-conjugation` to search the generated subprocesses
  for equal matrix elements;
- `--color-accuracy {lc,nlc,full}`;
- `--color-contraction {direct,symmetric-group-fft}`;
- `--fft {trace,adjoint}` as a shorthand selecting symmetric-group FFT and its basis;
- `--correlators FILE.json` to prepare named colour operators and allowed spin
  replacements for the [Python and native correlated APIs](../correlators.md);
- `--lc-flow-layout {topology-replay,all-flow-union}`;
- `--execution-mode {recurrence,compiled,eager,on-the-fly}`;
- `--backend {jit,asm,cpp}`;
- `--workers auto|N`;
- `--mode {error,append,replace}` or the `--force` shortcut for replacement;
- `--no-emit-api-bundle` when standalone drivers are not wanted;
- `--numerical-current-reuse` to search for certified numerical current
  relations at generation time (off by default because the high-precision
  probes are costly);
- `--post-build-validation` for an optional immediate native total-versus-
  resolved smoke after writing the artifact.

Planning is non-writing:

```console
pyamplicol generate "p p > Z j j" artifacts/unused \
  --model models/json/sm/sm.json \
  --multiparticle 'p=d,d~,g' --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2 \
  --execution-mode compiled --dry-run
```

Adaptive `--fft adjoint` is the recommended starting point for exact FFT colour
contraction. It selects certified DDM where valid and otherwise retains exact
fundamental-chain or trace tensors. Select contracted colour and
a supported execution lane (`recurrence` or `on-the-fly`) explicitly:

```console
pyamplicol generate "g g > g g g" artifacts/ggg_adjoint_fft \
  --model built-in-sm --color-accuracy full \
  --fft adjoint --execution-mode recurrence
```

Use `--execution-mode on-the-fly` for the compact OTF counterpart. The same
option works for quark processes, using their existing fundamental chains:

```console
pyamplicol generate "d d~ > z g g" artifacts/quark_zgg_fft \
  --model built-in-sm --color-accuracy full \
  --fft adjoint --execution-mode recurrence
```

For a certified `n`-gluon tree, DDM fixes two gluon anchors and retains
`(n-2)!` orderings instead of `(n-1)!`. Certified single-insertion HEFT also
uses DDM: see `examples/builtin_sm_heft.toml`, which explicitly sets `HIG = 1`.
Other interactions retain their exact trace representation; multi-quark-line
processes retain fundamental-chain products, not a JO primitive basis.
The saved `fft_basis_selection` states the requested and actual basis and the
selection reason. `--fft trace` forces the original representation.
Both requests preserve `full` or `nlc` accuracy; forced adjoint FFT rejects correlations.
In the benchmarked pure-gluon family the adjoint basis was faster than trace
for six or more external gluons (3.9x per sample and 4x faster generation at
ten gluons); at lower multiplicity the two are comparable, so compare warmed
evaluation and setup costs for other workloads.
`--color-contraction symmetric-group-fft` remains valid and defaults to adjoint.
Without an explicit contraction flag, NLC/full recurrence and on-the-fly
generation automatically try adaptive adjoint FFT and fall back to direct if
the FFT plan is unsupported. LC, compiled/eager, and correlated generation
remain direct/trace. Use `--color-contraction direct` to opt out.
Do not combine `--fft` and `--color-contraction`; use either spelling. Card
settings are overridden by dedicated flags, then by ordered `--set` options
as usual.

`--dry-run` performs the operation exposed as `Generator.plan()`. It creates no
artifact, output directory, or model-cache entry. A raw UFO or JSON source must
already have a reusable compiled-model cache entry, or be compiled explicitly,
because planning never compiles trusted model input as a side effect.

See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) before
combining execution modes and prepared-model backends.

### Correlated Born quantities

The [Born Correlations guide](../correlators.md#json-and-command-line) gives the
declaration format and complete runtime examples:

```console
pyamplicol generate 'g g > g g' artifacts/gg_correlated \
  --model built-in-sm --color-accuracy full --correlators correlators.json
```

`--correlators` is a generation option, separate from the ordinary TOML colour
settings. `--color-accuracy lc|nlc|full` selects the correlated approximation
(the ordinary default is LC). Complete amplitudes are generated in all cases;
direct contraction and generic compiled execution are selected, with those
adjustments recorded in the effective configuration. Partial
helicity/colour generation, append, and `--dry-run` are not supported on this
path. Numerical spin-vector setting and correlated evaluation use
`Runtime.set_spin_correlation_vectors(...)` and
`Runtime.evaluate_correlated(...)` or `Runtime.evaluate_correlated_many(...)`
in Python, or the corresponding [native SDK calls](native-apis.md#correlated-born-evaluations),
not the CLI `evaluate` command.

## Inspect and select a process

List an artifact's stable process IDs and coverage:

```console
pyamplicol inspect artifacts/pp_zjj
```

Focus on one public process ordering:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

The selector may be a stable process ID, an explicit alias ID, an exact stored
expression, or a unique side-preserving permutation of one. See
[Process Selection and Permutations](process-selection-and-permutations.md) for
the ordering contract.

Inspection does not execute evaluator state. Human output uses compact colored
tables; machine output is available with `--json`.

The `Process groups` section shows the artifact's
[process-group map](process-grouping.md): an overview row, then one table of
groups, each representative in bold followed by its members. Colours mark
`generated` (green) and `routed` (cyan) members, `C` (magenta), factors other
than 1 and warnings (yellow), and the basis of each relation. For example, for
`p p > j j j j` generated with `--flavor-blind --analyze-process-groups
numerical` (abridged):

```text
Process groups

+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+
| groups | subprocesses | generated | routed | conjugated | factor != 1 | bases            | analyses  | warnings |
+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+
|     23 |          486 |        23 |    463 |        140 |           0 | flavor-blind 463 | numerical |        0 |
+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+

+--------------------+---------------------------------------------------------------------------------------------+
| representative     | members                                                                                     |
+--------------------+---------------------------------------------------------------------------------------------+
| d d > d d d d~     |  4 routed    u u > u u u u~      s s > s s s s~      c c > c c c c~      b b > b b b b~     |
| p_p_to_j_j_j_j_1   |  5 routed C  d~ d~ > d d~ d~ d~  u~ u~ > u u~ u~ u~  s~ s~ > s s~ s~ s~  c~ c~ > c c~ c~ c~ |
| 10 subprocesses    |              b~ b~ > b b~ b~ b~                                                             |
+--------------------+---------------------------------------------------------------------------------------------+
...
+--------------------+---------------------------------------------------------------------------------------------+

routed: not generated, evaluated through its representative; C: charge-conjugated relation
compact view of 486 subprocesses; IDs and leg permutations: --json; one group row by row: --process REPRESENTATIVE
numerical analysis: 3 points, tolerance 1e-09
```

`routed` members are not generated and are evaluated through their
representative; `C` marks a charge-conjugated relation. A map with at most 80
subprocesses is listed one member per row, with its process ID, role, and
basis, plus a factor column when a factor differs from 1, a `C` column when a
relation is conjugated, a permutation column when a leg permutation is not the
identity, and a `note` column with `[n]` references to the warnings printed
below the table. Larger maps list each group's members as a grid of process
strings, one block per relation; subprocesses alone in their group are listed
as `ungrouped`. When every process is its own group, the section is the single
line `Process groups: none; each of the N processes is its own group`. With
`--process`, only the selected process's group is shown, so `--process
REPRESENTATIVE` lists a large group member by member. A routed member has no
runtime process of its own: without `--full-physics`, selecting it stops with
an error that names its representative. `--json` reports the
complete map under `process_groups`: `stored`, `analyses`, and per group the
`representative_id`, `representative`, and `members` (`id`, `process`, `pdgs`,
`initial_count`, `factor` as a fraction string, `permutation`, `conjugated`,
`basis`, `generated`, `warning`).

`--process MEMBER --full-physics` on a grouped process adds a `Process Group`
table, and a `process_group` object to the JSON. For a member, it gives the
representative, factor, conjugation and the leg permutation from the
representative to the requested leg order; for a representative, the size of
its group, and that permutation when the requested leg order is not the stored
one:

```text
+-----------------------------------------------------------------------------------------+
|                                      Process Group                                      |
+----------------+------------------------------------------------------------------------+
| field          | value                                                                  |
+----------------+------------------------------------------------------------------------+
| role           | routed member: not generated, evaluated through the representative     |
| representative | p_p_to_j_j_j_j_52  d u~ > d d d~ u~                                    |
| factor         | 1                                                                      |
| conjugated     | yes (C)                                                                |
| permutation    | p = [1 0 4 5 3 2]: representative leg r takes the momentum of leg p[r] |
| basis          | flavor-blind                                                           |
| group          | 40 subprocesses: 1 generated, 39 routed, 20 conjugated                 |
+----------------+------------------------------------------------------------------------+
```

## Group equivalent subprocesses

Every artifact records in `process-groups.json` which subprocesses share one
matrix element, up to a rational factor and a leg permutation. By default each
generated subprocess is its own group.

`generate --flavor-blind` assumes that subprocesses related by a relabelling of
flavours with equal spin, colour, mass, and width (and, unless
`--no-consider-charge-conjugation`, by charge conjugation) are equal. Only one
representative per class is generated; the others are recorded as its members
and exported to UMAMI through the representative's runtime. The assumption is
not verified: a warning names the members whose relabelling is not a symmetry
of the model couplings, for example electroweak couplings of different quark
flavours.

Generation logs the outcome of this pre-grouping before any representative is
generated:

```text
INFO pyamplicol.generation: Flavor-blind grouping: 486 subprocesses -> 23 generated representatives; 463 routed members (140 by charge conjugation); 0 relabelling warnings
```

The summary table printed at the end has a `process grouping` row, coloured
like the `inspect` display, read from the written `process-groups.json`
(abridged):

```text
| processes              | 23 requests                                                                              |
| process grouping       | 486 subprocesses -> 23 groups: 23 generated, 463 routed                                  |
|                        | 463 flavor-blind relations (140 charge-conjugated); 0 warnings                           |
|                        | numerical analysis: 3 points, tolerance 1e-09                                            |
```

Its first line counts the subprocesses, groups, and generated and routed
subprocesses; the second, the relations by basis, those that are
charge-conjugated or have a factor other than 1, and the relabelling warnings;
the last ones, each recorded numerical analysis. The `Generated Processes`
table gains a `group` column: the size of each representative's group, or
`member of REPRESENTATIVE` for a member generated in its own right, followed by
its factor when it is not 1 and `C` when the relation is conjugated, as in
`member of d_dbar_to_a_a (x16)`. When
every subprocess is its own group, the column is omitted, and so is the row
unless an analysis was recorded; the row then reads `no relation found; each of
the N processes is its own group`.

The numerical analysis compares generated subprocesses with equal external
particle types at three random phase-space points and merges those whose
resolved components agree up to one common rational factor:

```console
pyamplicol generate "p p > j j" artifacts/pp_jj --analyze-process-groups numerical
pyamplicol group-processes artifacts/pp_jj --only-unit-multiplier
```

`group-processes` runs the numerical analysis (its default) on an existing
artifact and rewrites `process-groups.json` in place; the artifact ID and the
UMAMI export are unchanged. `--only-unit-multiplier` accepts only relations
with factor 1. Subprocesses generated with a selected helicity or colour subset
are never grouped numerically. [Process Grouping](process-grouping.md)
describes the relations, the warnings and the file format in full.

## Evaluate

Evaluate a momenta file with optional parameter updates:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --model-parameters data/model_parameters.json \
  --momenta data/pp_zjj_momenta.json
```

Request all physical components rather than only the optimized sum:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --resolved
```

Global selectors may be repeated:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process p_p_to_z_j_j_3 \
  --helicity 'h:-1,+1,-1,+1,+1' \
  --color-flow 1 \
  --momenta data/pp_zjj_momenta.json
```

`--color-flow` accepts either a stable flow ID or the one-based ordinal shown
by `inspect`. Flow selection is LC-only; NLC and full-color artifacts expose a
contracted color output and accept helicity selection only. More examples are
in [Runtime and Selectors](runtime-and-selectors.md).

## Profile

The profiler measures the same optimized total path as `Runtime.evaluate()`:

```console
pyamplicol profile artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 \
  --batch-size 128 \
  --color-flow 1 \
  --precision 16
```

It calibrates repetitions, keeps at least the requested independent sample
count, and reports uncertainty. `Ctrl-C` during sampling preserves completed
blocks and prints a result marked partial. See
[Profiling and Benchmarking](profiling-and-benchmarking.md) for metric meanings
and selector-aware profiling. If both profile selector axes are omitted, the
stored LC layout supplies a deterministic optimized selector; explicit
non-hot shapes are retained and produce at most one pre-loop warning per
loaded process.

## Models

Inspect a trusted UFO directory without generating a process:

```console
pyamplicol model inspect models/ufo/sm
```

Compile portable model IR:

```console
pyamplicol model compile models/ufo/sm models/sm.pyAmplicol-model.json
```

Prepare a reusable JIT-O2 kernel bundle for eager or recurrence execution:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2
```

Enumerate a broad request without writing a process artifact:

```console
pyamplicol model processes "p p > all all" \
  --model built-in-sm --flavor-scheme 1 --max-quark-lines 0
```

The distinction between portable model IR and prepared kernel bundles is
explained in [Models and Processes](models-and-processes.md).

## Examples and installed workspaces

List the available names before running one:

```console
pyamplicol examples list
pyamplicol examples run generate_pp_zjj_from_ufo_sm
```

The list uses a colored table on a terminal. Add `--json` for stable,
uncolored machine-readable output.

`examples run` accepts the stem of a shipped `.toml` card. `all_options` is a
reference template, not a runnable example. Unknown names fail with the full
available-name list.

For an editable and inspectable workspace, copying is usually clearer:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

See [Examples Gallery](examples-gallery.md) for the complete card inventory and
recommended sequences.

## Output, color, progress, and logging

The common presentation options are:

| Option | Values | Meaning |
| --- | --- | --- |
| `--json` | flag | Emit stable machine-readable stdout instead of the human-facing table |
| `--color` | `auto`, `always`, `never` | Detect a terminal, force ANSI color, or disable it |
| `--progress` | `auto`, `tty`, `log`, `off` | Select live bars, rate-limited log progress, or silence |
| `--log-level` | `debug`, `info`, `warning`, `error` | Control diagnostic verbosity |

JSON mode keeps stdout machine-readable; progress and diagnostics use stderr.
This makes pipelines predictable:

```console
pyamplicol inspect artifacts/pp_zjj --json > artifact.json
```

Use `--color always` only when the consumer understands ANSI escapes. The
default `auto` mode produces colored terminal tables and plain redirected
output.

## Diagnostics

Start with the two installed-package checks:

```console
pyamplicol doctor
pyamplicol self-test
```

Then inspect the relevant artifact and effective configuration:

```console
pyamplicol inspect artifacts/pp_zjj --json > inspect.json
pyamplicol config resolve run.toml --json > config.json
```

For common failures and the information to include in a bug report, see
[Troubleshooting](troubleshooting.md) and [Release and Support](release-and-support.md).

## Next steps

- [Configuration](configuration.md)
- [Models and Processes](models-and-processes.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Native APIs](native-apis.md)
