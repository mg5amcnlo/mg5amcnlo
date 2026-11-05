# Plan: a new tutorial mode for MG7

Status: design proposal, nothing implemented yet.

## 1. Where we are today

The tutorial mode is a *logger-level trick*, not a mode:

| Piece | Location |
|---|---|
| Three text modules | `madgraph/interface/tutorial_text.py`, `tutorial_text_nlo.py`, `tutorial_text_madloop.py` |
| Three loggers | `madgraph_interface.py:117-122` (`tutorial`, `tutorial_aMCatNLO`, `tutorial_MadLoop`) |
| Logger handlers/formatters | `madgraph/interface/.mg5_logging.conf` (+ a copy in `tests/.mg5_logging.conf`) |
| The dispatch | `CmdExtended.postcmd`, `madgraph_interface.py:272-318` |
| The command | `do_tutorial` `:4156`, `check_tutorial` `:1288`, `help_tutorial` `:474`, `complete_tutorial` `:2513`, `_tutorial_opts` `:3143` |
| Master delegation | `master_interface.py:532` (`do_tutorial`), `:363` (`check_tutorial`), `:453` (`complete_tutorial`), `:638` (`help_tutorial`) |

`do_tutorial NAME` raises one logger to `INFO` and drops the other two to `ERROR`.
After every command, `postcmd` builds a key from the command words
(`import model` -> `import_model`, else `generate`) and prints
`getattr(tutorial_text_module, key)` if such an attribute exists.

### Why that design cannot carry the tutorials we want

1. **One lesson per command name.** The lookup key *is* the command, so a
   tutorial can teach `generate` exactly once. Every tutorial requested here
   (syntax, model, standalone, BSM) is a sequence of five to ten `generate` or
   `import model` steps. This is the blocking limitation.
2. **No state.** The engine cannot know which step you are on, cannot repeat a
   step, cannot notice you typed something else, cannot end.
3. **Adding a tutorial is invasive.** A new logger, an edit to two
   `.mg5_logging.conf` files, and a fourth hard-coded `try/except` block in
   `postcmd`.
4. **Text only.** The tutorial cannot change the prompt, add tutorial-only
   commands (`next`, `hint`, `solution`), pre-check a command before it runs,
   set an option needed for the lesson, or refuse a step whose prerequisite is
   missing (`madspace` not installed, no `lhapdf`, ...).
5. **No menu.** `tutorial` with no argument silently means `MadGraph5`.

## 2. Proposed architecture: a tutorial layer on the switcher

Follow the pattern the codebase already uses for MadLoop / aMC@NLO, as
suggested: make the tutorial a real interface with real overrides, driven
through the switcher — not a logger.

### 2.1 The key constraint

`Switcher` (`master_interface.py:56`) holds `self.cmd`, a **class** used for
explicit-self dispatch (`self.cmd.do_generate(self, line)`), and swaps it in
`change_principal_cmd` (`:688`). `interface_names` (`:65`) has three entries:
`MadGraph`, `MadLoop`, `aMC@NLO`.

A tutorial must **not** become a fourth entry there. `Switcher.do_generate`
(`:260`) switches interface based on the process itself — `generate p p > t t~
[QCD]` forces `aMC@NLO` — so a tutorial parked on that axis would be destroyed
by the very command an NLO-syntax lesson asks the user to type. The tutorial
axis is *orthogonal* to the LO/NLO/loop axis.

### 2.2 The mechanism: splice the mixin into the live instance

> **Corrected during implementation.** The first draft of this plan proposed
> wrapping `self.cmd` and hooking `change_principal_cmd`. That does not work,
> and the reason is worth recording. `Switcher.self.cmd` is consulted *only*
> for the methods `Switcher` explicitly forwards; cmd dispatch itself resolves
> everything on the instance — `extended_cmd.Cmd.onecmd_orig` does
> `getattr(self, 'do_' + cmd)`, and `exec_cmd` calls `current_interface.postcmd`.
> So wrapping `self.cmd` would have reached neither `postcmd` nor any command
> the mixin adds (`hint`, `solution`, ...): those are not in the forwarding
> list, so they would never have been found at all.

The mixin goes in front of the **instance's own class** instead:

```python
# madgraph/interface/tutorials/mixin.py
def attach(interface, session):
    if not isinstance(interface, TutorialMixin):
        interface._tutorial_base_class = interface.__class__
        interface.__class__ = _wrap(interface.__class__)   # (TutorialMixin, base)
    interface._tutorial_session = session
```

`detach()` puts the original class back. `_wrap` is memoised per base class, so
attaching twice adds one layer, not two.

This turns out to be *simpler* than the original plan, not more complex: it
needs **no hook in `change_principal_cmd` at all**. The LO <-> NLO switch swaps
`self.cmd`, not the instance class, so a tutorial that crosses
`generate p p > t t~ [QCD]` keeps running with nothing extra — verified
end-to-end, the `syntax` tutorial walks straight through the switch into the
aMC@NLO interface and continues.

The one place the switcher does need to know about tutorials is
`debug_link_to_command` (`master_interface.py:81`), which warns about any
`do_*` on the instance that is not forwarded through `self.cmd`. The mixin's
commands are deliberately not forwarded, so they are added to `to_preserve`.

Because the mixin is only in the MRO while a tutorial is actually running,
tutorial mode costs exactly nothing when it is off — better than the old
version, which ran three `getattr`/`try`/`except` pairs after every command
whether or not anyone was learning.

### 2.3 What the mixin can do (this is the payoff)

| Hook | Use |
|---|---|
| `postcmd` | advance the step, print the next lesson (today's behaviour, now stateful) |
| `precmd` / `onecmd` | see the command *before* it runs: warn "that will take 20 min, use `p p > t t~` instead" |
| `default` | catch a mistyped command and point at the current step instead of a generic error |
| `prompt` | show progress: `MG7 tuto[syntax 4/9]> ` |
| `do_generate`, `do_output`, `do_import`, ... | per-step validation, or auto-set an option the lesson needs (`set group_subprocesses False` before the polarisation step) |
| new `do_next`, `do_back`, `do_repeat`, `do_hint`, `do_solution`, `do_skip` | tutorial-only commands, only reachable while wrapped; they print, never execute |
| `help_*` | append the tutorial's own note to the normal help |
| `preloop` / `setup` | print the intro, check prerequisites |

### 2.4 State: `TutorialSession` + declarative steps

```python
class Step:
    key        # 'generate', 'import_model', 'output_mg7', or a callable/regex
    text       # printed after the step's command succeeds
    hint       # printed by `hint`
    solution   # the exact command line; printed by `solution` and by `next`
    setup      # optional callable run before the step
    requires   # e.g. ['madspace'], checked before the step; degrade with a warning

class Exercise(Step):        # see 2.9
    question   # what the user is asked to do or answer
    check      # callable(interface, line) -> True / False / feedback string
    mistakes   # [(predicate, explanation)] - the known wrong answers, diagnosed

class Tutorial:
    name, title, description        # description feeds the menu
    intro, outro
    steps                           # list[Step]
    requires                        # checked at start
```

`TutorialSession` holds `(tutorial, index)`. Matching rule on `postcmd`:

* command matches the current step -> print its text, advance;
* command matches a *later* step -> jump there and print (user skipped ahead);
* command matches nothing -> print nothing, stay put (decision 3: never
  block, never nag);
* last step -> print `outro`, auto-stop, offer the next tutorial in the chain.

This is what today's design cannot express, and it is the whole reason repeated
`generate` steps work.

### 2.5 Layout

```
madgraph/interface/tutorials/
    __init__.py      registry: name -> Tutorial, discovery, plugin registration
    session.py       TutorialSession, Step, Tutorial
    mixin.py         TutorialMixin (the interface overrides)
    lo.py            ported from tutorial_text.py, then rewritten (§3.1)
    nlo.py           ported from tutorial_text_nlo.py
    madloop.py       ported from tutorial_text_madloop.py
    syntax.py        new
    mg7.py           new
    madevent.py      new
    model.py         new
    standalone.py    new
    bsm.py           new
    ...
```

`tutorial_text*.py` become thin shims re-exporting from the new package for one
release, then get deleted.

### 2.6 The single `tutorial` command with a menu

```
MG7> tutorial
   Which tutorial would you like?
     1. lo          first events at LO, with madspace (the default)
     2. syntax      process syntax: orders, decay chains, $ / , polarisation, NLO
     3. mg7         tuning the MG7 integrator: madspace phase space, MadNIS
     4. madevent    the MG5-compatible MadEvent path
     5. nlo         NLO and aMC@NLO runs
     6. madloop     loop matrix elements with MadLoop
     7. model       working with models: gauge, restrictions, add/customize model
     8. standalone  standalone matrix elements and how to call them from Python
     9. bsm         going beyond the SM: UFO models, widths, validation
    10. exercises   practise: we ask, you answer, we check
     ...
     0. stop        leave tutorial mode
   Enter a number or a name [1]:
```

Built on the existing `cmd.ask()` (`extended_cmd.py:1089`) so completion,
timeout and `--` script mode all behave. **Non-interactive safety:** with no tty,
in `--` script mode, or with `self.force`, `tutorial` with no argument must keep
today's meaning (`MadGraph5`) and never block.

**Naming (decided).** The primary names are `lo`, `nlo`, `madloop`. The old
names `MadGraph5`, `aMCatNLO`, `MadLoop` stay as aliases so existing test
scripts, proc cards and user habits keep working; they resolve to the same
tutorials and are hidden from the menu.

Sub-commands: `tutorial list`, `tutorial status`, `tutorial stop`.

### 2.7 Logging

Everything routes through the existing `tutorial` logger, so
`.mg5_logging.conf` need not change and no new logger is added. The
`tutorial_aMCatNLO` / `tutorial_MadLoop` loggers stay declared (external configs
reference them) but are no longer the transport.

**Format unified (decision 1).** The NLO and MadLoop tutorials use the `simple`
formatter today while the MG5 one uses the boxed `===== Tutorial =====` frame;
all three now get the frame. Since everything goes through the one `tutorial`
logger, this needs no `.mg5_logging.conf` change — but it does mean the M0
"identical output" regression diff will show the added frame on `nlo` and
`madloop`. Land the unification as its own commit so that diff stays readable.

### 2.8 What happens at `launch`

*Rewritten after PR #131.* This section used to describe two different
boundaries, because `launch` on an mg7 output shelled out to
`bin/generate_events` while a madevent output got an in-process child. That
split is gone: both now hand control to a child cmd interface via
`define_child_cmd_interface` (`extended_cmd.py`), in MG5's own process.

What that means for tutorials:

* The tutorial logger reaches the card question. The
  "Need help here? type 'help'" block that `extended_cmd.py:2295` emits at any
  question now appears for an mg7 `launch` exactly as it always did for NLO —
  verified by driving `tutorial lo` to the question under a pty.
* The `MG_TUTORIAL=lo:5` environment-variable handoff this section used to
  propose is unnecessary. There is no process to hand anything to.
* M7 — following a tutorial *into* the run, so a step can react to what the
  user does at the card question — is now a matter of propagating the session
  to the child interface, which is ordinary work rather than a boundary
  problem. It is still not done.

The one thing that has not changed is the pedagogy: `lo` explains the question
*after* the user has answered it, because before the fact it is a wall of text
about a screen they have not seen. That was originally a workaround for the
boundary and turned out to be the better shape anyway.

### 2.9 Exercise mode: ask, check, and diagnose the mistake

The tutorials above are show-and-tell. An **exercise** turns the same material
around: MG7 poses the task, the user answers, and the tutorial tells them
whether they got it — and, when they didn't, *which* mistake they made. This is
the format of the online tutorials, and it is the one thing a logger-based
tutorial could never do, so it is the strongest argument for the interface
design in §2.

**Check the result, not the string.** An exercise must never string-match the
typed line. `p p > t t~ QED=0` and `p p > t t~ QCD=2 QED=0` are both right;
`p p > t t~` with the orders MG5 inferred may also be right. Instead the
checker inspects the state the command produced, which the mixin already has in
hand as `self`:

* `self._curr_amps` and `amp.get_number_of_diagrams()`
  (`madgraph_interface.py:3570` does exactly this for its own log line);
* `amp.get('process')` -> legs, `orders`, `squared_orders`, `sqorders_types`,
  `decay_chains`, polarisation;
* `self._curr_model`, `self._export_format`, `self._done_export`.

So "generate the pure interference term for `p p > j j`" is checked as
*squared_orders == {QCD: 2, QED: 2} with `==` types*, not as a substring.

**Diagnose the mistake.** Each exercise carries a `mistakes` table of
`(predicate, explanation)` pairs evaluated in order when the check fails. These
are worth writing carefully, because the wrong answers are predictable and each
one is a teaching moment:

| Typical wrong answer | What the tutorial says |
|---|---|
| `QED=2 QCD=2` for the interference | those are *amplitude* orders; the interference is a statement about the squared ME, so it needs `^2` |
| `QED^2=2` | a bare `=` means `<=`; you want `==` |
| `p p > t t~ QED=0` when asked for all orders | you excluded the EW diagrams; drop the constraint and compare the diagram count |
| `$` where `$$` was wanted | `$` only removes the on-shell resonance; `$$` forbids the s-channel entirely |
| decay chain written with `>` instead of `,` | the comma opens a decay chain; `>` inside one core process means a required s-channel |
| polarised process generated without `set group_subprocesses False` | it will run, but the polarisation is grouped away — the run card part matters as much as the process line |
| forgetting `install madspace` before `launch` | it will bootstrap itself; here is what that output means |

Anything the table does not match falls through to a generic "not quite — here
is what your command actually produced, and what was asked for", printing the
observed state next to the expected one. That fallback must be good, because it
is what an unanticipated wrong answer gets.

**Never block** (decision 3). A wrong answer is commented on, never rejected:
the command has already run, the user sees what it did, and they can try again,
ask for a `hint`, or `skip`. Same for a right answer arrived at by an unusual
route — the checker passes it and says so.

**Two answer kinds.**

* *Command exercises* — the user types a real MG7 command; the mixin's
  `postcmd` runs the checker on the resulting state. This is the main kind.
* *Quiz answers* — "how many diagrams did that give?", "which gauge do you need
  for this?" — read through the existing `cmd.ask()` (`extended_cmd.py:1089`)
  so completion and timeout behave, with the answer checked against a value the
  tutorial computed itself rather than a hard-coded number (diagram counts
  change with the model).
* *Card and result exercises* — the answer is not a command at all but an edit
  to `param_card.dat` / `run_card.toml`, or a number in a run's output. The
  checker reads the card back (`madgraph/models/check_param_card.py` parses it already)
  or reads the scan summary. Exercise 9 below is the motivating case, and it is
  the one that pushes the `check` callable to be "inspect whatever state you
  need", not "inspect `self._curr_amps`".

**Scope.** Ship this as one `exercises` tutorial first (§3.8), built on the
reusable `Exercise` step type so any tutorial can later opt in to an
`--exercise` variant of its own material.

### 2.10 Only the user's commands count

Found while implementing M3, and it turned out to be a live bug in the old
tutorial too.

MG5 runs plenty of commands for itself. Importing a model alone issues half a
dozen `define` commands (`madgraph_interface.py:6436`), and those went through
the same `postcmd` the user's commands do. In the old tutorial that printed the
`define` lesson **six times in a row** the first time you typed `generate`
(reproduced on HEAD: six blocks). In a *sequenced* tutorial it is far worse — an
internal `define` matched a later step and skipped the user five lessons ahead,
which is exactly what happened on the first run of `syntax`.

The fix is a nesting counter. `exec_cmd` (`extended_cmd.py`) now maintains
`exec_cmd_depth`, and the mixin reacts only at the outermost level. The one
subtlety is command files: `import_command_file` resets the depth around each
line it feeds, because a command file's lines *are* the user's commands however
deep the `import` that reached them was.

  * interactive line: depth 0
  * command-file line: depth 1
  * anything MG5 issues from inside a `do_` method: depth 2 or more, ignored

This is a behaviour change against HEAD — the six duplicate blocks disappear —
so it lands as its own commit, after the byte-identical port.

## 3. The tutorial catalogue

### 3.1 `lo` — first events at LO, on madspace  *(refactored, not just ported)*

Two changes from today's `MadGraph5` tutorial.

**It runs on madspace.** `mg7` is already the default output format —
`check_output(self, args, default='mg7')` (`madgraph_interface.py:1777`), and
`:1990` says so in as many words — so a bare `output NAME` already produces the
MG7/madspace output. The tutorial should stop pretending otherwise, and should
carry the `install madspace` step that makes it work. The MG5-compatible
MadEvent path moves to its own tutorial (§3.3b) that `lo` signposts.

**It stops meandering.** Today's version reaches `launch` at step 4 and then
runs seven more steps through `import model MSSM_SLHA2`, `display particles`,
`customize_model`, `define`, `history`, `open` and `display diagrams` — a
grab-bag of "other useful commands" bolted on past the natural ending. Every one
of those now has a dedicated tutorial. `lo` becomes **the shortest honest path
from a cold start to events on disk**, with a signpost at each step to the
tutorial that goes deep.

Target: seven steps.

0. **`install madspace`** — the one prerequisite. `Step.requires` should detect
   an existing install and skip the step silently rather than making everyone
   sit through it. Mention that `mg7/launch.py` will bootstrap madspace on
   first `launch` anyway, so a user who skips this is not stuck — it just
   happens later and less visibly.
1. **`generate p p > t t~`** — multiparticles `p`/`j`, and the coupling orders
   MG5 adds for you.
   *Signpost:* everything else you can put in a process line — orders,
   interference-only, required and forbidden s-channels, decay chains,
   polarisation, NLO — is `tutorial syntax`.
2. **`display diagrams`** — look before you generate; a cheap habit worth
   building on step two rather than as an afterthought.
   *Signpost:* `tutorial checks`.
3. **`output MY_FIRST_LO_RUN`** — say plainly that this is madspace/MG7 output
   because that is the default, and show what the directory holds:
   `run_card.toml`, `Cards/`, the generated matrix elements.
   *Signposts:* `tutorial madevent` for the MG5-compatible directory layout;
   `tutorial standalone` if you want the matrix element rather than events;
   `tutorial model` / `tutorial bsm` if the physics you want isn't in the
   default SM.
4. **`launch %(run)s`** — since PR #131 this runs in MG5's own process, so
   the tutorial stays with the user throughout and `help` works at the card
   question. The step still says only "press Enter" up front and explains what
   the question held afterwards: before the fact it is a wall of text about a
   screen the reader has not seen. See §2.8.
   *Signposts:* `tutorial mg7` for tuning the integrator and turning on MadNIS;
   `tutorial run` for cuts, scales, PDFs, systematics and showering;
   `tutorial decays` for MadSpin.
5. **Reading the result** — where the cross section is, where the LHE file is,
   `open index.html`.
6. **Outro / session tips** — one compact step absorbing what used to be four:
   `history my_mg5_cmd.dat`, `import command`, `shell` and `!`, and
   `tutorial list` to pick what to learn next.

Nothing is lost, only relocated — the plan should state where each dropped
piece lands so a reviewer can check:

| Dropped from `lo` | New home |
|---|---|
| `import model MSSM_SLHA2`, `display modellist / particles / interactions` | `model` |
| `customize_model`, `customize_model --save` | `model` |
| `define v = w+ w- z a` | `model` (and mentioned in `syntax`) |
| `history`, `import command`, `open`, `shell` / `!` | folded into the `lo` outro step |
| `display diagrams` | promoted to `lo` step 2, expanded in `checks` |
| `output madevent` and everything downstream of it | `madevent` (§3.3b) |

**Content bug to fix while porting:** every existing tutorial text prompts with
`MG5_aMC>`, but the prompt has been `MG7> ` since `MG7_PROMPT`
(`madgraph_interface.py:125`). The ported text must be updated — ideally by
interpolating the live prompt rather than hard-coding it again.

### 3.1b `nlo` and `madloop` (ported)
Content unchanged, new engine, same prompt fix. `lo` signposts to `nlo` at the
point where the user asks "and at NLO?", and `syntax` hands off to it after the
`[QCD]` step.

### 3.2 `syntax` — process generation syntax  *(request 2)*

Content is grounded in `help_generate` (`madgraph_interface.py`) and
`help_polarization`.

1. `generate p p > t t~` — multiparticles `p`/`j`, `display multiparticles`,
   and the coupling orders MG5 picks for you.
2. **Coupling orders.** `QED=0`; `=` vs `==` vs `<=` vs `>` (note the trap:
   bare `=` means `<=`, only `==` means exactly).
3. **Interference only.** Its own step, not a footnote — this is the thing
   people get wrong. Squared-order constraints select a single term of the
   *squared* amplitude:

   * `generate p p > j j` — everything;
   * `generate p p > j j QCD^2==4 QED^2==0` — the pure QCD term;
   * `generate p p > j j QCD^2==2 QED^2==2` — the **interference only**;
   * `generate p p > j j QCD^2==0 QED^2==4` — the pure EW term;
   * and the `COUP^2==-I` = N^(-I+1)LO shorthand for the same expansion.

   Show that the three components sum to the full result — that check is what
   makes the syntax click. Then the caveats, all of which are real and
   currently undocumented for users:

   * a negative order constraint may be given on **one** coupling only, and
     either on squared orders or on amplitude orders, never both
     (`madgraph_interface.py:3515`);
   * interference **with a decay** (a 1 -> N process with squared orders) is
     flagged in the code as not fully validated, with the suggested cross-check
     being to regenerate under `set group_subprocesses True`
     (`madgraph_interface.py:3524`) — the tutorial should pass that warning on
     rather than let users meet it mid-run;
   * `check` does not accept the `^2` syntax, so do not send users to
     `check` to validate an interference process;
   * squared orders drive `split_orders` (`base_objects.py:3607-3616`), which
     is what makes the per-component matrix elements available downstream.
4. **Required s-channels.** `generate p p > w+ > l+ vl`; alternatives with `|`:
   `b b~ > W+ W- | H+ H- > ta+ vt ta- vt~`.
5. **Exclusions.** `$` (exclude on-shell s-channel) vs `$$` (forbid entirely)
   vs `/` (particle forbidden anywhere). Show the diagram-count difference and
   say plainly which one is gauge-safe.
6. **Decay chains.** `generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~`;
   identical particles are *all* decayed; on-shell approximation; when to use
   MadSpin at run time instead. One sentence of warning to plant here and cash
   in at exercise 9: a decay chain's cross section carries a branching ratio
   built from the *card's* widths, so changing a mass without changing the
   width silently corrupts it.
7. **Several processes.** `add process ... @2`, `display processes`,
   what the `@N` tag does to the output directory.
8. **Polarisation.** `generate p p > z{0} z{T}, z > e+ e-`; `{L} {T} {0} {A}`;
   then the run-time part that is easy to get wrong —
   `set group_subprocesses False`, `nhel = 1` and `me_frame` in the run card,
   with the warning that **`me_frame` indexes the normalised leg order, not the
   order in the process string**.
9. **NLO.** `generate p p > t t~ [QCD]`, and `[real=QCD]`, `[virt=QCD]`,
   `[noborn=QCD]` (loop-induced); orders before `[` restrict the Born
   amplitude, orders after `]` restrict the squared ME; no decay chains at NLO.
   Note the automatic interface switch as it happens.
10. `define` your own multiparticle; `display diagrams`; `output`.

This tutorial is the acceptance test for the engine: ten steps, six of them
`generate`, one crossing the LO->NLO interface switch.

### 3.3 `mg7` — tuning madspace and MadNIS  *(request 3)*

**Scope boundary:** `lo` already gets the user through a default madspace run.
This tutorial starts where `lo` ended and is about *tuning* — it should open by
saying so, and assume an mg7 output directory already exists.

1. Recap in one step: madspace is the phase-space + integration engine behind
   the default `output`, VEGAS is the default integrator, MadNIS is the
   trainable alternative.
2. `run_card.toml` section by section — `[run] [gridpack] [beam] [generation]
   [postprocessing] [vegas] [phasespace] [multiparticles] [cuts] [histograms]
   [madnis]` — and which ones a first-time tuner should actually touch.
3. **Phase space**: `[phasespace]` and `[vegas]` knobs, channel handling, and
   how to tell an integration is badly channelled rather than merely slow.
4. **Turning on MadNIS**: `madnis.enable = True`, and what changes in the run.
5. The `[madnis]` knobs in the order a user should touch them:
   `train_batches`, `batch_size_per_channel`, `lr` / `lr_scheduler`, and only
   then the network shape (`flow_*`, `discrete_*`, `cwnet_*`).
6. Running the training, reading the log; CPU vs GPU and
   what torch is needed for.
7. Measuring the win honestly: cross section, unweighting efficiency and wall
   time, VEGAS vs MadNIS vs MadEvent on the same process.
8. `gridpack.py`; `[postprocessing]`; and `RunCardLO_to_MG7_mapping.md` as the
   translation table for users arriving from a LO run card.
9. The rough edge the tutorial should say out loud rather than let users
   discover: the default PDF choice. (`set iseed` being inert for `output mg7`
   was the other one; PR #131 fixed it.)

### 3.3b `madevent` — the MG5-compatible path  *(new)*

For everyone with existing MG5 workflows, and for the features that have not
moved to MG7 yet.

1. `output madevent MY_MADEVENT_RUN` — and why you would ask for it explicitly
   now that `mg7` is the default.
2. The directory layout: `Cards/run_card.dat`, `Cards/param_card.dat`,
   `SubProcesses/`, `bin/generate_events`.
3. `launch MY_MADEVENT_RUN` and the card-editing questions; the same run from
   the shell with `./bin/generate_events`.
4. `run_card.dat` essentials: cuts, scales, PDF, `nhel`, number of events.
5. What plugs in here: MadSpin, `systematics`, `reweight`, Pythia8, Delphes.
6. **Parameter scans**: `scan:[...]` in the param card, the scan summary
   table, and the rule that a scan over a mass must recompute the widths that
   mass feeds — `DECAY <pdg> Auto`, which the scan machinery honours per point
   and reports as a `width#<pdg>` column. This is the material exercise 9
   (§3.8) tests.
7. Gridpacks and cluster/multicore submission.
8. Reading the HTML results and the LHE banner.
9. A short, honest "MG7 vs MadEvent today" section: what each one has, and how
   to move a run between them.

### 3.4 `model` — working with models  *(request 4)*

1. `import model sm`; where models live; `import model /path/to/UFO`;
   `--modelname` to keep the UFO particle names.
2. Inspecting: `display particles`, `display particles t`,
   `display interactions`, `display couplings`, `display parameters`.
3. **Restrictions.** `import model sm-no_b_mass`, `sm-full`,
   `sm-lepton_masses`; what a `restrict_*.dat` does and why the default
   restriction exists.
4. **Gauge.** `set gauge unitary` vs `set gauge Feynman`: goldstones, when
   Feynman gauge is required (loops), and the reload it triggers.
5. `set complex_mass_scheme True`; `set EWscheme`.
6. **Composing models.** `add model taudecay` (`help_add`, `:736`).
7. **`customize_model`** — the interactive restriction editor, and
   `customize_model --save=NAME` to keep the result.
9. `define` multiparticles and `save model`.
9. Widths: `compute_widths`, `decay_diagram`, and when to trust either.
10. Sanity checks after any model change: `check gauge`, `check lorentz`,
    `check permutation`.

### 3.5 `standalone` — standalone output and linking  *(request 5)*

Source material already exists: `docs/standalone_flavor_python.md`.

1. Why standalone: a matrix element as a callable, no MadEvent around it.
2. The formats actually available from `output`: `standalone`,
   `standalone_fortran`, `matrix`, `standalone_msP` / `standalone_msF` /
   `standalone_rw`, `matchbox_cpp`, `mg7`. Say explicitly that
   `output standalone_cpp` was removed (`madgraph_interface.py:1786` raises for
   it) and point at the C++ backends that replaced it.
3. `output standalone_fortran MYPROC --prefix=int` — what `--prefix=int|proc`
   buys you (`M0_SMATRIX`, `PY_M0_GET_VALUE`), and why you need it as soon as
   two process modules share a Python session.
4. `check_sa.f`: building it, feeding it momenta, and pinning a phase-space
   point with `MG_MOMFILE` so two backends can be compared digit by digit.
5. **The f2py path**: building the module, `flavor_dispatch.py`, and calling it.
6. **The flavour selector**: a merged process (`p p > j j`) serves several
   flavour combinations from one ME; index in `[1, NFLAV]` versus the
   per-leg group-position array; index 1 is always valid.
7. Loading a `param_card` into a standalone module.
8. The C++ / cudacpp path for the same process via the `madmatrix` plugin
   (`./bin/madgraph -m madmatrix`), and how to check the two agree.
9. Worked end: a short Python script that computes |M|^2 for a phase-space
   point and cross-checks it against `check_sa`.

### 3.6 `bsm` — beyond the Standard Model  *(request 6)*

1. Getting a model: FeynRules, the model database, `import model MODELNAME`
   with automatic download.
2. First look: `display particles`, `display interactions`, and finding the new
   coupling orders (`NP`, `QED`, ...) in `display couplings`.
3. The `param_card`: setting masses and couplings, and `compute_widths` for the
   new states — including why a hand-set width is often the honest choice, and
   the standing rule that any mass you change invalidates every width that mass
   feeds (`DECAY <pdg> Auto`, or recompute). This bites hardest in BSM scans,
   where nothing warns you.
4. Generating signal: `generate p p > x1 x1~ NP=2`, squared-order selection
   `NP^2==2` for pure BSM versus interference with the SM.
5. Resonances and decays: decay chain versus MadSpin, narrow-width validity.
6. **Validating a new model** — the part users skip and regret:
   `check gauge`, `check lorentz`, `check permutation`, `check brs`,
   `check full`; comparing unitary and Feynman gauge.
7. EFT pitfalls: dimension-six operators and order counting, growth with
   energy, when `set complex_mass_scheme` matters.
8. Restricting a large BSM model with `customize_model` so generation stays
   tractable.

### 3.7 Additional tutorials worth having  *(request 7)*

* **`decays`** — MadSpin and MadWidth: `decay_events`, spin correlations,
  the on-shell approximation, decay chains vs MadSpin vs Pythia, `compute_widths`.
* **`run`** — the physics of a run, backend-agnostic: cuts, scales, PDF choice,
  `nhel`, `systematics`, `reweight`, showering and matching/merging. The
  MadEvent-specific mechanics live in `madevent` (§3.3b) and the MG7 ones in
  `mg7` (§3.3); this tutorial is what is common to both, and should show each
  setting in both card formats.
* **`checks`** — the `check` command family as a debugging tool
  (`gauge`, `lorentz`, `permutation`, `brs`, `cms`, `timing`, `profile`,
  `stability`, `poles`), plus `display diagrams`, `open index.html`,
  and how to file a useful bug report.
* **`performance`** — MG7's speed work: `set group_subprocesses`, helicity
  recycling, crossing (`--use_crossing`), colour basis choice, the C++/cudacpp
  backends, and how to time a change honestly.
* **`install`** — `install` targets (`pythia8`, `lhapdf6`, `madspace`,
  `cudacpp`, ...) and `set` of the associated paths.

**Plugin-provided tutorials.** The plugin API already exposes `new_output`,
`new_cluster` and `new_interface` (see `madmatrix/__init__.py:29-50`); add
`new_tutorial = {'name': TutorialInstance}` alongside them, so an experiment or a
downstream tool can ship its own tutorial without patching MG7. The
`tutorial_plugin` logger (`extended_cmd.py:41`) already hints at that intent.
One caveat to handle: `debug_link_to_command` returns early when `self.plugin`
is set (`master_interface.py:83`), so the wrap hook must not rely on it.

### 3.8 `exercises` — practise, with the answers checked  *(request 6b)*

The engine of §2.9, with content following the online tutorial format. Each
exercise is a task, a check on the produced state, and a diagnosis when it
fails. A first set, deliberately ordered so each one has exactly one new idea:

1. Generate `p p > t t~` and say how many diagrams you got. *(Reads the count
   back from `_curr_amps`, so it stays right when the model changes.)*
2. Generate the same process with the EW diagrams included. *(Catches the
   reflex `QED=0`.)*
3. Generate the **interference only** for `p p > j j`. *(Catches `QED=2 QCD=2`
   and `QED^2=2` — the two mistakes from the table in §2.9.)*
4. Generate `p p > t t~` where the top decays leptonically, as a decay chain.
   *(Catches `>` used where `,` was meant, and the "identical particles are all
   decayed" surprise.)*
5. Generate `p p > e+ e-` with no photon s-channel — first with `/`, then with
   `$`, and explain why the diagram counts differ. *(The `$` / `$$` / `/`
   distinction, learned by contrast rather than by table.)*
6. Generate `p p > z z` with one longitudinal Z. *(Catches the missing
   `set group_subprocesses False`.)*
7. Produce a standalone output for a process of your choice and evaluate it at
   one phase-space point. *(Cross-checks against `check_sa`.)*
8. Take `p p > t t~` to NLO in QCD. *(Notices and explains the interface
   switch as it happens.)*
9. **The capstone: a top-mass scan, with and without the decay.** This one is
   worth building carefully — it is a trap real users fall into, and it cannot
   be spotted from a single run.

   *The task.* Scan the top mass over, say, `scan:[150, 160, 170, 180, 190]` in
   the `param_card` (the syntax is parsed at
   `madgraph/models/check_param_card.py:967-988`), and run it twice: once for
   `p p > t t~` and once for the decayed `p p > t t~, t > w+ b`. Then form the
   ratio of the two cross sections point by point.

   *The trap.* That ratio is a branching fraction, so it must be at most 1. It
   isn't. Changing `MT` in the card does **not** change the top width, which
   stays pinned at its nominal `DECAY 6 1.508...`. The partial width the decay
   chain actually computes from the model grows roughly like `m_t^3`, while the
   total width in the denominator is frozen — so as the scan walks upward the
   ratio climbs through 1 and the "decayed" cross section ends up larger than
   the undecayed one, which is impossible. At the nominal point everything
   looks fine, which is exactly why this survives into people's results.

   *The fix the exercise asks for.* Put the width back under the model's
   control by writing `DECAY 6 Auto` in the param card. The scan machinery
   already understands it — `ParamCardIterator` collects `auto` entries
   (`check_param_card.py:992`) and recomputes them at every point, and it even
   records the resulting width as a `width#6` column in the scan summary
   (`:1035-1039`). So the user can watch the width move with the mass and see
   the ratio drop back below 1. `compute_widths t` before each point, or a
   hand-set width, are acceptable alternative answers and the checker should
   pass them.

   *The generalisation, which is the real lesson.* Any scan that moves a mass
   must recompute every width that mass feeds. The same trap is waiting in any
   BSM scan over a coupling or a mediator mass.

   *Checking it.* Two acceptable checks, and the exercise should use both:
   read the param card back and confirm the width is no longer frozen; and read
   the scan summary and confirm the ratio never exceeds 1. This is the `check`
   callable's "inspect whatever state you need" case (§2.9).

   *Mistakes to diagnose.* Changing `MT` but not the width and concluding the
   BR really is > 1; setting `DECAY 6 Auto` on the wrong PDG code; running only
   the decayed process and never noticing (the exercise must insist on both
   runs, because the trap is invisible without the comparison); fixing it by
   rescaling the answer by hand instead of fixing the card.

   *Practical constraint.* This is the only exercise that requires two full
   runs, so keep it to four or five scan points at low statistics and state the
   expected wall time up front. (Before PR #131 it was also easier to
   instrument on the `madevent` path, because the mg7 scan ran behind a process
   boundary; both now run in process, so that no longer decides it.)

Later sets can follow the same shape for `model` / `bsm` (import a model,
restrict it, validate it with `check`) and for `mg7` (turn on MadNIS and beat
the VEGAS run).

## 4. Milestones

**Status: M0-M9 implemented, bar three tutorials.** Twelve tutorials ship and
walk end to end; `run`, `performance` and `install` are not written. What
landed:

| | |
|---|---|
| `madgraph/interface/tutorials/session.py` | `Step`, `Exercise`, `Tutorial`, `TutorialSession` |
| `.../mixin.py` | the interface layer, `attach`/`detach`, exercise marking |
| `.../__init__.py` | the registry, `see_also_block`, plugin `new_tutorial` |
| `.../_port.py` | the one substitution the ported text is allowed |
| `.../lo.py` `.../syntax.py` `.../mg7.py` `.../madevent.py` | |
| `.../model.py` `.../bsm.py` `.../standalone.py` `.../decays.py` | |
| `.../checks.py` `.../exercises.py` `.../nlo.py` `.../madloop.py` | |
| `tests/unit_tests/interface/test_tutorials.py` | 44 tests |

**The method that made this work**, and the thing to keep doing for the
remaining three: run every command before writing about it. Nothing here was
written from memory of how MG5 behaves, and that caught, among others:

 * MG5 does not add `QED=0` when you give no orders -- it searches for the
   lowest `WEIGHTED = QCD + 2*QED` that yields a diagram;
 * `[madnis] enable` is `"auto"`, not off, and auto also sizes the networks and
   picks the learning rate; there is no `train_madnis` driver, training is a
   phase inside `bin/generate_events`;
 * `run_card.toml` lives in `Cards/`, not at the top of the output;
 * `check` does not load a model for you, unlike `generate`;
 * `add model taudecay` -- the example MG5 itself prints -- fails against the
   shipped models, which are named `taudecay_UFO`;
 * a step must never ask for a command that blocks (`open Cards/...` hands the
   file to an editor and waits).

**Why `run` is not written.** Its distinctive value was showing each setting in
both card formats, and that is exactly the part that keeps turning out wrong:
`ickkw` and `xqcut` are not in the current `Template/LO/Cards/run_card.dat` at
all, and MG7's `[cuts]` is a group/observable scheme
(jet/bottom/lepton/missing/photon x pt/eta/dR/mass/sqrt_s) rather than a list of
named keys. Writing it means verifying both card formats key by key first.
`madevent` and `mg7` each cover their own card in the meantime, and
`see_also_block` drops the dead `run` link on its own.

**M0 — engine, no new content. [DONE]**
`madgraph/interface/tutorials/` with `Tutorial`/`Step`/`TutorialSession`, the
`TutorialMixin`, the `change_principal_cmd` wrap hook (both `MasterCmd` and
`MasterCmdWeb`), and the three existing tutorials ported to steps as `lo`,
`nlo`, `madloop`. `postcmd` in `madgraph_interface.py` loses its three
`try/except` blocks. Acceptance: the three print **exactly the same text at
exactly the same points** as before — capture the current output first and diff
it. Keep this commit content-free so that diff is meaningful; the prompt fix
and the `lo` rewrite are separate commits.

**M1 — command and menu. [DONE]**
`tutorial` with no argument opens the menu; `list` / `status` / `stop`;
aliases kept; non-interactive fallback verified in `--` script mode; help and
completion updated in both `madgraph_interface.py` and `master_interface.py`.

**M2 — tutorial-only commands. [DONE]** `next`, `back`, `repeat`, `hint`, `solution`,
`skip`, plus the progress prompt. These only exist while wrapped. Per decision
4, none of them execute anything — `next` and `solution` print the command and
the user types it.

**M3 — `syntax`. [DONE]** The engine's real acceptance test: repeated `generate`
steps and a step that crosses the LO->NLO interface switch.

**M3b — rewrite `lo` on madspace. [DONE]** Seven steps (eight with the optional
detour), the `install madspace` prerequisite step with detect-and-skip, the
card question explained after the user has answered it (§2.8), signposts, the
`MG5_aMC>` -> live-prompt fix, and the relocation table from §3.1 honoured. Lands after `syntax` so its
first signpost points somewhere real; each later tutorial adds its own signpost
to `lo` as it lands, which keeps `lo` the front door rather than a dead end.

**M3c — `madevent`. [DONE]** The MG5-compatible path, split out of the old `lo`. No
new engine work needed: `launch` without `--interactive` behaves like any other
command.

**M4 — `model` and `bsm`. [DONE]** Share a helper for the `check`/validation steps.

**M5 — `standalone`. [DONE]** Content lifted from `docs/standalone_flavor_python.md`;
add a runnable example script under the tutorial package.

**M6 — `mg7`. [DONE]** Tuning and MadNIS. Needs no new engine work either — see
§2.8: the mg7 `launch` is a subprocess, so this tutorial is written around that
boundary rather than through it.

**M7 — child-interface propagation [NOT DONE]**, which would let a tutorial
teach inside the run rather than around it. Since PR #131 both madevent and
mg7 outputs hand control to an in-process child cmd interface, so this is now
a matter of propagating the session to that child — ordinary work, not a
boundary problem. `decays` was written without it (the MadSpin card is
described rather than edited under guidance) and `run` still wants it.

**M8 — `exercises`. [DONE]** The `Exercise` step type, the check-the-state
convention, the mistake tables, and the first nine exercises (§3.8). Exercises
1-8 depend on `syntax` and `standalone` for their material and on nothing else;
the capstone (9) additionally needs the `madevent` tutorial's scan material and
is the only one that runs anything, so it can land separately if the runtime
turns out to be awkward. The `--exercise` variant of existing tutorials is
deliberately *not* in this milestone.

**M9 — `checks` and `decays` [DONE], plugin registration [DONE]**;
`run`, `performance` and `install` [NOT DONE].

## 5. Testing

* A golden test per tutorial: feed the tutorial's own `solution` lines through
  the interface with `--` scripting and assert the emitted step sequence. This
  makes every tutorial self-checking — if a command in a lesson stops working,
  the test fails.
* An M0 regression test diffing old vs new output for the three ported
  tutorials.
* A no-tty test asserting `tutorial` does not block.
* Keep `tests/.mg5_logging.conf` in sync if any logger changes.
* **Exercises are self-testing, and this is the point.** For every exercise,
  assert that (a) its own `solution` passes its `check`, and (b) every entry in
  its `mistakes` table is reachable — feed the wrong answer, assert that
  diagnosis fires. A mistake entry that can no longer be triggered means the
  syntax changed under us, and the test says so.

## 6. Decisions

1. ~~Unify the tutorial output format?~~ **Decided: yes.** All tutorials use the
   boxed `===== Tutorial =====` frame on the single `tutorial` logger. The NLO
   and MadLoop tutorials gain the frame they lack today; `.mg5_logging.conf`
   needs no change.
2. ~~Naming of the ported tutorials.~~ **Decided:** primary names are `lo`,
   `nlo`, `madloop`; `MadGraph5` / `aMCatNLO` / `MadLoop` remain as hidden
   aliases.
3. ~~How strict should tutorials be?~~ **Decided: comment only, never block.**
   No command is ever refused or rewritten; a wrong or off-script command runs,
   and the tutorial comments on what it did. This holds for exercises too
   (§2.9) — the `strict` flag is dropped from the design.
4. ~~Should `next` execute the step's solution?~~ **Decided: print only.**
   Neither `next` nor `solution` runs anything; the user always types the
   command themselves, which is the whole pedagogy. A step advances when the
   user actually runs a matching command, or on an explicit `skip`.
5. ~~Does the mg7 tutorial gate on madspace being installed?~~ **Decided:**
   offer to install, and do it in `lo` — `install madspace` is step 0 there,
   detected and skipped when madspace is already present.

Still open:

6. **Should `lo` name itself `lo` in the menu, or something friendlier?** `lo`
   is accurate but tells a newcomer nothing; the menu line can carry the
   meaning instead. Currently keeping `lo`, per decision 2.
7. **How far does exercise mode eventually go?** §2.9 ships one `exercises`
   tutorial on a reusable step type. Whether every tutorial later grows an
   `--exercise` variant of its own material is a call to make once the first
   set has been used by real people.
