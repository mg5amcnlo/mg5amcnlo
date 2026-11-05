################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Data model for the tutorial mode: Step, Tutorial and TutorialSession.

A tutorial is an ordered list of steps.  A step is triggered by a command the
user types; when it triggers, its text is printed and it becomes the current
step.  Two matching policies exist:

  'free'      any step whose key matches fires, wherever it sits in the list.
              This is the behaviour the pre-2026 tutorials had (they were a
              plain command-name -> text lookup) and the ported ones keep it.
  'sequence'  only the current step, or a later one, may fire.  A tutorial that
              teaches the same command several times over (`generate` five
              times with a different lesson each time) needs this.

Nothing here talks to the interface; the mixin does that.
"""

from __future__ import absolute_import

import inspect

import os


class Step(object):
    """One lesson of a tutorial.

    key       command key that triggers the step.  A command line is reduced to
              candidate keys by TutorialSession.keys_for(): 'import model sm'
              gives ('import_model', 'import'), 'generate p p > t t~' gives
              ('generate',).  A step matches if its key is any of them; a
              callable key is called with (keys, line, interface) instead.
    text      printed when the step fires.  Either a string, or a
              callable(interface) -> str for a lesson whose wording depends on
              the machine it runs on (whether madspace is already installed,
              say).
    hint      printed by the `hint` command; falls back to nothing.  May be a
              callable(interface) -> str, for a hint which has to name what
              the reader actually did -- the process they generated, which
              depends on whether they took a detour; resolve it with
              get_hint().
    solution  the command line this step is waiting for.  Printed by `next` and
              `solution` -- never executed, the user always types it.  May be a
              callable(interface) -> str, for a step whose command depends on
              what the user chose earlier (the name they gave `output`, say);
              resolve it with get_solution().
    requires  list of prerequisite names checked before the step is announced.
    setup     callable(interface) run before the step is announced.
    question_hint
              shown under any question MG7 asks while this step is current, in
              place of the generic "type 'help'" line.  May be a
              callable(interface) -> str. The card question a
              `launch` step leads to is the case that matters: it is asked by
              the run interface in the middle of the command, so this is the
              only way a step can say anything there.
    question_progress
              callable(interface, line) -> str or None, called with each answer
              given to such a question and printed under it when it returns
              something.  For a lesson which has more to say than fits at the
              top of a question: the rest is said as the reader gets to it,
              once the answer it follows from has been typed.
    on_failure
              printed when a command which would have triggered this step
              *raised* instead of running, before the generic "that command did
              not run" line.  For the command a lesson invites the user to try
              and which needs an argument they have no reason to guess.  May be
              a callable(interface) -> str.
    entry     for the first step of a side quest or a detour: the command that
              takes the reader into it.  The main line reaches every other step
              through the solution of the one before, but a detour is entered
              by a command the previous step only offers -- and `tutorial skip`
              needs to know it to rebuild the state a detour step expects.
    sticky    the step answers a command without consuming the lesson: its text
              is printed and the session stays where it is, so the same step
              can answer again.  For a lesson which invites the user to try
              several commands -- `display particles`, `display interactions`,
              ... -- none of which is the one it is waiting for.  A sticky step
              has to sit *before* any later step sharing its key, since
              step_for scans forward from the current position and would
              otherwise jump the user to that one.  When the invited commands
              cannot be told apart from the lesson's own by their first two
              words -- `generate ... $ a` against `generate ... / a` -- give it
              a callable key; step_for tries sticky callables first, so it
              still shields the steps behind it.
    gate      callable(interface, line) -> a string saying why the step may
              not fire, or False to refuse it without a word.  Anything else,
              None included, lets it fire, so a check written as `if wrong:
              return "why"` behaves.  For a lesson which only makes
              sense against the state its command was supposed to leave -- the
              bsm tutorial's second lesson reads the model's coupling orders
              and every line of it is false unless what got loaded is an EFT.
              The command has already run: a refused gate prints the string and
              leaves the session where it is, so the reader can load the right
              thing and take the same step again.  Checked by the mixin, not by
              step_for, which has to answer the same way with no interface at
              all (`tutorial index`, `skip N`).
    """

    def __init__(self, key, text, hint=None, solution=None, requires=None,
                 setup=None, title=None, question_hint=None, on_failure=None,
                 sticky=False, entry=None, gate=None, question_progress=None):
        self.key = key
        self.entry = entry
        self.text = text
        self.hint = hint
        self.solution = solution
        self.requires = list(requires) if requires else []
        self.setup = setup
        self.title = title
        self.question_hint = question_hint
        self.question_progress = question_progress
        self.on_failure = on_failure
        self.sticky = sticky
        self.gate = gate

    def refusal(self, interface=None, line=None):
        """Why this step may not fire yet, or None when it may.

        Only an explicit refusal keeps a lesson back: a string (what the
        reader is told), or False.  A gate which returns nothing lets the step
        through, since the shape a check is naturally written in --
        `if wrong: return "why"` -- returns None when it is happy.  So does a
        gate that raises: a lesson is not worth withholding over a broken
        check.
        """

        if self.gate is None:
            return None
        try:
            verdict = self.gate(interface, line)
        except Exception:
            return None
        if isinstance(verdict, str):
            return verdict or None
        if verdict is False:
            return ('This lesson needs what that command was meant to leave '
                    'behind, so the tutorial stays where it is.')
        return None

    def get_failure_advice(self, interface=None):
        """What to say when a command meant for this step did not run."""

        if callable(self.on_failure):
            try:
                return self.on_failure(interface)
            except Exception:
                return None
        return self.on_failure

    def render(self, interface=None, line=None):
        """The text to print for this step.

        A callable text takes the interface, and the command line too when it
        declares a second argument -- which a sticky step needs, since what it
        has to say depends on which command the user tried.
        """

        if not callable(self.text):
            return self.text
        try:
            nb_args = len(inspect.signature(self.text).parameters)
        except (TypeError, ValueError):
            nb_args = 1
        if nb_args > 1:
            return self.text(interface, line)
        return self.text(interface)

    def get_hint(self, interface=None):
        """The hint for this step, resolved against the session."""

        if callable(self.hint):
            try:
                return self.hint(interface)
            except Exception:
                return None
        return self.hint

    def get_solution(self, interface=None):
        """The command this step is waiting for, resolved against the session.

        Falls back to a callable's answer for a missing interface, so `hint`
        and the tests both work before anything has been generated.
        """

        if callable(self.solution):
            try:
                return self.solution(interface)
            except Exception:
                return None
        return self.solution

    def matches(self, keys, line=None, interface=None):
        """True if this step is triggered by a command reduced to `keys`."""

        if callable(self.key):
            return bool(self.key(keys, line, interface))
        if isinstance(self.key, (list, tuple, set)):
            return any(k in keys for k in self.key)
        return self.key in keys

    def __repr__(self):
        return '<Step %s>' % (self.title or self.key)


class Exercise(Step):
    """A step that asks the user to do something and checks whether they did.

    question  what the user is asked to do.  Printed when the exercise becomes
              current, which is when the *previous* step is completed.
    check     callable(interface, line) -> True to pass, False to fail, or a
              string to fail with that message.  It must inspect the state the
              command produced -- interface._curr_amps and friends -- never the
              text of the line: `p p > t t~ QED=0` and `p p > t t~ QCD=2 QED=0`
              are both right, and string matching would fail whoever typed the
              second one.
    mistakes  [(predicate(interface, line), explanation)], tried in order when
              the check fails.  These are the wrong answers worth naming; each
              one is a teaching moment.
    report    optional callable(interface) -> str appended to the pass message,
              for saying what the user's command actually produced.

    A failed attempt never blocks and never advances: the command has already
    run, the user sees what it did, and they can try again, ask for a `hint`,
    or `skip`.
    """

    def __init__(self, key, question, check, mistakes=(), hint=None,
                 solution=None, title=None, praise=None, report=None,
                 question_hint=None):
        Step.__init__(self, key, question, hint=hint, solution=solution,
                      title=title, question_hint=question_hint)
        self.question = question
        self.check = check
        self.mistakes = list(mistakes)
        self.praise = praise
        self.report = report

    def evaluate(self, interface, line):
        """Mark an attempt.  Returns (passed, message)."""

        try:
            verdict = self.check(interface, line)
        except Exception as error:
            return False, ("That command did not leave anything to check "
                           "(%s). Try again, or `skip`." % error)

        if verdict is True:
            message = self.praise or 'That is it.'
            if self.report:
                try:
                    message = '%s\n\n%s' % (message, self.report(interface))
                except Exception:
                    pass
            return True, message

        if isinstance(verdict, str):
            return False, verdict

        for predicate, explanation in self.mistakes:
            try:
                if predicate(interface, line):
                    return False, explanation
            except Exception:
                continue
        return False, self.fallback(interface, line)

    def fallback(self, interface, line):
        """What an answer we did not anticipate gets told.

        This has to be good: it is the only thing standing between an unusual
        wrong answer and a bare "no".
        """

        return ('Not quite -- and not a mistake this exercise knows by name.\n'
                '\nWhat was asked:\n  %s\n'
                '\nWhat your command produced:\n%s\n'
                '\nTry again, or `hint`, or `solution` to see one right '
                'answer, or `skip`.'
                % (self.solution or self.title or 'see the question above',
                   describe_state(interface)))


def describe_state(interface, indent='  '):
    """A short readable summary of what the interface currently holds.

    Used by Exercise.fallback, so an unanticipated answer still gets told what
    it actually did rather than just that it was wrong.
    """

    amps = getattr(interface, '_curr_amps', None)
    if not amps:
        return indent + 'no process is defined'

    lines = []
    for amp in amps[:4]:
        try:
            process = core_process(amp)
            bits = [process.base_string()]
            orders = dict(process.get('orders'))
            squared = dict(process.get('squared_orders'))
            if orders:
                bits.append('orders %s' % _fmt_orders(orders))
            if squared:
                types = dict(process.get('sqorders_types'))
                bits.append('squared orders %s'
                            % _fmt_orders(squared, types))
            bits.append('%d diagrams' % amp.get_number_of_diagrams())
            lines.append(indent + ', '.join(bits))
        except Exception:
            continue
    if len(amps) > 4:
        lines.append(indent + '... and %d more' % (len(amps) - 4))
    return '\n'.join(lines) or (indent + 'nothing that could be summarised')


def lhapdf_configured(interface):
    """Whether this MG7 has a usable lhapdf-config.

    Scale and PDF variations are computed by systematics.py, which imports the
    python lhapdf module and finds it through lhapdf-config
    (mg7/launch.py:_lhapdf_config_path).  Without it the run still produces a
    cross section and its integration error -- only the theory uncertainty is
    missing.  Reported as a tri-state so a tutorial can say "not configured
    here" when it knows, and stay general when it cannot tell.
    """

    try:
        import madgraph.various.misc as misc
    except Exception:
        return None

    try:
        configured = interface.options.get('lhapdf')
    except Exception:
        configured = None

    for candidate in (configured, 'lhapdf-config'):
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return True
        elif misc.which(candidate):
            return True
    return False


def pythia8_available(interface):
    """Whether this MG7 has a usable Pythia8.

    MG5 nulls options['pythia8_path'] at startup when the headers are not where
    it points, so that option alone tells: aMC@NLO builds its own Pythia8
    driver and MadEvent runs Pythia8's main164.
    """

    try:
        options = interface.options
    except Exception:
        return False
    return options.get('pythia8_path') not in (None, '', 'None')


def output_name(interface, default):
    """The directory the user's last `output` actually made.

    A tutorial says "you now have a directory called X", and X has to be the
    name the user chose -- they are not obliged to use the one we suggested.
    `_done_export` is [path, mode]; falls back to the suggested name before any
    output exists.
    """

    try:
        done = getattr(interface, '_done_export', None)
        if done:
            name = os.path.basename(os.path.normpath(done[0]))
            if name:
                return name
    except Exception:
        pass
    return default


def last_run_info(interface=None):
    """The info.json of the most recent run in the output directory, or None.

    Lets a step show the reader their own numbers instead of invented ones.
    None means there is nothing to read -- no output yet, or a run that made no
    events.
    """

    import json

    try:
        done = getattr(interface, '_done_export', None)
        if not done:
            return None
        events = os.path.join(done[0], 'Events')
        runs = [os.path.join(events, name) for name in os.listdir(events)]
        runs = [d for d in runs
                if os.path.isfile(os.path.join(d, 'info.json'))]
        if not runs:
            return None
        latest = max(runs, key=os.path.getmtime)
        with open(os.path.join(latest, 'info.json')) as handle:
            return json.load(handle)
    except Exception:
        return None


def run_line(interface=None):
    """What the last run came to, `**X +- dX pb**`, or '' if there is none.

    Rebuilt the way the run builds it: the channel means summed, their errors
    in quadrature.
    """

    import math

    info = last_run_info(interface)
    try:
        channels = info['channels']
        mean = sum(c['mean'] for c in channels)
        error = math.sqrt(sum(c['error'] ** 2 for c in channels))
    except Exception:
        return ''
    if not mean:
        return ''
    return '**%.4g +- %.2g pb**' % (mean, error)


VI_FAMILY = ('vi', 'vim', 'nvim', 'vim.basic', 'vim.tiny')


def text_editor(interface=None):
    """(editor, where it comes from): the program a card will open in.

    Resolved the way MG7 resolves it when the card is opened
    (misc.open_file.resolve_text_editor), so a tutorial can say in advance
    which editor the reader is about to be put in.  (None, ...) if there is
    none at all.
    """

    import madgraph.various.misc as misc

    try:
        configured = (getattr(interface, 'options', None) or {}).get(
            'text_editor')
    except Exception:
        configured = None
    editor = misc.open_file.resolve_text_editor(configured, quiet=True)
    if configured and editor == configured:
        source = 'your `text_editor` option'
    elif editor and editor == os.environ.get('EDITOR'):
        source = 'your `$EDITOR`'
    else:
        source = ("MG7's first choice when neither `text_editor` nor "
                  "`$EDITOR` names one")
    return editor, source


def is_vi(editor):
    """True for vi and its relatives, which need a word of explanation."""

    if not editor:
        return False
    return os.path.basename(editor.split()[0]) in VI_FAMILY


def model_line(interface=None):
    """What model is loaded, in its own numbers, or '' if none is.

    The counterpart of counts_line() for an `import model` step: the lesson
    after it opens on the model the reader actually loaded rather than on the
    one the text assumed.
    """

    model = getattr(interface, '_curr_model', None)
    if not model:
        return ''
    try:
        name = model.get('name')
        particles = len(model.get('particles'))
        interactions = len(model.get('interactions'))
    except Exception:
        return ''
    if not name:
        return ''
    return ('**%s** is loaded: %d particles, %d interactions.\n'
            % (name, particles, interactions))


def check_line(interface=None):
    """The verdict of the last `check permutation`, or '' if there is none.

    `do_check` keeps the permutation comparisons on the interface
    (`_comparisons`), so the lesson that follows can quote the reader's own
    numbers instead of describing a table they have to trust.  Only the
    permutation check is stored -- gauge, lorentz and flavor print and move on.
    """

    comparisons = getattr(interface, '_comparisons', None)
    if not comparisons:
        return ''
    try:
        results = [r for r in comparisons[0] if len(r.get('values', [])) > 1]
    except Exception:
        return ''
    if not results:
        return ''
    passed = sum(1 for r in results if r.get('passed'))
    worst = max(r.get('difference', 0.0) for r in results)
    return ('**%d/%d passed**, the largest relative difference %.1e.\n'
            % (passed, len(results), worst))


def counts(interface=None):
    """`**N processes with M diagrams**`, or '' when there is nothing to count.

    These are the numbers MG5 has just printed for the command the reader
    typed: `total_diagrams` sums exactly what its own `Total:` line reports,
    decay chains and accumulated `add process` included.
    """

    amplitudes = getattr(interface, '_curr_amps', None) or []
    total = total_diagrams(interface)
    if not total:
        return ''
    text = '**%d process%s with %d diagram%s**' % (
        len(amplitudes), '' if len(amplitudes) == 1 else 'es',
        total, '' if total == 1 else 's')

    # A decay chain's total is production PLUS decays, generated separately
    # (DecayChainAmplitude.get_number_of_diagrams sums them); `output` then
    # stitches them into full diagrams.  `p p > t t~, t > w+ b, t~ > w- b~`
    # prints 6 here and writes 4.  Quote MG5's number, but say what it adds up.
    production = [_production_diagrams(a) for a in amplitudes]
    if production and None not in production:
        made = sum(production)
        text += ' -- %d for the production, %d for the decays' % (
            made, total - made)
    return text


def _production_diagrams(amplitude):
    """The production's own diagrams, for a decay chain; None otherwise."""

    try:
        if not amplitude.get('decay_chains'):
            return None
        return sum(len(a.get('diagrams'))
                   for a in amplitude.get('amplitudes'))
    except Exception:
        return None


def counts_line(interface=None):
    """The counts sentence a lesson opens on, or nothing at all.

    Every step begins with this: the reader is told what the command they just
    typed produced before the next subject starts, rather than being moved on
    from a standing start.  It carries its own newline, so the hand-wrapped
    prose after it starts fresh -- a markup span that wrapped would colour the
    next line's indentation.
    """

    found = counts(interface)
    return '%s.\n' % found if found else ''


def total_diagrams(interface):
    """How many diagrams the current process(es) came to, over all of them."""

    total = 0
    for amplitude in getattr(interface, '_curr_amps', None) or []:
        try:
            total += amplitude.get_number_of_diagrams()
        except Exception:
            continue
    return total


def applied_orders(interface):
    """The coupling orders MG5 actually put on the current process(es).

    Returns a list of the distinct order dicts, so a tutorial can describe what
    really happened rather than assert what usually happens.  MG5 has more than
    one way of settling this -- a minimal-WEIGHTED search, or an explicit order
    -- and which one you get depends on the process.
    """

    seen = []
    for amplitude in getattr(interface, '_curr_amps', None) or []:
        try:
            orders = dict(core_process(amplitude).get('orders'))
        except Exception:
            continue
        if orders not in seen:
            seen.append(orders)
    return seen


def describe_applied_orders(interface, indent=''):
    """Prose for what MG5 chose, matched to what it actually chose."""

    found = [orders for orders in applied_orders(interface) if orders]
    if not found:
        return (indent + 'MG5 put no coupling-order constraint on the process '
                'at all, so every diagram the model allows is included.')

    # an amplitude order means '<=' -- MG5 warns "Interpreting 'QED=2' as
    # 'QED<=2'" and prints WEIGHTED<=2 -- so quote it the way the user just
    # saw it
    shown = ' and '.join(_fmt_orders(orders, default='<=') or 'no constraint'
                         for orders in found[:3])

    if any('WEIGHTED' in orders for orders in found):
        return (indent + 'MG5 settled on **%s**.\n\n'
                % shown +
                indent + 'WEIGHTED counts QCD + 2*QED, and with no orders given '
                'MG5 searches:\nit takes the lowest WEIGHTED that produces any '
                'diagram at all. Here that\nlands on the QCD diagrams. It is a '
                'search, not a statement of physics.')
    return (indent + 'MG5 settled on **%s**.\n\n' % shown +
            indent + 'With no orders given MG5 works out a constraint for you '
            'and applies it\ndirectly. It is a choice made on your behalf, not '
            'a statement of physics.')


def _fmt_orders(orders, types=None, default='='):
    types = types or {}
    return ' '.join('%s%s%s' % (name, types.get(name, default), value)
                    for name, value in sorted(orders.items()))


def core_process(amplitude):
    """The production process of an amplitude, decay chain or not.

    A DecayChainAmplitude has no 'process' of its own -- asking for one raises
    KeyError -- so the core process has to be dug out of its first amplitude.
    """

    try:
        return amplitude.get('process')
    except Exception:
        return amplitude.get('amplitudes')[0].get('process')


# What `tutorial skip N` runs to rebuild the state step N expects: the commands
# that define the process and the directories later steps work in.  The rest
# only print, write a file or take minutes -- a `launch` is a full run and asks
# questions -- and no later step needs their effect, so they are skipped.
REPLAYED_COMMANDS = ('import', 'define', 'set', 'generate', 'add', 'output')


def replay_line(command):
    """The line `tutorial skip` runs for `command`, or None to skip it.

    `output` is forced: the directory may already exist from an earlier pass,
    and the overwrite question would stop the replay.
    """

    words = command.split()
    if not words or words[0] not in REPLAYED_COMMANDS:
        return None
    if words[0] == 'output' and '-f' not in words:
        return command + ' -f'
    return command


# Commands whose second word says which command it is: `display particles` is
# not `display modellist`, `check gauge` is not `check lorentz`.
SUBCOMMANDED = ('display', 'import', 'set', 'add', 'install', 'save')
OUTPUT_FORMATS = ('madevent', 'standalone_fortran', 'standalone_msP',
                  'standalone_msF', 'matrix', 'standalone_rw', 'aloha',
                  'matchbox_cpp', 'matchbox', 'mg7_v5', 'mg7', 'standalone')
CHECKS = ('full', 'timing', 'stability', 'profile', 'permutation', 'gauge',
          'lorentz', 'brs', 'cms', 'flavor', 'language', 'precision')


def command_signature(line):
    """What makes a command the one a lesson asks for, and nothing more.

    The command, and for the ones that have one its sub-command -- the output
    format, the check, the thing displayed.  Not the arguments the reader is
    free to choose: another directory name, another process, another model.
    """

    words = (line or '').split()
    if not words:
        return ()
    command = words[0]
    rest = words[1] if len(words) > 1 else ''
    if command == 'output':
        # `output DIR` is the default format
        return ('output', rest if rest in OUTPUT_FORMATS else 'mg7')
    if command == 'check':
        return ('check', rest if rest in CHECKS else 'full')
    if command in SUBCOMMANDED:
        return (command, rest)
    return (command,)


class Tutorial(object):
    """An ordered list of steps, addressable by name.

    name        primary name, what `tutorial NAME` takes and what the menu shows.
    title       one line for the menu.
    description longer blurb for `tutorial list`.
    steps       list of Step.
    aliases     other accepted names; hidden from the menu.  Used to keep the
                pre-2026 names (MadGraph5 / aMCatNLO / MadLoop) working.
    order       'free' or 'sequence', see the module docstring.
    requires    prerequisites checked when the tutorial starts.
    hidden      keep out of the menu (for tutorials still being written).
    see_also    names of related tutorials, rendered as a "where to go next"
                block.  Names that are not registered are dropped, so a
                tutorial can point at one that has not been written yet
                without ever advertising a dead end.
    section     which group of the menu this belongs to: 'basic',
                'advanced', 'more' or 'exercises'.  'basic' and 'advanced' are
                the developer-validated ones and refuse an AI-generated
                tutorial, which is what lets the menu label them as validated
                without anyone having to keep that claim in step by hand.  The
                default is 'more', so a tutorial lands in the validated groups
                only when someone puts it there on purpose.
    ai_generated
                True when the content was written by an AI and has not been
                validated by the developers.  The menu says so, per section,
                and this is what keeps that notice honest for the tutorials
                carried over from the pre-2026 hand-written text.
    """

    SECTIONS = ('basic', 'advanced', 'more', 'exercises')
    # sections whose heading claims developer validation
    VALIDATED_SECTIONS = ('basic', 'advanced')

    def __init__(self, name, title, steps, description='', aliases=(),
                 order='free', requires=None, hidden=False, see_also=(),
                 section='more', ai_generated=True):
        self.name = name
        self.title = title
        self.description = description or title
        self.steps = list(steps)
        self.aliases = tuple(aliases)
        if order not in ('free', 'sequence'):
            raise ValueError("unknown step order %r" % order)
        self.order = order
        self.requires = list(requires) if requires else []
        self.hidden = hidden
        self.see_also = tuple(see_also)
        if section not in self.SECTIONS:
            raise ValueError('unknown tutorial section %r' % section)
        self.ai_generated = bool(ai_generated)
        if self.ai_generated and section in self.VALIDATED_SECTIONS:
            raise ValueError("tutorial %r is AI-generated and cannot sit in "
                             "the %r section, which the menu presents as "
                             "validated by the developers" % (name, section))
        self.section = section

    @property
    def names(self):
        return (self.name,) + self.aliases

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return '<Tutorial %s (%d steps)>' % (self.name, len(self.steps))


class TutorialSession(object):
    """A tutorial plus where the user has got to in it."""

    def __init__(self, tutorial):
        self.tutorial = tutorial
        self.index = -1          # -1: not started; else index of the last step fired
        self.seen = set()
        # set by do_tutorial for informational sub-commands ('list', 'status')
        # so that they do not re-trigger the intro step
        self.suppress_next = False

    # -- command line -> candidate keys ---------------------------------------

    @staticmethod
    def keys_for(line):
        """Reduce a command line to the keys a step may match.

        Mirrors exactly what madgraph_interface.CmdExtended.postcmd used to do:
        the two-word key first ('import model sm' -> 'import_model', with the
        second word truncated at a dot so 'open index.html' -> 'open_index'),
        then the bare command.  Order matters: the more specific key wins.
        """

        args = line.split()
        if not args:
            return ()
        if len(args) == 1:
            return (args[0],)
        return (args[0] + '_' + args[1].split('.')[0], args[0])

    # -- advancing -------------------------------------------------------------

    def step_for(self, line, interface=None):
        """Return (index, step) for the step triggered by `line`, or None.

        In a sequenced tutorial that has started, only the command the current
        step asks for moves it on -- its solution, or the entry of a detour it
        offers -- and only has to be close to it (command_signature): the
        directory, the process, the model are the reader's to choose, the
        command is not.  A command the lesson merely mentions -- `display
        modellist` in passing -- used to fire the first later step sharing its
        key, skipping the reader lessons ahead.  Sticky steps still answer in
        place, and an exercise still marks whatever answer it is given, from
        the state the command left.

        A free-order tutorial, and the intro, keep the plain lookup (_scan).
        The session is left untouched; advance() commits.
        """

        if self.tutorial.order == 'sequence' and self.index >= 0:
            return self._expected(line, interface)
        return self._scan(line, interface)

    def _expected(self, line, interface=None):
        """step_for in a running sequenced tutorial: see there."""

        keys = self.keys_for(line)
        if not keys:
            return None
        steps = self.tutorial.steps
        ahead = range(self.index + 1, len(steps))

        # a sticky step with a callable key answers a command the lesson
        # invited; it is specific by construction, so it goes first
        for i in ahead:
            if (steps[i].sticky and callable(steps[i].key)
                    and steps[i].matches(keys, line, interface)):
                return i, steps[i]

        following = steps[self.index + 1] if self.index + 1 < len(steps) else None
        # an exercise judges the state the answer produced, never its text
        if isinstance(following, Exercise):
            if following.matches(keys, line, interface):
                return self.index + 1, following
            return None

        signature = command_signature(line)
        # the command this step asks for leads where its solution leads
        solution = self.current.get_solution(interface) if self.current else None
        if solution and command_signature(solution) == signature:
            found = self._scan(solution, interface)
            if found is not None and not found[1].sticky:
                return found
        # or into the detour the step offers
        if (following is not None and following.entry
                and command_signature(following.entry) == signature):
            return self.index + 1, following
        # a step that names no command: the next step's own key says what it
        # waits for -- the next one only, never one further on
        if (not solution and following is not None and not following.sticky
                and following.matches(keys, line, interface)):
            return self.index + 1, following

        # a plain-key sticky step answers in place what is left
        for i in ahead:
            if (steps[i].sticky and not callable(steps[i].key)
                    and steps[i].matches(keys, line, interface)):
                return i, steps[i]
        return None

    def _scan(self, line, interface=None):
        """The plain lookup: the first allowed step whose key matches.

        Candidate keys are tried most-specific first, and for each key every
        allowed step is scanned.  That ordering -- key-major, not step-major --
        is what reproduces the old getattr() chain exactly: 'open index.html'
        resolves to the 'open_index' step even if a plain 'open' step sits
        earlier in the list.  Steps with a callable key are tried last, since
        they cannot be indexed by key -- except a *sticky* one, which is tried
        first (see below).  Also what finds where a step's solution leads.
        """

        keys = self.keys_for(line)
        if not keys:
            return None

        steps = self.tutorial.steps
        if self.tutorial.order == 'free':
            allowed = list(range(len(steps)))
        else:
            # strictly *after* the current step: a sequenced tutorial teaches
            # the same command several times over, so re-matching the step we
            # just fired would pin the user on lesson one forever
            allowed = list(range(self.index + 1, len(steps)))

        # A sticky step exists to answer a command the current lesson invited
        # without ending it, which only works if it wins over the later step
        # that would otherwise swallow that command.  A plain-keyed one wins by
        # sitting at a lower index; a callable-keyed one -- the only way to
        # separate `generate ... $ a` from `generate ... / a`, which share both
        # their keys -- needs the priority stated, since the callable pass runs
        # after the plain one.
        for i in allowed:
            step = steps[i]
            if (step.sticky and callable(step.key)
                    and step.matches(keys, line, interface)):
                return i, step

        for key in keys:
            for i in allowed:
                if callable(steps[i].key):
                    continue
                if steps[i].matches((key,), line, interface):
                    return i, steps[i]

        for i in allowed:
            if callable(steps[i].key) and steps[i].matches(keys, line, interface):
                return i, steps[i]
        return None

    def path_to(self, target):
        """The commands that take a fresh session from its intro to `target`.

        The reader's own route: from each step, the command it asks for -- its
        solution, or for an exercise the answer the next one expects -- until
        the target fires.  When that route jumps straight past the target, the
        target sits in a detour the step before only offers, and the detour's
        `entry` is taken instead.  Returns None when there is no such route: a
        free-order tutorial, a step nothing leads to, a sticky step (it answers
        a command, it is not somewhere to be).
        """

        steps = self.tutorial.steps
        if (self.tutorial.order != 'sequence' or not 0 <= target < len(steps)
                or steps[target].sticky):
            return None

        walk = TutorialSession(self.tutorial)
        walk.index = 0                       # `tutorial NAME` fired the intro
        commands = []
        while walk.index < target:
            following = steps[walk.index + 1]
            if isinstance(following, Exercise):
                command = following.get_solution()
            else:
                command = steps[walk.index].get_solution()
            found = walk.step_for(command) if command else None
            if found is not None and found[0] > target:
                # the main line steps over the target: go in by the detour
                command = following.entry
                found = walk.step_for(command) if command else None
            if found is None or found[0] <= walk.index:
                return None
            commands.append(command)
            walk.advance(found[0])
        return commands if walk.index == target else None

    def advance(self, index):
        self.index = index
        self.seen.add(index)

    @property
    def current(self):
        if 0 <= self.index < len(self.tutorial.steps):
            return self.tutorial.steps[self.index]
        return None

    @property
    def next_step(self):
        if self.index + 1 < len(self.tutorial.steps):
            return self.tutorial.steps[self.index + 1]
        return None

    @property
    def finished(self):
        return self.index >= len(self.tutorial.steps) - 1

    def question_hint(self, interface=None):
        """The hint for whatever step is current, or None.

        The step that *asked* for the command is the current one while that
        command runs, which is what makes this reach the card question a
        `launch` step leads to.
        """

        step = self.current
        hint = step.question_hint if step is not None else None
        if callable(hint):
            # a hint whose wording depends on the machine -- which editor a
            # card will open in, say
            try:
                return hint(interface)
            except Exception:
                return None
        return hint

    def question_progress(self, line, interface=None):
        """What the current step has to say about `line`, answered to a
        question it is watching, or None.

        The step that asked for the command is the current one while that
        command runs, exactly as for question_hint.
        """

        step = self.current
        hook = step.question_progress if step is not None else None
        if not callable(hook):
            return None
        try:
            return hook(interface, line)
        except Exception:
            return None

    def progress(self):
        """(done, total) for the prompt and `status`."""
        return (max(self.index, 0) + (1 if self.index >= 0 else 0),
                len(self.tutorial.steps))
