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
"""The interface layer of the tutorial mode.

While a tutorial runs, `TutorialMixin` is spliced in front of the *instance's*
own class -- attach() rebinds interface.__class__ -- rather than in front of
the class the switcher dispatches to.  That is deliberate and it is the only
thing that works: cmd dispatch resolves `do_XXX`, `postcmd` and `default` on
the instance (extended_cmd.Cmd.onecmd_orig does `getattr(self, 'do_' + cmd)`),
while Switcher.self.cmd is only consulted for the commands Switcher explicitly
forwards.  Wrapping self.cmd would therefore reach neither postcmd nor any
command the mixin adds.

Splicing at the instance also makes the tutorial automatically survive
Switcher.change_principal_cmd: the LO <-> NLO switch swaps self.cmd, not the
instance class, so a tutorial that crosses `generate p p > t t~ [QCD]` keeps
running with no extra hook.

The mixin never blocks a command: a wrong or off-script command runs exactly as
it would outside tutorial mode, and the tutorial only comments on it.
"""

from __future__ import absolute_import

import logging

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.various.misc as misc
from madgraph.interface.tutorials._style import to_terminal
from madgraph.interface.tutorials.session import Exercise, replay_line

logger_tuto = logging.getLogger('tutorial')
logger = logging.getLogger('madgraph')


def emit(text):
    """Print one tutorial block, in the format the tutorial logger frames."""

    logger_tuto.info(to_terminal(text).replace('\n', '\n\t'))


# The prompt while a tutorial runs says where the reader is -- `TUTO [syntax]
# 3/14>` -- in the colours of the MG7 prompt it replaces.
TUTORIAL_PROMPT = "\001\033[1;94m\002TUTO [%s] %d/%d> \001\033[0m\002"


def tutorial_prompt(session):
    """The prompt for `session`: its name, and step X of Y (1-based)."""

    done, total = session.progress()
    return TUTORIAL_PROMPT % (session.tutorial.name, done, total)


class TutorialMixin(object):
    """Interface overrides active only while a tutorial is running."""

    # the session is stored on the interface instance as _tutorial_session
    # and the pre-tutorial class as _tutorial_base_class

    # -- the hook that advances the tutorial ----------------------------------

    def notify_failed_command(self, line):
        """The user's command raised instead of running.

        Say so and repeat what the tutorial is waiting for: the alternative is
        an error message followed by silence, which reads as if the tutorial
        had stopped working. The step is *not* advanced -- the command did not
        do what the lesson asked -- and the line is remembered so that postcmd,
        which the interactive path still reaches afterwards, does not advance
        it either.
        """

        super(TutorialMixin, self).notify_failed_command(line)

        session = getattr(self, '_tutorial_session', None)
        if session is None or getattr(self, 'exec_cmd_depth', 0) > 0:
            return
        if getattr(self, '_tutorial_failed_line', None) is not None:
            # a command of a script failed, and the 'import' which was running
            # it fails in turn: one report is enough
            return
        self._tutorial_failed_line = line

        # what the current step waits for is its own solution (before the
        # intro, there is no current step yet); next_step's is the one after
        expected = None
        step = session.current or session.next_step
        if step is not None:
            expected = step.get_solution(self)

        # a step which knows why this particular command fails says so first
        advice = None
        found = session.step_for(line, self)
        if found is not None:
            advice = found[1].get_failure_advice(self)

        text = 'That command did not run, so the tutorial stays where it is.'
        if advice:
            text = '%s\n%s' % (advice, text)
        if expected:
            text += '\nIt is still waiting for:\n  %s' % expected
        text += "\nType `hint` for a hint, or `tutorial stop` to leave."
        emit(text)

    def postcmd(self, stop, line):
        stop = self._tutorial_postcmd(stop, line)
        # after every command, not only the ones that move the tutorial: the
        # LO/NLO switch resets the prompt, and `back`/`skip` move without a step
        # firing
        session = getattr(self, '_tutorial_session', None)
        if session is not None:
            self.prompt = tutorial_prompt(session)
        return stop

    def _tutorial_postcmd(self, stop, line):
        stop = super(TutorialMixin, self).postcmd(stop, line)
        if stop is False:
            return False

        session = getattr(self, '_tutorial_session', None)
        if session is None:
            return stop

        # the command raised: notify_failed_command has already spoken, and a
        # step must not be advanced by a command which did not run
        failed = getattr(self, '_tutorial_failed_line', None)
        if failed is not None:
            self._tutorial_failed_line = None
            if failed == line:
                return stop

        # Only react to what the user actually typed.  MG5 runs plenty of
        # commands for itself -- importing a model issues half a dozen 'define'
        # commands, `display diagrams` issues an `open` -- and those used to
        # fire tutorial steps, printing the same block six times over and, in a
        # sequenced tutorial, skipping the user several lessons ahead.
        # exec_cmd tracks the nesting depth and a user command sits at 0, both
        # interactively and from a command file; anything deeper is MG5 talking
        # to itself.
        if getattr(self, 'exec_cmd_depth', 0) > 0:
            return stop

        if session.suppress_next:
            session.suppress_next = False
            return stop

        found = session.step_for(line, self)
        if found is None:
            return stop

        index, step = found
        # a lesson built on state the command was meant to leave: it did not,
        # so say what is missing and stay where we are (Step.gate)
        refusal = step.refusal(self, line)
        if refusal is not None:
            emit(refusal)
            return stop
        if step.setup:
            step.setup(self)

        if step.sticky:
            # answers the command without consuming the lesson: the session
            # stays where it is, so the next one is answered too
            emit(step.render(self, line))
            return stop

        if isinstance(step, Exercise):
            passed, message = step.evaluate(self, line)
            if not passed:
                # never blocks and never advances: the command has already run,
                # so the user can simply try again
                emit(message)
                return stop
            session.advance(index)
            emit(self._tutorial_join(message, session))
            return stop

        session.advance(index)
        emit(step.render(self))
        return stop

    def _tutorial_join(self, message, session):
        """A passed exercise's verdict, followed by whatever comes next.

        An exercise is triggered by the user's answer, so the *next* question
        has to be printed here rather than waiting for a command that would
        trigger it.
        """

        following = session.next_step
        if following is None:
            return '%s\n\nThat was the last one.' % message
        if isinstance(following, Exercise):
            return '%s\n\n%s' % (message, following.question)
        # a plain step after the exercises: the closing text.  Nothing will
        # ever trigger it, so show it now and mark the tutorial finished.
        session.advance(session.index + 1)
        return '%s\n%s' % (message, following.render(self))

    # -- tutorial-only commands ----------------------------------------------
    #
    # None of these execute anything: `next` and `solution` print the command
    # and the user types it.

    def do_hint(self, line):
        """Not in help: show a hint for the current tutorial step"""
        step = self._tutorial_step_or_warn()
        if step is None:
            return
        solution = step.get_solution(self)
        hint = step.get_hint(self)
        if hint:
            emit(hint)
        elif solution:
            emit("Try:\n%s%s" % (self._tutorial_prompt_text(), solution))
        else:
            emit("No hint for this step -- try 'solution'.")

    def do_solution(self, line):
        """Not in help: show the command the current tutorial step expects"""
        self._tutorial_show_solution(self._tutorial_step_or_warn())

    def do_next(self, line):
        """Not in help: show the next command of the tutorial"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        step = session.current or session.next_step
        self._tutorial_show_solution(step)

    def do_repeat(self, line):
        """Not in help: print the current tutorial step again"""
        step = self._tutorial_step_or_warn()
        if step is not None:
            emit(step.render(self))

    def do_back(self, line):
        """Not in help: go back one tutorial step"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        if session.index <= 0:
            emit("You are at the first step of this tutorial.")
            return
        session.index -= 1
        emit(session.current.render(self))

    def do_skip(self, line):
        """Not in help: skip the current tutorial step, or `skip N` to go to
        step N"""
        session = self._tutorial_session_or_warn()
        if session is None:
            return
        if line.strip():
            return self._tutorial_jump(session, line.strip())
        step = session.next_step
        if step is None:
            emit("That was the last step of this tutorial.\n"
                 "Type 'tutorial' to pick another one, or 'tutorial stop'.")
            return
        session.advance(session.index + 1)
        emit(step.render(self))

    def _tutorial_jump(self, session, number):
        """`skip N`: go to step N, rebuilding the state it expects.

        The commands the reader would have typed to get there are run for them
        -- the process definitions and the outputs; a `launch` or a `check`
        only prints or takes minutes, and no later step needs it -- and then
        step N is shown as if its own command had just fired it.
        """

        steps = session.tutorial.steps
        try:
            target = int(number) - 1
        except ValueError:
            emit("`skip` takes a step number -- `tutorial index` lists them.")
            return
        if not 0 <= target < len(steps):
            emit("There is no step %s: this tutorial has %d. `tutorial index` "
                 "lists them." % (number, len(steps)))
            return
        if steps[target].sticky:
            emit("Step %d answers a command in place rather than being a "
                 "lesson of its own, so there is nothing to go to. `tutorial "
                 "index` lists the steps." % (target + 1))
            return

        # an exercise is current while the reader answers it, which is once
        # the step before it has passed: rebuild that, then show the question
        exercise = isinstance(steps[target], Exercise)
        reach = target - 1 if exercise else target

        commands = []
        if session.tutorial.order == 'sequence' and reach > 0:
            commands = session.path_to(reach)
            if commands is None:
                emit("Step %d cannot be reached by replaying the commands "
                     "before it -- it is shown when the step before is "
                     "completed. `tutorial index` lists the steps."
                     % (target + 1))
                return

        ran, skipped = [], []
        for command in commands:
            line = replay_line(command)
            if line is None:
                skipped.append(command)
                continue
            try:
                self.exec_cmd(line, printcmd=False, precmd=True)
            except Exception as error:
                emit("Getting to step %d stopped at `%s`: %s\nThe tutorial "
                     "stays where it was." % (target + 1, command, error))
                return
            ran.append(line)

        report = "Step %d of %d." % (target + 1, len(steps))
        if ran:
            report += " To get here, this ran:\n" + "\n".join(
                "  %s" % line for line in ran)
        if skipped:
            report += ("\nand skipped what only prints or runs, which no "
                       "later step needs:\n" + "\n".join(
                           "  %s" % line for line in skipped))
        session.advance(reach)
        emit(report)
        emit(steps[target].question if exercise
             else steps[target].render(self))

    # -- helpers ---------------------------------------------------------------

    def _tutorial_session_or_warn(self):
        session = getattr(self, '_tutorial_session', None)
        if session is None:
            logger.warning('No tutorial is running.')
        return session

    def _tutorial_step_or_warn(self):
        session = self._tutorial_session_or_warn()
        if session is None:
            return None
        step = session.current
        if step is None:
            step = session.next_step
        return step

    def _tutorial_show_solution(self, step):
        if step is None:
            return
        solution = step.get_solution(self)
        if not solution:
            emit("This step has no single command to give -- read it again "
                 "with 'repeat'.")
            return
        emit("The tutorial expects:\n%s%s"
             % (self._tutorial_prompt_text(), solution))

    @staticmethod
    def _tutorial_prompt_text():
        """The bare prompt, for quoting commands in tutorial text."""
        import madgraph.interface.madgraph_interface as mg
        return mg.MG7_PROMPT_TEXT


#===============================================================================
# splicing the mixin in and out of a live interface
#===============================================================================

_WRAPPED = {}       # base class -> class with the mixin in front


def _wrap(base):
    if base not in _WRAPPED:
        _WRAPPED[base] = type('%sTutorial' % base.__name__,
                              (TutorialMixin, base), {})
    return _WRAPPED[base]


def is_attached(interface):
    return isinstance(interface, TutorialMixin)


def attach(interface, session):
    """Start `session` on `interface`, splicing the mixin in if needed."""

    if not is_attached(interface):
        interface._tutorial_saved_prompt = getattr(interface, 'prompt', None)
        interface._tutorial_base_class = interface.__class__
        interface.__class__ = _wrap(interface.__class__)
        _suspend_crash_on_error(interface)
    interface._tutorial_session = session
    _arm_question_hooks(session, interface)
    return session


def detach(interface):
    """Stop any running tutorial and restore the plain interface class."""

    session = getattr(interface, '_tutorial_session', None)
    interface._tutorial_session = None
    if is_attached(interface):
        base = getattr(interface, '_tutorial_base_class', None)
        if base is not None:
            interface.__class__ = base
        interface._tutorial_base_class = None
        _restore_crash_on_error(interface)
        saved = getattr(interface, '_tutorial_saved_prompt', None)
        if saved is not None:
            interface.prompt = saved
        interface._tutorial_saved_prompt = None
    _disarm_question_hooks()
    return session


def _suspend_crash_on_error(interface):
    """Turn crash_on_error off for the life of the tutorial.

    A tutorial is a place to make mistakes -- that is what the exercises are
    for -- and with crash_on_error set, a mistyped command does not just fail,
    it tears down the whole session. Suspend it while a tutorial runs and put
    the user's setting back on `tutorial stop`.

    TMP_variable installs the new value on construction and restores it in
    __exit__, so it can span the tutorial rather than a single block; it
    addresses `options` by key, leaving any other option the user changes
    meanwhile alone.
    """

    interface._tutorial_crash_guard = None
    options = getattr(interface, 'options', None)
    if not isinstance(options, dict) or 'crash_on_error' not in options:
        return
    if not options['crash_on_error']:
        return          # already off; nothing to suspend or restore
    interface._tutorial_crash_guard = misc.TMP_variable(
        options, 'crash_on_error', False)


def _restore_crash_on_error(interface):
    guard = getattr(interface, '_tutorial_crash_guard', None)
    if guard is not None:
        guard.__exit__(None, None, None)
    interface._tutorial_crash_guard = None


def mixin_command_names():
    """`do_XXX` names the mixin adds -- Switcher.debug_link_to_command skips
    these, since they are deliberately not forwarded through self.cmd."""

    return [name for name in vars(TutorialMixin) if name.startswith('do_')]


def _arm_question_hooks(session, interface=None):
    """Let the running tutorial speak at any question, and stop the clock.

    Both are module-level switches in extended_cmd because a question is often
    asked by an object the mixin is not attached to -- the launch card question
    belongs to the run interface, not to the command the user typed `launch`
    at.
    """

    extended_cmd.question_hint = lambda: _question_hint(session, interface)
    extended_cmd.question_progress = \
        lambda line: _question_progress(session, line, interface)
    extended_cmd.suppress_timeout = True


def _disarm_question_hooks():
    extended_cmd.question_hint = None
    extended_cmd.question_progress = None
    extended_cmd.suppress_timeout = False


def _question_hint(session, interface=None):
    """The current step's hint, styled, or None to keep the generic line."""

    hint = session.question_hint(interface)
    return to_terminal(hint) if hint else None


def _question_progress(session, line, interface=None):
    """What the current step says about an answer just given, styled."""

    text = session.question_progress(line, interface)
    return to_terminal(text) if text else None
