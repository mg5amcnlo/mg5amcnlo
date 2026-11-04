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
"""The `lo` tutorial -- from a cold start to events on disk.

The shortest honest path to a leading-order event sample, on madspace (which is
what `output` produces by default), with a signpost at each step to the
tutorial that goes deeper.  Everything the pre-2026 version taught after
`launch` -- loading a model, restricting it, defining multiparticles -- has
moved to the tutorials that own it.
"""

from __future__ import absolute_import

import json
import math
import os

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  describe_applied_orders,
                                                  is_vi, last_run_info,
                                                  text_editor,
                                                  lhapdf_configured,
                                                  output_name,
                                                  total_diagrams)

P = 'MG7>'
RUN = 'MY_FIRST_LO_RUN'


def _mg7_bootstrap():
    """The mg7 bootstrap module, or None if it cannot be imported.

    Everything the tutorial wants to know about madspace goes through it, so
    the tutorial cannot disagree with the launcher about where madspace is or
    whether it is there (see madgraph/mg7/bootstrap.py).
    """

    try:
        from madgraph.mg7 import bootstrap
    except Exception:
        return None
    return bootstrap


def madspace_is_installed():
    """True if this MG7 already has a usable madspace build."""

    bootstrap = _mg7_bootstrap()
    if bootstrap is None:
        return False
    try:
        return bootstrap.madspace_is_installed()
    except Exception:
        return False


def _import_madspace():
    """This MG7's own compiled madspace, or None if it has none.

    A bare ``import madspace`` here would resolve against whatever happens to
    be importable -- a system-wide copy, or the repo's own madspace/ source
    directory as a namespace package -- and cache that wrong module in
    sys.modules, where mg7/launch.py's ``import madspace as ms`` would then
    pick it up and fail on the first attribute the real package has. So only
    import when the bundled build is actually installed, and with its install
    prefix on sys.path, exactly as the launcher does.
    """

    bootstrap = _mg7_bootstrap()
    if bootstrap is None or not madspace_is_installed():
        return None
    try:
        # Already installed, so this only prepends the install directory.
        bootstrap.ensure_madspace()
        import madspace
    except Exception:
        return None
    finally:
        # Same reason launch.py drops it: the install directory also carries
        # madspace's build dependencies and would shadow the caller's.
        try:
            bootstrap.drop_install_path()
        except Exception:
            pass
    return madspace


# Shown under the card question `launch` asks, in place of the generic
# "type 'help'" line. It has to be short: it sits between the question and the
# prompt, and the reader is mid-command.
LAUNCH_QUESTION_HINT = """
The two cards worth knowing here:

  param_card.dat   masses, widths and couplings
  run_card.toml    beams, cuts, number of events, integrator settings

The rest belong to whichever programs are switched on above them.
`tutorial mg7` goes through the run card properly.
%(editor)s
**For this tutorial the defaults are fine -- just press Enter.**
"""


def _editor_at_question(interface=None):
    """Which editor a card opens in -- said before the reader opens one."""

    editor, source = text_editor(interface)
    if not editor:
        return ("\nNo text editor was found, so a card cannot be opened from "
                "here;\n`set text_editor nano` (or emacs, code, ...) at the "
                "MG7 prompt fixes that.\n")
    text = "\nA card you open comes up in **%s** -- that is %s." % (
        editor, source)
    if is_vi(editor):
        text += ("\nIf vi is new to you, type `help vi` here first: it lists "
                 "the handful of\nkeys it takes to edit a card and get out "
                 "again.")
    return text + "\n"


def _launch_question_hint(interface=None):
    return LAUNCH_QUESTION_HINT % {'editor': _editor_at_question(interface)}


def _editor_back_at_the_prompt(interface=None):
    """How to change the editor, once the run is over."""

    editor, source = text_editor(interface)
    opened = ("Cards open in **%s** here -- that is %s." % (editor, source)
              if editor else "No text editor was found for opening cards.")
    return """
%s For another
editor:

  set text_editor nano         for this session (or emacs, code, ...)
  save options text_editor     to keep it for the next ones
""" % opened


def _detour_text(interface=None):
    """The optional QED<=2 side-trip: say what changed, then send them back.

    Reached only if the user takes it; typing `display diagrams` at the
    previous step jumps straight past this one.
    """

    before = getattr(interface, '_tutorial_lo_ndiag', None)
    now = total_diagrams(interface)

    if before and now and now != before:
        count = ("You had **%d** diagrams a moment ago and now you have "
                 "**%d**." % (before, now))
    elif now:
        count = "You now have **%d** diagrams." % now
    else:
        count = ""

    return """
%(count)s

The new ones are all in the quark-antiquark subprocess. Without QED vertices
it can only go through a gluon; allowing two lets it go through a photon or a
Z as well, so `q q~ > t t~` picks up the two s-channel electroweak diagrams.
The gluon-gluon subprocess is unchanged -- there is no electroweak way to make
a top pair out of two gluons at this order.

Whether that matters is a physics question, not a syntax one. Here the
electroweak contribution is tiny next to QCD, which is exactly why MG5's search
left it out. For a process where it is not tiny -- anything with a Z or a W in
the final state -- leaving it to the search is how you quietly compute the
wrong thing.

That is the detour. Back to the main line:
%(p)s display diagrams
""" % {'count': count, 'p': P}


# shared with the other tutorials that run something
_last_run_info = last_run_info


def _result_row(info):
    """The `Result:` row the run printed, in its own notation.

    The cross section and its error are rebuilt the way the run builds them --
    the channel means summed, their errors in quadrature -- and formatted by
    madspace itself rather than by a second copy of its digit-of-error rule.
    """

    if not info:
        return '503.1(1.4)'
    try:
        channels = info['channels']
        mean = sum(c['mean'] for c in channels)
        error = math.sqrt(sum(c['error'] ** 2 for c in channels))
    except Exception:
        return '503.1(1.4)'
    if not mean:
        return '503.1(1.4)'
    madspace = _import_madspace()
    if madspace is not None:
        try:
            return madspace.format_with_error(mean, error)
        except Exception:
            pass
    # madspace not importable here: say the same thing the long way
    return '%.4g +- %.2g' % (mean, error)


def _systematics_rows(info):
    """The Scale/PDF rows of the run's Systematics box, or illustrative ones.

    The percentages come from systematics_summary, the same helper the run's
    own box and the scan summary use, so the tutorial cannot quote a number
    computed a third way.
    """

    illustrative = ('    Scale variation:   +12.4%    -9.6%\n'
                    '    PDF variation:     +2.1%    -2.1%')
    if not info:
        return illustrative
    try:
        from madgraph.mg7 import systematics_summary
    except Exception:
        return illustrative

    summary = info.get('systematics')
    if not summary:
        return illustrative
    rows = []
    scale = systematics_summary.scale_percentages(summary)
    if scale:
        rows.append('    Scale variation:   +%.3g%%    -%.3g%%' % scale)
    for _entry, up, down in systematics_summary.pdf_percentages(summary):
        rows.append('    PDF variation:     +%.3g%%    -%.3g%%' % (up, down))
        break
    return '\n'.join(rows) if rows else illustrative


def _lhapdf_note(interface=None):
    """The theory-uncertainty caveat, matched to whether LHAPDF is here.

    madspace evaluates the PDF members itself, but it still has to *have* the
    set: it resolves and, if need be, downloads it through lhapdf-config
    (mg7/launch.py, ensure_pdf_set). Without one the scale variations still
    work -- they need no new PDF -- but the error members cannot be found, and
    the run says so rather than failing.
    """

    if lhapdf_configured(interface):
        return ""
    return ("\n**Expect no PDF row here**: that half needs the error set on "
            "disk, and finding\nit needs LHAPDF, which this MG7 does not have. "
            "`install lhapdf6` fixes it. The\nscale row and the cross section "
            "are unaffected.\n")


def intro(interface=None):
    """The welcome text, which depends on whether madspace is already here."""

    head = """
Welcome. This tutorial takes you from nothing to a leading-order event sample,
in about six commands, and points you at the tutorial that goes deeper at each
step along the way.

As you go: `hint` and `solution` print the command a step expects -- they never
run it for you, you always type it. `repeat` prints the step again, `skip`
moves on, `tutorial status` shows how far you have got, and `tutorial stop`
leaves at any time. `tutorial help` lists the lot.
"""

    if madspace_is_installed():
        return head + """
The integration engine, madspace, is already installed here, so we can go
straight to physics. Pick a process:
%(p)s generate p p > t t~

`p` is a multiparticle label -- shorthand for a set of particles. A space
between particle names is mandatory.
""" % {'p': P}

    return head + """
One thing first. Events are integrated by madspace, and this MG7 does not have
it built yet:
%(p)s install madspace

(You can skip this -- the run itself will build madspace the first time it
needs it. Doing it now just means you see the build separately from the
physics, which is easier to read if anything goes wrong.)
""" % {'p': P}


tutorial = Tutorial(
    name='lo',
    title='first events at leading order',
    description='from a cold start to an event sample, with madspace',
    aliases=('MadGraph5',),
    section='basic',
    ai_generated=False,
    order='sequence',
    see_also=('syntax', 'madevent', 'mg7', 'model', 'bsm', 'standalone',
              'nlo', 'run', 'decays', 'checks', 'exercises'),
    steps=[

Step('tutorial', intro,
     title='welcome',
     hint="If madspace is already installed, go straight to `generate`.",
     solution='generate p p > t t~'),

Step('install', """
madspace is in place. Now pick a process:
%(p)s generate p p > t t~

`p` is a multiparticle label -- shorthand for a set of particles. A space
between particle names is mandatory.
""" % {'p': P},
     title='install madspace',
     entry='install madspace',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
Look at what MG5 printed back. You gave it no coupling orders, so it chose
some for you.

%(orders)s

You can ask for the electroweak diagrams back by allowing QED vertices:

%(p)s generate p p > t t~ QED<=2

Note that `tutorial syntax` walks through the various ways to generate
diagrams.

The typical next step is to inspect the generated diagrams to check them:

%(p)s display diagrams
""" % {'p': P, 'orders': describe_applied_orders(interface)},
     title='generate a process',
     hint="A space between every particle name, and `>` separates initial from final state.",
     setup=lambda interface: setattr(interface, '_tutorial_lo_ndiag',
                                     total_diagrams(interface)),
     solution='display diagrams'),

Step('generate', lambda interface: _detour_text(interface),
     title='the electroweak diagrams (detour)',
     entry='generate p p > t t~ QED<=2',
     hint="Nothing to do here -- `display diagrams` picks the main line back up.",
     solution='display diagrams'),

Step('display', """
That opens the diagrams in a viewer (or writes them, depending on your setup).
Two minutes here saves an afternoon: a process with far more or far fewer
diagrams than you expected usually means the process line said something other
than what you meant.

`tutorial checks` covers the heavier validation -- `check gauge`,
`check lorentz`, `check permutation` -- for when the diagrams look right but
the number does not.

Now produce the output:
%(p)s output %(run)s
""" % {'p': P, 'run': RUN},
     title='look at the diagrams',
     solution='output %s' % RUN),

Step('output', lambda interface: """
You now have a directory called %(run)s.

Note that `output` can take a format: `output FORMAT PATH`. No format is our
default output for event generation.

Other things `output` can make, each with a tutorial of its own (give them a
different name so they sit beside this one):
  `output madevent DIRNAME`     the MG5-compatible directory layout
  `output standalone DIRNAME`   the matrix element as a callable, no events

Now run it:
%(p)s launch
or
%(p)s launch %(run)s

-- with no argument it takes the output you just made.

Then it runs. If madspace was not built earlier this is where it gets built, so
the first run takes longer than the ones after it.

(To stop a long run and carry on with the tutorial, press Ctrl-C.)
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='produce the output',
     hint="`output NAME` with no format gives you the default MG7 output.",
     question_hint=_launch_question_hint,
     solution=lambda interface: 'launch %s' % output_name(interface, RUN)),

Step('launch', lambda interface: """
That is a full leading-order event sample. The events are an LHE file under
`%(run)s/Events/`, and the run prints two different uncertainties.

The **statistical** error travels with the cross section:

    Result:   %(result)s

That is the Monte-Carlo integration error.

The **theoretical** uncertainty gets its own box at the end:

%(systematics)s

`tutorial mg7` covers how it is computed and how to change what is varied.
%(lhapdf)s%(editor)s
Save what you typed, so you can do this again without remembering it:

%(p)s history my_first_run.dat
""" % {'p': P, 'run': output_name(interface, RUN),
       'result': _result_row(_last_run_info(interface)),
       'systematics': _systematics_rows(_last_run_info(interface)),
       'lhapdf': _lhapdf_note(interface),
       'editor': _editor_back_at_the_prompt(interface)},
     title='run it',
     hint="`history FILE` saves the session; `open FILE` shows a file from the output.",
     solution='history my_first_run.dat'),

Step(('history', 'open', 'display'), lambda interface: """
That is the whole path: generate, look, output, run, read.

You now have a file that replays the whole session:

  MG7> import command my_first_run.dat

or, from the shell, `./bin/madgraph my_first_run.dat`. That is how a run stops
being something you remember and starts being something you can repeat.

Two more things worth knowing:
  * any shell command works from here -- `shell ls`, or just `!ls`;
  * `help COMMAND` prints the syntax for any command, and `help` on its own
    lists them all.

%(see_also)s

Leave tutorial mode with `tutorial stop`. Thanks for using the tutorial!
""" % {'see_also': tutorials.where_next()},
     title='where to go next'),

    ],
)
