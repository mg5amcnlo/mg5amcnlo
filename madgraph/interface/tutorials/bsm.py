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
"""The `bsm` tutorial -- physics beyond the SM, through an effective theory.

Uses SMEFTatNLO, the dimension-six Standard Model EFT at NLO in QCD.  It is in
the MG5 model database, so the first `import model` downloads it.  It teaches
the parts of BSM work that are not the SM's: a dedicated coupling order, an
order hierarchy under which the default search keeps the wrong diagrams, a
convention for that order which changes the process line, and the linear and
quadratic pieces of an EFT cross section.

The second half is the model rather than the process.  The same process is
written out twice, before and after `customize_model`, and the param card that
comes with the code is what the two lessons compare -- the Wilson coefficients
block by block, and the widths, which in an EFT move with the coefficients.
The last lessons are about those widths: what a small one costs an integrator,
and why a large one needs the complex mass scheme rather than a width dropped
into the propagators.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  applied_orders, counts_line,
                                                  is_vi, model_line,
                                                  output_name, text_editor)

P = 'MG7>'
MODEL = 'SMEFTatNLO-NLO'
DEFAULT = 'generate p p > t t~'
ONE_INSERTION = 'generate p p > t t~ NP<=2'
LINEAR = 'generate p p > t t~ NP^2==2'
RESTRICTION = 'top'
CUSTOMIZE = 'customize_model --save=%s' % RESTRICTION
RELOAD = 'import model SMEFTatNLO-%s' % RESTRICTION
# the same process written out twice, before and after the restriction: the
# param card that comes with the code is what the two lessons compare
BEFORE = 'TTX_EFT'
AFTER = 'TTX_TOP'
OUTPUT_BEFORE = 'output standalone %s' % BEFORE
OUTPUT_AFTER = 'output standalone %s' % AFTER
# `open` looks in the last output directory and in its Cards/, so the bare
# name is enough and the same line works for both directories
OPEN_CARD = 'open param_card.dat'
CMS = 'set complex_mass_scheme True'


def _launch(interface=None):
    """`launch DIR` for the directory the reader's last `output` made.

    A standalone `launch` asks one question -- how to build, and which
    parameters to use -- then compiles and evaluates a phase-space point.  The
    question is the reason it is here: `set decay all auto` is a command of a
    card question and of nothing else, so a reader who only ever opens the
    card in an editor never gets to type it.
    """

    return 'launch %s' % output_name(interface, BEFORE)


def _launch_question_hint(interface=None):
    """What to do at the param card question the launch opens.

    The widths are the lesson: this is where they can be handed over to MG7
    instead of being kept up to date by hand, and the question is the only
    place `set decay all auto` exists.
    """

    editor, source = text_editor(interface)
    if not editor:
        opening = ('\nNo text editor was found here, so the card cannot be '
                   'opened from this question -- the `set` lines above do not '
                   'need it.\n')
    else:
        opening = ('\nThe number the question puts in front of `param : '
                   'param_card.dat` opens it in **%s** (%s); the `set` lines '
                   'above do not need it.\n' % (editor, source))
        if is_vi(editor):
            opening += ('`help vi` here lists the handful of keys it takes to '
                        'get out of vi again.\n')

    return """
One question, two halves. The top one is the build -- which `backend` to
compile (`auto` takes the widest your machine has), which `P*` subprocess,
how many cores -- and its defaults are what you want here. The rest is the
param card, as a question rather than as a file: every parameter of the model
can be set from it, and `set` is how.

  set decay all auto     every width computed from the parameters you set
  set decay 6 auto       the top width alone
  set dim6 1 2000        one parameter by block and number (here Lambda)

`set decay all auto` is the one that matters in an EFT: the widths move with
the coefficients, so a number typed in today is wrong as soon as you vary one.
MG7 computes each of them on the way out, from the parameters actually set --
it generates the decays and integrates them, which takes a minute or two and
is worth watching once.
%(opening)s
Then press Enter. It compiles each subprocess and evaluates one phase-space
point with the card you answered -- enough to know the code builds and the
card is read.
""" % {'opening': opening}

# the two blocks the reader switches off at the customize_model question
LEPTON_BLOCKS = ('dim64f2l', 'dim64f4l')


def _coupling_using(interface=None, blocks=LEPTON_BLOCKS):
    """The name of a coupling of this model built on one of those blocks.

    The question invites `display couplings NAME`, and the name has to be one
    of the reader's own model: `display couplings` matches on the coupling's
    name, so `GC_73` is something they would otherwise have to hunt for. The
    shortest expression wins -- a coupling which is one coefficient over
    Lambda^2 makes the point with nothing else in the way. None when the model
    cannot be read, or has no such coupling.
    """

    import re

    model = getattr(interface, '_curr_model', None)
    try:
        names = set(param.name for param
                    in model.get('parameters').get(('external',), [])
                    if param.lhablock.lower() in blocks)
        couplings = sum(model.get('couplings').values(), [])
    except Exception:
        return None

    best = None
    for coupling in couplings:
        try:
            expression = str(coupling.expr)
        except Exception:
            continue
        variables = set(re.findall(r'\w+', expression))
        # MG5 prefixes the parameters it reads from the UFO ('mdl_cQe1');
        # the question displays them as the UFO wrote them
        if not any(name in variables or name.replace('mdl_', '') in variables
                   for name in names):
            continue
        if best is None or len(expression) < len(best[1]):
            best = (coupling.name, expression)
    return best[0] if best else None


def _customize_question_hint(interface=None):
    """What to type at the customize_model question.

    Only what the reader needs to start. What `display` is good for waits
    until they have made a choice for it to show (_customize_question_progress):
    a wall of text at the top of a question is read once, at the moment none
    of it applies yet.
    """

    return """
Switch off what a top pair never sees -- the operators with leptons in them,
two whole blocks of the param card:

  set DIM64F2L all 0         two quarks, two leptons
  set DIM64F4L all 0         four leptons

then `done`. `set NAME 0` does it for one coefficient, and `set NAME = OTHER`
ties two together. Leave `DIM6` alone as a block: it also holds `Lambda`, and
a zero scale is a division by zero.
"""


def _customize_question_progress(interface=None, line=None):
    """The rest of the lesson, said as the reader gets to it.

    Two things follow an answer rather than preceding it: after a `set`, that
    the question can show what it did; after the `display` that follows, how
    to read what came back. Each is said once -- the reader is answering a
    question, not reading a page -- which is what the marks on the interface
    remember.
    """

    words = (line or '').split()
    if not words:
        return None
    seen = getattr(interface, '_tutorial_bsm_said', None)
    if seen is None:
        seen = set()
        try:
            interface._tutorial_bsm_said = seen
        except Exception:
            pass

    if words[0] == 'set' and 'display' not in seen:
        seen.add('display')
        example = _coupling_using(interface)
        look = ('  display couplings %s        a coupling built on one of them'
                % example if example else
                '  display couplings NAME         a coupling of the model, by '
                'name')
        return """
That is one choice made. Look at what it did -- not a step you have to take,
but this is where checking a restriction is free:

  display parameters DIM64F2L    the block, every line of it now marked `-> 0`
%s
""" % look

    if words[0] == 'display' and 'read' not in seen:
        seen.add('read')
        return """
`display parameters` says which parameters your choices fix and to what.
`display couplings` prints the coupling's expression, unfolds it down to the
parameters of the param card, marks the ones your choices restrict and ends
on one line saying whether that coupling survives them -- the impact of what
you typed, before `done` commits it.
"""
    return None


def _orders(interface=None):
    """The coupling orders the loaded model declares, and their hierarchy.

    Read back rather than asserted: which orders a model defines, and what
    each weighs, is the whole subject of the step.
    """

    model = getattr(interface, '_curr_model', None)
    try:
        orders = sorted(model.get('coupling_orders'))
        hierarchy = model.get('order_hierarchy') or {}
    except Exception:
        return ''
    if not orders:
        return 'This model declares no coupling order at all.\n'
    names = ['**%s**' % order for order in orders]
    listed = (', '.join(names[:-1]) + ' and ' + names[-1]
              if len(names) > 1 else names[0])
    text = 'This model declares %s' % listed
    if all(order in hierarchy for order in orders):
        text += (',\nweighed %s in its hierarchy'
                 % ', '.join('%s %s' % (order, hierarchy[order])
                             for order in sorted(orders,
                                                 key=lambda o: hierarchy[o])))
    return text + '.\n'


def _is_smeft(interface=None, line=None):
    """Whether the model now loaded is an SMEFT one -- the gate on lesson two.

    Everything after the intro is about this model's own new physics: the
    lesson coming up reads the coupling orders and is written around `NP`, and
    the ones after it count operator insertions in powers of 1/Lambda and
    switch off blocks of Wilson coefficients.  Against the Standard Model there
    is no `NP` to read and nothing any of that acts on, so an `import model`
    which loaded something else does not move the tutorial on: it says what is
    loaded and what to load instead.  Told by the model's name, which carries
    its restriction -- `SMEFTatNLO-NLO`, and later the reader's own
    `SMEFTatNLO-top`.
    """

    model = getattr(interface, '_curr_model', None)
    try:
        name = model.get('name') or ''
    except Exception:
        name = ''
    if 'smeft' in name.lower():
        return True
    return ("""%(loaded)sThis lesson and every one after it are about an EFT's own new physics: the
coupling order that labels it, the powers of 1/Lambda that count operator
insertions, the Wilson coefficients you switch off one block at a time. None
of that is there%(subject)s, so the tutorial stays here.

Load the EFT and it moves on:
%(p)s import model %(model)s""" % {
        'p': P, 'model': MODEL,
        'loaded': model_line(interface) or 'No model is loaded.\n',
        'subject': ' in **%s**' % name if name else ''})


def _saved_restriction(interface=None):
    """Where customize_model wrote the restriction card, if it did."""

    import os

    try:
        path = os.path.join(interface._curr_model.get('modelpath'),
                            'restrict_%s.dat' % RESTRICTION)
    except Exception:
        return ''
    if not os.path.isfile(path):
        return ''
    return 'The restriction is saved as\n`%s`.\n' % path


def _card_summary(interface=None):
    """The blocks of the param card the reader's last `output` wrote.

    The point of the two output lessons is a before/after on that card, and
    the numbers have to be the reader's own: their directory, their model,
    and after `customize_model` their restriction.  '' when there is no card
    to read -- the lessons still make sense without it.
    """

    import os

    try:
        done = getattr(interface, '_done_export', None)
        path = os.path.join(done[0], 'Cards', 'param_card.dat')
        with open(path) as handle:
            lines = handle.readlines()
    except Exception:
        return ''

    blocks, widths = [], 0
    for line in lines:
        words = line.split()
        if not words:
            continue
        if words[0].lower() == 'block' and len(words) > 1:
            name = words[1].upper()
            # QNUMBERS is one block per particle: quantum numbers, not inputs
            if name != 'QNUMBERS' and name not in blocks:
                blocks.append(name)
        elif words[0].lower() == 'decay':
            widths += 1
    if not blocks:
        return ''
    return ('`%s/Cards/param_card.dat`: **%d** blocks, **%d** widths.\n'
            '  %s\n' % (output_name(interface, BEFORE), len(blocks), widths,
                        ', '.join(blocks)))


def _last_generate(interface=None):
    """The `generate` line the reader last typed, or the main-line one.

    The interference step is a detour, so by the time the restriction is
    built the process in memory is `NP<=2` for a reader who stayed on the
    main line and `NP^2==2` for one who took it. The lesson which asks for
    that process to be generated again under the new model has to ask for
    theirs, and the one after it compares diagram counts across the two.
    """

    try:
        for line in reversed(interface.history):
            if line.split()[0] == 'generate':
                return line
    except Exception:
        pass
    return ONE_INSERTION


def _widths_changed(interface=None):
    """The widths the reader's launch recomputed, against the shipped ones.

    Every standalone output leaves a pristine `param_card_default.dat` beside
    the card the question edits, so the two together say what the automatic
    widths did -- and the reader is shown their own numbers rather than mine.
    '' when nothing moved: pressing Enter is a perfectly good answer to that
    question.
    """

    import os

    def widths(path):
        out = {}
        with open(path) as handle:
            for line in handle:
                words = line.split()
                if len(words) > 2 and words[0].lower() == 'decay':
                    try:
                        out[int(words[1])] = (float(words[2]),
                                              line.split('#')[-1].strip())
                    except ValueError:
                        continue
        return out

    try:
        cards = os.path.join(interface._done_export[0], 'Cards')
        now = widths(os.path.join(cards, 'param_card.dat'))
        shipped = widths(os.path.join(cards, 'param_card_default.dat'))
    except Exception:
        return ''

    moved = [(pdg, value, shipped[pdg][0], name)
             for pdg, (value, name) in sorted(now.items())
             if pdg in shipped and value != shipped[pdg][0]]
    if not moved:
        return ''
    return ('Your card came back with new widths:\n%s\n'
            % '\n'.join('  %-4s %-6s %.5g   (the model shipped %.5g)'
                         % (pdg, name, value, was)
                         for pdg, value, was, name in moved[:4]))


def _editor_note(interface=None):
    """Which editor the card is about to open in, and how to leave it."""

    editor, _ = text_editor(interface)
    if not editor:
        return ('No text editor was found here, so `open` may not show you '
                'anything: read the file with whatever you normally use.\n')
    if is_vi(editor):
        return ('It opens in **%s**: `:q!` then Enter leaves it without '
                'saving (`help vi` has the rest).\n' % editor)
    return ('It opens in **%s**. Close it without saving and the tutorial '
            'carries on.\n' % editor)


def _search_result(interface=None):
    """The orders MG5 put on a process given none, as it printed them."""

    found = [orders for orders in applied_orders(interface) if orders]
    if not found:
        return 'no coupling-order constraint at all'
    return ' and '.join(
        '**%s**' % ' '.join('%s<=%s' % (name, value)
                            for name, value in sorted(orders.items()))
        for orders in found[:2])


tutorial = Tutorial(
    name='bsm',
    title='beyond the Standard Model',
    description='an EFT model: its coupling order, its conventions, a restriction of your own, and the widths it moves',
    order='sequence',
    section='advanced',
    ai_generated=False,
    see_also=('model', 'syntax', 'decays', 'checks'),
    steps=[

Step('tutorial', """
Nothing in MG7 knows about the Standard Model specifically. Hand it a different
model and everything -- diagram generation, matrix elements, events -- works
the same way. This tutorial is about the parts that are *not* the same, on the
most common kind of BSM model today: an effective field theory.

Where BSM models come from:
  * **FeynRules** turns a Lagrangian into a UFO model. If your model does not
    exist yet, this is how it starts.
  * **The model database** -- `display modellist` shows what MG7 can download,
    and `import model NAME` fetches it.
  * **A colleague**, as a UFO directory you point `import model` at.

We will use SMEFTatNLO, the dimension-six Standard Model EFT at NLO in QCD,
from the database -- the first import downloads it. `-NLO` is the restriction
the model provides for NLO QCD work:
%(p)s import model %(model)s
""" % {'p': P, 'model': MODEL},
     title='welcome',
     hint="`import model NAME` also downloads a model from the database.",
     solution='import model %s' % MODEL),

Step('import_model', lambda interface: """
%(model)sNot one new particle: that is the Standard Model and its ghosts. The
new physics is in the interactions -- dimension-six operators, each entering
through a Wilson coefficient over Lambda^2. The coefficients are parameters of
the model, in the param card's `DIM6`, `DIM62F` and `DIM64F` blocks, and the
new vertices are what they multiply.

Before generating anything, see how the model labels that new physics:
%(p)s display coupling_order
""" % {'p': P, 'model': model_line(interface)},
     title='load an EFT model',
     # the whole lesson is about `NP`: an import of anything else waits here
     gate=_is_smeft,
     solution='display coupling_order'),

Step('display', lambda interface: """
%(orders)sThis is the camp of models with a dedicated order: `NP` marks the new
physics, so you ask for it in the process line rather than by naming new
particles. Two things about it are this model's own, and both matter:

  * **NP counts powers of 1/Lambda.** Every operator comes with 1/Lambda^2, so
    one operator insertion is `NP=2`, not 1 -- and `NP<=1` quietly removes
    every EFT vertex. Other models count an insertion as `NP=1`; a model's
    documentation says which.
  * **NP is the cheapest order.** When you give no orders, MG5 keeps the
    cheapest diagrams it can find, weighing each order by that hierarchy.

See what that does to the simplest process there is:
%(p)s %(default)s
""" % {'p': P, 'default': DEFAULT, 'orders': _orders(interface)},
     title='how the model labels new physics',
     hint="`generate p p > t t~`, with no orders on purpose.",
     solution=DEFAULT),

Step('generate', lambda interface: """
%(counts)sMG5 settled on %(search)s -- and not one of those diagrams is the
Standard Model. A top pair from QCD weighs 4, two QCD vertices at 2 each; one
operator insertion weighs 2. The search took the lowest weight that produces
anything, which is the EFT on its own: a pure 1/Lambda^4 piece with nothing to
interfere with, and a prediction of nothing.

That is the trap `tutorial syntax` warns about, and in a model like this one it
is the default. Naming *one* order is enough to call the search off, and the
orders you do not name are then left unrestricted -- not held at whatever the
search would have picked for them. So `NP<=2` below bounds `NP` alone and
leaves `QCD` and `QED` free, which is exactly how the Standard Model diagrams
come back. `set default_unset_couplings N` changes that: the orders you leave
out are capped at N instead of being free, and MG7 prints which ones it
applied it to. The default is 99, which is no cap at all.

Say the orders yourself -- the Standard Model plus at most one operator
insertion:
%(p)s %(one)s
""" % {'p': P, 'one': ONE_INSERTION, 'counts': counts_line(interface),
       'search': _search_result(interface)},
     title='the default search, in an EFT',
     hint="One operator insertion is `NP=2` in this model.",
     solution=ONE_INSERTION),

Step('generate', lambda interface: """
%(counts)sThe Standard Model diagrams, and every single-insertion one beside
them. Squared, that amplitude has three pieces:

  SM x SM      the Standard Model                     NP^2==0
  SM x EFT     linear in the coefficients, 1/Lambda^2   NP^2==2
  EFT x EFT    quadratic, 1/Lambda^4                  NP^2==4

The linear term is the one a dimension-six analysis is built on. The quadratic
one is formally of the same order as the dimension-eight operators you did not
include, so whether to keep it is a choice to make and to state. Asking for
one of them on its own is a detour worth taking once:
%(p)s %(linear)s

That is the process settled. What is left is the model: a dimension-six EFT
carries a coefficient for every operator, and most of them never touch a top
pair. Write the process out and you get the whole list on paper -- the param
card comes with the code. `standalone` is the output to ask for here: the
matrix element and its card, none of the run machinery, and a `launch` which
puts that card in front of you before it builds anything:
%(p)s %(output)s
""" % {'p': P, 'linear': LINEAR, 'output': OUTPUT_BEFORE,
       'counts': counts_line(interface)},
     title='linear and quadratic',
     hint="`output standalone DIR` picks up the main line. The detour is "
          "`%s`: `^2` constrains the squared matrix element." % LINEAR,
     solution=OUTPUT_BEFORE),

Step('generate', lambda interface: """
%(counts)sThe same diagrams as a moment ago: as in `tutorial syntax`, a `^2`
constraint picks which products survive the squaring, not which diagrams
exist. It is also where the model's convention bites a second time. In a
model counting an insertion as `NP=1`, this same term is `NP^2==1` -- which
here asks for a term that does not exist, and generates nothing.

That is the detour. Back to the main line -- write the process out:
%(p)s %(output)s
""" % {'p': P, 'output': OUTPUT_BEFORE, 'counts': counts_line(interface)},
     title='the interference on its own (detour)',
     entry=LINEAR,
     hint="Nothing to do here -- `output standalone DIR` picks the main "
          "line back up.",
     solution=OUTPUT_BEFORE),

Step('output', lambda interface: """
%(card)sThat card is the model as you will actually set it, and the blocks say
which is which. The Wilson coefficients are the `DIM*` ones, one line per
coefficient, with `Lambda` itself as entry 1 of `DIM6` -- every coefficient in
the card is a number divided by that scale. `MASS`, `SMINPUTS` and `YUKAWA`
are the Standard Model inputs, and each `DECAY` line is one particle's width.

You can read it with `open param_card.dat`, but `launch` is better: it puts
the same card in front of you as a question you can set parameters from, then
compiles the matrix element and evaluates a phase-space point with whatever
you answered. The compiling is why it is not instant:
%(p)s %(launch)s
""" % {'p': P, 'launch': _launch(interface), 'card': _card_summary(interface)},
     title='the parameters, on paper',
     # current while the launch asks the param card question
     question_hint=_launch_question_hint,
     hint=lambda interface: ("`%s` -- the directory your last `output` made."
                            % _launch(interface)),
     solution=_launch),

Step('launch', lambda interface: """
%(widths)sWhatever you answered there is what the matrix element was
evaluated with. If you asked for `auto` widths, MG7 computed them on the way
out -- the two-body decays and the numerical integration in the log. Widths
deserve that attention here: `ctW` enters `t > W b`, so the top width moves
with the very coefficients you are varying, and a number typed in by hand is
wrong as soon as it does.

The coefficients are the other half, and most of them do nothing for this
process. They are in the card because they are in the model, and they will be
in every card, every scan and every reweighting until you take them out.
`customize_model` builds a restriction of your own -- coefficients fixed to
zero, to one, or tied together -- and `--save` keeps it, so it loads like the
ones the model ships:
%(p)s %(customize)s
""" % {'p': P, 'customize': CUSTOMIZE, 'widths': _widths_changed(interface)},
     title='widths, and what to do about them',
     # what the question has already been told, forgotten each time the lesson
     # is reached: `back`, `skip` or a second run of the tutorial start over
     setup=lambda interface: setattr(interface, '_tutorial_bsm_said', set()),
     # current while customize_model asks its question
     question_hint=_customize_question_hint,
     question_progress=_customize_question_progress,
     hint="`customize_model --save=NAME` opens a question; `done` closes it.",
     solution=CUSTOMIZE),

Step('customize_model', lambda interface: """
%(model)s%(saved)sThat is now the model loaded here, and it started from where
you were: the operators `-NLO` switches off were already in the question at 0,
and your two blocks went on top. The lepton operators are gone, and so are the
vertices they multiplied.

Next to the restriction, `--save` wrote `param_%(restriction)s.dat`: the
default values it comes with, which is what fills the card from now on. In a
later session `%(reload)s` loads the pair -- nothing to do here, that model is
already the one in front of you.

The process still in memory was generated with the *previous* model, so
generate it again under this one:
%(p)s %(again)s
""" % {'p': P, 'again': _last_generate(interface), 'reload': RELOAD,
       'restriction': RESTRICTION, 'model': model_line(interface),
       'saved': _saved_restriction(interface)},
     title='a restriction of your own',
     hint=lambda interface: ("The same process line as before: `%s`."
                            % _last_generate(interface)),
     solution=_last_generate),

Step('generate', lambda interface: """
%(counts)sThe same count as before the restriction, which is the whole point:
the operators you switched off never entered `p p > t t~`, so the physics is
untouched and what you dropped was overhead. In a model where the restriction
*does* remove a vertex your diagrams use, this is the step that tells you --
the count changes.

Write it out again, next to the first one:
%(p)s %(output)s
""" % {'p': P, 'output': OUTPUT_AFTER, 'counts': counts_line(interface)},
     title='the same process, a smaller model',
     hint="`output standalone DIR` again, under a new name so the first "
          "directory survives.",
     solution=OUTPUT_AFTER),

Step('output', lambda interface: """
%(card)sTwo blocks short of the first card: `DIM64F2L` and `DIM64F4L` are not
in this one, and neither are the coefficients they held.

A shorter card is not cosmetic. It is what a scan iterates over, what
`reweight` reads, and what anyone repeating your analysis has to be handed.
See it for yourself -- no need to launch this one, `open` finds the card of
your last output. %(editor)s
%(p)s %(open)s
""" % {'p': P, 'open': OPEN_CARD, 'card': _card_summary(interface),
       'editor': _editor_note(interface)},
     title='the card, after the restriction',
     hint="`open param_card.dat` -- your last output was the new directory.",
     solution=OPEN_CARD),

Step('open', """
One more thing about the `DECAY` block, because a width is not just a number
to keep up to date. Small and large widths are two different problems.

**Small widths.** A narrow resonance is a near-singular integrand, so each
s-channel one gets a Breit-Wigner channel of its own, and `bwcutoff` in the
run card (15) says how far off shell still counts as on it. Below
`small_width_treatment * mass` (1e-6, same card) the width in the propagator
is lifted to that floor and the cross section corrected in the narrow-width
approximation. A width of exactly zero makes the particle stable.

**Large widths.** A broad resonance is the harder case, and not only
numerically. Putting `Gamma` into the propagators and nowhere else -- the
fixed-width scheme -- is not gauge invariant: the Ward identities tie
propagators to vertices, so a width in one and not the other leaves a
violation that grows with energy and can show up as a visibly wrong tail. The
gauge-invariant treatment is the **complex mass scheme**: the mass is made
complex once, `m^2 -> m^2 - i*m*Gamma`, everywhere it appears -- in the
propagators and in every coupling built from it, `sin(theta_W)` included --
so the identities hold term by term. That is one option in MG7:
%(p)s %(cms)s
""" % {'p': P, 'cms': CMS},
     title='small widths and large ones',
     hint="`set complex_mass_scheme True` -- it is a model-level option, so "
          "MG7 reloads the model itself.",
     solution=CMS),

Step('set', lambda interface: """
%(model)sMG7 re-imported the model for you: the scheme is not a flag applied
at the end, the complex masses have to be there when the couplings are built,
so the option only means anything through a fresh import -- your restriction
and its `param_%(restriction)s.dat` defaults included, as the log above shows.
Widths now come from the param card into the masses themselves, which is also
why `Auto` widths and this scheme belong together.

Two things to know before you rely on it. At tree level the check to run is
`check gauge`: gauge invariance is the thing a width in the propagators
breaks, so it is also the thing that says whether your treatment of it holds
up. It computes the same process in unitary, Feynman, axial and FD gauge and
compares the four. Give it a process with a massive vector inside, so the
Goldstone bosons have something to do:

  check gauge a e- > e- ve ve~ NP=0

Four numbers agreeing and `Passed` is what gauge invariance looks like from
here. Say the orders as always -- `NP=0` asks about the Standard Model part,
`NP<=2` includes the operators you kept, and each is a question worth asking
of a model you did not write. `check cms` is the one dedicated to the scheme
itself, comparing against the narrow-width approximation with the width
shrinking; `tutorial checks` covers the family.

And read `help set complex_mass_scheme` first: it says that masses become the
electroweak input parameters, and it warns that loop processes are not
supported.

Three more things that bite in EFT work:
  * **The convention is the physics.** Which `NP` value is one insertion, and
    so which squared order is the linear term, is the model's choice. Read its
    documentation before you write a process line.
  * **EFT amplitudes grow with energy**, by construction. A cross section
    dominated by events above your cutoff says the expansion has broken down,
    not that you found something.
  * **A restriction is part of your result.** `restrict_%(restriction)s.dat`
    and `param_%(restriction)s.dat` say which operators you kept and what the
    others were fixed to. `explain_restriction` reads one back, and
    `tutorial checks` is the tutorial to take before you trust a model you
    did not write.

%(see_also)s

Leave with `tutorial stop`.
""" % {'restriction': RESTRICTION, 'model': model_line(interface),
       'see_also': tutorials.where_next()},
     title='the complex mass scheme, and EFT pitfalls'),

    ],
)
