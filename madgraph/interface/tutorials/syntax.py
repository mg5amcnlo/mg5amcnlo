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
"""The `syntax` tutorial -- everything you can put in a process line.

This is a *sequenced* tutorial: most steps are triggered by `generate`, each
one teaching something different.  That is exactly what the old
command-name -> text lookup could not express.
"""

from __future__ import absolute_import

from madgraph.interface.tutorials.session import (Step, Tutorial, counts,
                                                  counts_line, output_name,
                                                  total_diagrams)

P = 'MG7>'

DECAY_CHAIN = 'generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~'
# the pure QCD term: what the interference lesson leaves on, and what the side
# quest comes back to.  It has to be a command of that lesson's own -- leaving
# on the decay chain would ask the user for it here and again in the lesson
# that teaches it.
PURE_QCD = 'generate p p > j j QCD^2==4 QED^2==0'


EXCLUSION = 'generate p p > e+ e- / a'
# the polarised process the tutorial ends on: two different decays, so the
# two Z bosons can be told apart and each has a rest frame of its own
POLARISED = 'generate p p > z{0} z{T}, (z > e+ e-), (z > mu+ mu-)'
POL_DIR = 'POL_ZZ'

# what the tutorial signs off with.  It is reached whether or not the reader
# takes the optional run at the end, so it lives here rather than in a step.
CLOSING = """That is the syntax tour. A few things worth remembering:

 * `define` makes your own multiparticle label, e.g.
   `define v = w+ w- z a`, usable anywhere `p` or `j` is;
 * `display diagrams` draws what you generated before you commit to an
   output -- a cheap habit;
 * `help generate` prints the whole grammar in one screen.

Where to go next:
 * `tutorial lo`         take a process all the way to events
 * `tutorial nlo`        the same process line at next-to-leading order
 * `tutorial exercises`  practise, with the answers checked
 * `tutorial list`       everything on offer

Leave tutorial mode with `tutorial stop`."""


def _tried_an_exclusion(keys, line, interface=None):
    """A `generate ... $ x` or `$$ x` line: a comparison the lesson invites.

    Keyed on the operator rather than the command name, because the command
    that ends the lesson is a `generate` over the same particles: `generate p`
    and `generate` are the keys of all three, and only the line tells them
    apart.
    """

    return 'generate' in keys and '$' in (line or '')


def _exclusion_note(interface, line=None):
    """Answer a `$` or `$$` without ending the exclusion lesson.

    The diagram count is read back from what the command just produced, since
    that is the whole point of trying them: `$` leaves it alone and `$$` does
    not.
    """

    total = total_diagrams(interface)
    # its own line, and short: the prose around it is hand-wrapped, and a
    # markup span that wrapped would colour the next line's indentation
    count = ('\n\nThat came to **%d diagram%s**.'
             % (total, '' if total == 1 else 's')) if total else ''

    if '$$' in (line or ''):
        what = (
"""`$$` forbids the photon as an s-channel outright, so the diagram is dropped.
Fewer than you started with -- and what is left is a subset of an amplitude,
so the gauge warning from the lesson above applies to it in full.""")
    else:
        what = (
"""`$` kept every diagram and subtracted only the photon's on-shell
contribution. The count is the one you get with no constraint at all, because
nothing was deleted: what changed is where the propagator may sit, not which
diagrams exist. That is why this one is the safe operator.""")

    return """
%(what)s%(count)s

Try the other one too if you have not. The tutorial waits here; what carries
on is:
%(p)s %(exclusion)s
""" % {'what': what, 'count': count, 'p': P, 'exclusion': EXCLUSION}


def _remember_counts(interface):
    """Keep the interference counts, so the next lesson can compare against
    them.  Same trick as `lo`'s detour: the numbers belong to the user's own
    session, not to the text."""

    setattr(interface, '_tutorial_syntax_ndiag', total_diagrams(interface))


def _pure_qcd_comparison(interface=None):
    """What the pure QCD term came to, set beside the interference before it.

    Both numbers are read back from the session; `repeat` after a restart has
    no remembered count and simply says less.
    """

    now = total_diagrams(interface)
    before = getattr(interface, '_tutorial_syntax_ndiag', None)

    opening = counts(interface)
    if opening:
        if before and before != now:
            opening += ', against **%d** a moment ago' % before
        opening += '.\n'

    return opening + (
"""More subprocesses than the interference had, and fewer diagrams: the
gluon-initiated ones are back, because they do have a QCD-squared term, and
every electroweak diagram has gone. It is also, to the diagram, what a bare
`generate p p > j j` gives you -- the search from the first lesson, choosing
the pure QCD term on your behalf.""")


def _run_dir(interface=None):
    """The madevent directory the user actually wrote, by whatever name."""

    return output_name(interface, POL_DIR)


def _standalone_dir(interface=None):
    """The directory `output standalone` just wrote, named if we know it."""

    name = output_name(interface, None) if interface is not None else None
    # its own line: the name is any length the user chose, and a markup span
    # that wrapped would colour the next line's indentation
    return '\nIt landed in `%s`.' % name if name else ''


tutorial = Tutorial(
    name='syntax',
    title='the process generation syntax',
    description='orders, interference, decay chains, s-channels, gauge traps, polarisation',
    order='sequence',
    section='advanced',
    ai_generated=False,
    steps=[

Step('tutorial', """
Each lesson hands you a command to type at the MG7 prompt; type it and the
next lesson appears. The tutorial only watches -- it never runs anything for
you. If you are stuck, `hint` and `solution` print what is expected, `skip`
moves on, `repeat` prints the step again, `tutorial status` shows how far you
have got, and `tutorial help` lists the lot.

The subject is the process line itself: everything you can put after
`generate`, one idea at a time. It picks up where `tutorial lo` leaves off.

What the lo tutorial already showed, in three lines:

  generate p p > t t~     initial state, `>`, final state; the spaces matter
  p  j  l+  l-  vl        multiparticle labels (`display multiparticles`)
  no orders given         MG5 searches for coupling orders and applies its own

As said in the lo tutorial, that third line is the one to watch. It typically
lands on the QCD-dominant contribution -- fine when that is the physics you
are after. It is a trap in a BSM model, where the order counting may be
declared in a way that makes the lowest solution the wrong one. Stating the
orders yourself costs one word and settles it.

So, to begin, type:
%(p)s generate p p > t t~ QED<=2
""" % {'p': P},
     title='welcome',
     hint="`tutorial lo` is the one that starts from nothing; this one starts "
          "from the process line.",
     solution='generate p p > t t~ QED<=2'),

Step('generate', lambda interface: """
%(counts)sThose are the QCD diagrams plus the photon and Z exchanges in the
quark-antiquark subprocess, which the search leaves out when you say nothing.
`QED<=2` constrained the *amplitude*: at most two QED vertices per diagram.
The whole family reads

  QED=0    at most 0 QED vertices     ('=' means '<=' -- this trips people up)
  QED==0   exactly 0 QED vertices
  QED<=2   at most 2
  QED>2    more than 2

and every one of them counts vertices in the diagram.

A constraint with `^2` applies to the *squared* matrix element, not to the
amplitude, so it selects one term of |M|^2. For a process with both a QCD and
an EW amplitude, |M|^2 has three pieces:

  generate p p > j j QCD^2==4 QED^2==0     the pure QCD term
  generate p p > j j QCD^2==2 QED^2==2     the INTERFERENCE only
  generate p p > j j QCD^2==0 QED^2==4     the pure EW term

and the three add up to `generate p p > j j` with no constraints at all.

Generate the interference term on its own:
%(p)s generate p p > j j QCD^2==2 QED^2==2
""" % {'p': P, 'counts': counts_line(interface)},
     title='coupling orders: amplitude and squared',
     hint="'^2' makes the constraint apply to the squared matrix element.",
     solution='generate p p > j j QCD^2==2 QED^2==2'),

Step('generate', lambda interface: """
%(counts)sThose diagrams are the ordinary ones: `display diagrams` draws the
gluon exchange next to the photon and the Z exchange, exactly what you get
with no `^2` constraint at all.

That is not a bug, and it is why the diagram count can never tell you which
term you asked for. A `^2` constraint does not select diagrams: the amplitude
stays whole. It selects which pairs of amplitudes survive when that amplitude
is squared, and the squaring happens inside the matrix element, downstream of
anything `display` can draw.

What it does decide is which subprocesses exist at all. `g g > g g` has no
electroweak amplitude to interfere with, so its interference term is empty and
the subprocess is gone: only the quark ones are left.

**Side quest** -- two commands, and you see it instead of taking my word:
%(p)s output standalone

Either way, what carries on is the pure QCD term of the same process -- the
first line of the table above, and worth generating for what it does to the
subprocess list:
%(p)s %(qcd)s
""" % {'counts': counts_line(interface), 'p': P, 'qcd': PURE_QCD},
     title='the interference is not in the diagrams',
     setup=_remember_counts,
     hint="`output standalone` takes the side quest; `%s` carries on."
          % PURE_QCD,
     solution=PURE_QCD),

Step('output', lambda interface: """
That wrote a small standalone program: the matrix element and nothing else --
no integration, no cuts, no PDFs.%(dir)s

It asks two questions, a backend and a subprocess, and Enter takes the default
for both. Then it compiles, picks one phase-space point, and prints a
`Matrix element` value for every flavour combination.

Run it:
%(p)s launch
""" % {'p': P, 'dir': _standalone_dir(interface)},
     title='side quest: a standalone matrix element',
     entry='output standalone',
     hint="`launch` with no argument runs the directory you just wrote.",
     question_hint="Enter takes the default at both questions. To go straight "
                   "to the interesting numbers, pick the `q q~ > q q~` "
                   "subprocess at the second one.",
     solution='launch'),

Step('launch', """
Look through the numbers for the `q q~ > q q~` subprocess (`P1_QQx_QQx`) and
find a negative one. There are several.

**No squared matrix element is ever negative.** What MG7 printed is not one: it
is the cross term alone -- the QCD amplitude times the conjugate of the
electroweak one, twice its real part -- and a cross term carries the relative
sign of the two amplitudes, so it is free to come out negative. Add the two
positive terms back and the total is positive again, as it has to be.

You will also find entries that are exactly zero or highly suppressed although
the subprocess has diagrams -- `d s > d s` is one. What decides that is not
s-channel against t-channel; it is whether the process offers *two* colour
flows at all. With a single topology, s or t alike, the gluon and the photon
differ by one colour generator on each quark line, and the trace of a single
generator is zero. The entries that survive are the ones with a second flow --
identical flavours, or a W exchange beside the gluon -- and that is where the
negative numbers are.

That is the side quest. Back to the main line, the command the lesson before
it asked for:
%(p)s %(qcd)s
""" % {'p': P, 'qcd': PURE_QCD},
     title='side quest: the sign gives it away',
     hint="Nothing to do here -- `%s` picks the main line back up." % PURE_QCD,
     on_failure="`launch` needs a directory to run: write one first with "
                "`output standalone`.",
     solution=PURE_QCD),

Step('generate', lambda interface: """
%(compared)s

That is the end of the coupling orders. A comma opens a decay chain:
everything after it decays a particle of the process before it, and
parentheses nest.

Two things to keep in mind:
 * identical particles are ALL decayed -- you cannot decay one top and leave
   the other alone by writing the decay once;
 * a decay chain is neither on-shell nor a branching ratio. It is the full
   matrix element, production times propagator times decay, with the spin
   correlations kept and the resonance left off shell -- spread over its
   Breit-Wigner out to `bwcutoff` widths from the pole. What it drops is the
   diagrams that do not go through that resonance.

This is also the *safe* way to ask for a resonance. Production and decay are
each a complete set of diagrams, so each is gauge invariant on its own, and
what you dropped is stated plainly: the diagrams that do not go through the
resonance. The next two lessons do a similar-looking job by reaching inside a
single amplitude, and that is where it gets delicate.

Your turn:
%(p)s %(decay)s
""" % {'p': P, 'decay': DECAY_CHAIN,
       'compared': _pure_qcd_comparison(interface)},
     title='decay chains',
     hint="Use ',' to open the decay, and parentheses to nest a second one.",
     solution='generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~'),

Step('generate', lambda interface: """
%(counts)sThe production and the decays are generated separately and counted
together; `output` stitches them back into full diagrams, every one of them
going through the tops you asked to decay.

Now the operators that reach inside one amplitude and keep part of it.

A second `>` names a required s-channel: only diagrams going through that
particle are kept.

%(p)s generate p p > w+ > l+ vl

Alternative s-channels are separated by `|`, which is how you keep an
interfering pair together -- the Z/photon pair being the everyday case:

  generate p p > z | a > e+ e-

Several particles on one side of the `|` are required together, so a
two-per-side form asks for one pair or the other. They have to be distinct
names: `> z z >` is refused rather than quietly read as `> z >`.

**Read this before you use it.** A single Feynman diagram is not an
observable. Only the complete set for an amplitude is gauge invariant, and the
cancellations between diagrams can be enormous. Keep some and drop the rest
and the answer may depend on the gauge it was computed in -- which means it is
not a prediction of anything.

Sometimes the subset you kept is a complete gauge-invariant set by itself and
all is well: photon versus Z exchange in `u u~ > e+ e-` is the textbook case.
Sometimes it is not, and nothing in the output tells you which you are in.
`check gauge` does, and the next lesson runs it.

Ask for the W explicitly:
%(p)s generate p p > w+ > l+ vl
""" % {'p': P, 'counts': counts_line(interface)},
     title='required s-channels',
     hint="Put the intermediate particle between two '>'.",
     solution='generate p p > w+ > l+ vl'),

Step('generate', lambda interface: """
%(counts)sEverything that did not go through the W is gone -- which is the
point of the operator, and also its danger.

The mirror image: excluding things. Three operators, and the difference
between them is exactly the gauge question from the last lesson.

  $   exclude the ON-SHELL contribution of an s-channel particle. The diagram
      is KEPT and only the resonance peak is subtracted, so this one is safe
      by construction.
  $$  forbid that s-channel entirely -- the diagram is dropped.
  /   forbid a particle ANYWHERE in the diagram, internal or external.

`generate p p > e+ e- $ a` and `generate p p > e+ e- $$ a` are both worth
typing here, in either order: the contrast between their diagram counts is the
lesson, and neither of them ends this one.

`$$` and `/` delete diagrams, so they carry the same warning as `> A >` above.
Worth running once, so that you have watched it happen:

  check gauge e+ e- > w+ w-           passes
  check gauge e+ e- > w+ w- $$ a      FAILS
  check gauge e+ e- > a > w+ w-       FAILS

Dropping the photon from W pair production leaves a matrix element several
times the full one, and destroys the cancellation that keeps it from growing
with energy. `check gauge` computes the same thing in four gauges and compares
them; `tutorial checks` is the one that goes through it. (`check` needs a model
imported, which `generate` does for you but `check` does not.)

Forbid the photon everywhere and see what is left:
%(p)s %(exclusion)s
""" % {'p': P, 'exclusion': EXCLUSION, 'counts': counts_line(interface)},
     title='excluding particles and s-channels',
     hint="'/ a' forbids the photon everywhere; '$ a' only removes it on shell.",
     solution=EXCLUSION),

# answers the `$` and `$$` the lesson above invites, and stays put: without it
# either one would count as the lesson's command and jump the reader a step
Step(_tried_an_exclusion, _exclusion_note,
     title='what $ and $$ did',
     sticky=True,
     hint="`$` and `$$` are the comparison, not the way on -- `%s` is."
          % EXCLUSION,
     solution=EXCLUSION),

Step('generate', lambda interface: """
%(counts)sWith `/` the photon is gone from everywhere it could have sat,
internal or external, and the Z exchange is all that survives.

`add process` puts a second process into the same output. It takes exactly the
same syntax as `generate`, and it accumulates instead of replacing.

`display processes` lists everything defined so far, `display diagrams` draws
all of it, and everything you have added goes into the next `output`.

Add a second process beside the one you have:
%(p)s add process p p > w+ j, w+ > l+ vl
""" % {'p': P, 'counts': counts_line(interface)},
     title='several processes at once',
     hint="'add process' takes the same syntax as 'generate'.",
     solution='add process p p > w+ j, w+ > l+ vl'),

Step('add', lambda interface: """
%(counts)sTwo processes now rather than one: `add process` accumulated instead
of replacing, and the count covers both of them.

Polarisation. `{X}` after a (multi)particle fixes its helicity: `{L}` and
`{R}` for left and right, `{T}` transverse, `{0}` longitudinal, `{A}` auxiliary.
It works on external particles, massless or massive, and on massive internal
particles before a decay chain.

Give the two Z bosons different decays, so that each one can be told from the
other -- it matters in a moment:
%(p)s %(pol)s
""" % {'p': P, 'counts': counts_line(interface), 'pol': POLARISED},
     title='polarisation',
     hint="Append '{0}' or '{T}' to a particle name, and give each Z its own "
          "decay in its own parentheses.",
     solution=POLARISED),

Step('generate', lambda interface: """
%(counts)sThe same count as the unpolarised process: a polarisation does not
delete diagrams, it changes which helicities are summed over in the square.

That sum is not Lorentz invariant. A polarisation means nothing until you say
in which frame it is measured, and the frame is not part of the process line
at all -- it is a run-card setting. This is what the two different decays
bought you: `e+ e-` and `mu+ mu-` tell the two Z bosons apart, so there are
three frames worth asking for here, where `z > e+ e-` on both would have left
only one.

%(closing)s

If you would rather see the frame being chosen than read that it exists, you
can write the code out and stay a minute longer. Today that setting lives in
the madevent run card, so take that path:
%(p)s output madevent %(dir)s
""" % {'counts': counts_line(interface), 'p': P, 'dir': POL_DIR,
       'closing': CLOSING},
     title='polarisation needs a frame, and that is the tour',
     hint="`output madevent NAME` writes the MG5-compatible directory, which "
          "is the one whose run card carries `me_frame`. Or stop here: the "
          "tutorial is done.",
     solution='output madevent %s' % POL_DIR),

Step('output', lambda interface: """
That wrote `%(dir)s`. `launch` runs it, and on the way it offers you the
cards -- which is where the frame is chosen.

In the run card, `me_frame` is a list of legs whose momenta are summed to
define the rest frame:

  1, 2    the incoming partons. Their sum is the ZZ system at this order, so
          this is the ZZ rest frame. It is the default.
  3, 4    the `e+ e-` pair: the rest frame of the Z you made longitudinal.
  5, 6    the `mu+ mu-` pair: the rest frame of the transverse one.

Those numbers are positions in the *normalised* leg order -- 1 and 2 incoming,
then `e+ e- mu+ mu-` -- not the order you typed. All three are legitimate and
they are different measurements: a polarised cross section quoted without its
frame does not mean anything. (`tutorial madevent` goes through the cards
properly; this is the one line of them that the process line cannot carry.)

%(p)s launch
""" % {'p': P, 'dir': _run_dir(interface)},
     title='choosing the frame at launch',
     hint="`launch` with no argument runs the directory you just wrote.",
     question_hint="Open the run card -- type `run`, the name the menu gives "
                   "it -- and set `me_frame`: `1, 2` is the ZZ rest frame "
                   "(the default), `3, 4` the `e+ e-` Z, `5, 6` the "
                   "`mu+ mu-` one. `0` or Enter takes the cards as they "
                   "stand and runs.",
     on_failure="`launch` needs the directory to exist -- run "
                "`output madevent` first.",
     solution='launch'),

Step('launch', """
That is a polarised cross section, and it is only a number next to the frame
you picked -- quote the two together or it says nothing.

That really is the end. `tutorial stop` leaves tutorial mode, and
`tutorial list` shows what else there is.
""",
     title='wrap-up'),

    ],
)
