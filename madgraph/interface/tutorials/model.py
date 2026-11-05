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
"""The `model` tutorial -- loading, inspecting and shaping a model.

Takes over the material the pre-2026 `lo` tutorial tacked on after `launch`:
importing a model, looking inside it, restrictions, `customize_model` and
`define`.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import Step, Tutorial

P = 'MG7>'


# ---------------------------------------------------------------------------
# The lesson above lists half a dozen `display` commands and invites the user
# to try them.  Each one has something worth saying -- how to narrow it down,
# which convention the output follows -- and none of them is the command the
# tutorial is waiting for, so this step answers and stays put.
# ---------------------------------------------------------------------------

DISPLAY_NOTES = {

'particles': """Every particle of the model, with its PDG code.

The names are the MG5 ones: `ta-` for the tau, `vt` for its neutrino, a
trailing `~` for an antiparticle. That translation is what `--modelname`
switches off, and the names it maps to live in `input/default_particle.dat`.

One particle at a time gives you its mass, width, spin and colour:
  display particles t""",

'interactions': """Every vertex of the model -- several hundred of them for the
SM, which is rarely what you want.

Ask for the ones you care about instead, by particle:
  display interactions t t~ g
or for a single one by number, as they are numbered in that listing:
  display interactions 12""",

'couplings': """The couplings as MG5 built them, grouped by what they depend on.

One at a time, by name, shows the definition it came from in the UFO:
  display couplings GC_12""",

'parameters': """The parameters the couplings are built from: the external ones
-- the entries of the param_card -- first, then everything derived from them.

This one has no filter: it prints the lot. `display couplings GC_12` is the
way in when you are chasing where a number comes from.""",

'multiparticles': """The labels that stand for a set of particles.

`p`, `j`, `l+`, `l-`, `vl` and `vl~` come predefined -- that is why `p p > j j`
means what it means.

`p` and `j` are the gluon plus the quarks the model treats as massless, so
their content follows the number of massless flavours: four here, and you will
see them pick up the b when it goes massless in a moment.

You will add your own with `define` later in this tutorial.""",

'modellist': """The models MG7 can download for you, which is many more than
the handful shipped in `models/`. `import model NAME` fetches one on demand.""",

'coupling_order': """The coupling orders the model declares -- QCD, QED and
whatever else it defines -- with the hierarchy MG5 uses to decide what a
process means when you do not spell the orders out.

Worth a look before you generate anything in a model you do not know: those
names are what you constrain to keep the diagrams you want and drop the rest,
as in `generate p p > t t~ QED=0`. The `syntax` tutorial does that properly.""",

'lorentz': """The Lorentz structures the vertices are built from.
`display lorentz NAME` shows one of them.""",
}


def display_note(interface, line):
    """What to say about the `display` the user just tried."""

    args = (line or '').split()
    what = args[1].lower() if len(args) > 1 else ''
    note = DISPLAY_NOTES.get(what)
    if note is None:
        note = ("`display %s` is not one of the things this lesson lists; "
                "`help display` has the full set." % what if what else
                "`display` needs to be told what to show -- `display particles`, "
                "`display interactions`, ... -- see `help display`.")
    return '%s\n\nWhen you have seen enough, carry on with:\n%s import model sm-no_b_mass' \
           % (note, P)


tutorial = Tutorial(
    name='model',
    title='working with models',
    description='import, inspect, restrict and customise the physics model',
    order='sequence',
    section='advanced',
    ai_generated=False,
    see_also=('bsm', 'syntax', 'lo', 'checks'),
    steps=[

Step('tutorial', """
Every process you generate is read out of a model: the particles, the vertices,
the parameters and the couplings. MG7 ships with the Standard Model loaded, and
most of the time you never think about it -- until you need a different one, or
a different corner of the same one.

This tutorial covers loading a model, looking inside it, cutting it down, and
changing how it is treated.

Start by loading the SM explicitly:
%(p)s import model sm
""" % {'p': P},
     title='welcome',
     solution='import model sm'),

Step('import_model', """
A model is a UFO directory: Python files describing particles, vertices,
parameters and couplings, usually written by FeynRules. `import model NAME`
looks for `NAME` in `models/`, then downloads it if it is in the model
database. You can also give a path directly:

  import model /path/to/my_UFO_model

One option worth knowing now: `--modelname` keeps the model's own particle
names instead of translating them to the MG5 ones, which are defined in
`input/default_particle.dat`.

Look at what you just loaded:
%(p)s display particles
""" % {'p': P},
     title='load a model',
     hint="`import model NAME`, where NAME is a directory under models/.",
     solution='display particles'),

Step('display', """
`display` is how you interrogate a model without generating anything:

  display particles            every particle, with its name and PDG code
  display particles t          everything about one particle: mass, width,
                               spin, colour, its antiparticle
  display interactions         every vertex
  display interactions t t~ g  the vertices involving exactly those particles
  display couplings            the coupling values
  display parameters           the parameters they are built from
  display multiparticles       the labels p, j, l+, l- and any you defined
  display modellist            the models MG7 can download for you -- many of
                               them; `import model NAME` fetches one on demand
  display coupling_order       the coupling orders the model defines

`display interactions` on its own is long; the filtered form is what you
usually want.

Now something less obvious. The SM you just loaded is *restricted*: several
parameters are fixed and some interactions removed, because carrying them adds
diagrams that contribute nothing. Load a different restriction and compare:

%(p)s import model sm-no_b_mass
""" % {'p': P},
     title='look inside the model',
     hint="`display particles`, `display interactions`, `display parameters`.",
     solution='import model sm-no_b_mass'),

Step(('display', 'display_multiparticles'), display_note,
     title='trying the display commands',
     sticky=True,
     hint="`display particles`, `display interactions t t~ g`, ...",
     solution='import model sm-no_b_mass'),

Step('import_model', """
`MODEL-RESTRICTION` loads `restrict_RESTRICTION.dat` from the model directory.
With no restriction named you get `restrict_default.dat`, which is why the
plain `sm` already has the light-quark masses and most CKM mixing switched off.

The ones the SM ships with:
  sm                  the default restriction
  sm-full             nothing restricted at all
  sm-no_b_mass        massless b, for 5-flavour-scheme calculations
  sm-lepton_masses    keep the charged-lepton masses

A restriction file is just a param card: any parameter set to zero is removed
from the model along with the interactions that need it, and identical values
are merged into one parameter. That is why a restricted model generates fewer
diagrams and runs faster -- and why a result that surprises you is worth
re-checking against `-full`.

`explain_restriction` tells you what the card you loaded actually did: for each
of its entries, which couplings it drops and how many interactions go with
them. Run it whenever a diagram you expected is missing.

Next, how the model is *treated* rather than what is in it:

%(p)s set gauge Feynman
""" % {'p': P},
     title='restrictions',
     hint="Append `-RESTRICTION` to the model name.",
     solution='set gauge Feynman'),

Step('explain_restriction', """
That report is the restriction card read back, one entry at a time.

Each line is one thing the card fixes -- a parameter set to zero or to one, or
a family it gives a common value -- followed by what that entry *on its own*
takes out of the model. `dropped` counts the couplings which then evaluate to
zero, so the vertices needing them go; `fused` counts the ones which become
equal to another and are merged into a single coupling.

Two things are worth noticing in it. Setting a mass to zero often drops no
coupling at all -- a mass lives in the propagator, the Yukawa of the same
particle is the one sitting in the vertices. And a coupling can be listed under
two entries: either of them alone is enough to kill it, which the closing line
counts.

`explain_restriction --all` names every coupling instead of the first few, and
`explain_restriction sm-full` works on any card, not only the one you loaded.

Back to the tutorial:
%(p)s set gauge Feynman
""" % {'p': P},
     title='a detour: explain_restriction',
     sticky=True,
     hint="`explain_restriction` reports on the card the model was loaded with.",
     solution='set gauge Feynman'),

Step('set', """
`set gauge` chooses the gauge for the non-QCD part, and reloads the model:

  unitary   the default: no goldstones, fewer diagrams, bigger cancellations
  Feynman   goldstones present, better numerical behaviour at high energy,
            and the only choice for loop processes
  axial     the parton-shower gauge, massless particles only
  FD        Feynman Diagram gauge, the extension of axial to massive
            particles (arXiv:2203.10440, 2405.01256)

Comparing unitary and Feynman is also the cheapest test that a model is
self-consistent -- that is exactly what `check gauge` does for you. It compares
them for a process, so give it one: `check gauge p p > e+ e-`.

Two more model-level settings:
  set complex_mass_scheme True   widths in the propagator *and* in the
                                 couplings, for results that stay gauge
                                 invariant near a resonance
  set EWscheme                   which electroweak inputs are taken as
                                 independent

Now something you will use constantly:
%(p)s define v = w+ w- z a
""" % {'p': P},
     title='gauge and scheme',
     hint="`set gauge unitary|Feynman|axial|FD`.",
     solution='define v = w+ w- z a'),

Step('check', """
`check gauge` compares the gauges for *a process*, so it needs one:

  check gauge p p > e+ e-

It generates that process in both gauges and compares the matrix elements
point by point; they have to agree to numerical precision. `check full` runs
this and the other checks together -- the `checks` tutorial goes through them.

Back to the model. Give a name to a set of particles:
%(p)s define v = w+ w- z a
""" % {'p': P},
     title='a detour: check gauge',
     entry='check gauge p p > e+ e-',
     hint="`check gauge PROCESS`, for instance `check gauge p p > e+ e-`.",
     on_failure="`check` compares a *process* between two computations, so it "
                "needs one. Try:\n  check gauge p p > e+ e-",
     solution='define v = w+ w- z a'),

Step('define', """
`define` makes a multiparticle label. `v` now stands for any electroweak vector
boson and works anywhere a particle name does:

  generate p p > v v

`p`, `j`, `l+`, `l-`, `vl` and `vl~` are predefined the same way, which is why
`p p > j j` means what it means. What `p` and `j` hold is not fixed: the gluon,
and every quark the model treats as massless. Their content follows the number
of massless flavours, so it grew by one when you loaded `sm-no_b_mass` a moment
ago -- four flavours in the default SM, five with a massless b.

A definition can use `/` to exclude, as in `define aUPC = a j / g`.

See them all, including the one you just made:
%(p)s display multiparticles
""" % {'p': P},
     title='multiparticle labels',
     hint="`define LABEL = particle particle ...`",
     solution='display multiparticles'),

Step('display_multiparticles', lambda interface: """
Three more things, worth knowing they exist:

  * **`customize_model`** opens an interactive menu of the switches a model
    exposes -- the flavour scheme, how many leptons are massive, a diagonal
    CKM, a parameter or a whole block set to zero. It is the practical way to
    write a restriction file: `customize_model --save=NAME` saves the answers
    as `restrict_NAME.dat` next to the model, and `import model MODEL-NAME`
    loads them back.
  * **`add model OTHER`** merges a second model into the current one, which is
    how you bolt an extra sector onto the SM. It writes a combined model
    directory (`sm__OTHER`) and keeps your multiparticle definitions.

And `save model PATH` writes the current model, restrictions and all, so a
collaborator gets exactly what you had.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='customise, merge, save'),

    ],
)
