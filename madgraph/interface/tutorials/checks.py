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
"""The `checks` tutorial -- the `check` command as a debugging tool."""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  check_line, model_line)

P = 'MG7>'
PROC = 'p p > e+ e-'


tutorial = Tutorial(
    name='checks',
    title='checking a process before you trust it',
    description='the check command family, and what each one can tell you',
    order='sequence',
    see_also=('model', 'bsm', 'syntax'),
    steps=[

Step('tutorial', """
Every check below takes seconds and most people never run one. They are worth
knowing about for two reasons: when a number looks wrong they tell you *where*
to look, and when you are handed an unfamiliar model they tell you whether to
trust it at all.

All of them work on a process definition, without producing any output
directory.

One thing first, and it is a difference from `generate`: `check` does **not**
load a model for you. `generate` quietly imports the SM if nothing is loaded;
`check` refuses with "No model currently active". So:

%(p)s import model sm
""" % {'p': P},
     title='welcome',
     solution='import model sm'),

Step('import_model', lambda interface: """
%(model)sThat is all `check` was waiting for. Every check below compares the
model against itself, which is why none of them will start without one.

Now the cheapest of them:
%(p)s check permutation %(proc)s
""" % {'p': P, 'proc': PROC, 'model': model_line(interface)},
     title='load a model first',
     hint="`check` needs a model loaded explicitly, unlike `generate`.",
     solution='check permutation %s' % PROC),

Step('check', lambda interface: """
%(verdict)sThat regenerated the process with the external legs relabelled
and compared |M|^2 across the permutations. A relabelling cannot change
physics, so any disagreement is a bug -- in the model, in the colour or
helicity bookkeeping, or in MG7 itself.

Read the columns: min, max, relative difference, verdict. What matters is the
relative difference. Machine precision is around 1e-16; anything up to 1e-10
or so is fine, and 1e-3 is a bug however confident the "Passed" looks.

Next, the sharpest one:
%(p)s check gauge %(proc)s
""" % {'p': P, 'proc': PROC, 'verdict': check_line(interface)},
     title='permutation',
     hint="`check permutation PROCESS`",
     solution='check gauge %s' % PROC),

Step('check', """
`check gauge` computes the same matrix element in the unitary, Feynman, axial
and FD gauges and compares. The gauge is not physical, so they must agree --
and they only do if the model's couplings are consistent with each other.

This is the single most useful test of a UFO model you did not write. A model
that fails it has a bug in its Lagrangian or in the FeynRules export, and no
amount of careful running will fix the number it produces.

(It is not available for loop processes.)

%(p)s check lorentz %(proc)s
""" % {'p': P, 'proc': PROC},
     title='gauge invariance',
     hint="`check gauge PROCESS`",
     solution='check lorentz %s' % PROC),

Step('check', """
Same amplitude, different frames. Catches a Lorentz structure that is written
down wrongly -- again, mostly a model problem rather than an MG7 one.

Now one that is specific to MG7. Because of flavour grouping, a single matrix
element serves several flavour combinations -- `p p > e+ e-` is really
`Q Qx > Lx L`. That merging is an optimisation, and an optimisation is
something to verify:

%(p)s check flavor %(proc)s
""" % {'p': P, 'proc': PROC},
     title='Lorentz invariance',
     hint="`check lorentz PROCESS`",
     solution='check flavor %s' % PROC),

Step('check', """
One row per flavour the merged matrix element serves -- sixteen of them for
this process, and `Summary: 16/16 passed` under them -- each compared against
the same thing computed without any grouping. All of them matching to machine
precision is what says the merging is doing only what it claims to.

The rest of the family:

  check brs        Ward identities, for a process with a massless gauge boson
                   as an external leg
  check cms        complex-mass-scheme consistency -- compares against the
                   narrow-width approximation off shell, with the width
                   progressively reduced
  check full       permutation, brs, gauge and lorentz together, plus the
                   flavour check when the model uses grouping
  check language   the same point through the Python, Fortran and C++
                   back-ends. Needs gfortran and g++, and does not accept the
                   `^2` order syntax.

And three that measure rather than verify:
  check timing     where the time goes
  check profile    timing plus stability
  check stability  how the result behaves in awkward corners of phase space

%(p)s history my_checks_session.dat
""" % {'p': P},
     title='flavour grouping',
     hint="`check flavor PROCESS`",
     solution='history my_checks_session.dat'),

Step('history', lambda interface: """
**When a check fails**, before assuming MG7 is broken:

  1. Read the relative difference, not the verdict. A "Failed" at 1e-9 usually
     means a tight tolerance meeting an awkward phase-space point; a "Passed"
     at 1e-3 would be the real problem.
  2. Try the unrestricted model (`import model sm-full`, or the equivalent for
     yours). A restriction that removed something it should not have shows up
     as a check failure.
  3. Try `set gauge Feynman` and rerun. Unitary-gauge cancellations at high
     energy can look like a bug and are not.
  4. Narrow it down: fewer legs, one flavour, no multiparticles. A failing
     2 -> 2 is a bug report someone can act on; a failing 2 -> 6 is not.

**Two other things worth reaching for when something looks wrong:**
  * `display diagrams` -- far more problems are visible in the diagram count
    than in the cross section;
  * `./bin/madgraph --debug` and the `MG5_debug` file MG7 writes on a crash,
    which is what to attach to a bug report.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='when a check fails'),

    ],
)
