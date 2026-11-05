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
"""The `nlo` tutorial -- running aMC@NLO.  Ported from tutorial_text_nlo.py."""

from __future__ import absolute_import

import madgraph.interface.tutorial_text_nlo as legacy
from madgraph.interface.tutorials._port import retarget
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  pythia8_available)


SHOWER_NOTE_PY8 = """
-- About the parton shower ---------------------------------------------------

`parton_shower` in the run card decides which shower the events are matched to,
and MC@NLO counterterms are shower-specific: the parton-level file you get can
only be showered with the one you chose. The default is **PYTHIA8**, which this
MG7 has installed, so `launch` will work as written.

(HERWIG6 is still selectable, but it is Fortran from the early 2000s and no
longer compiles with current gfortran, so do not reach for it expecting it to
build.)
"""

SHOWER_NOTE_NO_PY8 = """
-- About the parton shower: read this before you launch ----------------------

`parton_shower` in the run card decides which shower the events are matched to.
The default is PYTHIA8, and **this MG7 does not have it**. Install it first:

  MG7> install pythia8

It takes a while, but it is the path that works.

If you would rather not wait, you can run the fixed-order parton level only:

  MG7> launch -p

**But understand what that gives you.** The .lhe file MC@NLO produces is
UNPHYSICAL on its own: the events carry the counterterms that cancel against
the shower's first emission, so individual events (and any distribution you
make from them) are only meaningful once showered. Use `launch -p` to check
that the machinery runs, never to get a number.

Do not reach for HERWIG6 as a substitute either: it is Fortran from the early
2000s and no longer compiles with current gfortran.
"""


def _output_text(interface=None):
    """The ported text, plus what to do about the shower on this machine."""

    note = (SHOWER_NOTE_PY8 if pythia8_available(interface)
            else SHOWER_NOTE_NO_PY8)
    return retarget(legacy.output) + note


tutorial = Tutorial(
    name='nlo',
    title='NLO computations with aMC@NLO',
    description='generate, output and run a process at next-to-leading order',
    aliases=('aMCatNLO',),
    section='basic',
    ai_generated=False,
    order='free',
    steps=[
        Step('tutorial', retarget(legacy.tutorial),
             title='welcome',
             solution='generate p p > t t~ [QCD]'),
        Step('generate', retarget(legacy.generate),
             title='generate a process at NLO',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step('display_processes', retarget(legacy.display_processes),
             title='list the defined processes',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step('add_process', retarget(legacy.add_process),
             title='add a second process',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step(('output', 'open_index'), _output_text,
             title='produce an output',
             solution='launch MY_FIRST_AMCATNLO_RUN'),
        Step('launch', retarget(legacy.launch),
             title='run it'),
    ],
)
