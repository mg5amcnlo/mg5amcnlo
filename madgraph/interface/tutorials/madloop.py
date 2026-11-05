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
"""The `madloop` tutorial.  Ported from tutorial_text_madloop.py."""

from __future__ import absolute_import

import madgraph.interface.tutorial_text_madloop as legacy
from madgraph.interface.tutorials._port import retarget
from madgraph.interface.tutorials.session import Step, Tutorial


tutorial = Tutorial(
    name='madloop',
    title='loop matrix elements with MadLoop',
    description='generate and check standalone loop matrix elements',
    aliases=('MadLoop',),
    order='free',
    section='advanced',
    ai_generated=False,
    steps=[
        Step('tutorial', retarget(legacy.tutorial),
             title='welcome',
             solution='generate p p > t t~ [virt=QCD]'),
        Step('generate', retarget(legacy.generate),
             title='generate a loop process',
             solution='output MY_FIRST_MADLOOP_RUN'),
        Step('display_processes', retarget(legacy.display_processes),
             title='list the defined processes',
             solution='output MY_FIRST_MADLOOP_RUN'),
        Step('display_diagrams', retarget(legacy.display_diagrams),
             title='draw the loop diagrams',
             solution='output MY_FIRST_MADLOOP_RUN'),
        Step('add_process', retarget(legacy.add_process),
             title='add a second process',
             solution='output MY_FIRST_MADLOOP_RUN'),
        Step('output', retarget(legacy.output),
             title='produce an output',
             solution='launch MY_FIRST_MADLOOP_RUN'),
        Step('launch', retarget(legacy.launch),
             title='run it'),
        Step('check', retarget(legacy.check),
             title='check the result'),
        Step('check_profile', retarget(legacy.check_profile),
             title='profile the computation'),
    ],
)
