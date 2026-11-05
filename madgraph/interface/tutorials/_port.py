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
"""Helpers for the tutorials still reading their text from tutorial_text*.py.

Only the stale prompt is rewritten; everything else is passed through
untouched, and tests/unit_tests/interface/test_tutorials.py applies the same
substitution to the expected text so that any *other* drift still fails.
"""

from __future__ import absolute_import

# the prompt has been 'MG7> ' since MG7_PROMPT, but the tutorial text was
# written when it was 'MG5_aMC> ' (and, in places, 'MG_aMC> ')
STALE_PROMPTS = ('MG5_aMC>', 'MG_aMC>')
PROMPT = 'MG7>'


def retarget(text):
    """Bring a legacy tutorial string up to date with the current prompt."""

    for stale in STALE_PROMPTS:
        text = text.replace(stale, PROMPT)
    return text
