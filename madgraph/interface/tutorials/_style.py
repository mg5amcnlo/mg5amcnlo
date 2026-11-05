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
"""Turn the light markup tutorials are written in into terminal styling.

Tutorial text is authored with `**emphasis**` and `` `code` `` because that is
readable in the source; a terminal shows those literally, asterisks and all.
This translates them, at emit time, into the markers MG5's ColorFormatter
already understands:

    **text**   ->  $_BOLD text $_RESET     bold
    *text*     ->  underlined              the terminal's nearest thing to
                                           italic, which has no MG5 marker
    `text`     ->  $GREEN text $_RESET     green, MG5's convention for
                                           commands and file names

`$_BOLD`, `$_RESET` and the colour names are substituted unconditionally by
ColorFormatter.format(), unlike `$BOLD`/`$RESET`/`$COLOR`, which are dropped
for an INFO record that did not ask for a colour -- so the markers here are
deliberately the underscored ones.

Converting on the way out rather than in the source keeps the tutorials
readable and reviewable, and means one place to change if the styling does.
"""

from __future__ import absolute_import

import re

BOLD_ON, BOLD_OFF = '$_BOLD', '$_RESET'
CODE_ON, CODE_OFF = '$GREEN', '$_RESET'
# no formatter marker exists for underline, so the escape goes in directly;
# ColorFormatter passes anything it does not recognise straight through
ITALIC_ON, ITALIC_OFF = '\033[4m', '\033[0m'

# Both are line-local on purpose: a span that wrapped would colour the leading
# indentation of the next line, and there is a test keeping the sources free of
# wrapping spans.
_BOLD = re.compile(r'\*\*(?P<text>[^*\n]+?)\*\*')
# a single-* span, run after the ** ones have gone. It must not swallow a
# bullet ("  * a point"), hence requiring non-space just inside each marker.
_ITALIC = re.compile(r'(?<![\w*])\*(?=\S)(?P<text>[^*\n]+?)(?<=\S)\*(?![\w*])')
_CODE = re.compile(r'`(?P<text>[^`\n]+?)`')

# The formatter eats "$" followed by one of its keywords, so text that means a
# literal one has to be protected. MG5's own process syntax uses "$" for
# forbidden s-channels ("p p > e+ e- $ z"), which never collides -- but the
# `syntax` tutorial talks about "$" a lot, so this is checked rather than
# assumed.
_FORMATTER_KEYWORDS = (
    '_BOLD', '_RESET', 'BR', 'COLOR', 'RESET', 'BOLD', 'BLACK', 'RED',
    'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE', 'WARNING', 'INFO',
    'DEBUG', 'CRITICAL', 'ERROR', 'BG-', 'BG',
)
_KEYWORD = re.compile(r'\$(%s)' % '|'.join(re.escape(k)
                                           for k in _FORMATTER_KEYWORDS))


def has_formatter_keyword(text):
    """True if `text` contains a "$" the ColorFormatter would swallow."""

    return bool(_KEYWORD.search(text))


def to_terminal(text):
    """Render tutorial markup for a terminal."""

    if not text:
        return text
    text = _BOLD.sub(lambda m: '%s%s%s' % (BOLD_ON, m.group('text'), BOLD_OFF),
                     text)
    text = _ITALIC.sub(
        lambda m: '%s%s%s' % (ITALIC_ON, m.group('text'), ITALIC_OFF), text)
    text = _CODE.sub(lambda m: '%s%s%s' % (CODE_ON, m.group('text'), CODE_OFF),
                     text)
    return text
