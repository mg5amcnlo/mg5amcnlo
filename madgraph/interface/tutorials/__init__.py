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
"""Registry of the available tutorials.

Adding a tutorial means writing one module here and listing it in _MODULES; no
logger, no logging.conf entry and no edit to the interface is needed.  Plugins
register their own through register().
"""

from __future__ import absolute_import

import importlib
import logging

from madgraph.interface.tutorials.session import Step, Tutorial, TutorialSession

logger = logging.getLogger('madgraph')

# modules of this package holding a module-level `tutorial`, in menu order
_MODULES = ['lo', 'syntax', 'mg7', 'madevent', 'model', 'bsm',
            'standalone', 'decays', 'nlo', 'madloop', 'checks', 'exercises']

_REGISTRY = []          # list of Tutorial, in menu order
_BY_NAME = {}           # name or alias (lowercased) -> Tutorial
_LOADED = False


def register(tutorial, override=False):
    """Add a Tutorial to the registry.  Used by _load() and by plugins."""

    for name in tutorial.names:
        key = name.lower()
        if key in _BY_NAME and not override:
            raise ValueError('tutorial name %r is already taken by %r'
                             % (name, _BY_NAME[key].name))
    existing = _BY_NAME.get(tutorial.name.lower())
    if existing is not None and existing in _REGISTRY:
        _REGISTRY[_REGISTRY.index(existing)] = tutorial
    else:
        _REGISTRY.append(tutorial)
    for name in tutorial.names:
        _BY_NAME[name.lower()] = tutorial
    return tutorial


def load_plugin_tutorials(plugin_path):
    """Register tutorials a plugin exposes as `new_tutorial`.

    Sits beside the plugin API's `new_output` / `new_cluster` / `new_interface`
    hooks: a plugin declares

        new_tutorial = {'mytool': mytool_tutorial.tutorial}

    and `tutorial mytool` works, with no edit to MG7.  Called from the
    interface, which is what knows the plugin path.
    """

    import madgraph.various.misc as misc

    _load()
    names = misc.from_plugin_import(plugin_path, 'new_tutorial', keyname=None,
                                    warning=True) or []
    for name in names:
        if get(name) is not None:
            logger.debug('plugin tutorial %s shadowed by a built-in', name)
            continue
        tutorial = misc.from_plugin_import(
            plugin_path, 'new_tutorial', keyname=name, warning=True,
            info='Using tutorial %(key)s from plugin %(plug)s')
        if tutorial is not None:
            register(tutorial, override=True)


def _load():
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for name in _MODULES:
        try:
            module = importlib.import_module(
                'madgraph.interface.tutorials.%s' % name)
        except ImportError as error:
            # a tutorial module that is simply absent is fine (a plugin may add
            # it later); one that fails to import for any other reason is a bug
            # and must not be swallowed
            if getattr(error, 'name', None) != 'madgraph.interface.tutorials.%s' % name:
                raise
            logger.debug('tutorial %s not available: %s', name, error)
            continue
        register(module.tutorial)


def all_tutorials(include_hidden=False):
    """Every registered tutorial, in menu order."""

    _load()
    return [t for t in _REGISTRY if include_hidden or not t.hidden]


def names(include_aliases=False, include_hidden=False):
    """Names accepted by the `tutorial` command (menu order, aliases last)."""

    tutos = all_tutorials(include_hidden=include_hidden)
    out = [t.name for t in tutos]
    if include_aliases:
        for tuto in tutos:
            out.extend(tuto.aliases)
    return out


SECTION_TITLES = {
    'basic': 'Basic',
    'advanced': 'Advanced',
    'more': 'More topics',
    'exercises': 'Exercises',
}

# shown under a section heading whenever anything in it is AI-generated
AI_NOTICE = 'AI-generated, not yet validated by the developers'
# ... and its counterpart, for the sections that carry none
VALIDATED_NOTICE = 'validated by the developers'


def by_section(include_hidden=False):
    """The tutorials grouped for the menu.

    Yields (section_key, title, [tutorial, ...], notice) in menu order,
    skipping empty sections.  `notice` says where the content comes from: the
    AI caveat when any tutorial in the section carries it, and the validated
    counterpart otherwise.  It follows the content rather than the heading, so
    the tutorials carried over from the hand-written text are not tarred with
    it -- and Tutorial refuses to put an AI-generated one in a section whose
    heading would then be claiming more than is true.
    """

    tutos = all_tutorials(include_hidden=include_hidden)
    for key in Tutorial.SECTIONS:
        group = [t for t in tutos if t.section == key]
        if not group:
            continue
        notice = (AI_NOTICE if any(t.ai_generated for t in group)
                  else VALIDATED_NOTICE)
        yield key, SECTION_TITLES.get(key, key.title()), group, notice


def get(name):
    """Look a tutorial up by name or alias.  None if unknown."""

    _load()
    return _BY_NAME.get(str(name).lower())


def start(name):
    """Return a fresh TutorialSession for `name`, or None if unknown."""

    tutorial = get(name)
    if tutorial is None:
        return None
    return TutorialSession(tutorial)


def where_next():
    """The closing pointer every tutorial ends on.

    Tutorials used to end by listing the others with their descriptions --
    eleven lines that grew every time one was added, and that `tutorial list`
    already prints on demand. One sentence instead, defined here so the
    wording is the same everywhere.
    """

    return "Type `tutorial list` to see the other tutorials."
