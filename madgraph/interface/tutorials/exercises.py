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
"""The `exercises` tutorial -- we ask, you answer, we check.

Every check inspects the state the command produced, never the text of the
line: `p p > t t~ QED=0` and `p p > t t~ QCD=2 QED=0` are both right, and a
string match would fail whoever typed the second one.
"""

from __future__ import absolute_import

import madgraph.core.diagram_generation as diagram_generation
import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Exercise, Step, Tutorial,
                                                  core_process, describe_state,
                                                  total_diagrams)

P = 'MG7>'

# Flavour grouping means a leg's id is often a merged code rather than a PDG:
# `p` legs come through as 21 or +-81, and even `e+` becomes +-82.  So checks
# key on structure -- orders, forbidden particles, decay chains -- wherever
# they can, and only on ids for particles that are never merged.
TOP, ANTITOP = 6, -6


#===============================================================================
# reading the state
#===============================================================================

def amplitudes(interface):
    return list(getattr(interface, '_curr_amps', []) or [])


def processes(interface):
    out = []
    for amp in amplitudes(interface):
        try:
            out.append(core_process(amp))
        except Exception:
            continue
    return out


def any_process(interface, predicate):
    return any(predicate(process) for process in processes(interface))


def all_processes(interface, predicate):
    procs = processes(interface)
    return bool(procs) and all(predicate(process) for process in procs)


def final_ids(process):
    return [leg.get('id') for leg in process.get('legs') if leg.get('state')]


def orders_of(process):
    return dict(process.get('orders'))


def squared_of(process):
    return dict(process.get('squared_orders'))


def types_of(process):
    return dict(process.get('sqorders_types'))


def has_decay_chain(interface):
    return any(isinstance(amp, diagram_generation.DecayChainAmplitude)
               for amp in amplitudes(interface))


def report_diagrams(interface):
    return ('For the record, that gave %d diagrams over %d subprocess(es):\n%s'
            % (total_diagrams(interface), len(amplitudes(interface)),
               describe_state(interface)))


#===============================================================================
# the exercises
#===============================================================================

def _is_ttbar(process):
    ids = final_ids(process)
    return TOP in ids and ANTITOP in ids and len(ids) == 2


def check_ttbar(interface, line):
    return any_process(interface, _is_ttbar)


def check_ttbar_with_ew(interface, line):
    """The EW diagrams must be there, so QED cannot be pinned to zero."""

    if not any_process(interface, _is_ttbar):
        return False
    for process in processes(interface):
        if not _is_ttbar(process):
            continue
        orders = orders_of(process)
        if orders.get('QED', 0) >= 2 or ('QED' not in orders
                                         and 'WEIGHTED' not in orders):
            return True
    return False


def check_interference(interface, line):
    """Squared orders that select the interference term only."""

    for process in processes(interface):
        squared, types = squared_of(process), types_of(process)
        if squared.get('QCD') == 2 and squared.get('QED') == 2 \
                and types.get('QCD') == '==' and types.get('QED') == '==':
            return True
    return False


def mistake_amplitude_orders(interface, line):
    """QED=2 QCD=2 instead of QED^2==2 QCD^2==2."""

    for process in processes(interface):
        if not squared_of(process) and orders_of(process).get('QED') == 2 \
                and orders_of(process).get('QCD') == 2:
            return True
    return False


def mistake_single_equals(interface, line):
    """QED^2=2 -- a bare '=' means '<=', so this is not the interference."""

    for process in processes(interface):
        types = types_of(process)
        if types and any(kind != '==' for kind in types.values()):
            return True
    return False


def check_decay_chain(interface, line):
    if not has_decay_chain(interface):
        return False
    return any_process(interface, _is_ttbar)


def mistake_no_comma(interface, line):
    """A `>` where a `,` was meant makes a required s-channel, not a decay."""

    return not has_decay_chain(interface) and bool(processes(interface))


def check_forbidden_particle(interface, line):
    return all_processes(interface,
                         lambda p: bool(p.get('forbidden_particles')))


def mistake_dollar_not_slash(interface, line):
    """`$ a` removes the on-shell photon; `/ a` forbids it everywhere."""

    return any_process(interface,
                       lambda p: bool(p.get('forbidden_onsh_s_channels')
                                      or p.get('forbidden_s_channels')))


def check_forbidden_onshell(interface, line):
    return all_processes(interface,
                         lambda p: bool(p.get('forbidden_onsh_s_channels')))


def mistake_slash_not_dollar(interface, line):
    return any_process(interface, lambda p: bool(p.get('forbidden_particles')))


def check_nlo(interface, line):
    """An NLO process.

    Read it off the process definitions, whose NLO_mode is 'tree' at LO and
    'all' / 'real' / 'virt' / 'noborn' otherwise.  Not off `_fks_multi_proc`,
    which stays set after a later LO `generate` and would keep passing this
    exercise forever once it had passed once.
    """

    for definition in getattr(interface, '_curr_proc_defs', []) or []:
        try:
            if definition.get('NLO_mode') not in (None, 'tree'):
                return True
        except Exception:
            continue
    return False


tutorial = Tutorial(
    name='exercises',
    title='practise, with the answers checked',
    description='we ask, you answer, we tell you which mistake you made',
    order='sequence',
    section='exercises',
    see_also=('syntax', 'lo', 'madevent', 'standalone'),
    steps=[

Step('tutorial', """
The other tutorials show you things. This one asks.

Each exercise is a task. Type a command that does it and MG7 checks the result
-- not the text you typed, the process you actually built, so there is more
than one right answer to most of these. If you get it wrong you are told
*which* mistake it was, and you can try again as often as you like. Nothing is
ever blocked or refused.

`hint` nudges, `solution` prints one right answer, `skip` moves on,
`tutorial status` shows how far you have got, and `tutorial help` lists the
lot.

--- Exercise 1 ---------------------------------------------------------------

Generate top-quark pair production at a hadron collider.
""",
     title='welcome',
     solution='generate p p > t t~'),

Exercise('generate', """
--- Exercise 1 ---------------------------------------------------------------

Generate top-quark pair production at a hadron collider.
""",
     check=check_ttbar,
     report=report_diagrams,
     hint="`p` is the proton multiparticle; `>` separates initial from final.",
     solution='generate p p > t t~',
     title='top pair production',
     mistakes=[
        (lambda i, l: any_process(i, lambda p: len(final_ids(p)) != 2),
         "You built something, but not a 2 -> 2 process. The final state here\n"
         "is just `t t~`."),
        (lambda i, l: not amplitudes(i),
         "Nothing was generated. Check the particle names -- `t t~`, with a\n"
         "space between them -- and that you used `generate`."),
     ]),

Exercise('generate', """
--- Exercise 2 ---------------------------------------------------------------

Now the same process *including* the electroweak diagrams.

MG5 left them out a moment ago: with no orders given it searched for the
lowest WEIGHTED = QCD + 2*QED that produces anything, and stopped at the pure
QCD result. Ask for more.
""",
     check=check_ttbar_with_ew,
     report=report_diagrams,
     hint="Coupling orders go at the end of the line. You want to allow QED vertices, not forbid them.",
     solution='generate p p > t t~ QED=2',
     title='letting the EW diagrams in',
     mistakes=[
        (lambda i, l: any_process(i, lambda p: orders_of(p).get('QED') == 0),
         "`QED=0` goes the wrong way -- that *removes* the electroweak\n"
         "diagrams. This is the single most common reflex in MG5, and it is\n"
         "wrong about half the time it is typed. You want to allow QED\n"
         "vertices: `QED=2`."),
        (lambda i, l: not any_process(i, _is_ttbar),
         "That is not `p p > t t~` any more. Keep the process and change only\n"
         "the coupling orders."),
     ]),

Exercise('generate', """
--- Exercise 3 ---------------------------------------------------------------

Dijet production, `p p > j j`, has a QCD amplitude and an electroweak one, so
the squared matrix element has three pieces: QCD squared, EW squared, and the
interference between them.

Generate **the interference term only**.
""",
     check=check_interference,
     report=report_diagrams,
     hint="A constraint on the squared matrix element, not on the amplitude. The syntax has a `^2` in it.",
     solution='generate p p > j j QCD^2==2 QED^2==2',
     title='interference only',
     mistakes=[
        (mistake_amplitude_orders,
         "Close, but `QED=2 QCD=2` constrains the *amplitude*: it says each\n"
         "diagram may have at most two of each vertex, and you get all three\n"
         "terms of |M|^2 back.\n\n"
         "Interference is a statement about the squared matrix element, so the\n"
         "constraint needs `^2`: `QCD^2==2 QED^2==2`."),
        (mistake_single_equals,
         "You used `^2`, so you are asking about the squared matrix element --\n"
         "but with a single `=`. In MG5 a bare `=` means `<=`, so that admits\n"
         "the other terms too. Only `==` means exactly:\n"
         "  generate p p > j j QCD^2==2 QED^2==2"),
        (lambda i, l: not any_process(i, lambda p: bool(squared_of(p))),
         "No squared-order constraint reached the process. The orders go after\n"
         "the final state, like `p p > j j QCD^2==2 QED^2==2`."),
     ]),

Exercise('generate', """
--- Exercise 4 ---------------------------------------------------------------

Generate top pair production where **both** tops decay -- say
`t > w+ b` and `t~ > w- b~` -- as a decay chain in the process line.
""",
     check=check_decay_chain,
     report=report_diagrams,
     hint="A comma opens a decay chain. `>` inside one process means something else entirely.",
     solution='generate p p > t t~, t > w+ b, t~ > w- b~',
     title='decay chains',
     mistakes=[
        (mistake_no_comma,
         "That generated a process, but not a decay chain.\n\n"
         "The separator is a **comma**: `p p > t t~, t > w+ b`. A `>` inside a\n"
         "single process means a required s-channel instead -- a completely\n"
         "different request, which is why it did not error.\n\n"
         "One thing to know while you are here: you do not need to write the\n"
         "decay twice for identical particles. Every `t` in the process is\n"
         "decayed by one `t > ...`."),
        (lambda i, l: not amplitudes(i),
         "Nothing was generated -- check the decay is something the model\n"
         "allows, like `t > w+ b`."),
     ]),

Exercise('generate', """
--- Exercise 5 ---------------------------------------------------------------

Generate `p p > e+ e-` with the photon forbidden **everywhere in the diagram**,
internal lines included -- not merely removed when it is on shell.

(Then try the other one afterwards and compare the diagram counts. The contrast
is the whole lesson.)
""",
     check=check_forbidden_particle,
     report=report_diagrams,
     hint="Three operators do related things: `$`, `$$` and `/`. Only one forbids a particle anywhere.",
     solution='generate p p > e+ e- / a',
     title='forbidding a particle',
     mistakes=[
        (mistake_dollar_not_slash,
         "You used `$` (or `$$`), which talks about the s-channel:\n"
         "  `$ a`   keeps the diagram, removes the on-shell photon. Gauge-safe.\n"
         "  `$$ a`  drops the s-channel photon diagram entirely.\n\n"
         "To forbid the photon *anywhere*, internal lines included, the\n"
         "operator is `/`:\n"
         "  generate p p > e+ e- / a"),
     ]),

Exercise('generate', """
--- Exercise 6 ---------------------------------------------------------------

Now the other side of the same coin. Generate `p p > e+ e-` again, this time
removing only the **on-shell** Z contribution -- keeping the diagram, killing
the resonance.
""",
     check=check_forbidden_onshell,
     report=report_diagrams,
     hint="The gauge-safe one of the three exclusion operators.",
     solution='generate p p > e+ e- $ z',
     title='excluding a resonance',
     mistakes=[
        (mistake_slash_not_dollar,
         "`/ z` forbids the Z everywhere, which throws away the diagram rather\n"
         "than the resonance -- and, in general, gauge invariance with it.\n\n"
         "To remove only the on-shell contribution, use `$`:\n"
         "  generate p p > e+ e- $ z"),
     ]),

Exercise('generate', """
--- Exercise 7 ---------------------------------------------------------------

Last one. Take top pair production to next-to-leading order in QCD.

Watch what MG7 does as it runs: the process itself makes it switch to the
aMC@NLO interface.
""",
     check=check_nlo,
     hint="Square brackets after the process, naming the perturbative order.",
     solution='generate p p > t t~ [QCD]',
     title='going to NLO',
     mistakes=[
        (lambda i, l: bool(processes(i)) and not check_nlo(i, l),
         "That is still a leading-order process. The NLO request goes in\n"
         "square brackets at the end:\n"
         "  generate p p > t t~ [QCD]\n\n"
         "`[real=QCD]` and `[virt=QCD]` ask for only the real-emission or only\n"
         "the loop part, and `[noborn=QCD]` is for loop-induced processes."),
     ]),

Step('__done__', lambda interface: """
That is the set.

What these were really about, in one line each:
  1-2  MG5 chooses coupling orders for you, and `QED=0` is not the same
       request as "the QCD part".
  3    interference lives in the *squared* matrix element, so it needs `^2`
       and `==`.
  4    a comma opens a decay chain; a `>` does something else and will not
       warn you.
  5-6  `/` forbids a particle, `$` removes a resonance, `$$` drops the
       s-channel. They are not interchangeable and only one is gauge-safe.
  7    the process line decides which interface you are in.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='what those were about'),

    ],
)
