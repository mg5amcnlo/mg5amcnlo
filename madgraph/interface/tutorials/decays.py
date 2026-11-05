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
"""The `decays` tutorial -- decay chains, MadSpin, MadWidth and the shower."""

from __future__ import absolute_import

import os

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line, output_name,
                                                  run_line)

P = 'MG7>'

CHAIN = 'generate p p > t t~, t > w+ b, t~ > w- b~'
CHAIN_DIR = 'TT_DECAY'
UNDECAYED = 'generate p p > t t~'
SPIN_DIR = 'TT_MADSPIN'
# the closing exercise: a W+ produced beside the tops, and another one inside
# the top decay -- the parentheses are the only thing telling them apart
NESTED = 'generate p p > t t~ w+, (t > w+ b, w+ > l+ vl), t~ > w- b~, w+ > j j'
FLAT = 'generate p p > t t~ w+, t > w+ b, w+ > l+ vl, t~ > w- b~'


# shown at the card question `launch` asks, not before it: that is where the
# reader can act on it (the same arrangement as lo's LAUNCH_QUESTION_HINT)
WIDTH_AT_LAUNCH = """
One thing to settle before it runs: the top width.

It sits in the resonance propagator, and nothing checks it against the masses
and couplings in the param card. Set it to `Auto` and MG7 computes it from the
model when the run starts, so it follows the mass whenever you change it:

  set width 6 auto

For the SM the shipped value already agrees with the model, so here it changes
nothing -- it is the habit that matters, for the day you change a mass or load
another model. Then `0` or Enter to start.
"""

MADSPIN_AT_LAUNCH = """
Switch MadSpin on and give it the decays:

  madspin=ON
  decay t > w+ b
  decay t~ > w- b~

Each `decay` line replaces that particle's line in the MadSpin card. The
default card also decays the W bosons into light fermions; these two keep them
undecayed, as in the decay chain, so the two runs describe the same final
state.

MadSpin needs the right width as much as the decay chain did: it uses it for
the Breit-Wigner of each top it decays, and to turn the decay it computes into
a branching ratio. This is a fresh param card, so set it again:

  set width 6 auto

In general that is what you will do with every width: leave it to `Auto`
unless you have a reason not to. Then `0` or Enter to start.
"""


def _written_width(interface=None, pdg=6):
    """What `Auto` wrote into the run's param card: the width, and under it
    the branching ratios -- quoted from the file, the lines the reader can go
    and look at.  '' when there is no card to read (no output, or a width that
    was never set to Auto and so has no BR lines under it).
    """

    try:
        done = getattr(interface, '_done_export', None)
        path = os.path.join(done[0], 'Cards', 'param_card.dat')
        with open(path) as handle:
            lines = handle.read().split('\n')
    except Exception:
        return ''

    width, channels = None, []
    for i, line in enumerate(lines):
        bits = line.split('#')[0].split()
        if (len(bits) >= 3 and bits[0].upper() == 'DECAY'
                and bits[1] == str(pdg)):
            try:
                width = float(bits[2])
            except ValueError:
                return ''
            for rest in lines[i + 1:]:
                data = rest.split('#')[0].split()
                if not data:
                    continue
                if data[0].upper() in ('DECAY', 'BLOCK'):
                    break
                try:
                    channels.append((float(data[0]),
                                     [int(x) for x in data[2:]]))
                except ValueError:
                    break
            break
    if width is None or not channels:
        return ''

    model = getattr(interface, '_curr_model', None)

    def name(pid):
        try:
            part = model.get_particle(abs(pid))
            return part.get('name') if pid > 0 else part.get('antiname')
        except Exception:
            return str(pid)

    rows = '\n'.join('     %5.4g%%   %s'
                     % (100 * br, ' '.join(name(p) for p in ids))
                     for br, ids in channels[:4])
    return ("It wrote the result back into `%s/Cards/param_card.dat`: the "
            "width, and\nunder it the branching ratios it found --\n\n"
            "  DECAY  %d  %.6e\n%s\n\n"
            "so the card now holds numbers computed from the model, not the "
            "benchmark\nit shipped with.\n\n"
            % (os.path.basename(os.path.normpath(done[0])), pdg, width, rows))


def _ran(interface, what):
    """One line on the run that just finished, in its own number."""

    result = run_line(interface)
    if not result:
        return ''
    return 'The run gave %s for %s.\n' % (result, what)


tutorial = Tutorial(
    name='decays',
    title='decaying unstable particles',
    description='decay chains, MadSpin, widths, and which to use when',
    order='sequence',
    section='advanced',
    ai_generated=False,
    see_also=('syntax', 'madevent', 'model', 'bsm'),
    steps=[

Step('tutorial', """
Most massive particles are unstable and decay. There are three places where
the code can simulate a decay, they do different things, and choosing wrongly
is one of the easier ways to get a wrong answer that looks right.

  1. **In the process line** -- a decay chain. Exact spin correlations, the
     exact matrix element, and decay products you can put cuts on.
  2. **MadSpin**, after generation. Spin correlations kept, production
     untouched, so the cost barely grows with the length of the cascade.
  3. **The parton shower** (Pythia8). Cheapest, and it throws the spin
     correlations away.

All three rely on the narrow-width approximation to be valid -- the width small
against the mass -- and all three depend on widths you have to get right. This
tutorial runs the first two on the same top pair, then ends on the syntax that
trips people up in the first.

Start with a decay chain:
%(p)s %(chain)s
""" % {'p': P, 'chain': CHAIN},
     title='welcome',
     solution=CHAIN),

Step('generate', lambda interface: """
%(counts)sMG7 generates the production and the decays separately and reports
their diagrams added up. `output` stitches them back together -- one full
diagram per production diagram here, since each decay has only one, so as
many as plain `p p > t t~` has. What a decay chain buys is the full matrix
element: production, propagator and decay together, with each top's spin
correlations carried into its decay products.

And it uses no branching ratio. The decay is inside the matrix element, so the
b quarks and the W bosons are genuine final-state particles you can put cuts
on -- which a branching ratio applied afterwards would not allow.

Run it:
%(p)s output %(dir)s
""" % {'p': P, 'dir': CHAIN_DIR, 'counts': counts_line(interface)},
     title='decay chains',
     hint="A comma opens the decay; each decaying particle gets one "
          "statement.",
     solution='output %s' % CHAIN_DIR),

Step('output', lambda interface: """
That wrote `%(dir)s`: the process code for the decay chain, with its cards.
`launch` compiles it, integrates it and writes the events:
%(p)s launch
""" % {'p': P, 'dir': output_name(interface, CHAIN_DIR)},
     title='the decay chain, ready to run',
     hint="`launch` with no argument runs the directory you just wrote.",
     question_hint=WIDTH_AT_LAUNCH,
     solution='launch'),

Step('launch', lambda interface: """
%(ran)sThe log shows `Computing 'auto' width(s) for 6` before the integration:
that is MadWidth, run for you because of `Auto`.

%(width)sNow the same tops, decayed by MadSpin instead. The process line loses
its decays -- MadSpin will add them to the events afterwards:
%(p)s %(undecayed)s
""" % {'p': P, 'undecayed': UNDECAYED,
       'ran': _ran(interface, 't t~ with both tops decayed to W b'),
       'width': _written_width(interface)},
     title='the decay chain, run',
     solution=UNDECAYED),

Step('generate', lambda interface: """
%(counts)sThe undecayed production, nothing more: MadSpin never touches these
diagrams, which is why a long cascade costs it so little.

%(p)s output %(dir)s
""" % {'p': P, 'dir': SPIN_DIR, 'counts': counts_line(interface)},
     title='MadSpin: produce first',
     solution='output %s' % SPIN_DIR),

Step('output', lambda interface: """
That wrote `%(dir)s`: the undecayed top pair, with its cards -- MadSpin's
among them. Run it:
%(p)s launch
""" % {'p': P, 'dir': output_name(interface, SPIN_DIR)},
     title='MadSpin, ready to run',
     hint="`launch` with no argument runs the directory you just wrote.",
     question_hint=MADSPIN_AT_LAUNCH,
     solution='launch'),

Step('launch', lambda interface: """
%(ran)sMadSpin then decayed those events, keeping the spin correlations by
reweighting against the decay matrix element. It also says how much of each
resonance it kept -- the `Breit-Wigner truncation` line in its log -- and
scales the cross section it reports by that fraction.

**Which to use.**
  * decay chain -- the exact treatment, including the off-shell and
    interference effects of the full matrix element;
  * MadSpin -- long cascades, or many decay modes you want to vary without
    regenerating the production;
  * shower -- hadron and tau decays, where the correlations do not matter to
    your observable.

Whichever you pick, the width stays the trap: change a mass and forget the
width, and the effective fraction goes above 1 -- a decayed cross section
larger than the undecayed one. That is what `Auto` is for.

One last thing, about the decay-chain syntax. Add a W boson to the production,
and let the top decay through one too. Two different W bosons, and the
parentheses are what says which decay belongs to which:
%(p)s %(nested)s
""" % {'p': P, 'nested': NESTED,
       'ran': _ran(interface, 'the undecayed tops')},
     title='MadSpin, run -- and which to use',
     solution=NESTED),

Step('generate', lambda interface: """
%(counts)sRead it from the inside out. `(t > w+ b, w+ > l+ vl)` is one unit:
the `w+ > l+ vl` inside the parentheses decays the W that the top produced.
The `w+ > j j` outside them sits at the level of the production, so it decays
the W produced beside the tops. The top's W gives the lepton, the other one
the jets.

Take the parentheses away and the same symbols mean something else:

  %(flat)s

Now `w+ > l+ vl` is at the production level: it decays the W produced beside
the tops, and the W from the top decay is left undecayed. Nothing warns you --
it is a valid process, just not the one you meant. Whenever the same particle
appears at two levels of a chain, the parentheses are what says which is which.

That is the end of the decays tutorial. `tutorial syntax` has the rest of the
process-line grammar, `tutorial list` shows what else there is, and
`tutorial stop` leaves tutorial mode.
""" % {'counts': counts_line(interface), 'flat': FLAT},
     title='which W is which'),

    ],
)
