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
"""The `madevent` tutorial -- the MG5-compatible output.

For everyone with existing MG5 workflows, and for the tools that have not moved
to the MG7 integrator yet.  `lo` covers the default (madspace) path; this one
covers what `output madevent` gives you instead.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line, output_name)

P = 'MG7>'
RUN = 'MY_MADEVENT_RUN'


tutorial = Tutorial(
    name='madevent',
    title='the MG5-compatible MadEvent path',
    description='output madevent: the classic directory, cards and tools',
    order='sequence',
    see_also=('lo', 'mg7', 'run', 'decays', 'syntax'),
    steps=[

Step('tutorial', """
`output` on its own now gives you the MG7 output, integrated by madspace. This
tutorial is about the other one: `output madevent`, the directory layout MG5
has always produced.

Reach for it when you have scripts, gridpacks or analysis code built around the
classic layout, or when you need one of the tools that has not moved across
yet. The physics is the same; the plumbing is not.

Start with a process:
%(p)s generate p p > t t~
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
%(counts)sNothing special so far -- the process line is the same whichever
output you ask for. The difference is in the next command, where you name the
format:

%(p)s output madevent %(run)s
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface)},
     title='generate a process',
     hint="Name the format explicitly: `output madevent DIRNAME`.",
     solution='output madevent %s' % RUN),

Step('output', lambda interface: """
That wrote `%(run)s`. This is the layout a lot of existing code expects:

  Cards/run_card.dat     beams, cuts, scales, PDF, number of events
  Cards/param_card.dat   masses, widths, couplings
  Cards/*_card.dat       MadSpin, reweighting, Pythia8, Delphes, systematics
  SubProcesses/          the generated matrix elements, one directory per
                         subprocess group
  bin/generate_events    the runner

Two differences from the MG7 output worth knowing. The cards are `.dat` in the
old key/value format rather than one `run_card.toml`; and each tool that plugs
in brings its own card, which is why there are so many of them. A card only
takes effect once you rename `X_card_default.dat` to `X_card.dat`, or answer
yes when the run offers to edit it.

`launch` will ask which cards you want to edit and then run
`bin/generate_events` for you -- the same thing you would get by running that
script yourself from inside the directory. For a first run, change nothing,
and Ctrl-C stops a long run and returns you here.

%(p)s launch %(run)s
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='produce the output',
     solution=lambda interface: 'launch %s' % output_name(interface, RUN)),

Step('launch', lambda interface: """
Same physics, familiar output: a cross section, and an LHE file under
`%(run)s/Events/`.

The run card is where most of your time will go. The settings people change
first:

  nevents            how many events to produce
  ebeam1, ebeam2     beam energies
  lpp1, lpp2         beam type (1 proton, -1 antiproton, 0 no PDF, ...)
  pdlabel            the PDF set
  ptj, etaj, drjj    the jet cuts
  fixed_ren_scale,   scale choices, and the dynamical scale otherwise
  fixed_fac_scale
  nhel               0 sums the helicities, 1 samples them. A speed/variance
                     choice; polarisation does not need either value
  me_frame           for a polarised process: the legs whose sum defines the
                     rest frame the polarisation is measured in

`me_frame` is the one to read twice, because a polarisation only means
something once you say in which frame. It indexes the NORMALISED leg order,
not the order you wrote in the process line: for
`p p > w+ z j j, w+ > l+ vl, z > l+ l-` the WZ rest frame is
`me_frame = [3,4,5,6]`. Get it wrong and you still get a polarisation, just
not the one you asked for. (Nothing has to be done at generation time --
`set group_subprocesses False` is not needed for polarised processes.)

Now something that matters more than it looks. A cross section is only as
good as the widths in your param card, and the widths are not automatically
kept in step with the masses. Compute one and see:

%(p)s compute_widths t --body_decay=2 --output=./my_widths.dat
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='run it',
     hint="`compute_widths PARTICLE --output=FILE` writes a param card with the widths filled in.",
     solution='compute_widths t --body_decay=2 --output=./my_widths.dat'),

Step('compute_widths', """
That wrote a param card with the top width computed from the model rather than
copied from the card. Note the `--output=`: without it, `compute_widths`
overwrites the param card *in the model directory*, which is rarely what you
want.

Why this matters for MadEvent in particular: it is the output that makes
parameter scans easy, and scans are where widths go wrong.

Put this in `Cards/param_card.dat` in place of the top mass value:

  scan:[150, 160, 170, 180, 190]

and the run repeats for each point, collecting the cross sections into a
summary table. Now do the same with a decay chain like
`p p > t t~, t > w+ b` and watch the ratio to the undecayed cross section.
It plays the role of a branching fraction, so it ought to stay under 1 -- and
it does not, because changing the mass did not change the width. The partial width the decay chain
computes grows with the mass while the total width in the card stays frozen.

The fix is to hand the width back to the model:

  DECAY 6 Auto

The scan machinery understands `Auto`, recomputes it at every point, and
records the result as a `width#6` column in the summary, so you can watch the
width follow the mass and the ratio drop back under 1.

The rule generalises: any scan that moves a mass must recompute every width
that mass feeds. Nothing warns you.

%(p)s history my_madevent_run.dat
""" % {'p': P},
     title='widths and parameter scans',
     hint="`history FILE` writes down the session so far.",
     solution='history my_madevent_run.dat'),

Step('history', lambda interface: """
That file replays the session -- `import command my_madevent_run.dat`, or
`./bin/madgraph my_madevent_run.dat` from a shell.

What else lives on this path:

  * **The tools.** MadSpin (decays with spin correlations), `systematics`
    (scale and PDF variations), `reweight` (new parameters without
    regenerating), Pythia8 and Delphes. Each has a card in `Cards/`, and the
    run offers to edit it.
  * **Gridpacks.** Set `gridpack = True` in the run card to get a
    self-contained tarball that generates events anywhere without MG5.
  * **Clusters.** `set run_mode 1` for a cluster and `set cluster_type` for
    the scheduler; `set run_mode 2` for local multicore.
  * **The HTML summary.** `%(run)s/crossx.html` collects every run in the
    directory, with the cross sections, the cards used and the plots.

MG7 or MadEvent? MG7 is the default and where the integrator work is going --
madspace phase space, MadNIS, the C++ and GPU backends. MadEvent is where the
older tooling still lives. The process line is identical, so moving a study
across is a matter of changing one word in the `output` command and
translating the run card.

%(see_also)s

Leave with `tutorial stop`.
""" % {'run': output_name(interface, RUN),
       'see_also': tutorials.where_next()},
     title='tools, gridpacks and clusters'),

    ],
)
