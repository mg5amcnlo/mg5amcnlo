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
"""The `mg7` tutorial -- tuning the integrator.

Starts where `lo` ended: a default madspace run exists, and this is about
making it faster or more accurate, up to and including training MadNIS.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line,
                                                  output_name)

P = 'MG7>'
RUN = 'MY_MG7_RUN'


tutorial = Tutorial(
    name='mg7',
    title='tuning the MG7 integrator',
    description='madspace phase space, VEGAS, and training MadNIS',
    order='sequence',
    see_also=('lo', 'madevent', 'run'),
    steps=[

Step('tutorial', """
This one assumes you have already run something -- `tutorial lo` if not. It is
about the settings that decide how long a run takes and how good the answer is.

The pieces:
  * **madspace** builds the phase space and evaluates the matrix elements. It
    is what `output` produces by default.
  * **VEGAS** is the default integrator: adaptive binning, no training, works
    everywhere.
  * **MadNIS** is the alternative: a normalizing flow trained on your process.
    It costs training time up front and pays it back in unweighting
    efficiency. It is not something you switch on -- `[madnis] enable` is
    `"auto"` by default, and MG7 decides per subprocess after a probe run.

Everything below lives in `Cards/run_card.toml`. Make an output to look at:

%(p)s generate p p > t t~ j
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~ j'),

Step('generate', lambda interface: """
%(counts)sEnough channels for the integrator to have decisions to make,
which is what makes this worth tuning -- `p p > t t~` would not be.

%(p)s output %(run)s
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface)},
     title='pick a process worth tuning',
     solution='output %s' % RUN),

Step('output', lambda interface: """
That is an MG7 output, integrated by madspace: `%(run)s/` holds the process
code, and everything below is one file inside it.

`Cards/run_card.toml`, section by section. The ones you will actually touch
are marked.

  [run]             seed, device (cpu/gpu), cpu_mode, madspace_cpu_mode,
                    the thread pools, output_format, verbosity          <-
  [beam]            beam energies, PDF
  [generation]      events, the survey iterations and target precision,
                    batch sizes, systematics                            <-
  [cuts]            the cuts
  [phasespace]      how phase space is built                            <-
  [vegas]           the default integrator                              <-
  [madnis]          the trainable one                                   <-
  [multiparticles]  labels, as in the process line
  [histograms]      what to plot
  [postprocessing]  systematics after the fact
  [gridpack]        self-contained event generation

Start with the phase space, because a badly channelled integration cannot be
rescued by a better integrator:

  mode                  multichannel (default), flat, or both
  sde_strategy          diagrams or denominators -- how channels map to
                        diagrams
  t_channel, flat_mode  propagator, rambo or chili for the t-channel
  bw_cutoff             how far off-shell a Breit-Wigner is followed
  drop_qcd_s_channel    drop channels with no QCD resonance once the channel
                        count would exceed this -- the lever for processes
                        with too many channels
  combine_channel_threshold, adaptive_symmetry_sampling

The symptom of bad channelling is a survey that will not converge and an
unweighting efficiency in the per-mille range, whatever you do to [vegas].

Then [vegas] itself: `bins`, `damping`, `start_batch_size`, `max_batch_size`,
`optimization_patience` and `optimization_threshold`. Raising the batch sizes
buys a better grid at a proportional cost; `damping` is what to reach for when
the grid oscillates instead of settling.

%(p)s history my_mg7_session.dat
""" % {'p': P, 'run': output_name(interface, RUN)},
     title='the run card, section by section',
     solution='history my_mg7_session.dat'),

Step('history', lambda interface: """
That file replays the session -- `import command my_mg7_session.dat`, or
`./bin/madgraph my_mg7_session.dat` from a shell.

**MadNIS, and why you probably do not need to configure it.**

`[madnis] enable` is `"auto"`, and it means what it says. MG7 surveys the phase
space first, measures how badly the integrand varies (a relative standard
deviation across the channels), and then decides *per subprocess*:

  * two outgoing particles -- never. There is nothing for a flow to learn that
    the multichannel phase space has not already done. This is why
    `p p > t t~` trains nothing.
  * three -- only if the survey came out noisy, or you asked for more than a
    million events, or you are building a gridpack.
  * four or more -- always.

Auto mode also sizes the networks and picks the learning rate from that same
measurement, so the numbers printed in `Cards/run_card.toml` are starting
points, not the values that will be used.

There is no separate training command: training is a phase inside
`bin/generate_events`, which is what `launch` runs. It uses the GPU if
`[run] device` says so and torch can see one, and the CPU otherwise.

**When you do want to override it**, set `enable = true` or `false` outright,
and then touch things in this order:

  1. `train_batches`, `batch_size_per_channel` -- how much training, and how
     much per channel. Everything else is secondary.
  2. `lr`, `lr_scheduler` (none or cosine). If the loss is noisy,
     lower the rate before touching the network.
  3. only then the network shape: `flow_hidden_dim`, `flow_layers`,
     `flow_spline_bins`, `flow_activation`; `discrete_*` for the discrete
     dimensions; `cwnet_*` for the channel-weight network.

Setting any of these by hand overrides what auto would have chosen for it.

**Judge it honestly.** Compare three runs of the same process -- forced off,
forced on, and MadEvent -- on cross section (they must agree inside the quoted
errors), unweighting efficiency, and total wall time *including* training. A
flow that wins on efficiency and loses on wall time has not helped you, and
that is the trade auto mode is trying to call for you.

**One rough edge**, so you meet it here rather than mid-study: check which PDF
you are getting; the default is not the one a MadEvent run would have used.

Seeding works the way you would expect -- `set iseed 42` at the launch question
sets `[run] seed`, and so does editing the run card directly.

**Coming from a LO run card?** The "Coming from MadEvent" page of the
documentation maps the old names onto the new sections.

**Gridpacks** work here too, via the `[gridpack]` section. The "Gridpacks"
page of the documentation describes them, including `bin/event_stream.py`, a
Python API that streams events in memory.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='training MadNIS'),

    ],
)
