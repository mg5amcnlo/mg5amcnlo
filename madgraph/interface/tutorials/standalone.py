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
"""The `standalone` tutorial -- the matrix element as a callable.

The longer reference for the flavour-aware API and the f2py linking is
docs/standalone_flavor_python.md.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line, output_name)

P = 'MG7>'
RUN = 'MY_SA_RUN'
CPP = 'MY_SA_CPP'


LAUNCH_QUESTION_HINT = """
The first box is the build: `backend=scalar`, `nb_core=4`, or the number of a
line to step through its values. The second is the param_card, edited with
`set`, as for any other output.

**For this tutorial the defaults are fine -- just press Enter.**
"""


tutorial = Tutorial(
    name='standalone',
    title='standalone matrix elements',
    description='get |M|^2 as a callable, and link it into Python or C++',
    order='sequence',
    see_also=('syntax', 'checks', 'lo'),
    steps=[

Step('tutorial', """
Sometimes you do not want events. You want |M|^2 at a phase-space point you
choose, from your own code -- to reweight someone else's events, to feed a
fitter or a neural network, to compare two calculations point by point, or to
check MG7 against a number you computed by hand.

That is what the standalone outputs are for: no integrator, no run card,
just the matrix element as a function you call.

We will use a merged process, because it shows the one thing about the
standalone API that is not obvious:

%(p)s generate p p > j j QCD=2 QED=0
""" % {'p': P},
     title='welcome',
     solution='generate p p > j j QCD=2 QED=0'),

Step('generate', lambda interface: """
%(counts)sNo events anywhere in sight, and none coming: this path stops at the
matrix element.

Now the output. The formats, and what each is for:

  standalone_fortran   the Fortran standalone -- this tutorial's main path
  standalone           the MadMatrix C++ / CUDA standalone
  matrix               just the matrix-element files, nothing around them
  standalone_msP,      variants for specific external tools
  standalone_msF,
  standalone_rw

(`output standalone_cpp` was removed; `standalone` is the C++ one now.)

`--prefix` is the flag that matters if you ever load two processes into one
Python session. It prefixes the routine names -- `--prefix=int` gives `M1_`,
`M2_`, ... per subprocess group, `--prefix=proc` uses the process name -- so
the symbols and COMMON blocks of two modules cannot collide. With no prefix
you get bare `SMATRIX`, which is fine for exactly one module and a trap for
two.

%(p)s output standalone_fortran %(run)s --prefix=int
""" % {'p': P, 'run': RUN, 'counts': counts_line(interface)},
     title='generate a process',
     hint="`output standalone_fortran DIR --prefix=int`",
     solution='output standalone_fortran %s --prefix=int' % RUN),

Step('output', lambda interface: """
Look at `%(run)s/SubProcesses/`: one directory per subprocess group, each with

  matrix.f                the matrix element
  check_sa.f              a ready-made driver that evaluates one point
  f2py_matrix_wrapper.f   the Python-facing entry points
  flavor_dispatch.py      a helper that hides their naming
  makefile

and alongside them `all_matrix.f` and `f2py_wrapper.f`, which wrap *every*
group behind one interface.

Now the thing that is not obvious. Open `P1_gg_QQx/matrix.f` and you will find

  PARAMETER (NFLAV=4)

One matrix element, four physical flavour combinations. That is flavour
merging: `p p > j j` does not generate a separate ME for uu~ > dd~, cc~ > ss~
and so on where the structure is identical. So every standalone entry point
takes a **flavour selector**, in either of two equivalent forms:

  * an **index** in [1, NFLAV] -- the column in the allowed-flavour table.
    Index 1 is always valid; an index that is out of range or not allowed
    returns 0.
  * an **array** of length NEXTERNAL giving, per leg, the 1-based position of
    the actual particle inside its merged group (1 for an unmerged leg). It is
    resolved to an index internally.

A process with no merged particles is just the single-flavour case, index 1.
Forgetting the selector is the usual first mistake, and it does not crash --
it quietly gives you one flavour when you wanted another.

For contrast, produce the C++ one too:
%(p)s output standalone %(cpp)s
""" % {'p': P, 'run': output_name(interface, RUN), 'cpp': CPP},
     title='what the output contains',
     setup=lambda interface: setattr(interface, '_tutorial_sa_fortran_dir',
                                     output_name(interface, RUN)),
     hint="`output standalone DIR` gives the C++/CUDA standalone.",
     solution='output standalone %s' % CPP),

Step('output', lambda interface: """
That is a different animal: `SubProcesses/` now holds `check_sa.cc`,
`color_sum.cc`, `GpuAbstraction.h` and friends. Same matrix element, C++ and
CUDA, with the vectorised and GPU paths the Fortran one does not have.

It also has a `launch`, which asks one question before it builds anything --
how to build, and which parameters to use -- then compiles it and evaluates
one point:
%(p)s launch %(cpp)s
""" % {'p': P, 'cpp': output_name(interface, CPP)},
     title='the C++ standalone',
     hint="`launch DIR` builds the C++ standalone and runs `check_sa.exe`.",
     question_hint=LAUNCH_QUESTION_HINT,
     solution=lambda interface: 'launch %s' % output_name(interface, CPP)),

Step('launch', """
That compiled `check_sa.exe` in each `P*` directory and ran it once: a fixed
phase-space point, and |M|^2 there for every flavour combination the group
serves -- the NFLAV of the Fortran output again. The executable stays behind,
so the next launch with the same backend does not recompile.

The switches you saw are the build:

  backend      the SIMD width on CPU (`auto` picks the widest your machine
               has), or `cuda` / `hip` when their compiler is installed
  subprocess   one `P*` directory, or all of them
  nb_core      parallel make jobs

and the card box is the same param_card editing as everywhere else --
`set mt 172.5`, `set decay 23 auto`, or a path to a card you already have.
The C++ reads `Cards/param_card.dat` at run time, so a new card needs no
recompilation.

Which output to reach for:
  * **standalone_fortran** to link into Fortran or to call from Python via
    f2py. Simplest, and the reference the others are checked against.
  * **standalone** (C++/CUDA) when you need throughput -- many points, a GPU,
    or a C++ host program.
  * **`output mg7`** if what you actually wanted was events after all.

%(p)s history my_standalone_session.dat
""" % {'p': P},
     title='build and run it',
     solution='history my_standalone_session.dat'),

Step('history', lambda interface: (lambda FORTRAN_RUN: """
That file replays the session -- `import command my_standalone_session.dat`,
or `./bin/madgraph my_standalone_session.dat` from a shell.

The rest happens outside MG7, in the Fortran output directory.

**Evaluate one point with no Python at all.** `check_sa` is built for you;
run it from a subprocess directory and it prints |M|^2 for a phase-space point.
To pin the point rather than take a random one, set `MG_MOMFILE` to a file of
momenta -- that is how you compare two backends digit by digit.

**Build the Python module.** Two routes:

  cd %(run)s/SubProcesses && make f2py
      builds `all_matrix2py`, one module covering every subprocess group. Its
      entry points select the process by PDG codes:
      `SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2, NHEL, ANS)`, plus
      `INITIALISE(path_to_param_card)`, `GET_PDG_ORDER`, and `CHANGE_PARA` /
      `UPDATE_ALL_COUP` to move a parameter without regenerating.

  cd %(run)s/SubProcesses/P1_gg_QQx && make matrix2py.so
      builds `matrix2py` for that one group. Its entries come in pairs --
      `PY_M1_SMATRIX` taking the flavour array and `PY_M1_SMATRIX_IDX` taking
      the index -- and `flavor_dispatch.FlavorDispatch` hides both the pairing
      and the prefix:

        from flavor_dispatch import FlavorDispatch
        me = FlavorDispatch(matrix2py)
        me.initialisemodel('../../Cards/param_card.dat')
        val = me.get_value(P, alphas, nhel, 2)          # flavour by index
        val = me.get_value(P, alphas, nhel, [1,1,1,1])  # or by array

Either way f2py (from numpy) must be on PATH, and the module links against a
shared library sitting next to it -- so run Python from that directory, or put
it on `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on macOS).

**The full reference**, including the flavour tables and every call signature,
is `docs/standalone_flavor_python.md` in the MG7 source tree.

%(see_also)s

Leave with `tutorial stop`.
""" % {'run': FORTRAN_RUN,
       'see_also': tutorials.where_next()})(
         getattr(interface, '_tutorial_sa_fortran_dir', RUN)),
     title='build it and call it'),

    ],
)
