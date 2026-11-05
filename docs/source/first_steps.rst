First steps
===========

To start your first process, enter the following commands at the MadGraph7 prompt::

    generate p p > t t~
    output my_first_run
    launch

The ``launch`` command asks which programs to run and lets you edit the cards. Press Enter to
accept the defaults. The events are written to ``my_first_run/Events``.

Interactive tutorial
--------------------

Type ``tutorial`` at the prompt for a guided walkthrough that runs inside the interpreter. It
shows a menu of lessons. You pick one by name, for example::

    tutorial lo
    tutorial mg7

The first lessons are a good start:

``lo``
    From a cold start to a first event sample.

``syntax``
    The process syntax: orders, interference, decay chains and polarization.

``mg7``
    The run card, the phase-space options and training MadNIS.

Other lessons cover models, decays, standalone matrix elements, NLO and MadLoop, the
``check`` commands and the MadEvent mode. The ``exercises`` lesson checks your answers
and tells you which mistake you made. Leave a lesson with ``tutorial stop``.

Where to go next
----------------

:doc:`run_card`
    Every setting of the run card ``Cards/run_card.toml``, section by section.

:doc:`gridpacks`
    Save a finished run and generate more events from it without repeating the survey and
    the training.

:doc:`madevent_to_mg7`
    The MadEvent settings and their MG7 equivalents, if you are used to ``run_card.dat``.

:doc:`installation`
    Requirements and how to install MadGraph7 and ``madspace``.

:doc:`madspace/examples`
    Use the phase-space and integration library from Python.

Graphical interface
-------------------

`MadBoard <https://github.com/MadGraphTeam/MadBoard>`_ is a separate package providing a
modern web interface to MadGraph, for setting up processes, following runs from the
browser and inspecting their results. You can install it with::

    pip install madboard

and launch it by running the ``madboard`` command within your MadGraph7 directory.
