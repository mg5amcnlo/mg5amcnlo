MadGraph7 documentation
=======================

MadGraph7 is the next major release of the MadGraph event generator. Its new features
include vectorization and GPU support for matrix element evaluation and phase-space
sampling, machine-learning-accelerated phase-space integration and sampling through
MadNIS, and more accurate handling of spin correlations in MadSpin.

Alpha release
-------------

This is an **alpha release**, meant for testing and feedback, **not for production**.
The MadGraph7 workflow currently covers leading-order event generation. The LO workflow
from MG5_aMC@NLO is still reachable through the ``output madevent`` mode. The NLO
workflow is also available (with only minor updates). AmpliCol mode is not included in
this release.

If you need a stable release, use the
`MadGraph5_aMC@NLO repository <https://github.com/mg5amcnlo/mg5amcnlo>`_ or the
`Launchpad <http://launchpad.net/madgraph5>`_.

.. toctree::
   :maxdepth: 1
   :caption: Usage
   :hidden:

   installation
   first_steps
   run_card
   custom_functions
   diagram_filters
   gridpacks
   madevent_to_mg7

.. toctree::
   :maxdepth: 1
   :caption: MadSpace
   :hidden:

   madspace/installation
   madspace/examples
   madspace/umami-api
   madspace/python-api
   madspace/cpp-api

.. toctree::
   :maxdepth: 1
   :caption: pyAmpliCol
   :hidden:

   pyamplicol/README
   pyamplicol/user/installation
   pyamplicol/user/gentle-walkthrough
   pyamplicol/user/configuration
   pyamplicol/user/python-api
   pyamplicol/user/profiling-and-benchmarking
   pyamplicol/user/architecture-overview
   pyamplicol/development/README
