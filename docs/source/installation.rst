Installation
============

Download the latest release tarball from the
`releases page <https://github.com/MadGraphTeam/MadGraph7/releases>`_, unpack it and
start the interface::

    tar xzf MG7_v0.2.0.tar.gz
    cd MG7_v0_2_0
    ./bin/madgraph

Python 3.12 or higher and a C++ compiler are required. The ``madspace`` library is
installed automatically the first time you launch a run. To build it yourself instead,
run ``install madspace`` from the MadGraph7 prompt, or see the
:doc:`MadSpace installation instructions <madspace/installation>`.

The ``pyamplicol`` package, used by the ``[amplicol]`` matrix-element backend, is
installed in the same way with ``install pyamplicol``; see the
:doc:`pyAmpliCol installation instructions <pyamplicol/user/installation>`.
