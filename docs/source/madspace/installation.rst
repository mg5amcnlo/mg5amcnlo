Installation
============

From PyPI
---------

Pre-compiled packages are available for Linux (x86_64) and MacOS X (Apple silicon),
for Python 3.11 to 3.14. The Linux wheels are built with the CUDA and HIP backends
included. Install the package with::

    pip install madspace

Building from source
--------------------

MadSpace is developed as part of `MadGraph7 <https://github.com/MadGraphTeam/MadGraph7>`_
and is built with the ``install.py`` script in its ``madspace`` directory. The script
installs the build dependencies, locates a suitable ``cmake`` (3.15 or newer) and runs
the build::

    git clone git@github.com:MadGraphTeam/MadGraph7.git
    cd MadGraph7
    python madspace/install.py --source --system

With ``--system``, MadSpace is installed into the active Python environment and can be
imported like any other package. Without arguments the script is interactive and asks
which backends to enable. The same choices are available as command line flags for
non-interactive use, for example::

    python madspace/install.py --source --system --cuda --cuda-arch "75;80;86"
    python madspace/install.py --source --system --hip --simd --debug

Run ``python madspace/install.py --help`` for the full list of options. The options of
the last build are remembered, so ``--source --system -y`` rebuilds with the same
settings. A flag given on the command line still wins over them, per option, so
``--source --system -y --cuda --cuda-arch 80`` rebuilds the remembered configuration
with CUDA added. They are stored in ``madspace/install_settings.json``; delete it to go
back to the defaults.

The build directory ``madspace/build`` is kept between builds, so you can run ``make``
there directly for faster incremental builds during development. Note that this does not
update the Python module itself, for which the install command has to be run again.

Rebuilds are therefore incremental. ``--clean`` deletes ``madspace/build`` and
``madspace/install`` first, for when that build tree is in the way -- after a toolchain
change, say, or a half-finished build -- at the price of recompiling MadSpace and its
vendored OpenBLAS from scratch::

    python madspace/install.py --source --system --clean

The remembered options are kept: they are stored outside both directories.

Installation for MadGraph7
--------------------------

Without ``--system``, the script installs MadSpace into ``madspace/install``, where
MadGraph7 picks it up without touching your Python environment::

    python madspace/install.py

This is also what the ``install madspace`` command at the MadGraph7 prompt runs, and
what happens automatically the first time you launch a run.
