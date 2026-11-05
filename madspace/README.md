<p align="center">
  <img src="https://raw.githubusercontent.com/MadGraphTeam/MadGraph7/refs/heads/main/docs/source/_static/logo-light-madspace.png" width="500", alt="MadSpace">
</p>

<h3 align="center">Modular and GPU-ready phase-space library</h3>

<p align="center">
<a href="https://arxiv.org/abs/2602.06895"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2602.06895-b31b1b.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

### Installation

#### From PyPI

Pre-compiled packages are available for Linux (x86_64) and MacOS X (Apple silicon),
for Python 3.11 to 3.14. The Linux wheels are built with the CUDA and HIP backends
included.

```sh
pip install madspace
```

#### Building from source

MadSpace is developed as part of [MadGraph7](https://github.com/MadGraphTeam/MadGraph7)
and is built with the `install.py` script in its `madspace` directory. The script
installs the build dependencies, locates a suitable `cmake` (3.15 or newer) and runs the
build:

```sh
git clone git@github.com:MadGraphTeam/MadGraph7.git
cd MadGraph7
python madspace/install.py --source --system
```

With `--system`, MadSpace is installed into the active Python environment and can be
imported like any other package. Without arguments the script is interactive and asks
which backends to enable. The same choices are available as command line flags for
non-interactive use, for example

```sh
python madspace/install.py --source --system --cuda --cuda-arch "75;80;86"
python madspace/install.py --source --system --hip --simd --debug
```

Run `python madspace/install.py --help` for the full list of options. The options of the
last build are remembered, so `--source --system -y` rebuilds with the same settings.

The build directory `madspace/build` is kept between builds, so you can run `make` there
directly for faster incremental builds during development. Note that this does not update
the Python module itself, for which the install command has to be run again.

#### Installation for MadGraph7

Without `--system`, the script installs MadSpace into `madspace/install`, where MadGraph7
picks it up without touching your Python environment:

```sh
python madspace/install.py
```

This is also what the `install madspace` command at the MadGraph7 prompt runs, and what
happens automatically the first time you launch a run.

### Tests

To run the tests, you need to have the `pytest`, `numpy` and `torch` packages installed.
One test optionally requires the `lhapdf` package (can be installed via conda or built from
source) and the `NNPDF40_nlo_as_01180` PDF set.

To run the tests, go to the root directory of the repository and run
```sh
pytest madspace/tests
```

### Citation

If you use this MadSpace or parts of it, please cite:

    @article{Heimel:2026hgp,
    author = "Heimel, Theo and Mattelaer, Olivier and Winterhalder, Ramon",
    title = "{MadSpace -- Event Generation for the Era of GPUs and ML}",
    eprint = "2602.06895",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "MCNET-26-01, IRMP-CP3-26-04, TIF-UNIMI-2026-1",
    month = "2",
    year = "2026"}
