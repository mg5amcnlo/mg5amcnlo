# MadGraph5_aMC@NLO
MadGraph5_aMC@NLO is a framework that aims at providing all the elements necessary for SM and BSM phenomenology, such as the computations of cross sections, the generation of hard events and their matching with event generators, and the use of a variety of tools relevant to event manipulation and analysis.
Processes can be simulated to LO accuracy for any user-defined Lagrangian, an the NLO accuracy in the case of models that support this kind of calculations -- prominent among these are QCD and EW corrections to SM processes.
Matrix elements at the tree- and one-loop-level can also be obtained.

MadGraph5_aMC@NLO is the new version of both MadGraph5 and aMC@NLO that unifies the LO and NLO lines of development of automated tools within the MadGraph family.
It therefore supersedes all the MadGraph5 1.5.x versions and all the beta versions of aMC@NLO.
As such, the code allows one to simulate processes in virtually all configurations of interest, in particular for hadronic and e+e- colliders; starting from version 3.2.0, the latter include Initial State Radiation and beamstrahlung effects.

## Citing
The standard reference for the use of the code is:
> J. Alwall et al, "The automated computation of tree-level and next-to-leading order differential cross sections, and their matching to parton shower simulations", arXiv:1405.0301 [hep-ph].

In addition to that, computations in mixed-coupling expansions and/or of NLO corrections in theories other than QCD (eg NLO EW) require the citation of:
> R. Frederix et al, "The automation of next-to-leading order electroweak calculations", arXiv:1804.10017 [hep-ph].

A more complete list of references can be found [here](http://amcatnlo.web.cern.ch/amcatnlo/list_refs.htm).

## Getting Started
### Releases
There are two supported versions of MadGraph5_aMC@NLO:
- The latest release contains the latest updates in both physics and performance
- The long term stable (LTS) release lacks the latest features but is regularly updated with all the bug fixes

### Requirements
MadGraph5_aMC@NLO runs on:
- Linux with:
    - Python 2.7 or
    - Python 3.7 (or higher)
    - gfortran/gcc 4.6 (or higher)
- macOS with:
    - [gfortran](http://hpc.sourceforge.net/)
    - xcode developers tool (`xcode-select --install`)

Windows users are recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

### Guide
1. Download the desired release from [Launchpad](http://launchpad.net/madgraph5).
2. Unpack the tar.gz by running `tar -xzpvf [TARBALL]`
3. Add the bin directory to your `PATH`, for example if the unpacked tar is in `~/MG5_aMC`, this could be achieved by including `export PATH="PATH:$HOME/MG5_aMC"` in `~/.bashrc`
4. Run `mg5_aMC`
5. Type:
    - `help` for list of commands
    - `help [command]` for help on individual commands
    - `tutorial` for an interactive quick-start tutorial for LO computations
    - `tutorial aMCatNLO` for a tutorial on aMC@NLO runs for NLO computations
    - `tutorial MadLoop` for a tutorial on MadLoop for learning how to output standalone codes for loop process evaluation for given PS points

## Contribute
MadGraph5_aMC@NLO is licensed under an [adapted University of Illinois/NCSA license](madgraph/LICENSE).

Contributors are encouraged to read the [contribution guidelines](CONTRIBUTING.md)
