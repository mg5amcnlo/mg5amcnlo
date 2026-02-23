# MadGraph5_aMC@NLO
MadGraph5_aMC@NLO is a framework that aims at providing all the elements necessary for SM and BSM phenomenology, such as the computations of cross sections, the generation of hard events and their matching with event generators, and the use of a variety of tools relevant to event manipulation and analysis.
Processes can be simulated to LO accuracy for any user-defined Lagrangian, an the NLO accuracy in the case of models that support this kind of calculations -- prominent among these are QCD and EW corrections to SM processes.
Matrix elements at the tree- and one-loop-level can also be obtained.

MadGraph5_aMC@NLO is the new version of both MadGraph5 and aMC@NLO that unifies the LO and NLO lines of development of automated tools within the MadGraph family.
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
MadGraph5_aMC@NLO requires:
 - Python 3.7 (or higher)
 - gfortran/gcc 4.6 (or higher)


### Guide
1. Download the desired release from [Launchpad](http://launchpad.net/madgraph5).
2. Unpack the tar.gz by running `tar -xzpvf [TARBALL]`
3. [optional] Add the bin directory to your `PATH`, for example if the unpacked tar is in `~/MG5_aMC`, this could be achieved by including `export PATH="$PATH:$HOME/MG5_aMC/bin" in `~/.bashrc`
4. Run `mg5_aMC`
5. Type:
    - `help` for list of commands
    - `help [command]` for help on individual commands
    - `tutorial` for an interactive quick-start tutorial for LO computations
    - `tutorial aMCatNLO` for a tutorial on aMC@NLO runs for NLO computations

### scripting MG5aMC

To avoid to use the interactive interface you can enter all your command in a file. For example:
```generate p p > t t~
output
launch
shower=Pythia8
set mt scan:[150+i for i in range(21)]
set wt auto
```

and run it via 
`mg5_aMC PATH_TO_FILE`

### Lectures and Tutorial
The list of all school and tutorial are available [here](https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/MGTutorial)
The latest school about madgraph (Iwate-japan 2024) contains lecture on how MG5aMC is working and various tutorial: [slide of the lectures/tutorial](https://ics.sgk.iwate-u.ac.jp/index.php/program-2024/)

### Help and Support

FAQ are available on [launchpad](https://answers.launchpad.net/mg5amcnlo/+faqs)
Physics Question are possible (no attachment) are possible trough [launchpad](https://answers.launchpad.net/mg5amcnlo) or github issue.
Bug report/Question with attachment are posible trough [launchpad](https://bugs.launchpad.net/mg5amcnlo) or github issue/PR.


## LICENSE
MadGraph5_aMC@NLO is licensed under an [adapted University of Illinois/NCSA license](madgraph/LICENSE).

## Github Contribution
Contributors are encouraged to implement test associated to their new feature and to allow to run the CI/CD in their repo to check that their commit does not break the tests implemented before them.
PR can target the following branch: 
  - bug fixing should target the latest LTS release (which contains the bug fix)
    - so either 2.9.(X+1) (up to december 2025) or 3.5.(X+1)  
  - small improvment of the code should target 3.X.(Y+1) when the latest release is 3.X.Y
  - large implementation should target 3.(X+1).0 when the latest release is 3.X.Y


