################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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


tutorial_aMCatNLO = """
You have entered tutorial mode. This will introduce you to the main
syntax options of MadGraph5_aMC@NLO for the computation of processes involving
NLO corrections.

To learn more about the different options for a command, you can use
MG_aMC>help A_CMD
To see a list of all commands, use
MG_aMC>help 

The goal of this tutorial is to learn how to generate a process and to
produce an output for MadGraph5_aMC@NLO. In this part we will learn
a) How to generate a process involving NLO corrections
b) How to create the corresponding output for MadGraph5_aMC@NLO
c) How to run this output for computing NLO corrections

If you have FastJet (v3 or later) installed on your computer and you wish
to link it, please update the mg5_configuration file or type
MG5_aMC>set fastjet /path/to/fastjet-config
Otherwise the basic fastjet functionalities included in FJcore (shipped
with MadGraph5_aMC@NLO) will be used.


Let's start with the first point, how to generate a process at NLO:
MG5_aMC>generate p p > e+ ve [QCD]
Note that a space is mandatory between the particle names and that '[QCD]' 
specifies that you want to consider QCD NLO corrections. 
Couplings different than QCD cannot be perturbed yet.
"""

tutorial = tutorial_aMCatNLO

generate = """
You have just generated a new process.
You can find more information on supported syntax by using:
MG_aMC>help generate
To list all defined processes, type
MG_aMC>display processes

If you want to know more about particles and multiparticles present,
write
MG_aMC>display particles
MG_aMC>display multiparticles

If you want to add a second process, use the add process command:
MG_aMC>add process p p > e+ e- [QCD] @2

At this stage you can export your processes.
This is done simply by typing:
MG_aMC>output MY_FIRST_AMCATNLO_RUN
"""

display_processes = """
You have seen a list of the already defined processes.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create a valid output for
MadGraph5_aMC@NLO. This is done simply by typing:
MG_aMC>output MY_FIRST_AMCTANLO_RUN
"""

add_process = """
You have added a process to your process list.

At this stage you can export your processes.
This is done simply by typing:
MG_aMC>output MY_FIRST_AMCATNLO_RUN
"""
output = """
If you are following the tutorial, a directory MY_FIRST_AMCATNLO_RUN has
been created which can be used in order to run simulation including NLO corrections.

Additionally to the commands in the bin directory (see 
MY_FIRST_AMCATNLO_RUN/README), you can also generate your events/compute the 
cross-section from this interface. 
You will generate events to be showered a la MC@NLO, compute the theoretical
and PDF error on the fly (if asked for in the run_card.dat) and shower the 
events with the parton_shower MonteCarlo specified in the run_card.dat, 
generating a file in the StdHEP format. 
Please note that, since shower-specific counterterms have to be included in the
calculation, the parton level sample you will obtain can only be showered
with the selected MonteCarlo. 
Note also that, because of the way they have been generated, the parton-level
events in the .lhe file are UNPHYSICAL. 
In order to obtain physical results, please use the .hep file

Please enter
MG_aMC> launch 

If you just want to generate the parton level .lhe file, please enter
MG_aMC> launch -p

(you can interrupt the computation to continue the tutorial by pressing Ctrl-C)

At any time, you can access more commands/options for running the output by 
switching to an interactive interface for a given output folder 'MyFolder'.
You can do so by typing:
MG_aMC> launch -i MyFolder

Please see MY_FIRST_AMCATNLO_RUN/README to know about the available commands.
To know the possible options/modes for each command, simply tiple
'MyFolder'> help COMMAND
from the interface bound to the 'MyFolder' output.

"""

open_index = output

launch = """This step ends the tutorial of the basic commands of for running
processes including NLO corrections.
You can always use the help to see the options available for different
commands. For example, if you want to know how to launch on multicore/cluster
just type
MG_aMC>help launch

To learn more about MadLoop StandAlone checks and runs, you can now follow
its tutorial with:
MG_aMC>tutorial MadLoop

To simply close this tutorial, enter
MG_aMC>tutorial stop
If you want to exit MG5, enter
MG_aMC>exit
"""
