################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################


tutorial_aMCatNLO = """
You have entered tutorial mode. This will introduce you to the main
syntax options of aMC@NLO. Most of the syntax is identical to the MadGraph one 

As in MadGraph, to learn more about the different options for a command, 
you can use
aMC@NLO>help A_CMD
To see a list of all commands, use
aMC@NLO>help 

The goal of this tutorial is to learn how to generate a process and to
produce an output for aMC@NLO. In this part we will learn
a) How to generate a process
b) How to create output for aMC@NLO
c) How to run the aMC@NLO output

IMPORTANT NOTE: fastjet v3+ is needed for aMC@NLO to run.
Please update the mg5_configuration file or type
mg5>set fastjet /path/to/fastjet-config

Let's start with the first point, how to generate a process at NLO:
mg5>generate p p > e+ ve [QCD]
Note that a space is mandatory between the particle names and that '[QCD]' 
specifies that you want to consider QCD NLO corrections. 
Couplings different than QCD cannot be perturbed yet.
"""

tutorial = tutorial_aMCatNLO

generate = """
You have just generated a new process.
You can find more information on supported syntax by using:
aMC@NLO>help generate
To list all defined processes, type
aMC@NLO>display processes

If you want to know more about particles and multiparticles present,
write
aMC@NLO>display particles
aMC@NLO>display multiparticles

If you want to add a second process, use the add process command:
aMC@NLO>add process p p > e+ e- [QCD] @2

At this stage you can export your processes.
This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCATNLO_RUN
"""

display_processes = """
You have seen a list of the already defined processes.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create a valid output for
aMC@NLO. This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCTANLO_RUN
"""

add_process = """
You have added a process to your process list.

At this stage you can export your processes.
This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCATNLO_RUN
"""
output = """
If you are following the tutorial, a directory MY_FIRST_AMCATNLO_RUN has
been created which can be used in order to run aMC@NLO.

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
aMC@NLO> launch 

If you just want to generate the parton level .lhe file, please enter
aMC@NLO> launch -p

(you can interrupt the computation to continue the tutorial by pressing Ctrl-C)

At any time, you can access more commands/options for running aMC@NLO by switching to an interactive interface aMC@NLO_run for a given output folder 'MyFolder'.
You can do so by typing:
aMC@NLO> launch -i MyFolder

Please see MY_FIRST_AMCATNLO_RUN/README to know about the available commands.
To know the possible options/modes for each command, simply tipe
aMC@NLO_run> help COMMAND
from the aMC@NLO_run interface

"""

open_index = output

launch = """This step ends the tutorial of the basic commands of aMCatNLO.
You can always use the help to see the options available for different
commands. For example, if you want to know how to launch on multicore/cluster
just type
aMC@NLO>help launch

To learn more about MadLoop standalone checks and runs, you can now follow
its tutorial with:
aMC@NLO>tutorial MadLoop

To simply close this tutorial, enter
aMC@NLO>tutorial stop
If you want to exit MG5, enter
aMC@NLO>exit
"""
