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

tutorial_MadLoop = """
You have entered tutorial mode. This will introduce you to the main
syntax options for MadLoop which are mostly similar to the MadGraph5 one. 
If you have not done so already, please follow MadGraph5 tutorial before 
this one.  

Remember that exactly as in MadGraph5, you can learn more about the different 
options for any command by typing
aMC@NLO>help A_CMD
And to see a list of all commands, use
aMC@NLO>help 

MadLoop is the part of MadGraph5 used by aMC@NLO to generate the code for
evaluating the loop diagrams. This tutorial teaches you how to use MadLoop
as standalone tool for studying loops within particular processes.
Therefore in this mode, you can only consider definite processes, meaning 
without multiparticle labels.

This tutorial has three parts:
a) How to generate a process.
b) How to cross-check / profile an output.
c) How to compute the loop matrix element squared for local phase-space points.

Let's start with the first point, how to generate a process with MadLoop in
standalone mode. Keep in mind that this means only the loop and born diagrams
are generated.

mg5>generate g g > t t~ [virt=QCD]

Note that a space is mandatory between the particle names and that '[virt=QCD]' 
specifies that you want to consider QCD NLO corrections. The keyword option
'virt' before '=' within the squared brackets precisely specifies you are only
interested in the virtual contribution. 
You will notice that MG5 recognizes you want to use standalone MadLoop because
the header of your interface will change from 'mg5>' to 'ML5>'. 
"""

generate = """
You have just generated a new process.
You can find more information on supported syntax by using:
ML5>help generate
To list all defined processes, type
ML5>display processes

You can display a pictorial representation of the diagrams with 
ML5>display diagrams
Notice you can add the option 'loop' or 'born' if you only want those diagrams
to be displayed.

If you want to add a second process, you can use the add process command:
ML5> add process u u~ > t t~ [virt=QCD]
But keep in mind that you must still consider only virtual corrections and 
cannot employ multiparticle labels. Also decay chains are not available for
loops.

At this stage you can export your processes.
This is done simply by typing:
ML5>output MY_FIRST_MADLOOP_RUN
"""

display_processes = """
You have seen a list of the already defined processes.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create a valid output for
MadEvent. This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCTANLO_RUN
"""

add_process = """
You have added a process to your process list.

At this stage you can export your processes.
This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCATNLO_RUN
"""
output = """
If you are following the tutorial, a directory MY_FIRST_MG5_RUN has
been created which can be used in order to run aMC@NLO.

Additionally to the commands in the bin directory (see 
MY_FIRST_MG5_RUN/README), you can also generate your events/compute the 
cross-section from this interface. 
You will generate events to be showered a la MC@NLO, compute the theoretical
and PDF error on the fly (if asked for in the run_card.dat) and shower the events 
with the parton_shower MonteCarlo specified in the run_card.dat, generating a file in
the StdHEP format.
Please Enter:
aMC@NLO> launch
(you can interrupt the computation to continue the tutorial by pressing Ctrl-C)
"""

open_index = output

launch = """This step ends the tutorial of the basic commands of aMCatNLO. You can
always use the help to see the options available for different
commands. For example, if you want to know how to launch on multicore/cluster
just type
aMC@NLO>help launch

In order to close this tutorial, enter
aMC@NLO>tutorial stop
If you want to exit MG5, enter
aMC@NLO>exit
"""
