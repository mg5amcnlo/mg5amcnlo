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

tutorial_nlo = """
You have entered tutorial mode. This will introduce you to the main
syntax options of aMC@NLO. Mos of the syntax corresponds to the MadGraph one 

As in MadGraph, to learn more about the different options for a command, 
you can use
aMC@NLO>help A_CMD
To see a list of all commands, use
aMC@NLO>help 

The goal of this tutorial is to learn how to generate a process and to
produce the output for aMC@NLO. In this part we will learn
a) How to generate a process
b) How to create output for aMC@NLO
c) How to run the aMC@NLO output

Let's start with the first point, how to generate a process at NLO:
mg5>generate p p > e+ ve [QCD]
Note that a space is mandatory between the particle names and that the [QCD] 
specifies that you want to do NLO in QCD. Couplings different than QCD cannot
be perturbed yet.
Note also that the interface automatically switched to the aMC@NLO one.
"""

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
aMC@NLO>add process p p > e+ e- @2

At this stage you can export your processes.
This is done simply by typing:
aMC@NLO>output MY_FIRST_AMCATNLO_RUN
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
You will generate events to be showered à la MC@NLO, compute the theoretical
and PDF error on the fly (if asked for in the run_card.dat) and shower the events 
with the parton_shower montecarlo you chosed in the run_card.dat 
Please Enter:
aMC@NLO> launch
(you can interrupt the computation to continue the tutorial by pressing Ctrl-C)
"""

open_index = output

launch = """This step ends the tutorial of the basic commands of aMCatNLO. You can
always use the help to see the options available for different
commands. For example, if you want to know how to lunch on multicore/cluster
just type
aMC@NLO>help launch

In order to close this tutorial please enter
aMC@NLO>tutorial stop
If you want to exit MG5 please enter
aMC@NLO>exit

But you can also continue the tutorial to learn some other useful
commands:
d) How to load a model
e) How to define a multi-particle label 
f) How to store a history of the commands in a session
g) How to call shell commands from MG5
h) How to draw the diagrams for your processes without generating
   MadEvent output

To import a model, write:
aMC@NLO>import model mssm
"""

import_model ="""
You have successfully imported a model. If you followed the tutorial
this is the MSSM.

If you want to know more information about this model you can use the
following commands:
aMC@NLO>display particles
aMC@NLO>display interactions
aMC@NLO>display multiparticles
which show information on the particles and the vertices of the model
or presently defined multiparticle labels.

To define a multiparticle label, i.e. a label corresponding to a set
of particles, write:
aMC@NLO>define v = w+ w- z a
This defines the symbol \"v\" to correspond to any EW vector boson.
"""
import_model_v4 = import_model

define = """
You have just defined a multiparticle label.
If you followed the tutorial, the label is \"v\"

Note that some multiparticles such as as p, j, l+, l- are
predefined. Type
aMC@NLO>display multiparticles
to see their definitions.

MG5 allows you to store a file with the list of command that you have
used in an interactive session:
aMC@NLO>history my_mg5_cmd.dat
"""

history = """
You have written a history file. If you followed the tutorial this
should be ./my_mg5_cmd.dat. In order to load a history file and
execute the commands in it, you can do:
aMC@NLO>import command my_mg5_cmd.dat
or from the shell:
./bin/mg5 my_mg5_cmd.dat

It is also possible to display this file directly from MG5:
aMC@NLO>open ./my_mg5_cmd.dat
"""


open_index = output

open = """
Note that in order to open some file, you might be need to use a shell command.
Any shell command can be launched by MG5, by running \"shell\" or
starting the line by an exclamation mark (!).

The final command of the tutorial is display diagrams. This allows you to draw and
look at the diagrams for your processes (in eps format) before
creating an output for a given format. This can be useful for a fast
check of your process. For this last command, we will also show how combine
different command in a single line: 
aMC@NLO>generate p p > go go; display diagrams

Note that when you run output [madevent_v4], the diagrams are
automatically written to the matrix.ps files in subprocess
directory, just like with MadGraph 4.
"""

display_diagrams = """
This command was the last step of the tutorial. 
Quit the tutorial by typing:
aMC@NLO>tutorial stop

Thanks for using MG5.
"""




