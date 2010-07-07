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

demo = """
You have just enter in the demonstration mode. This will introduce you the main
syntax and the main options in order to have the basic knowledge of MadGraph5.

In order to know more about the different options on a given command, you can use  
mg5> help A_CMD
In order to know all the possible command, you can type:
mg5> help 

The first goal of this interactive demonstration is to learn to you how to generate
a process and to produce the output for MadEvent. In this part we will learn 
a) How to include a model
b) How to define multi-particles 
c) How to generate a process
d) How to create an output


Let's start with the first point: How to charge a model.
In order to do that you have to use the following command:
mg5> import model_v4 sm

In order to continue the demo, please import a model as describe above.
"""

error = """
Your command fails. The text above explains you why. This probably means that
you are not following the instruction provided in this demo.
 
Please enter the command suggested in the demonstration. 
"""

import_model_v4 ="""
You have just import successfully a model. If you follow the demo this is the 
Standard Model. 

If you want to know more information about this model you can run the following 
codes:
mg5>display particles
mg5>display interactions
which show to you information on respectively the particles and the vertex.

Now that you have define your model, you can generate some process. 
mg5>generate g g > t t~
Note that a space is mandatory between the particle name.

In order to continue the demo, please generate a process as describe above.
"""

display = """
You have just print some information on the model. Those information should help
you in order to generate a process.

In order to generate a process please run
mg5> generate g g > t t~
Note that a space is mandatory between the particle name.

In order to continue the demo, please generate a process as describe above.
"""

generate = """
You have just generate a new process.

More information on the different syntax supported, and how to add a second 
process is accessible in by typing:
mg5> help generate

Note that you can also define some multiparticles. i.e some label corresponding 
to a set of particles:
mg5>define p g u u~ d d~
define the symbol \"p\" as symbol corresponding either to a gluon or to a up or 
down quark.

At this stage your are able to export your process to a given format. 
In this demonstration, we will explain how to create a valid output for MadEvent.
This is done easily by typing:
mg5> setup madevent_v4 MY_FIRST_MG5_RUN
"""

setup_madevent_v4 = """
This last command creates a directory MY_FIRST_MG5_RUN which can be use in order 
to run madevent, exactly as if it was coming from MG4.

This steps finish the demonstration of the basic commands of MG5.
Don\'t forget that you can always use the help to learn the different options of 
the different command. 

For example, if you want to know all the valid output format, you can enter
mg5> help setup

in order to close this demonstration please enter
mg5> demo stop

If you want to close MG5 please enter
mg5> exit

But If you want, you can continue the demo with some additional example of 
some helpfull commmand.

Let's focus for example on how MG5 authorizes you to write in a file the list
of command that you have enter in a session. This is done easisly by
mg5> history my_mg5_cmd.dat

In order to continue the demo, please write an history file
"""

history = """
You have write an history file. In order to use a file containing valid MG5 
instruction, you can do simply
mg5> import command my_mg5_cmd.dat
or from the shell:
./madgraph5/bin/mg5 my_mg5_cmd.dat

It's also possible to open this files from MG5. Simply by launching a shell 
command from MG5. For example:
! vi my_mg5_cmd.dat
"""

shell = """
As you have seen any shell command can be launch by MG5. If you start the line 
by a point-mark. 

The final command is draw. This one authorizes to draw the diagram
(in eps format) before creating an output for a given format. This can be usefull
for fast check of your process and for improving the plots for publications.
In order to draw diagrams, you need to specify a directory where the eps files
will be written:
mg5> draw .
"""

draw = """
This command was the last step of the demonstration. 
Please quit the demo in typing:
mg5> demo stop

Thanks for using MG5.
"""

add_process = """
At this stage your are able to export your process to a given format. 
In this demonstration, we will explain how to create a valid output for MadEvent.
This is done easily by typing:
mg5> setup madevent_v4 MY_FIRST_MG5_RUN
"""





