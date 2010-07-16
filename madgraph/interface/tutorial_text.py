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

tutorial = """
You have just entered in the tutorial mode. This will introduce you to
the main syntax options of MadGraph5.

To learn more about the different options for a command, you can use
mg5>help A_CMD
To see a list of all commands, use
mg5>help 

The goal of this tutorial is to learn how to generate a process and to
produce the output for MadEvent. In this part we will learn
a) How to load a model
b) How to generate a process
c) How to create output for MadEvent

Let's start with the first point, how to load a model:
mg5>import model_v4 sm
"""


import_model_v4 ="""
You have successfully imported a model. If you followed the tutorial
this is the Standard Model.

If you want to know more information about this model you can use the
following commands:
mg5>display particles
mg5>display interactions
mg5>display multiparticles
which show information on the particles and the vertices of the model
or presently defined multiparticle labels.

You can now generate a process, by running
mg5>generate p p > t t~ QED=0
Note that a space is mandatory between the particle names.
"""

display_model = """
You have just seen some information about the model, which can help
you in order to generate a process.

You can now generate a process, by running
mg5>generate p p > t t~ QED=0
Note that a space is mandatory between the particle names.
"""
display_particles = display_model
display_interactions = display_model

generate = """
You have just generated a new process.

You can find more information on supported syntax by using:
mg5>help generate
To list all defined processes, type
mg5>display processes

To add a second process, please use the add process command:
mg5>add process p p > W+ j QED=1, W+ > l+ vl
This adds a decay chain process, with the W+ decaying
leptonically.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create output for MadEvent.
This is done simply by typing:
mg5>setup madevent_v4 MY_FIRST_MG5_RUN -f
"""

display_processes = """
You have seen a list of the already defined processes.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create a valid output for
MadEvent. This is done simply by typing:
mg5>setup madevent_v4 MY_FIRST_MG5_RUN -f
"""

add_process = """
You have added a process to your process list.

At this stage you can export your processes to different formats. In
this tutorial, we will explain how to create a valid output for
MadEvent. This is done simply by typing:
mg5>setup madevent_v4 MY_FIRST_MG5_RUN -f
"""

setup_madevent_v4 = """
If you are following the tutorial, a directory MY_FIRST_MG5_RUN has
been created which can be used in order to run MadEvent exactly as if
it was coming from MG4, see MY_FIRST_MG5_RUN/README.

This step ends the tutorial of the basic commands of MG5. You can
always use the help to see the options available for different
commands. For example, if you want to know all valid output formats,
you can enter
mg5>help setup

In order to close this tutorial please enter
mg5>tutorial stop
If you want to exit MG5 please enter
mg5>exit

But you can also continue the tutorial to learn some other useful
commands:
d) How to define a multi-particle label 
e) How to store a history of the commands in a session
f) How to call shell commands from MG5
g) How to draw the diagrams for your processes without generating
   MadEvent output

To define a multiparticle label, i.e. a label corresponding to a set
of particles, write:
mg5>define v = w+ w- z a
This defines the symbol \"v\" to correspond to any EW vector boson.
"""

define = """
You have just defined a multiparticle label.
If you followed the tutorial, the label is \"v\"

Note that some multiparticles such as as p, j, l+, l- are
predefined. Type
mg5>display multiparticles
to see their definitions.

MG5 allows you to store a file with the list of command that you have
used in an interactive session:
mg5>history my_mg5_cmd.dat
"""

history = """
You have written a history file. If you followed the tutorial this
should be ./my_mg5_cmd.dat. In order to load a history file and
execute the commands in it, you can do:
mg5>import command my_mg5_cmd.dat
or from the shell:
./madgraph5/bin/mg5 my_mg5_cmd.dat

It is also possible to display this file directly from MG5 by
launching a shell command. For example:
mg5>shell less my_mg5_cmd.dat
"""

shell = """
Any shell command can be launched by MG5, by running \"shell\" or
starting the line by an exclamation mark (!).

The final command of the tutorial is draw. This allows you to draw and
look at the diagrams for your processes (in eps format) before
creating an output for a given format. This can be useful for a fast
check of your process.  In order to draw diagrams, you need to specify
a directory where the eps files will be written:
mg5>draw .

Note that when you run setup madevent_v4, the diagrams are
automatically written to the matrix.ps files in subprocess
directory, just like with MadGraph 4.
"""

draw = """
You can look at the diagrams for example by running
mg5>!gv ./diagrams_0_gg_ttx.eps
or on MacOS X
mg5>!open ./diagrams_0_gg_ttx.eps

This command was the last step of the tutorial. 
Quit the tutorial by typing:
mg5>tutorial stop

Thanks for using MG5.
"""




