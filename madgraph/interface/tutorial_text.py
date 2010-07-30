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
You have just entered in the tutorial mode. This will introduce you the main
syntax options of MadGraph5.

To learn more about the different options on a given command, you can use  
mg5>help A_CMD
To see a list of all commands, use
mg5>help 

The goal of this tutorial is to learn how to generate
a process and to produce the output for MadEvent. In this part we will learn 
a) How to load a model
b) How to define multi-particles 
c) How to generate a process
d) How to create an output

Let's start with the first point: How to load a model:
mg5>import model sm
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
or presently defined multiparticle labels

Now you can define a multiparticle label, i.e a label corresponding 
to a set of particles:
mg5>define v w+ w- z a
defines the symbol \"v\" to correspond to any vector.
"""
import_model = import_model_v4

display_model = """
You have just seen some information about the model, which can help
you in order to generate a process.

Now you can define a multiparticle label, i.e a label corresponding 
to a set of particles:
mg5>define v w+ w- z a
defines the symbol \"v\" to correspond to any vector.
"""
display_particles = display_model
display_interactions = display_model

display_processes = """
You have seen a list of the already defined processes.

At this stage you can export your processes to a given format. In this
tutorial, we will explain how to create a valid output for
MadEvent.
This is done by typing:
mg5>setup madevent_v4 MY_FIRST_MG5_RUN
"""

define = """
You have just defined a multiparticle label.
If you followed the tutorial, the label is \"v\"

Note that some multiparticles such as as p, j, l+, l- are predefined. Type
mg5>display multiparticles
to know more about their definition

To generate a process using the multiparticle label, please run
mg5>generate p p > t t~ QED=0
Note that a space is mandatory between the particle names.
"""

generate = """
You have just generated a new process.

More information on the different syntax supported is accessible by
using:
mg5>help generate
To list all defined processes, type
mg5>display processes

To add a second process, please use the add process command:
mg5>add process p p > v j

At this stage you can export your processes to a given format. In this
tutorial, we will explain how to create a valid output for
MadEvent.
This is done by typing:
mg5>setup madevent_v4 MY_FIRST_MG5_RUN
"""

setup_madevent_v4 = """
If you are following the tutorial, a directory MY_FIRST_MG5_RUN has been
created which can be use in order to run MadEvent, exactly as if it
was coming from MG4.

This step ends the tutorial of the basic commands of MG5.  Don\'t
forget that you can always use the help to learn the different options
of the different command. For example, if you want to know all the
valid output format, you can enter
mg5>help setup

In order to close this tutorial please enter
mg5>tutorial stop

If you want to exit MG5 please enter
mg5>exit

But if you want, you can continue the tutorial with some additional
example of some other useful commands. MG5 authorizes you to write in
a file the list of command that you write in a interactive session:
mg5>history my_mg5_cmd.dat
"""

history = """
You have written a history file. If you followed the tutorial
this should be ./my_mg5_cmd.dat. In order to execute a file of
instruction you can do:
mg5>import command my_mg5_cmd.dat
or from the shell:
./madgraph5/bin/mg5 my_mg5_cmd.dat

It is also possible to look at this files from MG5 by launching
a shell command. For example:
mg5>shell less my_mg5_cmd.dat
"""

shell = """
Any shell command can be launched by MG5, by running \"shell\" or
starting the line by an exclamation mark (!).

The final command is draw. This allows you to draw the diagrams for
your processes (in eps format) before creating an output for a given
format. This can be useful for fast check of your process.  In order
to draw diagrams, you need to specify a directory where the eps files
will be written:
mg5>draw .
"""

draw = """
You can look at the diagrams for example by running
mg5>! gv ./diagrams_0_gg_ttx.eps
or on MacOS X
mg5>! open ./diagrams_0_gg_ttx.eps

This command was the last step of the tutorial. 
Quit the tutorial by typing:
mg5>tutorial stop

Thanks for using MG5.
"""

add_process = """
You have added a process to your process list.
At this stage your are able to export your processes to a given
format. In this tutorial, we will explain how to create a valid
output for MadEvent.  This is easily done by typing:

mg5>setup madevent_v4 MY_FIRST_MG5_RUN
"""





