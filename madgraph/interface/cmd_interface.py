##############################################################################
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
##############################################################################

"""A user friendly command line interface to access MadGraph features."""

import cmd
import madgraph.iolibs.misc as misc

class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    prompt = 'MG5>'

    intro = "************************************************************" + \
            "*           W E L C O M E  to  M A D G R A P H             *" + \
            "*                                                          *"


    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
