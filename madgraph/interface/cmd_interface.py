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

    info = misc.get_pkg_info()
    info_line = ""

    if info.has_key('version') and  info.has_key('date'):
        len_version = len(info['version'])
        len_date = len(info['date'])
        if len_version + len_date < 30:
            info_line = "*         VERSION %s %s %s         *\n" % \
                        (info['version'],
                        (30 - len_version - len_date) * ' ',
                        info['date'])

    intro = "************************************************************\n" + \
            "*          W E L C O M E  to  M A D G R A P H  5           *\n" + \
            "*                                                          *\n" + \
            info_line + \
            "*                                                          *\n" + \
            "************************************************************\n"



    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
