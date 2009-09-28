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
import sys

import madgraph.iolibs.misc as misc

class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    def preloop(self):
        """Initializing before starting the main loop"""

        self.prompt = 'mg>'

        # If possible, build an info line with current version number and date, from
        # the VERSION text file

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

        self.intro = "************************************************************\n" + \
                "*          W E L C O M E  to  M A D G R A P H  5           *\n" + \
                "*                                                          *\n" + \
                info_line + \
                "*                                                          *\n" + \
                "*    The MadGraph Development Team - Please visit us at    *\n" + \
                "*              https://launchpad.net/madgraph5             *\n" + \
                "*                                                          *\n" + \
                "************************************************************"

    # Various ways to quit

    def do_quit(self, arg):
        sys.exit(1)

    def help_quit(self):
        print "syntax: quit",
        print "-- terminates the application"

    def do_EOF(self, line):
        sys.exit(1)

    # shortcuts
    do_q = do_quit

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
