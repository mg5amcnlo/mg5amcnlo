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
import os

import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4

import madgraph.core.base_objects as base_objects

class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    _curr_model = base_objects.Model()

    _import_formats = ['v4']

    def split_arg(self, line):
        """Split a line of arguments"""
        args = line.split()
        return args

    def list_completion(self, text, list):
        """Propose completions of text in list"""
        if not text:
                completions = list
        else:
            completions = [ f
                            for f in list
                            if f.startswith(text)
                            ]
        return completions
#
#    def path_completion(self, text, line):
#        """Propose completions of text to compose a valid path"""
#        if not text:
#                completions = []
#        else:
#            first = os.path.dirname(text)
#            last = os.path.basename(text)
#            completions = [ os.path.join(first, f)
#                            for f in os.listdir(first)
#                            if f.startswith(last) and os.path.isfile(first + '/' + f)
#                            ]
#            completions = completions + \
#                            [  os.path.join(first, f) + '/'
#                            for f in os.listdir(first)
#                            if f.startswith(last) and os.path.isdir(first + '/' + f)
#                            ]
#        return completions

    def preloop(self):
        """Initializing before starting the main loop"""

        self.prompt = 'mg5>'

        # If possible, build an info line with current version number 
        # and date, from the VERSION text file

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
                "*                                                          *\n" + \
                "*          W E L C O M E  to  M A D G R A P H  5           *\n" + \
                "*                                                          *\n" + \
                info_line + \
                "*                                                          *\n" + \
                "*    The MadGraph Development Team - Please visit us at    *\n" + \
                "*              https://launchpad.net/madgraph5             *\n" + \
                "*                                                          *\n" + \
                "*               Type 'help' for in-line help.              *\n" + \
                "*                                                          *\n" + \
                "************************************************************"

    # Import files

    def do_import(self, line):
        """Import files with external formats"""

        args = self.split_arg(line)
        if len(args) != 2:
            self.help_import()
            return False

        if args[0] == 'v4':
            #Try to guess which function to call according to the given path
            if os.path.isdir(args[1]):
                pass
            elif os.path.isfile(args[1]):
                filename = os.path.basename(args[1])
                if filename == 'particles.dat':
                    self._curr_model.set('particles',
                                         files.act_on_file(args[1],
                                                import_v4.read_particles_v4))
            else:
                print "Path %s is not a valid pathname" % args[1]

    # Access to shell
    def do_shell(self, line):
        "Run a shell command"

        if line.strip() is '':
            self.help_shell()
        else:
            print "running shell command:", line
            print os.popen(line).read(),

    # Various ways to quit
    def do_quit(self, line):
        sys.exit(1)

    def do_EOF(self, line):
        sys.exit(1)

    # In-line help
    def help_import(self):
        print "syntax: import (v4|...) FILENAME",
        print "-- imports files in various formats"

    def help_shell(self):
        print "syntax: shell CMD",
        print "-- run the shell command CMD and catch output"

    def help_quit(self):
        print "syntax: quit",
        print "-- terminates the application"

    def help_help(self):
        print "syntax: help",
        print "-- access to the in-line help"

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
