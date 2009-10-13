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

"""A user friendly command line interface to access MadGraph features."""

import cmd
import functools
import os
import subprocess
import sys

import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4

import madgraph.core.base_objects as base_objects

#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    _curr_model = base_objects.Model()

    _import_formats = ['v4']

    def split_arg(self, line):
        """Split a line of arguments"""
        return line.split()

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

    def path_completion(self, text, base_dir=None):
        """Propose completions of text to compose a valid path"""

        if base_dir is None:
            base_dir = os.getcwd()

        completion = [f
                       for f in os.listdir(base_dir)
                       if f.startswith(text) and \
                            os.path.isfile(os.path.join(base_dir, f))
                       ]

        completion = completion + \
                     [f + '/'
                       for f in os.listdir(base_dir)
                       if f.startswith(text) and \
                            os.path.isdir(os.path.join(base_dir, f))
                     ]
        return completion


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

        self.intro = \
        "************************************************************\n" + \
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
                                         files.read_from_file(
                                                args[1],
                                                import_v4.read_particles_v4))
                if filename == 'interactions.dat':
                    self._curr_model.set('interactions',
                                         files.read_from_file(
                                                args[1],
                                                import_v4.read_interactions_v4,
                                                self._curr_model['particles']))
            else:
                print "Path %s is not a valid pathname" % args[1]

    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, ['v4'])

        # Filename if directory is not given
        if len(self.split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(self.split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[2])

    # Display
    def do_display(self, line):
        """Display current internal status"""

        args = self.split_arg(line)

        if args[0] == 'particles':
            print "Current model contains %i particles:" % \
                    len(self._curr_model['particles']),
            for part in self._curr_model['particles']:
                print part['name'],
            print ''

        if args[0] == 'interactions':
            print "Current model contains %i interactions" % \
                    len(self._curr_model['interactions'])

    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        display_opts = ['particles',
                        'interactions']
        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, display_opts)

    # Access to shell
    def do_shell(self, line):
        "Run a shell command"

        if line.strip() is '':
            self.help_shell()
        else:
            print "running shell command:", line
            subprocess.call(line, shell=True)

    # Quit
    def do_quit(self, line):
        sys.exit(1)

    # In-line help
    def help_import(self):
        print "syntax: import v4|... FILENAME",
        print "-- imports file(s) in various formats"

    def help_display(self):
        print "syntax: display particles|interactions",
        print "-- display a the status of various internal state variables"

    def help_shell(self):
        print "syntax: shell CMD (or ! CMD)",
        print "-- run the shell command CMD and catch output"

    def help_quit(self):
        print "syntax: quit",
        print "-- terminates the application"

    def help_help(self):
        print "syntax: help",
        print "-- access to the in-line help"

    # ALiases

    do_EOF = do_quit
    help_EOF = help_quit

#===============================================================================
# __main__
#===============================================================================

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
