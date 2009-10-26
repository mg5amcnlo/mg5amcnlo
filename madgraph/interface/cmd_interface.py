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
import os
import subprocess
import sys
import time

import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    __curr_model = base_objects.Model()
    __curr_amp = diagram_generation.Amplitude()

    __import_formats = ['v4']

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
                    self.__curr_model.set('particles',
                                         files.read_from_file(
                                                args[1],
                                                import_v4.read_particles_v4))
                if filename == 'interactions.dat':
                    self.__curr_model.set('interactions',
                                         files.read_from_file(
                                                args[1],
                                                import_v4.read_interactions_v4,
                                                self.__curr_model['particles']))
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
        
        if len(args) != 1:
            self.help_display()
            return False
        
        if args[0] == 'particles':
            print "Current model contains %i particles:" % \
                    len(self.__curr_model['particles']),
            for part in self.__curr_model['particles']:
                print part['name'],
            print ''

        if args[0] == 'interactions':
            print "Current model contains %i interactions" % \
                    len(self.__curr_model['interactions'])
            for inter in self.__curr_model['interactions']:
                print str(inter['id']) + ':',
                for part in inter['particles']:
                    if part['is_part']:
                        print part['name'],
                    else:
                        print part['antiname'],
                print
        
        if args[0] == 'amplitude':
            print self.__curr_amp['process'].nice_string()
            print self.__curr_amp['diagrams'].nice_string()

    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        display_opts = ['particles',
                        'interactions',
                        'amplitude']
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
    
    # Generate a new amplitude
    def do_generate(self, line):
        """Generate an amplitude for a given process"""
    
        args = self.split_arg(line)
        
        if len(args) < 1:
            self.help_display()
            return False
        
        if len(self.__curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False
        
        if len(self.__curr_model['interactions']) == 0:
            print "No interaction list currently active," + \
            " please create one first!"
            return False
        
        myleglist = base_objects.LegList()
        state = 'initial'
        number = 1
        
        for part_name in args:
            if part_name == '>':
                state = 'final'
                continue
            
            mypart = self.__curr_model['particles'].find_name(part_name)
            
            if mypart:
                myleglist.append(base_objects.Leg({'id':mypart.get_pdg_code(),
                                                   'number':number,
                                                   'state':state}))
                number = number + 1
            else:
                print "Error with particle %s: skipped" % part_name
        
        if myleglist and state == 'final':
            myproc = base_objects.Process({'legs':myleglist,
                                            'orders':{},
                                            'model':self.__curr_model})
            self.__curr_amp.set('process', myproc)
            
            cpu_time1 = time.time()
            ndiags = len(self.__curr_amp.generate_diagrams())
            cpu_time2 = time.time()
            
            print "%i diagrams generated in %0.3f s" % (ndiags, (cpu_time2 - \
                                                               cpu_time1))

        else:
            print "Empty or wrong format process, please try again."

    # Quit
    def do_quit(self, line):
        sys.exit(1)

    # In-line help
    def help_import(self):
        print "syntax: import v4|... FILENAME",
        print "-- imports file(s) in various formats"

    def help_display(self):
        print "syntax: display particles|interactions|amplitude",
        print "-- display a the status of various internal state variables"
    
    def help_generate(self):
        print "syntax: generate INITIAL STATE > FINAL STATE",
        print "-- generate amplitude for a given process (list of particles"
        print "   separated by >)"

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
