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

"""A user friendly command line interface to access MadGraph features.
   Uses the cmd package for command interpretation and tab completion.
"""

import cmd
import os
import subprocess
import sys
import time
import readline
import atexit
import re

import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.export_v4 as export_v4

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects

#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    __curr_model = base_objects.Model()
    __curr_amps = diagram_generation.AmplitudeList()
    __curr_matrix_elements = helas_objects.HelasMultiProcess()
    __curr_fortran_model = export_v4.HelasFortranModel()
    __multiparticles = {}

    __display_opts = ['particles',
                      'interactions',
                      'processes',
                      'multiparticles']
    __import_formats = ['v4']
    __export_formats = ['v4standalone']

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

        readline.parse_and_bind("tab: complete")

        # initialize command history
        history_file = os.path.join(os.environ['HOME'], '.mg5history')
        try:
            readline.read_history_file(history_file)
        except IOError:
            pass
        atexit.register(readline.write_history_file, history_file)

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

        def import_v4file(self, filepath):
            """Helper function to load a v4 file from file path filepath"""
            filename = os.path.basename(filepath)
            if filename == 'particles.dat':
                self.__curr_model.set('particles',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_particles_v4))
                print "%d particles imported" % \
                      len(self.__curr_model['particles'])
            if filename == 'interactions.dat':
                if len(self.__curr_model['particles']) == 0:
                    print "No particle list currently active,",
                    print "please create one first!"
                    return False
                self.__curr_model.set('interactions',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_interactions_v4,
                                            self.__curr_model['particles']))
                print "%d interactions imported" % \
                      len(self.__curr_model['interactions'])

        args = self.split_arg(line)
        if len(args) != 2:
            self.help_import()
            return False

        if args[0] == 'v4':

            files_to_import = ('particles.dat', 'interactions.dat')

            if os.path.isdir(args[1]):
                for filename in files_to_import:
                    if os.path.isfile(os.path.join(args[1], filename)):
                        import_v4file(self, os.path.join(args[1], filename))

            elif os.path.isfile(args[1]):
                if os.path.basename(args[1]) in files_to_import:
                    import_v4file(self, args[1])
                else:
                    print "%s is not a valid v4 file name" % \
                                        os.path.basename(args[1])
            else:
                print "Path %s is not a valid pathname" % args[1]



    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__import_formats)

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
                    len(self.__curr_model['particles'])
            part_antipart = [part for part in self.__curr_model['particles'] \
                             if not part['self_antipart']]
            part_self = [part for part in self.__curr_model['particles'] \
                             if part['self_antipart']]
            for part in part_antipart:
                print part['name'] + '/' + part['antiname'],
            print ''
            for part in part_self:
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

        if args[0] == 'processes':
            for amp in self.__curr_amps:
                print amp.get('process').nice_string()
                print amp.get('diagrams').nice_string()

        if args[0] == 'multiparticles':
            print 'Multiparticle labels:'
            for key in self.__multiparticles:
                print key, " = ", self.__multiparticles[key]

    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__display_opts)

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

        if len(line) < 1:
            self.help_generate()
            return False

        if len(self.__curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False

        if len(self.__curr_model['interactions']) == 0:
            print "No interaction list currently active," + \
            " please create one first!"
            return False

        # Use regular expressions to extract s-channel propagators,
        # forbidden s-channel propagators/particles, coupling orders
        # starting from the back

        # Start with coupling orders (identified by "=")
        order_pattern = re.compile("^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(line)
        orders = {}
        while order_re:
            orders[order_re.group(2)] = int(order_re.group(3))
            line = order_re.group(1)
            order_re = order_pattern.match(line)

        # Particle names always lowercase
        line = line.lower()

        # Now check for forbidden particles, specified using "/"
        forbidden_particles_re = re.match("^(.+)\s*/\s*(.+)\s*$", line)
        forbidden_particles = ""
        if forbidden_particles_re:
            forbidden_particles = forbidden_particles_re.group(2)
            line = forbidden_particles_re.group(1)

        # Now check for forbidden schannels, specified using "$"
        forbidden_schannels_re = re.match("^(.+)\s*\$\s*(.+)\s*$", line)
        forbidden_schannels = ""
        if forbidden_schannels_re:
            forbidden_schannels = forbidden_schannels_re.group(2)
            line = forbidden_schannels_re.group(1)

        # Now check for required schannels, specified using "> >"
        required_schannels_re = re.match("^(.+?)>(.+?)>(.+)$", line)
        required_schannels = ""
        if required_schannels_re:
            required_schannels = required_schannels_re.group(2)
            line = required_schannels_re.group(1) + ">" + \
                   required_schannels_re.group(3)

        args = self.split_arg(line)

        # Reset Helas matrix elements
        self.__curr_matrix_elements = helas_objects.HelasMultiProcess()

        myleglist = base_objects.MultiLegList()
        state = 'initial'
        number = 1

        # Extract process
        for part_name in args:

            if part_name == '>':
                if not len(myleglist):
                    print "Empty or wrong format process, please try again."
                    return False
                state = 'final'
                continue

            mylegids = []
            if part_name in self.__multiparticles:
                mylegids.extend(self.__multiparticles[part_name])
            else:
                mypart = self.__curr_model['particles'].find_name(part_name)
                if mypart:
                    mylegids.append(mypart.get_pdg_code())

            if mylegids:
                myleglist.append(base_objects.MultiLeg({'ids':mylegids,
                                                        'state':state}))
            else:
                print "No particle %s in model: skipped" % part_name

        if filter(lambda leg: leg.get('state') == 'final', myleglist):
            # We have a valid process

            # Now extract restrictions
            forbidden_particle_ids = []
            forbidden_schannel_ids = []
            required_schannel_ids = []
            
            if forbidden_particles:
                args = self.split_arg(forbidden_particles)
                for part_name in args:
                    if part_name in self.__multiparticles:
                        forbidden_particle_ids.extend(self.__multiparticles[part_name])
                    else:
                        mypart = self.__curr_model['particles'].find_name(part_name)
                        if mypart:
                            forbidden_particle_ids.append(mypart.get_pdg_code())

            if forbidden_schannels:
                args = self.split_arg(forbidden_schannels)
                for part_name in args:
                    if part_name in self.__multiparticles:
                        forbidden_schannel_ids.extend(self.__multiparticles[part_name])
                    else:
                        mypart = self.__curr_model['particles'].find_name(part_name)
                        if mypart:
                            forbidden_schannel_ids.append(mypart.get_pdg_code())

            if required_schannels:
                args = self.split_arg(required_schannels)
                for part_name in args:
                    if part_name in self.__multiparticles:
                        required_schannel_ids.extend(self.__multiparticles[part_name])
                    else:
                        mypart = self.__curr_model['particles'].find_name(part_name)
                        if mypart:
                            required_schannel_ids.append(mypart.get_pdg_code())

                

            myprocdef = base_objects.ProcessDefinitionList([\
                base_objects.ProcessDefinition({'legs': myleglist,
                                                'model': self.__curr_model,
                                                'orders': orders,
                                                'forbidden_particles': forbidden_particle_ids,
                                                'forbidden_s_channels': forbidden_schannel_ids,
                                                'required_s_channels': required_schannel_ids \
                                                })])
            myproc = diagram_generation.MultiProcess({'process_definitions':\
                                                      myprocdef})

            cpu_time1 = time.time()
            self.__curr_amps = myproc.get('amplitudes')
            cpu_time2 = time.time()

            nprocs = len(filter(lambda amp: amp.get("diagrams"),
                                self.__curr_amps))
            ndiags = sum([len(amp.get('diagrams')) for \
                              amp in self.__curr_amps])
            print "%i processes with %i diagrams generated in %0.3f s" % \
                  (nprocs, ndiags, (cpu_time2 - cpu_time1))

        else:
            print "Empty or wrong format process, please try again."

    # Generate a new amplitude
    def do_export(self, line):
        """Export a generated amplitude to file"""

        def export_v4standalone(self, filepath):
            """Helper function to write a v4 file to file path filepath"""

            if not self.__curr_matrix_elements.get('matrix_elements'):
                cpu_time1 = time.time()
                self.__curr_matrix_elements = \
                             helas_objects.HelasMultiProcess(\
                                           self.__curr_amps)
                cpu_time2 = time.time()

                ndiags = sum([len(me.get('diagrams')) for \
                              me in self.__curr_matrix_elements.\
                              get('matrix_elements')])

            calls = 0
            for me in self.__curr_matrix_elements.get('matrix_elements'):
                filename = filepath + '/matrix_' + \
                           me.get('processes')[0].shell_string() + ".f"
                if os.path.isfile(filename):
                    print "Overwriting existing file %s" % filename
                else:
                    print "Creating new file %s" % filename
                calls = calls + files.write_to_file(filename,
                                                    export_v4.write_matrix_element_v4_standalone,
                                                    me,
                                                    self.__curr_fortran_model)

            print "Generated helas calls for %d subprocesses (%d diagrams) in %0.3f s" % \
                  (len(self.__curr_matrix_elements.get('matrix_elements')),
                   ndiags,
                   (cpu_time2 - cpu_time1))

            print "Wrote %d helas calls" % calls

        args = self.split_arg(line)

        if len(args) < 1:
            self.help_export()
            return False

        if len(args) != 2 or args[0] not in self.__export_formats:
            self.help_export()
            return False

        if not filter(lambda amp: amp.get("diagrams"), self.__curr_amps):
            print "No process generated, please generate a process!"
            return False

        if not os.path.isdir(args[1]):
            print "%s is not a valid directory for export file" % args[1]

        if args[0] == 'v4standalone':
            export_v4standalone(self, args[1])

    def complete_export(self, text, line, begidx, endidx):
        "Complete the export command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__export_formats)

        # Filename if directory is not given
        if len(self.split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(self.split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[2])

    # Define a multiparticle label
    def do_define(self, line):
        """Define a multiparticle"""

        # Particle names always lowercase
        line = line.lower()

        args = self.split_arg(line)

        if len(args) < 1:
            self.help_define()
            return False

        if len(self.__curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False

        label = args[0]
        pdg_list = []

        for part_name in args[1:]:

            mypart = self.__curr_model['particles'].find_name(part_name)

            if mypart:
                pdg_list.append(mypart.get_pdg_code())
            else:
                print "No particle %s in model: skipped" % part_name

        if not pdg_list:
            print """Empty or wrong format for multiparticle.
            Please try again."""

        self.__multiparticles[label] = pdg_list

    # Quit
    def do_quit(self, line):
        sys.exit(1)

    # In-line help
    def help_import(self):
        print "syntax: import " + "|".join(self.__import_formats) + \
              " FILENAME"
        print "-- imports file(s) in various formats"

    def help_display(self):
        print "syntax: display " + "|".join(self.__display_opts)
        print "-- display a the status of various internal state variables"

    def help_generate(self):
        print "syntax: generate INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2"
        print "-- generate diagrams for a given process"
        print "   Example: u d~ > w+ > m+ vm g $ a / z h QED=3 QCD=0"

    def help_define(self):
        print "syntax: define multipart_name [ part_name_list ]"
        print "-- define a multiparticle"
        print "   Example: define p u u~ c c~ d d~ s s~"

    def help_export(self):
        print "syntax: export " + "|".join(self.__export_formats) + \
              " FILEPATH"
        print """-- export matrix elements in various formats. The resulting
        file will be FILEPATH/matrix_\"process_string\".f"""

    def help_shell(self):
        print "syntax: shell CMD (or ! CMD)"
        print "-- run the shell command CMD and catch output"

    def help_quit(self):
        print "syntax: quit"
        print "-- terminates the application"

    def help_help(self):
        print "syntax: help"
        print "-- access to the in-line help"

    # Aliases

    do_EOF = do_quit
    help_EOF = help_quit

#===============================================================================
# __main__
#===============================================================================

if __name__ == '__main__':
    MadGraphCmd().cmdloop()
