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

import atexit
import cmd
import copy
import logging
import os
import re
import readline
import subprocess
import sys
import time

import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files

import madgraph.iolibs.import_v4 as import_v4
#import madgraph.iolibs.save_model as save_model
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.export_v4 as export_v4

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw

#position of MG5
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.split(root_path)[0]

#position of MG_ME
MGME_dir = None
MGME_dir_possibility = [os.path.join(root_path,os.path.pardir),
                os.path.join(os.getcwd(),os.path.pardir),
                os.getcwd()]

for position in MGME_dir_possibility:
    if os.path.exists(os.path.join(position, 'MGMEVersion.txt')) and \
                    os.path.exists(os.path.join(position, 'UpdateNotes.txt')):
        MGME_dir = os.path.realpath(position)
        del MGME_dir_possibility
        break
#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(cmd.Cmd):
    """The command line processor of MadGraph"""

    __curr_model = base_objects.Model()
    __model_dir = None
    __curr_amps = diagram_generation.AmplitudeList()
    # __org_amps holds the original amplitudes before export (for
    # decay chains, __curr_amps are replaced after export)
    __org_amps = diagram_generation.AmplitudeList()
    __curr_matrix_elements = helas_objects.HelasMultiProcess()
    __curr_fortran_model = export_v4.HelasFortranModel()
    __multiparticles = {}

    __display_opts = ['particles',
                      'interactions',
                      'processes',
                      'multiparticles']
    __add_opts = ['process']
    __save_opts = ['model', 'processes']
    __import_formats = ['model_v4', 'proc_v4','command']
    __export_formats = ['standalone_v4', 'sa_dirs_v4', 'madevent_v4']
    __export_dir = None 

    class MadGraphCmdError(Exception):
        """Exception raised if an error occurs in the execution
        of command."""
        pass
    
    def __init__(self,*arg,**opt):
        """ add a tracker of the history """
        
        self.history = []
        cmd.Cmd.__init__(self, *arg, **opt)
        
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
        
    def precmd(self, line):
        """ force the printing of the line if this is executed with an stdin """
        
        # Update the history of this suite of command
        self.history.append(line)
        
        # Print the calling line in the non interactive mode    
        if not self.use_rawinput:
            print line
        
        # Apply the line
        return line
    
    
    def default(self, line):
        """Default action if line is not recognized"""
        if line[0] == "#":
            # This is a comment - do nothing
            return
        else:
            # Faulty command
            print "Command \"%s\" not recognized, please try again" % \
                  line.split()[0]

    # Import files
    def do_import(self, line):
        """Import files with external formats"""

        def import_v4file(self, filepath):
            """Helper function to load a v4 file from file path filepath"""
            filename = os.path.basename(filepath)
            if filename.endswith('particles.dat'):
                self.__curr_model.set('particles',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_particles_v4))
                print "%d particles imported" % \
                      len(self.__curr_model['particles'])
                return True
            elif filename.endswith('interactions.dat'):
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
                return True

           
            #not valid File
            return False

        args = self.split_arg(line)
        if len(args) != 2 or args[0] not in self.__import_formats:
            self.help_import()
            return False
        
        


        if args[0] == 'model_v4':
            # Check for a file
            if os.path.isfile(args[1]):
                suceed = import_v4file(self, args[1])
                if not suceed:
                    print "%s is not a valid v4 file name" % \
                                        os.path.basename(args[1])
                else:
                    self.__model_dir = os.path.dirname(args[1])
                return
            
            # Check for a valid directory
            if os.path.isdir(args[1]):
                self.__model_dir = args[1]
            elif os.path.isdir(os.path.join(MGME_dir, 'Models', args[1])):
                self.__model_dir = os.path.join(MGME_dir, 'Models', args[1])
            else:
                print "Path %s is not a valid pathname" % args[1]
                return False
            
            #Load the directory
            if os.path.exists(os.path.join(self.__model_dir, 'model.pkl')):
                self.do_load('model %s' % os.path.join(self.__model_dir, \
                                                                   'model.pkl'))
                return
            files_to_import = ('particles.dat', 'interactions.dat')
            for filename in files_to_import:
                if os.path.isfile(os.path.join(self.__model_dir, filename)):
                    import_v4file(self, os.path.join(self.__model_dir, filename))
                else:
                    print "%s files doesn't exist in %s directory" % \
                                        (filename, os.path.basename(args[1]))
            #save model for next usage
            self.do_save('model %s ' % os.path.join(self.__model_dir, 'model.pkl'))
        
        elif args[0] == 'proc_v4':
            if len(args) == 1 and self.__export_dir:
                proc_card = os.path.join(self.__export_dir, '')            
            elif len(args) == 2:
                proc_card = args[1]
            else:
                logging.error('No default directory are setup')

            self.import_mg4_proc_card(proc_card)   
                                     
        elif args[0] == 'command':
            if not os.path.isfile(args[1]):
                print "Path %s is not a valid pathname" % args[1]
            else:
                self.import_mg5_proc_card(args[1])

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

    def import_mg4_proc_card(self, filepath):
        """ read a V4 proc card, convert it and run it in mg5"""
        
        # change the status of this line in the history -> pass in comment
        self.history[-1] = '#%s' % self.history[-1]
         
        # read the proc_card.dat
        reader = files.read_from_file(filepath, import_v4.read_proc_card_v4)
        
        if MGME_dir:
            #model_dir = os.path.join(MGME_dir, 'Models')
            line = self.onecmd_full('import model_v4 %s' % (reader.model))
        else:
            logging.error('No MG_ME installation detected')
            return    


        # Now that we have the model we can split the information
        lines = reader.treat_data(self.__curr_model)
        for line in lines:
            if 'setup' in line and self.__export_dir:
                continue
            self.onecmd_full(line)
            
        return 

    def import_mg5_proc_card(self, filepath):
        # change the status of this line in the history -> pass in comment
        self.history[-1] = '#%s' % self.history[-1]
 
        # Read the lines of the file and execute them
        for line in cmdfile(filepath):
            #remove pointless spaces and \n
            line = line.replace('\n','').strip()
            # execute the line if this one is not empty
            if line:
                self.onecmd_full(line)
        return


    def onecmd_full(self,line):
        """for third party call, call the line with pre and postfix treatment """
        print line
        line = self.precmd(line)
        stop = self.onecmd(line)
        stop = self.postcmd(stop, line)
        return stop      
        
        
    def do_save(self, line):
        """Save information to file"""

        args = self.split_arg(line)
        if len(args) != 2:
            self.help_save()
            return False

        if args[0] == 'model':
            if self.__curr_model:
                #save_model.save_model(args[1], self.__curr_model)
                if save_load_object.save_to_file(args[1], self.__curr_model):
                    print 'Saved model to file ', args[1]
            else:
                print 'No model to save!'
        elif args[0] == 'processes':
            if self.__curr_amps:
                if save_load_object.save_to_file(args[1], self.__curr_amps):
                    print 'Saved processes to file ', args[1]
            else:
                print 'No processes to save!'
        else:
            self.help_save()
            
    def do_setup(self, line):
        """Initialize a new Template or reinitialize one"""
        
        args = self.split_arg(line)
        clean = '-noclean' not in args
        force = '-f' in args 
        dir= '-d' in args
        if dir:
            mgme_dir = args[args.find('-d')+1]
        else:
            mgme_dir = MGME_dir
                        
        if len(args) < 2:
            self.help_setup()
            return False
        
        if not self.__model_dir:
            print 'No model found. Please import a model first and then retry'
            print '  for example do : import v4 sm'
            return False
        
        dir_path = os.path.join(mgme_dir,args[1])
        if not force and os.path.isdir(dir_path):
            print 'INFO: directory %s already exists.' %  args[1]
            if clean:
                print 'If you continue this directory will be cleaned'

            answer = raw_input('Do you want to continue? [y/n]')
            if answer != 'y':
                print 'stop'
                return False

        export_v4.copy_v4template(mgme_dir, dir_path, self.__model_dir, clean)
        # Import the model
        print 'import model files %s in directory %s' % \
                       (os.path.basename(self.__model_dir), args[1])        
        export_v4.export_model(self.__model_dir, dir_path)
        self.__export_dir = dir_path
        
        
    def complete_setup(self, text, line, begidx, endidx):
        "Complete the setup command"

        possible_option = ['-d ', '-f', '-noclean']
        possible_option2 = ['d ', 'f', 'noclean']
        possible_format = ['madevent_v4']
        #don't propose directory use by MG_ME
        forbidden_name = ['MadGraphII','Template','pythia-pgs', 'CVS', 
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis', 'mg5',
                            'DECAY', 'EventConverter', 'Models', 'ExRootAnalysis', 
                            'HELAS', 'Transfer_Fct']
        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, possible_format)
        
        #name of the run =>proposes old run name
        if len(self.split_arg(line[0:begidx])) == 2:
            mgme_pos = [os.path.join(root_path,os.path.pardir),
                        os.path.join(os.getcwd(),os.path.pardir),
                        os.getcwd()]
            for pos in mgme_pos:
                if os.path.isdir(os.path.join(pos, 'Template')):
                    content =[name for name in os.listdir(pos) if \
                                    name not in forbidden_name and \
                                    os.path.isdir(os.path.join(pos, name))]
                
                    return self.list_completion(text, content)

        # Returning options
        if len(self.split_arg(line[0:begidx])) > 2:
            if self.split_arg(line[0:begidx])[-1] == '-d':
                return self.path_completion(text)
            elif  self.split_arg(line[0:begidx])[-2] == '-d' and line[-1]!=' ':
                return self.path_completion(text, self.split_arg(line[0:begidx])[-1])
            elif self.split_arg(line[0:begidx])[-1] == '-':
                return self.list_completion(text, possible_option2)
            else:
                return self.list_completion(text, possible_option)

    def do_load(self, line):
        """Load information from file"""

        args = self.split_arg(line)
        if len(args) != 2:
            self.help_load()
            return False

        cpu_time1 = time.time()
        if args[0] == 'model':
            self.__curr_model = save_load_object.load_from_file(args[1])
            #save_model.save_model(args[1], self.__curr_model)
            if isinstance(self.__curr_model, base_objects.Model):
                cpu_time2 = time.time()
                print "Loaded model from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1)
            else:
                print 'Error: Could not load model from file ', args[1]
        elif args[0] == 'processes':
            self.__org_amps = diagram_generation.AmplitudeList()
            self.__curr_amps = save_load_object.load_from_file(args[1])
            if isinstance(self.__curr_amps, diagram_generation.AmplitudeList):
                cpu_time2 = time.time()
                print "Loaded processes from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1)
                if self.__curr_amps and not self.__curr_model.get('name'):
                    self.__curr_model = self.__curr_amps[0].\
                                        get('process').get('model')
                    print "Model set from process."
            else:
                print 'Error: Could not load processes from file ', args[1]
        else:
            self.help_save()

    def complete_save(self, text, line, begidx, endidx):
        "Complete the save command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__save_opts)

        # Filename if directory is not given
        if len(self.split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(self.split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[2])

    def complete_load(self, text, line, begidx, endidx):
        "Complete the load command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__save_opts)

        # Filename if directory is not given
        if len(self.split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(self.split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[2])

    def complete_draw(self, text, line, begidx, endidx):
        "Complete the import command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.path_completion(text)


        #option
        if len(self.split_arg(line[0:begidx])) >= 2:
            option = ['external=', 'horizontal=', 'add_gap=', 'max_size=', \
                                'contract_non_propagating=']
            return self.list_completion(text, option)

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
                print amp.nice_string()
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

        # Reset Helas matrix elements
        self.__curr_matrix_elements = helas_objects.HelasMultiProcess()

        try:
            if line.find(',') == -1:
                myprocdef = self.extract_process(line)
            else:
                myprocdef, line = self.extract_decay_chain_process(line)
        except self.MadGraphCmdError as error:
            print "Empty or wrong format process, please try again. Error:\n" \
                  + str(error)
            myprocdef = None
            
        if myprocdef:

            cpu_time1 = time.time()

            myproc = diagram_generation.MultiProcess(myprocdef)

            self.__curr_amps = myproc.get('amplitudes')
            self.__org_amps = copy.copy(self.__curr_amps)

            cpu_time2 = time.time()

            nprocs = len(self.__curr_amps)
            ndiags = sum([amp.get_number_of_diagrams() for \
                              amp in self.__curr_amps])
            print "%i processes with %i diagrams generated in %0.3f s" % \
                  (nprocs, ndiags, (cpu_time2 - cpu_time1))

        else:
            print "Empty or wrong format process, please try again."


    # Helper functions
    def extract_decay_chain_process(self, line, level_down = False):
        """Recursively extract a decay chain process definition from a
        string. Returns a ProcessDefinition."""

        index_comma = line.find(",")
        index_par = line.find(")")
        min_index = index_comma
        if index_par > -1 and (index_par < min_index or min_index == -1):
            min_index = index_par
        
        if min_index > -1:
            core_process = self.extract_process(line[:min_index])
        else:
            core_process = self.extract_process(line)

        #level_down = False

        while index_comma > -1:
            line = line[index_comma + 1:]
            index_par = line.find(')')
            if line.lstrip()[0] == '(':
                # Go down one level in process hierarchy
                #level_down = True
                line = line.lstrip()[1:]
                # This is where recursion happens
                decay_process, line = \
                            self.extract_decay_chain_process(line,
                                                             level_down = True)
                index_comma = line.find(",")
                index_par = line.find(')')
            else:
                index_comma = line.find(",")
                min_index = index_comma
                if index_par > -1 and \
                       (index_par < min_index or min_index == -1):
                    min_index = index_par
                if min_index > -1:
                    decay_process = self.extract_process(line[:min_index])
                else:
                    decay_process = self.extract_process(line)

            core_process.get('decay_chains').append(decay_process)

            if level_down:
                if index_par == -1:
                    raise self.MadGraphCmdError,\
                      "Missing ending parenthesis for decay process"

                if index_par < index_comma:
                    line = line[index_par + 1:]
                    level_down = False
                    break

        if level_down:
            index_par = line.find(')')
            if index_par == -1:
                raise self.MadGraphCmdError,\
                      "Missing ending parenthesis for decay process"
            line = line[index_par + 1:]
            
        # Return the core process (ends recursion when there are no
        # more decays)
        return core_process, line
        
    def extract_process(self, line):
        """Extract a process definition from a string. Returns
        a ProcessDefinition."""

        # Use regular expressions to extract s-channel propagators,
        # forbidden s-channel propagators/particles, coupling orders
        # and process number, starting from the back

        # Start with process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)\s*@\s*(\d+)\s*$")
        proc_number_re = proc_number_pattern.match(line)
        proc_number = 0
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            line = proc_number_re.group(1)

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

        myleglist = base_objects.MultiLegList()
        state = 'initial'
        number = 1

        # Extract process
        for part_name in args:

            if part_name == '>':
                if not myleglist:
                    raise self.MadGraphCmdError, \
                          "No final state particles"
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
                raise self.MadGraphCmdError,\
                      "No particle %s in model" % part_name

        if filter(lambda leg: leg.get('state') == 'final', myleglist):
            # We have a valid process

            # Now extract restrictions
            forbidden_particle_ids = []
            forbidden_schannel_ids = []
            required_schannel_ids = []

            decay_process = len(filter(lambda leg: \
                                       leg.get('state') == 'initial',
                                       myleglist)) == 1

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

            return \
                base_objects.ProcessDefinition({'legs': myleglist,
                                'model': self.__curr_model,
                                'id': proc_number,
                                'orders': orders,
                                'forbidden_particles': forbidden_particle_ids,
                                'forbidden_s_channels': forbidden_schannel_ids,
                                'required_s_channels': required_schannel_ids
                                 })
        #                       'is_decay_chain': decay_process\
        else:
            return None

    # Add a process to the existing multiprocess definition
    # Generate a new amplitude
    def do_add(self, line):
        """Generate an amplitude for a given process and add to
        existing amplitudes"""

        if len(line) < 1:
            self.help_add()
            return False

        if len(self.__curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False

        if len(self.__curr_model['interactions']) == 0:
            print "No interaction list currently active," + \
            " please create one first!"
            return False

        args = self.split_arg(line)
        if len(args) < 2:
            self.help_import()
            return False

        if args[0] == 'process':
            # Rejoin line
            line = ' '.join(args[1:])

            # Reset Helas matrix elements
            self.__curr_matrix_elements = helas_objects.HelasMultiProcess()

            try:
                if line.find(',') == -1:
                    myprocdef = self.extract_process(line)
                else:
                    myprocdef, line = self.extract_decay_chain_process(line)
            except self.MadGraphCmdError as error:
                print "Empty or wrong format process, please try again. Error:\n" \
                      + str(error)
                myprocdef = None

            if myprocdef:

                cpu_time1 = time.time()

                myproc = diagram_generation.MultiProcess(myprocdef)

                for amp in myproc.get('amplitudes'):
                    if amp not in self.__org_amps:
                        self.__org_amps.append(amp)
                    else:
                        print "Warning: Already in processes:"
                        print amp.nice_string_processes()

                self.__curr_amps = copy.copy(self.__org_amps)

                cpu_time2 = time.time()

                nprocs = len(myproc.get('amplitudes'))
                ndiags = sum([amp.get_number_of_diagrams() for \
                                  amp in myproc.get('amplitudes')])
                print "%i processes with %i diagrams generated in %0.3f s" % \
                      (nprocs, ndiags, (cpu_time2 - cpu_time1))
                ndiags = sum([amp.get_number_of_diagrams() for \
                                  amp in self.__curr_amps])
                print "Total: %i processes with %i diagrams" % \
                      (len(self.__curr_amps), ndiags)                
            else:
                print "Empty or wrong format process, please try again."


    def complete_add(self, text, line, begidx, endidx):
        "Complete the add command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self.__add_opts)
        
    # Write the list of command line use in this session
    def do_history(self,line):
        """write in a file the suite of command that was used"""
        
        # Start of the actual routine

        args = self.split_arg(line)

        if len(args) > 1:
            self.help_history()
            return False
        
        if len(args) == 0:
            print '\n'.join(self.history[:-1]) #don't print history
            return False
        
        output_file = open(args[0],'w')
        # Define a simple header for the file
        creation_time = time.asctime() 
        time_info = \
        '#     automaticaly generated the %s%s*\n' % (creation_time, ' '*(27-len(creation_time)))
        text = \
        '#************************************************************\n' + \
        '#                        MadGraph 5                         *\n' + \
        '#                                                           *\n' + \
        '#     The MadGraph Development Team - Please visit us at    *\n' + \
        '#             https://launchpad.net/madgraph5               *\n' + \
        '#                                                           *\n' + \
        '#************************************************************\n' + \
        '#                                                           *\n' + \
        '#               Command File for MadGraph 5                 *\n' + \
        '#                                                           *\n' + \
        '#     run as ./bin/mg5  filename                            *\n' + \
        time_info + \
        '#                                                           *\n' + \
        '#************************************************************\n'
        # Add the comand used 
        text += '\n'.join(self.history[:-1]) + '\n' #don't print history
        
        #write this information in a file
        output_file.write(text)
        output_file.close()

    def complete_history(self, text, line, begidx, endidx):
        "Complete the add command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.path_completion(text)

    # Export a matrix element
    def do_export(self, line):
        """Export a generated amplitude to file"""

        def generate_matrix_elements(self):
            """Helper function to generate the matrix elements before
            exporting"""

            cpu_time1 = time.time()
            if not self.__curr_matrix_elements.get('matrix_elements'):
                self.__curr_matrix_elements = \
                             helas_objects.HelasMultiProcess(\
                                           self.__curr_amps)
            cpu_time2 = time.time()

            ndiags = sum([len(me.get('diagrams')) for \
                          me in self.__curr_matrix_elements.\
                          get('matrix_elements')])

            return ndiags, cpu_time2 - cpu_time1

        # Start of the actual routine

        args = self.split_arg(line)

        if len(args) == 0 or (len(args) == 1 and not self.__export_dir) or \
                                            args[0] not in self.__export_formats:
            print len(args), (len(args) == 1 and not self.__export_dir),args[0] not in self.__export_formats
            print args
            self.help_export()
            return False

        if not self.__curr_amps:
            print "No process generated, please generate a process!"
            return False

        if len(args) == 1:
            path = os.path.join(self.__export_dir,'SubProcesses')
        else:
            path = args[1]

        if not os.path.isdir(path):
            print "%s is not a valid directory for export file" % path
            return False

        ndiags, cpu_time = generate_matrix_elements(self)
        calls = 0

        if args[0] == 'standalone_v4':
            for me in self.__curr_matrix_elements.get('matrix_elements'):
                filename = os.path.join(path, 'matrix_' + \
                           me.get('processes')[0].shell_string() + ".f")
                if os.path.isfile(filename):
                    print "Overwriting existing file %s" % filename
                else:
                    print "Creating new file %s" % filename
                calls = calls + files.write_to_file(filename,
                                                    export_v4.write_matrix_element_v4_standalone,
                                                    me,
                                                    self.__curr_fortran_model)


        if args[0] == 'sa_dirs_v4':
            for me in self.__curr_matrix_elements.get('matrix_elements'):
                calls = calls + \
                        export_v4.generate_subprocess_directory_v4_standalone(\
                            me, self.__curr_fortran_model, path)

        if args[0] == 'madevent_v4':
            for me in self.__curr_matrix_elements.get('matrix_elements'):
                calls = calls + \
                        export_v4.generate_subprocess_directory_v4_madevent(\
                            me, self.__curr_fortran_model, path)
            
            card_path = os.path.join(path,os.path.pardir,'Cards','proc_def.dat')
            export_v4.write_mg4_proc_card(card_path,
                                os.path.split(self.__model_dir)[-1],
                                self.__curr_matrix_elements.get('matrix_elements'))
                
        print ("Generated helas calls for %d subprocesses " + \
              "(%d diagrams) in %0.3f s") % \
              (len(self.__curr_matrix_elements.get('matrix_elements')),
               ndiags, cpu_time)

        print "Wrote %d helas calls" % calls

        # Replace the amplitudes with the actual amplitudes from the
        # matrix elements, which allows proper diagram drawing also of
        # decay chain processes
        self.__curr_amps = diagram_generation.AmplitudeList(\
               [me.get('base_amplitude') for me in \
                self.__curr_matrix_elements.get('matrix_elements')])

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

    def do_draw(self, line):
        """ draw the Feynman diagram for the given process """

        args = self.split_arg(line)

        if len(args) < 1:
            self.help_draw()
            return False
        
        if not self.__curr_amps:
            print "No process generated, please generate a process!"
            return False

        if not os.path.isdir(args[0]):
            print "%s is not a valid directory for export file" % args[1]

        start = time.time()
        option = draw_lib.DrawOption()
        if len(args) > 1:
            for data in args[1:]:
                try:
                    key, value = data.split('=')
                except:
                    print "invalid syntax: '%s'. Please try again" % data
                    self.help_draw()
                    return False
                option.set(key, value)

        # Collect amplitudes
        amplitudes = diagram_generation.AmplitudeList()

        for amp in self.__curr_amps:
            amplitudes.extend(amp.get_amplitudes())            

        for amp in amplitudes:
            filename = os.path.join(args[0], 'diagrams_' + \
                                    amp.get('process').shell_string() + ".eps")
            plot = draw.MultiEpsDiagramDrawer(amp['diagrams'],
                                              filename,
                                              model=self.__curr_model,
                                              amplitude='')

            logging.info("Drawing " + \
                         amp.get('process').nice_string())
            plot.draw(opt=option)
            print "Wrote file " + filename

        stop = time.time()
        print 'time to draw', stop - start

    def do_makehtml(self, line):
        """ make the html output for a MAdevent directory """
        
        args = self.split_arg(line)
        
        if len(args)<1:
            self.help_makehtml()
            return False
        
        if args[0] != 'madevent_v4':
            self.help_makehtml()
            return False
        
        if (len(args) == 1 and not self.__export_dir) or \
                        (len(args)>1 and not os.path.isdir(args[1])):
            self.help_makehtml()
            return False            
        
        if self.__export_dir:
            dir_path = self.__export_dir
        else: 
            dir_path = args[1]
        
        print 'creating html pages'    
        export_v4.create_v4_webpage(dir_path)
        
    def complete_makehtml(self, text, line, begidx, endidx):
        """ format: makehtlm madevent_v4 [PATH]"""
        
        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, ['madevent_v4'])
       
        # Filename if directory is not given
        if len(self.split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(self.split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[2]) 


    # Quit
    def do_quit(self, line):
        sys.exit(1)

    # In-line help
    def help_save(self):
        print "syntax: save %s FILENAME" % "|".join(self.__save_opts)
        print "-- save information as file FILENAME"


    def help_load(self):
        print "syntax: load %s FILENAME" % "|".join(self.__save_opts)
        print "-- load information from file FILENAME"

    def help_import(self):
        
        print "syntax: import " + "|".join(self.__import_formats) + \
              " FILENAME"
        print "-- imports file(s) in various formats"
        print
        print "   import model_v4 MODEL_info :"
        print "      Import a MG4 Model in MG5 Model"""
        print "      Model_info could be the name of the model"""
        print "                 or the path to MG4_Model directory"""
        print
        print "   import proc_v4 [PATH] :"  
        print "      Execute MG5 based on a proc_card.dat in MG4 format."""
        print "      Path to the proc_card is optional if you have setup a"
        print "      madevent directory"
        print 
        print "   import command PATH :"
        print "      Execute the list of command in the file at PATH"
        
    def help_display(self):
        print "syntax: display " + "|".join(self.__display_opts)
        print "-- display a the status of various internal state variables"

    def help_setup(self):
        print "syntax madevent_v4 name [options]"
        print "-- Create a copy of the V4 Template 'name' in the MG_ME directory."
        print "   options:"
        print "      -f: force the cleaning of the directory if this one exist"
        print "      -d PATH: specify the directory where to create name"
        print "      -noclean: no cleaning perform in name (simple storing position)"
        print "   Example:"
        print "       setup madevent_v4 MYRUN"
        print "       setup madevent_v4 MYRUN -d ../MG_ME -f"
        
    def help_generate(self):

        print "syntax: generate INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2"
        print "-- generate diagrams for a given process"
        print "   Example: u d~ > w+ > m+ vm g $ a / z h QED=3 QCD=0 @1"
        print "Decay chain syntax:"
        print "   core process, decay1, (decay2, (decay3, ...)), ...  etc"
        print "   Example: g g > t~ t @2, (t~ > W- b~, W- > e- ve~), t > W+ b"
        print "   Note that identical particles will all be decayed"

    def help_add(self):

        print "syntax: add process INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2"
        print "-- generate diagrams for a process and add to existing processes"
        print "   Syntax example: u d~ > w+ > m+ vm g $ a / z h QED=3 QCD=0 @1"
        print "Decay chain syntax:"
        print "   core process, decay1, (decay2, (decay3, ...)), ...  etc"
        print "   Example: g g > t~ t @2, (t~ > W- b~, W- > e- ve~), t > W+ b"
        print "   Note that identical particles will all be decayed"

    def help_define(self):
        print "syntax: define multipart_name [ part_name_list ]"
        print "-- define a multiparticle"
        print "   Example: define p u u~ c c~ d d~ s s~"

    def help_export(self):
        print "syntax: export " + "|".join(self.__export_formats) + \
              " FILEPATH"
        print """-- export matrix elements. For standalone_v4, the resulting
        files will be FILEPATH/matrix_\"process_string\".f. For sa_dirs_v4,
        the result is a set of complete MG4 Standalone process directories.
        For madevent_v4, the path needs to be to a MadEvent SubProcesses
        directory, and the result is the Pxxx directories (including the
        diagram .ps and .jpg files) for the subprocesses as well as a
        correctly generated subproc.mg file. Note that if you have run the 
        'setup', FILEPATH is optional."""

    def help_history(self):
        print "syntax: history [FILEPATH=stdout]"
        print "-- write in the specified files all the call to MG5 that you have """
        print "   perform since that you have open this command line applet."""

    def help_makehtml(self):
        print "syntax: makehtlm madevent_v4 [PATH]"
        print "-- create the web page related to the directory PATH"
        print "   by default PATH is the directory defined with the setup command"
        

    def help_draw(self):
        print "syntax: draw FILEPATH [option=value]"
        print "-- draw the diagrams in eps format"
        print "   Files will be FILEPATH/diagrams_\"process_string\".eps"
        print "   Example: draw plot_dir ."
        print "   Possible option: "
        print "        horizontal [False]: force S-channel to be horizontal"
        print "        external [0]: authorizes external particles to end"
        print "             at top or bottom of diagram. If bigger than zero"
        print "             this tune the length of those line."
        print "        max_size [0]: this forbids external line bigger than "
        print "             max_size."
        print "        contract_non_propagating [True]:contracts non propagating lines"
        print "   Example: draw plot_dir external=1 horizontal=1"

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
# 
#===============================================================================
class cmdfile(file):
    
    def readline(self,*arg,**opt):
        """readline outputing a \n at the end ot the read line"""
        line = file.readline(self, *arg, **opt)
        if line.endswith('\n'):
            return line
        elif line:
            return line + '\n'
        else:
            return line

#===============================================================================
# __main__
#===============================================================================

if __name__ == '__main__':
    
    opt = sys.argv
    if len(opt) > 1:
        # The first argument of sys.argv is the name of the program
        input_file = open(opt[1],'rU')
        cmd_line = MadGraphCmd(stdin=input_file)
        cmd_line.use_rawinput = False #put it in non interactive mode
        cmd_line.cmdloop()
    else:
        # Interactive mode
        MadGraphCmd().cmdloop()
