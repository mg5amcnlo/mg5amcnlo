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
import logging
import optparse
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
root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
root_path = os.path.split(root_path)[0]

#position of MG_ME
MGME_dir = None
MGME_dir_possibility = [os.path.join(root_path, os.path.pardir),
                os.path.join(os.getcwd(), os.path.pardir),
                os.getcwd()]

for position in MGME_dir_possibility:
    if os.path.exists(os.path.join(position, 'MGMEVersion.txt')) and \
                    os.path.exists(os.path.join(position, 'UpdateNotes.txt')):
        MGME_dir = os.path.realpath(position)
        break
del MGME_dir_possibility

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
    """Extension of the cmd.Cmd command line.
    This extensions supports linesbreak, history
    support commentary
    """

    class MadGraphCmdError(Exception):
        """Exception raised if an error occurs in the execution
        of command."""
        pass

    def __init__(self, *arg, **opt):
        """add possibility of line break/history """
        
        self.history = []
        self.save_line = ''
        cmd.Cmd.__init__(self, *arg, **opt)

    def precmd(self, line):
        """ force the printing of the line if this is executed with an stdin """
        # Update the history of this suite of command
        self.history.append(line)
        
        # Print the calling line in the non interactive mode    
        if not self.use_rawinput:
            print line
        
        #Check if they are a save line:
        if self.save_line:
            line = self.save_line + line 
            self.save_line = ''
        
        #Check that the line is complete
        if line.endswith('\\'):
            self.save_line = line[:-1]
            return '' # do nothing   
        
        #remove comment
        if '#' in line:
            line = line.split('#')[0]
            
        # execute the line command
        return line

    def exec_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment"""
        print line
        line = self.precmd(line)
        stop = self.onecmd(line)
        stop = self.postcmd(stop, line)
        return stop      

    def emptyline(self):
        """If empty line, do nothing. Default is repeat previous command."""
        pass
    
    def default(self, line):
        """Default action if line is not recognized"""

        # Faulty command
        print "Command \"%s\" not recognized, please try again" % \
                                                                 line.split()[0]
    # Quit
    def do_quit(self, line):
        sys.exit(1)

    # Aliases
    do_EOF = do_quit
   

#===============================================================================
# Helper function
#=============================================================================
def split_arg(line):
    """Split a line of arguments"""
    return line.split()
 

#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routine for the MadGraphCmd"""    
    
    def help_save(self):
        print "syntax: save %s FILENAME" % "|".join(self._save_opts)
        print "-- save information as file FILENAME"


    def help_load(self):
        print "syntax: load %s FILENAME" % "|".join(self._save_opts)
        print "-- load information from file FILENAME"

    def help_import(self):
        
        print "syntax: import " + "|".join(self._import_formats) + \
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
        print "syntax: display " + "|".join(self._display_opts)
        print "-- display a the status of various internal state variables"

    def help_setup(self):
        print "syntax madevent_v4 name|.|auto [options]"
        print "-- Create a copy of the V4 Template in the MG_ME directory."
        print "   name will be the name of the copy the madevent Template that"
        print "     will be use by default in following steps"
        print "   If you put '.' instead of a name, the code will try to locate"
        print "     a valid copy of the Template under focus"
        print "   If you put 'auto' instead an automatic name will be created"  
        print "   options:"
        print "      -f: force the cleaning of the directory if this one exist"
        print "      -d PATH: specify the directory where to create name"
        print "      -noclean: no cleaning perform in name"
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
        print "syntax: export " + "|".join(self._export_formats) + \
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
        print "syntax: history [FILEPATH] [-clean] "
        print "-- write in the specified files all the call to MG5 that you have"""
        print "   perform since that you have open this command line applet."""
        print "   -clean option will remove all the entry of the history"""

    def help_makehtml(self):
        print "syntax: makehtlm madevent_v4 [PATH]"
        print "-- create the web page related to the directory PATH"
        print "   by default PATH is the directory defined with the setup command"
        

    def help_draw(self):
        _draw_parser.print_help()

    def help_shell(self):
        print "syntax: shell CMD (or ! CMD)"
        print "-- run the shell command CMD and catch output"

    def help_quit(self):
        print "syntax: quit"
        print "-- terminates the application"
    
    help_EOF = help_quit
    
    def help_help(self):
        print "syntax: help"
        print "-- access to the in-line help" 

#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of help routine for the MadGraphCmd"""
    
    def check_add(self, args):
        """check the validity of line
        syntax: add process PROCESS 
        """
        
        if len(args) < 1:
            self.help_generate()
            return False
        
        if len(self._curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False

        if len(self._curr_model['interactions']) == 0:
            print "No interaction list currently active," + \
            " please create one first!"
            return False

        if len(args) < 2:
            self.help_import()
            return False
        
        if args[0] != 'process':
            print "Empty or wrong format process, please try again."
            return False
        
        return True
    
    def check_define(self, args):
        """check the validity of line
        syntax: define multipart_name [ part_name_list ]
        """
        
        if len(args) < 1:
            self.help_define()
            return False

        if len(self._curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False
        
        return True

    def check_display(self, args):
        """check the validity of line
        syntax: display XXXXX
        """
            
        if len(args) != 1:
            self.help_display()
            return False
        else:
            return True
    
    def check_draw(self, args):
        """check the validity of line
        syntax: draw FILEPATH [option=value]
        """
        if len(args) < 1:
            self.help_draw()
            return False
        
        if not self._curr_amps:
            print "No process generated, please generate a process!"
            return False

        if not os.path.isdir(args[0]):
            print "%s is not a valid directory for export file" % args[1]
            return False
        
        return True
    
    def check_export(self, args):
        """check the validity of line
        syntax: export MODE FILEPATH
        """  
        if len(args) == 0 or (len(args) == 1 and not self._export_dir) or \
                                           args[0] not in self._export_formats:
            self.help_export()
            return False

        if not self._curr_amps:
            print "No process generated, please generate a process!"
            return False

        if len(args) == 1:
            path = os.path.join(self._export_dir, 'SubProcesses')
        else:
            path = args[1]

        if not os.path.isdir(path):
            print "%s is not a valid directory for export file" % path
            return False

        return path
    
    def check_generate(self, line):
        """check the validity of line"""
        
        if len(line) < 1:
            self.help_generate()
            return False

        if len(self._curr_model['particles']) == 0:
            print "No particle list currently active, please create one first!"
            return False

        if len(self._curr_model['interactions']) == 0:
            print "No interaction list currently active," + \
            " please create one first!"
            return False
    
        return True
    
    def check_history(self, args):
        """check the validity of line"""
        
        if len(args) > 1:
            self.help_history()
            return False
        
        if len(args):
            if args[0] not in ['clean', '.']:
                dirpath = os.path.dirname(args[0])
                if not os.path.exists(dirpath):
                    print "invalid path"
                    return False
            elif args[0] == '.' and not self._export_dir:
                print 'no export dir configure'
                return False
        return True
    
    def check_import(self, args):
        """check the validity of line"""
        
        if len(args) != 2 or args[0] not in self._import_formats:
            self.help_import()
            return False
        
        return True
    
    def check_makehtml(self, args):
        """check the validity of the line"""
    
        if len(args) < 1:
            self.help_makehtml()
            return False
        
        if args[0] != 'madevent_v4':
            self.help_makehtml()
            return False
        
        if (len(args) == 1 and not self._export_dir) or \
                        (len(args) > 1 and not os.path.isdir(args[1])):
            self.help_makehtml()
            return False
        return True
    
#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(object):
    """ The Series of help routine for the MadGraphCmd"""
    
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
    
            
    def complete_makehtml(self, text, line, begidx, endidx):
        """ format: makehtlm madevent_v4 [PATH] [--nojpeg]"""
        
        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, ['madevent_v4'])
        
        if text.startswith('-'):
            return self.list_completion(text, ['--nojpeg'])

        # Filename if directory is not given
        if len(split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[2])

    def complete_export(self, text, line, begidx, endidx):
        "Complete the export command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._export_formats)

        # Filename if directory is not given
        if len(split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[2])
    
    def complete_history(self, text, line, begidx, endidx):
        "Complete the add command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.path_completion(text)
        
    def complete_add(self, text, line, begidx, endidx):
        "Complete the add command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._add_opts)
        
    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._display_opts)

    def complete_draw(self, text, line, begidx, endidx):
        "Complete the import command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.path_completion(text)


        #option
        if len(split_arg(line[0:begidx])) >= 2:
            option = ['external=', 'horizontal=', 'add_gap=', 'max_size=', \
                                'contract_non_propagating=']
            return self.list_completion(text, option)

    def complete_load(self, text, line, begidx, endidx):
        "Complete the load command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._save_opts)

        # Filename if directory is not given
        if len(split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[2])

    def complete_save(self, text, line, begidx, endidx):
        "Complete the save command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._save_opts)

        # Filename if directory is not given
        if len(split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[2])

    def complete_setup(self, text, line, begidx, endidx):
        "Complete the setup command"

        possible_option = ['-d ', '-f', '-noclean']
        possible_option2 = ['d ', 'f', 'noclean']
        possible_format = ['madevent_v4']
        #don't propose directory use by MG_ME
        forbidden_name = ['MadGraphII', 'Template', 'pythia-pgs', 'CVS',
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis', 
                            'mg5', 'DECAY', 'EventConverter', 'Models', 
                            'ExRootAnalysis', 'HELAS', 'Transfer_Fct']
        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, possible_format)
        
        #name of the run =>proposes old run name
        if len(split_arg(line[0:begidx])) == 2: 
            content = [name for name in os.listdir(MGME_dir) if \
                                    name not in forbidden_name and \
                                    os.path.isdir(os.path.join(MGME_dir, name))]
            content += ['.', 'auto']
            return self.list_completion(text, content)

        # Returning options
        if len(split_arg(line[0:begidx])) > 2:
            if split_arg(line[0:begidx])[-1] == '-d':
                return self.path_completion(text)
            elif  split_arg(line[0:begidx])[-2] == '-d' and line[-1] != ' ':
                return self.path_completion(text, \
                                             split_arg(line[0:begidx])[-1])
            elif split_arg(line[0:begidx])[-1] == '-':
                return self.list_completion(text, possible_option2)
            else:
                return self.list_completion(text, possible_option)

    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._import_formats)

        # Filename if directory is not given
        if len(split_arg(line[0:begidx])) == 2:
            return self.path_completion(text)

        # Filename if directory is given
        if len(split_arg(line[0:begidx])) == 3:
            return self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[2])

  


    
#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd_Web(CmdExtended, HelpToCmd, CheckValidForCmd):
    """The command line processor of MadGraph"""    
    
    _curr_model = base_objects.Model()
    _model_dir = None
    _curr_amps = diagram_generation.AmplitudeList()
    _curr_matrix_elements = helas_objects.HelasMultiProcess()
    _curr_fortran_model = export_v4.HelasFortranModel()
    _multiparticles = {}

    _display_opts = ['particles', 'interactions', 'processes', 'multiparticles']
    _add_opts = ['process']
    _save_opts = ['model', 'processes']
    _import_formats = ['model_v4', 'proc_v4', 'command']
    _export_formats = ['standalone_v4', 'sa_dirs_v4', 'madevent_v4']
    
    writing_dir = os.path.join(os.environ('MADGRAPH_DATA'),
                               os.environ('REMOTE_USER'))
    
    def __init__(self, *arg, **opt):
        """ add a tracker of the history """

        CmdExtended.__init__(self, *arg, **opt)
        self._generate_info = "" # store the first generated process
        
        # Detect If this script is launched from a valid copy of the Template
        #and if so store this position as standard output directory
        if 'Cards' in os.listdir('.'):
            #Check for ./
            self._export_dir = os.path.split(os.path.realpath('.'))[-1]
        elif 'Cards' in os.listdir('..'):
            #Check for ../
            self._export_dir = os.path.split(os.path.realpath('..'))[-1]
        elif self.stdin != sys.stdin:
            #Check for position defined by the input files
            input_path = os.path.realpath(self.stdin.name).split(os.path.sep)
            if input_path[-2] == 'Cards':
                self._export_dir = os.path.sep.join(input_path[:-2])
            else:
                self._export_dir = None
        else:
            self._export_dir = None
            
    # Add a process to the existing multiprocess definition
    # Generate a new amplitude
    def do_add(self, line):
        """Generate an amplitude for a given process and add to
        existing amplitudes
        syntax:
        """

        args = split_arg(line)
        if not self.check_add(args):
            return

        if args[0] == 'process':
            # Rejoin line
            line = ' '.join(args[1:])
            
            # store the first process (for the perl script)
            if not self._generate_info:
                self._generate_info = line
                
            # Reset Helas matrix elements
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()

            try:
                if line.find(',') == -1:
                    myprocdef = self.extract_process(line)
                else:
                    myprocdef, line = self.extract_decay_chain_process(line)
            except self.MadGraphCmdError as error:
                print "Empty or wrong format process, please try again. Error:" 
                print error
                myprocdef = None

            if myprocdef:

                cpu_time1 = time.time()

                myproc = diagram_generation.MultiProcess(myprocdef)

                for amp in myproc.get('amplitudes'):
                    if amp not in self._curr_amps:
                        self._curr_amps.append(amp)
                    else:
                        print "Warning: Already in processes:"
                        print amp.nice_string_processes()

                cpu_time2 = time.time()

                nprocs = len(myproc.get('amplitudes'))
                ndiags = sum([amp.get_number_of_diagrams() for \
                                  amp in myproc.get('amplitudes')])
                print "%i processes with %i diagrams generated in %0.3f s" % \
                      (nprocs, ndiags, (cpu_time2 - cpu_time1))
                ndiags = sum([amp.get_number_of_diagrams() for \
                                  amp in self._curr_amps])
                print "Total: %i processes with %i diagrams" % \
                      (len(self._curr_amps), ndiags)                
  
    # Define a multiparticle label
    def do_define(self, line):
        """Define a multiparticle"""

        # Particle names always lowercase
        line = line.lower()
        args = split_arg(line)
        if not self.check_define(args):
            return


        label = args[0]
        pdg_list = []

        for part_name in args[1:]:

            mypart = self._curr_model['particles'].find_name(part_name)

            if mypart:
                pdg_list.append(mypart.get_pdg_code())
            else:
                print "No particle %s in model: skipped" % part_name

        if not pdg_list:
            print """Empty or wrong format for multiparticle.
            Please try again."""

        self._multiparticles[label] = pdg_list
    
    # Display
    def do_display(self, line):
        """Display current internal status"""

        args = split_arg(line)
        if not self.check_display(args):
            return

        if args[0] == 'particles':
            print "Current model contains %i particles:" % \
                    len(self._curr_model['particles'])
            part_antipart = [part for part in self._curr_model['particles'] \
                             if not part['self_antipart']]
            part_self = [part for part in self._curr_model['particles'] \
                             if part['self_antipart']]
            for part in part_antipart:
                print part['name'] + '/' + part['antiname'],
            print ''
            for part in part_self:
                print part['name'],
            print ''

        if args[0] == 'interactions':
            print "Current model contains %i interactions" % \
                    len(self._curr_model['interactions'])
            for inter in self._curr_model['interactions']:
                print str(inter['id']) + ':',
                for part in inter['particles']:
                    if part['is_part']:
                        print part['name'],
                    else:
                        print part['antiname'],
                print

        if args[0] == 'processes':
            for amp in self._curr_amps:
                print amp.nice_string()
        if args[0] == 'multiparticles':
            print 'Multiparticle labels:'
            for key in self._multiparticles:
                print key, " = ", self._multiparticles[key]
    
    def do_draw(self, line):
        """ draw the Feynman diagram for the given process """

        args = split_arg(line)
        if not self.check_draw(args):
            return
        

        (options, args) = _draw_parser.parse_args(args)
        options = draw_lib.DrawOption(options)
        start = time.time()
        

        # Collect amplitudes
        amplitudes = diagram_generation.AmplitudeList()

        for amp in self._curr_amps:
            amplitudes.extend(amp.get_amplitudes())            

        for amp in amplitudes:
            filename = os.path.join(args[0], 'diagrams_' + \
                                    amp.get('process').shell_string() + ".eps")
            plot = draw.MultiEpsDiagramDrawer(amp['diagrams'],
                                              filename,
                                              model=self._curr_model,
                                              amplitude='')

            logging.info("Drawing " + \
                         amp.get('process').nice_string())
            plot.draw(opt=options)
            print "Wrote file " + filename

        stop = time.time()
        print 'time to draw', stop - start    
    
    # Export a matrix element
    def do_export(self, line):
        """Export a generated amplitude to file"""

        def generate_matrix_elements(self):
            """Helper function to generate the matrix elements before
            exporting"""

            cpu_time1 = time.time()
            if not self._curr_matrix_elements.get('matrix_elements'):
                self._curr_matrix_elements = \
                             helas_objects.HelasMultiProcess(\
                                           self._curr_amps)
            cpu_time2 = time.time()

            ndiags = sum([len(me.get('diagrams')) for \
                          me in self._curr_matrix_elements.\
                          get('matrix_elements')])

            return ndiags, cpu_time2 - cpu_time1

        # Start of the actual routine

        args = split_arg(line)
        path = self.check_export(args) #if valid return the path of the output
        if not path:
            return 

        ndiags, cpu_time = generate_matrix_elements(self)
        calls = 0

        if args[0] == 'standalone_v4':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                filename = os.path.join(path, 'matrix_' + \
                           me.get('processes')[0].shell_string() + ".f")
                if os.path.isfile(filename):
                    print "Overwriting existing file %s" % filename
                else:
                    print "Creating new file %s" % filename
                calls = calls + files.write_to_file(filename,
                                   export_v4.write_matrix_element_v4_standalone,
                                   me, self._curr_fortran_model)


        if args[0] == 'sa_dirs_v4':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                calls = calls + \
                        export_v4.generate_subprocess_directory_v4_standalone(\
                            me, self._curr_fortran_model, path)

        if args[0] == 'madevent_v4':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                calls = calls + \
                        export_v4.generate_subprocess_directory_v4_madevent(\
                            me, self._curr_fortran_model, path)
            
            card_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                                              'procdef_mg5.dat')
            if self._generate_info:
                export_v4.write_procdef_mg5(card_path,
                                os.path.split(self._model_dir)[-1],
                                self._generate_info)
                
        print ("Generated helas calls for %d subprocesses " + \
              "(%d diagrams) in %0.3f s") % \
              (len(self._curr_matrix_elements.get('matrix_elements')),
               ndiags, cpu_time)

        print "Wrote %d helas calls" % calls

        # Replace the amplitudes with the actual amplitudes from the
        # matrix elements, which allows proper diagram drawing also of
        # decay chain processes
        self._curr_amps = diagram_generation.AmplitudeList(\
               [me.get('base_amplitude') for me in \
                self._curr_matrix_elements.get('matrix_elements')])
    
    
    # Generate a new amplitude
    def do_generate(self, line):
        """Generate an amplitude for a given process"""

        if not self.check_generate(line):
            return

        # Reset Helas matrix elements
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        self._generate_info = line

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
            self._curr_amps = myproc.get('amplitudes')
            cpu_time2 = time.time()

            nprocs = len(self._curr_amps)
            ndiags = sum([amp.get_number_of_diagrams() for \
                              amp in self._curr_amps])
            print "%i processes with %i diagrams generated in %0.3f s" % \
                  (nprocs, ndiags, (cpu_time2 - cpu_time1))

        else:
            print "Empty or wrong format process, please try again."    
    
    
    def extract_process(self, line, proc_number = 0):
        """Extract a process definition from a string. Returns
        a ProcessDefinition."""

        # Use regular expressions to extract s-channel propagators,
        # forbidden s-channel propagators/particles, coupling orders
        # and process number, starting from the back

        # Start with process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(line)
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            line = proc_number_re.group(1) + \
                   proc_number_re.group(3)

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
        slash = line.find("/")
        dollar = line.find("$")
        forbidden_particles = ""
        if slash > 0:
            if dollar > slash:
                forbidden_particles_re = re.match("^(.+)\s*/\s*(.+\s*)(\$.*)$", line)
            else:
                forbidden_particles_re = re.match("^(.+)\s*/\s*(.+\s*)$", line)
            if forbidden_particles_re:
                forbidden_particles = forbidden_particles_re.group(2)
                line = forbidden_particles_re.group(1)
                if len(forbidden_particles_re.groups()) > 2:
                    line = line + forbidden_particles_re.group(3)

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

        args = split_arg(line)

        myleglist = base_objects.MultiLegList()
        state = False

        # Extract process
        for part_name in args:

            if part_name == '>':
                if not myleglist:
                    raise self.MadGraphCmdError, \
                          "No final state particles"
                state = True
                continue

            mylegids = []
            if part_name in self._multiparticles:
                mylegids.extend(self._multiparticles[part_name])
            else:
                mypart = self._curr_model['particles'].find_name(part_name)
                if mypart:
                    mylegids.append(mypart.get_pdg_code())

            if mylegids:
                myleglist.append(base_objects.MultiLeg({'ids':mylegids,
                                                        'state':state}))
            else:
                raise self.MadGraphCmdError, \
                      "No particle %s in model" % part_name

        if filter(lambda leg: leg.get('state') == True, myleglist):
            # We have a valid process

            # Now extract restrictions
            forbidden_particle_ids = []
            forbidden_schannel_ids = []
            required_schannel_ids = []

            #decay_process = len(filter(lambda leg: \
            #                           leg.get('state') == False,
            #                           myleglist)) == 1

            if forbidden_particles:
                args = split_arg(forbidden_particles)
                for part_name in args:
                    if part_name in self._multiparticles:
                        forbidden_particle_ids.extend( \
                                               self._multiparticles[part_name])
                    else:
                        mypart = self._curr_model['particles'].find_name( \
                                                                      part_name)
                        if mypart:
                            forbidden_particle_ids.append(mypart.get_pdg_code())

            if forbidden_schannels:
                args = split_arg(forbidden_schannels)
                for part_name in args:
                    if part_name in self._multiparticles:
                        forbidden_schannel_ids.extend(\
                                               self._multiparticles[part_name])
                    else:
                        mypart = self._curr_model['particles'].find_name(\
                                                                      part_name)
                        if mypart:
                            forbidden_schannel_ids.append(mypart.get_pdg_code())

            if required_schannels:
                args = split_arg(required_schannels)
                for part_name in args:
                    if part_name in self._multiparticles:
                        required_schannel_ids.extend(\
                                               self._multiparticles[part_name])
                    else:
                        mypart = self._curr_model['particles'].find_name(\
                                                                      part_name)
                        if mypart:
                            required_schannel_ids.append(mypart.get_pdg_code())

            return \
                base_objects.ProcessDefinition({'legs': myleglist,
                                'model': self._curr_model,
                                'id': proc_number,
                                'orders': orders,
                                'forbidden_particles': forbidden_particle_ids,
                                'forbidden_s_channels': forbidden_schannel_ids,
                                'required_s_channels': required_schannel_ids
                                 })
        #                       'is_decay_chain': decay_process\
        else:
            return None

    def extract_decay_chain_process(self, line, level_down=False):
        """Recursively extract a decay chain process definition from a
        string. Returns a ProcessDefinition."""

        # Start with process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(line)
        proc_number = 0
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            line = proc_number_re.group(1) + \
                   proc_number_re.group(3)
            print line
            
        index_comma = line.find(",")
        index_par = line.find(")")
        min_index = index_comma
        if index_par > -1 and (index_par < min_index or min_index == -1):
            min_index = index_par
        
        if min_index > -1:
            core_process = self.extract_process(line[:min_index], proc_number)
        else:
            core_process = self.extract_process(line, proc_number)

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
                                                             level_down=True)
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
                    raise self.MadGraphCmdError, \
                      "Missing ending parenthesis for decay process"

                if index_par < index_comma:
                    line = line[index_par + 1:]
                    level_down = False
                    break

        if level_down:
            index_par = line.find(')')
            if index_par == -1:
                raise self.MadGraphCmdError, \
                      "Missing ending parenthesis for decay process"
            line = line[index_par + 1:]
            
        # Return the core process (ends recursion when there are no
        # more decays)
        return core_process, line
    
    # Write the list of command line use in this session
    def do_history(self, line):
        """write in a file the suite of command that was used"""
        
        args = split_arg(line)
        if not self.check_history(args):
            return

        if len(args) == 0:
            print '\n'.join(self.history)
            return False
        elif args[0] == 'clean':
            self.history = []
            print 'history is cleaned'
            return False
        elif args[0] == '.':
            output_file = os.path.join(self._export_dir, 'Cards', \
                                                            'proc_card_mg5.dat')
            output_file = open(output_file, 'w')
        else:
            output_file = open(args[0], 'w')
            
        
        # Define a simple header for the file
        creation_time = time.asctime() 
        time_info = \
        '#     automaticaly generated the %s%s*\n' % (creation_time, ' ' * \
                                                      (26 - len(creation_time)))
        text = \
        '#***********************************************************\n' + \
        '#                         MadGraph 5                       *\n' + \
        '#                                                          *\n' + \
        "#                 *                       *                *\n" + \
        "#                   *        * *        *                  *\n" + \
        "#                     * * * * 5 * * * *                    *\n" + \
        "#                   *        * *        *                  *\n" + \
        "#                 *                       *                *\n" + \
        "#                                                          *\n" + \
        "#                                                          *\n" + \
        "#    The MadGraph Development Team - Please visit us at    *\n" + \
        "#    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        '#                                                          *\n' + \
        '#***********************************************************\n' + \
        '#                                                          *\n' + \
        '#               Command File for MadGraph 5                *\n' + \
        '#                                                          *\n' + \
        '#     run as ./bin/mg5  filename                           *\n' + \
        time_info + \
        '#                                                          *\n' + \
        '#***********************************************************\n'
        
        #Avoid repetition of header
        if self.history[0] == '#'+'*' * 59:
            text=''
        # Add the comand used 
        text += '\n'.join(self.history) + '\n' 
        
        #write this information in a file
        output_file.write(text)
        output_file.close()
    
    # Import files
    def do_import(self, line):
        """Import files with external formats"""

        def import_v4file(self, filepath):
            """Helper function to load a v4 file from file path filepath"""
            filename = os.path.basename(filepath)
            if filename.endswith('particles.dat'):
                self._curr_model.set('particles',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_particles_v4))
                print "%d particles imported" % \
                      len(self._curr_model['particles'])
                return True
            elif filename.endswith('interactions.dat'):
                if len(self._curr_model['particles']) == 0:
                    print "No particle list currently active,",
                    print "please create one first!"
                    return False
                self._curr_model.set('interactions',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_interactions_v4,
                                            self._curr_model['particles']))
                print "%d interactions imported" % \
                      len(self._curr_model['interactions'])
                return True
           
            #not valid File
            return False

        args = split_arg(line)
        if not self.check_import(args):
            return
        
        if args[0] == 'model_v4':
            # Check for a file
            if os.path.isfile(args[1]):
                suceed = import_v4file(self, args[1])
                if not suceed:
                    print "%s is not a valid v4 file name" % \
                                        os.path.basename(args[1])
                else:
                    self._model_dir = os.path.dirname(args[1])
                return
            
            # Check for a valid directory
            if os.path.isdir(args[1]):
                self._model_dir = args[1]
            elif MGME_dir and os.path.isdir(os.path.join(MGME_dir, 'Models', \
                                                                      args[1])):
                self._model_dir = os.path.join(MGME_dir, 'Models', args[1])
            elif not MGME_dir:
                print "Path %s is not a valid pathname" % args[1]
                print "and no MG_ME installation detected in other to search in Models"
                return False                
            else:
                print "Path %s is not a valid pathname" % args[1]
                return False
            
            #Load the directory
            if os.path.exists(os.path.join(self._model_dir, 'model.pkl')):
                self.do_load('model %s' % os.path.join(self._model_dir, \
                                                                   'model.pkl'))
                return
            files_to_import = ('particles.dat', 'interactions.dat')
            for filename in files_to_import:
                if os.path.isfile(os.path.join(self._model_dir, filename)):
                    import_v4file(self, os.path.join(self._model_dir, \
                                                                      filename))
                else:
                    print "%s files doesn't exist in %s directory" % \
                                        (filename, os.path.basename(args[1]))
            #save model for next usage
            self.do_save('model %s ' % os.path.join(self._model_dir, \
                                                                   'model.pkl'))
        
        elif args[0] == 'proc_v4':
            if len(args) == 1 and self._export_dir:
                proc_card = os.path.join(self._export_dir, '')            
            elif len(args) == 2:
                proc_card = args[1]
            else:
                logging.error('No default directory are setup')
            
            # Check the status of export and try to use file position is no
            #self._export dir are define
            self.check_for_export_dir(proc_card) 
            #convert and excecute the card
            self.import_mg4_proc_card(proc_card)   
                                     
        elif args[0] == 'command':
            if not os.path.isfile(args[1]):
                print "Path %s is not a valid pathname" % args[1]
            else:
                # Check the status of export and try to use file position is no
                #self._export dir are define
                self.check_for_export_dir(args[1])
                # Execute the card
                self.import_mg5_proc_card(args[1])    
    
    def import_mg4_proc_card(self, filepath):
        """ read a V4 proc card, convert it and run it in mg5"""
        
        # change the status of this line in the history -> pass in comment
        self.history[-1] = '#%s' % self.history[-1]
         
        # read the proc_card.dat
        reader = files.read_from_file(filepath, import_v4.read_proc_card_v4)
        
        if MGME_dir:
            #model_dir = os.path.join(MGME_dir, 'Models')
            line = self.exec_cmd('import model_v4 %s' % (reader.model))
        else:
            logging.error('No MG_ME installation detected')
            return    


        # Now that we have the model we can split the information
        lines = reader.extract_command_lines(self._curr_model)
        for line in lines:
            self.exec_cmd(line)
            
        return 

    def import_mg5_proc_card(self, filepath):
        # change the status of this line in the history -> pass in comment
        self.history[-1] = '#%s' % self.history[-1]
 
        # Read the lines of the file and execute them
        for line in CmdFile(filepath):
            #remove pointless spaces and \n
            line = line.replace('\n', '').strip()
            # execute the line if this one is not empty
            if line:
                self.exec_cmd(line)
        return
    
    def check_for_export_dir(self, filepath):
        """Check if the files is in a valid export directory and assing it to
        export path if if is"""
        
        # keep previous if a previous one is defined
        if self._export_dir:
            return
        
        path_split = filepath.split(os.path.sep)
        if path_split[-2] == 'Cards':
            self._export_dir = os.path.sep.join(path_split[:-2])    
    

    def do_load(self, line):
        """Load information from file"""

        args = split_arg(line)
        if len(args) != 2:
            self.help_load()
            return False

        cpu_time1 = time.time()
        if args[0] == 'model':
            self._curr_model = save_load_object.load_from_file(args[1])
            #save_model.save_model(args[1], self._curr_model)
            if isinstance(self._curr_model, base_objects.Model):
                cpu_time2 = time.time()
                print "Loaded model from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1)
            else:
                print 'Error: Could not load model from file ', args[1]
        elif args[0] == 'processes':
            self._curr_amps = save_load_object.load_from_file(args[1])
            if isinstance(self._curr_amps, diagram_generation.AmplitudeList):
                cpu_time2 = time.time()
                print "Loaded processes from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1)
                if self._curr_amps and not self._curr_model.get('name'):
                    self._curr_model = self._curr_amps[0].\
                                        get('process').get('model')
                    print "Model set from process."
            else:
                print 'Error: Could not load processes from file ', args[1]
        else:
            self.help_save()
    
    def do_save(self, line):
        """Save information to file"""

        args = split_arg(line)
        if len(args) != 2:
            self.help_save()
            return False

        if args[0] == 'model':
            if self._curr_model:
                #save_model.save_model(args[1], self._curr_model)
                if save_load_object.save_to_file(args[1], self._curr_model):
                    print 'Saved model to file ', args[1]
            else:
                print 'No model to save!'
        elif args[0] == 'processes':
            if self._curr_amps:
                if save_load_object.save_to_file(args[1], self._curr_amps):
                    print 'Saved processes to file ', args[1]
            else:
                print 'No processes to save!'
        else:
            self.help_save()
            
            
    def do_makehtml(self, line):
        """ make the html output for a MAdevent directory """
        
        args = split_arg(line)
        if not self.check_makehtml(args):
            return
                   
        
        if self._export_dir:
            dir_path = self._export_dir
        else: 
            dir_path = args[1]
            
        #look if the user ask to bypass the jpeg creation
        if '--nojpeg' in args:
            makejpg = False
        else:
            makejpg = True
        
        export_v4.create_v4_webpage(dir_path, makejpg)
        os.system('touch %s/done' % os.path.join(dir_path,'SubProcesses'))


    def do_setup(self, line):
        """Initialize a new Template or reinitialize one"""
        
        args = split_arg(line)
        clean = '-noclean' not in args
        force = '-f' in args 
        dir = '-d' in args
        if dir:
            mgme_dir = args[args.find('-d') + 1]
        else:
            mgme_dir = MGME_dir
                        
        if len(args) < 2:
            self.help_setup()
            return False
        
        if not self._model_dir:
            print 'No model found. Please import a model first and then retry'
            print '  for example do : import model_v4 sm'
            return False
        
        # Check for special directory treatment
        if args[1] == '.':
            if self._export_dir:
                args[1] = self._export_dir
            else:
                print 'No possible working directory are detected'
                self.help_setup()    
                return False
        elif args[1] == 'auto':
            name_dir = lambda i: 'PROC_%s_%s' % \
                                        (os.path.split(self._model_dir)[-1], i)
            auto_path = lambda i: os.path.join(self.writing_dir, name_dir(i))     
            
            for i in range(500):
                if os.path.isdir(auto_path(i)):
                    continue
                else:
                    args[1] = name_dir(i) 
                    break
                
        dir_path = os.path.join(self.writing_dir, args[1])
        if not force and os.path.isdir(dir_path):
            print 'INFO: directory %s already exists.' % args[1]
            if clean:
                print 'If you continue this directory will be cleaned'

            answer = raw_input('Do you want to continue? [y/n]')
            if answer != 'y':
                print 'stop'
                return False

        export_v4.copy_v4template(mgme_dir, dir_path, self._model_dir, clean)
        # Import the model
        print 'import model files %s in directory %s' % \
                       (os.path.basename(self._model_dir), args[1])        
        export_v4.export_model(self._model_dir, dir_path)
        self._export_dir = dir_path
        
 
#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(MadGraphCmd_Web, CompleteForCmd):
    """The command line processor of MadGraph""" 
    
    writing_dir = MGME_dir
    
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
        "*           W E L C O M E  to  M A D G R A P H  5          *\n" + \
        "*                                                          *\n" + \
        "*                                                          *\n" + \
        "*                 *                       *                *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                     * * * * 5 * * * *                    *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                 *                       *                *\n" + \
        "*                                                          *\n" + \
        info_line + \
        "*                                                          *\n" + \
        "*    The MadGraph Development Team - Please visit us at    *\n" + \
        "*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*                                                          *\n" + \
        "************************************************************"

    # Access to shell
    def do_shell(self, line):
        "Run a shell command"

        if line.strip() is '':
            self.help_shell()
        else:
            print "running shell command:", line
            subprocess.call(line, shell=True)
   
    
    
    
#===============================================================================
# 
#===============================================================================
class CmdFile(file):
    """ a class for command input file -in order to debug cmd \n problem"""
    
    def __init__(self, name, opt):
        
        file.__init__(self, name, 'rU')
        self.text = file.read(self)
        self.close()
        self.lines = self.text.split('\n')
    
    def readline(self, *arg, **opt):
        """readline method treating correctly a line whithout \n at the end
           (add it)
        """
        if self.lines:
            line = self.lines.pop(0)
        else:
            return ''
        
        if line.endswith('\n'):
            return line
        else:
            return line + '\n'    
    
  
#===============================================================================
# Draw Command Parser
#=============================================================================== 
_usage =  "draw FILEPATH [options]\n" + \
         "-- draw the diagrams in eps format\n" + \
         "   Files will be FILEPATH/diagrams_\"process_string\".eps \n" + \
         "   Example: draw plot_dir . \n"
_draw_parser = optparse.OptionParser(usage=_usage)
_draw_parser.add_option("", "--horizontal", default=False, 
                   action='store_true', help="force S-channel to be horizontal")
_draw_parser.add_option("", "--external", default=0, type='float',
                    help="authorizes external particles to end at top or " + \
                    "bottom of diagram. If bigger than zero this tune the " + \
                    "length of those line.")
_draw_parser.add_option("", "--max_size", default=1.5, type='float',
                         help="this forbids external line bigger than max_size")
_draw_parser.add_option("", "--non_propagating", default=True, \
                          dest="contract_non_propagating", action='store_false', 
                          help= "avoid contractions of non propagating lines") 
_draw_parser.add_option("", "--add_gap", default=0, type='float', \
                          help= "set the x-distance between external particles")  
  
    
    
    
#===============================================================================
# __main__
#===============================================================================

if __name__ == '__main__':
    
    run_option = sys.argv
    if len(run_option) > 1:
        # The first argument of sys.argv is the name of the program
        input_file = open(run_option[1], 'rU')
        cmd_line = MadGraphCmd(stdin=input_file)
        cmd_line.use_rawinput = False #put it in non interactive mode
        cmd_line.cmdloop()
    else:
        # Interactive mode
        MadGraphCmd().cmdloop()    