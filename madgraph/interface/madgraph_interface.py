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
"""A user friendly command line interface to access MadGraph features at LO.
   Uses the cmd package for command interpretation and tab completion.
"""
from __future__ import division

import atexit
import cmath
import logging
import optparse
import os
import pydoc
import re
import signal
import subprocess
import sys
import shutil
import StringIO
import traceback
import time
import urllib
        

#usefull shortcut
pjoin = os.path.join

try:
    import readline
    GNU_SPLITTING = ('GNU' in readline.__doc__)
except:
    GNU_SPLITTING = True

import aloha
import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error


import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.drawing as draw_lib
import madgraph.core.helas_objects as helas_objects

import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.export_cpp as export_cpp
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.interface.extended_cmd as cmd
import madgraph.interface.tutorial_text as tutorial_text
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.interface.madevent_interface as madevent_interface

import madgraph.various.process_checks as process_checks
import madgraph.various.banner as banner_module
import madgraph.various.misc as misc
import madgraph.various.cluster as cluster

import models as ufomodels
import models.import_ufo as import_ufo
import models.write_param_card as param_writer
import models.check_param_card as check_param_card
import models.model_reader as model_reader

import aloha.aloha_fct as aloha_fct
import aloha.create_aloha as create_aloha
import aloha.aloha_lib as aloha_lib

import mg5decay.decay_objects as decay_objects

# Special logger for the Cmd Interface
logger = logging.getLogger('cmdprint') # -> stdout
logger_mg = logging.getLogger('madgraph') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr
logger_tuto = logging.getLogger('tutorial') # -> stdout include instruction in  
                                            #order to learn MG5

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
    """Particularisation of the cmd command for MG5"""

    #suggested list of command
    next_possibility = {
        'start': ['import model ModelName', 'import command PATH',
                      'import proc_v4 PATH', 'tutorial'],
        'import model' : ['generate PROCESS','define MULTIPART PART1 PART2 ...', 
                                   'display particles', 'display interactions'],
        'define': ['define MULTIPART PART1 PART2 ...', 'generate PROCESS', 
                                                    'display multiparticles'],
        'generate': ['add process PROCESS','output [OUTPUT_TYPE] [PATH]','display diagrams'],
        'add process':['output [OUTPUT_TYPE] [PATH]', 'display processes'],
        'output':['launch','open index.html','history PATH', 'exit'],
        'display': ['generate PROCESS', 'add process PROCESS', 'output [OUTPUT_TYPE] [PATH]'],
        'import proc_v4' : ['launch','exit'],
        'launch': ['open index.html','exit'],
        'tutorial': ['generate PROCESS', 'import model MODEL', 'help TOPIC']
    }
    
    debug_output = 'MG5_debug'
    error_debug = 'Please report this bug on https://bugs.launchpad.net/madgraph5\n'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'
    
    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/madgraph5\n'

    keyboard_stop_msg = """stopping all operation
            in order to quit mg5 please enter exit"""
    
    # Define the Error Class # Define how error are handle
    InvalidCmd = madgraph.InvalidCmd
    ConfigurationError = MadGraph5Error

    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
        # If possible, build an info line with current version number 
        # and date, from the VERSION text file
        info = misc.get_pkg_info()
        info_line = ""

        if info.has_key('version') and  info.has_key('date'):
            len_version = len(info['version'])
            len_date = len(info['date'])
            if len_version + len_date < 30:
                info_line = "#*         VERSION %s %s %s         *\n" % \
                            (info['version'],
                            (30 - len_version - len_date) * ' ',
                            info['date'])

        # Create a header for the history file.
        # Remember to fill in time at writeout time!
        self.history_header = \
        '#************************************************************\n' + \
        '#*                        MadGraph 5                        *\n' + \
        '#*                                                          *\n' + \
        "#*                *                       *                 *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                    * * * * 5 * * * *                     *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                *                       *                 *\n" + \
        "#*                                                          *\n" + \
        "#*                                                          *\n" + \
        info_line + \
        "#*                                                          *\n" + \
        "#*    The MadGraph Development Team - Please visit us at    *\n" + \
        "#*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        '#*                                                          *\n' + \
        '#************************************************************\n' + \
        '#*                                                          *\n' + \
        '#*               Command File for MadGraph 5                *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/mg5  filename                           *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        if info_line:
            info_line = info_line[1:]

        logger.info(\
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
        "*           Type 'tutorial' to learn how MG5 works         *\n" + \
        "*                                                          *\n" + \
        "************************************************************")
        
        cmd.Cmd.__init__(self, *arg, **opt)
    
    def postcmd(self,stop, line):
        """ finishing a command
        This looks if the command add a special post part.
        This looks if we have to write an additional text for the tutorial."""
        
        stop = super(CmdExtended, self).postcmd(stop, line)   
        # Print additional information in case of routines fails
        if stop == False:
            return False
        
        args=line.split()
        # Return for empty line
        if len(args)==0:
            return stop
        
        # try to print linked to the first word in command 
        #as import_model,... if you don't find then try print with only
        #the first word.
        if len(args)==1:
            command=args[0]
        else:
            command = args[0]+'_'+args[1].split('.')[0]
        
        try:
            logger_tuto.info(getattr(tutorial_text, command).replace('\n','\n\t'))
        except:
            try:
                logger_tuto.info(getattr(tutorial_text, args[0]).replace('\n','\n\t'))
            except:
                pass
        
        return stop

     
    def get_history_header(self):
        """return the history header""" 
        return self.history_header % misc.get_time_info()

#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(cmd.HelpCmd):
    """ The Series of help routine for the MadGraphCmd"""    
    
    def help_save(self):
        logger.info("syntax: save %s FILENAME" % "|".join(self._save_opts))
        logger.info("-- save information as file FILENAME")
        logger.info("   FILENAME is optional for saving 'options'.")
        logger.info('   By default it uses ./input/mg5_configuration.txt')
        logger.info('   If you put "global" for FILENAME it will use ~/.mg5/mg5_configuration.txt')
        logger.info('   If this files exists, it is uses by all MG5 on the system but continues')
        logger.info('   to read the local options files.')

    def help_load(self):
        logger.info("syntax: load %s FILENAME" % "|".join(self._save_opts))
        logger.info("-- load information from file FILENAME")

    def help_import(self):
        logger.info("syntax: import " + "|".join(self._import_formats) + \
              " FILENAME")
        logger.info("-- imports file(s) in various formats")
        logger.info("")
        logger.info("   import model MODEL[-RESTRICTION] [--modelname]:")
        logger.info("      Import a UFO model.")
        logger.info("      MODEL should be a valid UFO model name")
        logger.info("      Model restrictions are specified by MODEL-RESTRICTION")
        logger.info("        with the file restrict_RESTRICTION.dat in the model dir.")
        logger.info("        By default, restrict_default.dat is used.")
        logger.info("        Specify model_name-full to get unrestricted model.")
        logger.info("      '--modelname' keeps the original particle names for the model")
        logger.info("")
        logger.info("   import model_v4 MODEL [--modelname] :")
        logger.info("      Import an MG4 model.")
        logger.info("      Model should be the name of the model")
        logger.info("      or the path to theMG4 model directory")
        logger.info("      '--modelname' keeps the original particle names for the model")
        logger.info("")
        logger.info("   import proc_v4 [PATH] :"  )
        logger.info("      Execute MG5 based on a proc_card.dat in MG4 format.")
        logger.info("      Path to the proc_card is optional if you are in a")
        logger.info("      madevent directory")
        logger.info("")
        logger.info("   import command PATH :")
        logger.info("      Execute the list of command in the file at PATH")
        logger.info("")
        logger.info("   import banner PATH  [--no_launch]:")
        logger.info("      Rerun the exact same run define in the valid banner.")
 
    def help_install(self):
        logger.info("syntax: install " + "|".join(self._install_opts))
        logger.info("-- Download the last version of the program and install it")
        logger.info("   localy in the current Madgraph version. In order to have")
        logger.info("   a sucessfull instalation, you will need to have up-to-date")
        logger.info("   F77 and/or C and Root compiler.")
        logger.info(" ")
        logger.info("   \"install update\" check if your MG5 installation is the latest one.")
        logger.info("   If not it load the difference between your current version and the latest one,")
        logger.info("   and apply it to the code. Two options are available for this command:")
        logger.info("     -f: didn't ask for confirmation if it founds an update.")
        logger.info("     --timeout=: Change the maximum time allowed to reach the server.")
        
    def help_display(self):
        logger.info("syntax: display " + "|".join(self._display_opts))
        logger.info("-- display a the status of various internal state variables")
        logger.info("   for particles/interactions you can specify the name or id of the")
        logger.info("   particles/interactions to receive more details information.")
        logger.info("   Example: display particles e+.")
        logger.info("   For \"checks\", can specify only to see failed checks.")
        logger.info("   For \"diagrams\", you can specify where the file will be written.")
        logger.info("   Example: display diagrams ./")
        
        
    def help_launch(self):
        """help for launch command"""
        _launch_parser.print_help()

    def help_tutorial(self):
        logger.info("syntax: tutorial [" + "|".join(self._tutorial_opts) + "]")
        logger.info("-- start/stop the tutorial mode")

    def help_open(self):
        logger.info("syntax: open FILE  ")
        logger.info("-- open a file with the appropriate editor.")
        logger.info('   If FILE belongs to index.html, param_card.dat, run_card.dat')
        logger.info('   the path to the last created/used directory is used')
        logger.info('   The program used to open those files can be chosen in the')
        logger.info('   configuration file ./input/mg5_configuration.txt')
        
    def help_customize_model(self):
        logger.info("syntax: customize_model --save=NAME")
        logger.info("--  Open an invite where you options to tweak the model.")
        logger.info("    If you specify the option --save=NAME, this tweak will be")
        logger.info("    available for future import with the command 'import model XXXX-NAME'")
        
    def help_output(self):
        logger.info("syntax: output [" + "|".join(self._export_formats) + \
                    "] [path|.|auto] [options]")
        logger.info("-- Output any generated process(es) to file.")
        logger.info("   mode: Default mode is madevent. Default path is \'.\' or auto.")
        logger.info("   - If mode is madevent, create a MadEvent process directory.")
        logger.info("   - If mode is standalone, create a Standalone directory")
        logger.info("   - If mode is matrix, output the matrix.f files for all")
        logger.info("     generated processes in directory \"path\".")
        logger.info("   - If mode is standalone_cpp, create a standalone C++")
        logger.info("     directory in \"path\".")
        logger.info("   - If mode is pythia8, output all files needed to generate")
        logger.info("     the processes using Pythia 8. The files are written in")
        logger.info("     the Pythia 8 directory (default).")
        logger.info("     NOTE: The Pythia 8 directory is set in the ./input/mg5_configuration.txt")
        logger.info("   - If mode is aloha: Special syntax output:")
        logger.info("     syntax: aloha [ROUTINE] [--options]" )
        logger.info("     valid options for aloha output are:")
        logger.info("      --format=Fortran|Python|Cpp : defining the output language")
        logger.info("      --output= : defining output directory")
        logger.info("   path: The path of the process directory.")
        logger.info("     If you put '.' as path, your pwd will be used.")
        logger.info("     If you put 'auto', an automatic directory PROC_XX_n will be created.")
        logger.info("   options:")
        logger.info("      -f: force cleaning of the directory if it already exists")
        logger.info("      -d: specify other MG/ME directory")
        logger.info("      -noclean: no cleaning performed in \"path\".")
        logger.info("      -nojpeg: no jpeg diagrams will be generated.")
        logger.info("      -name: the postfix of the main file in pythia8 mode.")
        logger.info("   Examples:")
        logger.info("       output")
        logger.info("       output standalone MYRUN -f")
        logger.info("       output pythia8 ../pythia8/ -name qcdprocs")
        
    def help_check(self):

        logger.info("syntax: check [" + "|".join(self._check_opts) + "] [param_card] process_definition [--energy=]")
        logger.info("-- check a process or set of processes. Options:")
        logger.info("full: Perform all three checks described below:")
        logger.info("   permutation, gauge and lorentz_invariance.")
        logger.info("permutation: Check that the model and MG5 are working")
        logger.info("   properly by generating permutations of the process and")
        logger.info("   checking that the resulting matrix elements give the")
        logger.info("   same value.")
        logger.info("gauge: Check that processes with massless gauge bosons")
        logger.info("   are gauge invariant")
        logger.info("lorentz_invariance: Check that the amplitude is lorentz")
        logger.info("   invariant by comparing the amplitiude in different frames")        
        logger.info("If param_card is given, that param_card is used instead")
        logger.info("   of the default values for the model.")
        logger.info("For process syntax, please see help generate")
        logger.info("the options --energy allows to change the \sqrt(S)")

    def help_generate(self):

        logger.info("syntax: generate INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2 @N")
        logger.info("-- generate diagrams for a given process")
        logger.info("   Syntax example: l+ vl > w+ > l+ vl a $ z / a h QED=3 QCD=0 @1")
        logger.info("   Alternative required s-channels can be separated by \"|\":")
        logger.info("   b b~ > W+ W- | H+ H- > ta+ vt ta- vt~")
        logger.info("   If no coupling orders are given, MG5 will try to determine")
        logger.info("   orders to ensure maximum number of QCD vertices.")
        logger.info("   Note that if there are more than one non-QCD coupling type,")
        logger.info("   coupling orders need to be specified by hand.")
        logger.info("Decay chain syntax:")
        logger.info("   core process, decay1, (decay2, (decay2', ...)), ...  etc")
        logger.info("   Example: p p > t~ t QED=0, (t~ > W- b~, W- > l- vl~), t > j j b @2")
        logger.info("   Note that identical particles will all be decayed.")
        logger.info("To generate a second process use the \"add process\" command")

    def help_compute_widths(self):
        logger.info("syntax: calculate_width PART [other particles] [OPTIONS]")
        logger.info("Generate amplitudes for decay width calculation.")
        logger.info("  PART: name of the particle you want to calculate width")
        logger.info("        you can enter either the name or pdg code.\n")
        logger.info("  Various options:\n")
        logger.info("  --precision=XX: required precision for width")
        logger.info("        if an integer is provide it means the max number of decay products")
        logger.info("        default: 0.001")
        logger.info("  --path=XX: path for param_card (if not default)")
        logger.info("  --output=XX: path where to writte the resulting card. ")
        logger.info("        default: overwritte current default card.")
        logger.info("")
        logger.info(" example: calculate_width h --precision=2 --output=./param_card")
        
    def help_decay_diagram(self):
        logger.info("syntax: decay_diagram PART PRECISION [PARAM_CARD_PATH]")
        logger.info("Generate amplitudes for decay width calculation.")
        logger.info("  PART: name of the particle you want to calculate width")
        logger.info("  PRECISION: required precision for width")
        logger.info("             if an integer is provide it means the max number of decay products")
        logger.info("  PARAM_CARD_PATH: path for param_card (if not default)")
        logger.info("")
        logger.info(" example: decay_diagram h 2 ./param_card")
        
    def help_add(self):

        logger.info("syntax: add process INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2")
        logger.info("-- generate diagrams for a process and add to existing processes")
        logger.info("   Syntax example: l+ vl > w+ > l+ vl a $ z / a h QED=3 QCD=0 @1")
        logger.info("   Alternative required s-channels can be separated by \"|\":")
        logger.info("   b b~ > W+ W- | H+ H- > ta+ vt ta- vt~")
        logger.info("   If no coupling orders are given, MG5 will try to determine")
        logger.info("   orders to ensure maximum number of QCD vertices.")
        logger.info("Decay chain syntax:")
        logger.info("   core process, decay1, (decay2, (decay2', ...)), ...  etc")
        logger.info("   Example: p p > t~ t QED=0, (t~ > W- b~, W- > l- vl~), t > j j b @2")
        logger.info("   Note that identical particles will all be decayed.")

    def help_define(self):
        logger.info("syntax: define multipart_name [=] part_name_list")
        logger.info("-- define a multiparticle. part_name_list can include multiparticles.")
        logger.info("   Example: define p = g u u~ c c~ d d~ s s~ b b~")
        logger.info("   Special syntax: Use | for OR (used for required s-channels)")
        logger.info("   Special syntax: Use / to remove particles. Example: define q = p / g")

    def help_set(self):
        logger.info("syntax: set %s argument|default" % "|".join(self._set_options))
        logger.info("-- set options for generation or output.")
        logger.info("   group_subprocesses True/False/Auto: ")
        logger.info("     (default Auto) Smart grouping of subprocesses into ")
        logger.info("     directories, mirroring of initial states, and ")
        logger.info("     combination of integration channels.")
        logger.info("     Example: p p > j j j w+ gives 5 directories and 184 channels")
        logger.info("     (cf. 65 directories and 1048 channels for regular output)")
        logger.info("     Auto means False for decay computation and True for") 
        logger.info("     collisions.")
        logger.info("   ignore_six_quark_processes multi_part_label")
        logger.info("     (default none) ignore processes with at least 6 of any")
        logger.info("     of the quarks given in multi_part_label.")
        logger.info("     These processes give negligible contribution to the")
        logger.info("     cross section but have subprocesses/channels.")
        logger.info("   stdout_level DEBUG|INFO|WARNING|ERROR|CRITICAL")
        logger.info("     change the default level for printed information")
        logger.info("   fortran_compiler NAME")
        logger.info("      (default None) Force a specific fortran compiler.")
        logger.info("      If None, it tries first g77 and if not present gfortran.")
        logger.info("   gauge unitary|Feynman")        
        logger.info("      (default unitary) choose the gauge.")
        logger.info("   complex_mass_scheme True|False")        
        logger.info("      (default False) Set complex mass scheme.")
        logger.info("   timeout VALUE")
        logger.info("      (default 20) Seconds allowed to answer questions.")
        logger.info("      Note that pressing tab always stops the timer.")
        logger.info("   cluster_temp_path PATH")
        logger.info("      (default None) [Used in Madevent Output]")
        logger.info("      Allow to perform the run in PATH directory")
        logger.info("      This allow to not run on the central disk. This is not used")
        logger.info("      by condor cluster (since condor has it's own way to prevent it).")
       
#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(cmd.CheckCmd):
    """ The Series of help routine for the MadGraphCmd"""
    
    class RWError(MadGraph5Error):
        """a class for read/write errors"""
    
    def check_add(self, args):
        """check the validity of line
        syntax: add process PROCESS 
        """
    
        if len(args) < 2:
            self.help_add()
            raise self.InvalidCmd('\"add\" requires at least two arguments')
        
        if args[0] != 'process':
            raise self.InvalidCmd('\"add\" requires the argument \"process\"')

        if not self._curr_model:
            logger.info("No model currently active, so we import the Standard Model")
            self.do_import('model sm')
        

        if args[-1].startswith('--optimize'):
            if args[2] != '>':
                raise self.InvalidCmd('optimize mode valid only for 1->N processes. (See model restriction for 2->N)')
            if '=' in args[-1]:
                path = args[-1].split('=',1)[1]
                if not os.path.exists(path) or \
                                self.detect_file_type(path) != 'param_card':
                    raise self.InvalidCmd('%s is not a valid param_card')
            else:
                path=None
            # Update the default value of the model here.
            if not isinstance(self._curr_model, model_reader.ModelReader):
                self._curr_model = model_reader.ModelReader(self._curr_model)
            self._curr_model.set_parameters_and_couplings(path)
            self.check_process_format(' '.join(args[1:-1]))
        else:
            self.check_process_format(' '.join(args[1:]))

    def check_define(self, args):
        """check the validity of line
        syntax: define multipart_name [ part_name_list ]
        """  

        
        if len(args) < 2:
            self.help_define()
            raise self.InvalidCmd('\"define\" command requires at least two arguments')

        if args[1] == '=':
            del args[1]
            if len(args) < 2:
                self.help_define()
                raise self.InvalidCmd('\"define\" command requires at least one particles name after \"=\"')
        
        if '=' in args:
            self.help_define()
            raise self.InvalidCmd('\"define\" command requires symbols \"=\" at the second position')
        
        if not self._curr_model:
            logger.info('No model currently active. Try with the Standard Model')
            self.do_import('model sm')

        if self._curr_model['particles'].find_name(args[0]):
            raise self.InvalidCmd("label %s is a particle name in this model\n\
            Please retry with another name." % args[0])

    def check_display(self, args):
        """check the validity of line
        syntax: display XXXXX
        """
            
        if len(args) < 1:
            self.help_display()
            raise self.InvalidCmd, 'display requires an argument specifying what to display'
        if args[0] not in self._display_opts:
            self.help_display()
            raise self.InvalidCmd, 'Invalid arguments for display command: %s' % args[0]

        if not self._curr_model:
            raise self.InvalidCmd("No model currently active, please import a model!")

        if args[0] in ['processes', 'diagrams'] and not self._curr_amps:
            raise self.InvalidCmd("No process generated, please generate a process!")
        if args[0] == 'checks' and not self._comparisons:
            raise self.InvalidCmd("No check results to display.")
        
        if args[0] == 'variable' and len(args) !=2:
            raise self.InvalidCmd('variable need a variable name')


    def check_draw(self, args):
        """check the validity of line
        syntax: draw DIRPATH [option=value]
        """
        
        if len(args) < 1:
            args.append('/tmp')
        
        if not self._curr_amps:
            raise self.InvalidCmd("No process generated, please generate a process!")
            
        if not os.path.isdir(args[0]):
            raise self.InvalidCmd( "%s is not a valid directory for export file" % args[0])
            
    def check_check(self, args):
        """check the validity of args"""
        
        if  not self._curr_model:
            raise self.InvalidCmd("No model currently active, please import a model!")

        if self._model_v4_path:
            raise self.InvalidCmd(\
                "\"check\" not possible for v4 models")

        if len(args) < 2:
            self.help_check()
            raise self.InvalidCmd("\"check\" requires a process.")

        param_card = None
        if os.path.isfile(args[1]):
            param_card = args.pop(1)

        if args[0] not in self._check_opts:
            args.insert(0, 'full')

        if any([',' in elem for elem in args]):
            raise self.InvalidCmd('Decay chains not allowed in check')
        
        if not args[-1].startswith('--energy='):
            args.append('--energy=1000')
        
        self.check_process_format(" ".join(args[1:-1]))

        return param_card
    
    def check_generate(self, args):
        """check the validity of args"""
        # Not called anymore see check_add
        return self.check_add(args)
    
    def check_process_format(self, process):
        """ check the validity of the string given to describe a format """
        
        #check balance of paranthesis
        if process.count('(') != process.count(')'):
            raise self.InvalidCmd('Invalid Format, no balance between open and close parenthesis')
        #remove parenthesis for fututre introspection
        process = process.replace('(',' ').replace(')',' ')
        
        # split following , (for decay chains)
        subprocesses = process.split(',')
        if len(subprocesses) > 1:
            for subprocess in subprocesses:
                self.check_process_format(subprocess)
            return
        
        # request that we have one or two > in the process
        if process.count('>') not in [1,2]:
            raise self.InvalidCmd(
               'wrong format for \"%s\" this part requires one or two symbols \'>\', %s found' 
               % (process, process.count('>')))
        
        # we need at least one particles in each pieces
        particles_parts = process.split('>')
        for particles in particles_parts:
            if re.match(r'^\s*$', particles):
                raise self.InvalidCmd(
                '\"%s\" is a wrong process format. Please try again' % process)  
        
        # '/' and '$' sould be used only after the process definition
        for particles in particles_parts[:-1]:
            if re.search('\D/', particles):
                raise self.InvalidCmd(
                'wrong process format: restriction should be place after the final states')
            if re.search('\D\$', particles):
                raise self.InvalidCmd(
                'wrong process format: restriction should be place after the final states')
        
    
    def check_import(self, args):
        """check the validity of line"""
        
        modelname = False
        if '-modelname' in args:
            args.remove('-modelname')
            modelname = True   
        elif '--modelname' in args:
            args.remove('--modelname')
            modelname = True              
                
        if not args:
            self.help_import()
            raise self.InvalidCmd('wrong \"import\" format')
        
        if len(args) >= 2 and args[0] not in self._import_formats:
            self.help_import()
            raise self.InvalidCmd('wrong \"import\" format')            
        elif len(args) == 1:
            if args[0] in self._import_formats:
                if args[0] != "proc_v4":
                    self.help_import()
                    raise self.InvalidCmd('wrong \"import\" format')
                elif not self._export_dir:
                    self.help_import()
                    raise self.InvalidCmd('PATH is mandatory in the current context\n' + \
                                  'Did you forget to run the \"output\" command')
            # The type of the import is not given -> guess it
            format = self.find_import_type(args[0])
            logger.info('The import format was not given, so we guess it as %s' % format)
            args.insert(0, format)
            if self.history[-1].startswith('import'):
                self.history[-1] = 'import %s %s' % \
                                (format, ' '.join(self.history[-1].split()[1:]))
                                    
        if modelname:
            args.append('-modelname') 
        
        
          
    def check_install(self, args):
        """check that the install command is valid"""
        
        if len(args) < 1:
            self.help_install()
            raise self.InvalidCmd('install command require at least one argument')
        
        if args[0] not in self._install_opts:
            if not args[0].startswith('td'):
                self.help_install()
                raise self.InvalidCmd('Not recognize program %s ' % args[0])
            
        if args[0] in ["ExRootAnalysis", "Delphes", "Delphes2"]:
            if not misc.which('root'):
                raise self.InvalidCmd(
'''In order to install ExRootAnalysis, you need to install Root on your computer first.
please follow information on http://root.cern.ch/drupal/content/downloading-root''')
            if 'ROOTSYS' not in os.environ:
                raise self.InvalidCmd(
'''The environment variable ROOTSYS is not configured.
You can set it by adding the following lines in your .bashrc [.bash_profile for mac]:
export ROOTSYS=%s
export PATH=$PATH:$ROOTSYS/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTSYS/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$ROOTSYS/lib
This will take effect only in a NEW terminal
''' % os.path.realpath(pjoin(misc.which('root'), \
                                               os.path.pardir, os.path.pardir)))

                
    def check_launch(self, args, options):
        """check the validity of the line"""
        # modify args in order to be MODE DIR
        # mode being either standalone or madevent
        if not( 0 <= int(options.cluster) <= 2):
            return self.InvalidCmd, 'cluster mode should be between 0 and 2'
        
        if not args:
            if self._done_export:
                mode = self.find_output_type(self._done_export[0])
                if mode != self._done_export[1]:
                    raise self.InvalidCmd, \
                          '%s not valid directory for launch' % self._done_export[0]
                args.append(self._done_export[1])
                args.append(self._done_export[0])
                return
            else:
                logger.warning('output command missing, run it automatically (with default argument)')
                self.do_output('')
                logger.warning('output done: running launch')
                return self.check_launch(args, options)
        
        if len(args) != 1:
            self.help_launch()
            return self.InvalidCmd, 'Invalid Syntax: Too many argument'
        
        # search for a valid path
        if os.path.isdir(args[0]):
            path = os.path.realpath(args[0])
        elif os.path.isdir(pjoin(MG5DIR,args[0])):
            path = pjoin(MG5DIR,args[0])
        elif  MG4DIR and os.path.isdir(pjoin(MG4DIR,args[0])):
            path = pjoin(MG4DIR,args[0])
        else:    
            raise self.InvalidCmd, '%s is not a valid directory' % args[0]
                
        mode = self.find_output_type(path)
        
        args[0] = mode
        args.append(path)
        # inform where we are for future command
        self._done_export = [path, mode]
        

    def find_import_type(self, path):
        """ identify the import type of a given path 
        valid output: model/model_v4/proc_v4/command"""
        
        possibility = [pjoin(MG5DIR,'models',path), \
                     pjoin(MG5DIR,'models',path+'_v4'), path]
        if '-' in path:
            name = path.rsplit('-',1)[0]
            possibility = [pjoin(MG5DIR,'models',name), name] + possibility
        # Check if they are a valid directory
        for name in possibility:
            if os.path.isdir(name):
                if os.path.exists(pjoin(name,'particles.py')):
                    return 'model'
                elif os.path.exists(pjoin(name,'particles.dat')):
                    return 'model_v4'
        
        # Not valid directory so maybe a file
        if os.path.isfile(path):
            text = open(path).read()
            pat = re.compile('(Begin process|<MGVERSION>)', re.I)
            matches = pat.findall(text)
            if not matches:
                return 'command'
            elif len(matches) > 1:
                return 'banner'
            elif matches[0].lower() == 'begin process':
                return 'proc_v4'
            else:
                return 'banner'
        else:
            return 'proc_v4'
        
        

    
    def find_output_type(self, path):
        """ identify the type of output of a given directory:
        valid output: madevent/standalone/standalone_cpp"""
        
        card_path = pjoin(path,'Cards')
        bin_path = pjoin(path,'bin')
        src_path = pjoin(path,'src')
        include_path = pjoin(path,'include')
        subproc_path = pjoin(path,'SubProcesses')

        if os.path.isfile(pjoin(include_path, 'Pythia.h')):
            return 'pythia8'
        elif not os.path.isdir(os.path.join(path, 'SubProcesses')):
            raise self.InvalidCmd, '%s : Not a valid directory' % path
        
        if os.path.isdir(src_path):
            return 'standalone_cpp'
        elif os.path.isfile(pjoin(bin_path,'generate_events')):
            return 'madevent'
        elif os.path.isdir(card_path):
            return 'standalone'

        raise self.InvalidCmd, '%s : Not a valid directory' % path
        
    def check_load(self, args):
        """ check the validity of the line"""
        
        if len(args) != 2 or args[0] not in self._save_opts:
            self.help_load()
            raise self.InvalidCmd('wrong \"load\" format')
    
    def check_customize_model(self, args):
        """check the validity of the line"""
        
        # Check argument validity
        if len(args) >1 :
            self.help_customize_model()
            raise self.InvalidCmd('No argument expected for this command')
        
        if len(args):
            if not args[0].startswith('--save='):
                self.help_customize_model()
                raise self.InvalidCmd('Wrong argument for this command')
            if '-' in args[0][6:]:
                raise self.InvalidCmd('The name given in save options can\'t contain \'-\' symbol.')
            
        if self._model_v4_path:
            raise self.InvalidCmd('Restriction of Model is not supported by v4 model.')
        
        
    def check_save(self, args):
        """ check the validity of the line"""
        
        if len(args) == 0:
            args.append('options')
        
        if args[0] not in self._save_opts and args[0] != 'global':
            self.help_save()
            raise self.InvalidCmd('wrong \"save\" format')
        elif args[0] == 'global':
            args.insert(0, 'options')
        
        if args[0] != 'options' and len(args) != 2:
            self.help_save()
            raise self.InvalidCmd('wrong \"save\" format')
        elif args[0] != 'options' and len(args) == 2:
            basename = os.path.dirname(args[1])
            if not os.path.exists(basename):
                raise self.InvalidCmd('%s is not a valid path, please retry' % \
                                                                        args[1])
        
        if args[0] == 'options':
            has_path = None
            for arg in args[1:]:
                if arg in ['--auto', '--all']:
                    continue
                elif arg.startswith('--'):
                    raise self.InvalidCmd('unknow command for \'save options\'')
                elif arg == 'global':
                    if os.environ.has_key('HOME'):
                        args.remove('global')
                        args.insert(1,pjoin(os.environ['HOME'],'.mg5','mg5_configuration.txt'))
                        has_path = True
                else:
                    basename = os.path.dirname(arg)
                    if not os.path.exists(basename):
                        raise self.InvalidCmd('%s is not a valid path, please retry' % \
                                                                        arg)
                    elif has_path:
                        raise self.InvalidCmd('only one path is allowed')
                    else:
                        args.remove(arg)
                        args.insert(1, arg)
                        has_path = True
            if not has_path:
                args.insert(1, pjoin(MG5DIR,'input','mg5_configuration.txt'))     
                

    def check_set(self, args, log=True):
        """ check the validity of the line"""

        if len(args) == 1 and args[0] == 'complex_mass_scheme':
            args.append('True')

        if len(args) > 2 and '=' == args[1]:
            args.pop(1)
        
        if len(args) < 2:
            self.help_set()
            raise self.InvalidCmd('set needs an option and an argument')

        if args[1] == 'default':
            if args[0] in self.options_configuration:
                default = self.options_configuration[args[0]]
            elif args[0] in self.options_madgraph:
                default = self.options_madgraph[args[0]]
            elif args[0] in self.options_madevent:
                default = self.options_madevent[args[0]]
            else:
                raise self.InvalidCmd('%s doesn\'t have a valid default value' % args[0])
            if log:
                logger.info('Pass parameter %s to it\'s default value: %s' % 
                                                             (args[0], default))
            args[1] = str(default)

        if args[0] not in self._set_options:
            if not args[0] in self.options and not args[0] in self.options:
                self.help_set()
                raise self.InvalidCmd('Possible options for set are %s' % \
                                  self._set_options)

        if args[0] in ['group_subprocesses']:
            if args[1] not in ['False', 'True', 'Auto']:
                raise self.InvalidCmd('%s needs argument False, True or Auto' % \
                                      args[0])
        if args[0] in ['ignore_six_quark_processes']:
            if args[1] not in self._multiparticles.keys() and args[1] != 'False':
                raise self.InvalidCmd('ignore_six_quark_processes needs ' + \
                                      'a multiparticle name as argument')
        
        if args[0] in ['stdout_level']:
            if args[1] not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL'] and \
                                                          not args[1].isdigit():
                raise self.InvalidCmd('output_level needs ' + \
                                      'a valid level')       
        if args[0] in ['gauge']:
            if args[1] not in ['unitary','Feynman']:
                raise self.InvalidCmd('gauge needs argument unitary or Feynman.')       

        if args[0] in ['timeout']:
            if not args[1].isdigit():
                raise self.InvalidCmd('timeout values should be a integer')
            

            
    def check_open(self, args):
        """ check the validity of the line """
        
        if len(args) != 1:
            self.help_open()
            raise self.InvalidCmd('OPEN command requires exactly one argument')

        if args[0].startswith('./'):
            if not os.path.isfile(args[0]):
                raise self.InvalidCmd('%s: not such file' % args[0])
            return True

        # if special : create the path.
        if not self._done_export:
            if not os.path.isfile(args[0]):
                self.help_open()
                raise self.InvalidCmd('No command \"output\" or \"launch\" used. Impossible to associate this name to a file')
            else:
                return True
            
        path = self._done_export[0]
        if os.path.isfile(pjoin(path,args[0])):
            args[0] = pjoin(path,args[0])
        elif os.path.isfile(pjoin(path,'Cards',args[0])):
            args[0] = pjoin(path,'Cards',args[0])
        elif os.path.isfile(pjoin(path,'HTML',args[0])):
            args[0] = pjoin(path,'HTML',args[0])
        # special for card with _default define: copy the default and open it
        elif '_card.dat' in args[0]:   
            name = args[0].replace('_card.dat','_card_default.dat')
            if os.path.isfile(pjoin(path,'Cards', name)):
                files.cp(path + '/Cards/' + name, path + '/Cards/'+ args[0])
                args[0] = pjoin(path,'Cards', args[0])
            else:
                raise self.InvalidCmd('No default path for this file')
        elif not os.path.isfile(args[0]):
            raise self.InvalidCmd('No default path for this file')
                
                
    def check_output(self, args):
        """ check the validity of the line"""
          
        if args and args[0] in self._export_formats:
            self._export_format = args.pop(0)
        else:
            self._export_format = 'madevent'

        if not self._curr_model:
            text = 'No model found. Please import a model first and then retry.'
            raise self.InvalidCmd(text)

        if self._model_v4_path and \
               (self._export_format not in self._v4_export_formats):
            text = " The Model imported (MG4 format) does not contain enough\n "
            text += " information for this type of output. In order to create\n"
            text += " output for " + args[0] + ", you have to use a UFO model.\n"
            text += " Those model can be imported with mg5> import model NAME."
            logger.warning(text)
            raise self.InvalidCmd('')

        if self._export_format == 'aloha':
            return


        if not self._curr_amps:
            text = 'No processes generated. Please generate a process first.'
            raise self.InvalidCmd(text)





        if args and args[0][0] != '-':
            # This is a path
            path = args.pop(0)
            forbiden_chars = ['>','<',';','&']
            for char in forbiden_chars:
                if char in path:
                    raise self.invalidCmd('%s is not allowed in the output path' % char)
            # Check for special directory treatment
            if path == 'auto' and self._export_format in \
                     ['madevent', 'standalone', 'standalone_cpp']:
                self.get_default_path()
            elif path != 'auto':
                self._export_dir = path
            elif path == 'auto':
                if self.options['pythia8_path']:
                    self._export_dir = self.options['pythia8_path']
                else:
                    self._export_dir = '.'
        else:
            if self._export_format != 'pythia8':
                # No valid path
                self.get_default_path()
            else:
                if self.options['pythia8_path']:
                    self._export_dir = self.options['pythia8_path']
                else:
                    self._export_dir = '.'
                    
        self._export_dir = os.path.realpath(self._export_dir)
    
    def check_decay_diagram(self, args):
        """ check and format calculate decay width:
        Expected format: NAME [OTHER_NAMES] PREC/LEVEL [param_card]
        """    
        
        output = {'path': None, 'level':None, 'ids':set()}
        
        if len(args)<2:
            self.help_calculate_width()
            raise self.InvalidCmd('decay_diagram requires at least two arguments')

        if len(args) >= 3:
            if not os.path.exists(args[-1]):
                if os.path.exists(pjoin(MG5DIR, args[-1])):
                    output['path'] = pjoin(MG5DIR, args[-1])
                elif self._model_v4_path and  os.path.exists(pjoin(self._model_v4_path, args[-1])):
                         output['path'] = pjoin(self._curr_model_v4_path, args[-1])   
                elif os.path.exists(pjoin(self._curr_model.get('modelpath'), args[-1])):
                    output['path'] = pjoin(self._curr_model.get('modelpath'), args[-1])                
                else:
                    try:
                        precision = float(args[-1])
                    except Exception:
                         raise self.InvalidCmd('%s is not a valid path /precision' % args[-1])
                    else:
                        output['level'] = precision
            else:
                output['path'] = args[-1]
            # check that the path is indeed a param_card:
            if output['path'] and madevent_interface.MadEventCmd.detect_card_type(output['path']) != 'param_card.dat':
                raise self.InvalidCmd('%s should be a path to a param_card' % output['path'])
        else:
            try:
                precision = float(args[-1])
            except Exception:
                 raise self.InvalidCmd('%s is not a valid path /precision' % args[-1])
            else:
                output['level'] = precision
                

        if not output['level']:
            try:
                precision = float(args[-2])
            except Exception:
                raise self.InvalidCmd('%s is not a float (expected for precision args)' % args[-2])
            else:
                output['level'] = precision            
            max_pos = -2
        else:
            max_pos = -1

        for arg in args[:max_pos]:
            # check that the first argument is the particle name.
            if arg.isdigit():
                p = self._curr_model.get_particle(int(arg))
                if not p:
                    raise self.InvalidCmd('Model doesn\'t have pid %s for a particle' % arg)
                output['ids'].add(int(arg))
            elif arg in self._multiparticles:
                output['ids'].update(set(abs(id) for id in self._multiparticles[arg]))
            else:
                for p in self._curr_model['particles']:
                    if p['name'] == arg or p['antiname'] == arg:
                        output['ids'].add(abs(p.get_pdg_code()))
                        break
                else:
                    raise self.InvalidCmd('invalid particle name')
    
        return output
        
    def check_compute_widths(self, args):
        """ check and format calculate decay width:
        Expected format: NAME [other names] [--options]
        # fill the options if not present.
        # NAME can be either (anti-)particle name, multiparticle, pid
        """    
        
        if len(args)<1:
            self.help_calculate_width()
            raise self.InvalidCmd('''compute_widths requires at least the name of one particle.
            If you want to compute the width of all particles, type \'compute_widths all\'''')

        particles = set()
        options = {'precision': 0.0025, 'path':None, 'output':None}
        # check that the firsts argument is valid
        for i,arg in enumerate(args):
            if arg.startswith('--'):
                if not '=' in arg:
                    raise self.InvalidCmd('Options required an equal (and then the value)')
                arg, value = arg.split('=')
                if arg[2:] not in options:
                    raise self.InvalidCmd('%s not valid options' % arg)
                options[arg[2:]] = value
                continue
            # check for pid
            if arg.isdigit():
                p = self._curr_model.get_particle(int(arg))
                if not p:
                    raise self.InvalidCmd('Model doesn\'t have pid %s for any particle' % arg)
                particles.add(abs(int(arg)))
            elif arg in self._multiparticles:
                particles.update([abs(id) for id in self._multiparticles[args[0]]])
            else:
                for p in self._curr_model['particles']:
                    if p['name'] == args[0] or p['antiname'] == arg:
                        particles.add(abs(p.get_pdg_code()))
                        break
                else:
                    raise self.InvalidCmd('%s invalid particle name' % arg)
    
        if options['path'] and not os.path.isfile(options['path']):

            if os.path.exists(pjoin(MG5DIR, options['path'])):
                options['path'] = pjoin(MG5DIR, options['path'])
            elif self._model_v4_path and  os.path.exists(pjoin(self._model_v4_path, options['path'])):
                options['path'] = pjoin(self._curr_model_v4_path, options['path'])   
            elif os.path.exists(pjoin(self._curr_model.path, options['path'])):
                options['path'] = pjoin(self._curr_model.path, options['path'])                
                
            if os.path.isdir(options['path']) and os.path.isfile(pjoin(options['path'], 'param_card.dat')):
                options['path'] = pjoin(options['path'], 'param_card.dat')
            elif not os.path.isfile(options['path']):
                raise self.InvalidCmd('%s is not a valid path' % args[2])
            # check that the path is indeed a param_card:
            if madevent_interface.MadEventCmd.detect_card_type(options['path']) != 'param_card.dat':
                raise self.InvalidCmd('%s should be a path to a param_card' % options['path'])
  
        if not options['path']:
            param_card_text = self._curr_model.write_param_card()
            if not options['output']:
                dirpath = self._curr_model.get('modelpath')
                options['path'] = pjoin(dirpath, 'param_card.dat')
            else:
                options['path'] = options['output']
            ff = open(options['path'],'w')
            ff.write(param_card_text)
            ff.close()
        if not options['output']:
            options['output'] = options['path']

        return particles, options
                
        

    def get_default_path(self):
        """Set self._export_dir to the default (\'auto\') path"""

        if self._export_format in ['madevent', 'standalone']:
            # Detect if this script is launched from a valid copy of the Template,
            # if so store this position as standard output directory
            if 'TemplateVersion.txt' in os.listdir('.'):
                #Check for ./
                self._export_dir = os.path.realpath('.')
                return
            elif 'TemplateVersion.txt' in os.listdir('..'):
                #Check for ../
                self._export_dir = os.path.realpath('..')
                return
            elif self.stdin != sys.stdin:
                #Check for position defined by the input files
                input_path = os.path.realpath(self.stdin.name).split(os.path.sep)
                print "Not standard stdin, use input path"
                if input_path[-2] == 'Cards':
                    self._export_dir = os.path.sep.join(input_path[:-2])
                    if 'TemplateVersion.txt' in self._export_dir:
                        return
        
        if self._export_format.startswith('madevent'):            
            name_dir = lambda i: 'PROC_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format == 'standalone':
            name_dir = lambda i: 'PROC_SA_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))                
        elif self._export_format == 'standalone_cpp':
            name_dir = lambda i: 'PROC_SA_CPP_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format == 'pythia8':
            if self.options['pythia8_path']:
                self._export_dir = self.options['pythia8_path']
            else:
                self._export_dir = '.'
            return
        else:
            self._export_dir = '.'
            return
        for i in range(500):
            if os.path.isdir(auto_path(i)):
                continue
            else:
                self._export_dir = auto_path(i) 
                break
        if not self._export_dir:
            raise self.InvalidCmd('Can\'t use auto path,' + \
                                  'more than 500 dirs already')    
            
        
#===============================================================================
# CheckValidForCmdWeb
#===============================================================================
class CheckValidForCmdWeb(CheckValidForCmd):
    """ Check the validity of input line for web entry
    (no explicit path authorized)"""
    
    class WebRestriction(MadGraph5Error):
        """class for WebRestriction"""
    
    def check_draw(self, args):
        """check the validity of line
        syntax: draw FILEPATH [option=value]
        """
        raise self.WebRestriction('direct call to draw is forbidden on the web')
    
    def check_display(self, args):
        """ check the validity of line in web mode """
        
        if args[0] == 'mg5_variable':
            raise self.WebRestriction('Display internal variable is forbidden on the web')
        
        CheckValidForCmd.check_history(self, args)
    
    def check_check(self, args):
        """ Not authorize for the Web"""
        
        raise self.WebRestriction('Check call is forbidden on the web')
    
    def check_history(self, args):
        """check the validity of line
        No Path authorize for the Web"""
        
        CheckValidForCmd.check_history(self, args)

        if len(args) == 2 and args[1] not in ['.', 'clean']:
            raise self.WebRestriction('Path can\'t be specify on the web.')

        
    def check_import(self, args):
        """check the validity of line
        No Path authorize for the Web"""
        
        if not args:
            raise self.WebRestriction, 'import requires at least one option'
        
        if args[0] not in self._import_formats:
            args[:] = ['command', './Cards/proc_card_mg5.dat']
        elif args[0] == 'proc_v4':
            args[:] = [args[0], './Cards/proc_card.dat']
        elif args[0] == 'command':
            args[:] = [args[0], './Cards/proc_card_mg5.dat']

        CheckValidForCmd.check_import(self, args)
        
    def check_install(self, args):
        """ No possibility to install new software on the web """
        if args == ['update','--mode=mg5_start']:
            return
        
        raise self.WebRestriction('Impossible to install program on the cluster')
        
    def check_load(self, args):
        """ check the validity of the line
        No Path authorize for the Web"""

        CheckValidForCmd.check_load(self, args)        

        if len(args) == 2:
            if args[0] != 'model':
                raise self.WebRestriction('only model can be loaded online')
            if 'model.pkl' not in args[1]:
                raise self.WebRestriction('not valid pkl file: wrong name')
            if not os.path.realpath(args[1]).startswith(pjoin(MG4DIR, \
                                                                    'Models')):
                raise self.WebRestriction('Wrong path to load model')
        
    def check_save(self, args):
        """ not authorize on web"""
        raise self.WebRestriction('\"save\" command not authorize online')
    
    def check_open(self, args):
        """ not authorize on web"""
        raise self.WebRestriction('\"open\" command not authorize online')
    
    def check_output(self, args):
        """ check the validity of the line"""

        
        # first pass to the default
        CheckValidForCmd.check_output(self, args)
        
        args[:] = [self._export_format, '.', '-f']

        # Check that we output madevent
        if 'madevent' != self._export_format:
                raise self.WebRestriction, 'only available output format is madevent (at current stage)'

#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(cmd.CompleteCmd):
    """ The Series of help routine for the MadGraphCmd"""
    
 
    def model_completion(self, text, process):
        """ complete the line with model information """

        while ',' in process:
            process = process[process.index(',')+1:]
        args = self.split_arg(process)
        couplings = []

        # Force '>' if two initial particles.
        if len(args) == 2 and args[-1] != '>':
                return self.list_completion(text, '>')
            
        # Add non-particle names
        if len(args) > 0 and args[-1] != '>':
            couplings = ['>']
        if '>' in args and args.index('>') < len(args) - 1:
            couplings = [c + "=" for c in self._couplings] + \
                        ['@','$','/','>',',']
        return self.list_completion(text, self._particle_names + \
                                    self._multiparticles.keys() + couplings)
        
                    
    def complete_generate(self, text, line, begidx, endidx):
        "Complete the add command"

        # Return list of particle names and multiparticle names, as well as
        # coupling orders and allowed symbols
        args = self.split_arg(line[0:begidx])
        if len(args) > 2 and args[-1] == '@' or args[-1].endswith('='):
            return

        try:
            return self.model_completion(text, ' '.join(args[1:]))
        except Exception as error:
            print error
            
        #if len(args) > 1 and args[-1] != '>':
        #    couplings = ['>']
        #if '>' in args and args.index('>') < len(args) - 1:
        #    couplings = [c + "=" for c in self._couplings] + ['@','$','/','>']
        #return self.list_completion(text, self._particle_names + \
        #                            self._multiparticles.keys() + couplings)

    def complete_decay_diagram(self, text, line, begidx, endidx):
        "Complete the add command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) <= 1:
            return self.model_completion(text, '')
        elif len(args) == 2:
            return self.list_completion(text, ['2','3','4','0.'])
        elif len(args) == 3:
            return self.path_completion(text)
        else:
            return self.path_completion(text, pjoin(*[a for a in args \
                                                    if a.endswith(os.path.sep)]))

    def complete_compute_widths(self, text, line, begidx, endidx):
        "Complete the compute_widths command"

        args = self.split_arg(line[0:begidx])
        
        if args[-1] in  ['--path=', '--output=']:
            completion = {'path': self.path_completion(text)}
        elif line[begidx-1] == os.path.sep:
            current_dir = pjoin(*[a for a in args if a.endswith(os.path.sep)])
            if current_dir.startswith('--path='):
                current_dir = current_dir[7:]
            if current_dir.startswith('--output='):
                current_dir = current_dir[9:]                
            completion = {'path': self.path_completion(text, current_dir)}
        elif args[-1].startswith('--precision='):
            completion = {'precision': self.list_completion(text, ['2','3','4','0.\$'])}
        else:
            completion = {}            
            completion['options'] = self.list_completion(text, 
                            ['--precision=', '--path=', '--output='])
            completion['particles'] = self.model_completion(text, '')            
        
        return self.deal_multiple_categories(completion)

    def complete_add(self, text, line, begidx, endidx):
        "Complete the add command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._add_opts)

        return self.complete_generate(text, " ".join(args[1:]), begidx, endidx)

        # Return list of particle names and multiparticle names, as well as
        # coupling orders and allowed symbols
        couplings = []
        if len(args) > 2 and args[-1] != '>':
            couplings = ['>']
        if '>' in args and args.index('>') < len(args) - 1:
            couplings = [c + "=" for c in self._couplings] + ['@','$','/','>']
        return self.list_completion(text, self._particle_names + \
                                    self._multiparticles.keys() + couplings)
    
    def complete_customize_model(self, text, line, begidx, endidx):
        "Complete the customize_model command"
        
        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, ['--save='])
        
    
    def complete_check(self, text, line, begidx, endidx):
        "Complete the check command"

        out = {}
        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._check_opts)

        


        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text, pjoin(*[a for a in args \
                                                    if a.endswith(os.path.sep)]))
        # autocompletion for particles/couplings
        model_comp = self.model_completion(text, ' '.join(args[2:]))

        if len(args) == 2:
            out['particles'] = model_comp
            out['path to param_card'] = self.path_completion(text)
            out['options'] = self.list_completion(text, ['--energy='])
            return self.deal_multiple_categories(out)

        if len(args) > 2:
            out['particles'] = model_comp
            out['options'] = self.list_completion(text, ['--energy='])
            return self.deal_multiple_categories(out)
            
        
    def complete_tutorial(self, text, line, begidx, endidx):
        "Complete the tutorial command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._tutorial_opts)
        
    def complete_define(self, text, line, begidx, endidx):
        """Complete particle information"""
        return self.model_completion(text, line[6:])

    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        args = self.split_arg(line[0:begidx])
        # Format
        if len(args) == 1:
            return self.list_completion(text, self._display_opts)

        if len(args) == 2 and args[1] == 'checks':
            return self.list_completion(text, ['failed'])

        if len(args) == 2 and args[1] == 'particles':
            return self.model_completion(text, line[begidx:])

    def complete_draw(self, text, line, begidx, endidx):
        "Complete the draw command"

        args = self.split_arg(line[0:begidx])

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                                        only_dirs = True)
        # Format
        if len(args) == 1:
            return self.path_completion(text, '.', only_dirs = True)


        #option
        if len(args) >= 2:
            opt = ['horizontal', 'external=', 'max_size=', 'add_gap=',
                                'non_propagating', '--']
            return self.list_completion(text, opt)

    def complete_launch(self, text, line, begidx, endidx):
        """ complete the launch command"""
        args = self.split_arg(line[0:begidx])

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                                        only_dirs = True)
        # Format
        if len(args) == 1:
            out = {'Path from ./': self.path_completion(text, '.', only_dirs = True)}
            if MG5DIR != os.path.realpath('.'):
                out['Path from %s' % MG5DIR] =  self.path_completion(text,
                                     MG5DIR, only_dirs = True, relative=False)
            if MG4DIR and MG4DIR != os.path.realpath('.') and MG4DIR != MG5DIR:
                out['Path from %s' % MG4DIR] =  self.path_completion(text,
                                     MG4DIR, only_dirs = True, relative=False)

            
        #option
        if len(args) >= 2:
            out={}

        if line[0:begidx].endswith('--laststep='):
            opt = ['parton', 'pythia', 'pgs','delphes','auto']
            out['Options'] = self.list_completion(text, opt, line)
        else:
            opt = ['--cluster', '--multicore', '-i', '--name=', '-f','-m', '-n', 
               '--interactive', '--laststep=parton', '--laststep=pythia',
               '--laststep=pgs', '--laststep=delphes','--laststep=auto']
            out['Options'] = self.list_completion(text, opt, line)
        

        return self.deal_multiple_categories(out)

    def complete_load(self, text, line, begidx, endidx):
        "Complete the load command"

        args = self.split_arg(line[0:begidx])        

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._save_opts)

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

        # Filename if directory is not given
        if len(args) == 2:
            return self.path_completion(text)

    def complete_save(self, text, line, begidx, endidx):
        "Complete the save command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._save_opts)

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                                        only_dirs = True)

        # Filename if directory is not given
        if len(args) == 2:
            return self.path_completion(text) + self.list_completion(text, ['global'])

    @cmd.debug()    
    def complete_open(self, text, line, begidx, endidx): 
        """ complete the open command """
        
        args = self.split_arg(line[0:begidx])
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

        possibility = []
        if self._done_export:
            path = self._done_export[0]
            possibility = ['index.html']
            if os.path.isfile(pjoin(path,'README')):
                possibility.append('README')
            if os.path.isdir(pjoin(path,'Cards')):
                possibility += [f for f in os.listdir(pjoin(path,'Cards')) 
                                    if f.endswith('.dat')]
            if os.path.isdir(pjoin(path,'HTML')):
                possibility += [f for f in os.listdir(pjoin(path,'HTML')) 
                                  if f.endswith('.html') and 'default' not in f]
        else:
            possibility.extend(['./','../'])
        if os.path.exists('MG5_debug'):
            possibility.append('MG5_debug')
        if os.path.exists('ME5_debug'):
            possibility.append('ME5_debug')            

        return self.list_completion(text, possibility)
    
    @cmd.debug()
    def complete_output(self, text, line, begidx, endidx,
                        possible_options = ['f', 'noclean', 'nojpeg'],
                        possible_options_full = ['-f', '-noclean', '-nojpeg']):
        "Complete the output command"

        possible_format = self._export_formats
        #don't propose directory use by MG_ME
        forbidden_names = ['MadGraphII', 'Template', 'pythia-pgs', 'CVS',
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis',
                            'mg5', 'DECAY', 'EventConverter', 'Models',
                            'ExRootAnalysis', 'HELAS', 'Transfer_Fct', 'aloha']
        
        #name of the run =>proposes old run name
        args = self.split_arg(line[0:begidx])
        if len(args) >= 1: 
            if len(args) > 1 and args[1] == 'aloha':
                try:
                    return self.aloha_complete_output(text, line, begidx, endidx)
                except Exception, error:
                    print error
            # Directory continuation
            if args[-1].endswith(os.path.sep):
                return [name for name in self.path_completion(text,
                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True) if name not in forbidden_names]
            # options
            if args[-1][0] == '-' or len(args) > 1 and args[-2] == '-':
                return self.list_completion(text, possible_options)
            if len(args) > 2:
                return self.list_completion(text, possible_options_full)
            # Formats
            if len(args) == 1:
                format = possible_format + ['.' + os.path.sep, '..' + os.path.sep, 'auto']
                return self.list_completion(text, format)

            # directory names
            content = [name for name in self.path_completion(text, '.', only_dirs = True) \
                       if name not in forbidden_names]
            content += ['auto']
            return self.list_completion(text, content)

    def aloha_complete_output(self, text, line, begidx, endidx):
        "Complete the output aloha command"
        args = self.split_arg(line[0:begidx])
        completion_categories = {}
        
        forbidden_names = ['MadGraphII', 'Template', 'pythia-pgs', 'CVS',
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis',
                            'mg5', 'DECAY', 'EventConverter', 'Models',
                            'ExRootAnalysis', 'Transfer_Fct', 'aloha',
                            'apidoc','vendor']
        
        
        # options
        options = ['--format=Fortran', '--format=Python','--format=gpu','--format=CPP','--output=']
        options = self.list_completion(text, options)
        if options:
            completion_categories['options'] = options
        
        if args[-1] == '--output=' or args[-1].endswith(os.path.sep):
            # Directory continuation
            completion_categories['path'] =  [name for name in self.path_completion(text,
                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True) if name not in forbidden_names]

        else:
            ufomodel = ufomodels.load_model(self._curr_model.get('name'))
            wf_opt = []
            amp_opt = []
            opt_conjg = []
            for lor in ufomodel.all_lorentz:
                amp_opt.append('%s_0' % lor.name)
                for i in range(len(lor.spins)):
                    wf_opt.append('%s_%i' % (lor.name,i+1))
                    if i % 2 == 0 and lor.spins[i] == 2:
                        opt_conjg.append('%sC%i_%i' % (lor.name,i //2 +1,i+1))
            completion_categories['amplitude routines'] = self.list_completion(text, amp_opt) 
            completion_categories['Wavefunctions routines'] = self.list_completion(text, wf_opt)        
            completion_categories['conjugate_routines'] = self.list_completion(text, opt_conjg)
            
        return self.deal_multiple_categories(completion_categories)

    def complete_set(self, text, line, begidx, endidx):
        "Complete the set command"
        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = self.options.keys() 
            return self.list_completion(text, opts)

        if len(args) == 2:
            if args[1] in ['group_subprocesses', 'complex_mass_scheme']:
                return self.list_completion(text, ['False', 'True','default'])
            elif args[1] in ['ignore_six_quark_processes']:
                return self.list_completion(text, self._multiparticles.keys())
            elif args[1] == 'gauge':
                return self.list_completion(text, ['unitary', 'Feynman','default'])
            elif args[1] == 'stdout_level':
                return self.list_completion(text, ['DEBUG','INFO','WARNING','ERROR','CRITICAL','default'])
        
            elif args[1] == 'fortran_compiler':
                return self.list_completion(text, ['f77','g77','gfortran','default'])
            elif args[1] == 'nb_core':
                return self.list_completion(text, [str(i) for i in range(100)] + ['default'] )
            elif args[1] == 'run_mode':
                return self.list_completion(text, [str(i) for i in range(3)] + ['default'])
            elif args[1] == 'cluster_type':
                return self.list_completion(text, cluster.from_name.keys() + ['default'])
            elif args[1] == 'cluster_queue':
                return []
            elif args[1] == 'automatic_html_opening':
                return self.list_completion(text, ['False', 'True', 'default'])            
            else:
                # directory names
                second_set = [name for name in self.path_completion(text, '.', only_dirs = True)]
                return self.list_completion(text, second_set + ['default'])
        elif len(args) >2 and args[-1].endswith(os.path.sep):
                return self.path_completion(text,
                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True)
        
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        # Format
        if len(args) == 1:
            opt =  self.list_completion(text, self._import_formats)
            if opt:
                return opt
            mode = 'all'
        elif args[1] in self._import_formats:
            mode = args[1]
        else:
            args.insert(1, 'all')
            mode = 'all'


        completion_categories = {}
        # restriction continuation (for UFO)
        if mode in ['model', 'all'] and '-' in  text:
            # deal with - in readline splitting (different on some computer)
            path = '-'.join([part for part in text.split('-')[:-1]])
            # remove the final - for the model name
            # find the different possibilities
            all_name = self.find_restrict_card(path, no_restrict=False)
            all_name += self.find_restrict_card(path, no_restrict=False,
                                        base_dir=pjoin(MG5DIR,'models'))

            # select the possibility according to the current line           
            all_name = [name+' ' for name in  all_name if name.startswith(text)
                                                       and name.strip() != text]
            
                
            if all_name:
                completion_categories['Restricted model'] = all_name

        # Path continuation
        if os.path.sep in args[-1]:
            if mode.startswith('model') or mode == 'all':
                # Directory continuation
                try:
                    cur_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
                except:
                    pass
                else:
                    all_dir = self.path_completion(text, cur_path, only_dirs = True)
                    if mode in ['model_v4','all']:
                        completion_categories['Path Completion'] = all_dir
                    # Only UFO model here
                    new = []
                    data =   [new.__iadd__(self.find_restrict_card(name, base_dir=cur_path))
                                                                for name in all_dir]
                    if data:
                        completion_categories['Path Completion'] = all_dir + new
            else:
                try:
                    cur_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
                except:
                    pass
                else:
                    all_path =  self.path_completion(text, cur_path)
                    if mode == 'all':
                        new = [] 
                        data =   [new.__iadd__(self.find_restrict_card(name, base_dir=cur_path)) 
                                                               for name in all_path]
                        if data:
                            completion_categories['Path Completion'] = data[0]
                    else:
                        completion_categories['Path Completion'] = all_path
                
        # Model directory name if directory is not given
        if (len(args) == 2):
            is_model = True
            if mode == 'model':
                file_cond = lambda p : os.path.exists(pjoin(MG5DIR,'models',p,'particles.py'))
                mod_name = lambda name: name
            elif mode == 'model_v4':
                file_cond = lambda p :  (os.path.exists(pjoin(MG5DIR,'models',p,'particles.dat')) 
                                      or os.path.exists(pjoin(self._mgme_dir,'Models',p,'particles.dat')))
                mod_name = lambda name :(name[-3:] != '_v4' and name or name[:-3]) 
            elif mode == 'all':
                mod_name = lambda name: name
                file_cond = lambda p : os.path.exists(pjoin(MG5DIR,'models',p,'particles.py')) \
                                      or os.path.exists(pjoin(MG5DIR,'models',p,'particles.dat')) \
                                      or os.path.exists(pjoin(self._mgme_dir,'Models',p,'particles.dat')) 
            else:
                cur_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
                all_path =  self.path_completion(text, cur_path)
                completion_categories['model name'] = all_path
                is_model = False
            
            if is_model:
                model_list = [mod_name(name) for name in \
                                                self.path_completion(text,
                                                pjoin(MG5DIR,'models'),
                                                only_dirs = True) \
                                                if file_cond(name)]
                
                if mode == 'model_v4':
                    completion_categories['model name'] = model_list
                else:
                    # need to update the  list with the possible restriction
                    all_name = []
                    for model_name in model_list:
                        all_name += self.find_restrict_card(model_name, 
                                            base_dir=pjoin(MG5DIR,'models'))
                if mode == 'all':
                    cur_path = pjoin(*[a for a in args \
                                                        if a.endswith(os.path.sep)])
                    all_path =  self.path_completion(text, cur_path)
                    completion_categories['model name'] = all_path + all_name 
                elif mode == 'model':
                    completion_categories['model name'] = all_name 

        # Options
        if mode == 'all' and len(args)>1:
            mode = self.find_import_type(args[2])
   
        if len(args) >= 3 and mode.startswith('model') and not '-modelname' in line:
            if not text and not completion_categories:
                return ['--modelname']
            elif not (os.path.sep in args[-1] and line[-1] != ' '):
                completion_categories['options'] = self.list_completion(text, ['--modelname','-modelname'])
        if len(args) >= 3 and mode.startswith('banner') and not '--no_launch' in line:
            completion_categories['options'] = self.list_completion(text, ['--no_launch'])
        return self.deal_multiple_categories(completion_categories) 
        
        
            
    def find_restrict_card(self, model_name, base_dir='./', no_restrict=True):
        """find the restriction file associate to a given model"""

        # check if the model_name should be keeped as a possibility
        if no_restrict:
            output = [model_name]
        else:
            output = []
        
        # check that the model is a valid model
        if not os.path.exists(pjoin(base_dir, model_name, 'couplings.py')):
            # not valid UFO model
            return output
        
        if model_name.endswith(os.path.sep):
            model_name = model_name[:-1]
        
        # look for _default and treat this case
        if os.path.exists(pjoin(base_dir, model_name, 'restrict_default.dat')):
            output.append('%s-full' % model_name)
        
        # look for other restrict_file
        for name in os.listdir(pjoin(base_dir, model_name)):
            if name.startswith('restrict_') and not name.endswith('default.dat') \
                and name.endswith('.dat'):
                tag = name[9:-4] #remove restrict and .dat
                while model_name.endswith(os.path.sep):
                    model_name = model_name[:-1]
                output.append('%s-%s' % (model_name, tag))

        # return
        return output
    
    def complete_install(self, text, line, begidx, endidx):
        "Complete the import command"

        args = self.split_arg(line[0:begidx])
        
        # Format
        if len(args) == 1:
            return self.list_completion(text, self._install_opts)
        elif len(args) and args[0] == 'update':
            return self.list_completion(text, ['-f','--timeout='])     

#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmd(HelpToCmd, CheckValidForCmd, CompleteForCmd, CmdExtended):
    """The command line processor of MadGraph"""    

    writing_dir = '.'
  
    # Options and formats available
    _display_opts = ['particles', 'interactions', 'processes', 'diagrams', 
                     'diagrams_text', 'multiparticles', 'couplings', 'lorentz', 
                     'checks', 'parameters', 'options', 'coupling_order','variable']
    _add_opts = ['process']
    _save_opts = ['model', 'processes', 'options']
    _tutorial_opts = ['start', 'stop']
    _check_opts = ['full', 'permutation', 'gauge', 'lorentz_invariance']
    _import_formats = ['model_v4', 'model', 'proc_v4', 'command', 'banner']
    _install_opts = ['pythia-pgs', 'Delphes', 'MadAnalysis', 'ExRootAnalysis', 
                     'update', 'Delphes2']
    _v4_export_formats = ['madevent', 'standalone', 'matrix'] 
    _export_formats = _v4_export_formats + ['standalone_cpp', 'pythia8', 'aloha']
    _set_options = ['group_subprocesses',
                    'ignore_six_quark_processes',
                    'stdout_level',
                    'fortran_compiler',
                    'gauge',
                    'complex_mass_scheme']
    
    # The three options categories are treated on a different footage when a 
    # set/save configuration occur. current value are kept in self.options
    options_configuration = {'pythia8_path': './pythia8',
                       'madanalysis_path': './MadAnalysis',
                       'pythia-pgs_path':'./pythia-pgs',
                       'td_path':'./td',
                       'delphes_path':'./Delphes',
                       'exrootanalysis_path':'./ExRootAnalysis',
                       'timeout': 60,
                       'web_browser':None,
                       'eps_viewer':None,
                       'text_editor':None,
                       'fortran_compiler':None,
                       'auto_update':7,
                       'cluster_type': 'condor',
                       'cluster_temp_path': None,
                       'cluster_queue': None,
                       }
    
    options_madgraph= {'group_subprocesses': 'Auto',
                          'ignore_six_quark_processes': False,
                          'complex_mass_scheme': False,
                          'gauge':'unitary',
                          'stdout_level':None}
    options_madevent = {'automatic_html_opening':True,
                         'run_mode':2,
                         'nb_core': None,
                         }


    # Variables to store object information
    _curr_model = None  #base_objects.Model()
    _curr_amps = diagram_generation.AmplitudeList()
    _curr_matrix_elements = helas_objects.HelasMultiProcess()
    _curr_fortran_model = None
    _curr_cpp_model = None
    _curr_exporter = None
    _done_export = False
    _curr_decaymodel = None

    def preloop(self):
        """Initializing before starting the main loop"""

        self.prompt = 'mg5>'
        if madgraph.ReadWrite: # prevent on read-only disk  
            self.do_install('update --mode=mg5_start')
        
        # By default, load the UFO Standard Model
        logger.info("Loading default model: sm")
        self.exec_cmd('import model sm', printcmd=False, precmd=True)
        
        # preloop mother
        CmdExtended.preloop(self)

    
    def __init__(self, mgme_dir = '', *completekey, **stdin):
        """ add a tracker of the history """

        CmdExtended.__init__(self, *completekey, **stdin)
        
        # Set MG/ME directory path
        if mgme_dir:
            if os.path.isdir(pjoin(mgme_dir, 'Template')):
                self._mgme_dir = mgme_dir
                logger.info('Setting MG/ME directory to %s' % mgme_dir)
            else:
                logger.warning('Warning: Directory %s not valid MG/ME directory' % \
                             mgme_dir)
                self._mgme_dir = MG4DIR

        # Variables to store state information
        self._multiparticles = {}
        self.options = {}
        self._generate_info = "" # store the first generated process
        self._model_v4_path = None
        self._use_lower_part_names = False
        self._export_dir = None
        self._export_format = 'madevent'
        self._mgme_dir = MG4DIR
        self._comparisons = None

        # Load the configuration file
        self.set_configuration()

    def do_quit(self, line):
        """Do quit"""

        if self._done_export and \
                    os.path.exists(pjoin(self._done_export[0],'RunWeb')):
            os.remove(pjoin(self._done_export[0],'RunWeb'))
                
        value = super(MadGraphCmd, self).do_quit(line)
        if madgraph.ReadWrite: #prevent to run on Read Only disk
            self.do_install('update --mode=mg5_end')
        print

        return value
        
    # Add a process to the existing multiprocess definition
    # Generate a new amplitude
    def do_add(self, line):
        """Generate an amplitude for a given process and add to
        existing amplitudes
        """

        args = self.split_arg(line)
        
        # Check the validity of the arguments
        self.check_add(args)
        
        # special option for 1->N to avoid generation of kinematically forbidden
        #decay.
        if args[-1].startswith('--optimize'):
            optimize = True
            args.pop()
        else:
            optimize = False

        if args[0] == 'process':            
            # Rejoin line
            line = ' '.join(args[1:])
            
            # store the first process (for the perl script)
            if not self._generate_info:
                self._generate_info = line
                
            # Reset Helas matrix elements
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()

            # Extract process from process definition
            if ',' in line:
                myprocdef, line = self.extract_decay_chain_process(line)
            else:
                myprocdef = self.extract_process(line)

            # Check that we have something    
            if not myprocdef:
                raise self.InvalidCmd("Empty or wrong format process, please try again.")
            # Check that we have the same number of initial states as
            # existing processes
            if self._curr_amps and self._curr_amps[0].get_ninitial() != \
               myprocdef.get_ninitial():
                raise self.InvalidCmd("Can not mix processes with different number of initial states.")               
            
            cpu_time1 = time.time()

            # Generate processes
            if self.options['group_subprocesses'] == 'Auto':
                    collect_mirror_procs = True
            else:
                collect_mirror_procs = self.options['group_subprocesses']
            ignore_six_quark_processes = \
                           self.options['ignore_six_quark_processes'] if \
                           "ignore_six_quark_processes" in self.options \
                           else []

            

            myproc = diagram_generation.MultiProcess(myprocdef,
                                          collect_mirror_procs =\
                                          collect_mirror_procs,
                                          ignore_six_quark_processes = \
                                          ignore_six_quark_processes,
                                          optimize=optimize)

            for amp in myproc.get('amplitudes'):
                if amp not in self._curr_amps:
                    self._curr_amps.append(amp)
                else:
                    raise self.InvalidCmd, "Duplicate process %s found. Please check your processes." % \
                                                amp.nice_string_processes()


            # Reset _done_export, since we have new process
            self._done_export = False

            cpu_time2 = time.time()

            nprocs = len(myproc.get('amplitudes'))
            ndiags = sum([amp.get_number_of_diagrams() for \
                              amp in myproc.get('amplitudes')])
            logger.info("%i processes with %i diagrams generated in %0.3f s" % \
                  (nprocs, ndiags, (cpu_time2 - cpu_time1)))
            ndiags = sum([amp.get_number_of_diagrams() for \
                              amp in self._curr_amps])
            logger.info("Total: %i processes with %i diagrams" % \
                  (len(self._curr_amps), ndiags))        
                
  
    # Define a multiparticle label
    def do_define(self, line, log=True):
        """Define a multiparticle"""

        self.avoid_history_duplicate('define %s' % line, ['define'])
        if self._use_lower_part_names:
            # Particle names lowercase
            line = line.lower()
        # Make sure there are spaces around =, | and /
        line = line.replace("=", " = ")
        line = line.replace("|", " | ")
        line = line.replace("/", " / ")
        args = self.split_arg(line)
        # check the validity of the arguments
        self.check_define(args)

        label = args[0]
        remove_ids = []
        try:
            remove_index = args.index("/")
        except ValueError:
            pass
        else:
            remove_ids = args[remove_index + 1:]
            args = args[:remove_index]
        
        pdg_list = self.extract_particle_ids(args[1:])
        remove_list = self.extract_particle_ids(remove_ids)
        pdg_list = [p for p in pdg_list if p not in remove_list]

        self.optimize_order(pdg_list)
        self._multiparticles[label] = pdg_list
        if log:
            logger.info("Defined multiparticle %s" % \
                                             self.multiparticle_string(label))
    
    # Display
    def do_display(self, line, output=sys.stdout):
        """Display current internal status"""

        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_display(args)

        if args[0] == 'diagrams':
            self.draw(' '.join(args[1:]))

        if args[0] == 'particles' and len(args) == 1:
            propagating_particle = []
            nb_unpropagating = 0
            for particle in self._curr_model['particles']:
                if particle.get('propagating'):
                    propagating_particle.append(particle)
                else:
                    nb_unpropagating += 1
                    
            print "Current model contains %i particles:" % \
                    len(propagating_particle)
            part_antipart = [part for part in propagating_particle \
                             if not part['self_antipart']]
            part_self = [part for part in propagating_particle \
                             if part['self_antipart']]
            for part in part_antipart:
                print part['name'] + '/' + part['antiname'],
            print ''
            for part in part_self:
                print part['name'],
            print ''
            if nb_unpropagating:
                print 'In addition of %s un-physical particle mediating new interactions.' \
                                     % nb_unpropagating

        elif args[0] == 'particles':
            for arg in args[1:]:
                if arg.isdigit() or (arg[0] == '-' and arg[1:].isdigit()):
                    particle = self._curr_model.get_particle(abs(int(arg)))
                else:
                    particle = self._curr_model['particles'].find_name(arg)
                if not particle:
                    raise self.InvalidCmd, 'no particle %s in current model' % arg

                print "Particle %s has the following properties:" % particle.get_name()
                print str(particle)
            
        elif args[0] == 'interactions' and len(args) == 1:
            text = "Current model contains %i interactions\n" % \
                    len(self._curr_model['interactions'])
            for i, inter in enumerate(self._curr_model['interactions']):
                text += str(i+1) + ':'
                for part in inter['particles']:
                    if part['is_part']:
                        text += part['name']
                    else:
                        text += part['antiname']
                    text += " "
                text += " ".join(order + '=' + str(inter['orders'][order]) \
                                 for order in inter['orders'])
                text += '\n'
            pydoc.pager(text)

        elif args[0] == 'interactions' and len(args)==2 and args[1].isdigit():
            for arg in args[1:]:
                if int(arg) > len(self._curr_model['interactions']):
                    raise self.InvalidCmd, 'no interaction %s in current model' % arg
                if int(arg) == 0:
                    print 'Special interactions which identify two particles'
                else:
                    print "Interactions %s has the following property:" % arg
                    print self._curr_model['interactions'][int(arg)-1]

        elif args[0] == 'interactions':
            request_part = args[1:]
            text = ''
            for i, inter in enumerate(self._curr_model['interactions']):
                present_part = [part['is_part'] and part['name'] or part['antiname']
                                 for part in inter['particles']
                                if (part['is_part'] and  part['name'] in request_part) or
                                   (not part['is_part'] and part['antiname'] in request_part)]
                if len(present_part) < len(request_part):
                    continue
                # check that all particles are selected at least once
                if set(present_part) != set(request_part):
                    continue
                # check if a particle is asked more than once
                if len(request_part) > len(set(request_part)):
                    for p in request_part:
                        if request_part.count(p) > present_part.count(p):
                            continue
                        
                name = str(i+1) + ' : '
                for part in inter['particles']:
                    if part['is_part']:
                        name += part['name']
                    else:
                        name += part['antiname']
                    name += " "                   
                text += "\nInteractions %s has the following property:\n" % name
                text += str(self._curr_model['interactions'][i]) 

                text += '\n'
                print name
            if text =='':
                text += 'No matching for any interactions'
            pydoc.pager(text)


        elif args[0] == 'parameters' and len(args) == 1:
            text = "Current model contains %i parameters\n" % \
                    sum([len(part) for part in 
                                       self._curr_model['parameters'].values()])
            keys = self._curr_model['parameters'].keys()
            def key_sort(x, y):
                if ('external',) == x:
                    return -1
                elif ('external',) == y:
                    return +1
                elif  len(x) < len(y):
                    return -1
                else:
                    return 1
            keys.sort(key_sort)
            for key in keys:
                item = self._curr_model['parameters'][key]
                text += '\nparameter type: %s\n' % str(key)
                for value in item:
                    if hasattr(value, 'expr'):
                        if value.value is not None:
                            text+= '        %s = %s = %s\n' % (value.name, value.expr ,value.value)
                        else:
                            text+= '        %s = %s\n' % (value.name, value.expr)
                    else:
                        if value.value is not None:
                            text+= '        %s = %s\n' % (value.name, value.value)
                        else:
                            text+= '        %s \n' % (value.name)
            pydoc.pager(text)
            
        elif args[0] == 'processes':
            for amp in self._curr_amps:
                print amp.nice_string_processes()

        elif args[0] == 'diagrams_text':
            text = "\n".join([amp.nice_string() for amp in self._curr_amps])
            pydoc.pager(text)

        elif args[0] == 'multiparticles':
            print 'Multiparticle labels:'
            for key in self._multiparticles:
                print self.multiparticle_string(key)
                
        elif args[0] == 'coupling_order':
            hierarchy = self._curr_model['order_hierarchy'].items()
            #self._curr_model.get_order_hierarchy().items()
            def order(first, second):
                if first[1] < second[1]:
                    return -1
                else:
                    return 1
            hierarchy.sort(order)
            for order in hierarchy:
                print ' %s : weight = %s' % order 
            
        elif args[0] == 'couplings' and len(args) == 1:
            if self._model_v4_path:
                print 'No couplings information available in V4 model'
                return
            text = ''
            text = "Current model contains %i couplings\n" % \
                    sum([len(part) for part in 
                                        self._curr_model['couplings'].values()])
            keys = self._curr_model['couplings'].keys()
            def key_sort(x, y):
                if ('external',) == x:
                    return -1
                elif ('external',) == y:
                    return +1
                elif  len(x) < len(y):
                    return -1
                else:
                    return 1
            keys.sort(key_sort)
            for key in keys:
                item = self._curr_model['couplings'][key]
                text += '\ncouplings type: %s\n' % str(key)
                for value in item:
                    if value.value is not None:
                        text+= '        %s = %s = %s\n' % (value.name, value.expr ,value.value)
                    else:
                        text+= '        %s = %s\n' % (value.name, value.expr)

            pydoc.pager(text)

        elif args[0] == 'couplings':
            if self._model_v4_path:
                print 'No couplings information available in V4 model'
                return
            
            try:
                ufomodel = ufomodels.load_model(self._curr_model.get('name'))
                print 'Note that this is the UFO informations.'
                print ' "display couplings" present the actual definition'
                print 'prints the current states of mode'
                print eval('ufomodel.couplings.%s.nice_string()'%args[1])
            except:
                raise self.InvalidCmd, 'no couplings %s in current model' % args[1]
        
        elif args[0] == 'lorentz':
            if self._model_v4_path:
                print 'No lorentz information available in V4 model'
                return
            elif len(args) == 1: 
                raise self.InvalidCmd,\
                     'display lorentz require an argument: the name of the lorentz structure.'
                return
            try:
                ufomodel = ufomodels.load_model(self._curr_model.get('name'))
                print eval('ufomodel.lorentz.%s.nice_string()'%args[1])
            except:
                raise self.InvalidCmd, 'no lorentz %s in current model' % args[1]
            
        elif args[0] == 'checks':
            comparisons = self._comparisons[0]
            if len(args) > 1 and args[1] == 'failed':
                comparisons = [c for c in comparisons if not c['passed']]
            outstr = "Process check results:"
            for comp in comparisons:
                outstr += "\n%s:" % comp['process'].nice_string()
                outstr += "\n   Phase space point: (px py pz E)"
                for i, p in enumerate(comp['momenta']):
                    outstr += "\n%2s    %+.9e  %+.9e  %+.9e  %+.9e" % tuple([i] + p)
                outstr += "\n   Permutation values:"
                outstr += "\n   " + str(comp['values'])
                if comp['passed']:
                    outstr += "\n   Process passed (rel. difference %.9e)" % \
                          comp['difference']
                else:
                    outstr += "\n   Process failed (rel. difference %.9e)" % \
                          comp['difference']

            used_aloha = sorted(self._comparisons[1])
            outstr += "\nChecked ALOHA routines:"
            for aloha in used_aloha:
                aloha_str = aloha[0]
                if aloha[1]:
                    aloha_str += 'C' + 'C'.join([str(ia) for ia in aloha[1]])
                aloha_str += "_%d" % aloha[2]
                outstr += "\n" + aloha_str

            pydoc.pager(outstr)            
        
        elif args[0] == 'options':
            outstr = "                          MadGraph Options    \n"
            outstr += "                          ----------------    \n"
            for key, default in self.options_madgraph.items():
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)
            outstr += "\n"
            outstr += "                         MadEvent Options    \n"
            outstr += "                          ----------------    \n"
            for key, default in self.options_madevent.items():
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)  
            outstr += "\n"                 
            outstr += "                      Configuration Options    \n"
            outstr += "                      ---------------------    \n"
            for key, default in self.options_configuration.items():
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)

            output.write(outstr)
        elif args[0] in  ["variable"]:
            super(MadGraphCmd, self).do_display(line, output)
                
            
    def multiparticle_string(self, key):
        """Returns a nicely formatted string for the multiparticle"""

        if self._multiparticles[key] and \
               isinstance(self._multiparticles[key][0], list):
            return "%s = %s" % (key, "|".join([" ".join([self._curr_model.\
                                     get('particle_dict')[part_id].get_name() \
                                                     for part_id in id_list]) \
                                  for id_list in self._multiparticles[key]]))
        else:
            return "%s = %s" % (key, " ".join([self._curr_model.\
                                    get('particle_dict')[part_id].get_name() \
                                    for part_id in self._multiparticles[key]]))
            
  

    def do_tutorial(self, line):
        """Activate/deactivate the tutorial mode."""

        args = self.split_arg(line)
        if len(args) > 0 and args[0] == "stop":
            logger_tuto.info("\n\tThanks for using the tutorial!")
            logger_tuto.setLevel(logging.ERROR)
        else:
            logger_tuto.setLevel(logging.INFO)

        if not self._mgme_dir:
            logger_tuto.info(\
                       "\n\tWarning: To use all features in this tutorial, " + \
                       "please run from a" + \
                       "\n\t         valid MG_ME directory.")

    def draw(self, line):
        """ draw the Feynman diagram for the given process """

        args = self.split_arg(line)
        # Check the validity of the arguments
        self.check_draw(args)
        
        # Check if we plot a decay chain 
        if any([isinstance(a, diagram_generation.DecayChainAmplitude) for \
               a in self._curr_amps]) and not self._done_export:
            warn = 'WARNING: You try to draw decay chain diagrams without first running output.\n'
            warn += '\t  The decay processes will be drawn separately'
            logger.warning(warn)

        (options, args) = _draw_parser.parse_args(args)
        options = draw_lib.DrawOption(options)
        start = time.time()
        
        # Collect amplitudes
        amplitudes = diagram_generation.AmplitudeList()

        for amp in self._curr_amps:
            amplitudes.extend(amp.get_amplitudes())            

        for amp in amplitudes:
            filename = pjoin(args[0], 'diagrams_' + \
                                    amp.get('process').shell_string() + ".eps")
            plot = draw.MultiEpsDiagramDrawer(amp['diagrams'],
                                          filename,
                                          model=self._curr_model,
                                          amplitude='',
                                          legend=amp.get('process').input_string())

            logger.info("Drawing " + \
                         amp.get('process').nice_string())
            plot.draw(opt=options)
            logger.info("Wrote file " + filename)
            self.exec_cmd('open %s' % filename)

        stop = time.time()
        logger.info('time to draw %s' % (stop - start)) 
    
    # Generate a new amplitude
    def do_check(self, line):
        """Check a given process or set of processes"""

        args = self.split_arg(line)

        # Check args validity
        param_card = self.check_check(args)
        energy = float(args[-1].split('=')[1])
        args = args[:-1]
        line = " ".join(args[1:])
        
        myprocdef = self.extract_process(line)

        # Check that we have something    
        if not myprocdef:
            raise self.InvalidCmd("Empty or wrong format process, please try again.")

        # Disable diagram generation logger
        diag_logger = logging.getLogger('madgraph.diagram_generation')
        old_level = diag_logger.getEffectiveLevel()
        diag_logger.setLevel(logging.WARNING)

        # run the check
        cpu_time1 = time.time()
        # Run matrix element generation check on processes
        mass_scheme = self.options['complex_mass_scheme']

        comparisons = []
        gauge_result = []
        gauge_result_no_brs = []
        lorentz_result =[]
        nb_processes = 0
        
        if args[0] in  ['permutation', 'full']:
            comparisons = process_checks.check_processes(myprocdef,
                                                        param_card = param_card,
                                                        quick = True,
                                                        energy=energy)
            nb_processes += len(comparisons[0])

        if args[0] in ['lorentz_invariance', 'full']:
            lorentz_result = process_checks.check_lorentz(myprocdef,
                                                      param_card = param_card,
                                                      cmass_scheme = mass_scheme,
                                                      energy=energy)
            nb_processes += len(lorentz_result)
            
        if args[0] in  ['gauge', 'full']:
            gauge_result = process_checks.check_gauge(myprocdef,
                                                      param_card = param_card,
                                                      cmass_scheme = mass_scheme,
                                                      energy=energy)
            nb_processes += len(gauge_result)

        if args[0] in  ['gauge', 'full'] and len(self._curr_model.get('gauge')) == 2:
            
            gauge = str(self.options['gauge'])
            line = " ".join(args[1:])
            myprocdef = self.extract_process(line)
            model_name = self._curr_model['name']
            if gauge == 'unitary':
                myprocdef_unit = myprocdef
                self.do_set('gauge Feynman', log=False)
                self.do_import('model %s' % model_name)
                myprocdef_feyn = self.extract_process(line)
            else:
                myprocdef_feyn = myprocdef
                self.do_set('gauge unitary', log=False)
                self.do_import('model %s' % model_name)
                myprocdef_unit = self.extract_process(line)            
            
            nb_part_unit = len(myprocdef_unit.get('model').get('particles'))
            nb_part_feyn = len(myprocdef_feyn.get('model').get('particles'))
            
            if nb_part_feyn == nb_part_unit:
                logger.error('No Goldstone present for this check!!')
            gauge_result_no_brs = process_checks.check_unitary_feynman(
                                                myprocdef_unit, myprocdef_feyn,
                                                param_card = param_card,
                                                cmass_scheme = mass_scheme,
                                                energy=energy)
            
            
            # restore previous settings
            self.do_set('gauge %s' % gauge, log=False)
            self.do_import('model %s' % model_name)
            
            nb_processes += len(gauge_result_no_brs)            
            
            
        cpu_time2 = time.time()

        logger.info("%i checked performed in %0.3f s" \
                    % (nb_processes,
                      (cpu_time2 - cpu_time1)))
        if mass_scheme:
            text = "Note that Complex mass scheme gives gauge/lorentz invariant\n"
            text+= "results only for stable particles in final states.\n\n"
        else:
            text = "Note That all width have been set to zero for those checks\n\n"
            
        if gauge_result:
            text += 'Gauge results:\n'
            text += process_checks.output_gauge(gauge_result) + '\n'
        if gauge_result_no_brs:
            text += 'Gauge results (switching between Unitary/Feynman):\n'
            text += process_checks.output_unitary_feynman(gauge_result_no_brs) + '\n'

        if lorentz_result:
            text += 'Lorentz invariance results:\n'
            text += process_checks.output_lorentz_inv(lorentz_result) + '\n'

        if comparisons:
            text += 'Process permutation results:\n'
            text += process_checks.output_comparisons(comparisons[0]) + '\n'
            self._comparisons = comparisons

        logger.info(text)
        pydoc.pager(text)
        # Restore diagram logger
        diag_logger.setLevel(old_level)
        
        # clean the globals created.
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
    
    # Generate a new amplitude
    def do_generate(self, line):
        """Generate an amplitude for a given process"""

        # Reset amplitudes
        self._curr_amps = diagram_generation.AmplitudeList()
        # Reset Helas matrix elements
        self._curr_matrix_elements = None
        self._generate_info = line
        # Reset _done_export, since we have new process
        self._done_export = False
        # Also reset _export_format and _export_dir
        self._export_format = None

        # Remove previous generations from history
        self.clean_history(remove_bef_last='generate', keep_switch=True,
                     allow_for_removal= ['generate', 'add process', 'output'])


        # Call add process
        args = self.split_arg(line)
        args.insert(0, 'process')
        
        self.do_add(" ".join(args))
    
    def extract_process(self, line, proc_number = 0, overall_orders = {}):
        """Extract a process definition from a string. Returns
        a ProcessDefinition."""

        # Check basic validity of the line
        if not line.count('>') in [1,2]:
            self.do_help('generate')
            print
            raise self.InvalidCmd('Wrong use of \">\" special character.')
        

        # Perform sanity modifications on the lines:
        # Add a space before and after any > , $ / |
        space_before = re.compile(r"(?P<carac>\S)(?P<tag>[/\,\\$\\>|])(?P<carac2>\S)")
        line = space_before.sub(r'\g<carac> \g<tag> \g<carac2>', line)       
        
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

        # Then take coupling orders (identified by "=")
        order_pattern = re.compile("^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(line)
        orders = {}
        while order_re:
            orders[order_re.group(2)] = int(order_re.group(3))
            line = order_re.group(1)
            order_re = order_pattern.match(line)

        if self._use_lower_part_names:
            # Particle names lowercase
            line = line.lower()

        # Now check for forbidden particles, specified using "/"
        slash = line.find("/")
        dollar = line.find("$")
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

        # Now check for forbidden schannels, specified using "$$"
        forbidden_schannels_re = re.match("^(.+)\s*\$\s*\$\s*(.+)\s*$", line)
        forbidden_schannels = ""
        if forbidden_schannels_re:
            forbidden_schannels = forbidden_schannels_re.group(2)
            line = forbidden_schannels_re.group(1)

        # Now check for forbidden onshell schannels, specified using "$"
        forbidden_onsh_schannels_re = re.match("^(.+)\s*\$\s*(.+)\s*$", line)
        forbidden_onsh_schannels = ""
        if forbidden_onsh_schannels_re:
            forbidden_onsh_schannels = forbidden_onsh_schannels_re.group(2)
            line = forbidden_onsh_schannels_re.group(1)

        # Now check for required schannels, specified using "> >"
        required_schannels_re = re.match("^(.+?)>(.+?)>(.+)$", line)
        required_schannels = ""
        if required_schannels_re:
            required_schannels = required_schannels_re.group(2)
            line = required_schannels_re.group(1) + ">" + \
                   required_schannels_re.group(3)

        args = self.split_arg(line)

        myleglist = base_objects.MultiLegList()
        state = False

        # Extract process
        for part_name in args:
            if part_name == '>':
                if not myleglist:
                    raise self.InvalidCmd, "No final state particles"
                state = True
                continue

            mylegids = []
            if part_name in self._multiparticles:
                if isinstance(self._multiparticles[part_name][0], list):
                    raise self.InvalidCmd,\
                          "Multiparticle %s is or-multiparticle" % part_name + \
                          " which can be used only for required s-channels"
                mylegids.extend(self._multiparticles[part_name])
            else:
                mypart = self._curr_model['particles'].get_copy(part_name)
                if mypart:
                    mylegids.append(mypart.get_pdg_code())

            if mylegids:
                myleglist.append(base_objects.MultiLeg({'ids':mylegids,
                                                        'state':state}))
            else:
                raise self.InvalidCmd, \
                      "No particle %s in model" % part_name

        if filter(lambda leg: leg.get('state') == True, myleglist):
            # We have a valid process

            # Now extract restrictions
            forbidden_particle_ids = \
                              self.extract_particle_ids(forbidden_particles)
            if forbidden_particle_ids and \
               isinstance(forbidden_particle_ids[0], list):
                raise self.InvalidCmd,\
                      "Multiparticle %s is or-multiparticle" % part_name + \
                      " which can be used only for required s-channels"
            forbidden_onsh_schannel_ids = \
                              self.extract_particle_ids(forbidden_onsh_schannels)
            forbidden_schannel_ids = \
                              self.extract_particle_ids(forbidden_schannels)
            if forbidden_onsh_schannel_ids and \
               isinstance(forbidden_onsh_schannel_ids[0], list):
                raise self.InvalidCmd,\
                      "Multiparticle %s is or-multiparticle" % part_name + \
                      " which can be used only for required s-channels"
            if forbidden_schannel_ids and \
               isinstance(forbidden_schannel_ids[0], list):
                raise self.InvalidCmd,\
                      "Multiparticle %s is or-multiparticle" % part_name + \
                      " which can be used only for required s-channels"
            required_schannel_ids = \
                               self.extract_particle_ids(required_schannels)
            if required_schannel_ids and not \
                   isinstance(required_schannel_ids[0], list):
                required_schannel_ids = [required_schannel_ids]
            

            return \
                base_objects.ProcessDefinition({'legs': myleglist,
                              'model': self._curr_model,
                              'id': proc_number,
                              'orders': orders,
                              'forbidden_particles': forbidden_particle_ids,
                              'forbidden_onsh_s_channels': forbidden_onsh_schannel_ids,
                              'forbidden_s_channels': \
                                                forbidden_schannel_ids,
                              'required_s_channels': required_schannel_ids,
                              'overall_orders': overall_orders
                              })
      #                       'is_decay_chain': decay_process\

    def extract_particle_ids(self, args):
        """Extract particle ids from a list of particle names. If
        there are | in the list, this corresponds to an or-list, which
        is represented as a list of id lists. An or-list is used to
        allow multiple required s-channel propagators to be specified
        (e.g. Z/gamma)."""

        if isinstance(args, basestring):
            args.replace("|", " | ")
            args = self.split_arg(args)
        all_ids = []
        ids=[]
        for part_name in args:
            mypart = self._curr_model['particles'].get_copy(part_name)
            if mypart:
                ids.append([mypart.get_pdg_code()])
            elif part_name in self._multiparticles:
                ids.append(self._multiparticles[part_name])
            elif part_name == "|":
                # This is an "or-multiparticle"
                if ids:
                    all_ids.append(ids)
                ids = []
            else:
                raise self.InvalidCmd("No particle %s in model" % part_name)
        all_ids.append(ids)
        # Flatten id list, to take care of multiparticles and
        # or-multiparticles
        res_lists = []
        for i, id_list in enumerate(all_ids):
            res_lists.extend(diagram_generation.expand_list_list(id_list))
        # Trick to avoid duplication while keeping ordering
        for ilist, idlist in enumerate(res_lists):
            set_dict = {}
            res_lists[ilist] = [set_dict.setdefault(i,i) for i in idlist \
                         if i not in set_dict]

        if len(res_lists) == 1:
            res_lists = res_lists[0]

        return res_lists

    def optimize_order(self, pdg_list):
        """Optimize the order of particles in a pdg list, so that
        similar particles are next to each other. Sort according to:
        1. pdg > 0, 2. spin, 3. color, 4. mass > 0"""

        if not pdg_list:
            return
        if not isinstance(pdg_list[0], int):
            return
        
        model = self._curr_model
        pdg_list.sort(key = lambda i: i < 0)
        pdg_list.sort(key = lambda i: model.get_particle(i).is_fermion())
        pdg_list.sort(key = lambda i: model.get_particle(i).get('color'),
                      reverse = True)
        pdg_list.sort(key = lambda i: \
                      model.get_particle(i).get('mass').lower() != 'zero')

    def extract_decay_chain_process(self, line, level_down=False):
        """Recursively extract a decay chain process definition from a
        string. Returns a ProcessDefinition."""

        # Start with process number (identified by "@") and overall orders
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*((\w+\s*=\s*\d+\s*)*)$")
        proc_number_re = proc_number_pattern.match(line)
        proc_number = 0
        overall_orders = {}
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            line = proc_number_re.group(1)
            if proc_number_re.group(3):
                order_pattern = re.compile("^(.*?)\s*(\w+)\s*=\s*(\d+)\s*$")
                order_line = proc_number_re.group(3)
                order_re = order_pattern.match(order_line)
                while order_re:
                    overall_orders[order_re.group(2)] = int(order_re.group(3))
                    order_line = order_re.group(1)
                    order_re = order_pattern.match(order_line)
            logger.info(line)            
            
        index_comma = line.find(",")
        index_par = line.find(")")
        min_index = index_comma
        if index_par > -1 and (index_par < min_index or min_index == -1):
            min_index = index_par
        
        if min_index > -1:
            core_process = self.extract_process(line[:min_index], proc_number,
                                                overall_orders)
        else:
            core_process = self.extract_process(line, proc_number,
                                                overall_orders)

        #level_down = False

        while index_comma > -1:
            line = line[index_comma + 1:]
            if not line.strip():
                break
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
                    raise self.InvalidCmd, \
                      "Missing ending parenthesis for decay process"

                if index_par < index_comma:
                    line = line[index_par + 1:]
                    level_down = False
                    break

        if level_down:
            index_par = line.find(')')
            if index_par == -1:
                raise self.InvalidCmd, \
                      "Missing ending parenthesis for decay process"
            line = line[index_par + 1:]
            
        # Return the core process (ends recursion when there are no
        # more decays)
        return core_process, line
    

    # Import files
    def do_import(self, line):
        """Import files with external formats"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_import(args)
        
        if args[0].startswith('model'):
            self._model_v4_path = None
            # Clear history, amplitudes and matrix elements when a model is imported
            # Remove previous imports, generations and outputs from history
            self.clean_history(remove_bef_last='import', keep_switch=True,
                        allow_for_removal=['generate', 'add process', 'output'])
            # Reset amplitudes and matrix elements
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            # Import model
            if args[0].endswith('_v4'):
                self._curr_model, self._model_v4_path = \
                                 import_v4.import_model(args[1], self._mgme_dir)
                self._curr_fortran_model = \
                      helas_call_writers.FortranHelasCallWriter(\
                                                               self._curr_model)
            else:
                try:
                    self._curr_model = import_ufo.import_model(args[1])
                except import_ufo.UFOImportError, error:
                    if 'not a valid UFO model' in str(error):
                        logger_stderr.warning('WARNING: %s' % error)
                        logger_stderr.warning('Try to recover by running automatically `import model_v4 %s` instead.' \
                                                                      % args[1])
                    self.exec_cmd('import model_v4 %s ' % args[1], precmd=True)
                    return    
                if self.options['complex_mass_scheme']:
                    self._curr_model.change_mass_to_complex_scheme()
                    if hasattr(self._curr_model, 'set_parameters_and_couplings'):
                        self._curr_model.set_parameters_and_couplings()
                if self.options['gauge']=='unitary':
                    if 0 not in self._curr_model.get('gauge') :
                        logger.warning('Change the gauge to Feynman since the model does not allow unitary gauge') 
                        self.do_set('gauge Feynman', log=False)
                        self.do_import(line)
                        return                        
                else:
                    if 1 not in self._curr_model.get('gauge') :
                        logger.warning('Change the gauge to unitary since the model does not allow Feynman gauge')
                        self._curr_model = None
                        self.do_set('gauge unitary', log= False)
                        self.do_import(line)
                        return 
                
                self._curr_fortran_model = \
                      helas_call_writers.FortranUFOHelasCallWriter(\
                                                               self._curr_model)
                self._curr_cpp_model = \
                      helas_call_writers.CPPUFOHelasCallWriter(\
                                                               self._curr_model)

            if '-modelname' not in args:
                self._curr_model.pass_particles_name_in_mg_default()

            # Do post-processing of model
            self.process_model()
            # Reset amplitudes and matrix elements and global checks
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            process_checks.store_aloha = []
            
        elif args[0] == 'command':

            if not os.path.isfile(args[1]):
                raise self.InvalidCmd("Path %s is not a valid pathname" % args[1])
            else:
                # Check the status of export and try to use file position if no
                #self._export dir are define
                self.check_for_export_dir(args[1])
                # Execute the card
                self.import_command_file(args[1])
                            
        elif args[0] == 'banner':
            type = madevent_interface.MadEventCmd.detect_card_type(args[1])    
            if type != 'banner':
                raise self.InvalidCmd, 'The File should be a valid banner'
            ban = banner_module.Banner(args[1])
            # Check that this is MG5 banner
            if 'mg5proccard' in ban:
                for line in ban['mg5proccard'].split('\n'):
                    if line.startswith('#') or line.startswith('<'):
                        continue
                    self.exec_cmd(line)
            else:
                raise self.InvalidCmd, 'Only MG5 banner are supported'
            
            if not self._done_export:
                self.exec_cmd('output . -f')

            ban.split(self._done_export[0])
            logger.info('All Cards from the banner have been place in directory %s' % pjoin(self._done_export[0], 'Cards'))
            if '--no_launch' not in args:
                self.exec_cmd('launch')
            
            
        elif args[0] == 'proc_v4':
            self.history = []

            if len(args) == 1 and self._export_dir:
                proc_card = pjoin(self._export_dir, 'Cards', \
                                                                'proc_card.dat')
            elif len(args) == 2:
                proc_card = args[1]
                # Check the status of export and try to use file position is no
                # self._export dir are define
                self.check_for_export_dir(os.path.realpath(proc_card))
            else:
                raise MadGraph5Error('No default directory in output')

 
            #convert and excecute the card
            self.import_mg4_proc_card(proc_card)
            
    
    def import_ufo_model(self, model_name):
        """ import the UFO model """
        
        self._curr_model = import_ufo.import_model(model_name)
        self._curr_fortran_model = \
                helas_call_writers.FortranUFOHelasCallWriter(self._curr_model)
        self._curr_cpp_model = \
                helas_call_writers.CPPUFOHelasCallWriter(self._curr_model)
                
    def process_model(self):
        """Set variables _particle_names and _couplings for tab
        completion, define multiparticles"""

         # Set variables for autocomplete
        self._particle_names = [p.get('name') for p in self._curr_model.get('particles')\
                                                    if p.get('propagating')] + \
                 [p.get('antiname') for p in self._curr_model.get('particles') \
                                                    if p.get('propagating')]
        
        self._couplings = list(set(sum([i.get('orders').keys() for i in \
                                        self._curr_model.get('interactions')], [])))
        # Check if we can use case-independent particle names
        self._use_lower_part_names = \
            (self._particle_names == \
             [p.get('name').lower() for p in self._curr_model.get('particles')] + \
             [p.get('antiname').lower() for p in self._curr_model.get('particles')])

        self.add_default_multiparticles()
        
    
    def import_mg4_proc_card(self, filepath):
        """ read a V4 proc card, convert it and run it in mg5"""
        
        # change the status of this line in the history -> pass in comment
        if self.history and self.history[-1].startswith('import proc_v4'):
            self.history[-1] = '#%s' % self.history[-1]
         
        # read the proc_card.dat
        reader = files.read_from_file(filepath, import_v4.read_proc_card_v4)
        if not reader:
            raise self.InvalidCmd('\"%s\" is not a valid path' % filepath)
        
        if self._mgme_dir:
            # Add comment to history
            self.exec_cmd("# Import the model %s" % reader.model, precmd=True)
            line = self.exec_cmd('import model_v4 %s -modelname' % \
                                 (reader.model), precmd=True)
        else:
            logging.error('No MG_ME installation detected')
            return    


        # Now that we have the model we can split the information
        lines = reader.extract_command_lines(self._curr_model)

        for line in lines:
            self.exec_cmd(line, precmd=True)
    
        return 
    
    def add_default_multiparticles(self):
        """ add default particle from file interface.multiparticles_default.txt
        """
        
        defined_multiparticles = self._multiparticles.keys()
        removed_multiparticles = []
        # First check if the defined multiparticles are allowed in the
        # new model
        for key in self._multiparticles.keys():
            try:
                for part in self._multiparticles[key]:
                    self._curr_model.get('particle_dict')[part]
            except:
                del self._multiparticles[key]
                defined_multiparticles.remove(key)
                removed_multiparticles.append(key)
        
        # Now add default multiparticles
        for line in open(pjoin(MG5DIR, 'input', \
                                      'multiparticles_default.txt')):
            if line.startswith('#'):
                continue
            try:
                if self._use_lower_part_names:
                    multipart_name = line.lower().split()[0]
                else:
                    multipart_name = line.split()[0]
                if multipart_name not in self._multiparticles:
                    #self.do_define(line)
                    self.exec_cmd('define %s' % line, printcmd=False, precmd=True)
            except self.InvalidCmd, why:
                logger_stderr.warning('impossible to set default multiparticles %s because %s' %
                                        (line.split()[0],why))
        if defined_multiparticles:
            if 'all' in defined_multiparticles:
                defined_multiparticles.remove('all')
            logger.info("Kept definitions of multiparticles %s unchanged" % \
                                         " / ".join(defined_multiparticles))

        for removed_part in removed_multiparticles:
            if removed_part in self._multiparticles:
                removed_multiparticles.remove(removed_part)

        if removed_multiparticles:
            logger.info("Removed obsolete multiparticles %s" % \
                                         " / ".join(removed_multiparticles))
        
        # add all tag
        line = []
        for part in self._curr_model.get('particles'):
            line.append('%s %s' % (part.get('name'), part.get('antiname')))
        line = 'all =' + ' '.join(line)
        self.do_define(line)

    def do_install(self, line):
        """Install optional package from the MG suite."""
        
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_install(args)

        if sys.platform == "darwin":
            program = "curl"
        else:
            program = "wget"
        
        # special command for auto-update
        if args[0] == 'update':
            self.install_update(args, wget=program)
            return
           
        # Load file with path of the different program:
        import urllib
        path = {}
        try:
            data = urllib.urlopen('http://madgraph.phys.ucl.ac.be/package_info.dat')
        except:
            raise MadGraph5Error, '''Impossible to connect the server. 
            Please check your internet connection or retry later'''
        for line in data: 
            split = line.split()   
            path[split[0]] = split[1]
        
        if args[0] == 'Delphes':
            args[0] = 'Delphes3'
        
        name = {'td_mac': 'td', 'td_linux':'td', 'Delphes3':'Delphes',
                'pythia-pgs':'pythia-pgs', 'ExRootAnalysis': 'ExRootAnalysis',
                'MadAnalysis':'MadAnalysis', 'Delphes2': 'Delphes'}
        name = name[args[0]]
        
        try:
            os.system('rm -rf %s' % pjoin(MG5DIR, name))
        except:
            pass
        
        # Load that path
        logger.info('Downloading %s' % path[args[0]])
        if sys.platform == "darwin":
            misc.call(['curl', path[args[0]], '-o%s.tgz' % name], cwd=MG5DIR)
        else:
            misc.call(['wget', path[args[0]], '--output-document=%s.tgz'% name], cwd=MG5DIR)
        # Untar the file
        returncode = misc.call(['tar', '-xzpf', '%s.tgz' % name], cwd=MG5DIR, 
                                     stdout=open(os.devnull, 'w'))
        if returncode:
            raise MadGraph5Error, 'Fail to download correctly the File. Stop'
        
        
        # Check that the directory has the correct name
        if not os.path.exists(pjoin(MG5DIR, name)):
            created_name = [n for n in os.listdir(MG5DIR) if n.startswith(name) 
                                                  and not n.endswith('gz')]
            if not created_name:
                raise MadGraph5Error, 'The file was not loaded correctly. Stop'
            else:
                created_name = created_name[0]
            files.mv(pjoin(MG5DIR, created_name), pjoin(MG5DIR, name))
        logger.info('compile %s. This might takes a while.' % name)
        
        # Modify Makefile for pythia-pgs on Mac 64 bit
        if args[0] == "pythia-pgs":
            if sys.maxsize > 2**32:
                path = os.path.join(MG5DIR, 'pythia-pgs', 'src', 'make_opts')
                text = open(path).read()
                text = text.replace('MBITS=32','MBITS=64')
                open(path, 'w').writelines(text)
            if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs', 'libraries','pylib','lib')):
                os.mkdir(pjoin(MG5DIR, 'pythia-pgs', 'libraries','pylib','lib'))
                
        # Compile the file
        # Check for F77 compiler
        if 'FC' not in os.environ or not os.environ['FC']:
            if self.options['fortran_compiler'] and self.options['fortran_compiler'] != 'None':
                compiler = self.options['fortran_compiler']
            elif misc.which('gfortran'):
                compiler = 'gfortran'
            elif misc.which('g77'):
                compiler = 'g77'
            else:
                raise self.InvalidCmd('Require g77 or Gfortran compiler')
            if compiler == 'gfortran' and args[0] == "pythia-pgs":
                path = os.path.join(MG5DIR, 'pythia-pgs', 'src', 'make_opts')
                text = open(path).read()
                text = text.replace('FC=g77','FC=gfortran')
                open(path, 'w').writelines(text)    
            elif compiler == 'gfortran' and args[0] == 'MadAnalysis':
                path = os.path.join(MG5DIR, 'MadAnalysis', 'makefile')
                text = open(path).read()
                text = text.replace('FC=g77','FC=gfortran')
                open(path, 'w').writelines(text)
                            
        if logger.level <= logging.INFO:
            devnull = open(os.devnull,'w') 
            misc.call(['make', 'clean'], stdout=devnull, stderr=-2)
            if name == 'pythia-pgs':
                #SLC6 needs to have this first (don't ask why)
                status = misc.call(['make'], cwd = pjoin(MG5DIR, name, 'libraries', 'pylib'))
            status = misc.call(['make'], cwd = os.path.join(MG5DIR, name))
        else:
            self.compile(['clean'], mode='', cwd = os.path.join(MG5DIR, name))
            if name == 'pythia-pgs':
                #SLC6 needs to have this first (don't ask why)
                status = self.compile(mode='', cwd = pjoin(MG5DIR, name, 'libraries', 'pylib'))
            status = self.compile(mode='', cwd = os.path.join(MG5DIR, name))
        if not status:
            logger.info('compilation succeeded')
        else:
            logger.warning('Error detected during the compilation. Please check the compilation error and run make manually.')


        # Special treatment for TD/Ghostscript program (require by MadAnalysis)
        if args[0] == 'MadAnalysis':
            try:
                os.system('rm -rf td')
                os.mkdir(pjoin(MG5DIR, 'td'))
            except Exception, error:
                print error
                pass
            
            if sys.platform == "darwin":
                logger.info('Downloading TD for Mac')
                target = 'http://theory.fnal.gov/people/parke/TD/td_mac_intel.tar.gz'
                misc.call(['curl', target, '-otd.tgz'], 
                                                  cwd=pjoin(MG5DIR,'td'))      
                misc.call(['tar', '-xzpvf', 'td.tgz'], 
                                                  cwd=pjoin(MG5DIR,'td'))
                files.mv(MG5DIR + '/td/td_mac_intel',MG5DIR+'/td/td')
            else:
                logger.info('Downloading TD for Linux 32 bit')
                target = 'http://madgraph.phys.ucl.ac.be/Downloads/td'
                misc.call(['wget', target], cwd=pjoin(MG5DIR,'td'))      
                os.chmod(pjoin(MG5DIR,'td','td'), 0775)
                if sys.maxsize > 2**32:
                    logger.warning('''td program (needed by MadAnalysis) is not compile for 64 bit computer
                Please follow instruction in https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/TopDrawer .''')
            
            if not misc.which('gs'):
                logger.warning('''gosthscript not install on your system. This is not required to run MA.
                    but this prevent to create jpg files and therefore to have the plots in the html output.''')
                if sys.platform == "darwin":
                    logger.warning('''You can download this program at the following link: 
                    http://www.macupdate.com/app/mac/9980/gpl-ghostscript''')
            
        if args[0] == 'Delphes2':
            data = open(pjoin(MG5DIR, 'Delphes','data','DetectorCard.dat')).read()
            data = data.replace('data/', 'DELPHESDIR/data/')
            out = open(pjoin(MG5DIR, 'Template', 'Cards', 'delphes_card_default.dat'), 'w')
            out.write(data)
        if args[0] == 'Delphes3':
            files.cp(pjoin(MG5DIR, 'Delphes','examples','delphes_card_CMS.tcl'),
                     pjoin(MG5DIR,'Template', 'Cards', 'delphes_card_default.dat'))  
        

    def install_update(self, args, wget):
        """ check if the current version of mg5 is up-to-date. 
        and allow user to install the latest version of MG5 """

        # load options
        mode = [arg.split('=',1)[1] for arg in args if arg.startswith('--mode=')]
        if mode:
            mode = mode[-1]
        else:
            mode = "userrequest"
        force = any([arg=='-f' for arg in args])
        timeout = [arg.split('=',1)[1] for arg in args if arg.startswith('--timeout=')]
        if timeout:
            try:
                timeout = int(timeout[-1])
            except ValueError:
                raise self.InvalidCmd('%s: invalid argument for timeout (integer expected)'%timeout[-1])
        else:
            timeout = self.options['timeout']
        
        options = ['y','n','on_exit']
        if mode == 'mg5_start':
            timeout = 2
            default = 'n'
            update_delay = self.options['auto_update'] * 24 * 3600
            if update_delay == 0:
                return
        elif mode == 'mg5_end':
            timeout = 5
            default = 'n'
            update_delay = self.options['auto_update'] * 24 * 3600
            if update_delay == 0:
                return
            options.remove('on_exit')
        elif mode == "userrequest":
            default = 'y'
            update_delay = 0
        else:
            raise self.InvalidCmd('Unknown mode for command install update')
        
        if not os.path.exists(os.path.join(MG5DIR,'input','.autoupdate')) or \
                os.path.exists(os.path.join(MG5DIR,'.bzr')):
            error_text = """This version of MG5 doesn\'t support auto-update. Common reasons are:
            1) This version was loaded via bazaar (use bzr pull to update instead).
            2) This version is a beta release of MG5."""
            if mode == 'userrequest':
                raise self.ConfigurationError(error_text)
            return 
        
        if not misc.which('patch'):
            error_text = """Not able to find program \'patch\'. Please reload a clean version
            or install that program and retry."""
            if mode == 'userrequest':
                raise self.ConfigurationError(error_text)
            return            
        
        
        # read the data present in .autoupdate
        data = {}
        for line in open(os.path.join(MG5DIR,'input','.autoupdate')):
            if not line.strip():
                continue
            sline = line.split()
            data[sline[0]] = int(sline[1])

        #check validity of the file
        if 'version_nb' not in data:
            if mode == 'userrequest':
                error_text = 'This version of MG5 doesn\'t support auto-update. (Invalid information)'
                raise self.ConfigurationError(error_text)
            return
        elif 'last_check' not in data:
            data['last_check'] = time.time()
        
        #check if we need to update.
        if time.time() - data['last_check'] < update_delay:
            return
        
        logger.info('Checking if MG5 is up-to-date... (takes up to %ss)' % timeout)
        class TimeOutError(Exception): pass
        
        def handle_alarm(signum, frame): 
            raise TimeOutError
        
        signal.signal(signal.SIGALRM, handle_alarm)
        signal.alarm(timeout)
        to_update = 0
        try:
            filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/mg5_build_nb')
            signal.alarm(0)
            web_version = int(filetext.read().strip())            
        except (TimeOutError, ValueError, IOError):
            signal.alarm(0)
            print 'failed to connect server'
            if mode == 'mg5_end':
                # wait 24h before next check
                fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
                fsock.write("version_nb   %s\n" % data['version_nb'])
                fsock.write("last_check   %s\n" % \
                int(time.time()) - 3600 * 24 * (self.options['auto_update'] -1))
                fsock.close()
            return
        
        if web_version == data['version_nb']:
            logger.info('No new version of MG5 available')
            # update .autoupdate to prevent a too close check
            fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
            fsock.write("version_nb   %s\n" % data['version_nb'])
            fsock.write("last_check   %s\n" % int(time.time()))
            fsock.close()
            return
        elif data['version_nb'] > web_version:
            logger_stderr.info('impossible to update: local %s web %s' % (data['version_nb'], web_version))
            fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
            fsock.write("version_nb   %s\n" % data['version_nb'])
            fsock.write("last_check   %s\n" % int(time.time()))
            fsock.close()
            return
        else:
            if not force:
                answer = self.ask('New Version of MG5 available! Do you want to update your current version?',
                                  default, options)
            else:
                answer = default

        
        if answer == 'y':
            logger.info('start updating code')
            fail = 0
            for i in range(data['version_nb'], web_version):
                try:
                    filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/patch/build%s.patch' %(i+1))
#                    filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/patch_test/build%s.patch' %(i+1))
                except:
                    print 'fail to load patch to build #%s' % (i+1)
                    fail = i
                    break
                print 'apply patch %s' % (i+1)
                text = filetext.read()
                p= subprocess.Popen(['patch', '-p1'], stdin=subprocess.PIPE, 
                                                                  cwd=MG5DIR)
                p.communicate(text)
            
            fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
            if not fail:
                fsock.write("version_nb   %s\n" % web_version)
            else:
                fsock.write("version_nb   %s\n" % fail)
            fsock.write("last_check   %s\n" % int(time.time()))
            fsock.close()
            logger.info('Checking current version. (type ctrl-c to bypass the check)')
            subprocess.call([os.path.join('tests','test_manager.py')],
                                                                  cwd=MG5DIR)
            
            print 'new version installed, please relaunch mg5'
            sys.exit(0)
        elif answer == 'n':
            # prevent for a future check
            fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
            fsock.write("version_nb   %s\n" % data['version_nb'])
            fsock.write("last_check   %s\n" % int(time.time()))
            fsock.close()
            logger.info('Update bypassed.')
            logger.info('The next check for a new version will be performed in %s days' \
                        % abs(self.options['auto_update']))
            logger.info('In order to change this delay. Enter the command:')
            logger.info('set auto_update X')
            logger.info('Putting X to zero will prevent this check at anytime.')
            logger.info('You can upgrade your version at any time by typing:')
            logger.info('install update')
        else: #answer is on_exit
            #ensure that the test will be done on exit
            #Do not use the set command here!!
            self.options['auto_update'] = -1 * self.options['auto_update']


    
    def set_configuration(self, config_path=None, final=True):
        """ assign all configuration variable from file 
            ./input/mg5_configuration.txt. assign to default if not define """

        if not self.options:
            self.options = dict(self.options_configuration)
            self.options.update(self.options_madgraph)
            self.options.update(self.options_madevent) 

        if not config_path:
            if os.environ.has_key('MADGRAPH_BASE'):
                config_path = pjoin(os.environ['MADGRAPH_BASE'],'mg5_configuration.txt')
                self.set_configuration(config_path, final)
                return
            if 'HOME' in os.environ:
                config_path = pjoin(os.environ['HOME'],'.mg5', 
                                                        'mg5_configuration.txt')
                if os.path.exists(config_path):
                    self.set_configuration(config_path, final=False)
            config_path = os.path.relpath(pjoin(MG5DIR,'input',
                                                       'mg5_configuration.txt'))     
            return self.set_configuration(config_path, final)
        
        if not os.path.exists(config_path):
            files.cp(pjoin(MG5DIR,'input','.mg5_configuration_default.txt'), config_path)
        config_file = open(config_path)

        # read the file and extract information
        logger.info('load MG5 configuration from %s ' % config_file.name)
        for line in config_file:
            if '#' in line:
                line = line.split('#',1)[0]
            line = line.replace('\n','').replace('\r\n','')
            try:
                name, value = line.split('=')
            except ValueError:
                pass
            else:
                name = name.strip()
                value = value.strip()
                if name != 'mg5_path':
                    self.options[name] = value
                if value.lower() == "none":
                    self.options[name] = None

        self.options['stdout_level'] = logging.getLogger('madgraph').level
        if not final:
            return self.options # the return is usefull for unittest

        # Treat each expected input
        # 1: Pythia8_path
        # try relative path
        for key in self.options:
            if key == 'pythia8_path':
                if self.options['pythia8_path'] in ['None', None]:
                    self.options['pythia8_path'] = None
                    continue
                pythia8_dir = pjoin(MG5DIR, self.options['pythia8_path'])
                if not os.path.isfile(pjoin(pythia8_dir, 'include', 'Pythia.h')):
                    if not os.path.isfile(pjoin(self.options['pythia8_path'], 'include', 'Pythia.h')):
                        self.options['pythia8_path'] = None
                    else:
                        continue
                    
            elif key.endswith('path'):
                pass
            elif key in ['run_mode', 'auto_update']:
                self.options[key] = int(self.options[key])
            elif key in ['cluster_type','automatic_html_opening']:
                pass
            elif key not in ['text_editor','eps_viewer','web_browser', 'stdout_level']:
                # Default: try to set parameter
                try:
                    self.do_set("%s %s --no_save" % (key, self.options[key]), log=False)
                except MadGraph5Error, error:
                    print error
                    logger.warning("Option %s from config file not understood" \
                                   % key)
                else:
                    if key in self.options_madgraph:
                        self.history.append('set %s %s' % (key, self.options[key]))             
        
        # Configure the way to open a file:
        launch_ext.open_file.configure(self.options)
          
        return self.options
     
    def check_for_export_dir(self, filepath):
        """Check if the files is in a valid export directory and assign it to
        export path if if is"""

        # keep previous if a previous one is defined
        if self._export_dir:
            return
        
        if os.path.exists(pjoin(os.getcwd(), 'Cards')):    
            self._export_dir = os.getcwd()
            return
    
        path_split = filepath.split(os.path.sep)
        if len(path_split) > 2 and path_split[-2] == 'Cards':
            self._export_dir = os.path.sep.join(path_split[:-2])
            return

    def do_launch(self, line):
        """Ask for editing the parameter and then 
        Execute the code (madevent/standalone/...)
        """
        
        start_cwd = os.getcwd()
        
        args = self.split_arg(line)
        # check argument validity and normalise argument
        (options, args) = _launch_parser.parse_args(args)
        self.check_launch(args, options)
        options = options.__dict__
        # args is now MODE PATH

        if args[0].startswith('standalone'):
            ext_program = launch_ext.SALauncher(self, args[1], options=self.options, **options)
        elif args[0] == 'madevent':
            if options['interactive']:
                if hasattr(self, 'do_shell'):
                    ME = madevent_interface.MadEventCmdShell(me_dir=args[1], options=self.options)
                else:
                    ME = madevent_interface.MadEventCmd(me_dir=args[1],options=self.options)
                    ME.pass_in_web_mode()
                stop = self.define_child_cmd_interface(ME)                
                return stop
            
            #check if this is a cross-section
            if not self._generate_info:
                # This relaunch an old run -> need to check if this is a 
                # cross-section or a width
                info = open(pjoin(args[1],'SubProcesses','procdef_mg5.dat')).read()
                generate_info = info.split('# Begin PROCESS',1)[1].split('\n')[1]
                generate_info = generate_info.split('#')[0]
            else:
                generate_info = self._generate_info
            
            if len(generate_info.split('>')[0].strip().split())>1:
                ext_program = launch_ext.MELauncher(args[1], self,
                                shell = hasattr(self, 'do_shell'),
                                options=self.options,**options)
            else:
                # This is a width computation
                ext_program = launch_ext.MELauncher(args[1], self, unit='GeV',
                                shell = hasattr(self, 'do_shell'),
                                options=self.options,**options)

        elif args[0] == 'pythia8':
            ext_program = launch_ext.Pythia8Launcher( args[1], self, **options)
        else:
            os.chdir(start_cwd) #ensure to go to the initial path
            raise self.InvalidCmd , '%s cannot be run from MG5 interface' % args[0]
        
        
        ext_program.run()
        os.chdir(start_cwd) #ensure to go to the initial path
        
        
        
    
    def do_load(self, line):
        """Not in help: Load information from file"""

        args = self.split_arg(line)
        # check argument validity
        self.check_load(args)

        cpu_time1 = time.time()
        if args[0] == 'model':
            self._curr_model = save_load_object.load_from_file(args[1])
            if self._curr_model.get('parameters'):
                # This is a UFO model
                self._model_v4_path = None
                self._curr_fortran_model = \
                  helas_call_writers.FortranUFOHelasCallWriter(self._curr_model)
            else:
                # This is a v4 model
                self._model_v4_path = import_v4.find_model_path(\
                    self._curr_model.get('name').replace("_v4", ""),
                    self._mgme_dir)
                self._curr_fortran_model = \
                  helas_call_writers.FortranHelasCallWriter(self._curr_model)

            # Do post-processing of model
            self.process_model()
                
            #save_model.save_model(args[1], self._curr_model)
            if isinstance(self._curr_model, base_objects.Model):
                cpu_time2 = time.time()
                logger.info("Loaded model from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1))
            else:
                raise self.RWError('Could not load model from file %s' \
                                      % args[1])
        elif args[0] == 'processes':
            amps = save_load_object.load_from_file(args[1])
            if isinstance(amps, diagram_generation.AmplitudeList):
                cpu_time2 = time.time()
                logger.info("Loaded processes from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1))
                if amps:
                    model = amps[0].get('process').get('model')
                    if not model.get('parameters'):
                        # This is a v4 model.  Look for path.
                        self._model_v4_path = import_v4.find_model_path(\
                                   model.get('name').replace("_v4", ""),
                                   self._mgme_dir)
                        self._curr_fortran_model = \
                                helas_call_writers.FortranHelasCallWriter(\
                                                              model)
                    else:
                        self._model_v4_path = None
                        self._curr_fortran_model = \
                                helas_call_writers.FortranUFOHelasCallWriter(\
                                                              model)
                    # If not exceptions from previous steps, set
                    # _curr_amps and _curr_model
                    self._curr_amps = amps                    
                    self._curr_model = model
                    logger.info("Model set from process.")
                    # Do post-processing of model
                    self.process_model()
                self._done_export = None
            else:
                raise self.RWError('Could not load processes from file %s' % args[1])
    
    
    def do_customize_model(self, line):
        """create a restriction card in a interactive way"""

        args = self.split_arg(line)
        self.check_customize_model(args)

        try:
            model_path = import_ufo.find_ufo_path(self._curr_model.get('name'))
        except import_ufo.UFOImportError:
            name = self._curr_model.get('name').rsplit('-',1)[0]
            try:
                model_path = import_ufo.find_ufo_path(name)
            except import_ufo.UFOImportError:
                print name
                raise self.InvalidCmd('''Invalid model.''')
                
        if not os.path.exists(pjoin(model_path,'build_restrict.py')):
            raise self.InvalidCmd('''Model not compatible with this option.''')
        
        # (re)import the full model (get rid of the default restriction)
        self._curr_model = import_ufo.import_full_model(model_path)
        
        #1) create the full param_card
        out_path = StringIO.StringIO()
        param_writer.ParamCardWriter(self._curr_model, out_path)
        # and load it to a python object
        param_card = check_param_card.ParamCard(out_path.getvalue().split('\n'))
        
        #2) Import the option available in the model
        ufo_model = ufomodels.load_model(model_path)
        all_categories = ufo_model.build_restrict.all_categories
        
        #3) making the options
        def change_options(name, all_categories):
            for category in all_categories:
                for options in category:            
                    if options.name == name:
                        options.status = not options.status

        # asking the question to the user                        
        while 1:
            question = ''
            answers = ['0']
            cat = {} 
            for category in all_categories:
                question += category.name + ':\n'
                for options in category:
                    if not options.first:
                        continue
                    question += '    %s: %s [%s]\n' % (len(answers), options.name, 
                                options.display(options.status))
                    cat[str(len(answers))] = options.name
                    answers.append(len(answers))
            question += 'Enter a number to change it\'s status or press enter to validate'
            answers.append('done')
            value = self.ask(question,'0',answers)
            if value not in ['0','done']:
                change_options(cat[value], all_categories)
            else:
                break

        ## Make a Temaplate for  the restriction card. (card with no restrict)
        for block in param_card:
            value_dict = {}
            for param in param_card[block]:
                value = param.value
                if value == 0:
                    param.value = 0.000001e-99
                elif value == 1:
                    param.value = 9.999999e-1                
                elif abs(value) in value_dict:
                    param.value += value_dict[abs(value)] * 1e-4 * param.value
                    value_dict[abs(value)] += 1
                else:
                    value_dict[abs(value)] = 1 
        
        for category in all_categories:
            for options in category:
                if not options.status:
                    continue
                param = param_card[options.lhablock].get(options.lhaid)
                param.value = options.value
        
        logger.info('Loading the resulting model')
        # Applying the restriction 
        self._curr_model = import_ufo.RestrictModel(self._curr_model)
        model_name = self._curr_model.get('name')
        if model_name == 'mssm':
            keep_external=True
        else:
            keep_external=False
        self._curr_model.restrict_model(param_card,keep_external=keep_external)
        
        if args:
            name = args[0].split('=',1)[1]
            path = pjoin(model_path,'restrict_%s.dat' % name)
            logger.info('Save restriction file as %s' % path)
            param_card.write(path)
            self._curr_model['name'] += '-%s' % name
        
        
        
        
        
        
        
        
    
    
    def do_save(self, line, check=True, to_keep={}, log=True):
        """Not in help: Save information to file"""

        args = self.split_arg(line)
        # Check argument validity
        if check:
            self.check_save(args)

        if args[0] == 'model':
            if self._curr_model:
                #save_model.save_model(args[1], self._curr_model)
                if save_load_object.save_to_file(args[1], self._curr_model):
                    logger.info('Saved model to file %s' % args[1])
            else:
                raise self.InvalidCmd('No model to save!')
        elif args[0] == 'processes':
            if self._curr_amps:
                if save_load_object.save_to_file(args[1], self._curr_amps):
                    logger.info('Saved processes to file %s' % args[1])
            else:
                raise self.InvalidCmd('No processes to save!')
        
        elif args[0] == 'options':
            # First look at options which should be put in MG5DIR/input
            to_define = {}
            for key, default in self.options_configuration.items():
                if  self.options_configuration[key] != self.options[key] != None:
                    to_define[key] = self.options[key]
                
            if not '--auto' in args:
                for key, default in self.options_madevent.items():
                    if self.options_madevent[key] != self.options[key] != None:
                        to_define[key] = self.options[key]
                    elif key == 'cluster_queue' and self.options[key] is None:
                        to_define[key] = self.options[key]
                        
            if '--all' in args:
                for key, default in self.options_madgraph.items():
                    if self.options_madgraph[key] != self.options[key] != None and \
                      key != 'stdout_level':
                        to_define[key] = self.options[key]
            elif not '--auto' in args:
                for key, default in self.options_madgraph.items():
                    if self.options_madgraph[key] != self.options[key] != None and  key != 'stdout_level':
                        logger.info('The option %s is modified [%s] but will not be written in the configuration files.' \
                                    % (key,self.options_madgraph[key]) )
                        logger.info('If you want to make this value the default for future session, you can run \'save options --all\'')
            if len(args) >1 and not args[1].startswith('--'):
                filepath = args[1]
            else:
                filepath = pjoin(MG5DIR, 'input', 'mg5_configuration.txt')
            basefile = pjoin(MG5DIR, 'input', '.mg5_configuration_default.txt')
            basedir = MG5DIR
            
            if to_keep:
                to_define = to_keep
            self.write_configuration(filepath, basefile, basedir, to_define)
    
    # Set an option
    def do_set(self, line, log=True):
        """Set an option, which will be default for coming generations/outputs
        """
        # Be carefull:        
        # This command is associated to a post_cmd: post_set.
        args = self.split_arg(line)
        
        # Check the validity of the arguments
        self.check_set(args)
            
        if args[0] == 'ignore_six_quark_processes':
            if args[1] == 'False':
                self.options[args[0]] = False
                return
            self.options[args[0]] = list(set([abs(p) for p in \
                                      self._multiparticles[args[1]]\
                                      if self._curr_model.get_particle(p).\
                                      is_fermion() and \
                                      self._curr_model.get_particle(abs(p)).\
                                      get('color') == 3]))
            if log:
                logger.info('Ignore processes with >= 6 quarks (%s)' % \
                        ",".join([\
                            self._curr_model.get_particle(q).get('name') \
                            for q in self.options[args[0]]]))
            
        elif args[0] == 'group_subprocesses':
            if args[1] != 'Auto':
                self.options[args[0]] = eval(args[1])
            else:
                self.options[args[0]] = 'Auto'
            if log:
                logger.info('Set group_subprocesses to %s' % \
                        str(self.options[args[0]]))
                logger.info('Note that you need to regenerate all processes')
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()

        elif args[0] == "stdout_level":
            if args[1].isdigit():
                level = int(args[1])
            else:
                level = eval('logging.' + args[1])
            logging.root.setLevel(level)
            logging.getLogger('madgraph').setLevel(level)
            logging.getLogger('madevent').setLevel(level)
            if log:
                logger.info('set output information to level: %s' % level)

        elif args[0] == "complex_mass_scheme":
            old = self.options[args[0]] 
            self.options[args[0]] = eval(args[1])
            aloha.complex_mass = eval(args[1])
            aloha_lib.KERNEL.clean()
            if not self._curr_model:
                pass
            elif self.options[args[0]]:
                if old:
                    if log:
                        logger.info('Complex mass already activated.')
                    return
                if log:
                    logger.info('Activate complex mass scheme.')
                self._curr_model.change_mass_to_complex_scheme()
                if hasattr(self._curr_model, 'set_parameters_and_couplings'):
                        self._curr_model.set_parameters_and_couplings()
            else:
                if not old:
                    if log:
                        logger.info('Complex mass already desactivated.')
                    return
                if log:
                    logger.info('Desactivate complex mass scheme.')
                self.exec_cmd('import model %s' % self._curr_model.get('name'))

        elif args[0] == "gauge":
            
            # Treat the case where they are no model loaded
            if not self._curr_model:
                if args[1] == 'unitary':
                    aloha.unitary_gauge = True
                else:
                    aloha.unitary_gauge = False
                aloha_lib.KERNEL.clean()
                self.options[args[0]] = args[1]
                if log: logger.info('Pass to gauge %s.' % args[1])
                return
            
            # They are a valid model
            able_to_mod = True
            if args[1] == 'unitary':
                if 1 in self._curr_model.get('gauge'):		   
                    aloha.unitary_gauge = True
                else:
                    able_to_mod = False
                    if log: logger.warning('Note that unitary gauge is not allowed for your current model %s' \
		                                     % self._curr_model.get('name'))
            else:
                if 0 in self._curr_model.get('gauge'):		   
                    aloha.unitary_gauge = False
                else:
                    able_to_mod = False
                    if log: logger.warning('Note that Feynman gauge is not allowed for your current model %s' \
		                                     % self._curr_model.get('name'))
            self.options[args[0]] = args[1]

            #re-init all variable
            model_name = self._curr_model.get('name')
            self._curr_model = None
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            self._curr_fortran_model = None
            self._curr_cpp_model = None
            self._curr_exporter = None
            self._done_export = False
            import_ufo._import_once = []
            logger.info('Pass to gauge %s.' % args[1])
            
            if able_to_mod:
                self.do_import('model %s' % model_name)
            elif log:
                logger.info('Note that you have to reload the model') 

		
        elif args[0] == 'fortran_compiler':
            if args[1] != 'None':
                if log:
                    logger.info('set fortran compiler to %s' % args[1])
                self.options['fortran_compiler'] = args[1]
            else:
                self.options['fortran_compiler'] = None
        elif args[0] in ['timeout', 'auto_update']:
                self.options[args[0]] = int(args[1]) 
        elif args[0] in self.options:
            if args[1] in ['None','True','False']:
                self.options[args[0]] = eval(args[1])
            else:
                self.options[args[0]] = args[1]             

    def post_set(self, stop, line):
        """Check if we need to save this in the option file"""
        
        args = self.split_arg(line)
        # Check the validity of the arguments
        try:
            self.check_set(args, log=False)
        except Exception:
            return stop
        
        if args[0] in self.options_configuration and '--no_save' not in args:
            self.exec_cmd('save options --auto', log=False)
        elif args[0] in self.options_madevent:
            if not '--no_save' in line:
                logger.info('This option will be the default in any output that you are going to create in this session.')
                logger.info('In order to keep this changes permanent please run \'save options\'')
        else:
            #madgraph configuration
            if not self.history or self.history[-1].split() != line.split():
                self.history.append('set %s' % line)
                self.avoid_history_duplicate('set %s' % args[0], ['define', 'set']) 
        return stop

    def do_open(self, line):
        """Open a text file/ eps file / html file"""
        
        args = self.split_arg(line)
        # Check Argument validity and modify argument to be the real path
        self.check_open(args)
        file_path = args[0]
        
        launch_ext.open_file(file_path)
                 
    def do_output(self, line):
        """Initialize a new Template or reinitialize one"""

        args = self.split_arg(line)
        # Check Argument validity
        self.check_output(args)

        # Remove previous outputs from history
        self.clean_history(allow_for_removal = ['output'], keep_switch=True,
                           remove_bef_last='output')
        
        noclean = '-noclean' in args
        force = '-f' in args 
        nojpeg = '-nojpeg' in args
        main_file_name = ""
        try:
            main_file_name = args[args.index('-name') + 1]
        except:
            pass
        
        ################
        # ALOHA OUTPUT #
        ################
        if self._export_format == 'aloha':
            # catch format
            format = [d[9:] for d in args if d.startswith('--format=')]
            if not format:
                format = 'Fortran'
            else:
                format = format[-1]
            # catch output dir
            output = [d for d in args if d.startswith('--output=')]
            if not output:
                output = import_ufo.find_ufo_path(self._curr_model['name'])
                output = pjoin(output, format)
                if not os.path.isdir(output):
                    os.mkdir(output)
            else:
                output = output[-1]
                if not os.path.isdir(output):
                    raise self.InvalidCmd('%s is not a valid directory' % output)
            logger.info('creating routines in directory %s ' % output)
            # build the calling list for aloha
            names = [d for d in args if not d.startswith('-')]
            wanted_lorentz = aloha_fct.guess_routine_from_name(names)
            # Create and write ALOHA Routine
            aloha_model = create_aloha.AbstractALOHAModel(self._curr_model.get('name'))
            aloha_model.add_Lorentz_object(self._curr_model.get('lorentz'))
            if wanted_lorentz:
                aloha_model.compute_subset(wanted_lorentz)
            else:
                aloha_model.compute_all(save=False)
            aloha_model.write(output, format)
            return

        #################
        ## Other Output #
        #################
        if not force and not noclean and os.path.isdir(self._export_dir)\
               and self._export_format in ['madevent', 'standalone']:
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % self._export_dir)
            logger.info('If you continue this directory will be deleted and replaced.')
            answer = self.ask('Do you want to continue?', 'y', ['y','n'])
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')
            else:
                shutil.rmtree(self._export_dir)

        # Make a Template Copy
        if self._export_format in ['madevent', 'standalone', 'matrix']:
            self._curr_exporter = export_v4.ExportV4Factory(self, noclean)
        elif self._export_format == 'standalone_cpp':
            export_cpp.setup_cpp_standalone_dir(self._export_dir, self._curr_model)
        if self._export_format not in \
                ['madevent', 'standalone', 'standalone_cpp'] and \
                not os.path.isdir(self._export_dir):
            os.makedirs(self._export_dir)

        if self._export_format in ['madevent', 'standalone']:
            self._curr_exporter.copy_v4template(modelname=self._curr_model.get('name'))

        # Reset _done_export, since we have new directory
        self._done_export = False

        # Perform export and finalize right away
        self.export(nojpeg, main_file_name)

        # Automatically run finalize
        self.finalize(nojpeg)

        # Remember that we have done export
        self._done_export = (self._export_dir, self._export_format)

        # Reset _export_dir, so we don't overwrite by mistake later
        self._export_dir = None

    # Export a matrix element
    def export(self, nojpeg = False, main_file_name = ""):
        """Export a generated amplitude to file"""

        def generate_matrix_elements(self):
            """Helper function to generate the matrix elements before
            exporting"""

            # Sort amplitudes according to number of diagrams,
            # to get most efficient multichannel output
            self._curr_amps.sort(lambda a1, a2: a2.get_number_of_diagrams() - \
                                 a1.get_number_of_diagrams())

            # Check if we need to group the SubProcesses or not
            group = True
            if self.options['group_subprocesses'] is False:
                group = False
            elif self.options['group_subprocesses'] == 'Auto' and \
                                         self._curr_amps[0].get_ninitial() == 1:
                group = False 



            cpu_time1 = time.time()
            ndiags = 0
            if not self._curr_matrix_elements.get_matrix_elements():
                if group:
                    cpu_time1 = time.time()
                    dc_amps = diagram_generation.DecayChainAmplitudeList(\
                        [amp for amp in self._curr_amps if isinstance(amp, \
                                        diagram_generation.DecayChainAmplitude)])
                    non_dc_amps = diagram_generation.AmplitudeList(\
                             [amp for amp in self._curr_amps if not \
                              isinstance(amp, \
                                         diagram_generation.DecayChainAmplitude)])
                    subproc_groups = group_subprocs.SubProcessGroupList()
                    if non_dc_amps:
                        subproc_groups.extend(\
                                   group_subprocs.SubProcessGroup.group_amplitudes(\
                                                                       non_dc_amps))
                    if dc_amps:
                        dc_subproc_group = \
                                 group_subprocs.DecayChainSubProcessGroup.\
                                                           group_amplitudes(dc_amps)
                        subproc_groups.extend(\
                                  dc_subproc_group.\
                                        generate_helas_decay_chain_subproc_groups())

                    ndiags = sum([len(m.get('diagrams')) for m in \
                              subproc_groups.get_matrix_elements()])
                    self._curr_matrix_elements = subproc_groups
                    # assign a unique id number to all groups
                    uid = 0 
                    for group in subproc_groups:
                        uid += 1 # update the identification number
                        for me in group.get('matrix_elements'):
                            me.get('processes')[0].set('uid', uid)
                else: # Not grouped subprocesses
                    self._curr_matrix_elements = \
                        helas_objects.HelasMultiProcess(self._curr_amps)
                    ndiags = sum([len(me.get('diagrams')) for \
                                  me in self._curr_matrix_elements.\
                                  get_matrix_elements()])
                    # assign a unique id number to all process
                    uid = 0 
                    for me in self._curr_matrix_elements.get_matrix_elements():
                        uid += 1 # update the identification number
                        me.get('processes')[0].set('uid', uid)
                        
            cpu_time2 = time.time()


            return ndiags, cpu_time2 - cpu_time1

        # Start of the actual routine

        ndiags, cpu_time = generate_matrix_elements(self)

        calls = 0

        path = self._export_dir
        if self._export_format in ['standalone_cpp', 'madevent', 'standalone']:
            path = pjoin(path, 'SubProcesses')
            
        cpu_time1 = time.time()

        # First treat madevent and pythia8 exports, where we need to
        # distinguish between grouped and ungrouped subprocesses

        # MadEvent
        if self._export_format == 'madevent':
            if isinstance(self._curr_matrix_elements, group_subprocs.SubProcessGroupList):
                for (group_number, me_group) in enumerate(self._curr_matrix_elements):
                    calls = calls + \
                         self._curr_exporter.generate_subprocess_directory_v4(\
                                me_group, self._curr_fortran_model,
                                group_number)
            else:
                for me_number, me in \
                   enumerate(self._curr_matrix_elements.get_matrix_elements()):
                    calls = calls + \
                            self._curr_exporter.generate_subprocess_directory_v4(\
                                me, self._curr_fortran_model, me_number)


            # Write the procdef_mg5.dat file with process info
            card_path = pjoin(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path,
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except:
                    pass
                
        # Pythia 8
        if self._export_format == 'pythia8':
            # Output the process files
            process_names = []
            if isinstance(self._curr_matrix_elements, group_subprocs.SubProcessGroupList):
                for (group_number, me_group) in enumerate(self._curr_matrix_elements):
                    exporter = export_cpp.generate_process_files_pythia8(\
                            me_group.get('matrix_elements'), self._curr_cpp_model,
                            process_string = me_group.get('name'),
                            process_number = group_number, path = path)
                    process_names.append(exporter.process_name)
            else:
                exporter = export_cpp.generate_process_files_pythia8(\
                            self._curr_matrix_elements, self._curr_cpp_model,
                            process_string = self._generate_info, path = path)
                process_names.append(exporter.process_file_name)

            # Output the model parameter and ALOHA files
            model_name, model_path = export_cpp.convert_model_to_pythia8(\
                            self._curr_model, self._export_dir)

            # Generate the main program file
            filename, make_filename = \
                      export_cpp.generate_example_file_pythia8(path,
                                                               model_path,
                                                               process_names,
                                                               exporter,
                                                               main_file_name)

        # Pick out the matrix elements in a list
        matrix_elements = \
                        self._curr_matrix_elements.get_matrix_elements()

        # Fortran MadGraph Standalone
        if self._export_format == 'standalone':
            for me in matrix_elements:
                calls = calls + \
                        self._curr_exporter.generate_subprocess_directory_v4(\
                            me, self._curr_fortran_model)

        # Just the matrix.f files
        if self._export_format == 'matrix':
            for me in matrix_elements:
                filename = pjoin(path, 'matrix_' + \
                           me.get('processes')[0].shell_string() + ".f")
                if os.path.isfile(filename):
                    logger.warning("Overwriting existing file %s" % filename)
                else:
                    logger.info("Creating new file %s" % filename)
                calls = calls + self._curr_exporter.write_matrix_element_v4(\
                    writers.FortranWriter(filename),\
                    me, self._curr_fortran_model)

        # C++ standalone
        if self._export_format == 'standalone_cpp':
            for me in matrix_elements:
                export_cpp.generate_subprocess_directory_standalone_cpp(\
                              me, self._curr_cpp_model,
                              path = path)
                
        cpu_time2 = time.time() - cpu_time1

        logger.info(("Generated helas calls for %d subprocesses " + \
              "(%d diagrams) in %0.3f s") % \
              (len(matrix_elements),
               ndiags, cpu_time))

        if calls:
            if "cpu_time2" in locals():
                logger.info("Wrote files for %d helas calls in %0.3f s" % \
                            (calls, cpu_time2))
            else:
                logger.info("Wrote files for %d helas calls" % \
                            (calls))
                
        if self._export_format == 'pythia8':
            logger.info("- All necessary files for Pythia 8 generated.")
            logger.info("- Run \"launch\" and select %s.cc," % filename)
            logger.info("  or go to %s/examples and run" % path)
            logger.info("      make -f %s" % make_filename)
            logger.info("  (with process_name replaced by process name).")
            logger.info("  You can then run ./%s to produce events for the process" % \
                        filename)

        # Replace the amplitudes with the actual amplitudes from the
        # matrix elements, which allows proper diagram drawing also of
        # decay chain processes
        self._curr_amps = diagram_generation.AmplitudeList(\
               [me.get('base_amplitude') for me in \
                matrix_elements])

    def finalize(self, nojpeg, online = False):
        """Make the html output, write proc_card_mg5.dat and create
        madevent.tar.gz for a MadEvent directory"""
        
        if self._export_format in ['madevent', 'standalone']:
            # For v4 models, copy the model/HELAS information.
            if self._model_v4_path:
                logger.info('Copy %s model files to directory %s' % \
                            (os.path.basename(self._model_v4_path), self._export_dir))
                self._curr_exporter.export_model_files(self._model_v4_path)
                self._curr_exporter.export_helas(pjoin(self._mgme_dir,'HELAS'))
            else:
                logger.info('Export UFO model to MG4 format')
                # wanted_lorentz are the lorentz structures which are
                # actually used in the wavefunctions and amplitudes in
                # these processes
                wanted_lorentz = self._curr_matrix_elements.get_used_lorentz()
                wanted_couplings = self._curr_matrix_elements.get_used_couplings()
                self._curr_exporter.convert_model_to_mg4(self._curr_model,
                                               wanted_lorentz,
                                               wanted_couplings)
        if self._export_format == 'standalone_cpp':
            logger.info('Export UFO model to C++ format')
            # wanted_lorentz are the lorentz structures which are
            # actually used in the wavefunctions and amplitudes in
            # these processes
            wanted_lorentz = self._curr_matrix_elements.get_used_lorentz()
            wanted_couplings = self._curr_matrix_elements.get_used_couplings()
            export_cpp.convert_model_to_cpp(self._curr_model,
                                            pjoin(self._export_dir),
                                            wanted_lorentz,
                                            wanted_couplings)
            export_cpp.make_model_cpp(self._export_dir)

        elif self._export_format == 'madevent':          
            # Create configuration file [path to executable] for madevent
            filename = os.path.join(self._export_dir, 'Cards', 'me5_configuration.txt')
            self.do_save('options %s' % filename.replace(' ', '\ '), check=False, 
                         to_keep={'mg5_path':MG5DIR})

        if self._export_format in ['madevent', 'standalone']:
            
            self._curr_exporter.finalize_v4_directory( \
                                           self._curr_matrix_elements,
                                           [self.history_header] + \
                                           self.history,
                                           not nojpeg,
                                           online,
                                           self.options['fortran_compiler'])

        if self._export_format in ['madevent', 'standalone', 'standalone_cpp']:
            logger.info('Output to directory ' + self._export_dir + ' done.')

        if self._export_format == 'madevent':              
            logger.info('Type \"launch\" to generate events from this process, or see')
            logger.info(self._export_dir + '/README')
            logger.info('Run \"open index.html\" to see more information about this process.')

    def do_help(self, line):
        """ propose some usefull possible action """
        
        super(MadGraphCmd,self).do_help(line)
        
        if line:
            return
        
        if len(self.history) == 0:
            last_action_2 = 'mg5_start'
            last_action = 'mg5_start'
        else:
            args = self.history[-1].split()
            last_action = args[0]
            if len(args)>1: 
                last_action_2 = '%s %s' % (last_action, args[1])
            else: 
                last_action_2 = 'none'
    
    
    
    # Calculate decay width
    def do_compute_widths(self, line, model=None):
        """Not in help: Generate amplitudes for decay width calculation, with fixed
           number of final particles (called level)
           syntax; compute_widths particle [other particles] [--options=]
           
            - particle/other particles can also be multiparticle name (can also be
           pid of the particle)
          
           --precision=X [default=0.001] allow to choose the precision.
                if X>1 this is means compute all X body decay
            
           --path=X. Use a given file for the param_card. (default UFO built-in)
           
           special argument: 
               - skip_2body: allow to not consider those decay (use FR)
               - model: use the model pass in argument.
           
        """

        warning_text = """Be carefull automatic computation of the width is 
ONLY valid in Narrow-Width Approximation and at Tree-Level."""
        logger.warning(warning_text)
      
        # check the argument and return those in a dictionary format
        particles, opts = self.check_compute_widths(self.split_arg(line))
        
        precision = float(opts['precision'])
        if not model:
            modelname = self._curr_model['name']
            with misc.MuteLogger(['madgraph'], ['INFO']):
                model = import_ufo.import_model(modelname, decay=True)
        if not isinstance(model, model_reader.ModelReader):
            model = model_reader.ModelReader(model)
        data = model.set_parameters_and_couplings(opts['path'])
                

        # find UFO particles linked to the require names.
        skip_2body = True
        decay_info = {}   
        for pid in particles:
            particle = model.get_particle(pid)
            if not hasattr(particle, 'partial_widths'):
                skip_2body = False
                break
            elif not decay_info:
                logger_mg.info('Get two body decay from FeynRules formula')
            decay_info[pid] = []
            mass = abs(eval(str(particle.get('mass')), data).real)
            data = model.set_parameters_and_couplings(opts['path'], scale= mass)
            total = 0
            
            for mode, expr in particle.partial_widths.items():
                tmp_mass = mass    
                for p in mode:
                    tmp_mass -= abs(eval(str(p.mass), data))
                if tmp_mass <=0:
                    continue
                
                decay_to = [p.get('pdg_code') for p in mode]
                value = eval(expr,{'cmath':cmath},data).real
                if -1e-10 < value < 0:
                    value = 0
                if -1e-5 < value < 0:
                    logger.warning('Partial width for %s > %s negative: %s automatically set to zero' %
                                   (particle.get('name'), ' '.join([p.get('name') for p in mode]), value))
                    value = 0
                elif value < 0:
                    raise Exception, 'Partial width for %s > %s negative: %s' % \
                                   (particle.get('name'), ' '.join([p.get('name') for p in mode]), value)
                decay_info[particle.get('pdg_code')].append([decay_to, value])
                total += value
        else:
            madevent_interface.MadEventCmd.update_width_in_param_card(decay_info, 
                                                   opts['path'], opts['output'])
        
        #
        # add info from decay module
        #

        self.do_decay_diagram('%s %s' % (' '.join([`id` for id in particles]), 
                                         precision), skip_2body=skip_2body)
        
        if self._curr_amps:
            logger.info('Pass to numerical integration for computing the widths:')
        else:
            return decay_info

        with misc.TMP_directory() as path:
            decay_dir = pjoin(path,'temp_decay')
            logger_mg.info('More info in temporary files:\n    %s/index.html' % (decay_dir))
            with misc.MuteLogger(['madgraph','ALOHA','cmdprint','madevent'], [40,40,40,40]):
                self.exec_cmd("set automatic_html_opening False --no-save")

                self.exec_cmd('output %s -f' % decay_dir)
                # Need to write the correct param_card in the correct place !!!
                files.cp(opts['path'], pjoin(decay_dir, 'Cards', 'param_card.dat'))
                if self._curr_model['name'] == 'mssm' or self._curr_model['name'].startswith('mssm-'):
                    check_param_card.convert_to_slha1(pjoin(decay_dir, 'Cards', 'param_card.dat'))
                #files.cp(pjoin(self.me_dir, 'Cards','run_card.dat'), pjoin(decay_dir, 'Cards', 'run_card.dat'))
                self.exec_cmd('launch -n decay -f')
            param = check_param_card.ParamCard(pjoin(decay_dir, 'Events', 'decay','param_card.dat'))
        for pid in particles:
            width = param['decay'].get((pid,)).value
            if not pid in param['decay'].decay_table:
                continue
            if pid not in decay_info:
                decay_info[pid] = []
            for BR in param['decay'].decay_table[pid]:
                decay_info[pid].append([BR.lhacode[1:], BR.value * width])
        
        madevent_interface.MadEventCmd.update_width_in_param_card(decay_info, 
                                                   opts['path'], opts['output'])
        
        if self._curr_model['name'] == 'mssm' or self._curr_model['name'].startswith('mssm-'):    
            check_param_card.convert_to_slha1(opts['output'])
        return


           
    # Calculate decay width
    def do_decay_diagram(self, line, skip_2body=False, model=None):
        """Not in help: Generate amplitudes for decay width calculation, with fixed
           number of final particles (called level)
           syntax; decay_diagram part_name level param_path           
           args; part_name level param_path
           part_name = name of the particle you want to calculate width
           level = a.) when level is int,
                       it means the max number of decay products
                   b.) when level is float,
                       it means the required precision for width.
           param_path = path for param_card
           (this is necessary to determine whether a channel is onshell or not)
           e.g. calculate width for higgs up to 2-body decays.
           calculate_width h 2 [path]
           N.B. param_card must be given so that the program knows which channel
           is on shell and which is not.
           
           special argument: 
               - skip_2body: allow to not consider those decay (use FR)
               - model: use the model pass in argument.
        """ 
           
        if model:
            self._curr_model = model

        args = self.split_arg(line)
        #check the validity of the arguments
        args = self.check_decay_diagram(args)
        #print args
        pids = args['ids']
        level = args['level']
        param_card_path = args['path']
            
        # Reset amplitudes
        self._curr_amps = diagram_generation.AmplitudeList()
        # Reset Helas matrix elements
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        # Reset _done_export, since we have new process
        self._done_export = False
        # Also reset _export_format and _export_dir
        self._export_format = None

        # Remove previous generations from history
        self.clean_history(to_remove=['add process'], remove_bef_last='generate',
                           to_keep=['add','import','set','load'],
                           allow_for_removal=['add process','generate','output'])


        # Setup before find_channels
        if not model:
            self._curr_decaymodel = decay_objects.DecayModel(self._curr_model,
                                                         True)        
            self._curr_decaymodel.read_param_card(param_card_path)
        else:
            self._curr_decaymodel = model
        model = self._curr_decaymodel
        
        if  isinstance(pids, int):
            pids = [pids]
            
        first =True
        for part_nb,pid in enumerate(pids):
            part = self._curr_decaymodel.get_particle(pid)
            if part.get('width').lower() == 'zero':
                continue
            logger_mg.info('get decay diagram for %s' % part['name'])
            # Find channels as requested
            if level // 1 == level and level >1:
                level = int(level)
                self._curr_decaymodel.find_channels(part, level)
                if not skip_2body:
                    amp = part.get_amplitudes(2)
                    if amp:
                        self._curr_amps.extend(amp)
                    
                for l in range(3, level+1):
                    amp = part.get_amplitudes(l)
                    if amp:
                        self._curr_amps.extend(amp)
            elif level < 1:
                precision = level
                if first:
                    model.find_all_channels(2)
                    first = False
                if not skip_2body:
                    amp = part.get_amplitudes(2)
                    if amp:
                        self._curr_amps.extend(amp)
                clevel = 2
                while part.get('apx_decaywidth_err') > precision:
                    clevel += 1
                    if clevel > 3:
                        logger_mg.info('    current estimated error: %s go to %s-body decay:' %\
                                        (part.get('apx_decaywidth_err'), clevel))
                    part.find_channels_nextlevel(model)
                    part.group_channels_2_amplitudes(clevel, model, precision)                 
                    amp = part.get_amplitudes(clevel)
                    if amp:
                        self._curr_amps.extend(amp)
                    part.update_decay_attributes(False, True, True, model)
            else:
                raise self.InvalidCmd('wrong type arguments!')
                #logger.info(self._curr_amps.nice_string())

        # Set _generate_info
        if len(self._curr_amps) > 0:
            process = self._curr_amps[0]['process'].nice_string()
            #print process
            self._generate_info = process[9:]
            #print self._generate_info
        else:
            print "No decay is found"

class MadGraphCmdWeb(CheckValidForCmdWeb, MadGraphCmd):
    """Temporary parser"""
                
#===============================================================================
# Command Parser
#=============================================================================== 
# DRAW
_draw_usage = "draw FILEPATH [options]\n" + \
         "-- draw the diagrams in eps format\n" + \
         "   Files will be FILEPATH/diagrams_\"process_string\".eps \n" + \
         "   Example: draw plot_dir . \n"
_draw_parser = optparse.OptionParser(usage=_draw_usage)
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
                          help="avoid contractions of non propagating lines") 
_draw_parser.add_option("", "--add_gap", default=0, type='float', \
                          help="set the x-distance between external particles")  

# LAUNCH PROGRAM
_launch_usage = "launch [DIRPATH] [options]\n" + \
         "-- execute the madevent/standalone/standalone_cpp/pythia8 output present in DIRPATH\n" + \
         "   By default DIRPATH is the latest created directory \n" + \
         "   (for pythia8, it should be the Pythia 8 main directory) \n" + \
         "   Example: launch PROC_sm_1 --name=run2 \n" + \
         "   Example: launch ../pythia8 \n"
_launch_parser = optparse.OptionParser(usage=_launch_usage)
_launch_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory in order to launch the different program")
_launch_parser.add_option("-n", "--name", default='', type='str',
                                help="Provide a name to the run (for madevent run)")
_launch_parser.add_option("-c", "--cluster", default=False, action='store_true',
                                help="submit the job on the cluster")
_launch_parser.add_option("-m", "--multicore", default=False, action='store_true',
                                help="submit the job on multicore core")

_launch_parser.add_option("-i", "--interactive", default=False, action='store_true',
                                help="Use Interactive Console [if available]")
_launch_parser.add_option("-s", "--laststep", default='', 
                                help="last program run in MadEvent run. [auto|parton|pythia|pgs|delphes]")
    
    
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
