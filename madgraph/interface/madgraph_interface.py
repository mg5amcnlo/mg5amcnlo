################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""A user friendly command line interface to access MadGraph5_aMC@NLO features at LO.
   Uses the cmd package for command interpretation and tab completion.
"""
from __future__ import division

import atexit
import collections
import cmath
import glob
import logging
import optparse
import os
import pydoc
import random
import re
import signal
import subprocess
import copy
import sys
import shutil
import StringIO
import traceback
import time
import inspect
import urllib
import random

#useful shortcut
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
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
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
import madgraph.interface.tutorial_text_nlo as tutorial_text_nlo
import madgraph.interface.tutorial_text_madloop as tutorial_text_madloop
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.amcatnlo_run_interface as amcatnlo_run

import madgraph.loop.loop_exporters as loop_exporters
import madgraph.loop.loop_helas_objects as loop_helas_objects

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
logger_check = logging.getLogger('check') # -> stdout
logger_mg = logging.getLogger('madgraph') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr
logger_tuto = logging.getLogger('tutorial') # -> stdout include instruction in
                                            #order to learn MG5
logger_tuto_nlo = logging.getLogger('tutorial_aMCatNLO') # -> stdout include instruction in
                                                        #order to learn aMC@NLO

logger_tuto_madloop = logging.getLogger('tutorial_MadLoop') # -> stoud for MadLoop tuto

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
    error_debug = 'Please report this bug on https://bugs.launchpad.net/mg5amcnlo\n'
    error_debug += 'More information is found in \'%(debug)s\'.\n'
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/mg5amcnlo\n'

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

        if os.path.exists(pjoin(MG5DIR, '.bzr')):
            proc = subprocess.Popen(['bzr', 'nick'], stdout=subprocess.PIPE)
            bzrname,_ = proc.communicate()
            proc = subprocess.Popen(['bzr', 'revno'], stdout=subprocess.PIPE)
            bzrversion,_ = proc.communicate() 
            bzrname, bzrversion = bzrname.strip(), bzrversion.strip() 
            len_name = len(bzrname)
            len_version = len(bzrversion)            
            info_line += "#*         BZR %s %s %s         *\n" % \
                            (bzrname,
                            (34 - len_name - len_version) * ' ',
                            bzrversion)

        # Create a header for the history file.
        # Remember to fill in time at writeout time!
        self.history_header = banner_module.ProcCard.history_header % {'info_line': info_line}
        banner_module.ProcCard.history_header = self.history_header

        if info_line:
            info_line = info_line.replace("#*","*")
            


        logger.info(\
        "************************************************************\n" + \
        "*                                                          *\n" + \
        "*                     W E L C O M E to                     *\n" + \
        "*              M A D G R A P H 5 _ a M C @ N L O           *\n" + \
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
        "*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
        "*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        "*                            and                           *\n" + \
        "*            http://amcatnlo.web.cern.ch/amcatnlo/         *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*           Type 'tutorial' to learn how MG5 works         *\n" + \
        "*    Type 'tutorial aMCatNLO' to learn how aMC@NLO works   *\n" + \
        "*    Type 'tutorial MadLoop' to learn how MadLoop works    *\n" + \
        "*                                                          *\n" + \
        "************************************************************")

        cmd.Cmd.__init__(self, *arg, **opt)

        self.history = banner_module.ProcCard()


    def default(self, line):
        """Default action if line is not recognized"""

        # Faulty command
        log=True
        if line.startswith('p') or line.startswith('e'):
            logger.warning("Command %s not recognized. Did you mean \'generate %s\'?. Please try again" %
                           (line.split()[0], line))
            log=False
        return super(CmdExtended,self).default(line, log=log)

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
        except Exception:
            try:
                logger_tuto.info(getattr(tutorial_text, args[0]).replace('\n','\n\t'))
            except Exception:
                pass

        try:
            logger_tuto_nlo.info(getattr(tutorial_text_nlo, command).replace('\n','\n\t'))
        except Exception:
            try:
                logger_tuto_nlo.info(getattr(tutorial_text_nlo, args[0]).replace('\n','\n\t'))
            except Exception:
                pass

        try:
            logger_tuto_madloop.info(getattr(tutorial_text_madloop, command).replace('\n','\n\t'))
        except Exception:
            try:
                logger_tuto_madloop.info(getattr(tutorial_text_madloop, args[0]).replace('\n','\n\t'))
            except Exception:
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
        logger.info("syntax: save %s FILENAME" % "|".join(self._save_opts),'$MG:color:BLUE')
        logger.info("-- save information as file FILENAME",'$MG:color:BLACK')
        logger.info("   FILENAME is optional for saving 'options'.")
        logger.info('   By default it uses ./input/mg5_configuration.txt')
        logger.info('   If you put "global" for FILENAME it will use ~/.mg5/mg5_configuration.txt')
        logger.info('   If this files exists, it is uses by all MG5 on the system but continues')
        logger.info('   to read the local options files.')

    def help_load(self):
        logger.info("syntax: load %s FILENAME" % "|".join(self._save_opts),'$MG:color:BLUE')
        logger.info("-- load information from file FILENAME",'$MG:color:BLACK')

    def help_import(self):
        logger.info("syntax: import " + "|".join(self._import_formats) + \
              " FILENAME",'$MG:color:BLUE')
        logger.info("-- imports file(s) in various formats",'$MG:color:GREEN')
        logger.info("")
        logger.info("   import model MODEL[-RESTRICTION] [OPTIONS]:",'$MG:color:BLACK')
        logger.info("      Import a UFO model.")
        logger.info("      MODEL should be a valid UFO model name")
        logger.info("      Model restrictions are specified by MODEL-RESTRICTION")
        logger.info("        with the file restrict_RESTRICTION.dat in the model dir.")
        logger.info("        By default, restrict_default.dat is used.")
        logger.info("        Specify model_name-full to get unrestricted model.")
        logger.info("      '--modelname' keeps the original particle names for the model")
        logger.info("")
        logger.info("   import model_v4 MODEL [--modelname] :",'$MG:color:BLACK')
        logger.info("      Import an MG4 model.")
        logger.info("      Model should be the name of the model")
        logger.info("      or the path to theMG4 model directory")
        logger.info("      '--modelname' keeps the original particle names for the model")
        logger.info("")
        logger.info("   import proc_v4 [PATH] :",'$MG:color:BLACK')
        logger.info("      Execute MG5 based on a proc_card.dat in MG4 format.")
        logger.info("      Path to the proc_card is optional if you are in a")
        logger.info("      madevent directory")
        logger.info("")
        logger.info("   import command PATH :",'$MG:color:BLACK')
        logger.info("      Execute the list of command in the file at PATH")
        logger.info("")
        logger.info("   import banner PATH  [--no_launch]:",'$MG:color:BLACK')
        logger.info("      Rerun the exact same run define in the valid banner.")

    def help_install(self):
        logger.info("syntax: install " + "|".join(self._install_opts),'$MG:color:BLUE')
        logger.info("-- Download the last version of the program and install it")
        logger.info("   locally in the current MadGraph5_aMC@NLO version. In order to have")
        logger.info("   a successful installation, you will need to have an up-to-date")
        logger.info("   F77 and/or C and Root compiler.")
        logger.info(" ")
        logger.info("   \"install update\"",'$MG:color:BLACK')
        logger.info("   check if your MG5 installation is the latest one.")
        logger.info("   If not it load the difference between your current version and the latest one,")
        logger.info("   and apply it to the code. Two options are available for this command:")
        logger.info("     -f: didn't ask for confirmation if it founds an update.")
        logger.info("     --timeout=: Change the maximum time allowed to reach the server.")

    def help_display(self):
        logger.info("syntax: display " + "|".join(self._display_opts),'$MG:color:BLUE')
        logger.info("-- display a the status of various internal state variables")
        logger.info("   for particles/interactions you can specify the name or id of the")
        logger.info("   particles/interactions to receive more details information.")
        logger.info("   Example: display particles e+.",'$MG:color:GREEN')
        logger.info(" > For \"checks\", can specify only to see failed checks.")
        logger.info(" > For \"diagrams\", you can specify where the file will be written.")
        logger.info("   Example: display diagrams ./",'$MG:color:GREEN')


    def help_launch(self):
        """help for launch command"""
        # Using the built-in parser help is not convenient when one wants to use
        # color schemes.
        #_launch_parser.print_help()
        logger.info("syntax: launch <dir_path> <options>",'$MG:color:BLUE')
        logger.info("-- execute the aMC@NLO/madevent/standalone/pythia8 output present in dir_path",'$MG:color:BLACK')
        logger.info("By default, dir_path points to the last created directory.")
        logger.info("(for pythia8, it should be the Pythia 8 main directory)")
        logger.info("")
        logger.info("Launch on madevent/pythia8/standalone outputs:",'$MG:color:BLACK')
        logger.info(" o Example: launch PROC_sm_1 --name=run2",'$MG:color:GREEN')
        logger.info(" o Example: launch ../pythia8",'$MG:color:GREEN')
        logger.info(" > Options:")
        logger.info("     -h, --help            show this help message and exit")
        logger.info("     -f, --force           Use the card present in the directory in order")
        logger.info("                           to launch the different program")
        logger.info("     -n NAME, --name=NAME  Provide a name to the run (for madevent run)")
        logger.info("     -c, --cluster         submit the job on the cluster")
        logger.info("     -m, --multicore       submit the job on multicore core")
        logger.info("     -i, --interactive     Use Interactive Console [if available]")
        logger.info("     -s LASTSTEP, --laststep=LASTSTEP")
        logger.info("                           last program run in MadEvent run.")
        logger.info("                           [auto|parton|pythia|pgs|delphes]")
        logger.info("")
        logger.info("Launch on MadLoop standalone output:",'$MG:color:BLACK')
        logger.info(" o Example: launch PROC_loop_sm_1 -f",'$MG:color:GREEN')
        logger.info(" > Simple check of a single Phase-space points.")
        logger.info(" > You will be asked whether you want to edit the MadLoop ")
        logger.info("   and model param card as well as the PS point, unless ")
        logger.info("   the -f option is specified. All other options are ")
        logger.info("   irrelevant for this kind of launch.")
        logger.info("")
        logger.info("Launch on aMC@NLO output:",'$MG:color:BLACK')
        logger.info(" > launch <dir_path> <mode> <options>",'$MG:color:BLUE')
        logger.info(" o Example: launch MyProc aMC@NLO -f -p",'$MG:color:GREEN')

    def help_tutorial(self):
        logger.info("syntax: tutorial [" + "|".join(self._tutorial_opts) + "]",'$MG:color:BLUE')
        logger.info("-- start/stop the MG5 tutorial mode (or stop any other mode)")
        logger.info("-- aMCatNLO: start aMC@NLO tutorial mode")
        logger.info("-- MadLoop: start MadLoop tutorial mode")

    def help_open(self):
        logger.info("syntax: open FILE  ",'$MG:color:BLUE')
        logger.info("-- open a file with the appropriate editor.",'$MG:color:BLACK')
        logger.info('   If FILE belongs to index.html, param_card.dat, run_card.dat')
        logger.info('   the path to the last created/used directory is used')
        logger.info('   The program used to open those files can be chosen in the')
        logger.info('   configuration file ./input/mg5_configuration.txt')

    def help_customize_model(self):
        logger.info("syntax: customize_model --save=NAME",'$MG:color:BLUE')
        logger.info("--  Open an invite where you options to tweak the model.",'$MG:color:BLACK')
        logger.info("    If you specify the option --save=NAME, this tweak will be")
        logger.info("    available for future import with the command 'import model XXXX-NAME'")

    def help_output(self):
        logger.info("syntax: output [" + "|".join(self._export_formats) + \
                    "] [path|.|auto] [options]",'$MG:color:BLUE')
        logger.info("-- Output any generated process(es) to file.",'$MG:color:BLACK')
        logger.info("   Default mode is madevent. Default path is \'.\' or auto.")
        logger.info("   mode:",'$MG:color:BLACK')
        logger.info("   - For MadLoop and aMC@NLO runs, there is only one mode and")
        logger.info("     it is set by default.")
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
        logger.info("   path: The path of the process directory.",'$MG:color:BLACK')
        logger.info("     If you put '.' as path, your pwd will be used.")
        logger.info("     If you put 'auto', an automatic directory PROC_XX_n will be created.")
        logger.info("   options:",'$MG:color:BLACK')
        logger.info("      -f: force cleaning of the directory if it already exists")
        logger.info("      -d: specify other MG/ME directory")
        logger.info("      -noclean: no cleaning performed in \"path\".")
        logger.info("      -nojpeg: no jpeg diagrams will be generated.")
        logger.info("      -name: the postfix of the main file in pythia8 mode.")
        logger.info("   Examples:",'$MG:color:GREEN')
        logger.info("       output",'$MG:color:GREEN')
        logger.info("       output standalone MYRUN -f",'$MG:color:GREEN')
        logger.info("       output pythia8 ../pythia8/ -name qcdprocs",'$MG:color:GREEN')

    def help_check(self):
        logger.info("syntax: check [" + "|".join(self._check_opts) + "] [param_card] process_definition [--energy=] [--split_orders=] [--reduction=]",'$MG:color:BLUE')
        logger.info("-- check a process or set of processes.",'$MG:color:BLACK')
        logger.info("General options:",'$MG:color:BLACK')
        logger.info("o full:",'$MG:color:GREEN')
        logger.info("   Perform all four checks described below:")
        logger.info("   permutation, brs, gauge and lorentz_invariance.")
        logger.info("o permutation:",'$MG:color:GREEN')
        logger.info("   Check that the model and MG5 are working properly")
        logger.info("   by generating permutations of the process and checking")
        logger.info("   that the resulting matrix elements give the same value.")
        logger.info("o gauge:",'$MG:color:GREEN')
        logger.info("   Check that processes are gauge invariant by ")
        logger.info("   comparing Feynman and unitary gauges.")
        logger.info("   This check is, for now, not available for loop processes.")
        logger.info("o brs:",'$MG:color:GREEN')
        logger.info("   Check that the Ward identities are satisfied if the ")
        logger.info("   process has at least one massless gauge boson as an")
        logger.info("   external particle.")
        logger.info("o lorentz_invariance:",'$MG:color:GREEN')
        logger.info("   Check that the amplitude is lorentz invariant by")
        logger.info("   comparing the amplitiude in different frames")
        logger.info("o cms:",'$MG:color:GREEN')
        logger.info("   Check the complex mass scheme consistency by comparing")
        logger.info("   it to the narrow width approximation in the off-shell")
        logger.info("   region of detected resonances and by progressively")
        logger.info("   decreasing the width. Additional options for this check are:")
        logger.info("    --offshellness=f : f is a positive or negative float specifying ")
        logger.info("      the distance from the pole as f*particle_mass. Default is 10.0")
        logger.info("    --seed=i : to force a specific RNG integer seed i (default is fixed to 0)")
        logger.info("    --cms=order1&order2;...,p1->f(p,lambdaCMS)&p2->f2(p,lambdaCMS);...")
        logger.info("      'order_i' specifies the expansion orders considered for the test.")
        logger.info("      The substitution lists specifies how internal parameter must be modified")
        logger.info("      with the width scaling 'lambdaCMS'. The default value for this option is:")
        logger.info("        --cms=QED&QCD,aewm1->10.0/lambdaCMS&as->0.1*lambdaCMS ")
        logger.info("      The number of order and parameters don't have to be the same.")
        logger.info("      The scaling must be specified so that one occurrence of the coupling order.")
        logger.info("      brings in exactly one power of lambdaCMS.")
        logger.info("    --recompute_width= never|first_time|always|auto")
        logger.info("      Decides when to use MadWidth to automatically recompute the width")
        logger.info("      'auto' (default) let MG5 chose the most appropriate behavior.")
        logger.info("      'never' uses the default width value for lambdaCMS=1.0.")
        logger.info("      'first_time' uses MadWidth to compute the width for lambdaCMS=1.0.")
        logger.info("      'first_time' and 'never' assume linear scaling of the widths with lambdaCMS")
        logger.info("      'always' uses MadWidth to compute the widths for all values of lambdaCMS")
        logger.info("               the test relies on linear scaling of the width, so 'always' is ")
        logger.info("               only for double-checks")
        logger.info("    --lambdaCMS = <python_list> : specifies the list of lambdaCMS values to ")
        logger.info("      use for the test. For example: '[(1/2.0)**exp\ for\ exp\ in\ range(0,20)]'")
        logger.info("      In the list expression, you must escape spaces. Also, this option")
        logger.info("      *must* appear last in the otpion list. Finally, the default value is '1.0e-6'")
        logger.info("      for which an optimal list of progressive values is picked up to 1.0e-6")
        logger.info("    --show_plot = True or False: Whether to show plot during analysis (default is True)")
        logger.info("    --report = concise or full: Whether return a concise or full report.")
        logger.info("Comments",'$MG:color:GREEN')
        logger.info(" > If param_card is given, that param_card is used ")
        logger.info("   instead of the default values for the model.")
        logger.info("   If that file is an (LHE) event file. The param_card of the banner")
        logger.info("   is used and the first event compatible with the requested process")
        logger.info("   is used for the computation of the square matrix elements")
        logger.info(" > \"--energy=\" allows to change the default value of sqrt(S).")
        logger.info(" > Except for the 'gauge' test, all checks above are also")
        logger.info("   available for loop processes with ML5 ('virt=' mode)")
        logger.info("Example: check full p p > j j",'$MG:color:GREEN')
        logger.info("Options for loop processes only:",'$MG:color:BLACK')
        logger.info("o timing:",'$MG:color:GREEN')
        logger.info("   Generate and output a process and returns detailed")
        logger.info("   information about the code and a timing benchmark.")
        logger.info("o stability:",'$MG:color:GREEN')
        logger.info("   Generate and output a process and returns detailed")
        logger.info("   statistics about the numerical stability of the code.")
        logger.info("o profile:",'$MG:color:GREEN')
        logger.info("   Performs both the timing and stability analysis at once")
        logger.info("   and outputs the result in a log file without prompting")
        logger.info("   it to the user.")
        logger.info("Comments",'$MG:color:GREEN')
        logger.info(" > These checks are only available for ML5 ('virt=' mode)")
        logger.info(" > For the 'profile' and 'stability' checks, you can chose")
        logger.info("   how many PS points should be used for the statistic by")
        logger.info("   specifying it as an integer just before the [param_card]")
        logger.info("   optional argument.")
        logger.info(" > Notice multiparticle labels cannot be used with these checks.")
        logger.info(" > \"--reduction=\" allows to change what reduction methods should be used.")
        logger.info(" > \"--split_orders=\" allows to change what specific combination of coupling orders to consider.")
        logger.info(" > For process syntax, please see help generate.")
        logger.info(" > In order to save the directory generated or the reuse an existing one")
        logger.info("   previously generated with the check command, one can add the '-reuse' ")
        logger.info("   keyword just after the specification of the type of check desired.")
        logger.info("Example: check profile g g > t t~ [virt=QCD]",'$MG:color:GREEN')


    def help_generate(self):

        logger.info("-- generate diagrams for a given process",'$MG:color:BLUE')
        logger.info("General leading-order syntax:",'$MG:color:BLACK')
        logger.info(" o generate INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2^2=ORDER2 @N")
        logger.info(" o Example: generate l+ vl > w+ > l+ vl a $ z / a h QED<=3 QCD=0 @1",'$MG:color:GREEN')
        logger.info(" > Alternative required s-channels can be separated by \"|\":")
        logger.info("   b b~ > W+ W- | H+ H- > ta+ vt ta- vt~")
        logger.info(" > If no coupling orders are given, MG5 will try to determine")
        logger.info("   orders to ensure maximum number of QCD vertices.")
        logger.info(" > Desired coupling orders combination can be specified directly for")
        logger.info("   the squared matrix element by appending '^2' to the coupling name.")
        logger.info("   For example, 'p p > j j QED^2==2 QCD^==2' selects the QED-QCD")
        logger.info("   interference terms only. The other two operators '<=' and '>' are")
        logger.info("   supported. Finally, a negative value COUP^2==-I refers to the")
        logger.info("   N^(-I+1)LO term in the expansion of the COUP order.")
        logger.info(" > allowed coupling operator are: \"==\", \"=\", \"<=\" and \">\".")
        logger.info("    \"==\" request exactly that number of coupling while \"=\" is interpreted as \"<=\".")
        logger.info(" > To generate a second process use the \"add process\" command")
        logger.info("Decay chain syntax:",'$MG:color:BLACK')
        logger.info(" o core process, decay1, (decay2, (decay2', ...)), ...  etc")
        logger.info(" o Example: generate p p > t~ t QED=0, (t~ > W- b~, W- > l- vl~), t > j j b @2",'$MG:color:GREEN')
        logger.info(" > Note that identical particles will all be decayed.")
        logger.info("Loop processes syntax:",'$MG:color:BLACK')
        logger.info(" o core process [ <NLO_mode=> LoopOrder1 LoopOrder2 ... ] SQUAREDCOUPi=ORDERi")
        logger.info(" o Example: generate p p > t~ t QED=0 QCD=2 [ all= QCD ] QCD=6",'$MG:color:GREEN')
        logger.info(" > Notice that in this format, decay chains are not allowed.")
        logger.info(" > The LoopOrder(s) defined specify the kind of loops to consider (only QCD for now).")
        logger.info(" > The coupling restrictions before '[' restrict the orders of born *amplitudes*.")
        logger.info("   So that in the example above QCD=2 restricts the born amplitude to have at")
        logger.info("   most QCD=2 and loop amplitudes at most QCD=2+2 (because QCD loops are considered)")
        logger.info(" > The coupling restrictions after ']' restrict the orders of the matrix element, ")
        logger.info("   namely the squared amplitudes. In the example above QCD=6 correspond to born")
        logger.info("   amplitudes with QCD=2 squared against loop amplitudes with QCD=4, adding up to 6.")
        logger.info(" > The optional <NLO_mode=> can be any of the following ('all=' by default if absent):")
        logger.info("     all=   : Generate all the real-emission and loop diagrams, ready for aMC@NLO runs.")
        logger.info("     virt=  : Generate only the loop diagrams, read for MadLoop standalone checks/runs.")
        logger.info("     real=  : Generate only the real-emission diagrams, for use with alternative OLP. ")
        logger.info(" > For processes without born amplitudes (i.e. loop-induced like g g > z), please use ")
        logger.info("   the 'virt=' NLO mode. aMC@NLO cannot integrate these processes, but standalone MadLoop5")
        logger.info("   can still handle these.")

    def help_add(self):
        logger.info("-- generate diagrams for a process and add to existing processes",'$MG:color:BLUE')
        logger.info("   OR merge two model",'$MG:color:BLUE')
        logger.info('')
        logger.info("-- generate diagrams for a process and add to existing processes",'$MG:color:BLUE')
        logger.info("General leading-order syntax:",'$MG:color:BLACK')
        logger.info(" o add process INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2 @N")
        logger.info(" o Example: add process l+ vl > w+ > l+ vl a $ z / a h QED=3 QCD=0 @1",'$MG:color:GREEN')
        logger.info(" > Alternative required s-channels can be separated by \"|\":")
        logger.info("   b b~ > W+ W- | H+ H- > ta+ vt ta- vt~")
        logger.info(" > If no coupling orders are given, MG5 will try to determine")
        logger.info("   orders to ensure maximum number of QCD vertices.")
        logger.info(" > Note that if there are more than one non-QCD coupling type,")
        logger.info("   coupling orders need to be specified by hand.")
        logger.info("Decay chain syntax:",'$MG:color:BLACK')
        logger.info(" o core process, decay1, (decay2, (decay2', ...)), ...  etc")
        logger.info(" o Example: add process p p > t~ t QED=0, (t~ > W- b~, W- > l- vl~), t > j j b @2",'$MG:color:GREEN')
        logger.info(" > Note that identical particles will all be decayed.")
        logger.info("Loop processes syntax:",'$MG:color:BLACK')
        logger.info(" o core process [ <NLO_mode=> LoopOrder1 LoopOrder2 ... ] SQUAREDCOUPi=ORDERi")
        logger.info(" o Example: add process p p > t~ t QED=0 QCD=2 [ all= QCD ] QCD=6",'$MG:color:GREEN')
        logger.info(" > Notice that in this format, decay chains are not allowed.")
        logger.info(" > The LoopOrder(s) defined specify the kind of loops to consider (only QCD for now).")
        logger.info(" > The coupling restrictions before '[' restrict the orders of born *amplitudes*.")
        logger.info("   So that in the example above QCD=2 restricts the born amplitude to have at")
        logger.info("   most QCD=2 and loop amplitudes at most QCD=2+2 (because QCD loops are considered)")
        logger.info(" > The coupling restrictions after ']' restrict the orders of the matrix element, ")
        logger.info("   namely the squared amplitudes. In the example above QCD=6 correspond to born")
        logger.info("   amplitudes with QCD=2 squared against loop amplitudes with QCD=4, adding up to 6.")
        logger.info(" > The optional <NLO_mode=> can be any of the following ('all=' by default if absent):")
        logger.info("     all=   : Generate all the real-emission and loop diagrams, ready for aMC@NLO runs.")
        logger.info("     virt=  : Generate only the loop diagrams, read for MadLoop standalone checks/runs.")
        logger.info("     real=  : Generate only the real-emission diagrams, for use with alternative OLP. ")
        logger.info(" > For processes without born amplitudes (i.e. loop-induced like g g > z), please use ")
        logger.info("   the 'virt=' NLO mode. aMC@NLO cannot integrate these processes, but standalone MadLoop5")
        logger.info("   can still handle these.")

        logger.info("--  merge two model to create a new one", '$MG:color:BLUE')
        logger.info("syntax:",'$MG:color:BLACK')
        logger.info(" o add model MODELNAME [OPTIONS]")
        logger.info(" o Example: add model taudecay",'$MG:color:GREEN')
        logger.info(" > Merge the two model in a single one. If that same merge was done before.")
        logger.info(" > Just reload the previous merge. (WARNING: This doesn't check if those model are modified)")
        logger.info(" > Options:")
        logger.info("   --output=  : Specify the name of the directory where the merge is done.")
        logger.info("                This allow to do \"import NAME\" to load that merge.")
        logger.info("   --recreate : Force to recreated the merge model even if the merge model directory already exists.")
        
    def help_compute_widths(self):
        logger.info("syntax: calculate_width PART [other particles] [OPTIONS]")
        logger.info("  Computes the width and partial width for a set of particles")
        logger.info("  Returns a valid param_card with this information.")
        logger.info(" ")
        logger.info("  PART: name of the particle you want to calculate width")
        logger.info("        you can enter either the name or pdg code.\n")
        logger.info("  Various options:\n")
        logger.info("  --body_decay=X: Parameter to control the precision of the computation")
        logger.info("        if X is an integer, we compute all channels up to X-body decay.")
        logger.info("        if X <1, then we stop when the estimated error is lower than X.")
        logger.info("        if X >1 BUT not an integer, then we X = N + M, with M <1 and N an integer")
        logger.info("              We then either stop at the N-body decay or when the estimated error is lower than M.")
        logger.info("        default: 4.0025")
        logger.info("  --min_br=X: All channel which are estimated below this value will not be integrated numerically.")
        logger.info("        default: precision (decimal part of the body_decay options) divided by four")
        logger.info("  --precision_channel=X: requested numerical precision for each channel")
        logger.info("        default: 0.01")
        logger.info("  --path=X: path for param_card")
        logger.info("        default: take value from the model")
        logger.info("  --output=X: path where to write the resulting card. ")
        logger.info("        default: overwrite input file. If no input file, write it in the model directory")
        logger.info("  --nlo: Compute NLO width [if the model support it]")
        logger.info("")
        logger.info(" example: calculate_width h --body_decay=2 --output=./param_card")

    def help_decay_diagram(self):
        logger.info("syntax: decay_diagram PART [other particles] [OPTIONS]")
        logger.info("  Returns the amplitude required for the computation of the widths")
        logger.info(" ")
        logger.info("  PART: name of the particle you want to calculate width")
        logger.info("        you can enter either the name or pdg code.\n")
        logger.info("  Various options:\n")
        logger.info("  --body_decay=X: Parameter to control the precision of the computation")
        logger.info("        if X is an integer, we compute all channels up to X-body decay.")
        logger.info("        if X <1, then we stop when the estimated error is lower than X.")
        logger.info("        if X >1 BUT not an integer, then we X = N + M, with M <1 and N an integer")
        logger.info("              We then either stop at the N-body decay or when the estimated error is lower than M.")
        logger.info("        default: 4.0025")
        logger.info("  --min_br=X: All channel which are estimated below this value will not be integrated numerically.")
        logger.info("        default: precision (decimal part of the body_decay options) divided by four")
        logger.info("  --precision_channel=X: requested numerical precision for each channel")
        logger.info("        default: 0.01")
        logger.info("  --path=X: path for param_card")
        logger.info("        default: take value from the model")
        logger.info("  --output=X: path where to write the resulting card. ")
        logger.info("        default: overwrite input file. If no input file, write it in the model directory")
        logger.info("")
        logger.info(" example: calculate_width h --body_decay=2 --output=./param_card")

    def help_define(self):
        logger.info("-- define a multiparticle",'$MG:color:BLUE')
        logger.info("Syntax:  define multipart_name [=] part_name_list")
        logger.info("Example: define p = g u u~ c c~ d d~ s s~ b b~",'$MG:color:GREEN')
        logger.info("Special syntax: Use | for OR (used for required s-channels)")
        logger.info("Special syntax: Use / to remove particles. Example: define q = p / g")

    def help_set(self):
        logger.info("-- set options for generation or output.",'$MG:color:BLUE')
        logger.info("syntax: set <option_name> <option_value>",'$MG:color:BLACK')
        logger.info("Possible options are: ")
        for opts in [self._set_options[i*3:(i+1)*3] for i in \
                                          range((len(self._set_options)//4)+1)]:
            logger.info("%s"%(','.join(opts)),'$MG:color:GREEN')
        logger.info("Details of each option:")
        logger.info("group_subprocesses True/False/Auto: ",'$MG:color:GREEN')
        logger.info(" > (default Auto) Smart grouping of subprocesses into ")
        logger.info("   directories, mirroring of initial states, and ")
        logger.info("   combination of integration channels.")
        logger.info(" > Example: p p > j j j w+ gives 5 directories and 184 channels",'$MG:color:BLACK')
        logger.info("   (cf. 65 directories and 1048 channels for regular output)",'$MG:color:BLACK')
        logger.info(" > Auto means False for decay computation and True for collisions.")
        logger.info("ignore_six_quark_processes multi_part_label",'$MG:color:GREEN')
        logger.info(" > (default none) ignore processes with at least 6 of any")
        logger.info("   of the quarks given in multi_part_label.")
        logger.info(" > These processes give negligible contribution to the")
        logger.info("   cross section but have subprocesses/channels.")
        logger.info("stdout_level DEBUG|INFO|WARNING|ERROR|CRITICAL",'$MG:color:GREEN')
        logger.info(" > change the default level for printed information")
        logger.info("fortran_compiler NAME",'$MG:color:GREEN')
        logger.info(" > (default None) Force a specific fortran compiler.")
        logger.info("   If None, it tries first g77 and if not present gfortran")
        logger.info("   but loop output use gfortran.")
        logger.info("loop_optimized_output True|False",'$MG:color:GREEN')
        logger.info(" > Exploits the open loop thechnique for considerable")
        logger.info("   improvement.")
        logger.info(" > CP relations among helicites are detected and the helicity")
        logger.info("   filter has more potential.")
        logger.info("loop_color_flows True|False",'$MG:color:GREEN')
        logger.info(" > Only relevant for the loop optimized output.")
        logger.info(" > Reduces the loop diagrams at the amplitude level")
        logger.info("   rendering possible the computation of the loop amplitude")
        logger.info("   for a fixed color flow or color configuration.")
        logger.info(" > This option can considerably slow down the loop ME")
        logger.info("   computation time, especially when summing over all color")
        logger.info("   and helicity configuration, hence turned off by default.")        
        logger.info("gauge unitary|Feynman",'$MG:color:GREEN')
        logger.info(" > (default unitary) choose the gauge of the non QCD part.")
        logger.info(" > For loop processes, only Feynman gauge is employable.")
        logger.info("complex_mass_scheme True|False",'$MG:color:GREEN')
        logger.info(" > (default False) Set complex mass scheme.")
        logger.info(" > Complex mass scheme is not yet supported for loop processes.")
        logger.info("timeout VALUE",'$MG:color:GREEN')
        logger.info(" > (default 20) Seconds allowed to answer questions.")
        logger.info(" > Note that pressing tab always stops the timer.")
        logger.info("cluster_temp_path PATH",'$MG:color:GREEN')
        logger.info(" > (default None) [Used in Madevent Output]")
        logger.info(" > Allow to perform the run in PATH directory")
        logger.info(" > This allow to not run on the central disk. ")
        logger.info(" > This is not used by condor cluster (since condor has")
        logger.info("   its own way to prevent it).")
#        logger.info("mg5amc_py8_interface_path PATH",'$MG:color:GREEN')
#        logger.info(" > Necessary when showering events with Pythia8 from Madevent.")
        logger.info("OLP ProgramName",'$MG:color:GREEN')
        logger.info(" > (default 'MadLoop') [Used for virtual generation]")
        logger.info(" > Chooses what One-Loop Program to use for the virtual")
        logger.info(" > matrix element generation via the BLAH accord.")
        logger.info("output_dependencies <mode>",'$MG:color:GREEN')
        logger.info(" > (default 'external') [Use for NLO outputs]")
        logger.info(" > Choses how the external dependences (such as CutTools)")
        logger.info(" > of NLO outputs are handled. Possible values are:")
        logger.info("     o external: Some of the libraries the output depends")
        logger.info("       on are links to their installation in MG5 root dir.")
        logger.info("     o internal: All libraries the output depends on are")
        logger.info("       copied and compiled locally in the output directory.")
        logger.info("     o environment_paths: The location of all libraries the ")
        logger.info("       output depends on should be found in your env. paths.")        
#        logger.info("max_npoint_for_channel <value>",'$MG:color:GREEN')
#        logger.info(" > (default '0') [Used for loop-induced outputs]")
#        logger.info(" > Sets the maximum 'n' of n-points loops to be used for")
#        logger.info(" > setting up the integration multichannels.") 
#        logger.info(" > The default value of zero automatically picks the apparent")
#        logger.info(" > appropriate choice which is to sometimes pick box loops")
#        logger.info(" > but never higher n-points ones.")

#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(cmd.CheckCmd):
    """ The Series of help routine for the MadGraphCmd"""

    class RWError(MadGraph5Error):
        """a class for read/write errors"""

    def check_add(self, args):
        """check the validity of line
        syntax: add process PROCESS | add model MODELNAME
        """

        if len(args) < 2:
            self.help_add()
            raise self.InvalidCmd('\"add\" requires at least two arguments')
        
        if args[0] not in  ['model', 'process']:
            raise self.InvalidCmd('\"add\" requires the argument \"process\" or \"model\"')    
    
        if args[0] == 'process':
            return self.check_generate(args)
    
        if args[0] == 'model':
            pass
            

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

# check that either _curr_amps or _fks_multi_proc exists
        if (args[0] in ['processes', 'diagrams'] and not self._curr_amps and not self._fks_multi_proc):
           raise self.InvalidCmd("No process generated, please generate a process!")
        if args[0] == 'checks' and not self._comparisons and not self._cms_checks:
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

        if len(args) < 2 and not args[0].lower().endswith('options'):
            self.help_check()
            raise self.InvalidCmd("\"check\" requires a process.")

        if args[0] not in self._check_opts and \
                                        not args[0].lower().endswith('options'):
            args.insert(0, 'full')

        param_card = None
        if args[0] not in ['stability','profile','timing'] and \
                                        len(args)>1 and os.path.isfile(args[1]):
            param_card = args.pop(1)

        if len(args)>1:
            if args[1] != "-reuse":
                args.insert(1, '-no_reuse')
        else:
            args.append('-no_reuse')

        if args[0] in ['timing'] and len(args)>2 and os.path.isfile(args[2]):
            param_card = args.pop(2)
        if args[0] in ['stability', 'profile'] and len(args)>1:
            # If the first argument after 'stability' is not the integer
            # specifying the desired statistics (i.e. number of points), then
            # we insert the default value 100
            try:
                int(args[2])
            except ValueError:
                args.insert(2, '100')

        if args[0] in ['stability', 'profile'] and os.path.isfile(args[3]):
            param_card = args.pop(3)
        if any([',' in elem for elem in args if not elem.startswith('--')]):
            raise self.InvalidCmd('Decay chains not allowed in check')
        
        user_options = {'--energy':'1000','--split_orders':'-1',
                   '--reduction':'1|2|3|4|5|6','--CTModeRun':'-1',
                   '--helicity':'-1','--seed':'-1'}
        
        if args[0] in ['cms'] or args[0].lower()=='cmsoptions':
            # increase the default energy to 5000
            user_options['--energy']='5000'
            # The first argument gives the name of the coupling order in which
            # the cms expansion is carried, and the expression following the 
            # comma gives the relation of an external parameter with the
            # CMS expansions parameter called 'lambdaCMS'.
            parameters = ['aewm1->10.0/lambdaCMS','as->0.1*lambdaCMS']
            user_options['--cms']='QED&QCD,'+'&'.join(parameters)
            # Widths are assumed to scale linearly with lambdaCMS unless
            # --force_recompute_width='always' or 'first_time' is used.
            user_options['--recompute_width']='auto'
            # It can be negative so as to be offshell below the resonant mass
            user_options['--offshellness']='10.0'
            # Pick the lambdaCMS values for the test. Instead of a python list
            # we specify here (low,N) which means that do_check will automatically
            # pick lambda values up to the value low and with N values uniformly
            # spread in each interval [1.0e-i,1.0e-(i+1)].
            # Some points close to each other will be added at the end for the
            # stability test.
            user_options['--lambdaCMS']='(1.0e-6,5)'
            # Set the RNG seed, -1 is default (random).
            user_options['--seed']=666
            # The option below can help the user re-analyze existing pickled check
            user_options['--analyze']='None'
            # Decides whether to show plot or not during the analysis
            user_options['--show_plot']='True'
            # Decides what kind of report 
            user_options['--report']='concise'
            # 'secret' option to chose by which lambda power one should divide
            # the nwa-cms difference. Useful to set to 2 when doing the Born check
            # to see whether the NLO check will have sensitivity to the CMS
            # implementation
            user_options['--diff_lambda_power']='1'
            # Sets the range of lambda values to plot
            user_options['--lambda_plot_range']='[-1.0,-1.0]'
            # Sets a filter to apply at generation. See name of available 
            # filters in loop_diagram_generations.py, function user_filter 
            user_options['--loop_filter']='None'
            # Apply tweaks to the check like multiplying a certain width by a
            # certain parameters or changing the analytical continuation of the 
            # logarithms of the UV counterterms
            user_options['--tweak']='default()'
            # Give a name to the run for the files to be saved
            user_options['--name']='auto'
            # Select what resonances must be run
            user_options['--resonances']='1'
        
        for arg in args[:]:
            if arg.startswith('--') and '=' in arg:
                parsed = arg.split('=')
                key, value = parsed[0],'='.join(parsed[1:])
                if key not in user_options:
                    raise self.InvalidCmd, "unknown option %s" % key
                user_options[key] = value
                args.remove(arg)

        # If we are just re-analyzing saved data or displaying options then we 
        # shouldn't check the process format.
        if not (args[0]=='cms' and '--analyze' in user_options and \
                              user_options['--analyze']!='None') and not \
                                            args[0].lower().endswith('options'):
            
            self.check_process_format(" ".join(args[1:]))

        for option, value in user_options.items():
            args.append('%s=%s'%(option,value))

        return param_card

    def check_generate(self, args):
        """check the validity of args"""

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
        nbsep = len(re.findall('>\D', process)) # not use process.count because of QCD^2>2
        if nbsep not in [1,2]:
            raise self.InvalidCmd(
               'wrong format for \"%s\" this part requires one or two symbols \'>\', %s found'
               % (process, nbsep))

        # we need at least one particles in each pieces
        particles_parts = re.split('>\D', process)
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


    def check_tutorial(self, args):
        """check the validity of the line"""
        if len(args) == 1:
            if not args[0] in self._tutorial_opts:
                self.help_tutorial()
                raise self.InvalidCmd('Invalid argument for tutorial')
        elif len(args) == 0:
            #this means mg5 tutorial
            args.append('MadGraph5')
        else:
            self.help_tutorial()
            raise self.InvalidCmd('Too many arguments for tutorial')



    def check_import(self, args):
        """check the validity of line"""

        modelname = False
        prefix = True
        if '-modelname' in args:
            args.remove('-modelname')
            modelname = True
        elif '--modelname' in args:
            args.remove('--modelname')
            modelname = True
            
        if '--noprefix' in args:
            args.remove('--noprefix')
            prefix = False  

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

        if not prefix:
            args.append('--noprefix')

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
                
                if not self._done_export[1].startswith(mode):
                    print mode, self._done_export[1]
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
        mw_path = pjoin(path,'Source','MadWeight')

        if os.path.isfile(pjoin(include_path, 'Pythia.h')) or \
            os.path.isfile(pjoin(include_path, 'Pythia8', 'Pythia.h')):
            return 'pythia8'
        elif not os.path.isdir(os.path.join(path, 'SubProcesses')):
            raise self.InvalidCmd, '%s : Not a valid directory' % path

        if os.path.isdir(src_path):
            return 'standalone_cpp'
        elif os.path.isdir(mw_path):
            return 'madweight'
        elif os.path.isfile(pjoin(bin_path,'madevent')):
            return 'madevent'
        elif os.path.isfile(pjoin(bin_path,'aMCatNLO')):
            return 'aMC@NLO'
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

        if len(args) == 1 and args[0] in ['complex_mass_scheme',\
                                          'loop_optimized_output',\
                                          'loop_color_flows',\
                                          'low_mem_multicore_nlo_generation']:
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

        if args[0] in ['timeout', 'max_npoint_for_channel']:
            if not args[1].isdigit():
                raise self.InvalidCmd('%s values should be a integer' % args[0])

        if args[0] in ['loop_optimized_output', 'loop_color_flows', 'low_mem_multicore_nlo_generation']:
            try:
                args[1] = banner_module.ConfigFile.format_variable(args[1], bool, args[0])
            except Exception:
                raise self.InvalidCmd('%s needs argument True or False'%args[0])

        if args[0] in ['gauge']:
            if args[1] not in ['unitary','Feynman']:
                raise self.InvalidCmd('gauge needs argument unitary or Feynman.')

        if args[0] in ['timeout']:
            if not args[1].isdigit():
                raise self.InvalidCmd('timeout values should be a integer')

        if args[0] in ['OLP']:
            if args[1] not in MadGraphCmd._OLP_supported:
                raise self.InvalidCmd('OLP value should be one of %s'\
                                               %str(MadGraphCmd._OLP_supported))

        if args[0].lower() in ['ewscheme']:
            if not self._curr_model:
                raise self.InvalidCmd("ewscheme acts on the current model please load one first.")
            if args[1] not in ['external']:
                raise self.InvalidCmd('Only valid ewscheme is "external". To restore default, please re-import the model.')

        if args[0] in ['output_dependencies']:
            if args[1] not in MadGraphCmd._output_dependencies_supported:
                raise self.InvalidCmd('output_dependencies value should be one of %s'\
                               %str(MadGraphCmd._output_dependencies_supported))

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


    def check_output(self, args, default='madevent'):
        """ check the validity of the line"""


        if args and args[0] in self._export_formats:
            self._export_format = args.pop(0)
        else:
            self._export_format = default

        if not self._curr_model:
            text = 'No model found. Please import a model first and then retry.'
            raise self.InvalidCmd(text)

        if self._model_v4_path and \
               (self._export_format not in self._v4_export_formats):
            text = " The Model imported (MG4 format) does not contain enough\n "
            text += " information for this type of output. In order to create\n"
            text += " output for " + args[0] + ", you have to use a UFO model.\n"
            text += " Those model can be imported with MG5> import model NAME."
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
                    raise self.InvalidCmd('%s is not allowed in the output path' % char)
            # Check for special directory treatment
            if path == 'auto' and self._export_format in \
                     ['madevent', 'standalone', 'standalone_cpp', 'matchbox_cpp', 'madweight',
                      'matchbox']:
                self.get_default_path()
                if '-noclean' not in args and os.path.exists(self._export_dir):
                    args.append('-noclean')
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
                if '-noclean' not in args and os.path.exists(self._export_dir):
                    args.append('-noclean')
                    
            else:
                if self.options['pythia8_path']:
                    self._export_dir = self.options['pythia8_path']
                else:
                    self._export_dir = '.'

        self._export_dir = os.path.realpath(self._export_dir)


    def check_compute_widths(self, args):
        """ check and format calculate decay width:
        Expected format: NAME [other names] [--options]
        # fill the options if not present.
        # NAME can be either (anti-)particle name, multiparticle, pid
        """

        if len(args)<1:
            self.help_compute_widths()
            raise self.InvalidCmd('''compute_widths requires at least the name of one particle.
            If you want to compute the width of all particles, type \'compute_widths all\'''')

        particles = set()
        options = {'path':None, 'output':None,
                   'min_br':None, 'body_decay':4.0025, 'precision_channel':0.01,
                   'nlo':False}
        # check that the firsts argument is valid
        
        for i,arg in enumerate(args):
            if arg.startswith('--'):
                if arg.startswith('--nlo'):
                    options['nlo'] =True  
                    continue
                elif not '=' in arg:
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
                if not self._curr_model['case_sensitive']:
                    arg = arg.lower()                
                for p in self._curr_model['particles']:
                    if p['name'] == arg or p['antiname'] == arg:
                        particles.add(abs(p.get_pdg_code()))
                        break
                else:
                    if arg == 'all':
                        #sometimes the multiparticle all is not define
                        particles.update([abs(p.get_pdg_code())
                                        for p in self._curr_model['particles']])
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

        if not options['min_br']:
            options['min_br'] = (float(options['body_decay']) % 1) / 5
        return particles, options


    check_decay_diagram = check_compute_widths

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


        if self._export_format == 'NLO':
            name_dir = lambda i: 'PROCNLO_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format.startswith('madevent'):
            name_dir = lambda i: 'PROC_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format.startswith('standalone'):
            name_dir = lambda i: 'PROC_SA_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))                
        elif self._export_format == 'madweight':
            name_dir = lambda i: 'PROC_MW_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format == 'standalone_cpp':
            name_dir = lambda i: 'PROC_SA_CPP_%s_%s' % \
                                    (self._curr_model['name'], i)
            auto_path = lambda i: pjoin(self.writing_dir,
                                               name_dir(i))
        elif self._export_format in ['matchbox_cpp', 'matchbox']:
            name_dir = lambda i: 'PROC_MATCHBOX_%s_%s' % \
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
            args[:] = ['command', './proc_card_mg5.dat']
        elif args[0] == 'proc_v4':
            args[:] = [args[0], './proc_card.dat']
        elif args[0] == 'command':
            args[:] = [args[0], './proc_card_mg5.dat']

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

    def check_output(self, args, default='madevent'):
        """ check the validity of the line"""

        # first pass to the default
        CheckValidForCmd.check_output(self, args, default=default)
        args[:] = ['.', '-f']

        self._export_dir = os.path.realpath(os.getcwd())
        # Check that we output madevent
        if 'madevent' != self._export_format:
                raise self.WebRestriction, 'only available output format is madevent (at current stage)'

#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(cmd.CompleteCmd):
    """ The Series of help routine for the MadGraphCmd"""

    def nlo_completion(self,args,text,line,allowed_loop_mode=None):
        """ complete the nlo settings within square brackets. It uses the
         allowed_loop_mode for the proposed mode if specified, otherwise, it
         uses self._nlo_modes_for_completion"""

        # We are now editing the loop related options
        # Automatically allow for QCD perturbation if in the sm because the
        # loop_sm would then automatically be loaded
        nlo_modes = allowed_loop_mode if not allowed_loop_mode is None else \
                                                  self._nlo_modes_for_completion
        if isinstance(self._curr_model,loop_base_objects.LoopModel):
            pert_couplings_allowed = ['all']+self._curr_model['perturbation_couplings']
        else:
            pert_couplings_allowed = []
        if self._curr_model.get('name').startswith('sm'):
            pert_couplings_allowed = pert_couplings_allowed + ['QCD']
        # Find wether the loop mode is already set or not
        loop_specs = line[line.index('[')+1:]
        try:
            loop_orders = loop_specs[loop_specs.index('=')+1:]
        except ValueError:
            loop_orders = loop_specs
        possibilities = []
        possible_orders = [order for order in pert_couplings_allowed if \
                                                  order not in loop_orders]

        # Simplify obvious loop completion
        single_completion = ''
        if len(nlo_modes)==1:
                single_completion = '%s= '%nlo_modes[0]
                if len(possible_orders)==1:
                    single_completion = single_completion + possible_orders[0] + ' ] '
        # Automatically add a space if not present after [ or =
        if text.endswith('['):
            if single_completion != '':
                return self.list_completion(text, ['[ '+single_completion])
            else:
                return self.list_completion(text,['[ '])

        if text.endswith('='):
            return self.list_completion(text,[' '])

        if args[-1]=='[':
            possibilities = possibilities + ['%s= '%mode for mode in nlo_modes]
            if single_completion != '':
                return self.list_completion(text, [single_completion])
            else:
                if len(possible_orders)==1:
                    return self.list_completion(text, [poss+' %s ] '%\
                              possible_orders[0] for poss in possibilities])
                return self.list_completion(text, possibilities)

        if len(possible_orders)==1:
            possibilities.append(possible_orders[0]+' ] ')
        else:
            possibilities.extend(possible_orders)
        if any([(order in loop_orders) for order in pert_couplings_allowed]):
            possibilities.append(']')
        return self.list_completion(text, possibilities)

    def model_completion(self, text, process, line, categories = True, \
                                                      allowed_loop_mode = None):
        """ complete the line with model information. If categories is True,
        it will use completion with categories. If allowed_loop_mode is
        specified, it will only complete with these loop modes."""

        # First check if we are within squared brackets so that specific
        # input for NLO settings must be completed
        args = self.split_arg(process)
        if len(args) > 2 and '>' in line and '[' in line and not ']' in line:
            return self.nlo_completion(args,text,line, allowed_loop_mode = \
                                                              allowed_loop_mode)

        while ',' in process:
            process = process[process.index(',')+1:]
        args = self.split_arg(process)
        couplings = []

        # Do no complete the @ for the process number.
        if len(args) > 1 and args[-1]=='@':
            return

        # Automatically allow for QCD perturbation if in the sm because the
        # loop_sm would then automatically be loaded
        if isinstance(self._curr_model,loop_base_objects.LoopModel):
            pert_couplings_allowed = ['all'] + self._curr_model['perturbation_couplings']
        else:
            pert_couplings_allowed = []
        if self._curr_model.get('name').startswith('sm'):
            pert_couplings_allowed = pert_couplings_allowed + ['QCD']

        # Remove possible identical names
        particles = list(set(self._particle_names + self._multiparticles.keys()))
        n_part_entered = len([1 for a in args if a in particles])

        # Force '>' if two initial particles.
        if n_part_entered == 2 and args[-1] != '>':
                return self.list_completion(text, '>')

        # Add non-particle names
        syntax = []
        couplings = []
        if len(args) > 0 and args[-1] != '>' and n_part_entered > 0:
            syntax.append('>')
        if '>' in args and args.index('>') < len(args) - 1:
            couplings.extend(sum([[c+"=",c+'^2'] for c in \
                                              self._couplings+['WEIGHTED']],[]))
            syntax.extend(['@','$','/','>',','])
            if '[' not in line and ',' not in line and len(pert_couplings_allowed)>0:
                syntax.append('[')
            
        # If information for the virtuals has been specified already, do not
        # propose syntax or particles input anymore
        if '[' in line:
            syntax = []
            particles = []
            # But still allow for defining the process id
            couplings.append('@')

        if not categories:
            # The direct completion (might be needed for some completion using
            # this function but adding some other completions (like in check)).
            # For those, it looks ok in the categorie mode on my mac, but if
            # someone sees wierd result on Linux systems, then use the
            # default completion for these features.
            return self.list_completion(text, particles+syntax+couplings)
        else:
            # A more elaborate one with categories
            poss_particles = self.list_completion(text, particles)
            poss_syntax = self.list_completion(text, syntax)
            poss_couplings = self.list_completion(text, couplings)
            possibilities = {}
            if poss_particles != []: possibilities['Particles']=poss_particles
            if poss_syntax != []: possibilities['Syntax']=poss_syntax
            if poss_couplings != []: possibilities['Coupling orders']=poss_couplings
            if len(possibilities.keys())==1:
                return self.list_completion(text, possibilities.values()[0])
            else:
                return self.deal_multiple_categories(possibilities)

    def complete_generate(self, text, line, begidx, endidx):
        "Complete the generate command"

        # Return list of particle names and multiparticle names, as well as
        # coupling orders and allowed symbols
        args = self.split_arg(line[0:begidx])

        valid_sqso_operators=['==','<=','>']
        if any(line.endswith('^2 %s '%op) for op in valid_sqso_operators):
            return
        if args[-1].endswith('^2'):
            return self.list_completion(text,valid_sqso_operators)
        match_op = [o for o in valid_sqso_operators if o.startswith(args[-1])]            
        if args[-2].endswith('^2') and len(match_op)>0:
            if args[-1] in valid_sqso_operators:
                return self.list_completion(text,' ')
            if len(match_op)==1:
                return self.list_completion(text,[match_op[0][len(args[-1]):]])
            else:
                return self.list_completion(text,match_op)

        if len(args) > 2 and args[-1] == '@' or ( args[-1].endswith('=') and \
                            (not '[' in line or ('[' in line and ']' in line))):
            return

        try:
            return self.model_completion(text, ' '.join(args[1:]),line)
        except Exception as error:
            print error

        #if len(args) > 1 and args[-1] != '>':
        #    couplings = ['>']
        #if '>' in args and args.index('>') < len(args) - 1:
        #    couplings = [c + "=" for c in self._couplings] + ['@','$','/','>']
        #return self.list_completion(text, self._particle_names + \
        #                            self._multiparticles.keys() + couplings)


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
        else:
            completion = {}
            completion['options'] = self.list_completion(text,
                            ['--path=', '--output=', '--min_br=0.\$',
                             '--precision_channel=0.\$', '--body_decay=', '--nlo'])
            completion['particles'] = self.model_completion(text, '', line)

        return self.deal_multiple_categories(completion)

    complete_decay_diagram = complete_compute_widths

    def complete_add(self, text, line, begidx, endidx):
        "Complete the add command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._add_opts)

        if args[1] == 'process':
            return self.complete_generate(text, " ".join(args[1:]), begidx, endidx)
        
        elif args[1] == 'model':
            completion_categories = self.complete_import(text, line, begidx, endidx, 
                                                         allow_restrict=False, treat_completion=False)
            completion_categories['options'] = self.list_completion(text,['--modelname=','--recreate'])
            return self.deal_multiple_categories(completion_categories) 
            
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


        cms_check_mode = len(args) >= 2 and args[1]=='cms'

        cms_options = ['--name=','--tweak=','--seed=','--offshellness=',
          '--lambdaCMS=','--show_plot=','--report=','--lambda_plot_range=','--recompute_width=',
          '--CTModeRun=','--helicity=','--reduction=','--cms=','--diff_lambda_power=',
          '--loop_filter=','--resonances=']

        options = ['--energy=']
        if cms_options:
            options.extend(cms_options)

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text, pjoin(*[a for a in args \
                                                    if a.endswith(os.path.sep)]))
        # autocompletion for particles/couplings
        model_comp = self.model_completion(text, ' '.join(args[2:]),line,
                                  categories = True, allowed_loop_mode=['virt'])

        model_comp_and_path = self.deal_multiple_categories(\
          {'Process completion': self.model_completion(text, ' '.join(args[2:]),
          line, categories = False, allowed_loop_mode=['virt']),
          'Param_card.dat path completion:':self.path_completion(text),
          'options': self.list_completion(text,options)})

        #Special rules for check cms completion
        if cms_check_mode:
            # A couple of useful value completions
            if line[-1]!=' ' and line[-2]!='\\' and not '--' in line[begidx:endidx] \
                              and args[-1].startswith('--') and '=' in args[-1]:
                examples = {
                  '--tweak=':
['default','alltweaks',"['default','allwidths->1.1*all_withds&seed333(Increased_widths_and_seed_333)','logp->logm&logm->logp(inverted_logs)']"],
                  '--lambdaCMS=':
['(1.0e-2,5)',"[float('1.0e-%d'%exp)\\ for\\ exp\\ in\\ range(8)]","[1.0,0.5,0.001]"],
                  '--lambda_plot_range=':
[' [1e-05,1e-02]','[0.01,1.0]'],
                  '--reduction=':
['1','1|2|3|4','1|2','3'],
                  '--cms=':
['QED&QCD,aewm1->10.0/lambdaCMS&as->0.1*lambdaCMS',
'NP&QED&QCD,aewm1->10.0/lambdaCMS&as->0.1*lambdaCMS&newExpansionParameter->newExpansionParameter*lambdaCMS'],
                  '--loop_filter=':
['None','n>3','n<4 and 6 in loop_pdgs and 3<=id<=7'],
                  '--resonances=':
['1','all','(24,(3,4))','[(24,(3,4)),(24,(4,5))]'],
                  '--analyze=':
['my_default_run.pkl',
'default_run.pkl,increased_widths.pkl(Increased_widths),logs_modified.pkl(Inverted_logs),seed_668.pkl(Different_seed)']
                    }
                for name, example in examples.items():
                    if  args[-1].startswith(name):
                        return self.deal_multiple_categories(
          {"Examples of completion for option '%s'"%args[-1].split('=')[0]:
#                    ['%d: %s'%(i+1,ex) for i, ex in enumerate(example)]},
                    ['%s'%ex for i, ex in enumerate(example)]},
                                                             forceCategory=True)
                if args[-1]=='--recompute_width=':
                    return self.list_completion(text,
                                         ['never','first_time','always','auto'])
                elif args[-1]=='--show_plot=':
                    return self.list_completion(text,['True','False'])
                elif args[-1]=='--report=':
                    return self.list_completion(text,['concise','full'])
                elif args[-1]=='--CTModeRun=':
                    return self.list_completion(text,['-1','1','2','3','4'])
                else:
                    return text
            if len(args)==2 or len(args)==3 and args[-1]=='-reuse':
                return self.deal_multiple_categories(
          {'Process completion': self.model_completion(text, ' '.join(args[2:]),
                        line, categories = False, allowed_loop_mode=['virt']),
                   'Param_card.dat path completion:': self.path_completion(text),
               'reanalyze result on disk / save output:':self.list_completion(
                                                  text,['-reuse','--analyze='])})
            elif not any(arg.startswith('--') for arg in args):
                if '>' in args:
                    return self.deal_multiple_categories({'Process completion': 
                        self.model_completion(text, ' '.join(args[2:]),
                        line, categories = False, allowed_loop_mode=['virt']),
                        'options': self.list_completion(text,options)})
                else:
                    return self.deal_multiple_categories({'Process completion': 
                        self.model_completion(text, ' '.join(args[2:]),
                        line, categories = False, allowed_loop_mode=['virt'])})
            else:
                return self.list_completion(text,options)
            
        if len(args) == 2:
            return model_comp_and_path
        elif len(args) == 3:
            try:
                int(args[2])
            except ValueError:
                return model_comp
            else:
                return model_comp_and_path
        elif len(args) > 3:
            return model_comp


    def complete_tutorial(self, text, line, begidx, endidx):
        "Complete the tutorial command"

        # Format
        if len(self.split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._tutorial_opts)

    def complete_define(self, text, line, begidx, endidx):
        """Complete particle information"""
        return self.model_completion(text, line[6:],line)

    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command"

        args = self.split_arg(line[0:begidx])
        # Format
        if len(args) == 1:
            return self.list_completion(text, self._display_opts)

        if len(args) == 2 and args[1] == 'checks':
            return self.list_completion(text, ['failed'])

        if len(args) == 2 and args[1] == 'particles':
            return self.model_completion(text, line[begidx:],line)

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
               '-p','--parton','--interactive', '--laststep=parton', '--laststep=pythia',
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
                            'ExRootAnalysis', 'HELAS', 'Transfer_Fct', 'aloha',
                            'matchbox', 'matchbox_cpp', 'tests']

        #name of the run =>proposes old run name
        args = self.split_arg(line[0:begidx])
        if len(args) >= 1:
            
            if len(args) > 1 and args[1] == 'pythia8':
                possible_options_full = list(possible_options_full) + ['--version=8.1','--version=8.2'] 
            
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
            content += possible_options_full
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
            opts = list(set(self.options.keys() + self._set_options))
            return self.list_completion(text, opts)

        if len(args) == 2:
            if args[1] in ['group_subprocesses', 'complex_mass_scheme',\
                           'loop_optimized_output', 'loop_color_flows',\
                           'low_mem_multicore_nlo_generation']:
                return self.list_completion(text, ['False', 'True', 'default'])
            elif args[1] in ['ignore_six_quark_processes']:
                return self.list_completion(text, self._multiparticles.keys())
            elif args[1].lower() == 'ewscheme':
                return self.list_completion(text, ["external"])
            elif args[1] == 'gauge':
                return self.list_completion(text, ['unitary', 'Feynman','default'])
            elif args[1] == 'OLP':
                return self.list_completion(text, MadGraphCmd._OLP_supported)
            elif args[1] == 'output_dependencies':
                return self.list_completion(text, 
                                     MadGraphCmd._output_dependencies_supported)
            elif args[1] == 'stdout_level':
                return self.list_completion(text, ['DEBUG','INFO','WARNING','ERROR',
                                                          'CRITICAL','default'])
            elif args[1] == 'fortran_compiler':
                return self.list_completion(text, ['f77','g77','gfortran','default'])
            elif args[1] == 'cpp_compiler':
                return self.list_completion(text, ['g++', 'c++', 'clang', 'default'])
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
        
    def complete_import(self, text, line, begidx, endidx, allow_restrict=True,
                        treat_completion=True):
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
                except Exception:
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
                except Exception:
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
                elif allow_restrict:
                    # need to update the  list with the possible restriction
                    all_name = []
                    for model_name in model_list:
                        all_name += self.find_restrict_card(model_name,
                                            base_dir=pjoin(MG5DIR,'models'))
                else:
                    all_name = model_list
                    
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
                completion_categories['options'] = self.list_completion(text, ['--modelname','-modelname','--noprefix'])
        if len(args) >= 3 and mode.startswith('banner') and not '--no_launch' in line:
            completion_categories['options'] = self.list_completion(text, ['--no_launch'])
        
        if treat_completion:
            return self.deal_multiple_categories(completion_categories) 
        else:
            #this means this function is called as a subgroup of another completion
            return completion_categories


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
    _add_opts = ['process', 'model']
    _save_opts = ['model', 'processes', 'options']
    _tutorial_opts = ['aMCatNLO', 'stop', 'MadLoop', 'MadGraph5']
    _switch_opts = ['mg5','aMC@NLO','ML5']
    _check_opts = ['full', 'timing', 'stability', 'profile', 'permutation',
                   'gauge','lorentz', 'brs', 'cms']
    _import_formats = ['model_v4', 'model', 'proc_v4', 'command', 'banner']
    _install_opts = ['pythia-pgs', 'Delphes', 'MadAnalysis', 'ExRootAnalysis',
                     'update', 'Delphes2', 'SysCalc', 'Golem95', 'PJFry',
                                                                      'QCDLoop']
    # The targets below are installed using the HEPToolsInstaller.py script
    _advanced_install_opts = ['ninja']
    
    # The options below are commented for now but already available
#    _advanced_install_opts += ['pythia8','zlib','boost','lhapdf6','lhapdf5','hepmc','mg5amc_py8_interface','oneloop']
    
    _install_opts.extend(_advanced_install_opts)

    _v4_export_formats = ['madevent', 'standalone', 'standalone_msP','standalone_msF',
                          'matrix', 'standalone_rw', 'madweight'] 
    _export_formats = _v4_export_formats + ['standalone_cpp', 'pythia8', 'aloha',
                                            'matchbox_cpp', 'matchbox']
    _set_options = ['group_subprocesses',
                    'ignore_six_quark_processes',
                    'stdout_level',
                    'fortran_compiler',
                    'cpp_compiler',
                    'loop_optimized_output',
                    'complex_mass_scheme',
                    'gauge',
                    'EWscheme',
                    'max_npoint_for_channel']
    _valid_nlo_modes = ['all','real','virt','sqrvirt','tree','noborn','LOonly']
    _valid_sqso_types = ['==','<=','=','>']
    _valid_amp_so_types = ['=','<=', '==', '>']
    _OLP_supported = ['MadLoop', 'GoSam']
    _output_dependencies_supported = ['external', 'internal','environment_paths']

    # The three options categories are treated on a different footage when a
    # set/save configuration occur. current value are kept in self.options
    options_configuration = {'pythia8_path': './pythia8',
                       'hwpp_path': './herwigPP',
                       'thepeg_path': './thepeg',
                       'hepmc_path': './hepmc',
                       'madanalysis_path': './MadAnalysis',
                       'pythia-pgs_path':'./pythia-pgs',
                       'td_path':'./td',
                       'delphes_path':'./Delphes',
                       'exrootanalysis_path':'./ExRootAnalysis',
                       'syscalc_path': './SysCalc',
                       'timeout': 60,
                       'web_browser':None,
                       'eps_viewer':None,
                       'text_editor':None,
                       'fortran_compiler':None,
                       'f2py_compiler':None,
                       'cpp_compiler':None,
                       'auto_update':7,
                       'cluster_type': 'condor',
                       'cluster_queue': None,
                       'cluster_status_update': (600, 30),
                       'fastjet':'fastjet-config',
                       'pjfry':'auto',
                       'golem':'auto',
                       'samurai':None,
                       'ninja':'./HEPTools/lib',
                       'lhapdf':'lhapdf-config',
                       'applgrid':'applgrid-config',
                       'amcfast':'amcfast-config',
                       'cluster_temp_path':None,
                       'cluster_local_path': None,
#                       'mg5amc_py8_interface_path': './HEPTools/MG5aMC_PY8_interface',
                       'OLP': 'MadLoop',
                       'cluster_nb_retry':1,
                       'cluster_retry_wait':300,
                       'cluster_size':100,
                       'output_dependencies':'external'
                       }

    options_madgraph= {'group_subprocesses': 'Auto',
                          'ignore_six_quark_processes': False,
                          'low_mem_multicore_nlo_generation': False,
                          'complex_mass_scheme': False,
                          'gauge':'unitary',
                          'stdout_level':None,
                          'loop_optimized_output':True,
                          'loop_color_flows':False,
                          'max_npoint_for_channel': 0 # 0 means automaticly adapted
                        }

    options_madevent = {'automatic_html_opening':True,
                         'run_mode':2,
                         'nb_core': None,
                         'notification_center': True
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

    helporder = ['Main commands', 'Documented commands']


    def preloop(self):
        """Initializing before starting the main loop"""

        self.prompt = 'MG5_aMC>'
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
        self._export_dir = None
        self._export_format = 'madevent'
        self._mgme_dir = MG4DIR
        self._cuttools_dir=str(os.path.join(self._mgme_dir,'vendor','CutTools'))
        self._iregi_dir=str(os.path.join(self._mgme_dir,'vendor','IREGI','src'))
        self._comparisons = None
        self._cms_checks = []
        self._nlo_modes_for_completion = ['all','virt','real','LOonly']

        # Load the configuration file,i.e.mg5_configuration.txt
        self.set_configuration()

    def setup(self):
        """ Actions to carry when switching to this interface """

        # Refresh all the interface stored value as things like generated
        # processes and amplitudes are not to be reused in between different
        # interfaces
        # Clear history, amplitudes and matrix elements when a model is imported
        # Remove previous imports, generations and outputs from history
        self.history.clean(remove_bef_last='import',keep_switch=True)
        # Reset amplitudes and matrix elements
        self._done_export=False
        self._curr_amps = diagram_generation.AmplitudeList()
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()

        self._v4_export_formats = ['madevent', 'standalone','standalone_msP','standalone_msF',
                                   'matrix', 'standalone_rw']
        self._export_formats = self._v4_export_formats + ['standalone_cpp', 'pythia8']
        self._nlo_modes_for_completion = ['all','virt','real']

    def do_quit(self, line):
        """Not in help: Do quit"""

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
        or merge two model
        """

        args = self.split_arg(line)

        
        warning_duplicate = True
        if '--no_warning=duplicate' in args:
            warning_duplicate = False
            args.remove('--no_warning=duplicate')

        # Check the validity of the arguments
        self.check_add(args)

        if args[0] == 'model':
            return self.add_model(args[1:])
        
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
                if ']' in line or '[' in line:
                    error_msg=\
"""The '[' and ']' syntax cannot be used in cunjunction with decay chains.
This implies that with decay chains:
  > Squared coupling order limitations are not available.
  > Loop corrections cannot be considered."""
                    raise MadGraph5Error(error_msg)
                else:
                    nb_proc = len([l for l in self.history if l.startswith(('generate','add process'))])
                    myprocdef, line = self.extract_decay_chain_process(line, proc_number=nb_proc)
                    # Redundant with above, but not completely as in the future
                    # one might think of allowing the core process to be 
                    # corrected by loops.
                    if myprocdef.are_decays_perturbed():
                        raise MadGraph5Error("Decay processes cannot be perturbed.")
                    # The two limitations below have some redundancy, but once
                    # again, they might be relieved (one at a time or together)
                    # int he future.
                    if myprocdef.decays_have_squared_orders() or \
                                                myprocdef['squared_orders']!={}:
                        raise MadGraph5Error("Decay processes cannot specify "+\
                                                  "squared orders constraints.")                        
                    if myprocdef.are_negative_orders_present():
                        raise MadGraph5Error("Decay processes cannot include negative"+\
                                                " coupling orders constraints.")                    
            else:
                nb_proc = len([l for l in self.history if l.startswith(('generate','add process'))])
                myprocdef = self.extract_process(line, proc_number=nb_proc)

            

            # Check that we have something
            if not myprocdef:
                raise self.InvalidCmd("Empty or wrong format process, please try again.")
            # Check that we have the same number of initial states as
            # existing processes
            if self._curr_amps and self._curr_amps[0].get_ninitial() != \
               myprocdef.get_ninitial():
                raise self.InvalidCmd("Can not mix processes with different number of initial states.")               
            
            # Negative coupling order contraints can be given on at most one
            # coupling order (and either in squared orders or orders, not both)
            if len([1 for val in myprocdef.get('orders').values()+\
                          myprocdef.get('squared_orders').values() if val<0])>1:
                raise MadGraph5Error("Negative coupling order constraints"+\
                  " can only be given on one type of coupling and either on"+\
                               " squared orders or amplitude orders, not both.")

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
                                     collect_mirror_procs = collect_mirror_procs,
                                     ignore_six_quark_processes = ignore_six_quark_processes,
                                     optimize=optimize)


            for amp in myproc.get('amplitudes'):
                if amp not in self._curr_amps:
                    self._curr_amps.append(amp)
                elif warning_duplicate:
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
                
    def add_model(self, args):
        """merge two model"""
        
        model_path = args[0]
        recreate = ('--recreate' in args)
        output_dir = [a.split('=',1)[1] for a in args if a.startswith('--output')]
        if output_dir:
            output_dir = output_dir[0]
            recreate = True
            restrict_name = ''
        else:
            name = os.path.basename(self._curr_model.get('modelpath'))
            restrict_name = self._curr_model.get('restrict_name')
            output_dir = pjoin(MG5DIR, 'models', '%s__%s' % (name,
                                                  os.path.basename(model_path)))
        
        if os.path.exists(output_dir):
            if recreate:
                shutil.rmtree(output_dir)
            else:
                logger.info('Model already created! Loading it from %s' % output_dir)
                oldmodel = self._curr_model.get('modelpath')
                new_model_name = output_dir
                if restrict_name:
                    new_model_name = '%s-%s' % (output_dir, restrict_name)
                try:
                    self.exec_cmd('import model %s' % new_model_name, errorhandling=False, 
                              printcmd=False, precmd=True, postcmd=True)
                except Exception, error:
                    logger.debug('fail to load model %s with error:\n %s' % (output_dir, error))
                    logger.warning('Fail to load the model. Restore previous model')
                    self.exec_cmd('import model %s' % oldmodel, errorhandling=False, 
                              printcmd=False, precmd=True, postcmd=True)                    
                    raise Exception('Invalid Model! Please retry with the option \'--recreate\'.')
                else:
                    return
        
        #Need to do the work!!!        
        import models.usermod as usermod
        base_model = usermod.UFOModel(self._curr_model.get('modelpath'))
        
        identify = dict(tuple(a.split('=')) for a in args if '=' in a)
        base_model.add_model(path=model_path, identify_particles=identify)
        base_model.write(output_dir)
        
        new_model_name = output_dir
        if restrict_name:
            new_model_name = '%s-%s' % (output_dir, restrict_name)
            
        if 'modelname' in self.history.get('full_model_line'):
            opts = '--modelname'
        else:
            opts='' 
        self.exec_cmd('import model %s %s' % (new_model_name, opts), errorhandling=False, 
                              printcmd=False, precmd=True, postcmd=True)         
        
        
    # Define a multiparticle label
    def do_define(self, line, log=True):
        """Define a multiparticle"""

        self.avoid_history_duplicate('define %s' % line, ['define'])
        if not self._curr_model:
            self.do_import('model sm')
            self.history.append('define %s' % line)
        if not self._curr_model['case_sensitive']:
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
            except Exception:
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
            except Exception:
                raise self.InvalidCmd, 'no lorentz %s in current model' % args[1]

        elif args[0] == 'checks':
            outstr = ''
            if self._comparisons:
                comparisons = self._comparisons[0]
                if len(args) > 1 and args[1] == 'failed':
                    comparisons = [c for c in comparisons if not c['passed']]
                outstr += "Process check results:"
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
                if used_aloha:
                    outstr += "\nChecked ALOHA routines:"
                for aloha in used_aloha:
                    aloha_str = aloha[0]
                    if aloha[1]:
                        aloha_str += 'C' + 'C'.join([str(ia) for ia in aloha[1]])
                    aloha_str += "_%d" % aloha[2]
                    outstr += "\n" + aloha_str
            
            outstr += '\n'
            for cms_check in self._cms_checks:
                outstr += '*'*102+'\n'
                outstr += 'Complex Mass Scheme check:\n'
                outstr += '    -> check %s\n'%cms_check['line']
                outstr += '*'*102+'\n'
                tmp_options = copy.copy(cms_check['options'])
                tmp_options['show_plot']=False
                outstr += process_checks.output_complex_mass_scheme(
                            cms_check['cms_result'], cms_check['output_path'], 
                                           tmp_options, self._curr_model) + '\n'
                outstr += '*'*102+'\n\n'
            pydoc.pager(outstr)

        elif args[0] == 'options':
            if len(args) == 1:
                to_print = lambda name: True
            else:
                to_print = lambda name: any(poss in name for poss in args[1:])

            outstr = "                          MadGraph5_aMC@NLO Options    \n"
            outstr += "                          ----------------    \n"
            keys = self.options_madgraph.keys()
            keys.sort()
            for key in keys:
                if not to_print(key):
                    continue
                default = self.options_madgraph[key] 
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)
            outstr += "\n"
            outstr += "                         MadEvent Options    \n"
            outstr += "                          ----------------    \n"
            keys = self.options_madevent.keys()
            keys.sort()
            for key in keys:
                if not to_print(key):
                    continue
                default = self.options_madevent[key]
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)
            outstr += "\n"
            outstr += "                      Configuration Options    \n"
            outstr += "                      ---------------------    \n"
            keys = self.options_configuration.keys()
            keys.sort()
            for key in keys:
                if not to_print(key):
                    continue
                default = self.options_configuration[key]
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
        self.check_tutorial(args)
        tutorials = {'MadGraph5': logger_tuto,
                     'aMCatNLO': logger_tuto_nlo,
                     'MadLoop': logger_tuto_madloop}
        try:
            tutorials[args[0]].setLevel(logging.INFO)
            for mode in [m for m in tutorials.keys() if m != args[0]]:
                tutorials[mode].setLevel(logging.ERROR)
        except KeyError:
            logger_tuto.info("\n\tThanks for using the tutorial!")
            logger_tuto.setLevel(logging.ERROR)
            logger_tuto_nlo.info("\n\tThanks for using the aMC@NLO tutorial!")
            logger_tuto_nlo.setLevel(logging.ERROR)
            logger_tuto_madloop.info("\n\tThanks for using MadLoop tutorial!")
            logger_tuto_madloop.setLevel(logging.ERROR)

        if not self._mgme_dir:
            logger_tuto.info(\
                       "\n\tWarning: To use all features in this tutorial, " + \
                       "please run from a" + \
                       "\n\t         valid MG_ME directory.")



    def draw(self, line,selection='all',type=''):
        """ draw the Feynman diagram for the given process.
        Type refers to born, real or loop"""

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

            if selection=='all' and type != 'loop':
                diags=amp.get('diagrams')
            elif selection=='born':
                diags=amp.get('born_diagrams')
            elif selection=='loop' or type == 'loop':
                diags=base_objects.DiagramList([d for d in
                        amp.get('loop_diagrams') if d.get('type')>0])
                if len(diags) > 5000:
                    logger.warning('Displaying only the first 5000 diagrams')
                    diags = base_objects.DiagramList(diags[:5000])

            plot = draw.MultiEpsDiagramDrawer(diags,
                                          filename,
                                          model=self._curr_model,
                                          amplitude=amp,
                                          legend=amp.get('process').input_string(),
                                          diagram_type=type)


            logger.info("Drawing " + \
                         amp.get('process').nice_string())
            plot.draw(opt=options)
            logger.info("Wrote file " + filename)
            self.exec_cmd('open %s' % filename)

        stop = time.time()
        logger.info('time to draw %s' % (stop - start))

    # Perform checks
    def do_check(self, line):
        """Check a given process or set of processes"""

        def create_lambda_values_list(lower_bound, N):
            """ Returns a list of values spanning the range [1.0, lower_bound] with
             lower_bound < 1.0 and with each interval [1e-i, 1e-(i+1)] covered
             by N values uniformly distributed. For example, lower_bound=1e-2
             and N=5 returns:
             [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]"""
            
            lCMS_values = [1]
            exp = 0
            n   = 0
            while lCMS_values[-1]>=lower_bound:
                n = (n+1)
                lCMS_values.append(float('1.0e-%d'%exp)*((N-n%N)/float(N)))
                if lCMS_values[-1]==lCMS_values[-2]:
                    lCMS_values.pop()
                exp = (n+1)//N
            
            lCMS_values = lCMS_values[:-1]
            if lCMS_values[-1]!=lower_bound:
                lCMS_values.append(lower_bound)
                
            return lCMS_values
        
        ###### BEGIN do_check

        args = self.split_arg(line)
        # Check args validity
        param_card = self.check_check(args)

        options= {'events':None} # If the momentum needs to be picked from a event file
        if param_card and 'banner' == madevent_interface.MadEventCmd.detect_card_type(param_card):
            logger_check.info("Will use the param_card contained in the banner and  the events associated")
            import madgraph.various.banner as banner
            options['events'] = param_card
            mybanner = banner.Banner(param_card)
            param_card = mybanner.charge_card('param_card')

        aloha_lib.KERNEL.clean()
        # Back up the gauge for later
        gauge = str(self.options['gauge'])
        options['reuse'] = args[1]=="-reuse"
        args = args[:1]+args[2:] 
        # For the stability check the user can specify the statistics (i.e
        # number of trial PS points) as a second argument
        if args[0] in ['stability', 'profile']:
            options['npoints'] = int(args[1])
            args = args[:1]+args[2:]
        
        MLoptions={}
        i=-1
        CMS_options = {}
        while args[i].startswith('--'):
            option = args[i].split('=')
            if option[0] =='--energy':
                options['energy']=float(option[1])
            elif option[0]=='--split_orders':
                options['split_orders']=int(option[1])
            elif option[0]=='--helicity':
                try:
                    options['helicity']=int(option[1])
                except ValueError:
                    raise self.InvalidCmd("The value of the 'helicity' option"+\
                                       " must be an integer, not %s."%option[1])
            elif option[0]=='--reduction':
                MLoptions['MLReductionLib']=[int(ir) for ir in option[1].split('|')]
            elif option[0]=='--CTModeRun':
                try:
                    MLoptions['CTModeRun']=int(option[1])  
                except ValueError:
                    raise self.InvalidCmd("The value of the 'CTModeRun' option"+\
                                       " must be an integer, not %s."%option[1])
            elif option[0]=='--offshellness':
                CMS_options['offshellness'] = float(option[1])
                if CMS_options['offshellness']<=-1.0:
                    raise self.InvalidCmd('Offshellness must be number larger or'+
                           ' equal to -1.0, not %f'%CMS_options['offshellness'])
            elif option[0]=='--analyze':
                options['analyze'] = option[1]
            elif option[0]=='--show_plot':
                options['show_plot'] = 'true' in option[1].lower()
            elif option[0]=='--report':
                options['report'] = option[1].lower()
            elif option[0]=='--seed':
                options['seed'] = int(option[1])
            elif option[0]=='--name':
                if '.' in option[1]:
                    raise self.InvalidCmd("Do not specify the extension in the"+
                                                             " name of the run")
                CMS_options['name'] = option[1]
            elif option[0]=='--resonances':
                if option[1]=='all':
                    CMS_options['resonances']  = 'all'
                else:
                    try:
                        resonances=eval(option[1])
                    except:
                        raise self.InvalidCmd("Could not evaluate 'resonances'"+
                                                       " option '%s'"%option[1])
                    if isinstance(resonances,int) and resonances>0:
                        CMS_options['resonances']  = resonances
                    elif isinstance(resonances,list) and all(len(res)==2 and 
                        isinstance(res[0],int) and all(isinstance(i, int) for i in 
                                                     res[1]) for res in resonances):
                        CMS_options['resonances']  = resonances
                    else:
                        raise self.InvalidCmd("The option 'resonances' can only be 'all'"+
                               " or and integer or a list of tuples of the form "+
                               "(resPDG,(res_mothers_ID)). You gave '%s'"%option[1])
            elif option[0]=='--tweak':
                # Lists the sets of custom and widths modifications to apply
                value = option[1]
                # Set a shortcuts for applying all relevant tweaks
                if value=='alltweaks':
                    value=str(['default','seed667(seed667)','seed668(seed668)',
                      'allwidths->0.9*allwidths(widths_x_0.9)',
                      'allwidths->0.99*allwidths(widths_x_0.99)',
                      'allwidths->1.01*allwidths(widths_x_1.01)',
                      'allwidths->1.1*allwidths(widths_x_1.1)',                      
                      'logp->logm(logp2logm)','logm->logp(logm2logp)'])
                try:
                    tweaks = eval(value)
                    if isinstance(tweaks, str):
                        tweaks = [value]                         
                    elif not isinstance(tweaks,list):
                        tweaks = [value]
                except:
                    tweaks = [value]
                if not all(isinstance(t,str) for t in tweaks):
                    raise self.InvalidCmd("Invalid specificaiton of tweaks: %s"%value)
                CMS_options['tweak'] = []
                for tweakID, tweakset in enumerate(tweaks):
                    specs =re.match(r'^(?P<tweakset>.*)\((?P<name>.*)\)$', tweakset)
                    if specs:
                        tweakset = specs.group('tweakset')
                        name    = specs.group('name')
                    else:
                        if tweakset!='default':
                            name = 'tweak_%d'%(tweakID+1)
                        else:
                            name = ''
                    new_tweak_set = {'custom':[],'params':{},'name':name}
                    for tweak in tweakset.split('&'):
                        if tweak=='default':
                            continue
                        if tweak.startswith('seed'):
                            new_tweak_set['custom'].append(tweak)
                            continue
                        try:
                            param, replacement = tweak.split('->')
                        except ValueError:
                            raise self.InvalidCmd("Tweak specification '%s'"%\
                                    tweak+" is incorrect. It should be of"+\
                                 " the form a->_any_function_of_(a,lambdaCMS).")
                        if param in ['logp','logm','log'] and \
                           replacement in ['logp','logm','log']:
                            new_tweak_set['custom'].append(tweak)
                            continue
                        try:
                            # for safety prefix parameters, because 'as' for alphas
                            # is a python reserved name for example
                            orig_param, orig_replacement = param, replacement
                            replacement = replacement.replace(param,
                                                        '__tmpprefix__%s'%param)
                            param = '__tmpprefix__%s'%param
                            res = float(eval(replacement.lower(),
                                         {'lambdacms':1.0,param.lower():98.85}))
                        except:                    
                            raise self.InvalidCmd("The substitution expression "+
                        "'%s' for the tweaked parameter"%orig_replacement+
                        " '%s' could not be evaluated. It must be an "%orig_param+
                        "expression of the parameter and 'lambdaCMS'.")
                        new_tweak_set['params'][param.lower()] = replacement.lower()
                    CMS_options['tweak'].append(new_tweak_set)

            elif option[0]=='--recompute_width':
                if option[1].lower() not in ['never','always','first_time','auto']:
                    raise self.InvalidCmd("The option 'recompute_width' can "+\
                  "only be 'never','always', 'first_time' or 'auto' (default).")
                CMS_options['recompute_width'] = option[1]
            elif option[0]=='--loop_filter':
                # Specify a loop, filter. See functions get_loop_filter and
                # user_filter in loop_diagram_generation.LoopAmplitude for
                # information on usage.
                CMS_options['loop_filter'] = '='.join(option[1:])
            elif option[0]=='--diff_lambda_power':
                #'secret' option to chose by which lambda power one should divide
                # the nwa-cms difference. Useful to set to 2 when doing the Born check
                # to see whether the NLO check will have sensitivity to the CMS
                # implementation
                try:
                    CMS_options['diff_lambda_power']=float(option[1])
                except ValueError:
                    raise self.InvalidCmd("the '--diff_lambda_power' option"+\
                            " must be an integer or float, not '%s'."%option[1])
            elif option[0]=='--lambda_plot_range':
                try:
                    plot_range=eval(option[1])
                except Exception as e:
                    raise self.InvalidCmd("The plot range specified %s"%option[1]+\
                                   " is not a valid syntax. Error:\n%s"%str(e))
                if not isinstance(plot_range,(list,tuple)) or \
                    len(plot_range)!=2 or any(not isinstance(p,(float,int)) 
                                                           for p in plot_range):                    
                    raise self.InvalidCmd("The plot range specified %s"\
                                                       %option[1]+" is invalid")
                CMS_options['lambda_plot_range']=list([float(p) for p in plot_range])
            elif option[0]=='--lambdaCMS':
                try:
                    lambda_values = eval(option[1])
                except SyntaxError:
                    raise self.InvalidCmd("'%s' is not a correct"%option[1]+
                                     " python expression for lambdaCMS values.")
                if isinstance(lambda_values,list):
                    if lambda_values[0]!=1.0:
                        raise self.InvalidCmd("The first value of the lambdaCMS values"+
                                " specified must be 1.0, not %s"%str(lambda_values))
                    for l in lambda_values:
                        if not isinstance(l,float):
                            raise self.InvalidCmd("All lambda CMS values must be"+
                                                          " float, not '%s'"%str(l))
                elif isinstance(lambda_values,(tuple,float)):
                    # Format here is then (lower_bound, N) were lower_bound is
                    # the minimum lambdaCMS value that must be probed and the
                    # integer N is the number of such values that must be 
                    # uniformly distributed in each intervale [1.0e-i,1.0e-(i+1)]
                    if isinstance(lambda_values, float):
                        # Use default of 10 for the number of lambda values
                        lower_bound = lambda_values
                        N = 10
                    else:
                        if isinstance(lambda_values[0],float) and \
                           isinstance(lambda_values[1],int):
                            lower_bound = lambda_values[0]
                            N = lambda_values[1]
                        else:
                            raise self.InvalidCmd("'%s' must be a "%option[1]+
                                               "tuple with types (float, int).")
                    lambda_values = create_lambda_values_list(lower_bound,N)
                else:
                    raise self.InvalidCmd("'%s' must be an expression"%option[1]+
                                          " for either a float, tuple or list.")
                lower_bound = lambda_values[-1]
                # and finally add 5 points for stability test on the last values
                # Depending on how the stab test will behave at NLO, we can 
                # consider automatically adding the values below
#                for stab in range(1,6):
#                    lambda_values.append((1.0+(stab/100.0))*lower_bound)

                CMS_options['lambdaCMS'] = lambda_values
            elif option[0]=='--cms':
                try:
                    CMS_expansion_orders, CMS_expansion_parameters = \
                                                            option[1].split(',')
                except ValueError:
                    raise self.InvalidCmd("CMS expansion specification '%s'"%\
                                                       args[i]+" is incorrect.")
                CMS_options['expansion_orders'] = [expansion_order for 
                             expansion_order in CMS_expansion_orders.split('&')]
                CMS_options['expansion_parameters'] = {}
                for expansion_parameter in CMS_expansion_parameters.split('&'):
                    try:
                        param, replacement = expansion_parameter.split('->')
                    except ValueError:
                        raise self.InvalidCmd("CMS expansion specification '%s'"%\
                          expansion_parameter+" is incorrect. It should be of"+\
                                 " the form a->_any_function_of_(a,lambdaCMS).")
                    try:
                        # for safety prefix parameters, because 'as' for alphas
                        # is a python reserved name for example
                        orig_param, orig_replacement = param, replacement
                        replacement = replacement.replace(param,
                                                        '__tmpprefix__%s'%param)
                        param = '__tmpprefix__%s'%param
                        res = float(eval(replacement.lower(),
                                         {'lambdacms':1.0,param.lower():98.85}))
                    except:                    
                        raise self.InvalidCmd("The substitution expression "+
                        "'%s' for CMS expansion parameter"%orig_replacement+
                        " '%s' could not be evaluated. It must be an "%orig_param+
                        "expression of the parameter and 'lambdaCMS'.")
                    # Put everything lower case as it will be done when
                    # accessing model variables
                    CMS_options['expansion_parameters'][param.lower()]=\
                                                             replacement.lower()
            else:
                raise self.InvalidCmd("The option '%s' is not reckognized."%option[0])

            i=i-1
        args = args[:i+1]
        
        if args[0]=='options':
            # Simple printout of the check command options
            logger_check.info("Options for the command 'check' are:")
            logger_check.info("{:<20}     {}".format('  name','default value'))
            logger_check.info("-"*40)
            for key, value in options.items():
                logger_check.info("{:<20} =   {}".format('--%s'%key,str(value)))
            return

        if args[0].lower()=='cmsoptions':
            # Simple printout of the special check cms options
            logger_check.info("Special options for the command 'check cms' are:")
            logger_check.info("{:<20}     {}".format('  name','default value'))
            logger_check.info("-"*40)
            for key, value in CMS_options.items():
                logger_check.info("{:<20} =   {}".format('--%s'%key,str(value)))
            return        
        
        # Set the seed here if not in cms check and if specified
        if args[0]!='cms' and options['seed']!=-1:
            # Not necessarily optimal as there could be additional call to
            # random() as the code develops, but at least it will encompass
            # everything in this way.
            logger_check.info('Setting random seed to %d.'%options['seed'])
            random.seed(options['seed'])
        
        proc_line = " ".join(args[1:])
        # Don't try to extract the process if just re-analyzing a saved run
        if not (args[0]=='cms' and options['analyze']!='None'):
            myprocdef = self.extract_process(proc_line)

            # Check that we have something
            if not myprocdef:
                raise self.InvalidCmd("Empty or wrong format process, please try again.")
            # For the check command, only the mode 'virt' make sense.
            if myprocdef.get('NLO_mode')=='all':
                myprocdef.set('NLO_mode','virt')
        else:
            myprocdef = None
            
        # If the test has to write out on disk, it should do so at the location
        # specified below where the user must be sure to have writing access.
        output_path = os.getcwd()

        if args[0] in ['timing','stability', 'profile'] and not \
                                        myprocdef.get('perturbation_couplings'):
            raise self.InvalidCmd("Only loop processes can have their "+
                                  " timings or stability checked.")

        if args[0]=='gauge' and \
                    not myprocdef.get('perturbation_couplings') in [[],['QCD']]:
            raise self.InvalidCmd(
"""Feynman vs unitary gauge comparisons can only be done if there are no loop
   propagators affected by this gauge. Typically, either processes at tree level
   or including only QCD perturbations can be considered here.""")

        if args[0]=='gauge' and len(self._curr_model.get('gauge')) < 2:
            raise self.InvalidCmd("The current model does not allow for both "+\
                                                   "Feynman and unitary gauge.")

        # Disable some loggers
        loggers = [logging.getLogger('madgraph.diagram_generation'),
                   logging.getLogger('madgraph.loop_diagram_generation'),
                   logging.getLogger('ALOHA'),
                   logging.getLogger('madgraph.helas_objects'),
                   logging.getLogger('madgraph.loop_exporter'),
                   logging.getLogger('madgraph.export_v4'),
                   logging.getLogger('cmdprint'),
                   logging.getLogger('madgraph.model'),
                   logging.getLogger('madgraph.base_objects')]
        old_levels = [log.level for log in loggers]
        for log in loggers:
            log.setLevel(logging.WARNING)

        # run the check
        cpu_time1 = time.time()
        # Run matrix element generation check on processes

        # The aloha python output has trouble when doing (tree level of course)
        # python output and that loop_mode is True at the beginning.
        # So as a temporary fix for the problem that after doing a check at NLO
        # then a check at LO will fail, I make sure I set it to False if the
        # process is a tree-level one
        if myprocdef:
            if myprocdef.get('perturbation_couplings')==[]:
                aloha.loop_mode = False

        comparisons = []
        gauge_result = []
        gauge_result_no_brs = []
        lorentz_result =[]
        nb_processes = 0
        timings = []
        stability = []
        profile_time = []
        profile_stab = []
        cms_results = []

        if "_cuttools_dir" in dir(self):
            CT_dir = self._cuttools_dir
        else:
            CT_dir =""
            if "MLReductionLib" in MLoptions:
                if 1 in MLoptions["MLReductionLib"]:
                    MLoptions["MLReductionLib"].remove(1)
        # directories for TIR
        TIR_dir={}
        if "_iregi_dir" in dir(self):
            TIR_dir['iregi_dir']=self._iregi_dir
        else:
            if "MLReductionLib" in MLoptions:
                if 3 in MLoptions["MLReductionLib"]:
                    logger_check.warning('IREGI not available on your system; it will be skipped.')                    
                    MLoptions["MLReductionLib"].remove(3)

        if 'pjfry' in self.options and isinstance(self.options['pjfry'],str):
            TIR_dir['pjfry_dir']=self.options['pjfry']
        else:
            if "MLReductionLib" in MLoptions:
                if 2 in MLoptions["MLReductionLib"]:
                    logger_check.warning('PJFRY not available on your system; it will be skipped.')                    
                    MLoptions["MLReductionLib"].remove(2)
                    
        if 'golem' in self.options and isinstance(self.options['golem'],str):
            TIR_dir['golem_dir']=self.options['golem']
        else:
            if "MLReductionLib" in MLoptions:
                if 4 in MLoptions["MLReductionLib"]:
                    logger_check.warning('GOLEM not available on your system; it will be skipped.')
                    MLoptions["MLReductionLib"].remove(4)
        
        if 'samurai' in self.options and isinstance(self.options['samurai'],str):
            TIR_dir['samurai_dir']=self.options['samurai']
        else:
            if "MLReductionLib" in MLoptions:
                if 5 in MLoptions["MLReductionLib"]:
                    logger_check.warning('Samurai not available on your system; it will be skipped.')
                    MLoptions["MLReductionLib"].remove(5)
        
        if 'ninja' in self.options and isinstance(self.options['ninja'],str):
            TIR_dir['ninja_dir']=self.options['ninja']
        else:
            if "MLReductionLib" in MLoptions:
                if 6 in MLoptions["MLReductionLib"]:
                    logger_check.warning('Ninja not available on your system; it will be skipped.')
                    MLoptions["MLReductionLib"].remove(6)
        
        if args[0] in ['timing']:
            timings = process_checks.check_timing(myprocdef,
                                                  param_card = param_card,
                                                  cuttools=CT_dir,
                                                  tir=TIR_dir,
                                                  options = options,
                                                  cmd = self,
                                                  output_path = output_path,
                                                  MLOptions = MLoptions
                                                  )        

        if args[0] in ['stability']:
            stability=process_checks.check_stability(myprocdef,
                                                  param_card = param_card,
                                                  cuttools=CT_dir,
                                                  tir=TIR_dir,
                                                  options = options,
                                                  output_path = output_path,
                                                  cmd = self,
                                                  MLOptions = MLoptions)

        if args[0] in ['profile']:
            # In this case timing and stability will be checked one after the
            # other without re-generating the process.
            profile_time, profile_stab = process_checks.check_profile(myprocdef,
                                                  param_card = param_card,
                                                  cuttools=CT_dir,
                                                  tir=TIR_dir,
                                                  options = options,
                                                  MLOptions = MLoptions,
                                                  output_path = output_path,
                                                  cmd = self)

        if args[0] in  ['gauge', 'full'] and \
          len(self._curr_model.get('gauge')) == 2 and\
                        myprocdef.get('perturbation_couplings') in [[],['QCD']]:

            line = " ".join(args[1:])
            myprocdef = self.extract_process(line)
            if gauge == 'unitary':
                myprocdef_unit = myprocdef
                self.do_set('gauge Feynman', log=False)
                myprocdef_feyn = self.extract_process(line)
            else:
                myprocdef_feyn = myprocdef
                self.do_set('gauge unitary', log=False)
                myprocdef_unit = self.extract_process(line)

            nb_part_unit = len(myprocdef_unit.get('model').get('particles'))
            nb_part_feyn = len(myprocdef_feyn.get('model').get('particles'))
            if nb_part_feyn == nb_part_unit:
                logger_check.error('No Goldstone present for this check!!')
            gauge_result_no_brs = process_checks.check_unitary_feynman(
                                                myprocdef_unit, myprocdef_feyn,
                                                param_card = param_card,
                                                options=options,
                                                cuttools=CT_dir,
                                                tir=TIR_dir,
                                                reuse = options['reuse'],
                                                output_path = output_path,
                                                cmd = self)

            # restore previous settings
            self.do_set('gauge %s' % gauge, log=False)
            nb_processes += len(gauge_result_no_brs)            

        if args[0] in  ['permutation', 'full']:
            comparisons = process_checks.check_processes(myprocdef,
                                            param_card = param_card,
                                            quick = True,
                                            cuttools=CT_dir,
                                            tir=TIR_dir,
                                            reuse = options['reuse'],
                                            cmd = self,
                                            output_path = output_path,
                                            options=options)
            nb_processes += len(comparisons[0])

        if args[0] in ['lorentz', 'full']:
            myprocdeff = copy.copy(myprocdef)
            lorentz_result = process_checks.check_lorentz(myprocdeff,
                                          param_card = param_card,
                                          cuttools=CT_dir,
                                          tir=TIR_dir,
                                          reuse = options['reuse'],
                                          cmd = self,
                                          output_path = output_path,
                                          options=options)
            nb_processes += len(lorentz_result)

        if args[0] in  ['brs', 'full']:
            gauge_result = process_checks.check_gauge(myprocdef,
                                          param_card = param_card,
                                          cuttools=CT_dir,
                                          tir=TIR_dir,
                                          reuse = options['reuse'],
                                          cmd = self,
                                          output_path = output_path,
                                          options=options)
            nb_processes += len(gauge_result)

        # The CMS check is typically more complicated and slower than others
        # so we don't run it automatically with 'full'.
        if args[0] in ['cms']:
            
            cms_original_setup = self.options['complex_mass_scheme']
            process_line = " ".join(args[1:])
            # Merge in the CMS_options to the options
            for key, value in CMS_options.items():
                if key=='tweak':
                    continue
                if key not in options:
                    options[key] = value
                else:
                    raise MadGraph5Error,"Option '%s' is both in the option"%key+\
                                                   " and CMS_option dictionary." 
            
            if options['analyze']=='None':
                cms_results = []
                for tweak in CMS_options['tweak']:
                    options['tweak']=tweak
                    # Try to guess the save path and try to load it before running
                    guessed_proc = myprocdef.get_process(
                      [leg.get('ids')[0] for leg in myprocdef.get('legs') 
                                                       if not leg.get('state')],
                      [leg.get('ids')[0] for leg in myprocdef.get('legs')
                                                           if leg.get('state')])
                    save_path = process_checks.CMS_save_path('pkl',
                    {'ordered_processes':[guessed_proc.base_string()],
                     'perturbation_orders':guessed_proc.get('perturbation_couplings')}, 
                             self._curr_model, options, output_path=output_path)
                    if os.path.isfile(save_path) and options['reuse']:
                        cms_result = save_load_object.load_from_file(save_path)
                        logger_check.info("The cms check for tweak %s is recycled from file:\n %s"%
                                                      (tweak['name'],save_path))
                        if cms_result is None:
                            raise self.InvalidCmd('The complex mass scheme check result'+
                            " file below could not be read.\n     %s"%save_path)
                    else:      
                        cms_result = process_checks.check_complex_mass_scheme(
                                              process_line,
                                              param_card = param_card,
                                              cuttools=CT_dir,
                                              tir=TIR_dir,
                                              cmd = self,
                                              output_path = output_path,
                                              MLOptions = MLoptions,
                                              options=options)
                        # Now set the correct save path
                        save_path = process_checks.CMS_save_path('pkl', cms_result, 
                             self._curr_model, options, output_path=output_path)
                    cms_results.append((cms_result,save_path,tweak['name']))
            else:
                cms_result = save_load_object.load_from_file(
                                               options['analyze'].split(',')[0])
                cms_results.append((cms_result,options['analyze'].split(',')[0],
                                               CMS_options['tweak'][0]['name']))
                if cms_result is None:
                    raise self.InvalidCmd('The complex mass scheme check result'+
                       " file below could not be read.\n     %s"
                                              %options['analyze'].split(',')[0])

            # restore previous settings
            self.do_set('complex_mass_scheme %s'%str(cms_original_setup),
                                                                      log=False)
            # Use here additional key 'ordered_processes'
            nb_processes += len(cms_result['ordered_processes'])

        cpu_time2 = time.time()
        logger_check.info("%i check performed in %s"% (nb_processes,
                                  misc.format_time(int(cpu_time2 - cpu_time1))))

        if args[0] in ['cms']:
                text = "Note that the complex mass scheme test in principle only\n"
                text+= "works for stable particles in final states.\n\ns"            
        if args[0] not in ['timing','stability', 'profile', 'cms']:
            if self.options['complex_mass_scheme']:
                text = "Note that Complex mass scheme gives gauge/lorentz invariant\n"
                text+= "results only for stable particles in final states.\n\ns"
            elif not myprocdef.get('perturbation_couplings'):
                text = "Note That all width have been set to zero for those checks\n\n"
            else:
                text = "\n"
        else:
            text ="\n"

        if timings:
            text += 'Timing result for the '+('optimized' if \
              self.options['loop_optimized_output'] else 'default')+' output:\n'

            text += process_checks.output_timings(myprocdef, timings)
        if stability:
            text += 'Stability result for the '+('optimized' if \
              self.options['loop_optimized_output'] else 'default')+' output:\n'
            text += process_checks.output_stability(stability,output_path)

        if profile_time and profile_stab:
            text += 'Timing result '+('optimized' if \
                    self.options['loop_optimized_output'] else 'default')+':\n'
            text += process_checks.output_profile(myprocdef, profile_stab,
                             profile_time, output_path, options['reuse']) + '\n'
        if lorentz_result:
            text += 'Lorentz invariance results:\n'
            text += process_checks.output_lorentz_inv(lorentz_result) + '\n'
        if gauge_result:
            text += 'Gauge results:\n'
            text += process_checks.output_gauge(gauge_result) + '\n'
        if gauge_result_no_brs:
            text += 'Gauge results (switching between Unitary/Feynman):\n'
            text += process_checks.output_unitary_feynman(gauge_result_no_brs) + '\n'
        if cms_results:
            text += 'Complex mass scheme results (varying width in the off-shell regions):\n'
            cms_result = cms_results[0][0]
            if len(cms_results)>1:
                analyze = []
                for i, (cms_res, save_path, tweakname) in enumerate(cms_results):
                    save_load_object.save_to_file(save_path, cms_res)
                    logger_check.info("Pickle file for tweak '%s' saved to disk at:\n ->%s"%
                                                          (tweakname,save_path))
                    if i==0:
                        analyze.append(save_path)
                    else:
                        analyze.append('%s(%s)'%(save_path,tweakname))
                options['analyze']=','.join(analyze)
                options['tweak']  = CMS_options['tweak'][0]
            
            self._cms_checks.append({'line':line, 'cms_result':cms_result,
                                  'options':options, 'output_path':output_path})
            text += process_checks.output_complex_mass_scheme(cms_result,
              output_path, options, self._curr_model,
              output='concise_text' if options['report']=='concise' else 'text')+'\n'

        if comparisons and len(comparisons[0])>0:
            text += 'Process permutation results:\n'
            text += process_checks.output_comparisons(comparisons[0]) + '\n'
            self._comparisons = comparisons

        # We use the reuse tag for an alternative way of skipping the pager.
        if len(text.split('\n'))>20 and not '-reuse' in line and text!='':
            if 'test_manager' not in sys.argv[0]:
                pydoc.pager(text)

        # Restore diagram logger
        for i, log in enumerate(loggers):
            log.setLevel(old_levels[i])

        # Output the result to the interface directly if short enough or if it
        # was anyway not output to the pager
        if len(text.split('\n'))<=20 or options['reuse']:
            # Useful to really specify what logger is used for ML acceptance tests
            logging.getLogger('madgraph.check_cmd').info(text)
        else:
            logging.getLogger('madgraph.check_cmd').debug(text)

        # clean the globals created.
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        if not options['reuse']:
            process_checks.clean_up(self._mgme_dir)

    # Generate a new amplitude
    def do_generate(self, line):
        """Main commands: Generate an amplitude for a given process"""

        aloha_lib.KERNEL.clean()
        # Reset amplitudes
        self._curr_amps = diagram_generation.AmplitudeList()
        # Reset Helas matrix elements
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        self._generate_info = line
        # Reset _done_export, since we have new process
        self._done_export = False
        # Also reset _export_format and _export_dir
        self._export_format = None


        # Call add process
        args = self.split_arg(line)
        args.insert(0, 'process')
        self.do_add(" ".join(args))

    def extract_process(self, line, proc_number = 0, overall_orders = {}):
        """Extract a process definition from a string. Returns
        a ProcessDefinition."""

        orig_line = line
        # Check basic validity of the line
        if not len(re.findall('>\D', line)) in [1,2]:
            self.do_help('generate')
            raise self.InvalidCmd('Wrong use of \">\" special character.')


        # Perform sanity modifications on the lines:
        # Add a space before and after any > , $ / | [ ]
        space_before = re.compile(r"(?P<carac>\S)(?P<tag>[\\[\\]/\,\\$\\>|])(?P<carac2>\S)")
        line = space_before.sub(r'\g<carac> \g<tag> \g<carac2>', line)

        # Use regular expressions to extract s-channel propagators,
        # forbidden s-channel propagators/particles, coupling orders
        # and process number, starting from the back

        # Start with process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(line)
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            line = proc_number_re.group(1)+ proc_number_re.group(3)
            #overall_order are already handle but it is better to pass the info to each group
        
        # Now check for perturbation orders, specified in between squared brackets
        perturbation_couplings_pattern = \
          re.compile("^(?P<proc>.+>.+)\s*\[\s*((?P<option>\w+)\s*\=)?\s*"+\
                               "(?P<pertOrders>(\w+\s*)*)\s*\]\s*(?P<rest>.*)$")
        perturbation_couplings_re = perturbation_couplings_pattern.match(line)
        perturbation_couplings = ""
        LoopOption= 'tree'
        HasBorn= True
        if perturbation_couplings_re:
            perturbation_couplings = perturbation_couplings_re.group("pertOrders")
            option=perturbation_couplings_re.group("option")
            if option:
                if option in self._valid_nlo_modes:
                    LoopOption=option
                    if option=='sqrvirt':
                        LoopOption='virt'
                        HasBorn=False
                    elif option=='noborn':
                        HasBorn=False
                else:
                    raise self.InvalidCmd, "NLO mode %s is not valid. "%option+\
                       "Valid modes are %s. "%str(self._valid_nlo_modes)
            else:
                LoopOption='all'

            line = perturbation_couplings_re.group("proc")+\
                     perturbation_couplings_re.group("rest")
                        
        ## Now check for orders/squared orders/constrained orders
        order_pattern = re.compile(\
           "^(?P<before>.+>.+)\s+(?P<name>(\w|(\^2))+)\s*(?P<type>"+\
                    "(=|(<=)|(==)|(===)|(!=)|(>=)|<|>))\s*(?P<value>-?\d+)\s*$")
        order_re = order_pattern.match(line)
        squared_orders = {}
        orders = {}
        constrained_orders = {}
        ## The 'split_orders' (i.e. those for which individual matrix element
        ## evalutations must be provided for each corresponding order value) are
        ## defined from the orders specified in between [] and any order for
        ## which there are squared order constraints.
        split_orders = []
        while order_re:
            type = order_re.group('type')
            if order_re.group('name').endswith('^2'):
                if type not in self._valid_sqso_types:
                    raise self.InvalidCmd, "Type of squared order "+\
                                 "constraint '%s'"% type+" is not supported."
                if type == '=':
                    name =  order_re.group('name')
                    value = order_re.group('value')
                    logger.warning("Interpreting '%(n)s=%(v)s' as '%(n)s<=%(v)s'" %\
                                       {'n':name, 'v': value})
                    type = "<="
                squared_orders[order_re.group('name')[:-2]] = \
                                         (int(order_re.group('value')),type)
            else:
                if type not in self._valid_amp_so_types:
                    raise self.InvalidCmd, \
                      "Amplitude order constraints can only be of type %s"%\
                    (', '.join(self._valid_amp_so_types))+", not '%s'."%type
                name = order_re.group('name')
                value = int(order_re.group('value'))
                if type in ['=', '<=']:
                    if type == '=' and value != 0:
                        logger.warning("Interpreting '%(n)s=%(v)s' as '%(n)s<=%(v)s'" %\
                                       {'n':name, 'v': value}) 
                    orders[name] = value
                elif type == "==":
                    constrained_orders[name] = (value, type)
                    if name not in squared_orders:
                        squared_orders[name] = (2 * value,'==')
                    if True:#name not in orders:
                        orders[name] = value
                    
                elif type == ">":
                    constrained_orders[name] = (value, type)
                    if name not in squared_orders:
                        squared_orders[name] = (2 * value,'>')
                                          
            line = order_re.group('before')
            order_re = order_pattern.match(line)          
            
        #only allow amplitue restrctions >/ == for LO/tree level
        if constrained_orders and LoopOption != 'tree':
            raise self.InvalidCmd, \
                          "Amplitude order constraints (for not LO processes) can only be of type %s"%\
                        (', '.join(['<=']))+", not '%s'."%type

        # If the squared orders are defined but not the orders, assume 
        # orders=sq_orders. In case the squared order has a negative value or is
        # defined with the '>' operato, then this order correspondingly set to 
        # be maximal (99) since there is no way to know, during generation, if 
        # the amplitude being contstructed will be leading or not.
        if orders=={} and squared_orders!={}:
            for order in squared_orders.keys():
                if squared_orders[order][0]>=0 and squared_orders[order][1]!='>':
                    orders[order]=squared_orders[order][0]
                else:
                    orders[order]=99
        
        if not self._curr_model['case_sensitive']:
            # Particle names lowercase
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
            elif part_name.isdigit() or part_name.startswith('-') and part_name[1:].isdigit():
                if int(part_name) in self._curr_model.get('particle_dict'):
                    mylegids.append(int(part_name))
                else:
                    raise self.InvalidCmd, \
                      "No pdg_code %s in model" % part_name
            else:
                mypart = self._curr_model['particles'].get_copy(part_name)
                if mypart:
                    mylegids.append(mypart.get_pdg_code())

            if mylegids:
                myleglist.append(base_objects.MultiLeg({'ids':mylegids,
                                                        'state':state}))
            else:
                raise self.InvalidCmd, "No particle %s in model" % part_name

        # Apply the keyword 'all' for perturbed coupling orders.
        if perturbation_couplings.lower()=='all':
            perturbation_couplings=' '.join(self._curr_model['perturbation_couplings'])

        if filter(lambda leg: leg.get('state') == True, myleglist):
            # We have a valid process
            # Extract perturbation orders
            perturbation_couplings_list = perturbation_couplings.split()
            if perturbation_couplings_list==['']:
                perturbation_couplings_list=[]
            # Correspondingly set 'split_order' from the squared orders and the
            # perturbation couplings list
            split_orders=list(set(perturbation_couplings_list+squared_orders.keys()))
            try:
                split_orders.sort(key=lambda elem: 0 if elem=='WEIGHTED' else
                                       self._curr_model['order_hierarchy']
                                       [elem if not elem.endswith('.sqrt') else elem[:-5]])
            except KeyError:
                raise self.InvalidCmd, "The loaded model does not defined a "+\
                    " coupling order hierarchy for these couplings: %s"%\
                      str([so for so in split_orders if so!='WEIGHTED' and so not 
                                 in self._curr_model['order_hierarchy'].keys()])

            # If the loopOption is 'tree' then the user used the syntax 
            # [tree= Orders] for the sole purpose of setting split_orders. We
            # then empty the perturbation_couplings_list at this stage.
            if LoopOption=='tree':
                perturbation_couplings_list = []
            if perturbation_couplings_list and LoopOption not in ['real', 'LOonly']:
                if not isinstance(self._curr_model,loop_base_objects.LoopModel):
                    raise self.InvalidCmd(\
                      "The current model does not allow for loop computations.")
                else:
                    for pert_order in perturbation_couplings_list:
                        if pert_order not in self._curr_model['perturbation_couplings']:
                            raise self.InvalidCmd(\
                                "Perturbation order %s is not among" % pert_order + \
                                " the perturbation orders allowed for by the loop model.")
            if not self.options['loop_optimized_output'] and \
                         LoopOption not in ['tree','real'] and split_orders!=[]:
                logger.warning('The default output mode (loop_optimized_output'+\
                  ' = False) does not support evaluations for given powers of'+\
                  ' coupling orders. MadLoop output will therefore not be'+\
                  ' able to provide such quantities.')
                split_orders = []
                       
            # Now extract restrictions
            forbidden_particle_ids = \
                              self.extract_particle_ids(forbidden_particles)
            if forbidden_particle_ids and \
               isinstance(forbidden_particle_ids[0], list):
                raise self.InvalidCmd(\
                      "Multiparticle %s is or-multiparticle" % part_name + \
                      " which can be used only for required s-channels")
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
            
            sqorders_values = dict([(k,v[0]) for k, v in squared_orders.items()])
            if len([1 for sqo_v in sqorders_values.values() if sqo_v<0])>1:
                raise self.InvalidCmd(
                  "At most one negative squared order constraint can be specified.")
            
            sqorders_types = dict([(k,v[1]) for k, v in squared_orders.items()]) 
                        
            
            out = base_objects.ProcessDefinition({'legs': myleglist,
                              'model': self._curr_model,
                              'id': proc_number,
                              'orders': orders,
                              'squared_orders':sqorders_values,
                              'sqorders_types':sqorders_types,
                              'constrained_orders': constrained_orders,
                              'forbidden_particles': forbidden_particle_ids,
                              'forbidden_onsh_s_channels': forbidden_onsh_schannel_ids,
                              'forbidden_s_channels': forbidden_schannel_ids,
                              'required_s_channels': required_schannel_ids,
                              'overall_orders': overall_orders,
                              'perturbation_couplings': perturbation_couplings_list,
                              'has_born':HasBorn,
                              'NLO_mode':LoopOption,
                              'split_orders':split_orders
                              })
            return out
        #                       'is_decay_chain': decay_process\


    def create_loop_induced(self, line, myprocdef=None):
        """ Routine to create the MultiProcess for the loop-induced case"""
        
        args = self.split_arg(line)
        
        warning_duplicate = True
        if '--no_warning=duplicate' in args:
            warning_duplicate = False
            args.remove('--no_warning=duplicate')
        
        # Check the validity of the arguments
        self.check_add(args)
        if args[0] == 'process':
            args = args[1:]
        
        # special option for 1->N to avoid generation of kinematically forbidden
        #decay.
        if args[-1].startswith('--optimize'):
            optimize = True
            args.pop()
        else:
            optimize = False

    
        if not myprocdef:
            myprocdef = self.extract_process(' '.join(args))
        
        myprocdef.set('NLO_mode', 'noborn')
            
        # store the first process (for the perl script)
        if not self._generate_info:
            self._generate_info = line
                
        # Reset Helas matrix elements
        #self._curr_matrix_elements = helas_objects.HelasLoopInducedMultiProcess()


        # Check that we have the same number of initial states as
        # existing processes
        if self._curr_amps and self._curr_amps[0].get_ninitial() != \
               myprocdef.get_ninitial():
            raise self.InvalidCmd("Can not mix processes with different number of initial states.")               
      
        if self._curr_amps and (not isinstance(self._curr_amps[0], loop_diagram_generation.LoopAmplitude) or \
             self._curr_amps[0]['has_born']):
            raise self.InvalidCmd("Can not mix loop induced process with not loop induced process")
            
        # Negative coupling order contraints can be given on at most one
        # coupling order (and either in squared orders or orders, not both)
        if len([1 for val in myprocdef.get('orders').values()+\
                      myprocdef.get('squared_orders').values() if val<0])>1:
            raise MadGraph5Error("Negative coupling order constraints"+\
              " can only be given on one type of coupling and either on"+\
                           " squared orders or amplitude orders, not both.")

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

        # Decide here wether one needs a LoopMultiProcess or a MultiProcess

        myproc = loop_diagram_generation.LoopInducedMultiProcess(myprocdef,
                                 collect_mirror_procs = collect_mirror_procs,
                                 ignore_six_quark_processes = ignore_six_quark_processes,
                                 optimize=optimize)

        for amp in myproc.get('amplitudes'):
            if amp not in self._curr_amps:
                self._curr_amps.append(amp)
                if amp['has_born']:
                    raise Exception
            elif warning_duplicate:
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

    @staticmethod
    def split_process_line(procline):
        """Takes a valid process and return
           a tuple (core_process, options). This removes
             - any NLO specifications.
             - any options
           [Used by MadSpin]
        """

        # remove the tag "[*]": this tag is used in aMC@LNO ,
        # but it is not a valid syntax for LO
        line=procline
        pos1=line.find("[")
        if pos1>0:
            pos2=line.find("]")
            if pos2 >pos1:
                line=line[:pos1]+line[pos2+1:]
        #
        # Extract the options:
        #
        # A. Remove process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(line)
        if proc_number_re:
            line = proc_number_re.group(1) + proc_number_re.group(3)

        # B. search for the beginning of the option string
        pos=1000
        # start with order
        order_pattern = re.compile("^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(line)
        if (order_re):
            pos_order=line.find(order_re.group(2))
            if pos_order>0 and pos_order < pos : pos=pos_order

        # then look for slash or dollar
        slash = line.find("/")
        if slash > 0 and slash < pos: pos=slash
        dollar = line.find("$")
        if dollar > 0 and dollar < pos: pos=dollar

        if pos<1000:
            proc_option=line[pos:]
            line=line[:pos]
        else:
            proc_option=""

        return line, proc_option

    def get_final_part(self, procline):
        """Takes a valid process and return
           a set of id of final states particles. [Used by MadSpin]
        """

        if not self._curr_model['case_sensitive']:
            procline = procline.lower()
        pids = self._curr_model.get('name2pdg')

        # method.
        # 1) look for decay.
        #     in presence of decay call this routine recursively and veto
        #     the particles which are decayed

        # Deal with decay chain
        if ',' in procline:
            core, decay = procline.split(',', 1)
            core_final = self.get_final_part(core)

            #split the decay
            all_decays = decay.split(',')
            nb_level,  tmp_decay = 0, ''
            decays = []
            # deal with ()
            for one_decay in all_decays:
                if '(' in one_decay:
                    nb_level += 1
                if ')' in one_decay:
                    nb_level -= 1

                if nb_level:
                    if tmp_decay:
                        tmp_decay += ', %s' % one_decay
                    else:
                        tmp_decay = one_decay
                elif tmp_decay:
                    final = '%s,%s' % (tmp_decay, one_decay)
                    final = final.strip()
                    assert final[0] == '(' and final[-1] == ')'
                    final = final[1:-1]
                    decays.append(final)
                    tmp_decay = ''
                else:
                    decays.append(one_decay)
            # remove from the final states all particles which are decayed
            for one_decay in decays:
                first = one_decay.split('>',1)[0].strip()
                if first in pids:
                    pid = set([pids[first]])
                elif first in self._multiparticles:
                    pid = set(self._multiparticles[first])
                else:
                    raise Exception, 'invalid particle name: %s. ' % first
                core_final.difference_update(pid)
                core_final.update(self.get_final_part(one_decay))

            return core_final

        # NO DECAY CHAIN
        final = set()
        final_states = re.search(r'> ([^\/\$\=\@>]*)(\[|\s\S+\=|\$|\/|\@|$)', procline)
        particles = final_states.groups()[0]
        for particle in particles.split():
            if particle in pids:
                final.add(pids[particle])
            elif particle in self._multiparticles:
                final.update(set(self._multiparticles[particle]))
        return final

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
            elif part_name.isdigit() or (part_name.startswith('-') and part_name[1:].isdigit()):
                ids.append([int(part_name)])
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

    def extract_decay_chain_process(self, line, level_down=False, proc_number=0):
        """Recursively extract a decay chain process definition from a
        string. Returns a ProcessDefinition."""

        # Start with process number (identified by "@") and overall orders
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*((\w+\s*=\s*\d+\s*)*)$")
        proc_number_re = proc_number_pattern.match(line)
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
            # special cases: parenthesis but no , => remove the paranthesis!
            if line.lstrip()[0] == '(' and index_par !=-1 and \
                                                    not ',' in line[:index_par]:
                par_start = line.find('(')
                line = '%s %s' % (line[par_start+1:index_par], line[index_par+1:]) 
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
    def do_import(self, line, force=False):
        """Main commands: Import files with external formats"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_import(args)
        if args[0].startswith('model'):
            self._model_v4_path = None
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
                # avoid loading the qcd/qed model twice
                if (args[1].startswith('loop_qcd_qed_sm') or\
                    args[1].split('/')[-1].startswith('loop_qcd_qed_sm')) and\
                     self.options['gauge']!='Feynman':
                    logger.info('Switching to Feynman gauge because '+\
                          'it is the only one supported by the model %s.'%args[1])
                    self._curr_model = None
                    self.do_set('gauge Feynman',log=False)
                prefix = not '--noprefix' in args
                if prefix:
                    aloha.aloha_prefix='mdl_'
                else:
                    aloha.aloha_prefix=''
                
                try:
                    self._curr_model = import_ufo.import_model(args[1], prefix=prefix,
                        complex_mass_scheme=self.options['complex_mass_scheme'])
                except import_ufo.UFOImportError, error:
                    if 'not a valid UFO model' in str(error):
                        logger_stderr.warning('WARNING: %s' % error)
                        logger_stderr.warning('Try to recover by running '+\
                         'automatically `import model_v4 %s` instead.'% args[1])
                    self.exec_cmd('import model_v4 %s ' % args[1], precmd=True)
                    return
                if self.options['gauge']=='unitary':
                    if not force and isinstance(self._curr_model,\
                                              loop_base_objects.LoopModel) and \
                         self._curr_model.get('perturbation_couplings') not in \
                                                                   [[],['QCD']]:
                        if 1 not in self._curr_model.get('gauge') :
                            logger_stderr.warning('This model does not allow Feynman '+\
                              'gauge. You will only be able to do tree level '+\
                                                'QCD loop cmputations with it.')
                        else:
                            logger.info('Change to the gauge to Feynman because '+\
                          'this loop model allows for more than just tree level'+\
                                                      ' and QCD perturbations.')
                            self.do_set('gauge Feynman', log=False)
                            return
                    if 0 not in self._curr_model.get('gauge') :
                        logger_stderr.warning('Change the gauge to Feynman since '+\
                                       'the model does not allow unitary gauge')
                        self.do_set('gauge Feynman', log=False)
                        return
                else:
                    if 1 not in self._curr_model.get('gauge') :
                        logger_stderr.warning('Change the gauge to unitary since the'+\
                          ' model does not allow Feynman gauge.'+\
                                                  ' Please re-import the model')
                        self._curr_model = None
                        self.do_set('gauge unitary', log= False)
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

    def remove_pointless_decay(self, param_card):
        """ For simple decay chain: remove diagram that are not in the BR.
            param_card should be a ParamCard instance."""

        assert isinstance(param_card, check_param_card.ParamCard)

        # Collect amplitudes
        amplitudes = diagram_generation.AmplitudeList()
        for amp in self._curr_amps:
            amplitudes.extend(amp.get_amplitudes())

        decay_tables = param_card['decay'].decay_table
        to_remove = []
        for amp in amplitudes:
            mother = [l.get('id') for l in amp['process'].get('legs') \
                                                        if not l.get('state')]
            if 1 == len(mother):
                try:
                    decay_table = decay_tables[abs(mother[0])]
                except KeyError:
                    logger.warning("No decay table for %s. decay of this particle with MadSpin should be discarded" % abs(mother[0]))
                    continue # No BR for this particle -> accept all.
                # create the tuple associate to the decay mode
                child = [l.get('id') for l in amp['process'].get('legs') \
                                                              if l.get('state')]
                if not mother[0] > 0:
                    child = [x if self._curr_model.get_particle(x)['self_antipart']
                             else -x for x in child]
                child.sort()
                child.insert(0, len(child))
                #check if the decay is present or not:
                if tuple(child) not in decay_table.keys():
                    to_remove.append(amp)

        def remove_amp(amps):
            for amp in amps[:]:
                if amp in to_remove:
                    amps.remove(amp)
                if isinstance(amp, diagram_generation.DecayChainAmplitude):
                    remove_amp(amp.get('decay_chains'))
                    for decay in amp.get('decay_chains'):
                        remove_amp(decay.get('amplitudes'))
        remove_amp(self._curr_amps)


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
            except Exception:
                del self._multiparticles[key]
                defined_multiparticles.remove(key)
                removed_multiparticles.append(key)

        # Now add default multiparticles
        for line in open(pjoin(MG5DIR, 'input', \
                                      'multiparticles_default.txt')):
            if line.startswith('#'):
                continue
            try:
                if not self._curr_model['case_sensitive']:
                    multipart_name = line.lower().split()[0]
                else:
                    multipart_name = line.split()[0]
                if multipart_name not in self._multiparticles:
                    #self.do_define(line)
                    self.exec_cmd('define %s' % line, printcmd=False, precmd=True)
            except self.InvalidCmd, why:
                logger_stderr.warning('impossible to set default multiparticles %s because %s' %
                                        (line.split()[0],why))
                if self.history[-1] == 'define %s' % line.strip():
                    self.history.pop(-1)
                else:
                    misc.sprint([self.history[-1], 'define %s' % line.strip()])

        scheme = "old"
        for qcd_container in ['p', 'j']:
            if qcd_container not in self._multiparticles:
                continue
            multi = self._multiparticles[qcd_container]
            b = self._curr_model.get_particle(5)
            if not b:
                break

            if 5 in multi:
                if b['mass'] != 'ZERO':
                    multi.remove(5)
                    multi.remove(-5)
                    scheme = 4
            elif b['mass'] == 'ZERO':
                multi.append(5)
                multi.append(-5)
                scheme = 5
                
        if scheme in [4,5]:
            logger.warning("Pass the definition of \'j\' and \'p\' to %s flavour scheme." % scheme)
            for container in ['p', 'j']:
                if container in defined_multiparticles:
                    defined_multiparticles.remove(container)
            self.history.append("define p = %s # pass to %s flavors" % \
                                (' ' .join([`i` for i in self._multiparticles['p']]), 
                                 scheme) 
                               )
            self.history.append("define j = p")
                
        
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

    def advanced_install(self, tool_to_install, 
                                            HepToolsInstaller_web_address=None,
                                            additional_options=[]):
        """ Uses the HEPToolsInstaller.py script maintened online to install
        HEP tools with more complicated dependences.
        Additional options will be added to the list when calling HEPInstaller"""

        # Always refresh the installer if already present
        if not os.path.isdir(pjoin(MG5DIR,'HEPTools','HEPToolsInstallers')):
            if HepToolsInstaller_web_address is None:
                raise MadGraph5Error, "The option 'HepToolsInstaller_web_address'"+\
                             " must be specified in function advanced_install"+\
                                " if the installers are not already downloaded."
            if not os.path.isdir(pjoin(MG5DIR,'HEPTools')):
                os.mkdir(pjoin(MG5DIR,'HEPTools'))
        elif not HepToolsInstaller_web_address is None:
            shutil.rmtree(pjoin(MG5DIR,'HEPTools','HEPToolsInstallers'))
        if not HepToolsInstaller_web_address is None:
            logger.info('Downloading the HEPToolInstaller at:\n   %s'%
                                                  HepToolsInstaller_web_address)
            # Guess if it is a local or web address
            if '//' in HepToolsInstaller_web_address:
                if sys.platform == "darwin":
                    misc.call(['curl', HepToolsInstaller_web_address, '-o%s' 
                      %pjoin(MG5DIR,'HEPTools','HEPToolsInstallers.tar.gz')],
                      stderr=open(os.devnull,'w'), stdout=open(os.devnull,'w'),
                                                                         cwd=MG5DIR)
                else:
                    misc.call(['wget', HepToolsInstaller_web_address, 
                      '--output-document=%s'% pjoin(MG5DIR,'HEPTools',
                      'HEPToolsInstallers.tar.gz')], stderr=open(os.devnull, 'w'),
                                           stdout=open(os.devnull, 'w'), cwd=MG5DIR)
            else:
                # If it is a local tarball, then just copy it
                shutil.copyfile(HepToolsInstaller_web_address,
                           pjoin(MG5DIR,'HEPTools','HEPToolsInstallers.tar.gz'))

            # Untar the file
            returncode = misc.call(['tar', '-xzpf', 'HEPToolsInstallers.tar.gz'],
                     cwd=pjoin(MG5DIR,'HEPTools'), stdout=open(os.devnull, 'w'))
            
            # Remove the tarball
            os.remove(pjoin(MG5DIR,'HEPTools','HEPToolsInstallers.tar.gz'))
            
############## FOR DEBUGGING ONLY, Take HEPToolsInstaller locally ##############
#            shutil.rmtree(pjoin(MG5DIR,'HEPTools','HEPToolsInstallers'))
#            shutil.copytree(os.path.abspath(pjoin(MG5DIR,os.path.pardir,
#           'HEPToolsInstallers')),pjoin(MG5DIR,'HEPTools','HEPToolsInstallers'))
################################################################################
            
        # Potential change in naming convention
        name_map = {}
        try:
            tool = name_map[tool_to_install]
        except:
            tool = tool_to_install
     
        # Compiler options
        compiler_options = []
        if not self.options['cpp_compiler'] is None:
            compiler_options.append('--cpp_compiler=%s'%
                                                   self.options['cpp_compiler'])
            compiler_options.append('--cpp_standard_lib=%s'%
               misc.detect_cpp_std_lib_dependence(self.options['cpp_compiler']))
        else:
            compiler_options.append('--cpp_standard_lib=%s'%
               misc.detect_cpp_std_lib_dependence(self.options['cpp_compiler']))

        if not self.options['fortran_compiler'] is None:
            compiler_options.append('--fortran_compiler=%s'%
                                               self.options['fortran_compiler'])

        # Add the path of pythia8 if known and the MG5 path
        if tool=='mg5amc_py8_interface':
            additional_options.append('--mg5_path=%s'%MG5DIR)
            # Warn about the soft dependency to gnuplot
            if misc.which('gnuplot') is None:
                logger.warning("==========")
                logger.warning("The optional dependency 'gnuplot' for the tool"+\
                 " 'mg5amc_py8_interface' was not found. We recommend that you"+\
                 " install it so as to be able to view the plots related to "+\
                                                      " merging with Pythia 8.")
                logger.warning("==========")
            if self.options['pythia8_path']:
                additional_options.append(
                               '--with_pythia8=%s'%self.options['pythia8_path'])

##### FOR DEBUGGING ONLY, until the mg5amc_py8_interface is put online  ########
#            additional_options.append('--mg5amc_py8_interface_tarball=%s'%
#                    pjoin(MG5DIR,os.path.pardir,'MG5aMC_PY8_interface',
#                                                 'MG5aMC_PY8_interface.tar.gz'))
################################################################################

        # Special rules for certain tools  
        if tool=='pythia8':
            # All what's below is to handle the lhapdf dependency of Pythia8
            lhapdf_config  = misc.which(self.options['lhapdf'])
            lhapdf_version = None
            if lhapdf_config is None:
                lhapdf_version = None
            else:
                try:
                    version = misc.Popen(
                           [lhapdf_config,'--version'], stdout=subprocess.PIPE)
                    lhapdf_version = int(version.stdout.read()[0])
                    if lhapdf_version not in [5,6]:
                        raise 
                except:
                    raise self.InvalidCmd('Could not detect LHAPDF version. Make'+
                           " sure '%s --version ' runs properly."%lhapdf_config)
        
            if lhapdf_version is None:
                answer = self.ask(question=
"\033[33;34mLHAPDF was not found. Do you want to install LHPADF6? "+
"(recommended) \033[0m \033[33;32my\033[0m/\033[33;31mn\033[0m >",
                                                default='y',text_format='33;32')
                if not answer.lower() in ['y','']:
                    lhapdf_path = None
                else:
                    self.advanced_install('lhapdf6',
                                          additional_options=additional_options)
                    lhapdf_path = pjoin(MG5DIR,'HEPTools','lhapdf6')
                    lhapdf_version = 6
            else:
                lhapdf_path = os.path.abspath(pjoin(os.path.dirname(\
                                                 lhapdf_config),os.path.pardir))
            if lhapdf_version is None:
                logger.warning('You decided not to link the Pythia8 installation'+
                  ' to LHAPDF. Beware that only built-in PDF sets can be used then.')
            else:
                logger.info('Pythia8 will be linked to LHAPDF v%d.'%lhapdf_version)
            logger.info('Now installing Pythia8. Be patient...','$MG:color:GREEN')
            lhapdf_option = []
            if lhapdf_version is None:
                lhapdf_option.append('--with_lhapdf6=OFF')
                lhapdf_option.append('--with_lhapdf5=OFF')                
            elif lhapdf_version==5:
                lhapdf_option.append('--with_lhapdf5=%s'%lhapdf_path)
                lhapdf_option.append('--with_lhapdf6=OFF')
            elif lhapdf_version==6:
                lhapdf_option.append('--with_lhapdf5=OFF')
                lhapdf_option.append('--with_lhapdf6=%s'%lhapdf_path)
            # Make sure each otion in additional_options appears only once
            additional_options = list(set(additional_options))
             # And that the option '--force' is placed last.
            additional_options = [opt for opt in additional_options if opt!='--force']+\
                        (['--force'] if '--force' in additional_options else [])
            return_code = misc.call([pjoin(MG5DIR,'HEPTools',
             'HEPToolsInstallers','HEPToolInstaller.py'),'pythia8',
             '--prefix=%s'%pjoin(MG5DIR,'HEPTools')]
                        + lhapdf_option + compiler_options + additional_options)
        else:
            logger.info('Now installing %s. Be patient...'%tool)
            # Make sure each otion in additional_options appears only once
            additional_options = list(set(additional_options))
             # And that the option '--force' is placed last.
            additional_options = [opt for opt in additional_options if opt!='--force']+\
                        (['--force'] if '--force' in additional_options else [])
            return_code = misc.call([pjoin(MG5DIR,'HEPTools',
              'HEPToolsInstallers', 'HEPToolInstaller.py'), tool,'--prefix=%s'%
              pjoin(MG5DIR,'HEPTools')] + compiler_options + additional_options)

        if return_code == 0:
            logger.info("%s successfully installed in %s."%(
                   tool_to_install, pjoin(MG5DIR,'HEPTools')),'$MG:color:GREEN')
        elif return_code == 66:
            answer = self.ask(question=
"""\033[33;34mTool %s already installed in %s."""%(tool_to_install, pjoin(MG5DIR,'HEPTools'))+
""" Do you want to overwrite its installation?\033[0m \033[33;32my\033[0m/\033[33;31mn\033[0m >"""
    ,default='y',text_format='33;32')
            if not answer.lower() in ['y','']:
                logger.info("Installation of %s aborted."%tool_to_install,
                                                              '$MG:color:GREEN')
                return
            else:
                return self.advanced_install(tool_to_install,
                              additional_options=additional_options+['--force'])            
        else:
            raise self.InvalidCmd("Installation of %s failed."%tool_to_install)

        # Post-installation treatment
        if tool == 'pythia8':
            self.options['pythia8_path'] = pjoin(MG5DIR,'HEPTools','pythia8')
            self.exec_cmd('save options')
            # Automatically re-install the mg5amc_py8_interface after a fresh
            # Pythia8 installation
            self.advanced_install('mg5amc_py8_interface',
                              additional_options=additional_options+['--force'])          
        elif tool == 'lhapdf6':
            self.options['lhapdf'] = pjoin(MG5DIR,'HEPTools','lhapdf6','bin',
                                                                'lhapdf-config')
            self.exec_cmd('save options')
        elif tool == 'lhapdf5':
            self.options['lhapdf'] = pjoin(MG5DIR,'HEPTools','lhapdf5','bin',
                                                                'lhapdf-config')
            self.exec_cmd('save options')            

        elif tool == 'mg5amc_py8_interface':
            self.options['mg5amc_py8_interface_path'] = \
                                 pjoin(MG5DIR,'HEPTools','MG5aMC_PY8_interface')
            self.exec_cmd('save options')      

        elif tool == 'ninja':
            if not misc.get_ninja_quad_prec_support(pjoin(
                                              MG5DIR,'HEPTools','ninja','lib')):
                logger.warning(
"""Successful installation of Ninja, but without support for quadruple precision
arithmetics. If you want to enable this (hence improving the treatment of numerically
unstable points in the loop matrix elements) you can try to reinstall Ninja with:
  MG5aMC>install ninja
After having made sure to have selected a C++ compiler in the 'cpp' option of
MG5aMC that supports quadruple precision (typically g++ based on gcc 4.6+).""")
            self.options['ninja'] = pjoin(os.curdir,'HEPTools','lib')
            self.exec_cmd('save options')
            
        # Now warn the user if he didn't add HEPTools first in his environment
        # variables.
        path_to_be_set = []
        if sys.platform == "darwin":
            library_variables = ["DYLD_LIBRARY_PATH"]
        else:
            library_variables = ["LD_LIBRARY_PATH"]
        for variable in library_variables:
            if (variable not in os.environ) or \
                not any(os.path.abspath(pjoin(MG5DIR,'HEPTools','lib'))==\
                os.path.abspath(path) for path in os.environ[variable].split(os.pathsep)):
                path_to_be_set.append((variable,
                               os.path.abspath(pjoin(MG5DIR,'HEPTools','lib'))))
        for variable in ["PATH"]:
            if (variable not in os.environ) or \
                not any(os.path.abspath(pjoin(MG5DIR,'HEPTools','bin'))==\
                os.path.abspath(path) for path in os.environ[variable].split(os.pathsep)):
                path_to_be_set.append((variable,
                               os.path.abspath(pjoin(MG5DIR,'HEPTools','bin'))))
            if (variable not in os.environ) or \
                not any(os.path.abspath(pjoin(MG5DIR,'HEPTools','include'))==\
                os.path.abspath(path) for path in os.environ[variable].split(os.pathsep)):
                path_to_be_set.append((variable,
                               os.path.abspath(pjoin(MG5DIR,'HEPTools','include'))))
       
        if len(path_to_be_set)>0:
            shell_type = misc.get_shell_type()
            if shell_type in ['bash',None]:
                modification_line = r"printf '# MG5aMC paths:\n%s' >> ~/.bashrc"%\
                (r'\n'.join('export %s=%s%s'%
                (var,path,'%s$%s'%(os.pathsep,var) if var in os.environ else '') 
                                                for var,path in path_to_be_set))
            elif shell_type=='tcsh':
                modification_line = r"printf '# MG5aMC paths:\n%s' >> ~/.cshrc"%\
                (r'\n'.join('setenv %s %s%s'%
                (var,path,'%s$%s'%(os.pathsep,var) if var in os.environ else '')
                                                for var,path in path_to_be_set))

            logger.warning("==========")
            logger.warning("We recommend that you add to the following paths"+\
             " to your environment variables, so that you are guaranteed that"+\
             " at runtime, MG5_aMC will use the tools you have just installed"+\
             " and not some other versions installed elsewhere on your system.\n"+\
             "You can do so by running the following command in your terminal:"
             "\n   %s"%modification_line) 
            logger.warning("==========")
    
        # Return true for successful installation
        return True
    
    def do_install(self, line, paths=None, additional_options=[]):
        """Install optional package from the MG suite.
        The argument 'additional_options' will be passed to the advanced_install
        functions. If it contains the option '--force', then the advanced_install
        function will overwrite any existing installation of the tool without 
        warnings.
        """

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

        advertisements = {'pythia-pgs':['arXiv:0603175'],
                          'Delphes':['arXiv:1307.6346'],
                          'Delphes2':['arXiv:0903.2225'],
                          'SysCalc':['arXiv:XXXX.YYYYY'],
                          'Golem95':['arXiv:0807.0605'],
                          'PJFry':['arXiv:1210.4095','arXiv:1112.0500'],
                          'QCDLoop':['arXiv:0712.1851'],
                          'pythia8':['arXiv:1410.3012'],
                          'lhapdf6':['arXiv:1412.7420'],
                          'lhapdf5':['arXiv:0605240'],
                          'hepmc':['CPC 134 (2001) 41-46'],
                          'mg5amc_py8_interface':['arXiv:1410.3012','arXiv:XXXX.YYYYY'],
                          'ninja':['arXiv:1203.0291','arXiv:1403.1229','arXiv:1604.01363'],
                          'oneloop':['arXiv:1007.4716']}

        if args[0] in advertisements:
            logger.info("------------------------------------------------------", '$MG:color:GREEN')
            logger.info("   You are installing '%s', please cite ref(s): "%args[0], '$MG:color:BLACK')
            for ad in advertisements[args[0]]:
              logger.info("                 %s"%ad, '$MG:color:GREEN')
            logger.info("   on top of the recommended MG5_aMC citations", '$MG:color:BLACK')
            logger.info("   when using results produced with this tool.", '$MG:color:BLACK')
            logger.info("------------------------------------------------------", '$MG:color:GREEN')


        # Load file with path of the different program:
        import urllib
        if paths:
            path = paths
        else:
            path = {}
    
            data_path = ['http://madgraph.phys.ucl.ac.be/package_info.dat',
                         'http://madgraph.hep.uiuc.edu/package_info.dat']
            r = random.randint(0,1)
            r = [r, (1-r)]
################################################################################
#           Force her to choose one particular server
#            r = [0]
################################################################################

            for index in r:
                cluster_path = data_path[index]
                try:
                    data = urllib.urlopen(cluster_path)
                except Exception:
                    continue
                break
            else:
                raise MadGraph5Error, '''Impossible to connect any of us servers.
                Please check your internet connection or retry later'''
    
            for line in data:
                split = line.split()
                path[split[0]] = split[1]

################################################################################
# TEMPORARY HACK WHERE WE ADD ENTRIES TO WHAT WILL BE EVENTUALLY ON THE WEB
################################################################################
#            path['XXX'] = 'YYY'
################################################################################

        if args[0] in self._advanced_install_opts:
            # Now launch the advanced installation of the tool args[0]
            # path['HEPToolsInstaller'] is the online adress where to downlaod
            # the installers if necessary.
            # Specify the path of the MG5_aMC_interface
            MG5aMC_PY8_interface_path = path['MG5aMC_PY8_interface'] if \
                                        'MG5aMC_PY8_interface' in path else 'NA'
            additional_options.append('--mg5amc_py8_interface_tarball=%s'%\
                                                      MG5aMC_PY8_interface_path)
            return self.advanced_install(args[0], path['HEPToolsInstaller'],
                                        additional_options = additional_options)

        if args[0] == 'PJFry' and not os.path.exists(
                                 pjoin(MG5DIR,'QCDLoop','lib','libqcdloop1.a')):
            logger.info("Installing PJFRY's dependence QCDLoop...")
            self.do_install('QCDLoop', paths=path)

        if args[0] == 'Delphes':
            args[0] = 'Delphes3'

        try:
            name = {'td_mac': 'td', 'td_linux':'td', 'Delphes2':'Delphes',
                'Delphes3':'Delphes', 'pythia-pgs':'pythia-pgs',
                'ExRootAnalysis': 'ExRootAnalysis','MadAnalysis':'MadAnalysis',
                'SysCalc':'SysCalc', 'Golem95': 'golem95',
                'PJFry':'PJFry','QCDLoop':'QCDLoop',
                }
            name = name[args[0]]
        except:
            pass

        #check outdated install
        if args[0] in ['Delphes2', 'pythia-pgs']:
            logger.warning("Please Note that this package is NOT maintained anymore by their author(s).\n"+\
                           "  You should consider using an up-to-date version of the code.")

        try:
            os.system('rm -rf %s' % pjoin(MG5DIR, name))
        except Exception:
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
            created_name = [n for n in os.listdir(MG5DIR) if n.lower().startswith(
                                         name.lower()) and not n.endswith('gz')]
            if not created_name:
                raise MadGraph5Error, 'The file was not loaded correctly. Stop'
            else:
                created_name = created_name[0]
            files.mv(pjoin(MG5DIR, created_name), pjoin(MG5DIR, name))


        logger.info('compile %s. This might take a while.' % name)

        # Modify Makefile for pythia-pgs on Mac 64 bit
        if args[0] == "pythia-pgs" and sys.maxsize > 2**32:
            path = os.path.join(MG5DIR, 'pythia-pgs', 'src', 'make_opts')
            text = open(path).read()
            text = text.replace('MBITS=32','MBITS=64')
            open(path, 'w').writelines(text)
            if not os.path.exists(pjoin(MG5DIR, 'pythia-pgs', 'libraries','pylib','lib')):
                os.mkdir(pjoin(MG5DIR, 'pythia-pgs', 'libraries','pylib','lib'))

        make_flags = [] #flags for the compilation        
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

            path = None
            base_compiler= ['FC=g77','FC=gfortran']
            if args[0] == "pythia-pgs":
                path = os.path.join(MG5DIR, 'pythia-pgs', 'src', 'make_opts')
            elif args[0] == 'MadAnalysis':
                path = os.path.join(MG5DIR, 'MadAnalysis', 'makefile')
            if path:
                text = open(path).read()
                for base in base_compiler:
                    text = text.replace(base,'FC=%s' % compiler)
                open(path, 'w').writelines(text)
            os.environ['FC'] = compiler
        
        # For Golem95, use autotools.
        if name == 'golem95':
            # Run the configure script
            ld_path = misc.Popen(['./configure', 
            '--prefix=%s'%str(pjoin(MG5DIR, name)),'FC=%s'%os.environ['FC']],
            cwd=pjoin(MG5DIR,'golem95'),stdout=subprocess.PIPE).communicate()[0]

        # For PJFry, use autotools.
        if name == 'PJFry':
            # Run the configure script
            ld_path = misc.Popen(['./configure', 
            '--prefix=%s'%str(pjoin(MG5DIR, name)),
            '--enable-golem-mode', '--with-integrals=qcdloop1',
            'LDFLAGS=-L%s'%str(pjoin(MG5DIR,'QCDLoop','lib')),
            'FC=%s'%os.environ['FC'],
            'F77=%s'%os.environ['FC']], cwd=pjoin(MG5DIR,name),
                                        stdout=subprocess.PIPE).communicate()[0]

        # For QCDLoop, use autotools.
        if name == 'QCDLoop':
            # Run the configure script
            ld_path = misc.Popen(['./configure', 
            '--prefix=%s'%str(pjoin(MG5DIR, name)),'FC=%s'%os.environ['FC'],
            'F77=%s'%os.environ['FC']], cwd=pjoin(MG5DIR,name),
                                        stdout=subprocess.PIPE).communicate()[0]

        # For Delphes edit the makefile to add the proper link to correct library
        if args[0] == 'Delphes3':
            #change in the makefile 
            #DELPHES_LIBS = $(shell $(RC) --libs) -lEG $(SYSLIBS)
            # to 
            #DELPHES_LIBS = $(shell $(RC) --libs) -lEG $(SYSLIBS) -Wl,-rpath,/Applications/root_v6.04.08/lib/
            rootsys = os.environ['ROOTSYS']
            text = open('./Delphes/Makefile').read()
            text = text.replace('DELPHES_LIBS = $(shell $(RC) --libs) -lEG $(SYSLIBS)', 
                         'DELPHES_LIBS = $(shell $(RC) --libs) -lEG $(SYSLIBS) -Wl,-rpath,%s/lib/' % rootsys)
            open('./Delphes/Makefile','w').write(text)
            
        # For SysCalc link to lhapdf
        if name == 'SysCalc':
            if self.options['lhapdf']:
                ld_path = misc.Popen([self.options['lhapdf'], '--libdir'],
                                     stdout=subprocess.PIPE).communicate()[0]
                ld_path = ld_path.replace('\n','')
                if 'LD_LIBRARY_PATH' not in os.environ:
                    os.environ['LD_LIBRARY_PATH'] = ld_path
                elif not os.environ['LD_LIBRARY_PATH']:
                    os.environ['LD_LIBRARY_PATH'] = ld_path
                elif ld_path not in os.environ['LD_LIBRARY_PATH']:
                    os.environ['LD_LIBRARY_PATH'] += ';%s' % ld_path
                if self.options['lhapdf'] != 'lhapdf-config':
                    if misc.which('lhapdf-config') != os.path.realpath(self.options['lhapdf']):
                        os.environ['PATH'] = '%s:%s' % (os.path.realpath(self.options['lhapdf']),os.environ['PATH']) 
            else:
                raise self.InvalidCmd('lhapdf is required to compile/use SysCalc. Specify his path or install it via install lhapdf6')
            if self.options['cpp_compiler']:
                make_flags.append('CXX=%s' % self.options['cpp_compiler'])
            

        if logger.level <= logging.INFO:
            devnull = open(os.devnull,'w')
            try:
                misc.call(['make', 'clean'], stdout=devnull, stderr=-2)
            except Exception:
                pass
            if name == 'pythia-pgs':
                #SLC6 needs to have this first (don't ask why)
                status = misc.call(['make'], cwd = pjoin(MG5DIR, name, 'libraries', 'pylib'))
            if name in ['golem95','QCDLoop','PJFry']:
                status = misc.call(['make','install'], 
                                               cwd = os.path.join(MG5DIR, name))
            else:
                status = misc.call(['make']+make_flags, cwd = os.path.join(MG5DIR, name))
        else:
            try:
                misc.compile(['clean'], mode='', cwd = os.path.join(MG5DIR, name))
            except Exception:
                pass
            if name == 'pythia-pgs':
                #SLC6 needs to have this first (don't ask why)
                status = self.compile(mode='', cwd = pjoin(MG5DIR, name, 'libraries', 'pylib'))
            if name in ['golem95','QCDLoop','PJFry']:
                status = misc.compile(['install'], mode='', 
                                          cwd = os.path.join(MG5DIR, name))
            else:
                status = self.compile(make_flags, mode='',
                                               cwd = os.path.join(MG5DIR, name))

        if not status:
            logger.info('Compilation succeeded')
        else:
            # For pythia-pgs check when removing the "-fno-second-underscore" flag
            if name == 'pythia-pgs':
                to_comment = ['libraries/PGS4/src/stdhep-dir/mcfio/arch_mcfio',
                              'libraries/PGS4/src/stdhep-dir/src/stdhep_Arch']
                for f in to_comment:
                    f = pjoin(MG5DIR, name, *f.split('/'))
                    text = "".join(l for l in open(f) if 'fno-second-underscore' not in l)
                    fsock = open(f,'w').write(text)
                try:
                    misc.compile(['clean'], mode='', cwd = os.path.join(MG5DIR, name))
                except Exception:
                    pass
                status = self.compile(mode='', cwd = os.path.join(MG5DIR, name))
            if not status:
                logger.info('Compilation succeeded')
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
                target = 'http://madgraph.phys.ucl.ac.be/Downloads/td_mac_intel.tar.gz'
                misc.call(['curl', target, '-otd.tgz'],
                                                  cwd=pjoin(MG5DIR,'td'))
                misc.call(['tar', '-xzpvf', 'td.tgz'],
                                                  cwd=pjoin(MG5DIR,'td'))
                files.mv(MG5DIR + '/td/td_mac_intel',MG5DIR+'/td/td')
            else:
                if sys.maxsize > 2**32:
                    logger.info('Downloading TD for Linux 64 bit')
                    target = 'http://madgraph.phys.ucl.ac.be/Downloads/td64/td'
                    logger.warning('''td program (needed by MadAnalysis) is not compile for 64 bit computer.
                In 99% of the case, this is perfectly fine. If you do not have plot, please follow 
                instruction in https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/TopDrawer .''')
                else:                    
                    logger.info('Downloading TD for Linux 32 bit')
                    target = 'http://madgraph.phys.ucl.ac.be/Downloads/td'
                misc.call(['wget', target], cwd=pjoin(MG5DIR,'td'))
            os.chmod(pjoin(MG5DIR,'td','td'), 0775)
            self.options['td_path'] = pjoin(MG5DIR,'td')

            if not misc.which('gs'):
                logger.warning('''gosthscript not install on your system. This is not required to run MA.
                    but this prevent to create jpg files and therefore to have the plots in the html output.''')
                if sys.platform == "darwin":
                    logger.warning('''You can download this program at the following link:
                    http://www.macupdate.com/app/mac/9980/gpl-ghostscript''')

        if args[0] == 'Delphes2':
            data = open(pjoin(MG5DIR, 'Delphes','data','DetectorCard.dat')).read()
            data = data.replace('data/', 'DELPHESDIR/data/')
            out = open(pjoin(MG5DIR, 'Template','Common', 'Cards', 'delphes_card_default.dat'), 'w')
            out.write(data)
        if args[0] == 'Delphes3':
            if os.path.exists(pjoin(MG5DIR, 'Delphes','cards')):
                card_dir = pjoin(MG5DIR, 'Delphes','cards')
            else:
                card_dir = pjoin(MG5DIR, 'Delphes','examples')
            files.cp(pjoin(card_dir,'delphes_card_CMS.tcl'),
                     pjoin(MG5DIR,'Template', 'Common', 'Cards', 'delphes_card_default.dat'))
            files.cp(pjoin(card_dir,'delphes_card_CMS.tcl'),
                     pjoin(MG5DIR,'Template', 'Common', 'Cards', 'delphes_card_CMS.dat'))
            files.cp(pjoin(card_dir,'delphes_card_ATLAS.tcl'),
                     pjoin(MG5DIR,'Template', 'Common', 'Cards', 'delphes_card_ATLAS.dat'))
            

        #reset the position of the executable
        options_name = {'Delphes': 'delphes_path',
                           'Delphes2': 'delphes_path',
                           'Delphes3': 'delphes_path',
                           'ExRootAnalysis': 'exrootanalysis_path',
                           'MadAnalysis': 'madanalysis_path',
                           'SysCalc': 'syscalc_path',
                           'pythia-pgs':'pythia-pgs_path',
                           'Golem95': 'golem',
                           'PJFry': 'pjfry'}

        if args[0] in options_name:
            opt = options_name[args[0]]
            if opt=='golem':
                self.options[opt] = pjoin(MG5DIR,name,'lib')
                self.exec_cmd('save options')
            elif opt=='pjfry':
                self.options[opt] = pjoin(MG5DIR,'PJFry','lib')
                self.exec_cmd('save options')            
            elif self.options[opt] != self.options_configuration[opt]:
                self.options[opt] = self.options_configuration[opt]
                self.exec_cmd('save options')



    def install_update(self, args, wget):
        """ check if the current version of mg5 is up-to-date.
        and allow user to install the latest version of MG5 """

        def apply_patch(filetext):
            """function to apply the patch"""
            text = filetext.read()
            pattern = re.compile(r'''=== renamed directory \'(?P<orig>[^\']*)\' => \'(?P<new>[^\']*)\'''')
            #=== renamed directory 'Template' => 'Template/LO'
            for orig, new in pattern.findall(text):
                shutil.copytree(pjoin(MG5DIR, orig), pjoin(MG5DIR, 'UPDATE_TMP'))
                full_path = os.path.dirname(pjoin(MG5DIR, new)).split('/')
                for i, name in enumerate(full_path):
                    path = os.path.sep.join(full_path[:i+1])
                    if path and not os.path.isdir(path):
                        os.mkdir(path)
                shutil.copytree(pjoin(MG5DIR, 'UPDATE_TMP'), pjoin(MG5DIR, new))
                shutil.rmtree(pjoin(MG5DIR, 'UPDATE_TMP'))
            # track rename since patch fail to apply those correctly.
            pattern = re.compile(r'''=== renamed file \'(?P<orig>[^\']*)\' => \'(?P<new>[^\']*)\'''')
            #=== renamed file 'Template/SubProcesses/addmothers.f' => 'madgraph/iolibs/template_files/addmothers.f'
            for orig, new in pattern.findall(text):
                print 'move %s to %s' % (orig, new)
                try:
                    files.cp(pjoin(MG5DIR, orig), pjoin(MG5DIR, new), error=True)
                except IOError:
                    full_path = os.path.dirname(pjoin(MG5DIR, new)).split('/')
                    for i, name in enumerate(full_path):
                        path = os.path.sep.join(full_path[:i+1])
                        if path and not os.path.isdir(path):
                            os.mkdir(path)
                files.cp(pjoin(MG5DIR, orig), pjoin(MG5DIR, new), error=True)
            # track remove/re-added file:
            pattern = re.compile(r'''^=== added file \'(?P<new>[^\']*)\'''',re.M)
            all_add = pattern.findall(text)
            #pattern = re.compile(r'''=== removed file \'(?P<new>[^\']*)\'''')
            #all_rm = pattern.findall(text)
            pattern=re.compile(r'''=== removed file \'(?P<new>[^\']*)\'(?=.*=== added file \'(?P=new)\')''',re.S)
            print 'this step can take a few minuts. please be patient'
            all_rm_add = pattern.findall(text)
            #=== added file 'tests/input_files/full_sm/interactions.dat'
            for new in all_add:
                if new in all_rm_add:
                    continue
                if os.path.isfile(pjoin(MG5DIR, new)):
                    os.remove(pjoin(MG5DIR, new))
            #pattern = re.compile(r'''=== removed file \'(?P<new>[^\']*)\'''')
            #=== removed file 'tests/input_files/full_sm/interactions.dat'
            #for old in pattern.findall(text):
            #    if not os.path.isfile(pjoin(MG5DIR, old)):
            #        full_path = os.path.dirname(pjoin(MG5DIR, old)).split('/')
            #        for i, _ in enumerate(full_path):
            #            path = os.path.sep.join(full_path[:i+1])
            #            if path and not os.path.isdir(path):
            #                os.mkdir(path)
            #        subprocess.call(['touch', pjoin(MG5DIR, old)])

            p= subprocess.Popen(['patch', '-p1'], stdin=subprocess.PIPE,
                                                              cwd=MG5DIR)
            p.communicate(text)

            # check file which are not move
            #=== modified file 'Template/LO/Cards/run_card.dat'
            #--- old/Template/Cards/run_card.dat     2012-12-06 10:01:04 +0000
            #+++ new/Template/LO/Cards/run_card.dat  2013-12-09 02:35:59 +0000
            pattern=re.compile('''=== modified file \'(?P<new>[^\']*)\'[^\n]*\n\-\-\- old/(?P<old>\S*)[^\n]*\n\+\+\+ new/(?P=new)''',re.S)
            for match in pattern.findall(text):
                new = pjoin(MG5DIR, match[0])
                old = pjoin(MG5DIR, match[1])
                if new == old:
                    continue
                elif os.path.exists(old):
                    if not os.path.exists(os.path.dirname(new)):
                        split = new.split('/')
                        for i in range(1,len(split)):
                            path = '/'.join(split[:i])
                            if not os.path.exists(path):
                                print 'mkdir', path
                                os.mkdir(path)
                    files.cp(old,new)
            #=== renamed file 'Template/bin/internal/run_delphes' => 'Template/Common/bin/internal/run_delphes'
            #--- old/Template/bin/internal/run_delphes       2011-12-09 07:28:10 +0000
            #+++ new/Template/Common/bin/internal/run_delphes        2012-10-23 02:41:37 +0000
            #pattern=re.compile('''=== renamed file \'(?P<old>[^\']*)\' => \'(?P<new>[^\']*)\'[^\n]*\n\-\-\- old/(?P=old)[^\n]*\n\+\+\+ new/(?P=new)''',re.S)
            #for match in pattern.findall(text):
            #    old = pjoin(MG5DIR, match[0])
            #    new = pjoin(MG5DIR, match[1])
            #    if new == old:
            #       continue
            #    elif os.path.exists(old):
            #        if not os.path.exists(os.path.dirname(new)):
            #            split = new.split('/')
            #            for i in range(1,len(split)):
            #                path = '/'.join(split[:i])
            #                if not os.path.exists(path):
            #                    print 'mkdir', path
            #                    os.mkdir(path)
            #        files.cp(old,new)

            # check that all files in bin directory are executable
            for path in misc.glob('*', pjoin(MG5DIR, 'bin')):
                misc.call(['chmod', '+x', path])
            for path in misc.glob(pjoin('*','bin','*'), pjoin(MG5DIR, 'Template')):
                misc.call(['chmod', '+x', path])
            for path in misc.glob(pjoin('*','bin','internal','*'), pjoin(MG5DIR, 'Template')):
                misc.call(['chmod', '+x', path])
            for path in misc.glob(pjoin('*','*', '*.py'), pjoin(MG5DIR, 'Template')):
                misc.call(['chmod', '+x', path])
            for path in misc.glob(pjoin('*','*','*.sh'), pjoin(MG5DIR, 'Template')):
                misc.call(['chmod', '+x', path])

            #add empty files/directory
            pattern=re.compile('''^=== touch (file|directory) \'(?P<new>[^\']*)\'''',re.M)
            for match in pattern.findall(text):
                if match[0] == 'file':
                    new = os.path.dirname(pjoin(MG5DIR, match[1]))
                else:
                    new = pjoin(MG5DIR, match[1])
                if not os.path.exists(new):
                    split = new.split('/')
                    for i in range(1,len(split)+1):
                        path = '/'.join(split[:i])
                        if path and not os.path.exists(path):
                            print 'mkdir', path
                            os.mkdir(path)
                if match[0] == 'file':
                    print 'touch ', pjoin(MG5DIR, match[1])
                    misc.call(['touch', pjoin(MG5DIR, match[1])])
            # add new symlink
            pattern=re.compile('''^=== link file \'(?P<new>[^\']*)\' \'(?P<old>[^\']*)\'''', re.M)
            for new, old in pattern.findall(text):
                    if not os.path.exists(pjoin(MG5DIR, new)):
                        files.ln(old, os.path.dirname(new), os.path.basename(new))

            # Re-compile CutTools and IREGI
            if os.path.isfile(pjoin(MG5DIR,'vendor','CutTools','includects','libcts.a')):
                misc.compile(cwd=pjoin(MG5DIR,'vendor','CutTools'))
            if os.path.isfile(pjoin(MG5DIR,'vendor','IREGI','src','libiregi.a')):
                misc.compile(cwd=pjoin(MG5DIR,'vendor','IREGI','src'))

            # check if it need to download binary:
            pattern = re.compile("""^Binary files old/(\S*).*and new/(\S*).*$""", re.M)
            if pattern.search(text):
                return True
            else:
                return False

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
        input_path = [arg.split('=',1)[1] for arg in args if arg.startswith('--input=')]

        if input_path:
            fsock = open(input_path[0])
            need_binary = apply_patch(fsock)
            logger.info('manual patch apply. Please test your version.')
            if need_binary:
                logger.warning('Note that some files need to be loaded separately!')
            sys.exit(0)

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
            filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/mg5amc_build_nb')
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
                except Exception:
                    print 'fail to load patch to build #%s' % (i+1)
                    fail = i
                    break
                need_binary = apply_patch(filetext)
                if need_binary:
                    path = "http://madgraph.phys.ucl.ac.be/binary/binary_file%s.tgz" %(i+1)
                    name = "extra_file%i" % (i+1)
                    if sys.platform == "darwin":
                        misc.call(['curl', path, '-o%s.tgz' % name], cwd=MG5DIR)
                    else:
                        misc.call(['wget', path, '--output-document=%s.tgz'% name], cwd=MG5DIR)
                    # Untar the file
                    returncode = misc.call(['tar', '-xzpf', '%s.tgz' % name], cwd=MG5DIR,
                                     stdout=open(os.devnull, 'w'))

            fsock = open(os.path.join(MG5DIR,'input','.autoupdate'),'w')
            if not fail:
                fsock.write("version_nb   %s\n" % web_version)
            else:
                fsock.write("version_nb   %s\n" % fail)
            fsock.write("last_check   %s\n" % int(time.time()))
            fsock.close()            
#            logger.info('Refreshing installation of MG5aMC_PY8_interface.')
#            self.do_install('mg5amc_py8_interface',additional_options=['--force'])
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
                self.set_configuration(config_path, final=False)
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
                if value.lower() == "none" or value=="":
                    self.options[name] = None

        self.options['stdout_level'] = logging.getLogger('madgraph').level
        if not final:
            return self.options # the return is usefull for unittest

        # Treat each expected input
        # 1: Pythia8_path and hewrig++ paths
        # try absolute and relative path
        for key in self.options:
            if key in ['pythia8_path', 'hwpp_path', 'thepeg_path', 'hepmc_path',
                       'mg5amc_py8_interface_path']:
                if self.options[key] in ['None', None]:
                    self.options[key] = None
                    continue
                path = self.options[key]
                #this is for pythia8
                if key == 'pythia8_path' and not os.path.isfile(pjoin(MG5DIR, path, 'include', 'Pythia8', 'Pythia.h')):
                    if not os.path.isfile(pjoin(path, 'include', 'Pythia8', 'Pythia.h')):
                        self.options['pythia8_path'] = None
                    else:
                        continue
                #this is for mg5amc_py8_interface_path
                if key == 'mg5amc_py8_interface_path' and not os.path.isfile(pjoin(MG5DIR, path, 'MG5aMC_PY8_interface')):
                    if not os.path.isfile(pjoin(path, 'MG5aMC_PY8_interface')):
                        self.options['pythia8_path'] = None
                    else:
                        continue

                #this is for hw++
                elif key == 'hwpp_path' and not os.path.isfile(pjoin(MG5DIR, path, 'include', 'Herwig++', 'Analysis', 'BasicConsistency.hh')):
                    if not os.path.isfile(pjoin(path, 'include', 'Herwig++', 'Analysis', 'BasicConsistency.hh')):
                        self.options['hwpp_path'] = None
                    else:
                        continue
                # this is for thepeg
                elif key == 'thepeg_path' and not os.path.isfile(pjoin(MG5DIR, path, 'include', 'ThePEG', 'ACDC', 'ACDCGenCell.h')):
                    if not os.path.isfile(pjoin(path, 'include', 'ThePEG', 'ACDC', 'ACDCGenCell.h')):
                        self.options['thepeg_path'] = None
                    else:
                        continue
                # this is for hepmc
                elif key == 'hepmc_path' and not os.path.isfile(pjoin(MG5DIR, path, 'include', 'HEPEVT_Wrapper.h')):
                    if not os.path.isfile(pjoin(path, 'include', 'HEPEVT_Wrapper.h')):
                        self.options['hepmc_path'] = None
                    else:
                        continue

            elif key in ['pjfry','golem','samurai']:
                if isinstance(self.options[key],str) and self.options[key].lower() == 'auto':
                    # try to find it automatically on the system                                                                                                                                            
                    program = misc.which_lib('lib%s.a'%key)
                    if program != None:
                        fpath, _ = os.path.split(program)
                        logger.info('Using %s library in %s' % (key,fpath))
                        self.options[key]=fpath
                    else:
                        # Try to look for it locally
                        local_install = {'pjfry':'PJFRY', 'golem':'golem95',
                                         'samurai':'samurai'}
                        if os.path.isfile(pjoin(MG5DIR,local_install[key],'lib', 'lib%s.a' % key)):
                            self.options[key]=pjoin(MG5DIR,local_install[key],'lib')
                        else:
                            self.options[key]=None
                    # Make sure that samurai version is recent enough
                    if key=='samurai' and \
                       isinstance(self.options[key],str) and \
                       self.options[key].lower() != 'auto':
                        if os.path.isfile(pjoin(self.options[key],os.pardir,'AUTHORS')):
                            try:
                                version = open(pjoin(self.options[key],os.pardir,
                                                          'VERSION'),'r').read()
                            except IOError:
                                version = None
                            if version is None:
                                self.options[key] = None
                                logger.info('--------')
                                logger.info(
"""The version of 'samurai' automatically detected seems too old to be compatible
with MG5aMC and it will be turned off. Ask the authors for the latest version if
you want to use samurai. 
If you want to enforce its use as-it-is, then specify directly its library folder
in the MG5aMC option 'samurai' (instead of leaving it to its default 'auto').""")
                                logger.info('--------')

            elif key.endswith('path'):
                pass
            elif key in ['run_mode', 'auto_update']:
                self.options[key] = int(self.options[key])
            elif key in ['cluster_type','automatic_html_opening']:
                pass
            elif key in ['notification_center']:
                if self.options[key] in ['False', 'True']:
                    self.allow_notification_center = eval(self.options[key])
                    self.options[key] = self.allow_notification_center
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
        
#        warnings = misc.mg5amc_py8_interface_consistency_warning(self.options)
#        if warnings:
#            logger.warning(warnings)

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
        """Main commands: Ask for editing the parameter and then
        Execute the code (madevent/standalone/...)
        """

        #ensure that MG option are not modified by the launch routine
        current_options = dict([(name, self.options[name]) for name in self.options_madgraph])
        start_cwd = os.getcwd()

        args = self.split_arg(line)
        # check argument validity and normalise argument
        (options, args) = _launch_parser.parse_args(args)
        self.check_launch(args, options)
        options = options.__dict__
        # args is now MODE PATH

        if args[0].startswith('standalone'):
            if os.path.isfile(os.path.join(os.getcwd(),args[1],'Cards',\
              'MadLoopParams.dat')) and not os.path.isfile(os.path.join(\
              os.getcwd(),args[1],'SubProcesses','check_poles.f')):
                ext_program = launch_ext.MadLoopLauncher(self, args[1], \
                                                options=self.options, **options)
            else:
                ext_program = launch_ext.SALauncher(self, args[1], \
                                                options=self.options, **options)
        elif args[0] == 'madevent':
            if options['interactive']:
                
                if isinstance(self, cmd.CmdShell):
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
                                shell = isinstance(self, cmd.CmdShell),
                                options=self.options,**options)
            else:
                # This is a width computation
                ext_program = launch_ext.MELauncher(args[1], self, unit='GeV',
                                shell = isinstance(self, cmd.CmdShell),
                                options=self.options,**options)

        elif args[0] == 'pythia8':
            ext_program = launch_ext.Pythia8Launcher( args[1], self, **options)

        elif args[0] == 'aMC@NLO':
            if options['interactive']:
                if isinstance(self, cmd.CmdShell):
                    ME = amcatnlo_run.aMCatNLOCmdShell(me_dir=args[1], options=self.options)
                else:
                    ME = amcatnlo_run.aMCatNLOCmd(me_dir=args[1],options=self.options)
                    ME.pass_in_web_mode()
                # transfer interactive configuration
                config_line = [l for l in self.history if l.strip().startswith('set')]
                for line in config_line:
                    ME.exec_cmd(line)
                stop = self.define_child_cmd_interface(ME)
                return stop
            ext_program = launch_ext.aMCatNLOLauncher( args[1], self,
                                                       shell = isinstance(self, cmd.CmdShell),
                                                        **options)
        elif args[0] == 'madweight':
            import madgraph.interface.madweight_interface as madweight_interface
            if options['interactive']:
                if isinstance(self, cmd.CmdShell):
                    MW = madweight_interface.MadWeightCmdShell(me_dir=args[1], options=self.options)
                else:
                    MW = madweight_interface.MadWeightCmd(me_dir=args[1],options=self.options)
                # transfer interactive configuration
                config_line = [l for l in self.history if l.strip().startswith('set')]
                for line in config_line:
                    MW.exec_cmd(line)
                stop = self.define_child_cmd_interface(MW)                
                return stop
            ext_program = launch_ext.MWLauncher( self, args[1],
                                                 shell = isinstance(self, cmd.CmdShell),
                                                 options=self.options,**options)            
        else:
            os.chdir(start_cwd) #ensure to go to the initial path
            raise self.InvalidCmd , '%s cannot be run from MG5 interface' % args[0]


        ext_program.run()
        os.chdir(start_cwd) #ensure to go to the initial path
        # ensure that MG options are not changed!
        for key, value in current_options.items():
            self.options[key] = value

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

        model_path = self._curr_model.get('modelpath')
        if not os.path.exists(pjoin(model_path,'build_restrict.py')):
            raise self.InvalidCmd('''Model not compatible with this option.''')

        # (re)import the full model (get rid of the default restriction)
        self._curr_model = import_ufo.import_model(model_path, restrict=False)

        #1) create the full param_card
        out_path = StringIO.StringIO()
        param_writer.ParamCardWriter(self._curr_model, out_path)
        # and load it to a python object
        param_card = check_param_card.ParamCard(out_path.getvalue().split('\n'))


        all_categories = self.ask('','0',[], ask_class=AskforCustomize)
        put_to_one = []
        ## Make a Temaplate for  the restriction card. (card with no restrict)
        for block in param_card:
            value_dict = {}
            for param in param_card[block]:
                value = param.value
                if value == 0:
                    param.value = 0.000001e-99
                elif value == 1:
                    if block != 'qnumbers':
                        put_to_one.append((block,param.lhacode))
                        param.value = random.random()
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

        # if some need to put on one
        if put_to_one:
            out_path = StringIO.StringIO()
            param_writer.ParamCardWriter(self._curr_model, out_path)
            # and load it to a python object
            param_card = check_param_card.ParamCard(out_path.getvalue().split('\n'))
            
            for (block, lhacode) in put_to_one:
                misc.sprint(block, lhacode)
                try:
                    param_card[block].get(lhacode).value = 1
                except:
                    pass # was removed of the model!
            self._curr_model.set_parameters_and_couplings(param_card)

            if args:
                name = args[0].split('=',1)[1]
                path = pjoin(model_path,'paramcard_%s.dat' % name)
                logger.info('Save default card file as %s' % path)
                param_card.write(path)

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
                if self.options_configuration[key] != self.options[key] and not self.options_configuration[key] is None:
                    to_define[key] = self.options[key]

            if not '--auto' in args:
                for key, default in self.options_madevent.items():
                    if self.options_madevent[key] != self.options[key] != None:
                        if '_path' in key and os.path.basename(self.options[key]) == 'None':
                            continue
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
    def do_set(self, line, log=True, model_reload=True):
        """Set an option, which will be default for coming generations/outputs.
        """

        # Be careful:
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
            if args[1] not in ['Auto', 'NLO']:
                self.options[args[0]] = eval(args[1])
            else:
                self.options[args[0]] = args[1]
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
        elif args[0].lower() == "ewscheme":
            logger.info("Change EW scheme to %s for the model %s. Note that YOU are responsible of the full validity of the input in that scheme." %\
                                              (self._curr_model.get('name'), args[1]))
            logger.info("Importing a model will restore the default scheme")
            self._curr_model.change_electroweak_mode(args[1])
        elif args[0] == "complex_mass_scheme":
            old = self.options[args[0]]
            self.options[args[0]] = eval(args[1])
            aloha.complex_mass = eval(args[1])
            aloha_lib.KERNEL.clean()
            if self.options[args[0]]:
                if old:
                    if log:
                        logger.info('Complex mass already activated.')
                    return
                if log:
                    logger.info('Activate complex mass scheme.')
            else:
                if not old:
                    if log:
                        logger.info('Complex mass already desactivated.')
                    return
                if log:
                    logger.info('Desactivate complex mass scheme.')
            if not self._curr_model:
                return
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
                if log: logger.info('Passing to gauge %s.' % args[1])
                return

            # They are a valid model
            able_to_mod = True
            if args[1] == 'unitary':
                if 0 in self._curr_model.get('gauge'):
                    aloha.unitary_gauge = True
                else:
                    able_to_mod = False
                    if log: logger.warning('Note that unitary gauge is not allowed for your current model %s' \
                                           % self._curr_model.get('name'))
            else:
                if 1 in self._curr_model.get('gauge'):
                    aloha.unitary_gauge = False
                else:
                    able_to_mod = False
                    if log: logger.warning('Note that Feynman gauge is not allowed for your current model %s' \
                                           % self._curr_model.get('name'))
            self.options[args[0]] = args[1]

            if able_to_mod and log and args[0] == 'gauge' and \
                args[1] == 'unitary' and not self.options['gauge']=='unitary' and \
                isinstance(self._curr_model,loop_base_objects.LoopModel) and \
                  not self._curr_model['perturbation_couplings'] in [[],['QCD']]:
                logger.warning('You will only be able to do tree level'+\
                                   ' and QCD corrections in the unitary gauge.')

            #re-init all variable
            model_name = self._curr_model.get('modelpath+restriction')
            self._curr_model = None
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            self._curr_fortran_model = None
            self._curr_cpp_model = None
            self._curr_exporter = None
            self._done_export = False
            import_ufo._import_once = []
            logger.info('Passing to gauge %s.' % args[1])

            if able_to_mod:
                # We don't want to go through the MasterCommand again
                # because it messes with the interface switching when
                # importing a loop model from MG5
                MadGraphCmd.do_import(self,'model %s' %model_name, force=True)
            elif log:
                logger.info('Note that you have to reload the model')

        elif args[0] == 'fortran_compiler':
            if args[1] != 'None':
                if log:
                    logger.info('set fortran compiler to %s' % args[1])
                self.options['fortran_compiler'] = args[1]
            else:
                self.options['fortran_compiler'] = None
        elif args[0] == 'f2py_compiler':
            if args[1] != 'None':
                if log:
                    logger.info('set f2py compiler to %s' % args[1])
                self.options['f2py_compiler'] = args[1]
            else:
                self.options['f2py_compiler'] = None
            
        elif args[0] == 'loop_optimized_output':
            if log:
                    logger.info('set loop optimized output to %s' % args[1])
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            self.options[args[0]] = args[1]
            if not self.options['loop_optimized_output'] and \
                                               self.options['loop_color_flows']:
                logger.warning("Turning off option 'loop_color_flows'"+\
                    " since it is not available for non-optimized loop output.")
                self.do_set('loop_color_flows False',log=False)
        elif args[0] == 'loop_color_flows':
            if log:
                    logger.info('set loop color flows to %s' % args[1])
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            self.options[args[0]] = args[1]
            if self.options['loop_color_flows'] and \
                                      not self.options['loop_optimized_output']:
                logger.warning("Turning on option 'loop_optimized'"+\
                                     " needed for loop color flow computation.")
                self.do_set('loop_optimized_output True',False)

        elif args[0] == 'fastjet':
            try:
                p = subprocess.Popen([args[1], '--version'], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
                output, error = p.communicate()
                res = 0
            except Exception:
                res = 1

            if res != 0 or error:
                logger.info('%s does not seem to correspond to a valid fastjet-config ' % args[1] + \
                 'executable (v3+). We will use fjcore instead.\n Please set the \'fastjet\'' + \
                 'variable to the full (absolute) /PATH/TO/fastjet-config (including fastjet-config).' +
                        '\n MG5_aMC> set fastjet /PATH/TO/fastjet-config\n')
                self.options[args[0]] = None
                self.history.pop()
            elif int(output.split('.')[0]) < 3:
                logger.warning('%s is not ' % args[1] + \
                        'v3 or greater. Please install FastJet v3+.')
                self.options[args[0]] = None
                self.history.pop()
            else: #everything is fine
                logger.info('set fastjet to %s' % args[1])
                self.options[args[0]] = args[1]

        elif args[0] in ['pjfry','golem','samurai','ninja'] and \
                           not (args[0]=='ninja' and args[1]=='./HEPTools/lib'):
            if args[1] in ['None',"''",'""']:
                self.options[args[0]] = None
            else:
                program = misc.which_lib(os.path.join(args[1],'lib%s.a'%args[0]))
                if program!=None:
                    res = 0
                    logger.info('set %s to %s' % (args[0],args[1]))
                    self.options[args[0]] = args[1]
                else:
                    res = 1
    
                if res != 0 :
                    logger.warning('%s does not seem to correspond to a valid %s lib ' % (args[1],args[0]) + \
                            '. Please enter the full PATH/TO/%s/lib .\n'%args[0] + \
                            'You will NOT be able to run %s otherwise.\n'%args[0])
                
        elif args[0] == 'lhapdf':
            try:
                res = misc.call([args[1], '--version'], stdout=subprocess.PIPE,
                                                             stderr=subprocess.PIPE)
                logger.info('set lhapdf to %s' % args[1])
                self.options[args[0]] = args[1]
            except Exception:
                res = 1
            if res != 0:
                logger.info('%s does not seem to correspond to a valid lhapdf-config ' % args[1] + \
                        'executable. \nPlease set the \'lhapdf\' variable to the (absolute) ' + \
                        '/PATH/TO/lhapdf-config (including lhapdf-config).\n' + \
                        'Note that you can still compile and run aMC@NLO with the built-in PDFs\n' + \
                        ' MG5_aMC> set lhapdf /PATH/TO/lhapdf-config\n')

        elif args[0] in ['timeout', 'auto_update', 'cluster_nb_retry',
                         'cluster_retry_wait', 'cluster_size', 'max_npoint_for_channel']:
                self.options[args[0]] = int(args[1])

        elif args[0] in ['cluster_local_path']:
            self.options[args[0]] = args[1].strip()

        elif args[0] == 'cluster_status_update':
            if '(' in args[1]:
                data = ' '.join([a for a in args[1:] if not a.startswith('-')])
                data = data.replace('(','').replace(')','').replace(',',' ').split()
                first, second = data[:2]
            else:
                first, second = args[1:3]

            self.options[args[0]] = (int(first), int(second))

        elif args[0] == 'OLP':
            # Reset the amplitudes, MatrixElements and exporter as they might
            # depend on this option
            self._curr_amps = diagram_generation.AmplitudeList()
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()
            self._curr_exporter = None
            self.options[args[0]] = args[1]

        elif args[0] =='output_dependencies':
            self.options[args[0]] = args[1]
        elif args[0] =='notification_center':
            if args[1] in ['None','True','False']:
                self.options[args[0]] = eval(args[1])
                self.allow_notification_center = self.options[args[0]]
            else:
                raise self.InvalidCmd('expected bool for notification_center')
        elif args[0] in ['cluster_queue']:
            self.options[args[0]] = args[1].strip()
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
            #MadGraph5_aMC@NLO configuration
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
        """Main commands: Initialize a new Template or reinitialize one"""

        args = self.split_arg(line)
        # Check Argument validity
        self.check_output(args)


        noclean = '-noclean' in args
        force = '-f' in args
        nojpeg = '-nojpeg' in args
        flaglist = []
                    
        if '--postpone_model' in args:
            flaglist.append('store_model')
        
        main_file_name = ""
        try:
            main_file_name = args[args.index('-name') + 1]
        except Exception:
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
        # Configuration of what to do:
        # check: check status of the directory
        # exporter: which exporter to use (v4/cpp/...)
        # output: [Template/dir/None] copy the Template, just create dir or do nothing
        config = {}
        config['madevent'] =       {'check': True,  'exporter': 'v4',  'output':'Template'}
        config['matrix'] =         {'check': False, 'exporter': 'v4',  'output':'dir'}
        config['standalone'] =     {'check': True, 'exporter': 'v4',  'output':'Template'}
        config['standalone_msF'] = {'check': False, 'exporter': 'v4',  'output':'Template'}
        config['standalone_msP'] = {'check': False, 'exporter': 'v4',  'output':'Template'}
        config['standalone_rw'] =  {'check': False, 'exporter': 'v4',  'output':'Template'}
        config['standalone_cpp'] = {'check': False, 'exporter': 'cpp', 'output': 'Template'}
        config['pythia8'] =        {'check': False, 'exporter': 'cpp', 'output':'dir'}
        config['matchbox_cpp'] =   {'check': True, 'exporter': 'cpp', 'output': 'Template'}
        config['matchbox'] =       {'check': True, 'exporter': 'v4',  'output': 'Template'}
        config['madweight'] =      {'check': True, 'exporter': 'v4',  'output':'Template'}

        options = config[self._export_format]
        # check
        if os.path.realpath(self._export_dir) == os.getcwd():
            if len(args) == 0:
                i=0
                while 1:
                    if os.path.exists('Pythia8_proc_%i' %i):
                        i+=1
                    else:
                        break
                os.mkdir('Pythia8_proc_%i' %i) 
                self._export_dir = pjoin(self._export_dir, 'Pythia8_proc_%i' %i)
                logger.info('Create output in %s' % self._export_dir)
            elif not args[0] in ['.', '-f']:
                raise self.InvalidCmd, 'Wrong path directory to create in local directory use \'.\''
        elif not noclean and os.path.isdir(self._export_dir) and options['check']:
            if not force:
                # Don't ask if user already specified force or noclean
                logger.info('INFO: directory %s already exists.' % self._export_dir)
                logger.info('If you continue this directory will be deleted and replaced.')
                answer = self.ask('Do you want to continue?', 'y', ['y','n'])
            else:
                answer = 'y'
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')
            else:
                shutil.rmtree(self._export_dir)

        # Choose here whether to group subprocesses or not, if the option was
        # set to 'Auto' and propagate this choice down the line:
        if self.options['group_subprocesses'] in [True, False]:
            group_processes = self.options['group_subprocesses']
        elif self.options['group_subprocesses'] == 'Auto':
            # By default we set it to True
            group_processes = True
            # But we turn if off for decay processes which
            # have been defined with multiparticle labels, because then
            # branching ratios necessitates to keep subprocesses independent.
            # That applies only if there is more than one subprocess of course.
            if self._curr_amps[0].get_ninitial() == 1 and \
                                                     len(self._curr_amps)>1:
                processes = [amp.get('process') for amp in self._curr_amps]            
                if len(set(proc.get('id') for proc in processes))!=len(processes):
                    # Special warning for loop-induced
                    if any(proc['perturbation_couplings'] != [] for proc in
                               processes) and self._export_format == 'madevent':
                        logger.warning("""
|| The loop-induced decay process you have specified contains several
|| subprocesses and, in order to be able to compute individual branching ratios, 
|| MG5_aMC will *not* group them. Integration channels will also be considered
|| for each diagrams and as a result integration will be inefficient.
|| It is therefore recommended to perform this simulation by setting the MG5_aMC
|| option 'group_subprocesses' to 'True' (before the output of the process).
|| Notice that when doing so, processes for which one still wishes to compute
|| branching ratios independently can be specified using the syntax:
||   -> add process <proc_def>
""")
                    group_processes = False

        #Exporter + Template
        if options['exporter'] == 'v4':
            self._curr_exporter = export_v4.ExportV4Factory(self, noclean, 
                                             group_subprocesses=group_processes)
            if options['output'] == 'Template':
                self._curr_exporter.copy_v4template(modelname=self._curr_model.get('name'))
        if  options['exporter'] == 'cpp' and options['output'] == 'Template':
            export_cpp.setup_cpp_standalone_dir(self._export_dir, self._curr_model)

        if options['output'] == 'dir' and not os.path.isdir(self._export_dir):
            os.makedirs(self._export_dir)

        # Reset _done_export, since we have new directory
        self._done_export = False

        if self._export_format == "madevent":
            # for MadEvent with MadLoop decide if we keep the box as channel of 
            #integration or not. Forbid them for matching and for h+j
            if self.options['max_npoint_for_channel']:
                base_objects.Vertex.max_n_loop_for_multichanneling = self.options['max_npoint_for_channel']
            else:
                base_objects.Vertex.max_n_loop_for_multichanneling = 3                        

        # Perform export and finalize right away
        self.export(nojpeg, main_file_name, group_processes, args)

        # Automatically run finalize
        self.finalize(nojpeg, flaglist=flaglist)

        # Remember that we have done export
        self._done_export = (self._export_dir, self._export_format)

        # Reset _export_dir, so we don't overwrite by mistake later
        self._export_dir = None

    # Export a matrix element
    def export(self, nojpeg = False, main_file_name = "", group_processes=True, 
                                                                       args=[]):
        """Export a generated amplitude to file."""

        version = [arg[10:] for arg in args if arg.startswith('--version=')]
        if version:
            version = version[-1]
        else:
            version = ''

        def generate_matrix_elements(self, group_processes=True):
            """Helper function to generate the matrix elements before
            exporting. Uses the main function argument 'group_processes' to decide 
            whether to use group_subprocess or not. (it has been set in do_output to
            the appropriate value if the MG5 option 'group_subprocesses' was set
            to 'Auto'."""

            if self._export_format in ['standalone_msP', 'standalone_msF', 'standalone_mw']:
                to_distinguish = []
                for part in self._curr_model.get('particles'):
                    if part.get('name') in args and part.get('antiname') in args and\
                       part.get('name') != part.get('antiname'):
                        to_distinguish.append(abs(part.get('pdg_code')))
            # Sort amplitudes according to number of diagrams,
            # to get most efficient multichannel output
            self._curr_amps.sort(lambda a1, a2: a2.get_number_of_diagrams() - \
                                 a1.get_number_of_diagrams())

            cpu_time1 = time.time()
            ndiags = 0
            if not self._curr_matrix_elements.get_matrix_elements():
                if group_processes:
                    cpu_time1 = time.time()
                    dc_amps = diagram_generation.DecayChainAmplitudeList(\
                        [amp for amp in self._curr_amps if isinstance(amp, \
                                        diagram_generation.DecayChainAmplitude)])
                    non_dc_amps = diagram_generation.AmplitudeList(\
                             [amp for amp in self._curr_amps if not \
                              isinstance(amp, \
                                         diagram_generation.DecayChainAmplitude)])
                    subproc_groups = group_subprocs.SubProcessGroupList()
                    matrix_elements_opts = {'optimized_output':
                                       self.options['loop_optimized_output']}
                    if non_dc_amps:
                        subproc_groups.extend(\
                          group_subprocs.SubProcessGroup.group_amplitudes(\
                          non_dc_amps, self._export_format, 
                                     matrix_elements_opts=matrix_elements_opts))

                    if dc_amps:
                        dc_subproc_group = \
                                  group_subprocs.DecayChainSubProcessGroup.\
                                  group_amplitudes(dc_amps, self._export_format,
                                      matrix_elements_opts=matrix_elements_opts)
                        subproc_groups.extend(dc_subproc_group.\
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
                    mode = {}
                    if self._export_format in [ 'standalone_msP' , 
                                             'standalone_msF', 'standalone_rw']:
                        mode['mode'] = 'MadSpin'
                    # The conditional statement tests whether we are dealing
                    # with a loop induced process.
                    if isinstance(self._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                        mode['optimized_output']=self.options['loop_optimized_output']
                        HelasMultiProcessClass = loop_helas_objects.LoopHelasProcess
                        compute_loop_nc = True
                    else:
                        HelasMultiProcessClass = helas_objects.HelasMultiProcess
                        compute_loop_nc = False
                    
                    self._curr_matrix_elements = HelasMultiProcessClass(
                      self._curr_amps, compute_loop_nc=compute_loop_nc,
                                                       matrix_element_opts=mode)
                    
                    ndiags = sum([len(me.get('diagrams')) for \
                                  me in self._curr_matrix_elements.\
                                  get_matrix_elements()])
                    # assign a unique id number to all process
                    uid = 0
                    for me in self._curr_matrix_elements.get_matrix_elements()[:]:
                        uid += 1 # update the identification number
                        me.get('processes')[0].set('uid', uid)

            cpu_time2 = time.time()


            return ndiags, cpu_time2 - cpu_time1

        # Start of the actual routine
        
        ndiags, cpu_time = generate_matrix_elements(self,group_processes)

        calls = 0

        path = self._export_dir
        if self._export_format in ['standalone_cpp', 'madevent', 'standalone',
                                   'standalone_msP', 'standalone_msF', 'standalone_rw',
                                   'matchbox_cpp', 'madweight', 'matchbox']:
            path = pjoin(path, 'SubProcesses')

        cpu_time1 = time.time()

        # First treat madevent and pythia8 exports, where we need to
        # distinguish between grouped and ungrouped subprocesses

        # MadEvent
        if self._export_format == 'madevent':
            calls += self._curr_exporter.export_processes(self._curr_matrix_elements,
                                                 self._curr_fortran_model)
            
            # Write the procdef_mg5.dat file with process info
            card_path = pjoin(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path,
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except Exception:
                    misc.sprint('command history fails.', 10)
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
                            process_number = group_number, path = path,
                            version = version)
                    process_names.append(exporter.process_name)
            else:
                exporter = export_cpp.generate_process_files_pythia8(\
                            self._curr_matrix_elements, self._curr_cpp_model,
                            process_string = self._generate_info, path = path)
                process_names.append(exporter.process_file_name)

            # Output the model parameter and ALOHA files
            model_name, model_path = exporter.convert_model_to_pythia8(\
                            self._curr_model, self._export_dir)

            # Generate the main program file
            filename, make_filename = \
                      export_cpp.generate_example_file_pythia8(path,
                                                               model_path,
                                                               process_names,
                                                               exporter,
                                                               main_file_name)

        # Pick out the matrix elements in a list
        matrix_elements = self._curr_matrix_elements.get_matrix_elements()

        # Fortran MadGraph MadWeight
        if self._export_format == 'madweight':
                        
            if isinstance(self._curr_matrix_elements, group_subprocs.SubProcessGroupList):
                #remove the merging between electron and muon
                self._curr_matrix_elements = self._curr_matrix_elements.split_lepton_grouping() 
                
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

        # Fortran MadGraph5_aMC@NLO Standalone
        if self._export_format in ['standalone', 'standalone_msP', 
                                   'standalone_msF', 'standalone_rw', 'matchbox']:
            for me in matrix_elements[:]:
                new_calls = self._curr_exporter.generate_subprocess_directory_v4(\
                            me, self._curr_fortran_model)
                if not new_calls:
                    matrix_elements.remove(me)
                calls = calls + new_calls

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
        if self._export_format in ['standalone_cpp', 'matchbox_cpp']:
            for me in matrix_elements:
                export_cpp.generate_subprocess_directory_standalone_cpp(\
                              me, self._curr_cpp_model,
                              path = path,
                              format=self._export_format)

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

    def finalize(self, nojpeg, online = False, flaglist=[]):
        """Make the html output, write proc_card_mg5.dat and create
        madevent.tar.gz for a MadEvent directory"""

        compiler_dict = {'fortran': self.options['fortran_compiler'],
                             'cpp': self.options['cpp_compiler'],
                             'f2py': self.options['f2py_compiler']}

        
        if self._export_format in ['madevent', 'standalone', 'standalone_msP', 
                                   'standalone_msF', 'standalone_rw', 'NLO', 'madweight',
                                   'matchbox']:

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
                # For a unique output of multiple type of exporter need to store this
                # information.             
                if hasattr(self, 'previous_lorentz'):
                    wanted_lorentz = list(set(self.previous_lorentz + wanted_lorentz))
                    wanted_couplings = list(set(self.previous_couplings + wanted_couplings))
                    del self.previous_lorentz
                    del self.previous_couplings
                if 'store_model' in flaglist:
                    self.previous_lorentz = wanted_lorentz
                    self.previous_couplings = wanted_couplings
                else:
                    self._curr_exporter.convert_model_to_mg4(self._curr_model,
                                               wanted_lorentz,
                                               wanted_couplings)
        if self._export_format in ['standalone_cpp', 'matchbox_cpp']:
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


        elif self._export_format in ['NLO']:
            ## write fj_lhapdf_opts file
            devnull = os.open(os.devnull, os.O_RDWR)
            try:
                res = misc.call([self.options['lhapdf'], '--version'], \
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                res = 1
            if res != 0:
                logger.info('The value for lhapdf in the current configuration does not ' + \
                        'correspond to a valid executable.\nPlease set it correctly either in ' + \
                        'input/mg5_configuration or with "set lhapdf /path/to/lhapdf-config" ' + \
                        'and regenrate the process. \nTo avoid regeneration, edit the ' + \
                        ('%s/Cards/amcatnlo_configuration.txt file.\n' % self._export_dir ) + \
                        'Note that you can still compile and run aMC@NLO with the built-in PDFs\n')



            self._curr_exporter.finalize_fks_directory( \
                                           self._curr_matrix_elements,
                                           self.history,
                                           not nojpeg,
                                           online,
                                           compiler_dict,
                                           output_dependencies = self.options['output_dependencies'],
                                           MG5DIR = MG5DIR)
            
            # Create configuration file [path to executable] for amcatnlo
            filename = os.path.join(self._export_dir, 'Cards', 'amcatnlo_configuration.txt')
            opts_to_keep = ['lhapdf', 'fastjet', 'pythia8_path', 'hwpp_path', 'thepeg_path', 
                                                                    'hepmc_path']
            to_keep = {}
            for opt in opts_to_keep:
                if self.options[opt]:
                    to_keep[opt] = self.options[opt]
            self.do_save('options %s' % filename.replace(' ', '\ '), check=False, \
                    to_keep = to_keep)

        elif self._export_format in ['madevent', 'madweight']:          
            # Create configuration file [path to executable] for madevent
            filename = os.path.join(self._export_dir, 'Cards', 'me5_configuration.txt')
            self.do_save('options %s' % filename.replace(' ', '\ '), check=False,
                         to_keep={'mg5_path':MG5DIR})

        if self._export_format in ['madevent', 'standalone', 'standalone_msP', 'standalone_msF',
                                   'standalone_rw', 'madweight', 'matchbox']:

            self._curr_exporter.finalize_v4_directory( \
                                           self._curr_matrix_elements,
                                           self.history,
                                           not nojpeg,
                                           online,
                                           compiler_dict)

        if self._export_format in ['madevent', 'standalone', 'standalone_cpp','madweight', 'matchbox']:
            logger.info('Output to directory ' + self._export_dir + ' done.')

        if self._export_format in ['madevent', 'NLO']:
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
    def do_compute_widths(self, line, model=None, do2body=True):
        """Documented commands:Generate amplitudes for decay width calculation, with fixed
           number of final particles (called level)
           syntax; compute_widths particle [other particles] [--options=]

            - particle/other particles can also be multiparticle name (can also be
           pid of the particle)

           --body_decay=X [default=4.0025] allow to choose the precision.
                if X is an integer: compute all X body decay
                if X is a float <1: compute up to the time that total error < X
                if X is a float >1: stops at the first condition.

           --path=X. Use a given file for the param_card. (default UFO built-in)

           special argument:
               - skip_2body: allow to not consider those decay (use FR)
               - model: use the model pass in argument.

        """



        self.change_principal_cmd('MadGraph')
        if '--nlo' not in line:
            warning_text = """Please note that the automatic computation of the width is
    only valid in narrow-width approximation and at tree-level."""
            logger.warning(warning_text)
            
        if not model:
            modelname = self._curr_model.get('modelpath+restriction')
            with misc.MuteLogger(['madgraph'], ['INFO']):
                model = import_ufo.import_model(modelname, decay=True)
        else:
            self._curr_model = model
            self._curr_fortran_model = \
                  helas_call_writers.FortranUFOHelasCallWriter(self._curr_model)
        if not isinstance(model, model_reader.ModelReader):
            model = model_reader.ModelReader(model)

        if '--nlo' in line:
            # call SMWidth to calculate NLO Width
            self.compute_widths_SMWidth(line, model=model)
            return

        # check the argument and return those in a dictionary format
        particles, opts = self.check_compute_widths(self.split_arg(line))

        if opts['path']:
            correct = True
            param_card = check_param_card.ParamCard(opts['path'])
            for param in param_card['decay']:
                if param.value == "auto":
                    param.value = 1
                    param.format = 'float'
                    correct = False
            if not correct:
                if opts['output']:
                    param_card.write(opts['output'])
                    opts['path'] = opts['output']
                else:
                    param_card.write(opts['path'])

        data = model.set_parameters_and_couplings(opts['path'])

        # find UFO particles linked to the require names.
        if do2body:
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
                        try:
                            value_mass = eval(str(p.mass), data)
                        except Exception:
                            # the p object can still be UFO reference. since the 
                            # mass name might hve change load back the MG5 one.
                            value_mass = eval(str(model.get_particle(p.pdg_code).get('mass')), data)
                        tmp_mass -= abs(value_mass)             
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
                    elif value < 0.1 and particle['color'] !=1:
                        logger.warning("partial width of particle %s lower than QCD scale:%s. Set it to zero. (%s)" \
                                   % (particle.get('name'), value, decay_to))
                        value = 0
                                     
                    decay_info[particle.get('pdg_code')].append([decay_to, value])
                    total += value
            else:
                madevent_interface.MadEventCmd.update_width_in_param_card(decay_info,
                                                       opts['path'], opts['output'])
                if float(opts['body_decay']) == 2:
                    return
        else:
            skip_2body = True

        #
        # add info from decay module
        #
        self.do_decay_diagram('%s %s' % (' '.join([`id` for id in particles]),
                                         ' '.join('--%s=%s' % (key,value)
                                                  for key,value in opts.items()
                                                  if key not in ['precision_channel'])
                                         ), skip_2body=skip_2body)

        if self._curr_amps:
            logger.info('Pass to numerical integration for computing the widths:')
        else:            
            logger.info('No need for N body-decay (N>2). Results are in %s' % opts['output'])
            
            
            
            return

        # Do the MadEvent integration!!
        with misc.TMP_directory(dir=os.getcwd()) as path:
            decay_dir = pjoin(path,'temp_decay')
            logger_mg.info('More info in temporary files:\n    %s/index.html' % (decay_dir))
            with misc.MuteLogger(['madgraph','ALOHA','cmdprint','madevent'], [40,40,40,40]):
                self.exec_cmd('output %s -f' % decay_dir)
                # Need to write the correct param_card in the correct place !!!
                if os.path.exists(opts['output']):
                    files.cp(opts['output'], pjoin(decay_dir, 'Cards', 'param_card.dat'))
                else:
                    files.cp(opts['path'], pjoin(decay_dir, 'Cards', 'param_card.dat'))
                if self._curr_model['name'] == 'mssm' or self._curr_model['name'].startswith('mssm-'):
                    check_param_card.convert_to_slha1(pjoin(decay_dir, 'Cards', 'param_card.dat'))
                # call a ME interface and define as it as child for correct error handling
                me_cmd = madevent_interface.MadEventCmd(decay_dir)
                #self.define_child_cmd_interface(me_cmd, interface=False)
                me_cmd.model_name = self._curr_model['name'] #needed for mssm
                me_cmd.options['automatic_html_opening'] = False

                me_opts=[('accuracy', opts['precision_channel']), # default 0.01
                         ('points', 1000),
                         ('iterations',9)]
                me_cmd.exec_cmd('survey decay -f %s' % (
                       " ".join(['--%s=%s' % val for val in me_opts])),
                      postcmd=False)
                me_cmd.exec_cmd('combine_events', postcmd=False)
                #me_cmd.exec_cmd('store_events', postcmd=False)
                me_cmd.collect_decay_widths()
                me_cmd.do_quit('')
                # cleaning
                del me_cmd

            param = check_param_card.ParamCard(pjoin(decay_dir, 'Events', 'decay','param_card.dat'))

        for pid in particles:
            width = param['decay'].get((pid,)).value
            particle = self._curr_model.get_particle(pid) 
            #if particle['color'] !=1 and 0 < width.real < 0.1:
            #    logger.warning("width of colored particle \"%s(%s)\" lower than QCD scale: %s. Set width to zero "
            #                   % (particle.get('name'), pid, width.real))
            #    width = 0
                
            
            if not pid in param['decay'].decay_table:
                continue
            if pid not in decay_info:
                decay_info[pid] = []
            for BR in param['decay'].decay_table[pid]:
                if len(BR.lhacode) == 3 and skip_2body:
                    continue
                if BR.value * width <0.1 and particle['color'] !=1:
                    logger.warning("partial width of particle %s lower than QCD scale:%s. Set it to zero. (%s)" \
                                   % (particle.get('name'), BR.value * width, BR.lhacode[1:]))
                                     
                    continue
                
                decay_info[pid].append([BR.lhacode[1:], BR.value * width])

        madevent_interface.MadEventCmd.update_width_in_param_card(decay_info,
                                                   opts['path'], opts['output'])

        if self._curr_model['name'] == 'mssm' or self._curr_model['name'].startswith('mssm-'):
            check_param_card.convert_to_slha1(opts['output'])
        return



    # Calculate decay width with SMWidth
    def compute_widths_SMWidth(self, line, model=None):
        """Compute widths with SMWidth.
        """

        # check the argument and return those in a dictionary format
        particles, opts = self.check_compute_widths(self.split_arg(line))

        if opts['path']:
            correct = True
            param_card = check_param_card.ParamCard(opts['path'])
            for param in param_card['decay']:
                if param.value == "auto":
                    param.value = 1
                    param.format = 'float'
                    correct = False
            if not correct:
                if opts['output']:
                    param_card.write(opts['output'])
                    opts['path'] = opts['output']
                else:
                    param_card.write(opts['path'])

        if not model:
            model_path = self._curr_model.get('modelpath')
            model_name = self._curr_model.get('name')
            currmodel = self._curr_model
        else:
            model_path = model.get('modelpath')
            model_name = model.get('name')
            currmodel = model

        if not os.path.exists(pjoin(model_path, 'SMWidth')):
            raise self.InvalidCmd, "Model %s is not valid for computing NLO width with SMWidth"%model_name

        # determine the EW scheme
        externparam = [(param.lhablock.lower(),param.name.lower()) for param \
                           in currmodel.get('parameters')[('external',)]]

        if ('sminputs','aewm1') in externparam:
            # alpha(MZ) scheme
            arg2 = "1"
        elif ('sminputs','mdl_gf') in externparam or ('sminputs','gf') in externparam:
            # Gmu scheme
            arg2 = "2"
        else:
            raise Exception, "Do not know the EW scheme in the model %s"%model_name

        #compile the code
        if not os.path.exists(pjoin(model_path, 'SMWidth','smwidth')):
            logger.info('Compiling SMWidth. This has to be done only once and'+\
                            ' can take a couple of minutes.','$MG:color:BLACK')
            current = misc.detect_current_compiler(pjoin(model_path, 'SMWidth',
                                                         'makefile_MW5'))
            new = 'gfortran' if self.options_configuration['fortran_compiler'] is None else \
                self.options_configuration['fortran_compiler']
            if current != new:
                misc.mod_compilator(pjoin(model_path, 'SMWidth'), new, current)
                misc.mod_compilator(pjoin(model_path, 'SMWidth','oneloop'), new, current)
                misc.mod_compilator(pjoin(model_path, 'SMWidth','hdecay'), new, current)
            misc.compile(cwd=pjoin(model_path, 'SMWidth'))

        # look for the ident_card.dat
        identpath=" "
        carddir=os.path.dirname(opts['path'])
        if 'ident_card.dat' in os.listdir(carddir):
            identpath=pjoin(carddir,'ident_card.dat')
        #run the code
        output,error = misc.Popen(['./smwidth',opts['path'],identpath,arg2],
                                  stdout=subprocess.PIPE,
                                  stdin=subprocess.PIPE,
                                  cwd=pjoin(model_path, 'SMWidth')).communicate()
        pattern = re.compile(r'''  decay\s+(\+?\-?\d+)\s+(\+?\-?\d+\.\d+E\+?\-?\d+)''',re.I)
        width_list = pattern.findall(output)
        width_dict = {}
        for pid,width in width_list:
            width_dict[int(pid)] = float(width)

        for pid in particles:
            if not pid in width_dict:
                width = 0
            else:
                width = width_dict[pid]
            param = param_card['decay'].get((pid,))
            param.value = width
            param.format = 'float'
            if pid not in param_card['decay'].decay_table:
                continue
            del param_card['decay'].decay_table[pid] # reset the BR
        # write the output file. (the new param_card)
        if opts['output']:
            param_card.write(opts['output'])
            logger.info('Results are written in %s' % opts['output'])
        else:
            param_card.write(opts['path'])
            logger.info('Results are written in %s' % opts['path'])
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
        particles, args = self.check_decay_diagram(args)
        #print args
        pids = particles
        level = float(args['body_decay'])
        param_card_path = args['path']
        min_br = float(args['min_br'])

        # Reset amplitudes
        self._curr_amps = diagram_generation.AmplitudeList()
        # Reset Helas matrix elements
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        # Reset _done_export, since we have new process
        self._done_export = False
        # Also reset _export_format and _export_dir
        self._export_format = None


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
                self._curr_decaymodel.find_channels(part, level, min_br)
                if not skip_2body:
                    amp = part.get_amplitudes(2)
                    if amp:
                        self._curr_amps.extend(amp)

                for l in range(3, level+1):
                    amp = part.get_amplitudes(l)
                    if amp:
                        self._curr_amps.extend(amp)
            else:
                max_level = level // 1
                if max_level < 2:
                    max_level = 999
                precision = level % 1
                if first:
                    model.find_all_channels(2,generate_abstract=False)
                    first = False
                if not skip_2body:
                    amp = part.get_amplitudes(2)
                    if amp:
                        self._curr_amps.extend(amp)
                clevel = 2
                while part.get('apx_decaywidth_err').real > precision:
                    clevel += 1
                    if clevel > max_level:
                        logger_mg.info('    stop to %s body-decay. approximate error: %s' %
                                   (max_level, part.get('apx_decaywidth_err')) )
                        break
                    if clevel > 3:
                        logger_mg.info('    current estimated error: %s go to %s-body decay:' %\
                                        (part.get('apx_decaywidth_err'), clevel))
                    part.find_channels_nextlevel(model, min_br)
                    #part.group_channels_2_amplitudes(clevel, model, min_br)
                    amp = part.get_amplitudes(clevel)
                    if amp:
                        self._curr_amps.extend(amp)
                    part.update_decay_attributes(False, True, True, model)


        # Set _generate_info
        if len(self._curr_amps) > 0:
            process = self._curr_amps[0]['process'].nice_string()
            #print process
            self._generate_info = process[9:]
            #print self._generate_info
        else:
            logger.info("No decay is found")

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
_draw_parser = misc.OptionParser(usage=_draw_usage)
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
         "-- execute the madevent/standalone/standalone_cpp/pythia8/NLO output present in DIRPATH\n" + \
         "   By default DIRPATH is the latest created directory \n" + \
         "   (for pythia8, it should be the Pythia 8 main directory) \n" + \
         "   Example: launch PROC_sm_1 --name=run2 \n" + \
         "   Example: launch ../pythia8 \n"
_launch_parser = misc.OptionParser(usage=_launch_usage)
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
_launch_parser.add_option("-R", "--reweight", default=False, action='store_true',
                            help="Run the reweight module (reweighting by different model parameter")
_launch_parser.add_option("-M", "--madspin", default=False, action='store_true',
                            help="Run the madspin package")

#===============================================================================
# Interface for customize question.
#===============================================================================
class AskforCustomize(cmd.SmartQuestion):
    """A class for asking a question where in addition you can have the
    set command define and modifying the param_card/run_card correctly"""

    def __init__(self, question, allow_arg=[], default=None,
                                            mother_interface=None, *arg, **opt):

        model_path = mother_interface._curr_model.get('modelpath')
        #2) Import the option available in the model
        ufo_model = ufomodels.load_model(model_path)
        self.all_categories = ufo_model.build_restrict.all_categories

        question = self.get_question()
        # determine the possible value and how they are linked to the restriction
        #options.
        allow_arg = ['0']
        self.name2options = {}
        for category in self.all_categories:
            for options in category:
                if not options.first:
                    continue
                self.name2options[str(len(allow_arg))] = options
                self.name2options[options.name.replace(' ','')] = options
                allow_arg.append(len(allow_arg))
        allow_arg.append('done')

        cmd.SmartQuestion.__init__(self, question, allow_arg, default, mother_interface)



    def default(self, line):
        """Default action if line is not recognized"""

        line = line.strip()
        args = line.split()
        if line == '' and self.default_value is not None:
            self.value = self.default_value
        # check if input is a file
        elif hasattr(self, 'do_%s' % args[0]):
            self.do_set(' '.join(args[1:]))
        elif line.strip() != '0' and line.strip() != 'done' and \
            str(line) != 'EOF' and line.strip() in self.allow_arg:
            option = self.name2options[line.strip()]
            option.status = not option.status
            self.value = 'repeat'
        else:
            self.value = line

        return self.all_categories

    def reask(self, reprint_opt=True):
        """ """
        reprint_opt = True
        self.question = self.get_question()
        cmd.SmartQuestion.reask(self, reprint_opt)

    def do_set(self, line):
        """ """
        self.value = 'repeat'

        args = line.split()
        if args[0] not in self.name2options:
            logger.warning('Invalid set command. %s not recognize options. Valid options are: \n  %s' %
                           (args[0], ', '.join(self.name2options.keys()) ))
            return
        elif len(args) != 2:
            logger.warning('Invalid set command. Not correct number of argument')
            return


        if args[1] in ['True','1','.true.','T',1,True,'true','TRUE']:
            self.name2options[args[0]].status = True
        elif args[1] in ['False','0','.false.','F',0,False,'false','FALSE']:
            self.name2options[args[0]].status = False
        else:
            logger.warning('%s is not True/False. Didn\'t do anything.' % args[1])



    def get_question(self):
        """define the current question."""
        question = ''
        i=0
        for category in self.all_categories:
            question += category.name + ':\n'
            for options in category:
                if not options.first:
                    continue
                i+=1
                question += '    %s: %s [%s]\n' % (i, options.name,
                                options.display(options.status))
            question += 'Enter a number to change it\'s status or press enter to validate.\n'
            question += 'For scripting this function, please type: \'help\''
        return question


    def complete_set(self, text, line, begidx, endidx):
        """ Complete the set command"""
        signal.alarm(0) # avoid timer if any
        args = self.split_arg(line[0:begidx])

        if len(args) == 1:
            possibilities = [x for x in self.name2options if not x.isdigit()]
            return self.list_completion(text, possibilities, line)
        else:
            return self.list_completion(text,['True', 'False'], line)


    def do_help(self, line):
        '''help message'''

        print 'This allows you to optimize your model to your needs.'
        print 'Enter the number associate to the possible restriction/add-on'
        print ' to change the status of this restriction/add-on.'
        print ''
        print 'In order to allow scripting of this function you can use the '
        print 'function \'set\'. This function takes two argument:'
        print 'set NAME VALUE'
        print '   NAME is the description of the option where you remove all spaces'
        print '   VALUE is either True or False'
        print ' Example: For the question'
        print '''     sm customization:
        1: diagonal ckm [True]
        2: c mass = 0 [True]
        3: b mass = 0 [False]
        4: tau mass = 0 [False]
        5: muon mass = 0 [True]
        6: electron mass = 0 [True]
    Enter a number to change it's status or press enter to validate.'''
        print ''' you can answer by'''
        print '   set diagonalckm False'
        print '   set taumass=0 True'

    def cmdloop(self, intro=None):
        cmd.SmartQuestion.cmdloop(self, intro)
        return self.all_categories



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
