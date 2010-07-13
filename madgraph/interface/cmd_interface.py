###############################################################################
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
import pydoc
import re
import subprocess
import sys
import traceback
import time

# Optional Library (not present on all platform)
try:
    import readline
except:
    readline = None
    



import madgraph.iolibs.misc as misc
import madgraph.iolibs.files as files

import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.import_ufo as import_ufo
#import madgraph.iolibs.save_model as save_model
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.export_pythia8 as export_pythia8
import madgraph.iolibs.convert_ufo2mg4 as ufo2mg4
import madgraph.iolibs.file_writers as writers

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw

import madgraph.interface.tutorial_text as tutorial_text

from madgraph import MG4DIR, MG5DIR, MadGraph5Error

import models as ufomodels
import aloha.create_helas as create_helas

# Special logger for the Cmd Interface
logger = logging.getLogger('cmdprint') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr
logger_tuto = logging.getLogger('tutorial') # -> stdout include instruction in  
                                            #order to learn MG5

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
    """Extension of the cmd.Cmd command line.
    This extensions supports line breaking, history, comments,
    internal call to cmdline,..."""

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
        '#*     automaticaly generated the %(time)s%(fill)s*\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        self.log = True
        self.history = []
        self.save_line = ''
        cmd.Cmd.__init__(self, *arg, **opt)
        self.__initpos = os.path.abspath(os.getcwd())
        
    def precmd(self, line):
        """ A suite of additional function needed for in the cmd
        this implement history, line breaking, comment treatment,...
        """
        
        if not line:
            return line

        # Update the history of this suite of command,
        # except for useless commands (empty history and help calls)
        if line != "history" and \
            not line.startswith('help') and \
            not line.startswith('#*') and \
            not line.startswith('now'):
            self.history.append(line)

        # Check if we are continuing a line:
        if self.save_line:
            line = self.save_line + line 
            self.save_line = ''
        
        # Check if the line is complete
        if line.endswith('\\'):
            self.save_line = line[:-1]
            return '' # do nothing   
        
        # Remove comment
        if '#' in line:
            line = line.split('#')[0]

        # execute the line command
        return line
    
    def onecmd(self, line):
        """catch all error and stop properly command accordingly"""
        
        try:
            cmd.Cmd.onecmd(self, line)
        except MadGraph5Error as error:
            # Make sure that we are at the initial position
            os.chdir(self.__initpos)
            if str(error):
                if line == self.history[-1]:
                    error_text = 'Command \"%s\" interrupted with error:\n' % line
                else:
                    error_text = 'Command \"%s\" interrupted in sub-command:\n' %line
                    error_text += '\"%s\" with error:\n' % self.history[-1] 
                error_text += '%s : %s' % (error.__class__.__name__, str(error).replace('\n','\n\t'))
                logger_stderr.error(error_text)
                #stop the execution if on a non interactive mode
                if self.use_rawinput == False:
                    sys.exit()
            return False
        except Exception as error:
            # Make sure that we are at the initial position
            os.chdir(self.__initpos)
            # Create the debug files
            self.log = False
            cmd.Cmd.onecmd(self, 'history MG5_debug')
            debug_file = open('MG5_debug', 'a')
            traceback.print_exc(file=debug_file)
            # Create a nice error output
            if line == self.history[-1]:
                error_text = 'Command \"%s\" interrupted with error:\n' % line
            else:
                error_text = 'Command \"%s\" interrupted in sub-command:\n' %line
                error_text += '\"%s\" with error:\n' % self.history[-1]
            error_text += '%s : %s\n' % (error.__class__.__name__, str(error).replace('\n','\n\t'))
            error_text += 'Please report this bug on https://bugs.launchpad.net/madgraph5\n'
            error_text += 'More information is found in \'%s\'.\n' % \
                          os.path.realpath("MG5_debug")
            error_text += 'Please attach this file to your report.'
            logger_stderr.critical(error_text)
            #stop the execution if on a non interactive mode
            if self.use_rawinput == False:
                sys.exit('Exit on error')
            return False

    def exec_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment"""
        
        logger.info(line)
        line = self.precmd(line)
        stop = cmd.Cmd.onecmd(self, line)
        stop = self.postcmd(stop, line)
        return stop      

    def run_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment"""
        
        logger.info(line)
        line = self.precmd(line)
        stop = self.onecmd(line)
        stop = self.postcmd(stop, line)
        return stop 
    
    def postcmd(self,stop, line):
        """ finishing a command
        This looks if we have to write an additional text for the tutorial."""
        
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
            command = args[0]+'_'+args[1]
        
        try:
            logger_tuto.info(getattr(tutorial_text, command).replace('\n','\n\t'))
        except:
            try:
                logger_tuto.info(getattr(tutorial_text, args[0]).replace('\n','\n\t'))
            except:
                pass
            


    def emptyline(self):
        """If empty line, do nothing. Default is repeat previous command."""
        pass
    
    def default(self, line):
        """Default action if line is not recognized"""

        # Faulty command
        logger.warning("Command \"%s\" not recognized, please try again" % \
                                                                line.split()[0])
    # Quit
    def do_quit(self, line):
        sys.exit(1)
        
    do_exit = do_quit

    # Aliases
    do_EOF = do_quit
    do_exit = do_quit

#===============================================================================
# Helper function
#=============================================================================
def split_arg(line):
    """Split a line of arguments"""
    
    split = line.split()
    out=[]
    tmp=''
    for data in split:
        if data[-1] == '\\':
            tmp += data[:-1]+' '
        elif tmp:
            out.append(tmp+data)
        else:
            out.append(data)
    return out

 

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

    def help_tutorial(self):
        print "syntax: tutorial [" + "|".join(self._tutorial_opts) + "]"
        print "-- start/stop the tutorial mode"

    def help_setup(self):
        print "syntax " + "|".join(self._setup_opts) + \
              " name|.|auto [options]"
        print "-- Create a copy of the V4 Template in the MG_ME directory"
        print "   with the model and Helas set up appropriately."
        print "   If standalone_v4 is chosen, the directory will be in"
        print "   Standalone format."
        print "   name is the name of the copy of Template."
        print "   If you put '.' instead of a name, the code will try to locate"
        print "     a valid copy of the Template under focus."
        print "   If you put 'auto' instead an automatic name will be created."
        print "   If you have generated a process, the process will "
        print "     automatically be exported to the directory."
        print "   options:"
        print "      -f: force the cleaning of the directory if this one exist"
        print "      -d PATH: specify the directory where to create name"
        print "      -noclean: no cleaning perform in name"
        print "      -nojpeg: no jpeg diagrams will be generated"
        print "   Example:"
        print "       setup madevent_v4 MYRUN"
        print "       setup madevent_v4 MYRUN -d ../MG_ME -f"
        
    def help_generate(self):

        print "syntax: generate INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2"
        print "-- generate diagrams for a given process"
        print "   Syntax example: l+ vl > w+ > l+ vl a $ z / a h QED=3 QCD=0 @1"
        print "Decay chain syntax:"
        print "   core process, decay1, (decay2, (decay2', ...)), ...  etc"
        print "   Example: p p > t~ t QED=0 @2, (t~ > W- b~, W- > l- vl~), t > j j b"
        print "   Note that identical particles will all be decayed."
        print "To generate a second process use the \"add process\" command"

    def help_add(self):

        print "syntax: add process INITIAL STATE > REQ S-CHANNEL > FINAL STATE $ EXCL S-CHANNEL / FORBIDDEN PARTICLES COUP1=ORDER1 COUP2=ORDER2"
        print "-- generate diagrams for a process and add to existing processes"
        print "   Syntax example: l+ vl > w+ > l+ vl a $ z / a h QED=3 QCD=0 @1"
        print "Decay chain syntax:"
        print "   core process, decay1, (decay2, (decay2', ...)), ...  etc"
        print "   Example: p p > t~ t QED=0 @2, (t~ > W- b~, W- > l- vl~), t > j j b"
        print "   Note that identical particles will all be decayed."

    def help_define(self):
        print "syntax: define multipart_name [ part_name_list ]"
        print "-- define a multiparticle"
        print "   Example: define p u u~ c c~ d d~ s s~"

    def help_export(self):
        print "syntax: export [" + "|".join(self._export_formats) + \
              " FILEPATH] [options]"
        print """-- export matrix elements.
        *Note* that if you have run the 'setup', export format and FILEPATH
        is optional.
        - For madevent_v4, the path needs to be to a MadEvent SubProcesses
        directory, and the result is the Pxxx directories (including the
        diagram .ps and .jpg files) for the subprocesses as well as a
        correctly generated subproc.mg file.
        - For standalone_v4, the result is a set of complete MG4 Standalone
        process directories.
        - For matrix_v4, the resulting files will be
        FILEPATH/matrix_\"process_string\".f
        - options available are \"-nojpeg\", to suppress generation of
        jpeg diagrams in MadEvent 4 subprocess directories.
        - For pythia8, the resulting files will be
        FILEPATH/Sigma_\"process_string\".h and
        FILEPATH/Sigma_\"process_string\".cc"""

    def help_history(self):
        print "syntax: history [FILEPATH|clean|.] "
        print "   If FILEPATH is \'.\' and \'setup\' is done,"
        print "   Cards/proc_card_mg5.dat will be used."
        print "   If FILEPATH is omitted, the history will be output to stdout."
        print "   clean option will remove all entries from the history."

    def help_finalize(self):
        print "syntax: finalize [" + "|".join(self._setup_opts) + \
              " PATH] [-nojpeg]"
        print "-- finalize MadEvent or Standalone directory in PATH. "
        print "   For MadEvent, create web pages, jpeg diagrams,"
        print "   proc_card_mg5 and madevent.tar.gz files."
        print "   For Standalone, just generate proc_card_mg5."
        print "   By default, PATH and madevent_v4/standalone_v4 are "
        print "   defined by the setup command."
        print "   Add option -nojpeg to suppress jpeg diagrams."
        print "   This command is automatically run by \"export\"."

    def help_demo(self):
        """ demo help"""
        
        print "this command starts a simple demonstration on how use MG5"
        print " In order to stop this demonstration you can enter "
        print " mg5> demo stop"

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
    
    class InvalidCmd(MadGraph5Error):
        """a class for the invalid syntax call"""
    
    class RWError(MadGraph5Error):
        """a class for read/write errors"""
    
    def check_add(self, args):
        """check the validity of line
        syntax: add process PROCESS 
        """
        
        if len(self._curr_model['particles']) == 0:
            raise self.InvalidCmd("No particle list currently active, " + \
                                              "please create one first!")

        if len(self._curr_model['interactions']) == 0:
            raise self.InvalidCmd("No interaction list currently active," + \
            " please create one first!")

        if len(args) < 2:
            self.help_add()
            raise self.InvalidCmd('\"add\" requires two arguments')
        
        if args[0] != 'process':
            raise self.InvalidCmd('\"add\" requires the argument \"process\"')
    
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
        
        if len(self._curr_model['particles']) == 0:
            raise self.InvalidCmd("No particle list currently active, please import a model first")

        if self._curr_model['particles'].find_name(args[0]):
            raise MadGraph5Error("label %s is a particle name in this model\n\
            Please retry with another name." % args[0])

    def check_display(self, args):
        """check the validity of line
        syntax: display XXXXX
        """
            
        if len(args) != 1 or args[0] not in self._display_opts:
            self.help_display()
            raise self.InvalidCmd

        if not self._curr_model['particles'] or not self._curr_model['interactions']:
            raise self.InvalidCmd("No model currently active, please import a model!")

        if args[0] in ['processes', 'diagrams'] and not self._curr_amps:
            raise self.InvalidCmd("No process generated, please generate a process!")

    def check_draw(self, args):
        """check the validity of line
        syntax: draw DIRPATH [option=value]
        """
        
        if len(args) < 1:
            self.help_draw()
            raise self.InvalidCmd('\"draw\" command requires a directory path')
        
        if not self._curr_amps:
            raise self.InvalidCmd("No process generated, please generate a process!")
            

        if not os.path.isdir(args[0]):
            raise self.InvalidCmd( "%s is not a valid directory for export file" % args[1])
            
    def check_export(self, args):
        """check the validity of line
        syntax: export MODE FILEPATH
        """  

        nojpeg = ""
        if '-nojpeg' in args:
            nojpeg = '-nojpeg'
            args = filter(lambda arg: arg != nojpeg, args)
    
        if len(args) == 0:
            if not self._export_format:
                self.help_export()
                raise self.InvalidCmd('\"export\" require at least a format of output')
        elif(len(args) == 1 and not self._export_dir):
            self.help_export()
            raise self.InvalidCmd(\
        'No output position defined (either explicitely or via a setup command)')
        elif(args[0] not in self._export_formats):
            self.help_export()
            raise self.InvalidCmd('%s is a valid export format.' % args[0])

        if not self._curr_amps:
            raise self.InvalidCmd("No process generated, please generate a process!")

        if len(args) <= 1:
            path = self._export_dir
        else:
            path = args[1]

        if not os.path.isdir(path):
            text = "%s is not a valid directory for export file\n" % path
            if args[0] == 'madevent_v4':
                text += " to create a valid output directory you can use the command\n"
                text += " $> setup madevent_v4 name|.|auto\n"
                text += " and then run export as follow:\n"
                text += " $> export madevent_v4\n" 
            raise self.InvalidCmd(text)
        

        if args and args[0] == ('madevent_v4' or 'standalone_v4') and \
               not os.path.isdir(os.path.join(path,'SubProcesses')):
            text = "%s is not a valid directory for export file" % path
            text += "to create a valid output directory you can use the command"
            text += "$> setup madevent_v4 auto" 
            text += " and then run export as follow:"
            text += "$> export madevent_v4" 
            raise self.InvalidCmd(text)

        if nojpeg:
            args.append(nojpeg)
    
        return path
    
    def check_generate(self, line):
        """check the validity of line"""
        
        if len(line) < 1:
            self.help_generate()
            raise self.InvalidCmd("\"generate\" requires an argument.")
            

        if not self._curr_model['particles'] or not self._curr_model['interactions']:
            raise self.InvalidCmd("No model currently active, please import a model!")

        return True
    
    def check_history(self, args):
        """check the validity of line"""
        
        if len(args) > 1:
            self.help_history()
            raise self.InvalidCmd('\"history\" command requires at most one argument')
        
        if not len(args):
            return
        
        if args[0] =='.':
            if not self._export_dir:
                raise self.InvalidCmd("No default directory are yet define for \'.\' option")
        elif args[0] != 'clean':
                dirpath = os.path.dirname(args[0])
                if dirpath and not os.path.exists(dirpath) or \
                       os.path.isdir(args[0]):
                    raise self.InvalidCmd("invalid path %s " % dirpath)
    
    def check_import(self, args):
        """check the validity of line"""
        
        if not args or  args[0] not in self._import_formats:
            self.help_import()
            raise self.InvalidCmd('wrong \"import\" format')
        
        if args[0] != 'proc_v4' and len(args) != 2:
            self.help_import()
            raise self.InvalidCmd(' incorrect number of arguments')
        
        if args[0] == 'proc_v4' and len(args) != 2 and not self._export_dir:
            self.help_import()
            raise self.InvalidCmd('PATH is mandatory in the current context\n' + \
                                  'You maybe forget to run \"setup\" command')            
        
    def check_load(self, args):
        """ check the validity of the line"""
        
        if len(args) != 2 or args[0] not in self._save_opts:
            self.help_load()
            raise self.InvalidCmd('wrong \"load\" format')
            
        
    def check_finalize(self, args):
        """check the validity of the line"""

        nojpeg = ""
        if '-nojpeg' in args:
            nojpeg = '-nojpeg'
            args = filter(lambda arg: arg != nojpeg, args)
    
        if len(args) < 1 and not self._export_format in self._setup_opts:
            self.help_finalize()
            raise self.InvalidCmd('wrong \"finalize\" format')
        
        elif args and args not in self._setup_opts:
            self.help_finalize()
            raise self.InvalidCmd('%s is not recognized as a valid option' % args[0])
        
        if (len(args) <= 1 and not self._export_dir) or \
                        (len(args) > 1 and not os.path.isdir(args[1])):
            self.help_finalize()
            raise self.InvalidCmd('no valid directory path output.')

        if nojpeg:
            args.append(nojpeg)
    
    def check_save(self, args):
        """ check the validity of the line"""
        if len(args) != 2 or args[0] not in self._save_opts:
            self.help_save()
            raise self.InvalidCmd('wrong \"save\" format')
    
    def check_setup(self, args):
        """ check the validity of the line"""
        
        nojpeg = ""
        if '-nojpeg' in args:
            nojpeg = '-nojpeg'
            args = filter(lambda arg: arg != nojpeg, args)
    
        if len(args) < 2 or args[0] not in self._setup_opts:
            self.help_setup()
            raise self.InvalidCmd('wrong \"setup\" format')
        
        if args[1] == '.' and not self._export_dir:
            self.help_setup()
            text = "\".\" options is not available in the current situation\n" + \
                    "Do a  \"setup\" first" 
            raise self.InvalidCmd(text)
        
        if not self._model_dir:
            text = 'No model found. Please import a model first and then retry\n'
            text += 'for example do : import model_v4 sm'
            raise self.InvalidCmd(text)

        if nojpeg:
            args.append(nojpeg)

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
      
    def check_export(self, args):
        """check the validity of line
        syntax: export MODE FILEPATH
        No FilePath authorized on the web
        """  


        if len(args) > 0:
            raise self.WebRestriction('Path can\'t be specify on the web.' + \
                                      'use the setup command to avoid the ' + \
                                      'need to specify a path')
  
        return CheckValidForCmd.check_export(self, args)
    
    def check_history(self, args):
        """check the validity of line
        No Path authorize for the Web"""
        
        CheckValidForCmd.check_history(self, args)

        if len(args) == 2 and args[1] not in ['.', 'clean']:
            raise self.WebRestriction('Path can\'t be specify on the web.')

        
    def check_import(self, args):
        """check the validity of line
        No Path authorize for the Web"""
        
        CheckValidForCmd.check_import(self, args)
        
        if len(args) >= 2 and args[0] == 'proc_v4' and args[1] != '.':
            raise self.WebRestriction('Path can\'t be specify on the web.')

        if len(args) >= 2 and args[0] == 'command':
            if args[1] != './Cards/proc_card_mg5.dat': 
                raise self.WebRestriction('Path can\'t be specify on the web.')
        else:
            for arg in args:
                if '/' in arg:
                    raise self.WebRestriction('Path can\'t be specify on the web.')
        
    def check_finalize(self, args):
        """check the validity of the line
        No Path authorize for the Web"""
    
        CheckValidForCmd.check_finalize(self, args)
        
        if len(args) > 1:
            raise self.WebRestriction('Path can\'t be specify on the web.')
        
        
        
    def check_load(self, args):
        """ check the validity of the line
        No Path authorize for the Web"""

        CheckValidForCmd.check_load(self, args)        

        if len(args) == 2:
            if args[0] != 'model':
                raise self.WebRestriction('only model can be loaded online')
            if 'model.pkl' not in args[1]:
                raise self.WebRestriction('not valid pkl file: wrong name')
            if not os.path.realpath(args[1]).startswith(os.path.join(MG4DIR, \
                                                                    'Models')):
                raise self.WebRestriction('Wrong path to load model')
        
    def check_save(self, args):
        """ not authorize on web"""
        raise self.WebRestriction('\"save\" command not authorize online')
    
    def check_setup(self, args):
        """ check the validity of the line"""
        
        CheckValidForCmd.check_setup(self, args)
        
        if '/' in args[2]:
            raise self.WebRestriction('Path can\'t be specify on the web.')
    
#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(CheckValidForCmd):
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
    
            
    def complete_finalize(self, text, line, begidx, endidx):
        """ format: finalize madevent_v4|standalone_v4 [PATH] [-nojpeg]"""
        
        # Format
        if text.startswith('-'):
            return self.list_completion(text, ['-nojpeg'])

        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, ['madevent_v4'])
        
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
        
    def complete_tutorial(self, text, line, begidx, endidx):
        "Complete the tutorial command"

        # Format
        if len(split_arg(line[0:begidx])) == 1:
            return self.list_completion(text, self._tutorial_opts)

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
            opt = ['horizontal', 'external=', 'max_size=', 'add_gap=',
                                'non_propagating', '--']
            return self.list_completion(text, opt)

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
        possible_format = self._setup_opts
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
            content = [name for name in os.listdir(MG4DIR) if \
                                    name not in forbidden_name and \
                                    os.path.isdir(os.path.join(MG4DIR, name))]
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

    def complete_shell(self, text, line, begidx, endidx):
        """ add path for shell """

        # Filename if directory is given
        #
        if len(split_arg(line[0:begidx])) > 1 and line[begidx - 1] == os.path.sep:
            if not text:
                text = ''
            output = self.path_completion(text,
                                        base_dir=\
                                          split_arg(line[0:begidx])[-1])
        else:
            output = self.path_completion(text)
        return output

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
class MadGraphCmd(CmdExtended, HelpToCmd):
    """The command line processor of MadGraph"""    
    
    _curr_model = base_objects.Model()
    _curr_amps = diagram_generation.AmplitudeList()
    _curr_matrix_elements = helas_objects.HelasMultiProcess()
    _curr_fortran_model = export_v4.HelasFortranModel()
    _curr_cpp_model = export_pythia8.UFOHelasCPPModel()

    _display_opts = ['particles', 'interactions', 'processes', 'diagrams', 
                     'multiparticles', 'couplings']
    _add_opts = ['process']
    _save_opts = ['model', 'processes']
    _setup_opts = ['madevent_v4', 'standalone_v4']
    _tutorial_opts = ['start', 'stop']
    _import_formats = ['model_v4', 'model', 'proc_v4', 'command']
    _export_formats = ['madevent_v4', 'standalone_v4', 'matrix_v4', 'pythia8']
    _done_export = False
    _done_finalize = False
        
    def __init__(self, *arg, **opt):
        """ add a tracker of the history """

        CmdExtended.__init__(self, *arg, **opt)
        self._generate_info = "" # store the first generated process
        self._model_dir = None
        self._model_format = None
        self._multiparticles = {}
        self._ufo_model = None
        
        # Detect If this script is launched from a valid copy of the Template
        #and if so store this position as standard output directory
        if 'TemplateVersion.txt' in os.listdir('.'):
            #Check for ./
            self._export_dir = os.path.realpath('.')
        elif 'TemplateVersion.txt' in os.listdir('..'):
            #Check for ../
            self._export_dir = os.path.realpath('..')
        elif self.stdin != sys.stdin:
            #Check for position defined by the input files
            input_path = os.path.realpath(self.stdin.name).split(os.path.sep)
            if input_path[-2] == 'Cards':
                self._export_dir = os.path.sep.join(input_path[:-2])
            else:
                self._export_dir = None
        else:
            self._export_dir = None
        self._export_format = None
        
    # Add a process to the existing multiprocess definition
    # Generate a new amplitude
    def do_add(self, line):
        """Generate an amplitude for a given process and add to
        existing amplitudes
        syntax:
        """

        args = split_arg(line)
        
        # Check the validity of the arguments
        self.check_add(args)

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
            except MadGraph5Error, error:
                raise MadGraph5Error("Empty or wrong format process :\n" + \
                                     str(error))
                
            if myprocdef:

                cpu_time1 = time.time()

                myproc = diagram_generation.MultiProcess(myprocdef)

                for amp in myproc.get('amplitudes'):
                    if amp not in self._curr_amps:
                        self._curr_amps.append(amp)
                    else:
                        warning = "Warning: Already in processes:\n%s" % \
                                                    amp.nice_string_processes()
                        logger.warning(warning)

                # Reset _done_export and _done_finalize, since we have
                # new process
                self._done_export = False
                self._done_finalize = False

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

        # Particle names always lowercase
        line = line.lower()
        args = split_arg(line)
        # check the validity of the arguments
        self.check_define(args)

        label = args[0]
        
        pdg_list = self.extract_particle_ids(args[1:])
        self._multiparticles[label] = pdg_list
        if log:
            logger.info("Defined multiparticle %s" % \
                                             self.multiparticle_string(label))
    
    # Display
    def do_display(self, line):
        """Display current internal status"""

        args = split_arg(line)
        #check the validity of the arguments
        self.check_display(args)

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

        elif args[0] == 'interactions':
            text = "Current model contains %i interactions\n" % \
                    len(self._curr_model['interactions'])
            for inter in self._curr_model['interactions']:
                text += str(inter['id']) + ':'
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

        elif args[0] == 'processes':
            for amp in self._curr_amps:
                print amp.nice_string_processes()

        elif args[0] == 'diagrams':
            for amp in self._curr_amps:
                print amp.nice_string()

        elif args[0] == 'multiparticles':
            print 'Multiparticle labels:'
            for key in self._multiparticles:
                print self.multiparticle_string(key)
        
        elif args[0] == 'couplings':
            couplings = set()
            for interaction in self._curr_model['interactions']:
                for order in interaction['orders'].keys():
                    couplings.add(order)
            print ' / '.join(couplings)
            
    def multiparticle_string(self, key):
        """Returns a nicely formatted string for the multiparticle"""

        return "%s = %s" % (key, " ".join( \
                    [ self._curr_model.get('particle_dict')[part_id].\
                      get_name() for part_id in self._multiparticles[key]]))

    def do_tutorial(self, line):
        """Activate/deactivate the tutorial mode."""

        args = split_arg(line)
        if len(args) > 0 and args[0] == "stop":
            logger_tuto.setLevel(logging.ERROR)
        else:
            logger_tuto.setLevel(logging.INFO)

        if not MG4DIR:
            logger_tuto.info(\
        "  Warning: This tutorial should preferably be run " + \
        "from a valid MG_ME directory.")

        
    
    def do_draw(self, line):
        """ draw the Feynman diagram for the given process """

        args = split_arg(line)
        # Check the validity of the arguments
        self.check_draw(args)

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

            logger.info("Drawing " + \
                         amp.get('process').nice_string())
            #plot.draw(opt=options)
            plot.draw()
            logger.info("Wrote file " + filename)

        stop = time.time()
        logger.info('time to draw %s' % (stop - start)) 
    
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
        # Check the validity of the arguments and return the output path
        path = self.check_export(args)

        if self._done_export == path:
            # We have already done export in this path
            logger.info("Matrix elements already exported")
            return        

        ndiags, cpu_time = generate_matrix_elements(self)
        calls = 0

        nojpeg = '-nojpeg' in args
        args = filter(lambda arg: arg != '-nojpeg', args)

        if not args:
            args = [self._export_format]

        if args[0] in ['standalone_v4', 'madevent_v4']:
            self._export_format = args[0]
            self._export_dir = path
            path = os.path.join(path, 'SubProcesses')

        if args[0] == 'matrix_v4':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                filename = os.path.join(path, 'matrix_' + \
                           me.get('processes')[0].shell_string() + ".f")
                if os.path.isfile(filename):
                    logger.warning("Overwriting existing file %s" % filename)
                else:
                    logger.info("Creating new file %s" % filename)
                calls = calls + export_v4.write_matrix_element_v4_standalone(\
                    writers.FortranWriter(filename),\
                    me, self._curr_fortran_model)
                
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
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except:
                    pass
                
        if args[0] == 'standalone_v4':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                calls = calls + \
                        export_v4.generate_subprocess_directory_v4_standalone(\
                            me, self._curr_fortran_model, path)
            
        if args[0] == 'pythia8':
            for me in self._curr_matrix_elements.get('matrix_elements'):
                export_pythia8.generate_process_files_pythia8(\
                            me, self._curr_cpp_model, path)
                
        self._export_format = args[0]


        logger.info(("Generated helas calls for %d subprocesses " + \
              "(%d diagrams) in %0.3f s") % \
              (len(self._curr_matrix_elements.get('matrix_elements')),
               ndiags, cpu_time))

        if calls:
            logger.info("Wrote %d helas calls" % calls)

        # Replace the amplitudes with the actual amplitudes from the
        # matrix elements, which allows proper diagram drawing also of
        # decay chain processes
        self._curr_amps = diagram_generation.AmplitudeList(\
               [me.get('base_amplitude') for me in \
                self._curr_matrix_elements.get('matrix_elements')])

        # Remember that we have done export
        self._done_export = self._export_dir
        # Reset _done_finalize, since we now have new subprocesses
        self._done_finalize = False

        if self._export_format:
            # Automatically run finalize
            options = ''
            if nojpeg:
                options = '-nojpeg'
                
            self.do_finalize(options)
    
    # Generate a new amplitude
    def do_generate(self, line):
        """Generate an amplitude for a given process"""

        # Check line validity
        self.check_generate(line)

        # Reset Helas matrix elements
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        self._generate_info = line

        try:
            if ',' not in line:
                myprocdef = self.extract_process(line)
            else:
                myprocdef, line = self.extract_decay_chain_process(line)
        except MadGraph5Error as error:
            raise MadGraph5Error(str(error))
        
        # Check that we have something    
        if not myprocdef:
            raise MadGraph5Error("Empty or wrong format process, please try again.")

        # run the program
        cpu_time1 = time.time()
        myproc = diagram_generation.MultiProcess(myprocdef)
        self._curr_amps = myproc.get('amplitudes')
        cpu_time2 = time.time()

        # Reset _done_export and _done_finalize, since we have new process
        self._done_export = False
        self._done_finalize = False

        nprocs = len(self._curr_amps)
        ndiags = sum([amp.get_number_of_diagrams() for \
                              amp in self._curr_amps])
        logger.info("%i processes with %i diagrams generated in %0.3f s" % \
                  (nprocs, ndiags, (cpu_time2 - cpu_time1)))
    
    
    def extract_process(self, line, proc_number=0):
        """Extract a process definition from a string. Returns
        a ProcessDefinition."""

        # Check basic validity of the line
        if not line.count('>') in [1,2]:
            raise self.InvalidCmd('Wrong use of \">\" special character.')
        

        # Perform sanity modifications on the lines:
        # Add a space before any > , $ /
        space_before = re.compile(r"(?P<carac>\S)(?P<tag>[/\,\\$\\>])")
        line = space_before.sub(r'\g<carac> \g<tag>', line)       
        # Add a space after any + - ~ > , $ / 
        space_after = re.compile(r"(?P<tag>[+-/\,\\$\\>~])(?P<carac>[^\s+-])")
        line = space_after.sub(r'\g<tag> \g<carac>', line)
        
        
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
                    raise MadGraph5Error, "No final state particles"
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
                raise MadGraph5Error, \
                      "No particle %s in model" % part_name

        if filter(lambda leg: leg.get('state') == True, myleglist):
            # We have a valid process

            # Now extract restrictions
            forbidden_particle_ids = \
                               self.extract_particle_ids(forbidden_particles)
            forbidden_schannel_ids = \
                               self.extract_particle_ids(forbidden_schannels)
            required_schannel_ids = \
                               self.extract_particle_ids(required_schannels)

            #decay_process = len(filter(lambda leg: \
            #                           leg.get('state') == False,
            #                           myleglist)) == 1


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

    def extract_particle_ids(self, args):
        """Extract particle ids from a list of particle names"""

        if isinstance(args, basestring):
            args = split_arg(args)
        ids=[]
        for part_name in args:
            mypart = self._curr_model['particles'].find_name(part_name)
            if mypart:
                ids.append(mypart.get_pdg_code())
            elif part_name in self._multiparticles:
                ids += self._multiparticles[part_name]
            else:
                raise MadGraph5Error("No particle %s in model" % part_name)
        # Trick to avoid duplication
        set_dict = {}
        return [set_dict.setdefault(i,i) for i in ids if i not in set_dict]

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
            logger.info(line)
            
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
                    raise MadGraph5Error, \
                      "Missing ending parenthesis for decay process"

                if index_par < index_comma:
                    line = line[index_par + 1:]
                    level_down = False
                    break

        if level_down:
            index_par = line.find(')')
            if index_par == -1:
                raise MadGraph5Error, \
                      "Missing ending parenthesis for decay process"
            line = line[index_par + 1:]
            
        # Return the core process (ends recursion when there are no
        # more decays)
        return core_process, line
    
    # Write the list of command line use in this session
    def do_history(self, line):
        """write in a file the suite of command that was used"""
        
        args = split_arg(line)
        # Check arguments validity
        self.check_history(args)

        if len(args) == 0:
            print '\n'.join(self.history)
            return
        elif args[0] == 'clean':
            self.history = []
            logger.info('History is cleaned')
            return
        elif args[0] == '.':
            output_file = os.path.join(self._export_dir, 'Cards', \
                                                            'proc_card_mg5.dat')
            output_file = open(output_file, 'w')
        else:
            output_file = open(args[0], 'w')
            
        # Create the command file
        text = self.history_header % misc.get_time_info()
        text += ('\n'.join(self.history) + '\n') 
        
        #write this information in a file
        output_file.write(text)
        output_file.close()

        if self.log:
            logger.info("History written to " + output_file.name)
    
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
                logger.info("%d particles imported" % \
                      len(self._curr_model['particles']))
                      
            elif filename.endswith('interactions.dat'):
                if len(self._curr_model['particles']) == 0:
                    text =  "No particle list currently active,"
                    text += "please create one first!"
                    raise MadGraph5Error(text)
                self._curr_model.set('interactions',
                                     files.read_from_file(
                                            filepath,
                                            import_v4.read_interactions_v4,
                                            self._curr_model['particles']))
                logger.info("%d interactions imported" % \
                      len(self._curr_model['interactions']))
           
            else:
                #not valid File
                raise MadGraph5Error("%s is not a valid v4 file name" % \
                                     filepath)

        args = split_arg(line)
        # Check argument's validity
        self.check_import(args)
        
        
        if args[0] == 'model':
            self._ufo_model = ufomodels.load_model(args[1])
            ufo2mg5_converter = import_ufo.converter_ufo_mg5(self._ufo_model)
            self._curr_model = ufo2mg5_converter.load_model()
            if os.path.isdir(args[1]):
                self._model_dir = args[1]
            elif os.path.isdir(os.path.join(MG5DIR, 'models', args[1])):
                self._model_dir = os.path.join(MG5DIR, 'models', args[1])
            else:
                raise self.InvalidCmd('Invalid model path/name')
            self._curr_fortran_model = export_v4.UFOHelasFortranModel()
                    
        elif args[0] == 'model_v4':
            # Check for a file
            if os.path.isfile(args[1]):
                import_v4file(self, args[1])
                self._model_dir = os.path.dirname(args[1])
                return
            
            # Check for a valid directory
            elif os.path.isdir(args[1]):
                self._model_dir = args[1]
            elif MG4DIR and os.path.isdir(os.path.join(MG4DIR, 'Models', \
                                                                      args[1])):
                self._model_dir = os.path.join(MG4DIR, 'Models', args[1])
            elif not MG4DIR:
                error_text = "Path %s is not a valid pathname\n" % args[1]
                error_text += "and no MG_ME installation detected in order to search in Models"
                raise MadGraph5Error(error_text)
            else:
                raise MadGraph5Error("Path %s is not a valid pathname" % args[1])
            
            #Load the directory
            if os.path.exists(os.path.join(self._model_dir, 'model.pkl')):
                self.do_load('model %s' % os.path.join(self._model_dir, \
                                                                   'model.pkl'))
                self.add_default_multiparticles()
                return
            files_to_import = ('particles.dat', 'interactions.dat')
            for filename in files_to_import:
                if os.path.isfile(os.path.join(self._model_dir, filename)):
                    import_v4file(self, os.path.join(self._model_dir, \
                                                                      filename))
                else:
                    raise self.RWError("%s file doesn't exist in %s directory" % \
                                        (filename, os.path.basename(args[1])))
            #save model for next usage
            save_load_object.save_to_file(
                                      os.path.join(self._model_dir, 'model.pkl')
                                    , self._curr_model)
            self.add_default_multiparticles()
        
        elif args[0] == 'proc_v4':
            
            if len(args) == 1 and self._export_dir:
                proc_card = os.path.join(self._export_dir, 'Cards', \
                                                                'proc_card.dat')
            elif len(args) == 2:
                proc_card = args[1]
                # Check the status of export and try to use file position is no
                #self._export dir are define
                if os.path.isdir(args[1]):
                    proc_card = os.path.join(proc_card, 'Cards', \
                                                                'proc_card.dat')    
                self.check_for_export_dir(os.path.realpath(proc_card))
            else:
                raise MadGraph5Error('No default directory are setup')

 
            #convert and excecute the card
            self.import_mg4_proc_card(proc_card)   
                                     
        elif args[0] == 'command':
            if not os.path.isfile(args[1]):
                raise MadGraph5Error("Path %s is not a valid pathname" % args[1])
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
        if not reader:
            raise MadGraph5Error('\"%s\" is not a valid path' % filepath)
        
        if MG4DIR:
            # Add comment to history
            self.exec_cmd("# Import the model %s" % reader.model)
            #model_dir = os.path.join(MG4DIR, 'Models')
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
        # remove this call from history
        self.history.pop()
        
        # Read the lines of the file and execute them
        for line in CmdFile(filepath):
            #remove pointless spaces and \n
            line = line.replace('\n', '').strip()
            # execute the line
            if line:
                self.exec_cmd(line)

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
        for line in open(os.path.join(MG5DIR, 'input', \
                                      'multiparticles_default.txt')):
            if line.startswith('#'):
                continue
            try:
                multipart_name = line.lower().split()[0]
                if multipart_name not in self._multiparticles:
                    self.do_define(line)
                    
            except MadGraph5Error, why:
                logger_stderr.warning('impossible to set default multiparticles %s because %s' %
                                        (line.split()[0],why))
        if defined_multiparticles:
            logger.info("Kept definitions of multiparticles %s unchanged" % \
                                         " / ".join(defined_multiparticles))

        for removed_part in removed_multiparticles:
            if removed_part in self._multiparticles:
                removed_multiparticles.remove(removed_part)

        if removed_multiparticles:
            logger.info("Removed obsolete multiparticles %s" % \
                                         " / ".join(removed_multiparticles))
            
    def check_for_export_dir(self, filepath):
        """Check if the files is in a valid export directory and assign it to
        export path if if is"""
        
        # keep previous if a previous one is defined
        if self._export_dir:
            return
        
        path_split = filepath.split(os.path.sep)
        if len(path_split) > 2 and path_split[-2] == 'Cards':
            self._export_dir = os.path.sep.join(path_split[:-2])
                
    

    def do_load(self, line):
        """Load information from file"""

        args = split_arg(line)
        # check argument validity
        self.check_load(args)

        cpu_time1 = time.time()
        if args[0] == 'model':
            self._curr_model = save_load_object.load_from_file(args[1])
            #save_model.save_model(args[1], self._curr_model)
            if isinstance(self._curr_model, base_objects.Model):
                cpu_time2 = time.time()
                logger.info("Loaded model from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1))
            else:
                raise self.RWError('Could not load model from file %s' \
                                      % args[1])
        elif args[0] == 'processes':
            self._curr_amps = save_load_object.load_from_file(args[1])
            if isinstance(self._curr_amps, diagram_generation.AmplitudeList):
                cpu_time2 = time.time()
                logger.info("Loaded processes from file in %0.3f s" % \
                      (cpu_time2 - cpu_time1))
                if self._curr_amps and not self._curr_model.get('name'):
                    self._curr_model = self._curr_amps[0].\
                                        get('process').get('model')
                    logger.info("Model set from process.")
            else:
                raise self.RWError('Could not load processes from file %s' % args[1])
    
    def do_save(self, line):
        """Save information to file"""

        args = split_arg(line)
        # Check argument validity
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
            
    def do_finalize(self, line):
        """Make the html output, write proc_card_mg5.dat and create
        madevent.tar.gz for a MadEvent directory"""
        
        args = split_arg(line)
        # Check argument validity
        self.check_finalize(args)

        if self._export_dir:
            dir_path = self._export_dir
        else: 
            dir_path = args[1]
            
        #look if the user ask to bypass the jpeg creation
        if '-nojpeg' in args:
            makejpg = False
        else:
            makejpg = True

        if self._done_finalize == (dir_path, makejpg):
            # We have already done export in this path
            logger.info("Process directory already finalized")
            return        

        if self._export_format and self._export_format == 'madevent_v4' or \
               not self._export_format and args[0] == 'madevent_v4':

            os.system('touch %s/done' % os.path.join(dir_path,'SubProcesses'))        
            export_v4.finalize_madevent_v4_directory(dir_path, makejpg,
                                                     [self.history_header] + \
                                                     self.history)
        elif self._export_format and self._export_format == 'standalone_v4' \
                 or not self._export_format and args[0] == 'standalone_v4':

            export_v4.finalize_standalone_v4_directory(dir_path,
                                                     [self.history_header] + \
                                                     self.history)

        logger.info('Directory ' + dir_path + ' finalized')
        self._done_finalize = (dir_path, makejpg)

    def do_setup(self, line):
        """Initialize a new Template or reinitialize one"""
        
        args = split_arg(line)
        # Check Argument validity
        self.check_setup(args)
        
        noclean = '-noclean' in args
        force = '-f' in args 
        nojpeg = '-nojpeg' in args
    
        dir = '-d' in args
        if dir:
            mgme_dir = args[args.find('-d') + 1]
        elif MG4DIR:
            mgme_dir = MG4DIR
        else:
            raise MadGraph5Error('No installation of  MG_ME (version 4) detected')
                                
        # Check for special directory treatment
        if args[1] == '.':
                dir_path = self._export_dir
        elif args[1] == 'auto':
            if args[0] == 'madevent_v4':
                name_dir = lambda i: 'PROC_%s_%s' % \
                                        (os.path.split(self._model_dir)[-1], i)
                auto_path = lambda i: os.path.join(self.writing_dir,
                                                   name_dir(i))
            elif args[0] == 'standalone_v4':
                name_dir = lambda i: 'PROC_SA_%s_%s' % \
                                        (os.path.split(self._model_dir)[-1], i)
                auto_path = lambda i: os.path.join(self.writing_dir,
                                                   name_dir(i))                
            for i in range(500):
                if os.path.isdir(auto_path(i)):
                    continue
                else:
                    dir_path = auto_path(i) 
                    break
        else:    
            dir_path = os.path.join(self.writing_dir, args[1])
        
        if not force and not noclean and os.path.isdir(dir_path):
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % dir_path)
            logger.info('If you continue this directory will be cleaned')
            answer = raw_input('Do you want to continue? [y/n]')
            if answer != 'y':
                raise MadGraph5Error('Stopped by user request')

        if args[0] == 'madevent_v4':
            export_v4.copy_v4template(mgme_dir, dir_path,
                                      self._model_dir, not noclean)
            self._export_format = 'madevent_v4'
        if args[0] == 'standalone_v4':
            export_v4.copy_v4standalone(mgme_dir, dir_path,
                                        self._model_dir, not noclean)
            self._export_format = 'standalone_v4'

        # Import the model/HELAS
        if not self._ufo_model:
            logger.info('import v4model files %s in directory %s' % \
                       (os.path.basename(self._model_dir), args[1]))        
            export_v4.export_model_files(self._model_dir, dir_path)
            export_v4.export_helas(os.path.join(mgme_dir,'HELAS'), dir_path)
        else:
            logger.info('convert UFO model to MG4 format')
            ufo2mg4.export_to_mg4(self._ufo_model, 
                                        os.path.join(dir_path,'Source','MODEL'))
            create_helas.AbstractHelasModel(os.path.basename(self._model_dir),
                            write_dir=os.path.join(dir_path,'Source','DHELAS'))
            export_v4.make_model_symbolic_link(self._model_dir, dir_path)
        
        if args[0] == 'standalone_v4':
            export_v4.make_v4standalone(dir_path)

        self._export_dir = dir_path

        # Reset _done_export and _done_finalize, since we have new directory
        self._done_export = False
        self._done_finalize = False        

        if self._curr_amps:
            # Perform export and finalize right away
            options = ''
            if nojpeg:
                options = '-nojpeg'

            self.do_export(options)


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
        
        possibility = {
        'mg5_start': ['import model_v4 PATH', 'import command PATH', 
                                                 'import proc_v4 PATH', 'tutorial'],
        'import model_v4': ['generate PROCESS','define MULTIPART PART1 PART2 ...', 
                                   'display particles', 'display interactions'],
        'import model' : ['generate PROCESS','define MULTIPART PART1 PART2 ...', 
                                   'display particles', 'display interactions'],
        'define': ['define MULTIPART PART1 PART2 ...', 'generate PROCESS', 
                                                    'display multiparticles'],
        'generate': ['add process PROCESS','setup OUTPUT_TYPE PATH','draw .'],
        'add process':['setup OUTPUT_TYPE PATH', 'display processes'],
        'setup':['history PATH', 'exit'],
        'display': ['generate PROCESS', 'add process PROCESS', 'setup OUTPUT_TYPE PATH'],
        'draw': ['shell CMD'],
        'export':['finalize'],
        'finalize': ['history PATH', 'exit'],
        'import proc_v4' : ['exit'],
        'tutorial': ['import model_v4 sm','help']
        }
        
        print 'Contextual Help'
        print '==============='
        if last_action_2 in possibility.keys():
            options = possibility[last_action_2]
        elif last_action in possibility.keys():
            options = possibility[last_action]
        else:
            print 'No suggestion available for your last command'
            return
        
        text = 'The following command maybe usefull in order to continue.\n'
        for option in options:
            text+='\t %s \n' % option
        #text+='you can use help to have more information on those command'
        
        print text
        





#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmdWeb(MadGraphCmd, CheckValidForCmdWeb):
    """The command line processor of MadGraph"""
 
    def __init__(self, *arg, **opt):
    
        if os.environ.has_key('_CONDOR_SCRATCH_DIR'):
            self.writing_dir = os.path.join(os.environ['_CONDOR_SCRATCH_DIR'], \
                                                                 os.path.pardir)
        else:
            self.writing_dir = os.path.join(os.environ['MADGRAPH_DATA'],
                               os.environ['REMOTE_USER'])
        
        #standard initialization
        MadGraphCmd.__init__(self, *arg, **opt)

#===============================================================================
# MadGraphCmd
#===============================================================================
class MadGraphCmdShell(MadGraphCmd, CompleteForCmd, CheckValidForCmd):
    """The command line processor of MadGraph""" 
    
    writing_dir = MG4DIR
    
    def preloop(self):
        """Initializing before starting the main loop"""

        self.prompt = 'mg5>'
        
        if readline:
            readline.parse_and_bind("tab: complete")

        # initialize command history if HOME exists
        if os.environ.has_key('HOME') and readline:
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
        "*           Type 'tutorial' to learn how MG5 works         *\n" + \
        "*                                                          *\n" + \
        "************************************************************"

    # Access to shell
    def do_shell(self, line):
        "Run a shell command"

        if line.strip() is '':
            self.help_shell()
        else:
            logging.info("running shell command: " + line)
            subprocess.call(line, shell=True)

#===============================================================================
# 
#===============================================================================
class CmdFile(file):
    """ a class for command input file -in order to debug cmd \n problem"""
    
    def __init__(self, name, opt='rU'):
        
        file.__init__(self, name, opt)
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
    
    def __next__(self):
        return self.lines.__next__()    
    def __iter__(self):
        return self.lines.__iter__()
  
#===============================================================================
# Draw Command Parser
#=============================================================================== 
_usage = "draw FILEPATH [options]\n" + \
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
                          help="avoid contractions of non propagating lines") 
_draw_parser.add_option("", "--add_gap", default=0, type='float', \
                          help="set the x-distance between external particles")  
  
    
    
    
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
