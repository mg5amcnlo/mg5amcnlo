#! /usr/bin/env python
################################################################################
#
# Copyright (c) 2011 The MadGraph Development team and Contributors
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
""" This is the main script in order to generate events in MadEvent """

import sys
import os
import subprocess 
import logging
import logging.config
import re
import shutil
import subprocess
import time

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]

if not sys.version_info[0] == 2 or sys.version_info[1] < 6:
    sys.exit('MadEvent works with python 2.6 or higher (but not python 3.X).\n\
               Please upgrade your version of python.')

# Check if optimize mode is (and should be) activated
if __debug__ and (not os.path.exists(os.path.join(root_path,'../..', 'bin','create_release.py'))):
    print 'launch in debug mode'
    subprocess.call([sys.executable] + ['-O'] + sys.argv)
    sys.exit()




pjoin = os.path.join
sys.path.append(pjoin(root_path,'bin','internal'))
import madweight_interface as MW        

try: 
    import readline
except ImportError:
    try:
        import pyreadline as readline
    except:
        print "For tab completion and history, install module readline."
else:
    import rlcompleter

    if 'r261:67515' in sys.version and  'GCC 4.2.1 (Apple Inc. build 5646)' in sys.version:
        readline.parse_and_bind("bind ^I rl_complete")
        readline.__doc__ = 'libedit'  
    
    elif hasattr(readline, '__doc__'):
        if 'libedit' not in readline.__doc__:
            readline.parse_and_bind("tab: complete")
        else:
            readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.__doc__ = 'GNU'
        readline.parse_and_bind("tab: complete")
        
    # charge history file
    try:
        history_file = os.path.join(os.environ['HOME'], '.mg5', 'mw5history')
        readline.read_history_file(history_file)
    except:
        pass

try:
   import psyco
   psyco.full()
except:
   pass

if __debug__:
        print 'Running MG5 in debug mode'


def set_configuration():
    import coloring_logging
    logging.config.fileConfig(os.path.join(root_path, 'bin', 'internal', 'me5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madevent').setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)    


def treat_old_argument(argument):
    """Have the MW4 behavior for this script"""

    import internal.misc as misc
    import internal.madweight.MW_info as MW_info
    with misc.chdir(root_path):
        MWparam=MW_info.MW_info('MadWeight_card.dat')
        MWparam.set_run_opt(sys.argv)
    logger = logging.getLogger('madgraph')
    opt =  MWparam.run_opt
    cmds = [] 
    if opt['param']: #-1
        cmds.append('treatcards')
    if opt['analyzer']: #-2
        cmds.append('get_integration_channel')
    if opt['compilation']: # -3
        cmds.append('compile')
    if opt['event']: # -4
        cmds.append('check_events')
    if opt['dir']: # -5
        logger.warning('Option -5/dir not supported/required anymore.')
    if opt['launch']: # -6
        cmds.append('submit_jobs')
    if opt['control']: # -7
        logger.warning('Option -7/control not supported anymore.')
    if opt['status']: # -7
        logger.warning('Option "status" not supported anymore.')        
    if opt['collect']: # -8
        cmds.append('collect')   
    if opt['plot']: # -9
        logger.warning('Option -9/plot not supported anymore.')            
    if opt['relaunch']: # relaunch
        cmds.append('refine 1')
    if opt['refine']: # refine
         cmds.append('refine %s' % argument[-1])
    if opt['clean'] == 1: # clean
         cmds.append('clean')
    if opt['clean']: # clean
         cmds.append('clean %s' % opt['clean'])         
         
    return cmds






################################################################################  
##   EXECUTABLE
################################################################################                                
if '__main__' == __name__:
    # Check that python version is valid

    set_configuration()
    argument = sys.argv
    try:
        launch = MW.MadWeightCmd(me_dir=root_path)
        if '-h' in argument or '--help' in argument:
            launch.exec_cmd('help launch')
        elif len(argument) > 1 and '-f' not in argument:
            cmds = treat_old_argument(argument)
            for cmd in cmds:
                launch.run_cmd(cmd)
        else: 
            launch.run_cmd('launch %s' % ' '.join(argument[1:]))
            launch.run_cmd('quit')
    except KeyboardInterrupt:
        try:
            launch.run_cmd('quit')
        except:
            pass
    except MW.AlreadyRunning, error:
        logging.error(str(error))
        sys.exit()
    except Exception, error:
        if os.path.exists(pjoin(root_path, 'RunWeb')): 
            os.remove(pjoin(root_path, 'RunWeb'))
        raise 
                        
    if os.path.exists(pjoin(root_path, 'RunWeb')): 
        os.remove(pjoin(root_path, 'RunWeb'))      
    
    
    # reconfigure path for the web 
    #if len(argument) == 5:
    #    ME.pass_in_web_mode()

             
        

        
    
    
    
    
    
    
    
