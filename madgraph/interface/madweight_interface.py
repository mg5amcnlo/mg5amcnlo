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
"""
A user friendly interface to access all the function associated to MadWeight 
"""

import logging
import os
import subprocess

logger = logging.getLogger('cmdprint')

pjoin = os.path.join
try:
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR
    import madgraph.interface.extended_cmd as cmd
    import madgraph.interface.common_run_interface as common_run
    
    import madgraph.madweight.MW_info as MW_info
    import madgraph.madweight.change_tf as change_tf
    import madgraph.madweight.create_param as create_param
    import madgraph.madweight.create_run as create_run
    import madgraph.madweight.Cards as Cards
    import madgraph.madweight.write_MadWeight as write_MadWeight
    import madgraph.madweight.verif_event as verif_event
    
    import madgraph.various.misc as misc
    import madgraph.iolibs.files as files
    MADEVENT = False
except ImportError, error:
    logger.debug(error)
    print error
    raise
    from internal import InvalidCmd, MadGraph5Error
    import internal.extended_cmd as extended_cmd
    import internal.common_run_interface as common_run
    import internal.madweight.MW_info as MW_info
    import internal.madweight.change_tf as change_tf
    import internal.madweight.create_param as create_param
    import internal.madweight.create_run as create_run
    import internal.madweight.Cards as Cards
    import internal.madweight.write_MadWeight as write_MadWeight
    import internal.madweight.verif_event as verif_event
    import internal.files as files
    
    import internal.misc as misc 
    MADEVENT = True



#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
    """Particularisation of the cmd command for MadEvent"""

    #suggested list of command
    next_possibility = {
        'start': [],
    }
    
    debug_output = 'MW5_debug'
    error_debug = 'Please report this bug on https://bugs.launchpad.net/madgraph5\n'
    error_debug += 'with MadWeight in the title of the bug report.'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/madgraph5\n'


    keyboard_stop_msg = """stopping all operation
            in order to quit madweight please enter exit"""
    
    # Define the Error
    InvalidCmd = InvalidCmd
    ConfigurationError = MadGraph5Error

    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
        # Tag allowing/forbiding question
        self.force = False
        
        # If possible, build an info line with current version number 
        # and date, from the VERSION text file
        info = misc.get_pkg_info()
        info_line = ""
        if info and info.has_key('version') and  info.has_key('date'):
            len_version = len(info['version'])
            len_date = len(info['date'])
            if len_version + len_date < 30:
                info_line = "#*         VERSION %s %s %s         *\n" % \
                            (info['version'],
                            (30 - len_version - len_date) * ' ',
                            info['date'])
        else:
            version = open(pjoin(root_path,'MGMEVersion.txt')).readline().strip()
            info_line = "#*         VERSION %s %s                *\n" % \
                            (version, (24 - len(version)) * ' ')    

        # Create a header for the history file.
        # Remember to fill in time at writeout time!
        self.history_header = \
        '#************************************************************\n' + \
        '#*                        MadWeight 5                       *\n' + \
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
        '#*              Command File for MadWeight                  *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/madweight.py FILENAME                   *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        if info_line:
            info_line = info_line[1:]

        logger.info(\
        "************************************************************\n" + \
        "*                                                          *\n" + \
        "*           W E L C O M E  to  M A D G R A P H  5          *\n" + \
        "*                      M A D W E I G H T                   *\n" + \
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
        "************************************************************")
        
        cmd.Cmd.__init__(self, *arg, **opt)

class HelpToCmd(object):
    pass

class CompleteForCmd(object):
    pass

#===============================================================================
# MadWeightCmd
#===============================================================================
class MadWeightCmd(CmdExtended, HelpToCmd, CompleteForCmd, common_run.CommonRunCmd):
    
    _set_options = []
    prompt = 'MadWeight5>'
    
    ############################################################################
    def __init__(self, me_dir = None, options={}, *completekey, **stdin):
        """ add information to the cmd """

        CmdExtended.__init__(self, *completekey, **stdin)
        common_run.CommonRunCmd.__init__(self, me_dir, options)
        self.configured = 0 # time at which the last option configuration occur
    
    def configure(self):
        os.chdir(pjoin(self.me_dir))
        self.__CMD__initpos = self.me_dir
        
        time_mod = max([os.path.getctime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getctime(pjoin(self.me_dir,'Cards','MadWeight_card.dat'))])
        
        if self.configured > time_mod:
            return
                        
        self.MWparam = MW_info.MW_info(pjoin(self.me_dir,'Cards','MadWeight_card.dat'))
    
    def do_define_transfer_fct(self, line):
        """Define the current transfer function"""
        self.configure()
        args = self.split_arg(line)
        
        
        listdir=os.listdir('./Source/MadWeight/transfer_function/data')
        question = 'Please choose your transfer_function between\n'
        possibilities = [content[3:-4] for content in listdir \
                     if (content.startswith('TF') and content.endswith('dat'))]
        for i, tfname in enumerate(possibilities):
            question += ' %s / %s\n' % (i, tfname)
        possibilities += range(len(possibilities))
        
        if args and args[0] in possibilities:
            tfname = args[0]
        else:
            tfname = self.ask(question, 'dbl_gauss_pt_jet', possibilities)
        if tfname.isdigit():
            tfname = possibilities[int(tfname)]
        
        P_dir, MW_dir = MW_info.detect_SubProcess(P_mode=1)
        os.chdir('./Source/MadWeight/transfer_function')
        change_tf.create_TF_main(tfname,0,MW_dir)
        os.chdir('../../..')
        
    def do_treatcards(self, line):
        """create the various param_card // compile input for the run_card"""
        self.configure()
        args = self.split_arg(line)
        
        create_param.Param_card(run_name=self.MWparam)
        self.MWparam.update_nb_card()
        Cards.create_include_file(self.MWparam)
        create_run.update_cuts_status(self.MWparam)
        
    def do_get_integration_channel(self, line):
        self.configure()
        args = self.split_arg(line)        
    
        write_MadWeight.create_all_fortran_code(self.MWparam)
        
    def do_compile(self, line):
        """compile the code"""
        
        misc.compile(arg=["../lib/libtools.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libblocks.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libTF.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libpdf.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libdhelas.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libmodel.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libgeneric.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libcernlib.a"], cwd=pjoin(self.me_dir,'Source')) 
        
        for MW_dir in self.MWparam.MW_listdir:
            misc.compile(cwd=pjoin(self.me_dir,'SubProcesses', MW_dir))
            if not os.path.exists(pjoin(self.me_dir,'SubProcesses',MW_dir, 'comp_madweight')):
                raise Exception, 'compilation fails'
        logger.info('MadWeight code has been compiled.')
        
    
    def do_check_events(self, line):
        """check that the events are valid"""
        
        evt_file = pjoin(self.me_dir,'Events','input.lhco')
        if not os.path.exists(evt_file):
            question = 'Which LHCO file do you want to use?'
            default = ''            
            
            input_file = self.ask(question, default, path_msg='valid path')
            
            if not input_file:
                raise self.InvalidCmd('Please specify a valid LHCO File')
            
            if input_file.endswith('.gz'):
                fsock = open(evt_file, 'w') 
                subprocess.call(['gunzip', '-c', input_file], stdout=fsock)
            else:
                files.cp(input_file, evt_file)
            
        verif_event.verif_event(self.MWparam)
        
    def check_launch_jobs(self, args):
        """format the argument to retrun a list with two argument,
        The first corresponding to the fact if we need te create the output dir
        The second if we can launch the job on the cluster."""
        
        if not args:
            #use default
            args[:] = [True, True]
            return
        else:
            create_dir = True
            launch = True
            for arg in args:
                if arg.count('=') !=1 :
                    logger.warning('command launch_jobs does not recognized argument %s. This argument is ignored' % arg)
                restrict, value = arg.split('=')
                if restrict == '--create_dir=':
                    if value in self.True:
                        create_dir = True
                    else: 
                        create_dir = False
                elif restrict == '--submit=':
                    if value in self.True:
                        launch = True
                    else: 
                        launch = False                
            args[:] = [create_dir, launch]
            return
        
    def do_launch_jobs(self, line):
        
        args = self.split_arg(line)
        self.check_launch_jobs(args)
        # now args is of the type [True True]
        create_dir, launch_jobs = args[0], args[1]
        
        for nb_card in self.MWparam.actif_param:
            for dirname in self.MWparam.MW_listdir:
                for event_sample in range(self.MWparam.nb_event_MW[dirname]//self.MWparam['mw_run']['nb_event_by_node']):
                    self.submit_job(dirname, nb_card, event_sample)
                    
    
    def submit_job(self, dirname, nb_card, sample_nb):
        """launch on the cluster the job which creates the computation"""
        
        input_files = [pjoin(self.me_dir, 'SubProcesses', dirname, 'comp_madweight'), 
                       pjoin(self.me_dir, 'SubProcesses','randinit'),
                       pjoin(self.me_dir,'Source','MadWeight','src','MW_driver.py'),
                       pjoin(self.me_dir, 'Cards', 'param_card_%i.dat' % nb_card),
                       ]
        
        # add event_file:
        evt = 'input_%i.lhco' % (sample_nb // self.MWparam['mw_run']['event_packing'])
        first_event = (sample_nb % self.MWparam['mw_run']['event_packing'])
        name = self.MWparam.name
        input_files.append(pjoin(self.me_dir, 'SubProcesses', dirname, name, evt))
                           
        # Need to add PDF (maybe also symfact, ...) ?
        
        output_file = ['output_%s_%s.xml' % (nb_card, sample_nb)]
        
        # need to define
        #1) the bash script
        #2) the associated python script
        #3) the argument to be passed to the function this should include
        #   first_event, card_nb, MW_param.integration_point, sample_nb, evt_name
        #4) adding the loop on the events

        
        
        
        
        

    
    
#===============================================================================
# MadEventCmd
#===============================================================================
class MadWeightCmdShell(MadWeightCmd, cmd.CmdShell):
    """The command line processor of MadGraph"""  
    pass