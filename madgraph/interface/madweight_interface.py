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
import time
import glob
import math
import xml.sax.handler
import shutil
from cStringIO import StringIO

if __name__ == '__main__':
    import sys
    sys.path.append('/Users/omatt/Documents/eclipse/madweight/')

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
    import madgraph.madweight.MW_driver as MW_driver
    
    import madgraph.various.misc as misc
    import madgraph.various.banner as banner
    import madgraph.iolibs.files as files
    MADEVENT = False
except ImportError, error:
    logger.debug(error)
    from internal import InvalidCmd, MadGraph5Error
    import internal.extended_cmd as cmd
    import internal.common_run_interface as common_run
    import internal.madweight.MW_info as MW_info
    import internal.madweight.change_tf as change_tf
    import internal.madweight.create_param as create_param
    import internal.madweight.create_run as create_run
    import internal.madweight.Cards as Cards
    import internal.madweight.write_MadWeight as write_MadWeight
    import internal.madweight.verif_event as verif_event
    import internal.madweight.MW_driver as MW_driver
    
    
    import internal.misc as misc 
    import internal.banner as banner
    import internal.files as files
    MADEVENT = True


AlreadyRunning = common_run.AlreadyRunning

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
    error_debug = 'Please report this bug on https://bugs.launchpad.net/mg5amcnlo\n'
    error_debug += 'with MadWeight in the title of the bug report.'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/mg5amcnlo\n'


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
            root_path = pjoin(os.path.dirname(__file__), os.path.pardir,os.path.pardir)
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
    
    def help_collect(self):
        logger.info('collect [-refine]')
        logger.info('  combine the results of the jobs launched on the cluster.')
        logger.info('  This creates three type of output files:')
        logger.info('    - weights.out [weight for event/card]')
        logger.info('    - unnormalize-likelihood.out [-\sum ln(Weight)]')
        logger.info('    - output.xml [additional information]')
        logger.info('')
        logger.info('  The option \'-refine\' is to be added if this is not the first')
        logger.info('  cluster submission. Otherwise previous run submission will be lost.')
           
    def help_define_transfer_fct(self):
        """help for define transfer_fct"""
        logger.info('  Modify the current transfer functions')
        logger.info('  If no argument provided a question showing the list of possibilities.')
        logger.info('  will be ask. If the TF is provided as argument, no question is asked.')


class CompleteForCmd(object):
    
    
    def complete_collect(self,text, line, begidx, endidx):
        """ complete the collect command"""
        args = self.split_arg(line[0:begidx])

        return self.list_completion(text,['-refine','-f','--refine'], line)
    
    def complete_define_transfer_fct(self, text, line, begidx, endidx):
        """ complete the define_transfer_fct """
        
        path = pjoin(self.me_dir, 'Source', 'MadWeight', 'transfer_function', 'data')
        listdir=os.listdir(path)
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            
            possibilities = [content[3:-4] for content in listdir \
                     if (content.startswith('TF') and content.endswith('dat'))]
            return self.list_completion(text, possibilities, line)

#===============================================================================
# MadWeightCmd
#===============================================================================
class MadWeightCmd(CmdExtended, HelpToCmd, CompleteForCmd, common_run.CommonRunCmd):
    
    _set_options = []
    prompt = 'MadWeight5>'
    helporder = ['MadWeight Function', 'Documented commands', 'Advanced commands']
    
    def remove_fct(self):
        """Not in help: remove fct"""
        return None
    
    do_decay_events = remove_fct
    do_delphes = remove_fct
    do_pgs = remove_fct
    
    
    ############################################################################
    def __init__(self, me_dir = None, options={}, *completekey, **stdin):
        """ add information to the cmd """

        CmdExtended.__init__(self, *completekey, **stdin)
        common_run.CommonRunCmd.__init__(self, me_dir, options)
        self.configured = 0 # time at which the last option configuration occur
    
    def do_quit(self, *args, **opts):
        common_run.CommonRunCmd.do_quit(self, *args, **opts)
        CmdExtended.do_quit(self, *args, **opts)
        return True
    
    def configure(self):
        os.chdir(pjoin(self.me_dir))
        self.__CMD__initpos = self.me_dir
        
        time_mod = max([os.path.getctime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getctime(pjoin(self.me_dir,'Cards','MadWeight_card.dat'))])
        

        if self.configured > time_mod and \
                           hasattr(self,'MWparam') and hasattr(self,'run_card'):
            return
        self.configured = time.time()
        
        
                  
        self.MWparam = MW_info.MW_info(pjoin(self.me_dir,'Cards','MadWeight_card.dat'))
        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner.RunCard(run_card)
        
        if self.options['run_mode'] == 0:
            self.exec_cmd('set run_mode 2 --no_save')
            self.exec_cmd('set nb_core 1 --no_save')
        if not self.options['cluster_temp_path']:
            if self.options['run_mode'] == 2:
                logger.info('Options cluster_temp_path is required for MW run. Trying to run with /tmp',
                                '$MG:color:BLACK')
                self.exec_cmd('set cluster_temp_path /tmp --no_save')
            elif self.options['cluster_type'] != 'condor':
                raise Exception, 'cluster_temp_path needs to be define for MW. Please retry.'
        
    def do_define_transfer_fct(self, line):
        """MadWeight Function:Define the current transfer function"""
        
        with misc.chdir(self.me_dir):  
            self.configure()
            args = self.split_arg(line)
            
            path = pjoin(self.me_dir, 'Source', 'MadWeight', 'transfer_function', 'data')
            listdir=os.listdir(path)
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
            change_tf.create_TF_main(tfname,0, MW_dir)
            
        
    def do_treatcards(self, line):
        """MadWeight Function:create the various param_card // compile input for the run_card"""
        self.configure()
        args = self.split_arg(line)
        
        create_param.Param_card(run_name=self.MWparam)
        self.MWparam.update_nb_card()
        Cards.create_include_file(self.MWparam)
        create_run.update_cuts_status(self.MWparam)
        
    def do_get_integration_channel(self, line):
        """MadWeight Function:analyze the cards/diagram to find an way to integrate efficiently"""
        self.configure()
        args = self.split_arg(line)        
    
        write_MadWeight.create_all_fortran_code(self.MWparam)
        
    def do_compile(self, line, refine=False):
        """MadWeight Function:compile the code"""
        self.configure()
        
        misc.compile(arg=["../lib/libtools.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libblocks.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libTF.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libpdf.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libdhelas.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libmodel.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libgeneric.a"], cwd=pjoin(self.me_dir,'Source')) 
        misc.compile(arg=["../lib/libcernlib.a"], cwd=pjoin(self.me_dir,'Source')) 

#
#       here check validity of some parameters
        if self.MWparam['mw_run']['integrator']=='m' and  self.MWparam['mw_run']['montecarlo_perm']=='t':
           raise Exception, 'Cannot use mint if monte carlo over permutations'
        if self.MWparam['mw_run']['integrator']=='m' and  self.MWparam['mw_run']['use_sobol']=='t':
           raise Exception, 'sobol generator with mint not implemented'
 

        
        for MW_dir in self.MWparam.MW_listdir:
            logger.info('compile %s' %MW_dir)
            pdir = pjoin(self.me_dir,'SubProcesses',MW_dir)
            if refine and os.path.exists(pjoin(pdir, 'initialization.o')):
                os.remove(pjoin(pdir, 'initialization.o'))
            if refine and  os.path.exists(pjoin(pdir, 'comp_madweight')):
                os.remove(pjoin(pdir, 'comp_madweight'))
            misc.compile(cwd=pdir)
            if not os.path.exists(pjoin(pdir, 'comp_madweight')):
                raise Exception, 'compilation fails'
        logger.info('MadWeight code has been compiled.')
        
    
    def do_check_events(self, line):
        """MadWeight Function: check that the events are valid
        and write the events to MG mapping"""
        self.configure()
        evt_file = pjoin(self.me_dir,'Events','input.lhco')
        if not os.path.exists(evt_file):
            question = 'Which LHCO file do you want to use?'
            default = ''            
            if os.path.exists('%s.gz' % evt_file):
                input_file =  '%s.gz' % evt_file
            else:
                input_file = self.ask(question, default, path_msg='valid path')
            
            if not input_file:
                raise self.InvalidCmd('Please specify a valid LHCO File')
            
            if input_file.endswith('.gz'):
                misc.gunzip(input_file, keep=True, stdout=evt_file)
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
        
    def do_submit_jobs(self, line):
        """MadWeight Function:Submitting the jobs to the cluster"""
        

        self.configure()
        self.clean_old_run(keep_event=True)
        args = self.split_arg(line)
        self.check_launch_jobs(args)
        # now args is of the type [True True]
        create_dir, launch_jobs = args[0], args[1]

        for nb_card in self.MWparam.actif_param:
            for dirname in self.MWparam.MW_listdir:
                nb_job = self.MWparam.nb_event_MW[dirname]
                if self.MWparam['mw_run']['nb_event_by_node'] > 1:
                    nb_job = 1+ (nb_job-1) // self.MWparam['mw_run']['nb_event_by_node']
                                    
                for event_sample in range(nb_job):
                    self.submit_job(dirname, nb_card, event_sample)        
    
        starttime = time.time()
        #logger.info('     Waiting for submitted jobs to complete')
        update_status = lambda i, r, f: self.update_status((i, r, f, 'madweight'), 
                      starttime=starttime, level='madweight', update_results=False)

        try:
            self.cluster.wait(self.me_dir, update_status)
        except Exception:
            self.cluster.remove()
            raise                
        except KeyboardInterrupt:
            if not self.force:
                ans = self.ask('Error detected. Do you want to clean the queue?',
                             default = 'y', choices=['y','n'])
            else:
                ans = 'y'
            if ans == 'y':
                self.cluster.remove()
            raise
    
    def submit_job(self, dirname, nb_card, sample_nb, evt_file=None, restrict_evt=[]):
        """launch on the cluster the job which creates the computation"""
        
        input_files = [pjoin(self.me_dir, 'SubProcesses', dirname, 'comp_madweight'), 
                       pjoin(self.me_dir, 'Cards', 'param_card_%i.dat' % nb_card),
                       self.get_pdf_input_filename(),
                       pjoin(self.me_dir, 'Cards', 'ident_card.dat'),
                       pjoin(self.me_dir, 'Cards', 'run_card.dat')
                       ]
        
        # add event_file:
        if not evt_file:
            evt_file = (sample_nb // self.MWparam['mw_run']['event_packing'])
        evt = 'verif_%i.lhco' % evt_file
        first_event = (sample_nb % self.MWparam['mw_run']['event_packing']) * self.MWparam['mw_run']['nb_event_by_node'] 
        name = self.MWparam.name
        input_files.append(pjoin(self.me_dir, 'SubProcesses', dirname, name, evt))
        
        if restrict_evt:
            restrict_path = pjoin(self.me_dir, 'SubProcesses', dirname, name, 
                                                  'restrict%i_%i.dat' % (nb_card,evt_file))
            input_files.append(restrict_path)
            #open(restrict_path, 'w').write(' '.join(map(str, restrict_evt)))
        
        # Need to add PDF (maybe also symfact, ...) ?
        
        output_file = ['output_%s_%s.xml' % (nb_card, sample_nb)]
        exe = pjoin(self.me_dir, 'bin', 'internal', 'madweight', 'MW_driver.py')
        
        # expected args: card_nb, first_event, nb_event, evt, mw_int_points, log_level
        args = [str(nb_card), str(first_event),
                str(self.MWparam['mw_run']['nb_event_by_node']) ,evt, 
                str(self.MWparam['mw_run']['mw_int_points']),
                self.MWparam['mw_run']['log_level'], str(sample_nb)]
        cwd = pjoin(self.me_dir, 'SubProcesses', dirname, name)
        # Ensure that the code is working ONLY if TEMP_CLUSTER_PATH is define
        if self.options['run_mode'] == 0:
            raise Exception , 'need to check the validity'
        else:
            # ensure that this is running with NO central disk !!!
            if not self.options['cluster_temp_path'] and not self.options['cluster_type'] == 'condor':
                raise self.ConfigurationError, 'MadWeight requires temp_cluster_path options to be define'
            self.cluster.submit2(exe, args, cwd, input_files=input_files, output_files=output_file)



    def check_collect(self, args):
        """ """
        
        if len(args) >1:
            self.help_collect()
            raise self.InvalidCmd, 'Invalid Command format'
        elif len(args) == 1:
            if args not in ['-refine', '--refine']:
                args[0] = '-refine'
            else:
                self.help_collect()
                raise self.InvalidCmd, 'Invalid Command format'

    def do_collect(self, line):
        """MadWeight Function: making the collect of the results"""
        
        self.configure()
        args = self.split_arg(line)
        self.check_collect(args)
        xml_reader = MWParserXML()
        
        name = self.MWparam.name
        # 1. Concatanate the file. #############################################
        out_dir = pjoin(self.me_dir, 'Events', name)
        if '-refine' in args:
            out_path = pjoin(out_dir, 'refine.xml') 
        else:
            out_path = pjoin(out_dir, 'output.xml')
            if os.path.exists(out_path):
                logger.warning('Output file already exists. Current one will be tagged with _old suffix')
                logger.warning('Run "collect -refine to instead update your current results."')
                files.mv(pjoin(out_dir, 'output.xml'), pjoin(out_dir, 'output_old.xml'))
                files.mv(pjoin(out_dir, 'weights.out'), pjoin(out_dir, 'weights.out'))
                for MWdir in self.MWparam.MW_listdir:
                    out_dir = pjoin(self.me_dir, 'Events', name, MWdir)
                    files.mv(pjoin(out_dir, 'output.xml'), pjoin(out_dir, 'output_old.xml'))
                out_dir = pjoin(self.me_dir, 'Events', name)
                    
        fsock = open(out_path, 'w')
        fsock.write('<madweight>\n<banner>\n')
        # BANNER
        for card in ['proc_card_mg5.dat','MadWeight_card.dat','transfer_card.dat','param_card.dat','run_card.dat']:
            cname = card[:-4]
            fsock.write('<%s>\n' % cname)
            fsock.write(open(pjoin(self.me_dir,'Cards',card)).read().replace('<','!>'))
            fsock.write('</%s>\n' % cname)
        fsock.write('</banner>\n')
        at_least_one = False
        for MWdir in self.MWparam.MW_listdir:
            out_dir = pjoin(self.me_dir, 'Events', name, MWdir)
            input_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, name)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            if '-refine' in args:
                out_path = pjoin(out_dir, 'refine.xml') 
            else:
                out_path = pjoin(out_dir, 'output.xml')  
            fsock2 = open(out_path,'w')
            fsock.write('<subprocess id=\'%s\'>\n' % MWdir)
            fsock2.write('<subprocess id=\'%s\'>\n' % MWdir)
            for output in misc.glob('output_*_*.xml', input_dir):
                at_least_one = True
                text = open(output).read()
                fsock2.write(text)
                fsock.write(text)
                os.remove(output)
            fsock.write('</subprocess>\n')
            fsock2.write('</subprocess>\n')
            fsock2.close()
        fsock.write('\n</madweight>\n')          
        fsock.close()
        # 2. Special treatment for refine mode
        if '-refine' in args:
            xml_reader2 = MWParserXML(self.MWparam['mw_run']['log_level'])
            for MWdir in self.MWparam.MW_listdir:
                out_dir = pjoin(self.me_dir, 'Events',name, MWdir)
                ref_output = xml_reader2.read_file(pjoin(out_dir, 'refine.xml'))
                xml_reader2 = MWParserXML(self.MWparam['mw_run']['log_level'])
                base_output = xml_reader2.read_file(pjoin(out_dir, 'output.xml'))

                base_output.refine(ref_output)
                files.mv(pjoin(out_dir, 'output.xml'), pjoin(out_dir, 'output_old.xml'))
                base_output.write(pjoin(out_dir, 'output.xml'), MWdir)
        elif not at_least_one:
            logger.warning("Nothing to collect restore _old file as current.")
            out_dir = pjoin(self.me_dir, 'Events', name)
            files.mv(pjoin(out_dir, 'output_old.xml'), pjoin(out_dir, 'output.xml'))
            files.mv(pjoin(out_dir, 'weights_old.out'), pjoin(out_dir, 'weights.out'))
            for MWdir in self.MWparam.MW_listdir:
                out_dir = pjoin(self.me_dir, 'Events', name, MWdir)
                files.mv(pjoin(out_dir, 'output.xml'), pjoin(out_dir, 'output_old.xml'))
            
            
            
        # 3. Read the (final) log file for extracting data
        total = {}
        likelihood = {}
        err_likelihood = {}
        cards = set()
        events = set()
        tf_sets = set()
        for MW_dir in self.MWparam.MW_listdir:
            out_dir = pjoin(self.me_dir, 'Events', name, MW_dir)
            xml_reader = MWParserXML()
            data = xml_reader.read_file(pjoin(out_dir, 'output.xml'))
            #
            log_level = self.MWparam['mw_run']['log_level']            
            generator =  ((int(i),j,int(k),data[i][j][k]) for i in data 
                                                               for j in data[i] 
                                                               for k in data[i][j])
            for card, event, tf_set, obj in generator:
                # update the full list of events/cards
                cards.add(card)
                events.add(event)
                tf_sets.add(tf_set)
                # now compute the associate value, error[square]
                if (card,event, tf_set) in total:
                    value, error = total[(card, event, tf_set)]                    
                else:
                    value, error = 0, 0
                obj.calculate_total()
                value, error = (value + obj.value, error + obj.error**2) 
                total[(card, event, tf_set)] = (value, error)
                if tf_set == 1:
                    if value:
                        if card not in likelihood:
                            likelihood[card], err_likelihood[card] = 0, 0
                        likelihood[card] -= math.log(value)
                        err_likelihood[card] += error / value
                    else:
                        likelihood[card] = float('Inf')
                        err_likelihood[card] = float('nan')

                
        # write the weights file:
        fsock = open(pjoin(self.me_dir, 'Events', name, 'weights.out'), 'w')
        logger.info('Write output file with weight information: %s' % fsock.name)
        fsock.write('# Weight (un-normalize) for each card/event/set of transfer fct\n')
        fsock.write('# format: LHCO_event_number card_id tf_set_id value integration_error\n')
        events = list(events)
        events.sort()
        cards = list(cards)
        cards.sort()
        tf_sets = list(tf_sets)
        tf_sets.sort()
        for event in events:
            for card in cards:
                for tf_set in tf_sets:
                    try:
                        value, error = total[(card, event,tf_set)]
                    except KeyError:
                        continue
                    error = math.sqrt(error)
                    fsock.write('%s %s %s %s %s \n' % (event.replace('@', ' '), card, tf_set, value, error))
    
        # write the likelihood file:
        fsock = open(pjoin(self.me_dir, 'Events', name, 'un-normalized_likelihood.out'), 'w')
        fsock.write('# Warning:  this Likelihood needs a bin by bin normalization !\n')
        fsock.write('# IF more than one set of transfer function are define. ONLY the first one is ')
        fsock.write('# include in this file.')
        fsock.write('# format: card_id value integration_error\n')
        for card in cards:
            value, error = likelihood[card], err_likelihood[card]
            error = math.sqrt(error)
            fsock.write('%s %s %s \n' % (card, value, error))
            
    
    def do_clean(self, line):
        """MadWeight Function: syntax: clean [XXX]
           clean the previous run XXX (the last one by default)"""
           
        args = self.split_arg(line)
        self.configure()
        if len(args) == 0:
            name = self.MWparam.name
        else:
            name = args[0]
        
        ans = self.ask('Do you want to remove Events/%s directory?', 'y',['y','n'])
        if ans == 'y':
            try:
                shutil.rmtree(pjoin(self.me_dir, 'Events', name))
            except Exception, error:
                logger.warning(error)
        for Pdir in self.MWparam.MW_listdir:
            try:
                shutil.rmtree(pjoin(self.me_dir, 'SubProcesses', Pdir, name))
            except Exception, error:
                logger.warning(error)            
    
    def ask_edit_cards(self, cards, *arg, **opts):
        super(MadWeightCmd, self).ask_edit_cards(cards, *arg, **opts)
        self.configured = 0
        self.configure()
        
    def do_launch(self, line):
        """MadWeight Function:run the full suite of commands"""

        args = self.split_arg(line)
    
        #if not os.path.exists(pjoin(self.me_dir, 'Cards','transfer_card.dat')):
        #    self.exec_cmd('define_transfer_fct')
        
        cards = ['param_card.dat', 'run_card.dat', 'madweight_card.dat', 
                 'transfer_card.dat', 'input.lhco']
        if not self.force:
            self.ask_edit_cards(cards, mode='fixed', plot=False)
        else:
            self.configured = 0
            self.configure()
        with misc.chdir(self.me_dir): 
            if not os.path.exists(pjoin(self.me_dir, self.MWparam['mw_run']['inputfile'])):
                raise self.InvalidCmd('Please specify a valid LHCO File')
            if pjoin(self.me_dir, self.MWparam['mw_run']['inputfile']) not in \
                    [pjoin(self.me_dir, 'Events', 'input.lhco'), pjoin(self.me_dir, 'Events', 'input.lhco.gz')]:
                zipped = self.MWparam['mw_run']['inputfile'].endswith('.gz')
                if zipped:
                    files.cp(pjoin(self.me_dir, self.MWparam['mw_run']['inputfile']),
                             pjoin(self.me_dir, 'Events', 'input.lhco.gz'))
                    if os.path.exists(pjoin(self.me_dir, 'Events', 'input.lhco')):
                        os.remove(pjoin(self.me_dir, 'Events', 'input.lhco'))
                else:
                    files.cp(pjoin(self.me_dir, self.MWparam['mw_run']['inputfile']),
                             pjoin(self.me_dir, 'Events', 'input.lhco'))                
                     
            if not (os.path.exists(pjoin(self.me_dir, 'Events', 'input.lhco')) or \
                     os.path.exists(pjoin(self.me_dir, 'Events', 'input.lhco.gz'))):
                raise self.InvalidCmd('Please specify a valid LHCO File')
    
            self.exec_cmd('treatcards')
            self.exec_cmd('get_integration_channel')
            self.exec_cmd('compile')
            self.exec_cmd('check_events')
            self.exec_cmd('submit_jobs')
            self.exec_cmd('collect')
        
    
    def check_refine(self, args):
        """check the argument validity"""
        
        if len(args) != 1:
            raise self.InvalidCmd('refine requires a single argument')
        
        try:
            args[0] = float(args[0])
        except Exception:
            raise self.InvalidCmd('First argument of refine command should be a number')
            
        if args[0] < 0 or args[0]>1:
            raise self.InvalidCmd('The first argument should be a number between 0 and 1.')
    
    def clean_old_run(self, keep_event=False):
        
        for MWdir in self.MWparam.MW_listdir:
            input_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, self.MWparam.name)
            if not os.path.exists(input_dir):
                continue
            for filename in os.listdir(input_dir):
                if keep_event and filename.startswith('verif'):
                    continue
                os.remove(pjoin(input_dir, filename))
            
            
    
    def do_refine(self, line):
        """MadWeight Function:syntax: refine X
        relaunch the computation of the weight which have a precision lower than X"""
        args = self.split_arg(line)
        self.check_refine(args)
        self.configure()
        
        self.clean_old_run(keep_event=True)     
        if not self.force:
            cards = ['madweight_card.dat'] 
            self.ask_edit_cards(cards, mode='fixed', plot=False)
            self.exec_cmd("treatcards")
            self.do_compile('', refine=True) # force re-compilation
        
        nb_events_by_file = self.MWparam['mw_run']['nb_event_by_node'] * self.MWparam['mw_run']['event_packing'] 
        asked_events = self.MWparam['mw_run']['nb_exp_events']
        
        precision = args[0]
        name = self.MWparam.name
        allow_refine = []
        # events/cards to refine
        fsock = open(pjoin(self.me_dir, 'Events', name, 'weights.out'), 'r')
        for line in fsock:
            if '#' in line:
                line = line.split('#')[0]
            line = line.split()
            if len(line) == 5:
                lhco_nb, card_nb, tf_set, value, error = line
            else:
                continue
            if tf_set != '1':
                continue
            if float(value) * precision < float(error):
                allow_refine.append((int(card_nb), lhco_nb))
        logger.info("%s selected jobs for the refinment." % len(allow_refine))
        for MWdir in self.MWparam.MW_listdir:
            # We need to know in which file are written all the relevant event
            event_to_file = {}
            for evt_nb in range(asked_events//nb_events_by_file +1):
                evt = 'verif_%i.lhco' % evt_nb
                for line in open(pjoin(self.me_dir, 'SubProcesses', MWdir, name, evt)):
                    split = line.split()
                    if len(split) == 3:
                        event_to_file[split[1]] = evt_nb
            to_refine = {}
            out_dir = pjoin(self.me_dir, 'Events',name, MWdir)
            xml_reader = MWParserXML(keep_level='weight')
            data = xml_reader.read_file(pjoin(out_dir, 'output.xml'))
            generator =  ((int(i),j,int(k),data[i][j][k]) for i in data 
                                                               for j in data[i] 
                                                               for k in data[i][j])
            for card, event, tf_set, obj in generator:
                if tf_set != 1:
                    continue
                value, error = obj.value, obj.error
                if value * precision < error:
                    if card not in to_refine:
                        to_refine[card] = []
                    if (card,event) in allow_refine:
                        to_refine[card].append(event)
            if to_refine:
                self.resubmit(MWdir, to_refine, event_to_file)
        
        # control
        starttime = time.time()
        update_status = lambda i, r, f: self.update_status((i, r, f, 'madweight'), 
                  starttime=starttime, level='madweight', update_results=False)
        try:
            self.cluster.wait(self.me_dir, update_status)
        except Exception:
            self.cluster.remove()
            raise                
        except KeyboardInterrupt:
            if not self.force:
                ans = self.ask('Error detected. Do you want to clean the queue?',
                             default = 'y', choices=['y','n'])
            else:
                ans = 'y'
            if ans == 'y':
                self.cluster.remove()
            raise
        
        self.do_collect('-refine')
                
    def resubmit(self, M_path, to_refine, event_to_file):
        """resubmit various jobs"""

        for card, event_list in to_refine.items():
            packets = {}
            for event in event_list:
                evt_nb_file = event_to_file[event]
                if evt_nb_file in packets:
                    packets[evt_nb_file].append(event)
                else:
                    packets[evt_nb_file] = [event]

#                evt_file = (sample_nb // self.MWparam['mw_run']['event_packing'])
#        open(restrict_path, 'w').write(' '.join(map(str, restrict_evt)))   

            max_evts = self.MWparam['mw_run']['nb_event_by_node']
            for evt_nb, evt_list in packets.items():
                restrict_path = pjoin(self.me_dir, 'SubProcesses', M_path, self.MWparam.name, 
                                                  'restrict%i_%i.dat' % (card,evt_nb_file))
                open(restrict_path, 'w').write(' '.join(map(str, evt_list)))  
                
                nb_weights = len(evt_list)
                for i in range(1+ (nb_weights-1)//max_evts):
                    #sub_list = evt_list[max_evts * i: max_evts * (i+1)]
                    self.submit_job(M_path, card, i, evt_file=evt_nb,
                                    restrict_evt=True)                        
        
        
        
#===============================================================================
# MadEventCmd
#===============================================================================
class MadWeightCmdShell(MadWeightCmd, cmd.CmdShell):
    """The command line processor of MadGraph"""  
    pass


class CollectObj(dict):
    pass

    #2 #############################################################################  
    def refine(self, other):
        
        for card, CardDATA in other.items():
            if card not in self:
                self[card] = other[card]
                continue
            
            for event, obj2 in CardDATA.items():
                self[card][event] = obj2

        
    
    def write(self, out_path, MWdir):
        """ """
        fsock = open(out_path, 'w')
        fsock.write('<subprocess id=\'%s\'>\n' % MWdir)
        for card in self:
            fsock.write('<card id=\'%s\'>\n' % card)
            for event in self[card].values():
                event.write(fsock)
            fsock.write('</card>\n')
        
        fsock.write('</subprocess>')
        fsock.close()
        

#1 ################################################################################# 
class MWParserXML(xml.sax.handler.ContentHandler):
    """ This class will organize in python obect the TF_param.dat file 
        (written in xml) 
    """

    #2 #############################################################################        
    def __init__(self, keep_level='weight'):
        self.in_el = {'process': '', 'card':'', 'event':'', 'tfset':'' ,
                      'permutation':'', 'channel':'','full':''}
        if keep_level in ['weight', 'event', 'debug']:
            keep_level = 'tfset'
        self.all_event = {}
        self.keep_level = keep_level
        self.buffer=''
        self.output = CollectObj() 


    #2 #############################################################################  
    def startElement(self, name, attributes):
        
        # if in lower level than needed skip the collection of data
        if self.in_el[self.keep_level] != '' or name == 'log':
            return
        
        obj_class = {'event': MW_driver.Weight, 'permutation':MW_driver.Permutation,
         'channel': MW_driver.Channel, 'tfset':MW_driver.TFsets}
        
        
        if name == "process":
            pass
        elif name == 'card':
            id = attributes['id']
            if id not in self.output:
                self.output[id] = {}
            self.in_el[name] = self.output[id]
            self.curr_card_id = id            
        elif name == 'event':
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            card = self.curr_card_id
            if ((card,id) in self.all_event):
                data = self.all_event[(card,id)]
            else:
                data =  MW_driver.Weight(id, self.keep_level)
                self.all_event[(card,id)] = data
            data.value = value
            data.error = error
            self.in_el['event'] = data
            # assign it in the mother:
            card = self.in_el['card']
            card[id] = data
        elif name == 'tfset':
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            data =  MW_driver.TFsets(id)
            data.value = value
            data.error = error
            data.add('0', '', value, error, '')
            #assign it to the mother:
            event = self.in_el['event']
            event[id] = data
            self.in_el['tfset'] = data
        elif name == 'permutation':
            tfset = self.in_el['tfset']
            
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            data =  MW_driver.Permutation(id, '')
            data.value = value
            data.error = error
            self.in_el['permutation'] = data
            # assign it in the mother:
            tfset[id] = data
        elif name == 'channel':
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            data =  MW_driver.Channel(id, value, error)
            self.in_el['channel'] = data
            # assign it in the mother:
            permutation = self.in_el['permutation']
            permutation[id] = data
        elif name in ['log','subprocess','br']:           
            pass
        else:
            raise Exception, name
        if name != 'br':
            self.text = StringIO()
        
    def characters(self, content):
        self.text.write(content)
    
    def endElement(self, name):
        if name == 'log':
            data = self.in_el['event']
            data.log = self.text.getvalue()
            
        self.in_el[name] = ''
            
    #2 ############################################################################# 
    def read_file(self,filepos):
        """ parse the file and fulfill the object """
        self.output = CollectObj()
        parser = xml.sax.make_parser(  )
        parser.setContentHandler(self)
        parser.parse(filepos)
        return self.output
