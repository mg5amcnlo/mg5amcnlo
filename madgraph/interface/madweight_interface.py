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
        
    def do_compile(self, line):
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
        
        for MW_dir in self.MWparam.MW_listdir:
            misc.compile(cwd=pjoin(self.me_dir,'SubProcesses', MW_dir))
            if not os.path.exists(pjoin(self.me_dir,'SubProcesses',MW_dir, 'comp_madweight')):
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
        
    def do_submit_jobs(self, line):
        """MadWeight Function:Submitting the jobs to the cluster"""
        
        self.configure()
        args = self.split_arg(line)
        self.check_launch_jobs(args)
        # now args is of the type [True True]
        create_dir, launch_jobs = args[0], args[1]

        for nb_card in self.MWparam.actif_param:
            for dirname in self.MWparam.MW_listdir:
                nb_job = self.MWparam.nb_event_MW[dirname]
                if self.MWparam['mw_run']['nb_event_by_node'] > 1:
                    nb_job = (nb_job+1) // self.MWparam['mw_run']['nb_event_by_node']
                
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
                                                  'restrict_%i.dat' % evt_file)
            input_files.append(restrict_path)
            open(restrict_path, 'w').write(' '.join(map(str, restrict_evt)))
        
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
                raise self.ConfigurationError, 'MadWeight requires temp_cluste_path options to be define'
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
        for MWdir in self.MWparam.MW_listdir:
            out_dir = pjoin(self.me_dir, 'Events', name, MWdir)
            input_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, name)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            if '-refine' in args:
                out_path = pjoin(out_dir, 'refine.xml') 
            else:
                out_path = pjoin(out_dir, 'output.xml')
            fsock = open(out_path, 'w')
            fsock.write('<subprocess id=\'%s\'>\n' % MWdir)
            for output in glob.glob(pjoin(input_dir, 'output_*_*.xml')):
                fsock.write(open(output).read())
                os.remove(output)
            fsock.write('</subprocess>')
            fsock.close()

        # 2. Special treatment for refine mode
        if '-refine' in args:
            xml_reader2 = MWParserXML(self.MWparam['mw_run']['log_level'])
            for MWdir in self.MWparam.MW_listdir:
                out_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, name)
                base_output = xml_reader2.read_file(pjoin(out_dir, 'output.xml'))
                sec_output = xml_reader2.read_file(pjoin(out_dir, 'refine.xml'))
                base_output.refine(sec_output)
                files.mv(pjoin(out_dir, 'output.xml'), pjoin(out_dir, 'output_old.xml'))
                base_output.write(pjoin(out_dir, 'output.xml'), MWdir)

        # 3. Read the (final) log file for extracting data
        total = {}
        likelihood = {}
        err_likelihood = {}
        cards = set()
        events = set()
        for MW_dir in self.MWparam.MW_listdir:
            out_dir = pjoin(self.me_dir, 'Events', name, MW_dir)
            xml_reader = MWParserXML()
            data = xml_reader.read_file(pjoin(out_dir, 'output.xml'))
            generator =  ((int(i),int(j),data[i][j]) for i in data for j in data[i])
            for card, event, obj in generator:
                # update the full list of events/cards
                cards.add(card)
                events.add(event)
                # now compute the associate value, error[square]
                if (card,event) in total:
                    value, error = total[(card, event)]                    
                else:
                    value, error = 0, 0
                value, error = (value + obj.value, error + obj.error**2) 
                total[(card, event)] = (value, error)
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
        fsock.write('# Weight (un-normalize) for each card/event\n')
        fsock.write('# format: LHCO_event_number card_id value integration_error\n')
        events = list(events)
        events.sort()
        cards = list(cards)
        cards.sort()
        for event in events:
            for card in cards:
                try:
                    value, error = total[(card, event)]
                except KeyError:
                    continue
                error = math.sqrt(error)
                fsock.write('%s %s %s %s \n' % (event, card, value, error))
    
        # write the likelihood file:
        fsock = open(pjoin(self.me_dir, 'Events', name, 'un-normalized_likelihood.out'), 'w')
        fsock.write('# Warning:  this Likelihood needs a bin by bin normalization !\n')
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
        
        
    def do_launch(self, line):
        """MadWeight Function:run the full suite of commands"""

        args = self.split_arg(line)
    
        #if not os.path.exists(pjoin(self.me_dir, 'Cards','transfer_card.dat')):
        #    self.exec_cmd('define_transfer_fct')
        
        cards = ['param_card.dat', 'run_card.dat', 'madweight_card.dat', 
                 'transfer_card.dat', 'input.lhco']
        if not self.force:
            self.ask_edit_cards(cards, mode='fixed', plot=False)
        with misc.chdir(self.me_dir):          
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
    
    def do_refine(self, line):
        """MadWeight Function:syntax: refine X
        relaunch the computation of the weight which have a precision lower than X"""
        args = self.split_arg(line)
        self.check_refine(args)
        self.configure()
        
        
        
        nb_events_by_file = self.MWparam['mw_run']['nb_event_by_node'] * self.MWparam['mw_run']['event_packing'] 
        asked_events = self.MWparam['mw_run']['nb_exp_events']
        
        precision = args[0]
        name = self.MWparam.name
        allow_refine = []
        # events/cards to refine
        fsock = open(pjoin(self.me_dir, 'Events', name, 'weights.out'), 'r')
        for line in fsock:
            line = line.split('#')[0].split()
            if len(line) == 4:
                lhco_nb, card_nb, value, error = line
            else:
                continue
            if float(value) * precision < float(error):
                allow_refine.append((int(card_nb), int(lhco_nb)))

        xml_reader = MWParserXML(keep_level=self.MWparam['mw_run']['log_level'])        
        for MWdir in self.MWparam.MW_listdir:
            # We need to know in which file are written all the relevant event
            event_to_file = {}
            for evt_nb in range(asked_events//nb_events_by_file +1):
                evt = 'verif_%i.lhco' % evt_nb
                for line in open(pjoin(self.me_dir, 'SubProcesses', MWdir, name, evt)):
                    split = line.split()
                    if len(split) == 3:
                        event_to_file[int(split[1])] = evt_nb

            to_refine = {}
            out_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, name)
            data = xml_reader.read_file(pjoin(out_dir, 'output.xml'))

            generator =  ((int(i),int(j),data[i][j]) for i in data for j in data[i])
            for card, event, obj in generator:
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
        except KeyboardInterupt:
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
            max_evts = self.MWparam['mw_run']['nb_event_by_node']
            for evt_nb, evt_list in packets.items():
                nb_weights = len(evt_list)
                for i in range(1+ (nb_weights-1)//max_evts):
                    sub_list = evt_list[max_evts * i: max_evts * (i+1)]
                    self.submit_job(M_path, card, i, evt_file=evt_nb,
                                    restrict_evt=sub_list)                        
        

    def collect_for_refine(self):
        """Collect data for refine"""

        #1 Load the object
        #2 Load the new object
        #3 Merge them
        
        xml_reader = MWParserXML()
        for MW_dir in self.MWparam.MW_listdir:
            out_dir = pjoin(self.me_dir, 'SubProcesses', MWdir, name)
            data = xml_reader.read_file(pjoin(out_dir, 'output.xml'))
        
        



        
#===============================================================================
# MadEventCmd
#===============================================================================
class MadWeightCmdShell(MadWeightCmd, cmd.CmdShell):
    """The command line processor of MadGraph"""  
    pass


class CollectObj(dict):
    pass

    def refine(self, other):
        """ """
        
        for card in other:
            if not card in self:
                self[card] = other[card]
                continue
            for event_nb in other[card]:
                self[card][event_nb] = other[card][event_nb]        
        return
    
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
        

#1 ################################################################################# 
class MWParserXML(xml.sax.handler.ContentHandler):
    """ This class will organize in python obect the TF_param.dat file 
        (written in xml) 
    """

    #2 #############################################################################        
    def __init__(self, keep_level='weight'):
        self.in_el = {'process': '', 'card':'', 'event':'', 'permutation':'', 'channel':'',
                      'full':''}
        if keep_level == 'weight':
            keep_level = 'event'
        self.keep_level = keep_level
        self.buffer=''
        self.output = CollectObj() 

    #2 #############################################################################  
    def startElement(self, name, attributes):
        
        # if in lower level than needed skip the collection of data
        if self.in_el[self.keep_level] != '' or name == 'log':
            return
        
        obj_class = {'event': MW_driver.Weight, 'permutation':MW_driver.Permutation,
         'channel': MW_driver.Channel}
        
        
        if name == "process":
            pass
        elif name == 'card':
            id = attributes['id']
            if id not in self.output:
                self.output[id] = {}
            self.in_el[name] = self.output[id]            
        elif name == 'event':
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            data =  MW_driver.Weight(id, self.keep_level)
            data.value = value
            data.error = error
            self.in_el['event'] = data
            # assign it in the mother:
            card = self.in_el['card']
            card[id] = data
        elif name == 'permutation':
            id = attributes['id']
            value = float(attributes['value'])
            error = float(attributes['error'])
            data =  MW_driver.Permutation(id, '')
            data.value = value
            data.error = error
            self.in_el['permutation'] = data
            # assign it in the mother:
            event = self.in_el['event']
            event.append(data)
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
        parser = xml.sax.make_parser(  )
        parser.setContentHandler(self)
        parser.parse(filepos)
        return self.output
