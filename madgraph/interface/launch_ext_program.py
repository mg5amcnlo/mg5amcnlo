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

import glob
import logging
import os
import pydoc
import re
import sys
import subprocess
import thread
import time

import madgraph.iolibs.files as files
import madgraph.interface.extended_cmd as cmd
import madgraph.interface.madevent_interface as me_cmd
import madgraph.various.misc as misc
import madgraph.various.process_checks as process_checks
import madgraph.various.banner as banner_mod

from madgraph import MG4DIR, MG5DIR, MadGraph5Error
from madgraph.iolibs.files import cp
pjoin = os.path.join


logger = logging.getLogger('cmdprint.ext_program')

class ExtLauncher(object):
    """ Generic Class for executing external program """
    
    program_dir = ''
    executable = ''  # path from program_dir
    
    force = False
    
    def __init__(self, cmd, running_dir, card_dir='', **options):
        """ initialize an object """
        
        self.running_dir = running_dir
        self.card_dir = os.path.join(self.running_dir, card_dir)
        self.cmd_int = cmd
        #include/overwrite options
        for key,value in options.items():
            setattr(self, key, value)
            
        self.cards = [] # files can be modified (path from self.card_dir)
            
    def run(self):
        """ execute the main code """

        self.prepare_run()        
        for card in self.cards:
            self.treat_input_file(card, default = 'n')

        self.launch_program()

        
    def prepare_run(self):
        """ aditional way to prepare the run"""
        pass
    
    def launch_program(self):
        """launch the main program"""
        subprocess.call([self.executable], cwd=self.running_dir)
    
    def edit_file(self, path):
        """edit a file"""

        path = os.path.realpath(path)
        open_file(path)
    

    # Treat Nicely the timeout
    def timeout_fct(self,timeout):
        if timeout:
            # avoid to always wait a given time for the next answer
            self.force = True
   
    def ask(self, question, default, choices=[], path_msg=None):
        """nice handling of question"""
     
        if not self.force:
            return self.cmd_int.ask(question, default, choices=choices, 
                                path_msg=path_msg, fct_timeout=self.timeout_fct)
        else:
            return str(default)
         
        
    def treat_input_file(self, filename, default=None, msg=''):
        """ask to edit a file"""
        
        if msg == '' and filename == 'param_card.dat':
            msg = \
            "WARNING: If you edit this file don\'t forget to consistently "+\
            "modify the different parameters,\n especially the width of all "+\
            "particles."
                                         
        if not self.force:
            if msg:  print msg
            question = 'Do you want to edit file: %(card)s?' % {'card':filename}
            choices = ['y', 'n']
            path_info = 'path of the new %(card)s' % {'card':os.path.basename(filename)}
            ans = self.ask(question, default, choices, path_info)
        else:
            ans = default
        
        if ans == 'y':
            path = os.path.join(self.card_dir, filename)
            self.edit_file(path)
        elif ans == 'n':
            return
        else:
            path = os.path.join(self.card_dir, filename)
            files.cp(ans, path)
            
class MadLoopLauncher(ExtLauncher):
    """ A class to launch a simple Standalone test """
    
    def __init__(self, cmd_int, running_dir, **options):
        """ initialize the StandAlone Version """
        
        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **options)
        self.cards = ['param_card.dat','MadLoopParams.dat']

    def prepare_run(self):
        """ Possible preparatory actions to take."""
        pass

    def treat_input_file(self, filename, default=None, msg='', dir_path=None):
        """ask to edit a file"""

        if filename == 'PS.input':
            if not self.force:
                if msg!='':  print msg
                question = 'Do you want to specify the Phase-Space point: %(card)s?' % {'card':filename}
                choices = ['y', 'n']
                path_info = 'path of the PS.input file'
                ans = self.ask(question, default, choices, path_info)
            else:
                ans = default
            if ans == 'y':
                if not os.path.isfile(os.path.join(dir_path,'PS.input')):
                    PSfile = open(os.path.join(dir_path,'PS.input'), 'w')
                    if not os.path.isfile(os.path.join(dir_path,'result.dat')):
                        PSfile.write('\n'.join([' '.join(['%.16E'%0.0 for \
                                        pi in range(4)]) for pmom in range(1)]))
                    else:
                        default_ps = process_checks.LoopMatrixElementEvaluator.\
                            parse_check_output(file(os.path.join(dir_path,\
                                          'result.dat')),format='dict')['res_p']
                        PSfile.write('\n'.join([' '.join(['%.16E'%pi for pi \
                                             in pmom]) for pmom in default_ps]))                     
                    PSfile.write("\n\nEach line number 'i' like the above one sets"+\
                            " the momentum of particle number i, \nordered like in"+\
                            " the process definition. The format is (E,px,py,pz).")
                    PSfile.close()       
                self.edit_file(os.path.join(dir_path,'PS.input'))
        else:
            super(MadLoopLauncher,self).treat_input_file(filename,default,msg)
            if filename == 'MadLoopParams.dat':
                # Make sure to update the changes
                MadLoopparam = banner_mod.MadLoopParam(
                               os.path.join(self.card_dir, 'MadLoopParams.dat'))   
                # Unless user asked for it, don't doublecheck the helicity filter.
                MadLoopparam.set('DoubleCheckHelicityFilter', False, 
                                                             ifnotdefault=False)
                MadLoopparam.write(os.path.join(self.card_dir,os.path.pardir, 
                                           'SubProcesses', 'MadLoopParams.dat'))

    def launch_program(self):
        """launch the main program"""
        evaluator = process_checks.LoopMatrixElementTimer
        sub_path = os.path.join(self.running_dir, 'SubProcesses')
        for path in os.listdir(sub_path):
            if path.startswith('P') and \
                                    os.path.isdir(os.path.join(sub_path, path)):
                shell_name = path.split('_')[1]+' > '+path.split('_')[2]
                curr_path = os.path.join(sub_path, path)
                infos = {}
                logger.info("Initializing process %s."%shell_name)
                nps = me_cmd.MadLoopInitializer.run_initialization(
                                curr_path, sub_path, infos,
                                req_files = ['HelFilter.dat','LoopFilter.dat'])
                if nps == None:
                    raise MadGraph5Error,"MadLoop could not initialize the process %s"\
                      %shell_name
                logger.debug(("MadLoop initialization performed for %s"+\
                        " using %d PS points (%s)")\
                        %(shell_name,abs(nps),\
                    'double precision' if nps>0 else 'quadruple precision'))
                # Ask if the user wants to edit the PS point.
                self.treat_input_file('PS.input', default='n', 
                  msg='Phase-space point for process %s.'%shell_name,\
                                                             dir_path=curr_path)
                # We use mu_r=-1.0 to use the one defined by the user in the
                # param_car.dat
                me_cmd.MadLoopInitializer.fix_PSPoint_in_check(sub_path, 
                  read_ps = os.path.isfile(os.path.join(curr_path, 'PS.input')),
                  npoints = 1, mu_r=-1.0)
                # check
                t1, t2, ram_usage = me_cmd.MadLoopInitializer.make_and_run(curr_path)
                if t1==None or t2==None:
                    raise MadGraph5Error,"Error while running process %s."\
                                                                     %shell_name
                try:
                    rFile=open(os.path.join(curr_path,'result.dat'), 'r')
                except IOError:
                    raise MadGraph5Error,"Could not find result file %s."%\
                                       str(os.path.join(curr_path,'result.dat'))
                # The result are returned as a dictionary.
                result = evaluator.parse_check_output(rFile,format='dict')
                for line in self.format_res_string(result, shell_name):
                    if isinstance(line, str):
                        logger.info(line)
                    elif isinstance(line,tuple):
                        logger.info(line[0],line[1])

    def format_res_string(self, res, shell_name):
        """ Returns a good-looking string presenting the results.
        The argument the tuple ((fin,born,spole,dpole,me_pow), p_out)."""
        
        main_color='$MG:color:BLUE'
        
        def special_float_format(float):
            return '%s%.16e'%('' if float<0.0 else ' ',float)
        
        so_order_names = res['Split_Orders_Names']
        
        def format_so_orders(so_orders):
            return ' '.join(['%s=%d'%(so_order_names[i],so_orders[i]) for i in
                                                         range(len(so_orders))])

        ASCII_bar = ('|'+''.join(['='*96]),main_color)
        
        ret_code_h = res['return_code']//100
        ret_code_t = (res['return_code']-100*ret_code_h)//10
        ret_code_u = res['return_code']%10
        StabilityOutput=[]
        if ret_code_h==1:
            if ret_code_t==3 or ret_code_t==4:
                StabilityOutput.append('| Unknown numerical stability because '+\
                                      'MadLoop is in the initialization stage.')
            else:
                StabilityOutput.append('| Unknown numerical stability, check '+\
                                        'CTRunMode value in MadLoopParams.dat.')
        elif ret_code_h==2:
            StabilityOutput.append('| Stable kinematic configuration (SPS).')
        elif ret_code_h==3:
            StabilityOutput.append('| Unstable kinematic configuration (UPS).')
            StabilityOutput.append('| Quadruple precision rescue successful.')            
        elif ret_code_h==4:
            StabilityOutput.append('| Exceptional kinematic configuration (EPS).')
            StabilityOutput.append('| Both double and quadruple precision'+\
                                                  ' computations are unstable.')
        
        if ret_code_t==2 or ret_code_t==4:
            StabilityOutput.append('| Quadruple precision was used for this'+\
                                                                 'computation.')
        if ret_code_h!=1:
            if res['accuracy']>0.0:
                StabilityOutput.append('| Estimated relative accuracy = %.1e'\
                                                               %res['accuracy'])
            elif res['accuracy']==0.0:
                StabilityOutput.append('| Estimated relative accuracy = %.1e'\
                             %res['accuracy']+' (i.e. beyond double precision)')
            else:
                StabilityOutput.append('| Estimated accuracy could not be '+\
                                              'computed for an unknown reason.')

        PS_point_spec = ['|| Phase-Space point specification (E,px,py,pz)','|']
        PS_point_spec.append('\n'.join(['| '+' '.join(['%s'%\
           special_float_format(pi) for pi in pmom]) for pmom in res['res_p']]))
        PS_point_spec.append('|')
        
        str_lines=[]
        
        notZeroBorn=True
        if res['export_format']!='LoopInduced' and len(so_order_names) and \
                                     len([1 for k in res['Born_kept'] if k])==0:
            notZeroBorn = False
            str_lines.append(
("|  /!\\ There is no Born contribution for the squared orders specified in "+
                                  "the process definition/!\\",'$MG:color:RED'))
        
        if res['export_format']=='Default' and notZeroBorn:
            str_lines.extend(['\n',ASCII_bar,
  ('|| Results for process %s'%shell_name,main_color),
  ASCII_bar]+PS_point_spec+StabilityOutput+[
  '|',
  ('|| Total(*) Born contribution (GeV^%d):'%res['gev_pow'],main_color),
  ('|    Born        = %s'%special_float_format(res['born']),main_color),
  ('|| Total(*) virtual contribution normalized with born*alpha_S/(2*pi):',main_color),
  ('|    Finite      = %s'%special_float_format(res['finite']),main_color),
  ('|    Single pole = %s'%special_float_format(res['1eps']),main_color),
  ('|    Double pole = %s'%special_float_format(res['2eps']),main_color)])
        elif res['export_format']=='LoopInduced' and notZeroBorn:
            str_lines.extend(['\n',ASCII_bar,
  ('|| Results for process %s (Loop-induced)'%shell_name,main_color),
  ASCII_bar]+PS_point_spec+StabilityOutput+[
  '|',
  ('|| Loop amplitude squared, must be finite:',main_color),
  ('|    Finite      = %s'%special_float_format(res['finite']),main_color),
  '|(| Pole residues, indicated only for checking purposes: )',
  '|(    Single pole = %s )'%special_float_format(res['1eps']),
  '|(    Double pole = %s )'%special_float_format(res['2eps'])])

        if (len(res['Born_SO_Results'])+len(res['Born_SO_Results']))>0:
            if notZeroBorn:
                str_lines.append(
    ("|  (*) The results above sum all starred contributions below",main_color))

        str_lines.append('|')
        if not notZeroBorn:
            str_lines.append(
("|  The Born contributions below are computed but do not match these squared "+
                                               "orders constraints",main_color))

        if len(res['Born_SO_Results'])==1:
            str_lines.append('|| All Born contributions are of split orders *(%s)'\
                                %format_so_orders(res['Born_SO_Results'][0][0]))
        elif len(res['Born_SO_Results'])>1:
            for i, bso_contrib in enumerate(res['Born_SO_Results']):
                str_lines.append('|| Born contribution of split orders %s(%s) = %s'\
                                           %('*' if res['Born_kept'][i] else ' ',
                                               format_so_orders(bso_contrib[0]),
                                  special_float_format(bso_contrib[1]['BORN'])))
        
        if len(so_order_names):
            str_lines.append('|')

        if len(res['Loop_SO_Results'])==1:
            str_lines.append('|| All virtual contributions are of split orders *(%s)'\
                                %format_so_orders(res['Loop_SO_Results'][0][0]))
        elif len(res['Loop_SO_Results'])>1:
            if not notZeroBorn:
                str_lines.append(
    ("|  The coupling order combinations matching the squared order"+
                              " constraints are marked with a star",main_color))
            for i, lso_contrib in enumerate(res['Loop_SO_Results']):
                str_lines.append('|| Virtual contribution of split orders %s(%s):'\
                                        %('*' if res['Loop_kept'][i] else ' ',
                                              format_so_orders(lso_contrib[0])))
                str_lines.append('|    Accuracy    =  %.1e'%\
                                                         lso_contrib[1]['ACC']),
                str_lines.append('|    Finite      = %s'%\
                                   special_float_format(lso_contrib[1]['FIN'])),
                if res['export_format']=='LoopInduced':
                    str_lines.append('|(    Single pole = %s )'%\
                                   special_float_format(lso_contrib[1]['1EPS']))
                    str_lines.append('|(    Double pole = %s )'%\
                                   special_float_format(lso_contrib[1]['2EPS']))
                else:
                    str_lines.append('|    Single pole = %s'%\
                                   special_float_format(lso_contrib[1]['1EPS']))
                    str_lines.append('|    Double pole = %s'%\
                                   special_float_format(lso_contrib[1]['2EPS']))              
        str_lines.extend([ASCII_bar,'\n'])

        return str_lines

class SALauncher(ExtLauncher):
    """ A class to launch a simple Standalone test """
    
    def __init__(self, cmd_int, running_dir, **options):
        """ initialize the StandAlone Version"""
        
        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **options)
        self.cards = ['param_card.dat']

    
    def launch_program(self):
        """launch the main program"""
        sub_path = os.path.join(self.running_dir, 'SubProcesses')
        for path in os.listdir(sub_path):
            if path.startswith('P') and \
                                   os.path.isdir(os.path.join(sub_path, path)):
                cur_path =  os.path.join(sub_path, path)
                # make
                misc.compile(cwd=cur_path, mode='unknown')
                # check
                subprocess.call(['./check'], cwd=cur_path)

class MWLauncher(ExtLauncher):
    """ A class to launch a simple Standalone test """
    
    
    def __init__(self, cmd_int, running_dir, **options):
        """ initialize the StandAlone Version"""
        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **options)
        self.options = cmd_int.options

    def launch_program(self):
        """launch the main program"""
        
        import madgraph.interface.madweight_interface as MW
        # Check for number of cores if multicore mode
        mode = str(self.cluster)
        nb_node = 1
        if mode == "2":
            import multiprocessing
            max_node = multiprocessing.cpu_count()
            if max_node == 1:
                logger.warning('Only one core is detected on your computer! Pass in single machine')
                self.cluster = 0
                self.launch_program()
                return
            elif max_node == 2:
                nb_node = 2
            elif not self.force:
                nb_node = self.ask('How many core do you want to use?', max_node, range(2,max_node+1))
            else:
                nb_node=max_node
                
        import madgraph.interface.madevent_interface as ME
        
        stdout_level = self.cmd_int.options['stdout_level']
        if self.shell:
            usecmd = MW.MadWeightCmdShell(me_dir=self.running_dir, options=self.options)
        else:
            usecmd = MW.MadWeightCmd(me_dir=self.running_dir, options=self.options)
            usecmd.pass_in_web_mode()
        #Check if some configuration were overwritten by a command. If so use it    
        set_cmd = [l for l in self.cmd_int.history if l.strip().startswith('set')]
        for line in set_cmd:
            try:
                usecmd.do_set(line[3:], log=False)
            except Exception:
                pass
            
        usecmd.do_set('stdout_level %s'  % stdout_level,log=False)
        #ensure that the logger level 
        launch = self.cmd_int.define_child_cmd_interface(
                     usecmd, interface=False)

        command = 'launch'
        if mode == "1":
            command += " --cluster"
        elif mode == "2":
            command += " --nb_core=%s" % nb_node
        
        if self.force:
            command+= " -f"
        if self.laststep:
            command += ' --laststep=%s' % self.laststep
        
        try:
            os.remove('ME5_debug')
        except:
           pass
        launch.run_cmd(command)
        launch.run_cmd('quit')
        
        if os.path.exists('ME5_debug'):
            return True
        


class aMCatNLOLauncher(ExtLauncher):
    """A class to launch MadEvent run"""
    
    def __init__(self, running_dir, cmd_int, run_mode='', unit='pb', **option):
        """ initialize the StandAlone Version"""

        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **option)
        #self.executable = os.path.join('.', 'bin','generate_events')

        self.options = option
        assert hasattr(self, 'cluster')
        assert hasattr(self, 'multicore')
        assert hasattr(self, 'name')
        assert hasattr(self, 'shell')

        self.unit = unit
        self.run_mode = run_mode
        
        if self.cluster or option['cluster']:
            self.cluster = 1
        if self.multicore or option['multicore']:
            self.cluster = 2
        
        self.cards = []

        # Assign a valid run name if not put in options
        if self.name == '':
            self.name = me_cmd.MadEventCmd.find_available_run_name(self.running_dir)
    
    def launch_program(self):
        """launch the main program"""
        
        # Check for number of cores if multicore mode
        mode = str(self.cluster)
        nb_node = 1
        if mode == "2":
            import multiprocessing
            max_node = multiprocessing.cpu_count()
            if max_node == 1:
                logger.warning('Only one core is detected on your computer! Pass in single machine')
                self.cluster = 0
                self.launch_program()
                return
            elif max_node == 2:
                nb_node = 2
            elif not self.force:
                nb_node = self.ask('How many cores do you want to use?', max_node, range(2,max_node+1))
            else:
                nb_node=max_node
                
        import madgraph.interface.amcatnlo_run_interface as run_int
        
        if hasattr(self, 'shell') and self.shell:
            usecmd = run_int.aMCatNLOCmdShell(me_dir=self.running_dir, options = self.cmd_int.options)
        else:
            usecmd = run_int.aMCatNLOCmd(me_dir=self.running_dir, options = self.cmd_int.options)
        
        #Check if some configuration were overwritten by a command. If so use it    
        set_cmd = [l for l in self.cmd_int.history if l.strip().startswith('set')]
        all_options = usecmd.options_configuration.keys() +  usecmd.options_madgraph.keys() + usecmd.options_madevent.keys()
        for line in set_cmd:
            arg = line.split()
            if arg[1] not in all_options:
                continue
            try:
                usecmd.exec_cmd(line)
            except Exception, error:
                misc.sprint('Command %s fails with msg: %s'%(str(line), \
                                                                    str(error)))
                pass
        launch = self.cmd_int.define_child_cmd_interface(
                     usecmd, interface=False)
        #launch.me_dir = self.running_dir
        option_line = ' '.join([' --%s' % opt for opt in self.options.keys() \
                if self.options[opt] and not opt in ['cluster', 'multicore', 'name', 'appl_start_grid','shell']])
        if self.options['name']:
            option_line += ' --name %s' %  self.options['name']
        if 'appl_start_grid' in self.options and  self.options['appl_start_grid']:
            option_line += ' --appl_start_grid %s' %  self.options['appl_start_grid']
        command = 'launch ' + self.run_mode + ' ' + option_line

        if mode == "1":
            command += " -c"
        elif mode == "2":
            command += " -m" 
            usecmd.nb_core = int(nb_node)
        try:
            os.remove('ME5_debug')
        except:
           pass
        launch.run_cmd(command)
        launch.run_cmd('quit')
        
        

                
        
class MELauncher(ExtLauncher):
    """A class to launch MadEvent run"""
    
    def __init__(self, running_dir, cmd_int , unit='pb', **option):
        """ initialize the StandAlone Version"""

        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **option)
        #self.executable = os.path.join('.', 'bin','generate_events')
        self.pythia = cmd_int.options['pythia-pgs_path']
        self.delphes = cmd_int.options['delphes_path'],
        self.options = cmd_int.options

        assert hasattr(self, 'cluster')
        assert hasattr(self, 'multicore')
        assert hasattr(self, 'name')
        assert hasattr(self, 'shell')

        self.unit = unit
        
        if self.cluster:
            self.cluster = 1
        if self.multicore:
            self.cluster = 2
        
        self.cards = []

        # Assign a valid run name if not put in options
        if self.name == '':
            self.name = me_cmd.MadEventCmd.find_available_run_name(self.running_dir)
    
    def launch_program(self):
        """launch the main program"""
        
        # Check for number of cores if multicore mode
        mode = str(self.cluster)
        nb_node = 1
        if mode == "2":
            import multiprocessing
            max_node = multiprocessing.cpu_count()
            if max_node == 1:
                logger.warning('Only one core is detected on your computer! Pass in single machine')
                self.cluster = 0
                self.launch_program()
                return
            elif max_node == 2:
                nb_node = 2
            elif not self.force:
                nb_node = self.ask('How many cores do you want to use?', max_node, range(2,max_node+1))
            else:
                nb_node=max_node
                
        import madgraph.interface.madevent_interface as ME
        
        stdout_level = self.cmd_int.options['stdout_level']
        if self.shell:
            usecmd = ME.MadEventCmdShell(me_dir=self.running_dir, options=self.options)
        else:
            usecmd = ME.MadEventCmd(me_dir=self.running_dir, options=self.options)
            usecmd.pass_in_web_mode()
        #Check if some configuration were overwritten by a command. If so use it    
        set_cmd = [l for l in self.cmd_int.history if l.strip().startswith('set')]
        all_options = usecmd.options_configuration.keys() +  usecmd.options_madgraph.keys() + usecmd.options_madevent.keys()
        for line in set_cmd:
            arg = line.split()
            if arg[1] not in all_options:
                continue
            try:
                usecmd.do_set(line[3:], log=False)
            except usecmd.InvalidCmd:
                pass
        usecmd.do_set('stdout_level %s'  % stdout_level,log=False)
        #ensure that the logger level 
        launch = self.cmd_int.define_child_cmd_interface(
                     usecmd, interface=False)
        #launch.me_dir = self.running_dir
        if self.unit == 'pb':
            command = 'generate_events %s' % self.name
        else:
            warning_text = '''\
This command will create a new param_card with the computed width. 
This param_card makes sense only if you include all processes for
the computation of the width. For more efficient width computation:
see arXiv:1402.1178.'''
            logger.warning(warning_text)

            command = 'generate_events %s' % self.name
        if mode == "1":
            command += " --cluster"
        elif mode == "2":
            command += " --nb_core=%s" % nb_node
        
        if self.force:
            command+= " -f"
        
        if self.laststep:
            command += ' --laststep=%s' % self.laststep
        if self.reweight:
            command += ' -R '
        if self.madspin:
            command += ' -M '
        
        
        try:
            os.remove('ME5_debug')
        except:
           pass

        launch.run_cmd(command)
        launch.run_cmd('quit')
        
        if os.path.exists('ME5_debug'):
            return True
        
        # Display the cross-section to the screen
        path = os.path.join(self.running_dir, 'SubProcesses', 'results.dat') 
        if not os.path.exists(path):
            logger.error('Generation failed (no results.dat file found)')
            return
        fsock = open(path)
        line = fsock.readline()
        cross, error = line.split()[0:2]
        
        logger.info('more information in %s' 
                                 % os.path.join(self.running_dir, 'index.html'))
                

class Pythia8Launcher(ExtLauncher):
    """A class to launch Pythia8 run"""
    
    def __init__(self, running_dir, cmd_int, **option):
        """ initialize launching Pythia 8"""

        running_dir = os.path.join(running_dir, 'examples')
        ExtLauncher.__init__(self, cmd_int, running_dir, '.', **option)
        self.cards = []
    
    def prepare_run(self):
        """ ask for pythia-pgs/delphes run """

        # Find all main_model_process.cc files
        date_file_list = []
        for file in misc.glob('main_*_*.cc', self.running_dir):
            # retrieves the stats for the current file as a tuple
            # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
            # the tuple element mtime at index 8 is the last-modified-date
            stats = os.stat(file)
            # create tuple (year yyyy, month(1-12), day(1-31), hour(0-23), minute(0-59), second(0-59),
            # weekday(0-6, 0 is monday), Julian day(1-366), daylight flag(-1,0 or 1)) from seconds since epoch
            # note:  this tuple can be sorted properly by date and time
            lastmod_date = time.localtime(stats[8])
            date_file_list.append((lastmod_date, os.path.split(file)[-1]))

        if not date_file_list:
            raise MadGraph5Error, 'No Pythia output found'
        # Sort files according to date with newest first
        date_file_list.sort()
        date_file_list.reverse()
        files = [d[1] for d in date_file_list]
        
        answer = ''
        answer = self.ask('Select a main file to run:', files[0], files)

        self.cards.append(answer)
    
        self.executable = self.cards[-1].replace(".cc","")

        # Assign a valid run name if not put in options
        if self.name == '':
            for i in range(1000):
                path = os.path.join(self.running_dir, '',
                                    '%s_%02i.log' % (self.executable, i))
                if not os.path.exists(path):
                    self.name = '%s_%02i.log' % (self.executable, i)
                    break
        
        if self.name == '':
            raise MadGraph5Error, 'too many runs in this directory'

        # Find all exported models
        models = misc.glob("Processes_*", pjoin(self.running_dir,os.path.pardir))
        models = [os.path.split(m)[-1].replace("Processes_","") for m in models]
        # Extract model name from executable
        models.sort(key=len)
        models.reverse()
        model_dir = ""
        for model in models:
            if self.executable.replace("main_", "").startswith(model):
                model_dir = "Processes_%s" % model
                break
        if model_dir:
            self.model = model
            self.model_dir = os.path.realpath(os.path.join(self.running_dir,
                                                           os.path.pardir,
                                                           model_dir))
            self.cards.append(os.path.join(self.model_dir,
                                           "param_card_%s.dat" % model))
        
    def launch_program(self):
        """launch the main program"""

        # Make pythia8
        print "Running make for pythia8 directory"
        misc.compile(cwd=os.path.join(self.running_dir, os.path.pardir), mode='cpp')
        if self.model_dir:
            print "Running make in %s" % self.model_dir
            misc.compile(cwd=self.model_dir, mode='cpp')
        # Finally run make for executable
        makefile = self.executable.replace("main_","Makefile_")
        print "Running make with %s" % makefile
        misc.compile(arg=['-f', makefile], cwd=self.running_dir, mode='cpp')
        
        print "Running " + self.executable
        
        output = open(os.path.join(self.running_dir, self.name), 'w')
        if not self.executable.startswith('./'):
            self.executable = os.path.join(".", self.executable)
        subprocess.call([self.executable], stdout = output, stderr = output,
                        cwd=self.running_dir)
        
        # Display the cross-section to the screen
        path = os.path.join(self.running_dir, self.name) 
        pydoc.pager(open(path).read())

        print "Output of the run is found at " + \
              os.path.realpath(os.path.join(self.running_dir, self.name))

# old compatibility shortcut
open_file = misc.open_file


