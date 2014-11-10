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
""" Command interface for MadSpin """
from __future__ import division
import difflib
import logging
import math
import os
import re
import shutil
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT


pjoin = os.path.join

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.iolibs.files as files
import MadSpin.interface_madspin as madspin_interface
import madgraph.various.misc as misc
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.combine_plots as combine_plots

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card 
import MadSpin.decay as madspin


logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print



class ReweightInterface(extended_cmd.Cmd):
    """Basic interface for reweighting operation"""
    
    prompt = 'Reweight>'
    debug_output = 'Reweight_debug'
    
    @misc.mute_logger()
    def __init__(self, event_path=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""
        
        if not event_path:
            cmd_logger.info('************************************************************')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('*               Welcome to Reweight Module                 *')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)
        
        self.model = None
        self.has_standalone_dir = False
        
        self.options = {'curr_dir': os.path.realpath(os.getcwd())}
        
        self.events_file = None
        self.processes = {}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path)
            
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        
        #all the cross-section for convenience
        self.all_cross_section = {}
            
    def do_import(self, inputfile):
        """import the event file"""
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'], 
                                                      os.path.pardir, os.pardir)
            
        
        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)
        
        if inputfile.endswith('.gz'):
            misc.gunzip(inputfile)
            inputfile = inputfile[:-3]

        # Read the banner of the inputfile
        self.lhe_input = lhe_parser.EventFile(os.path.realpath(inputfile))
        if not self.lhe_input.banner:
            value = self.ask("What is the path to banner", 0, [0], "please enter a path", timeout=0)
            self.lhe_input.banner = open(value).read()
        self.banner = self.lhe_input.get_banner()
        
        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        if 'madspin' in self.banner:
            raise self.InvalidCmd('Reweight should be done before running MadSpin')
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if '[' in process:
            msg = 'Reweighting options is valid only for LO events'
            raise Exception, msg            
        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception, msg
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)


    def check_events(self):
        """Check some basic property of the events file"""
        
        sum_of_weight = 0
        sum_of_abs_weight = 0
        negative_event = 0
        positive_event = 0
        
        start = time.time()
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')

            event.check() #check 4 momenta/...

            sum_of_weight += event.wgt
            sum_of_abs_weight += abs(event.wgt)
            if event.wgt < 0 :
                negative_event +=1
            else:
                positive_event +=1
        
        logger.info("total cross-section: %s" % sum_of_weight)
        logger.info("total abs cross-section: %s" % sum_of_abs_weight) 
        logger.info("fraction of negative event %s", negative_event/(negative_event+positive_event))      
        logger.info("total number of events %s", (negative_event+positive_event))
        logger.info("negative event %s", negative_event)
        
        
        
        
    @extended_cmd.debug()
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        if len(args) == 1:
            base_dir = '.'
        else:
            base_dir = args[1]
        
        return self.path_completion(text, base_dir)
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            raise self.InvalidCmd('set command requires at least two argument.')
        
        valid = ['max_weight','seed','curr_dir']
        if args[0] not in self.options and args[0] not in valid:
            raise self.InvalidCmd('Unknown options %s' % args[0])        
    
        if args[0] == 'max_weight':
            try:
                args[1] = float(args[1].replace('d','e'))
            except ValueError:
                raise self.InvalidCmd('second argument should be a real number.')
        
        elif args[0] == 'BW_effect':
            if args[1] in [0, False,'.false.', 'F', 'f', 'False', 'no']:
                args[1] = 0
            elif args[1] in [1, True,'.true.', 'T', 't', 'True', 'yes']:
                args[1] = 1
            else:
                raise self.InvalidCmd('second argument should be either T or F.')
        
        elif args[0] == 'curr_dir':
            if not os.path.isdir(args[1]):
                raise self.InvalidCmd('second argument should be a path to a existing directory')

            
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.lhe_input:
            raise self.InvalidCmd("No events files defined.")

    def help_launch(self):
        """help for the launch command"""
        
        logger.info('''Add to the loaded events a weight associated to a 
        new param_card (to be define). The weight returned is the ratio of the 
        square matrix element by the squared matrix element of production.
        All scale are kept fix for this re-weighting.''')

    #@misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        args = self.split_arg(line)
        self.check_launch(args)

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.has_standalone_dir:
            self.create_standalone_directory()        
            
        ff = open(pjoin(self.me_dir, 'rw_me','Cards', 'param_card.dat'), 'w')
        ff.write(self.banner['slha'])
        ff.close()
        ff = open(pjoin(self.me_dir, 'rw_me','Cards', 'param_card_orig.dat'), 'w')
        ff.write(self.banner['slha'])
        ff.close()        
        cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                   ask=self.ask, pwd=pjoin(self.me_dir,'rw_me'))
        new_card = open(pjoin(self.me_dir, 'rw_me', 'Cards', 'param_card.dat')).read()        

        

        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner:
            if 'type=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup type=\'mg_reweighting\'\s*>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'mg_reweight_(?P<id>\d+)\'>(?P<info>[^<>]*)</weight>', re.S+re.I+re.M)
                mg_rwgt_info = pattern.findall(content)
                maxid = 0
                for i, diff in mg_rwgt_info:
                    if int(i) > maxid:
                        maxid = int(i)
                maxid += 1
                rewgtid = maxid
            else:
                header_rwgt_other = self.banner['initrwgt'] 
                mg_rwgt_info = []
                rewgtid = 1
        else:
            self.banner['initrwgt']  = ''
            header_rwgt_other = ''
            mg_rwgt_info = []
            rewgtid = 1
        
        # add the reweighting in the banner information:
        #starts by computing the difference in the cards.
        s_orig = self.banner['slha']
        s_new = new_card
        old_param = check_param_card.ParamCard(s_orig.splitlines())
        new_param =  check_param_card.ParamCard(s_new.splitlines())
        card_diff = old_param.create_diff(new_param)
        if card_diff == '':
            raise self.InvalidCmd, 'original card and new card are identical'
        mg_rwgt_info.append((str(rewgtid), card_diff))
        
        # re-create the banner.
        self.banner['initrwgt'] = header_rwgt_other
        self.banner['initrwgt'] += '\n<weightgroup type=\'mg_reweighting\'>\n'
        for tag, diff in mg_rwgt_info:
            self.banner['initrwgt'] += '<weight id=\'mg_reweight_%s\'>%s</weight>\n' % \
                                       (tag, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')
            
        output = open( self.lhe_input.name +'rw', 'w')
        
        
        logger.info('starts to compute weight for events with the following modification to the param_card:')
        logger.info(card_diff)
        
        #write the banner to the output file
        self.banner.write(output, close_tag=False)
        # prepare the output file for the weight plot
        if self.mother:
            out_path = pjoin(self.mother.me_dir, 'Events', 'reweight.lhe')
            output2 = open(out_path, 'w')
            self.banner.write(output2, close_tag=False)
            new_banner = banner.Banner(self.banner)
            if not hasattr(self, 'run_card'):
                self.run_card = new_banner.charge_card('run_card')
            self.run_card['run_tag'] = 'reweight_%s' % rewgtid
            new_banner['slha'] = s_new   
            del new_banner['initrwgt']
            #ensure that original banner is kept untouched
            assert new_banner['slha'] != self.banner['slha']
            assert 'initrwgt' in self.banner 
            ff = open(pjoin(self.mother.me_dir,'Events',self.mother.run_name, '%s_%s_banner.txt' % \
                          (self.mother.run_name, self.run_card['run_tag'])),'w') 
            new_banner.write(ff)
            ff.close()
        
        # Loop over all events
        tag_name = 'mg_reweight_%s' % rewgtid
        start = time.time()
        cross = 0
        
        os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'y'
        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),1000)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')


            weight = self.calculate_weight(event)
            cross += weight
            event.reweight_data[tag_name] = weight
            #write this event with weight
            output.write(str(event))
            if self.mother:
                event.wgt = weight
                event.reweight_data = {}
                output2.write(str(event))

        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))        
        
        output.write('</LesHouchesEvents>\n')
        output.close()
        os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'n'
        if self.mother:
            output2.write('</LesHouchesEvents>\n')
            output2.close()        
            # add output information
            if hasattr(self.mother, 'results'):
                run_name = self.mother.run_name
                results = self.mother.results
                results.add_run(run_name, self.run_card, current=True)
                results.add_detail('nb_event', event_nb+1)
                results.add_detail('cross', cross)
                results.add_detail('error', 'nan')
                self.mother.create_plot(mode='reweight', event_path=output2.name,
                                        tag=self.run_card['run_tag'])
                #modify the html output to add the original run
                if 'plot' in results.current.reweight:
                    html_dir = pjoin(self.mother.me_dir, 'HTML', run_name)
                    td = pjoin(self.mother.options['td_path'], 'td') 
                    MA = pjoin(self.mother.options['madanalysis_path'])
                    path1 = pjoin(html_dir, 'plots_parton')
                    path2 = pjoin(html_dir, 'plots_%s' % self.run_card['run_tag'])
                    outputplot = path2
                    combine_plots.merge_all_plots(path2, path1, outputplot, td, MA)
                #results.update_status(level='reweight')
                #results.update(status, level, makehtml=True, error=False)
                
                #old_name = self.mother.results.current['run_name']
                #new_run = '%s_rw_%s' % (old_name, rewgtid)
                #self.mother.results.add_run( new_run, self.run_card)
                #self.mother.results.add_detail('nb_event', event_nb+1)
                #self.mother.results.add_detail('cross', cross)
                #self.mother.results.add_detail('error', 'nan')
                #self.mother.do_plot('%s -f' % new_run)
                #self.mother.update_status('Reweight %s done' % rewgtid, 'madspin')
                #self.mother.results.def_current(old_name)
                #self.run_card['run_tag'] = self.run_card['run_tag'][9:]
                #self.mother.run_name = old_name
        self.lhe_input.close()
        files.mv(output.name, self.lhe_input.name)
        logger.info('Event %s have now the additional weight' % self.lhe_input.name)
        logger.info('new cross-section is : %g pb' % cross)
        self.terminate_fortran_executables(new_card_only=True)
        #store result
        self.all_cross_section[rewgtid] = cross
        


            
    def calculate_weight(self, event):
        
        event.parse_reweight()
        w_orig = self.calculate_matrix_element(event, 0)
        w_new =  self.calculate_matrix_element(event, 1)
        
        return w_new/w_orig*event.wgt
    
    
    def calculate_matrix_element(self, event, hypp_id):
        """routine to return the matrix element"""

        tag, order = event.get_tag_and_order()
        orig_order, Pdir = self.id_to_path[tag]

        run_id = (tag, hypp_id)

        if run_id in self.calculator:
            external = self.calculator[run_id]
            self.calculator_nbcall[run_id] += 1
        else:
            # create the executable for this param_card            

            tmpdir = pjoin(self.me_dir,'rw_me', 'SubProcesses', Pdir)
            executable_prod="./check"
            if not os.path.exists(pjoin(tmpdir, 'check')):
                misc.compile( cwd=tmpdir)
            external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, 
                                                      stderr=STDOUT, cwd=tmpdir)
            self.calculator[run_id] = external 
            self.calculator_nbcall[run_id] = 1       
            # set the param_card
            if hypp_id == 1:
                external.stdin.write('param_card.dat\n')
            elif hypp_id == 0:
                external.stdin.write('param_card_orig.dat\n')
        #import the value of alphas
        external.stdin.write('%g\n' % event.aqcd)
        stdin_text = event.get_momenta_str(orig_order)
        external.stdin.write(stdin_text)
        me_value = external.stdout.readline()
        try: 
            me_value = float(me_value)
        except Exception:
            print 'ZERO DETECTED'
            print stdin_text
            print me_value
            os.system('lsof -p %s' % external.pid)
            me_value = 0
        
        if len(self.calculator) > 100:
            logger.debug('more than 100 calculator. Perform cleaning')
            nb_calls = self.calculator_nbcall.values()
            nb_calls.sort()
            cut = max([nb_calls[len(nb_calls)//2], 0.001 * nb_calls[-1]])
            for key, external in list(self.calculator.items()):
                nb = self.calculator_nbcall[key]
                if nb < cut:
                    external.stdin.close()
                    external.stdout.close()
                    external.terminate()
                    del self.calculator[key]
                    del self.calculator_nbcall[key]
                else:
                    self.calculator_nbcall[key] = self.calculator_nbcall[key] //10
        
        return me_value
    
    def terminate_fortran_executables(self, new_card_only=False):
        """routine to terminate all fortran executables"""

        for (mode, production) in dict(self.calculator):
            
            if new_card_only and production == 0:
                continue
            external = self.calculator[(mode, production)]
            external.stdin.close()
            external.stdout.close()
            external.terminate()            
            del self.calculator[(mode, production)]
    
    def do_quit(self, line):
        
        if 'init' in self.banner:
            cross = 0 
            error = 0
            for line in self.banner['init'].split('\n'):
                split = line.split()
                if len(split) == 4:
                    cross, error = float(split[0]), float(split[1])
            logger.info('Original cross-section: %s +- %s pb' % (cross, error))
        
        logger.info('Computed cross-section:')
        keys = self.all_cross_section.keys()
        keys.sort()
        for key in keys:
            logger.info('%s : %s' % (key,self.all_cross_section[key]))  
        self.terminate_fortran_executables()
            
    def __del__(self):
        self.do_quit('')

    
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        



        
    
    @misc.mute_logger()
    def create_standalone_directory(self):
        """generate the various directory for the weight evaluation"""
        
        # 0. clean previous run ------------------------------------------------
        path_me = self.me_dir
        try:
            shutil.rmtree(pjoin(path_me,'rw_me'))
        except Exception: 
            pass

        # 1. Load model---------------------------------------------------------  
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
            elif line.startswith('define'):
                try:
                    self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except Exception:
                    pass 
                          
        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            mg_names = False
        else:
            mg_names = True
        model_name = self.banner.get('proc_card', 'model')
        if model_name:
            self.load_model(model_name, mg_names, complex_mass)
        else:
            raise self.InvalidCmd('Only UFO model can be loaded in this module.')
        
        mgcmd = self.mg5cmd
        modelpath = self.model.get('modelpath')
        if os.path.basename(modelpath) != mgcmd._curr_model['name']:
            name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
            if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])
            
        commandline="import model %s " % modelpath
        mgcmd.exec_cmd(commandline)
        
        # 2. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search('^\s*add\s+process', line)]   
        mgcmd.exec_cmd("set group_subprocesses False")

        logger.info('generating the square matrix element for reweighting')
        start = time.time()
        commandline=''
        for proc in processes:
            if '[' not in proc:
                commandline+="add process %s ;" % proc
            else:
                raise self.InvalidCmd('NLO processes can\'t be reweight')
        
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output standalone_rw %s' % pjoin(path_me,'rw_me')
        mgcmd.exec_cmd(commandline, precmd=True)        
        logger.info('Done %.4g' % (time.time()-start))
        self.has_standalone_dir = True

        
        # 3. Store id to directory information ---------------------------------
        matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
        
        self.id_to_path = {}
        for me in matrix_elements:
            for proc in me.get('processes'):
                initial = []    #filled in the next line
                final = [l.get('id') for l in proc.get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
                order = (initial, final)
                tag = proc.get_initial_final_ids()
                decay_finals = proc.get_final_ids_after_decay()

                if tag[1] != decay_finals:
                    order = (initial, list(decay_finals))
                    decay_finals.sort()
                    tag = (tag[0], tuple(decay_finals))
                Pdir = pjoin(path_me, 'rw_me', 'SubProcesses', 
                                  'P%s' % me.get('processes')[0].shell_string())
                assert os.path.exists(Pdir), "Pdir %s do not exists" % Pdir                        
                if tag in self.id_to_path:
                    if not Pdir == self.id_to_path[tag][1]:
                        misc.sprint(tag, Pdir, self.id_to_path[tag][1])
                        raise self.InvalidCmd, '2 different process have the same final states. This module can not handle such situation'
                    else:
                        continue
                self.id_to_path[tag] = [order, Pdir]


    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False

        logger.info('detected model: %s. Loading...' % name)
        model_path = name

        # Import model
        base_model = import_ufo.import_model(name, decay=False)


        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        if complex_mass:
            base_model.change_mass_to_complex_scheme()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()
        


    


        
