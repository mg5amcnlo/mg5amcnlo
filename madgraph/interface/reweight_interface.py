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
import sys
import tempfile
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
import madgraph.various.cluster as cluster
import madgraph.fks.fks_common as fks_common

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card 
import MadSpin.decay as madspin


logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

# global to check which f2py module have been already loaded. (to avoid border effect)
dir_to_f2py_free_mod = {}
nb_f2py_module = 0 # each time the process/model is changed this number is modified to 
                   # forced the python module to re-create an executable




class ReweightInterface(extended_cmd.Cmd):
    """Basic interface for reweighting operation"""
    
    prompt = 'Reweight>'
    debug_output = 'Reweight_debug'
    
    @misc.mute_logger()
    def __init__(self, event_path=None, allow_madspin=False, *completekey, **stdin):
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
        self.second_model = None
        self.second_process = None
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        self.output_type = "default"
        self.helicity_reweighting = True
        self.rwgt_dir = None
        
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path, allow_madspin=allow_madspin)
            
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        
        #all the cross-section for convenience
        self.all_cross_section = {}
            
    def do_import(self, inputfile, allow_madspin=False):
        """import the event file"""

        args = self.split_arg(inputfile)
        if not args:
            return self.InvalidCmd, 'import requires arguments'
        
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
        
        #get original cross-section/error
        if 'init' not in self.banner:
            self.orig_cross = (0,0)
            #raise self.InvalidCmd('Event file does not contain init information')
        else:
            for line in self.banner['init'].split('\n'):
                    split = line.split()
                    if len(split) == 4:
                        cross, error = float(split[0]), float(split[1])
            self.orig_cross = (cross, error)
        
        
        
        # Check the validity of the banner:
        if 'slha' not in self.banner:
            misc.sprint(self.banner)
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        if 'madspin' in self.banner and not allow_madspin:
            raise self.InvalidCmd('Reweight should be done before running MadSpin')
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if '[' in process:
            logger.warning("Remember that the reweighting is performed at Leading Order. NLO precision is not guarantee.")

        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception, msg
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)


    def get_LO_definition_from_NLO(self, proc):
        """return the LO definitions of the process corresponding to the born/real"""
        
        # split the line definition with the part before and after the NLO tag
        process, order, final = re.split('\[\s*(.*)\s*\]', proc)
        # add the part without any additional jet.
        commandline="add process %s %s --no_warning=duplicate;" % (process, final)
        if not order:
            #NO NLO tag => nothing to do actually return input
            return proc
        elif not order.startswith(('virt=','loonly=','noborn=')):
            # OK this a standard NLO process
            if '=' in order:
                # get the type NLO QCD/QED/...
                order = order.split('=',1)[1]
                
            # define the list of particles that are needed for the radiation
            pert = fks_common.find_pert_particles_interactions(self.model,
                                           pert_order = order)['soft_particles']
            commandline += "define pert_%s = %s;" % (order.replace(' ',''), ' '.join(map(str,pert)) )
            
            # check if we have to increase by one the born order
            if '%s=' % order in process:
                result=re.split(' ',process)
                process=''
                for r in result:
                    if '%s=' % order in r:
                        ior=re.split('=',r)
                        r='QCD=%i' % (int(ior[1])+1)
                    process=process+r+' '
            #handle special tag $ | / @
            result = re.split('([/$@]|\w+=\w+)', process, 1)                    
            if len(result) ==3:
                process, split, rest = result
                commandline+="add process %s pert_%s %s%s %s --no_warning=duplicate;" % (process, order.replace(' ','') ,split, rest, final)
            else:
                commandline +='add process %s pert_%s %s --no_warning=duplicate;' % (process,order.replace(' ',''), final)
        elif order.startswith(('noborn=')):
            # pass in sqrvirt=
            return "add process %s " % proc.replace('noborn=', 'sqrvirt=')
            
        else:
            #just return the input. since this Madloop.
            return "add process %s " % proc                                       
        return commandline


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

            try:
                event.check() #check 4 momenta/...
            except Exception, error:
                print event
                raise error
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
    
    def help_change(self):
        """help for change command"""
    
        print "change model X :use model X for the reweighting"
        print "change process p p > e+ e-: use a new process for the reweighting"
        print "change process p p > mu+ mu- --add : add one new process to existing ones"
    
    def do_change(self, line):
        """allow to define a second model/processes"""
        
        global nb_f2py_module
        
        args = self.split_arg(line)
        if len(args)<2:
            logger.critical("not enough argument (need at least two). Discard line")
        if args[0] == "model":
            nb_f2py_module += 1 # tag to force the f2py to reload
            self.second_model = " ".join(args[1:])
            if self.has_standalone_dir:
                self.terminate_fortran_executables()
                self.has_standalone_dir = False
        elif args[0] == "process":
            nb_f2py_module += 1
            if self.has_standalone_dir:
                self.terminate_fortran_executables()
                self.has_standalone_dir = False
            if args[-1] == "--add":
                self.second_process.append(" ".join(args[1:-1]))
            else:
                self.second_process = [" ".join(args[1:])]
        elif args[0] == "output":
            if args[1] in ['default', '2.0', 'unweight']:
                self.output_type = args[1]
        elif args[0] == "helicity":
            self.helicity_reweighting = banner.ConfigFile.format_variable(args[1], bool, "helicity")
        elif args[0] == "rwgt_dir":
            self.rwgt_dir = args[1]
            if not os.path.exists(self.rwgt_dir):
                os.mkdir(self.rwgt_dir)
        else:
            logger.critical("unknown option! %s.  Discard line." % args[0])
        
             
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.lhe_input:
            if isinstance(self.lhe_input, lhe_parser.EventFile):
                self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
            else:
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
            if self.rwgt_dir and os.path.exists(pjoin(self.rwgt_dir,'rw_me','rwgt.pkl')):
                self.load_from_pickle()
                self.me_dir = self.rwgt_dir
            else:
                self.create_standalone_directory()        
        
        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process:
            rw_dir = pjoin(path_me, 'rw_me_second')
        else:
            rw_dir = pjoin(path_me, 'rw_me')
        
        if not '--keep_card' in args:
            ff = open(pjoin(rw_dir,'Cards', 'param_card.dat'), 'w')
            ff.write(self.banner['slha'])
            ff.close()
            ff = open(pjoin(path_me, 'rw_me','Cards', 'param_card_orig.dat'), 'w')
            ff.write(self.banner['slha'])
            ff.close()      
            cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                   ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line)
            self.stored_line = None
        
        # check for potential scan in the new card 
        new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()
        pattern_scan = re.compile(r'''^[\s\d]*scan''', re.I+re.M) 
        param_card_iterator = []
        if pattern_scan.search(new_card):
            if not isinstance(self.mother, cmd.CmdShell): 
                raise Exception, "scan are not allowed on the Web"
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))                
        # check if "Auto" is present for a width parameter
        if "auto" in new_card.lower():            
            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()

        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner:
            if 'type=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup type=\'mg_reweighting\'\s*>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'mg_reweight_(?P<id>\d+)\'>(?P<info>[^<]*)</weight>', re.S+re.I+re.M)
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
        if not self.second_model:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  check_param_card.ParamCard(s_new.splitlines())
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process:
                logger.warning(' REWEIGHTING: original card and new card are identical. Bypass this run')
                return
                #raise self.InvalidCmd, 'original card and new card are identical'
            try:
                if old_param['sminputs'].get(3)- new_param['sminputs'].get(3) > 1e-3 * new_param['sminputs'].get(3):
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception, error:
                logger.debug("error in check of alphas: %s" % str(error))
                pass #this is a security                
            if not self.second_process:
                mg_rwgt_info.append((str(rewgtid), card_diff))
            else:
                str_proc = "\n change process  ".join([""]+self.second_process)
                mg_rwgt_info.append((str(rewgtid), str_proc + '\n'+ card_diff))
        else:
            str_info = "change model %s" % self.second_model
            if self.second_process:
                str_info += "\n change process  ".join([""]+self.second_process)
            card_diff = str_info
            str_info += '\n' + s_new
            mg_rwgt_info.append((str(rewgtid), str_info))

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
            lha_strategy = self.banner.get_lha_strategy() 
            self.banner.set_lha_strategy(4*lha_strategy/abs(lha_strategy)) 
            self.banner.write(output2, close_tag=False)
            self.banner.set_lha_strategy(lha_strategy)
            new_banner = banner.Banner(self.banner)
            if not hasattr(self, 'run_card'):
                self.run_card = new_banner.charge_card('run_card')
            self.run_card['run_tag'] = 'reweight_%s' % rewgtid
            new_banner['slha'] = s_new   
            del new_banner['initrwgt']
            assert 'initrwgt' in self.banner 
            ff = open(pjoin(self.mother.me_dir,'Events',self.mother.run_name, '%s_%s_banner.txt' % \
                          (self.mother.run_name, self.run_card['run_tag'])),'w') 
            new_banner.write(ff)
            ff.close()
        
        # Loop over all events
        tag_name = 'mg_reweight_%s' % rewgtid
        start = time.time()
        cross = 0
        ratio, ratio_square = 0, 0 # to compute the variance and associate error
        
        os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'y'
        if self.lhe_input.closed:
            misc.sprint("using", self.lhe_input.name)
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

#        Multicore option not really stable -> not use it
        nb_core = 1
#        if nb_core >1:
#            multicore = cluster.MultiCore(nb_core)

        self.lhe_input.seek(0)
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')

            if nb_core > 1:
                #        Multicore option not really stable -> not use it
                while 1:
                    if multicore.queue.qsize() < 100 * nb_core:
                        multicore.submit(self.write_reweighted_event, argument=[event, tag_name])
                        break
                    #else:
                    #    time.sleep(0.001)
                continue
            else:
                weight = self.calculate_weight(event)
                cross += weight
                ratio += weight/event.wgt
                ratio_square += (weight/event.wgt)**2
                if self.output_type == "default":
                    event.reweight_data[tag_name] = weight
                    #write this event with weight
                    output.write(str(event))
                    if self.mother:
                        event.wgt = weight
                        event.reweight_data = {}
                        output2.write(str(event))
    
                else:
                    event.wgt = weight
                    event.reweight_data = {}
                    if self.mother:
                        output2.write(str(event))
                    else:
                        output.write(str(event))
                
        # check normalisation of the events:
        if 'event_norm' in self.run_card:
            if self.run_card['event_norm'] == 'average':
                cross /= event_nb+1

                
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
                event_nb +=1
                variance = ratio_square/event_nb - (ratio/event_nb)**2
                orig_cross, orig_error = self.orig_cross
                error = variance/math.sqrt(event_nb) * orig_cross + ratio/event_nb * orig_error
                
                results.add_detail('error', error)
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
        if not self.mother or self.output_type != "default" :
            target = pjoin(self.mother.me_dir, 'Events', run_name, 'events.lhe')
        else:
            target = self.lhe_input.name
        
        if self.output_type == "default":
            files.mv(output.name, target)
        elif self.output_type == "unweight":
            output2.close()
            lhe = lhe_parser.EventFile(output2.name)
            nb_event = lhe.unweight(target)
            if self.mother and  hasattr(self.mother, 'results'):
                results = self.mother.results
                results.add_detail('nb_event', nb_event)
                results.current.parton.append('lhe')
                
        else:
            files.mv(output2.name, self.lhe_input.name)     
            if self.mother and  hasattr(self.mother, 'results'):
                results = self.mother.results
                results.current.parton.append('lhe')   

        logger.info('Event %s have now the additional weight' % self.lhe_input.name)
        logger.info('new cross-section is : %g pb (indicative error: %g pb)' % (cross,error))
        self.terminate_fortran_executables(new_card_only=True)
        #store result
        self.all_cross_section[rewgtid] = (cross, error)
        
        # perform the scanning
        if param_card_iterator:
            for card in param_card_iterator:
                card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                self.exec_cmd("launch --keep_card", printcmd=False, precmd=True)
        
    
    def do_set(self, line):
        "Not in help"
        
        logger.warning("Invalid Syntax. The command 'set' should be placed after the 'launch' one. Continuing by adding automatically 'launch'")
        self.stored_line = "set %s" % line
        return self.exec_cmd("launch")
    
    def default(self, line, log=True):
        """Default action if line is not recognized"""

        if os.path.isfile(line):
            if log:
                logger.warning("Invalid Syntax. The path to a param_card' should be placed after the 'launch' command. Continuing by adding automatically 'launch'")
            self.stored_line =  line
            return self.exec_cmd("launch")
        else:
            return super(ReweightInterface,self).default(line, log=log)
    
    def write_reweighted_event(self, event, tag_name, **opt):
        """a function for running in multicore"""
        
        if not hasattr(opt['thread_space'], "calculator"):
            opt['thread_space'].calculator = {}
            opt['thread_space'].calculator_nbcall = {}
            opt['thread_space'].cross = 0
            opt['thread_space'].output = open( self.lhe_input.name +'rw.%s' % opt['thread_id'], 'w')
            if self.mother:
                out_path = pjoin(self.mother.me_dir, 'Events', 'reweight.lhe.%s' % opt['thread_id'])
                opt['thread_space'].output2 = open(out_path, 'w')
                
        weight = self.calculate_weight(event, space=opt['thread_space'])
        opt['thread_space'].cross += weight
        if self.output_type == "default":
            event.reweight_data[tag_name] = weight
            #write this event with weight
            opt['thread_space'].output.write(str(event))
            if self.mother:
                event.wgt = weight
                event.reweight_data = {}
                opt['thread_space'].output2.write(str(event))
        else:
            event.wgt = weight
            event.reweight_data = {}
            if self.mother:
                opt['thread_space'].output2.write(str(event))
            else:
                opt['thread_space'].output.write(str(event))
        
        return 0
    
    def calculate_weight(self, event, space=None):
        """space defines where to find the calculator (in multicore)"""
        
        if not space: 
            space = self
        
        event.parse_reweight()
        
        w_orig = self.calculate_matrix_element(event, 0, space)
        w_new =  self.calculate_matrix_element(event, 1, space)
        if w_orig == 0:
            tag, order = event.get_tag_and_order()
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
            misc.sprint(w_orig, w_new)
            misc.sprint(event)
            raise Exception, "Invalid matrix element for original computation (weight=0)"
        return w_new/w_orig*event.wgt
     
    @staticmethod   
    def invert_momenta(p):
        """ fortran/C-python do not order table in the same order"""
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p
    
    @staticmethod
    def rename_f2py_lib(Pdir, tag):
        if tag == 2:
            return
        if os.path.exists(pjoin(Pdir, 'matrix%spy.so' % tag)):
            return
        else:
            open(pjoin(Pdir, 'matrix%spy.so' % tag),'w').write(open(pjoin(Pdir, 'matrix2py.so')
                                        ).read().replace('matrix2py', 'matrix%spy' % tag))
    
    def calculate_matrix_element(self, event, hypp_id, space):
        """routine to return the matrix element"""

        tag, order = event.get_tag_and_order()
        
        if (not self.second_model and not self.second_process) or hypp_id==0:
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
        else:
            orig_order, Pdir, hel_dict = self.id_to_path_second[tag]

        run_id = (tag, hypp_id)


        assert space == self
        start = False
        if run_id in space.calculator:
            external = space.calculator[run_id]
        elif (not self.second_model and not self.second_process) or hypp_id==0:
            # create the executable for this param_card  
            subdir = pjoin(self.me_dir,'rw_me ', 'SubProcesses')
            if self.me_dir not in sys.path:
                sys.path.insert(0,self.me_dir)
            if self.rwgt_dir and self.rwgt_dir not in sys.path:
                sys.path.insert(0,self.rwgt_dir)
            Pname = os.path.basename(Pdir)
            if hypp_id == 0:
                if (Pdir, 0) not in dir_to_f2py_free_mod:
                    metag = 1
                    dir_to_f2py_free_mod[(Pdir,0)] = (metag, nb_f2py_module)
                else:
                    metag, old_module = dir_to_f2py_free_mod[(Pdir,0)]
                    if old_module != nb_f2py_module:
                        metag += 1
                        dir_to_f2py_free_mod[(Pdir,0)] = (metag, nb_f2py_module)
                
                os.environ['MENUM'] = '2'
                if not self.rwgt_dir or not os.path.exists(pjoin(Pdir, 'matrix2py.so')):
                    misc.compile(['matrix2py.so'], cwd=Pdir)
                    self.rename_f2py_lib(Pdir, 2*metag)
                    
                mymod = __import__('rw_me.SubProcesses.%s.matrix%spy' % (Pname, 2*metag), globals(), locals(), [],-1)
                S = mymod.SubProcesses
                P = getattr(S, Pname)
                mymod = getattr(P, 'matrix%spy' % (2*metag))
                with misc.chdir(Pdir):
                    mymod.initialise('param_card_orig.dat')
                    
            if hypp_id == 1:
                #incorrect line
                metag = dir_to_f2py_free_mod[(Pdir,0)][0]
                newtag = 2*metag+1
                self.rename_f2py_lib(Pdir, newtag)
                try:
                    mymod = __import__('rw_me.SubProcesses.%s.matrix%spy' % (Pname, newtag), globals(), locals(), [],-1)
                except Exception, error:
                    os.remove(pjoin(Pdir, 'matrix%spy.so' % newtag))
                    newtag  = "L%s" % newtag
                    os.environ['MENUM'] = newtag
                    misc.compile(['matrix%spy.so' % newtag], cwd=Pdir)
                    mymod = __import__('rw_me.SubProcesses.%s.matrix%spy' % (Pname, newtag), globals(), locals(), [],-1)
                
                S = mymod.SubProcesses
                P = getattr(S, Pname)
                mymod = getattr(P, 'matrix%spy' % newtag)
                with misc.chdir(Pdir):
                    mymod.initialise('param_card.dat') 
            space.calculator[run_id] = mymod.get_me
            external = space.calculator[run_id]                      
        else:
            subdir = pjoin(self.me_dir,'rw_me_second', 'SubProcesses')
            if self.me_dir not in sys.path:
                sys.path.append(self.me_dir)

            assert hypp_id == 1
            Pname = os.path.basename(Pdir)
            os.environ['MENUM'] = '2'
            if not self.rwgt_dir or not os.path.exists(pjoin(Pdir, 'matrix2py.so')):
                misc.compile(['matrix2py.so'], cwd=pjoin(subdir, Pdir))
            if (Pdir, 1) not in dir_to_f2py_free_mod:
                metag = 1
                dir_to_f2py_free_mod[(Pdir,1)] = (metag, nb_f2py_module)
            else:
                metag, old_module = dir_to_f2py_free_mod[(Pdir,1)]
                if old_module != nb_f2py_module:
                    metag += 1
                    dir_to_f2py_free_mod[(Pdir,1)] = (metag, nb_f2py_module)
            self.rename_f2py_lib(Pdir, metag)
            
            try:
                mymod = __import__("rw_me_second.SubProcesses.%s.matrix%spy" % (Pname, metag))
            except ImportError:
                os.remove(pjoin(Pdir, 'matrix%spy.so' % metag ))
                metag = "L%s" % metag
                os.environ['MENUM'] = str(metag)
                misc.compile(['matrix%spy.so' % metag], cwd=pjoin(subdir, Pdir))
                mymod = __import__("rw_me_second.SubProcesses.%s.matrix%spy" % (Pname, metag))
            reload(mymod)
            S = mymod.SubProcesses
            P = getattr(S, Pname)
            mymod = getattr(P, 'matrix%spy' % metag)                
            with misc.chdir(Pdir):      
                mymod.initialise('param_card.dat')              
            space.calculator[run_id] = mymod.get_me
            external = space.calculator[run_id]                
        
        
        p = self.invert_momenta(event.get_momenta(orig_order))
        # add helicity information
        
        hel_order = event.get_helicity(orig_order)
        if self.helicity_reweighting and 9 not in hel_order:
            nhel = hel_dict[tuple(hel_order)]
        else:
            nhel = 0

        with misc.chdir(Pdir):
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                me_value = external(p,event.aqcd, nhel)
        # for NLO we have also the stability status code
        if isinstance(me_value, tuple):
            me_value, code = me_value
            #if code points unstability -> returns 0
            hundred_value = (code % 1000) //100
            if hundred_value in [4]:
                me_value = 0.
            
        return me_value
    
    def terminate_fortran_executables(self, new_card_only=False):
        """routine to terminate all fortran executables"""

        for (mode, production) in dict(self.calculator):
            
            if new_card_only and production == 0:
                continue            
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
            logger.info('%s : %s +- %s pb' % (key,self.all_cross_section[key][0],self.all_cross_section[key][1] ))  
        self.terminate_fortran_executables()
    
        if self.rwgt_dir:
            self.save_to_pickle()
            
    def __del__(self):
        self.do_quit('')

    
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        



        
    
    @misc.mute_logger()
    def create_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""
        
        # 0. clean previous run ------------------------------------------------
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
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
        
        #multiparticles
        for name, content in self.banner.get('proc_card', 'multiparticles'):
            mgcmd.exec_cmd("define %s = %s" % (name, content))
        
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
                commandline += "add process %s ;" % proc
            else:
                commandline += self.get_LO_definition_from_NLO(proc)
        
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
                # build the helicity dictionary
                hel_nb = 0
                hel_dict = {9:0} # unknown helicity -> use full ME
                for helicities in me.get_helicity_matrix():
                    hel_nb +=1 #fortran starts at 1
                    hel_dict[tuple(helicities)] = hel_nb

                self.id_to_path[tag] = [order, Pdir, hel_dict]        
        
        # 3. If we need a new model/process-------------------------------------
        if self.second_model or self.second_process:
            self.create_second_standalone_directory()    

    @misc.mute_logger()
    def create_second_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""
        
        # 0. clean previous run ------------------------------------------------
        path_me = self.me_dir
        try:
            shutil.rmtree(pjoin(path_me,'rw_me_second'))
        except Exception: 
            pass
        
        mgcmd = self.mg5cmd
        # 1. Load model---------------------------------------------------------  
        if self.second_model:
            use_mgdefault= True
            complex_mass = False
            if ' ' in self.second_model:
                args = self.second_model.split()
                if '--modelname' in args:
                    use_mgdefault = False
                model_name = args[0]
            else:
                model_name = self.second_model
            self.load_model(model_name, use_mgdefault, complex_mass)
                
            modelpath = self.model.get('modelpath')
            if os.path.basename(modelpath) != mgcmd._curr_model['name']:
                name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
                if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                    modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])
                
            commandline="import model %s " % modelpath
            if not use_mgdefault:
                commandline += ' -modelname '
            mgcmd.exec_cmd(commandline)
            
        # 2. compute the production matrix element -----------------------------
        if self.second_process:
            processes = self.second_process
        else:
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
            elif 'sqrvirt' in proc:
                commandline+="add process %s ;" % proc
            else:
                raise self.InvalidCmd('NLO processes can\'t be reweight (for Loop induced reweighting use [sqrvirt =])')
        
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        
        matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
        
        commandline = 'output standalone_rw %s' % pjoin(path_me,'rw_me_second')
        mgcmd.exec_cmd(commandline, precmd=True)        
        logger.info('Done %.4g' % (time.time()-start))
        
        # 3. Store id to directory information ---------------------------------
        matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
        
        self.id_to_path_second = {}
        to_check = [] # list of tag that do not have a Pdir at creation time. 
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
                Pdir = pjoin(path_me, 'rw_me_second', 'SubProcesses', 
                                  'P%s' % me.get('processes')[0].shell_string())
                if not os.path.exists(Pdir):
                    to_check.append(tag)
                    continue
                if tag in self.id_to_path_second:
                    if not Pdir == self.id_to_path_second[tag][1]:
                        misc.sprint(tag, Pdir, self.id_to_path_second[tag][1])
                        raise self.InvalidCmd, '2 different process have the same final states. This module can not handle such situation'
                    else:
                        continue
                
                # build the helicity dictionary
                hel_nb = 0
                hel_dict = {9:0} # unknown helicity -> use full ME
                for helicities in me.get_helicity_matrix():
                    hel_nb +=1 #fortran starts at 1
                    hel_dict[tuple(helicities)] = hel_nb    
                self.id_to_path_second[tag] = [order, Pdir, hel_dict]

        for tag in to_check:
            if tag not in self.id_to_path:
                logger.warning("no valid path for %s" % (tag,))
                #raise self.InvalidCmd, "no valid path for %s" % (tag,)
        
        # 4. Check MadLoopParam
        if os.path.exists(pjoin(path_me, 'rw_me_second', 'Cards', 'MadLoopParams.dat')):
            MLCard = banner.MadLoopParam(pjoin(path_me, 'rw_me_second', 'Cards', 'MadLoopParams.dat'))
            MLCard.set('WriteOutFilters', False)
            MLCard.set('UseLoopFilter', False)
            MLCard.set("DoubleCheckHelicityFilter", False)
            MLCard.set("HelicityFilterLevel", 0)
            MLCard.write(pjoin(path_me, 'rw_me_second', 'SubProcesses', 'MadLoopParams.dat'),
                         pjoin(path_me, 'rw_me_second', 'Cards', 'MadLoopParams.dat'), 
                         commentdefault=False)
            
            
            
        
        

    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False

        logger.info('detected model: %s. Loading...' % name)
        model_path = name

        # Import model
        base_model = import_ufo.import_model(name, decay=False,
                                               complex_mass_scheme=complex_mass)
    
        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()
        

    def save_to_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        to_save = {}
        to_save['id_to_path'] = self.id_to_path
        if hasattr(self, 'id_to_path_second'):
            to_save['id_to_path_second'] = self.id_to_path_second
        else:
            to_save['id_to_path_second'] = {}
        to_save['all_cross_section'] = self.all_cross_section
        to_save['processes'] = self.processes
        to_save['second_process'] = self.second_process
        if self.second_model:
            to_save['second_model'] =True
        else:
            to_save['second_model'] = None
        to_save['rwgt_dir'] = self.rwgt_dir

        name = pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl')
        save_load_object.save_to_file(name, to_save)

    def load_from_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        obj = save_load_object.load_from_file( pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl'))
        
        self.has_standalone_dir = True
        self.options = {'curr_dir': os.path.realpath(os.getcwd())}
        
        old_rwgt = obj['rwgt_dir']
           
        # path to fortran executable
        self.id_to_path = {}
        for key , (order, Pdir, hel_dict) in obj['id_to_path'].items():
            new_P = Pdir.replace(old_rwgt, self.rwgt_dir)
            self.id_to_path[key] = [order, new_P, hel_dict]
            
        # path to fortran executable (for second directory)
        self.id_to_path_second = {}
        for key , (order, Pdir, hel_dict) in obj['id_to_path_second'].items():
            new_P = Pdir.replace(old_rwgt, self.rwgt_dir)
            self.id_to_path_second[key] = [order, new_P, hel_dict]            
        
        self.all_cross_section = obj['all_cross_section']            
        self.processes = obj['processes']
        self.second_process = obj['second_process']
        self.second_model = obj['second_model']

        
        
        
        





        
