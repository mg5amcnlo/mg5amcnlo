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
""" Command interface for Re-Weighting """
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
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.iolibs.files as files
#import MadSpin.interface_madspin as madspin_interface
import madgraph.various.misc as misc
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.combine_plots as combine_plots
import madgraph.various.cluster as cluster
import madgraph.fks.fks_common as fks_common
import madgraph.core.diagram_generation as diagram_generation

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card 
#import MadSpin.decay as madspin


logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

# global to check which f2py module have been already loaded. (to avoid border effect)
dir_to_f2py_free_mod = {}
nb_f2py_module = 0 # each time the process/model is changed this number is modified to 
                   # forced the python module to re-create an executable

lhapdf = None


class ReweightInterface(extended_cmd.Cmd):
    """Basic interface for reweighting operation"""
    
    prompt = 'Reweight>'
    debug_output = 'Reweight_debug'
    rwgt_dir_possibility = ['rw_me','rw_me_second','rw_mevirt','rw_mevirt_second']
    
    @misc.mute_logger()
    def __init__(self, event_path=None, allow_madspin=False, mother=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""
        
        
        self.me_dir = os.getcwd()
        if not event_path:
            cmd_logger.info('************************************************************')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('*               Welcome to Reweight Module                 *')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)
        
        self.model = None
        self.has_standalone_dir = False
        self.mother= mother # calling interface
        self.multicore=False
        
        self.options = {'curr_dir': os.path.realpath(os.getcwd()),
                        'rwgt_name':None}

        self.events_file = None
        self.processes = {}
        self.f2pylib = {}
        self.second_model = None
        self.second_process = None
        self.dedicated_path = {}
        self.soft_threshold = None
        self.systematics = False # allow to run systematics in ouput2.0 mode
        self.mg5cmd = master_interface.MasterCmd()
        if mother:
            self.mg5cmd.options.update(mother.options)
        self.seed = None
        self.output_type = "default"
        self.helicity_reweighting = True
        self.rwgt_mode = '' # can be LO, NLO, NLO_tree, '' is default 
        self.has_nlo = False
        self.rwgt_dir = None
        self.exitted = False # Flag to know if do_quit was already called.
        
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
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        if 'madspin' in self.banner and not allow_madspin:
            raise self.InvalidCmd('Reweight should be done before running MadSpin')
        
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if '[' in process and isinstance(self.banner.get('run_card'), banner.RunCardNLO):
            if not self.banner.get_detail('run_card', 'store_rwgt_info'):
                logger.warning("The information to perform a proper NLO reweighting is not present in the event file.")
                logger.warning("       We will perform a LO reweighting instead. This does not guarantee NLO precision.")
                self.rwgt_mode = 'LO'

            if 'OLP' in self.mother.options:
                if self.mother.options['OLP'].lower() != 'madloop':
                    logger.warning("Accurate NLO mode only works for OLP=MadLoop not for OLP=%s. An approximate (LO) reweighting will be performed instead")
                    self.rwgt_mode = 'LO'
            
            if 'lhapdf' in self.mother.options and not self.mother.options['lhapdf']:
                logger.warning('NLO accurate reweighting requires lhapdf to be installed. Pass in approximate LO mode.')
                self.rwgt_mode = 'LO'
        else:
            self.rwgt_mode = 'LO'

        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception, msg
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        self.is_decay = len(process.split('>',1)[0].split()) == 1 
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)


    @staticmethod
    def get_LO_definition_from_NLO(proc, model, real_only=False):
        """return the LO definitions of the process corresponding to the born/real"""
        
        # split the line definition with the part before and after the NLO tag
        process, order, final = re.split('\[\s*(.*)\s*\]', proc)
        if process.strip().startswith(('generate', 'add process')):
            process = process.replace('generate', '')
            process = process.replace('add process','')
        
        # add the part without any additional jet.
        commandline="add process %s %s --no_warning=duplicate;" % (process, final)
        if not order:
            #NO NLO tag => nothing to do actually return input
            return proc
        elif not order.startswith(('virt','LOonly','noborn')):
            # OK this a standard NLO process            
            if real_only:
                commandline= '' 
            
            if '=' in order:
                # get the type NLO QCD/QED/...
                order = order.split('=',1)[1].strip()

            # define the list of particles that are needed for the radiation
            pert = fks_common.find_pert_particles_interactions(model,
                                           pert_order = order)['soft_particles']
            commandline += "define pert_%s = %s;" % (order.replace(' ',''), ' '.join(map(str,pert)) )
            
            # check if we have to increase by one the born order
            
            if '%s=' % order in process or '%s<=' % order in process:
                result=re.split(' ',process)
                process=''
                for r in result:
                    if '%s=' % order in r:
                        ior=re.split('=',r)
                        r='QCD=%i' % (int(ior[1])+1)
                    elif '%s<=' % order in r:
                        ior=re.split('=',r)
                        r='QCD<=%i' % (int(ior[1])+1)
                    process=process+r+' '
            #handle special tag $ | / @
            result = re.split('([/$@]|\w+(?:^2)?(?:=|<=|>)+\w+)', process, 1)                    
            if len(result) ==3:
                process, split, rest = result
                commandline+="add process %s pert_%s %s%s %s --no_warning=duplicate;" % (process, order.replace(' ','') ,split, rest, final)
            else:
                commandline +='add process %s pert_%s %s --no_warning=duplicate;' % (process,order.replace(' ',''), final)
        elif order.startswith(('noborn=')):
            # pass in sqrvirt=
            return "add process %s [%s] %s;" % (process, order.replace('noborn=', 'sqrvirt='), final)
        elif order.startswith('LOonly'):
            #remove [LOonly] flag
            return "add process %s %s;" % (process, final)
        else:
            #just return the input. since this Madloop.
            if order:
                return "add process %s [%s] %s ;" % (process, order,final)
            else:
                return "add process %s %s ;" % (process, final)
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
        print "change output [default|2.0|unweight]:"
        print "               default: add weight(s) to the current file"    
    
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
        elif args[0] in ['virtual_path', 'tree_path']:
            self.dedicated_path[args[0]] = os.path.abspath(args[1])
        elif args[0] == "output":
            if args[1] in ['default', '2.0', 'unweight']:
                self.output_type = args[1]
        elif args[0] == "helicity":
            self.helicity_reweighting = banner.ConfigFile.format_variable(args[1], bool, "helicity")
        elif args[0] == "mode":
            if args[1] != 'LO':
                if 'OLP' in self.mother.options and self.mother.options['OLP'].lower() != 'madloop':
                    logger.warning("Only LO reweighting is allowed for OLP!=MadLoop. Keeping the mode to LO.")
                    self.rwgt_mode = 'LO'
                elif not self.banner.get_detail('run_card','store_rwgt_info', default=False):
                    logger.warning("Missing information for NLO type of reweighting. Keeping the mode to LO.")
                    self.rwgt_mode = 'LO'
                elif 'lhapdf' in self.mother.options and not self.mother.options['lhapdf']:
                    logger.warning('NLO accurate reweighting requires lhapdf to be installed. Pass in approximate LO mode.')
                    self.rwgt_mode = 'LO'
                else:
                    self.rwgt_mode = args[1]
            else:
                self.rwgt_mode = args[1]
        elif args[0] == "rwgt_dir":
            self.rwgt_dir = args[1]
            if not os.path.exists(self.rwgt_dir):
                os.mkdir(self.rwgt_dir)
            self.rwgt_dir = os.path.abspath(self.rwgt_dir)
        elif args[0] == 'systematics':
            if self.output_type == 'default' and args[1].lower() not in ['none', 'off']:
                logger.warning('systematics can only be computed for non default output type. pass to output mode \'2.0\'')
                self.output_type = '2.0'
            if len(args) == 2:
                try:
                    self.systematics = banner.ConfigFile.format_variable(args[1], bool)
                except Exception, error:
                    self.systematics = args[1:]
            else:
                self.systematics = args[1:]
        elif args[0] == 'soft_threshold':
            self.soft_threshold = banner.ConfigFile.format_variable(args[1], float, 'soft_threshold')
        elif args[0] == 'multicore':
            pass 
            # this line is meant to be parsed by common_run_interface and change the way this class is called.
            #It has no direct impact on this class.
        else:
            logger.critical("unknown option! %s.  Discard line." % args[0])
        
             
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.lhe_input:
            if isinstance(self.lhe_input, lhe_parser.EventFile):
                self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
            else:
                raise self.InvalidCmd("No events files defined.")
            
        opts = {'rwgt_name':None}
        if any(a.startswith('--') for a in args):
            for a in args[:]:
                if a.startswith('--') and '=' in a:
                    key,value = a[2:].split('=')
                    opts[key] = value .replace("'","") .replace('"','')
        return opts

    def help_launch(self):
        """help for the launch command"""
        
        logger.info('''Add to the loaded events a weight associated to a 
        new param_card (to be define). The weight returned is the ratio of the 
        square matrix element by the squared matrix element of production.
        All scale are kept fix for this re-weighting.''')


    def get_weight_names(self):
        """ return the various name for the computed weights """
        
        if self.rwgt_mode == 'LO':
            return ['']
        elif self.rwgt_mode == 'NLO':
            return ['_nlo']
        elif self.rwgt_mode == 'LO+NLO':
            return ['_lo', '_nlo']
        elif self.rwgt_mode == 'NLO_tree':
            return ['_tree']        
        elif not self.rwgt_mode and self.has_nlo :
            return ['_nlo']
        else:
            return ['']

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        args = self.split_arg(line)
        opts = self.check_launch(args)
        if opts['rwgt_name']:
            self.options['rwgt_name'] = opts['rwgt_name']

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.has_standalone_dir:                           
            if self.rwgt_dir and os.path.exists(pjoin(self.rwgt_dir,'rw_me','rwgt.pkl')):
                self.load_from_pickle()
                if opts['rwgt_name']:
                    self.options['rwgt_name'] = opts['rwgt_name']
                if not self.rwgt_dir:
                    self.me_dir = self.rwgt_dir
                self.load_module()       # load the fortran information from the f2py module
            elif self.multicore == 'wait':
                i=0
                while not os.path.exists(pjoin(self.me_dir,'rw_me','rwgt.pkl')):
                    time.sleep(10+i)
                    i+=5
                    print 'wait for pickle'                  
                print "loading from pickle"
                if not self.rwgt_dir:
                    self.rwgt_dir = self.me_dir
                self.load_from_pickle(keep_name=True)
                self.load_module()
            else:
                self.create_standalone_directory()
                self.compile()
                self.load_module()  
                if self.multicore == 'create':
                    self.load_module()
                    if not self.rwgt_dir:
                        self.rwgt_dir = self.me_dir
                    self.save_to_pickle()      
        
        # get the mode of reweighting #LO/NLO/NLO_tree/...
        type_rwgt = self.get_weight_names()
        # get iterator over param_card and the name associated to the current reweighting.
        param_card_iterator, tag_name = self.handle_param_card(model_line, args, type_rwgt)
        
        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_second')
        else:
            rw_dir = pjoin(path_me, 'rw_me')
                
        start = time.time()
        # initialize the collector for the various re-weighting
        cross, ratio, ratio_square,error = {},{},{}, {}
        for name in type_rwgt + ['orig']:
            cross[name], error[name] = 0.,0.
            ratio[name],ratio_square[name] = 0., 0.# to compute the variance and associate error

        if self.output_type == "default":
            output = open( self.lhe_input.name +'rw', 'w')
            #write the banner to the output file
            self.banner.write(output, close_tag=False)
        else:
            output = {}
            if tag_name.isdigit():
                name_tag= 'rwgt_%s' % tag_name
            else:
                name_tag = tag_name
            base = os.path.dirname(self.lhe_input.name)
            for rwgttype in  type_rwgt:
                output[(name_tag,rwgttype)] = lhe_parser.EventFile(pjoin(base,'rwgt_events%s_%s.lhe.gz' %(rwgttype,tag_name)), 'w')
                #write the banner to the output file
                self.banner.write(output[(name_tag,rwgttype)], close_tag=False)
                
        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

        self.lhe_input.seek(0)
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
            if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')

            weight = self.calculate_weight(event)
            if not isinstance(weight, dict):
                weight = {'':weight}
            
            for name in weight:
                cross[name] += weight[name]
                ratio[name] += weight[name]/event.wgt
                ratio_square[name] += (weight[name]/event.wgt)**2

            # ensure to have a consistent order of the weights. new one are put 
            # at the back, remove old position if already defines
            for tag in type_rwgt:
                try:
                    event.reweight_order.remove('%s%s'  % (tag_name,tag))
                except ValueError:
                    continue
            
            event.reweight_order += ['%s%s' % (tag_name,name) for name in type_rwgt]  
            if self.output_type == "default":
                for name in weight:
                    if 'orig' in name:
                        continue             
                    event.reweight_data['%s%s' % (tag_name,name)] = weight[name]
                    #write this event with weight
                output.write(str(event))
            else:
                for i,name in enumerate(weight):
                    if 'orig' in name:
                        continue 
                    if weight[name] == 0:
                        continue
                    new_evt = lhe_parser.Event(str(event))
                    new_evt.wgt = weight[name]
                    new_evt.parse_reweight()
                    new_evt.reweight_data = {}  
                    output[(tag_name,name)].write(str(new_evt))

        # check normalisation of the events:
        if 'event_norm' in self.run_card:
            if self.run_card['event_norm'] in ['average','bias']:
                for key, value in cross.items():
                    cross[key] = value / (event_nb+1)
                
        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))        
        
        
        if self.output_type == "default":
            output.write('</LesHouchesEvents>\n')
            output.close()
        else:
            for key in output:
                output[key].write('</LesHouchesEvents>\n')
                output[key].close()
                if self.systematics and len(output) ==1:
                    try:
                        logger.info('running systematics computation')
                        import madgraph.various.systematics as syst
                        
                        if not isinstance(self.systematics, bool):
                            args = [output[key].name, output[key].name] + self.systematics
                        else:
                            args = [output[key].name, output[key].name]
                        if self.mother and self.mother.options['lhapdf']:
                            args.append('--lhapdf_config=%s' % self.mother.options['lhapdf'])
                        syst.call_systematics(args, result=open('rwg_syst_%s.result' % key[0],'w'),
                                              log=logger.info)
                    except Exception:
                        logger.error('fail to add systematics')
                        raise
        # add output information        
        if self.mother and hasattr(self.mother, 'results'):
            run_name = self.mother.run_name
            results = self.mother.results
            results.add_run(run_name, self.run_card, current=True)
            results.add_detail('nb_event', event_nb+1)
            name = type_rwgt[0]
            results.add_detail('cross', cross[name])
            event_nb +=1
            for name in type_rwgt:
                variance = ratio_square[name]/event_nb - (ratio[name]/event_nb)**2
                orig_cross, orig_error = self.orig_cross
                error[name] = math.sqrt(max(0,variance/math.sqrt(event_nb))) * orig_cross + ratio[name]/event_nb * orig_error
            results.add_detail('error', error[type_rwgt[0]])
            import madgraph.interface.madevent_interface as ME_interface

        self.lhe_input.close()
        if not self.mother:
            name, ext = self.lhe_input.name.rsplit('.',1)
            target = '%s_out.%s' % (name, ext)            
        elif self.output_type != "default" :
            target = pjoin(self.mother.me_dir, 'Events', run_name, 'events.lhe')
        else:
            target = self.lhe_input.name
        
        if self.output_type == "default":
            files.mv(output.name, target)
            logger.info('Event %s have now the additional weight' % self.lhe_input.name)
        elif self.output_type == "unweight":
            for key in output:
                output[key].write('</LesHouchesEvents>\n')
                output.close()
                lhe = lhe_parser.EventFile(output[key].name)
                nb_event = lhe.unweight(target)
                if self.mother and  hasattr(self.mother, 'results'):
                    results = self.mother.results
                    results.add_detail('nb_event', nb_event)
                    results.current.parton.append('lhe')
                logger.info('Event %s is now unweighted under the new theory' % lhe.name)                
        else:
            if self.mother and  hasattr(self.mother, 'results'):
                results = self.mother.results
                results.current.parton.append('lhe')       
            logger.info('Eventfiles is/are now created with new central weight')
        
        if self.multicore != 'create':
            for name in cross:
                if name == 'orig':
                    continue
                logger.info('new cross-section is %s: %g pb (indicative error: %g pb)' %\
                        ('(%s)' %name if name else '',cross[name], error[name]))
            
        self.terminate_fortran_executables(new_card_only=True)
        #store result
        for name in cross:
            if name == 'orig':
                self.all_cross_section[name] = (cross[name], error[name])
            else:
                self.all_cross_section[(tag_name,name)] = (cross[name], error[name])

        # perform the scanning
        if param_card_iterator:
            for i,card in enumerate(param_card_iterator):
                if self.options['rwgt_name']:
                    self.options['rwgt_name'] = '%s_%s' % (self.options['rwgt_name'].rsplit('_',1)[0], i+1)
                card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                self.exec_cmd("launch --keep_card", printcmd=False, precmd=True)
        
        self.options['rwgt_name'] = None


    def handle_param_card(self, model_line, args, type_rwgt):
        
        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_second')
        else:
            rw_dir = pjoin(path_me, 'rw_me')
        
        
        if not '--keep_card' in args:
            ff = open(pjoin(rw_dir,'Cards', 'param_card.dat'), 'w')
            ff.write(self.banner['slha'])
            ff.close()
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_me', 'rw_mevirt')
                files.ln(ff.name, starting_dir=pjoin(rwdir_virt, 'Cards')) 
            ff = open(pjoin(path_me, 'rw_me','Cards', 'param_card_orig.dat'), 'w')
            ff.write(self.banner['slha'])
            ff.close()      
            if self.has_nlo and self.rwgt_mode != "LO":
                files.ln(ff.name, starting_dir=pjoin(path_me, 'rw_mevirt', 'Cards'))
            cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                   ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line)
            self.stored_line = None
        
        # check for potential scan in the new card 
        new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()
        pattern_scan = re.compile(r'''^[\s\d]*scan''', re.I+re.M) 
        param_card_iterator = []
        if pattern_scan.search(new_card):
            try:
                import internal.extended_cmd as extended_internal
                Shell_internal = extended_internal.CmdShell
            except:
                Shell_internal = None
            import madgraph.interface.extended_cmd as extended_cmd
            if not isinstance(self.mother, (extended_cmd.CmdShell, Shell_internal)): 
                raise Exception, "scan are not allowed on the Web"
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)
            if self.options['rwgt_name']:
                self.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))  
                          
        # check if "Auto" is present for a width parameter
        tmp_card = new_card.lower().split('block',1)[1]
        if "auto" in tmp_card: 
            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()


        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner and self.output_type == 'default': 
            if 'name=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup name=\'mg_reweighting\'\s*weight_name_strategy=\'includeIdInWeightName\'>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                misc.sprint(blockpat, self.banner['initrwgt'])
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'(?:rwgt_(?P<id>\d+)|(?P<id2>[_\w]+))(?P<rwgttype>\s*|_\w+)\'>(?P<info>.*?)</weight>', re.S+re.I+re.M)
                mg_rwgt_info = pattern.findall(content)
                
                maxid = 0
                for k,(i, fulltag, nlotype, diff) in enumerate(mg_rwgt_info):
                    if i:
                        if int(i) > maxid:
                            maxid = int(i)
                        mg_rwgt_info[k] = (i, nlotype, diff) # remove the pointless fulltag tag
                    else:
                        mg_rwgt_info[k] = (fulltag, nlotype, diff) # remove the pointless id tag
                         
                maxid += 1
                rewgtid = maxid
                if self.options['rwgt_name']:
                    #ensure that the entry is not already define if so overwrites it
                    for (i, nlotype, diff) in mg_rwgt_info[:]:
                        for flag in type_rwgt:
                            if 'rwgt_%s' % i == '%s%s' %(self.options['rwgt_name'],flag) or \
                                i == '%s%s' % (self.options['rwgt_name'], flag):
                                    logger.warning("tag %s%s already defines, will replace it", self.options['rwgt_name'],flag)
                                    mg_rwgt_info.remove((i, nlotype, diff))
                                                
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
        self.new_param_card = check_param_card.ParamCard(s_new.splitlines())
        
        #define tag for the run
        if self.options['rwgt_name']:
            tag = self.options['rwgt_name']
        else:
            tag = str(rewgtid)
        
        if not self.second_model and not self.dedicated_path:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  self.new_param_card
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process:
                    logger.warning(' REWEIGHTING: original card and new card are identical.')
            try:
                if old_param['sminputs'].get(3)- new_param['sminputs'].get(3) > 1e-3 * new_param['sminputs'].get(3):
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception, error:
                logger.debug("error in check of alphas: %s" % str(error))
                pass #this is a security                
            if not self.second_process:
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, card_diff))
            else:
                str_proc = "\n change process  ".join([""]+self.second_process)
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, str_proc + '\n'+ card_diff))
        else:
            if self.second_model:
                str_info = "change model %s" % self.second_model
            else:
                str_info =''
            if self.second_process:
                str_info += "\n change process  ".join([""]+self.second_process)
            if self.dedicated_path:
                for k,v in self.dedicated_path.items():
                    str_info += "\n change %s %s" % (k,v)
            card_diff = str_info
            str_info += '\n' + s_new
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, str_info))
        # re-create the banner.
        self.banner['initrwgt'] = header_rwgt_other
        if self.output_type == 'default':
            self.banner['initrwgt'] += '\n<weightgroup name=\'mg_reweighting\' weight_name_strategy=\'includeIdInWeightName\'>\n'
        else:
            self.banner['initrwgt'] += '\n<weightgroup name=\'main\'>\n'
        for tag, rwgttype, diff in mg_rwgt_info:
            if tag.isdigit():
                self.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
            else:
                self.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')

        logger.info('starts to compute weight for events with the following modification to the param_card:')
        logger.info(card_diff.replace('\n','\nKEEP:'))
        self.run_card = banner.Banner(self.banner).charge_card('run_card')

        if self.options['rwgt_name']:
            tag_name = self.options['rwgt_name']
        else:
            tag_name = 'rwgt_%s' % rewgtid

        #initialise module.
        for (path,tag), module in self.f2pylib.items():
            with misc.chdir(pjoin(os.path.dirname(rw_dir), path)):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):
                    if 'second' in path or tag == 3:
                        module.initialise(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                    else:
                        module.initialise(pjoin(path_me, 'rw_me', 'Cards', 'param_card_orig.dat'))

        return param_card_iterator, tag_name

        
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
    
    def do_compute_widths(self, line):
        return self.mother.do_compute_widths(line)
    
    
    dynamical_scale_warning=True
    def change_kinematics(self, event):
 

        if isinstance(self.run_card, banner.RunCardLO):
            jac = event.change_ext_mass(self.new_param_card)
            new_event = event
        else:
            jac =1
            new_event = event

        if jac != 1:
            if self.output_type == 'default':
                logger.critical('mass reweighting requires dedicated lhe output!. Please include "change output 2.0" in your reweight_card')
                raise Exception
            mode = self.run_card['dynamical_scale_choice']
            if mode == -1:
                if self.dynamical_scale_warning:
                    logger.warning('dynamical_scale is set to -1. New sample will be with HT/2 dynamical scale for renormalisation scale')
                mode = 3
            new_event.scale = event.get_scale(mode)
            new_event.aqcd = self.lhe_input.get_alphas(new_event.scale, lhapdf_config=self.mother.options['lhapdf'])
         
        return jac, new_event
    
    
    def calculate_weight(self, event):
        """space defines where to find the calculator (in multicore)"""
        
        global lhapdf
        
        if self.has_nlo and self.rwgt_mode != "LO":
            return self.calculate_nlo_weight(event)
        
        event.parse_reweight()                    
        orig_wgt = event.wgt
        # LO reweighting    
        w_orig = self.calculate_matrix_element(event, 0)
        
        # reshuffle event for mass effect # external mass only
        # carefull that new_event can sometimes be = to event 
        # (i.e. change can be in place)
        jac, new_event = self.change_kinematics(event)
        
        
        if event.wgt != 0: # impossible reshuffling
            w_new =  self.calculate_matrix_element(new_event, 1)
        else:
            w_new = 0

        if w_orig == 0:
            tag, order = event.get_tag_and_order()
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
            misc.sprint(w_orig, w_new)
            misc.sprint(event)
            misc.sprint(self.invert_momenta(event.get_momenta(orig_order)))
            misc.sprint(event.get_momenta(orig_order))
            misc.sprint(event.aqcd)
            hel_order = event.get_helicity(orig_order)
            if self.helicity_reweighting and 9 not in hel_order:
                nhel = hel_dict[tuple(hel_order)]
            else:
                nhel = 0
            misc.sprint(nhel, Pdir, hel_dict)                        
            raise Exception, "Invalid matrix element for original computation (weight=0)"

        return {'orig': orig_wgt, '': w_new/w_orig*orig_wgt*jac}
     
    def calculate_nlo_weight(self, event):


        type_nlo = self.get_weight_names()
        final_weight = {'orig': event.wgt}
            
        event.parse_reweight()
        event.parse_nlo_weight(threshold=self.soft_threshold) 
        if self.output_type != 'default':
            event.nloweight.modified = True # the internal info will be changed
                                            # so set this flage to True to change
                                            # the writting of those data

        #initialise the input to the function which recompute the weight
        scales2 = []
        pdg = []
        bjx = []
        wgt_tree = [] # reweight for loop-improved type
        wgt_virt  = [] #reweight b+v together
        base_wgt = []
        gs=[]
        qcdpower = []
        ref_wgts = [] #for debugging

        orig_wgt = 0
        for cevent in event.nloweight.cevents:
            #check if we need to compute the virtual for that cevent
            need_V = False # the real is nothing else than the born for a N+1 config
            all_ctype = [w.type for w in cevent.wgts]
            if '_nlo' in type_nlo and any(c in all_ctype for c in [2,14,15]):
                need_V =True
            
            w_orig = self.calculate_matrix_element(cevent, 0)
            w_new =  self.calculate_matrix_element(cevent, 1)
            ratio_T = w_new/w_orig
            if need_V:
                scale2 = cevent.wgts[0].scales2[0]
                #for scale2 in set(c.scales2[1] for c in cevent.wgts): 
                w_origV = self.calculate_matrix_element(cevent, 'V0', scale2=scale2)
                w_newV =  self.calculate_matrix_element(cevent, 'V1', scale2=scale2)                    
                ratio_BV = (w_newV + w_new) / (w_origV + w_orig)
                ratio_V = w_newV/w_origV
            else:
                ratio_V = "should not be used"
                ratio_BV = "should not be used"
            for c_wgt in cevent.wgts:
                orig_wgt += c_wgt.ref_wgt
                #add the information to the input
                scales2.append(c_wgt.scales2)
                pdg.append(c_wgt.pdgs[:2])

                bjx.append(c_wgt.bjks)
                qcdpower.append(c_wgt.qcdpower)
                gs.append(c_wgt.gs)
                ref_wgts.append(c_wgt.ref_wgt)
                
                if '_nlo' in type_nlo:
                    if c_wgt.type in  [2,14,15]:
                        R = ratio_BV
                    else:
                        R = ratio_T
                    
                    new_wgt = [c_wgt.pwgt[0] * R,
                               c_wgt.pwgt[1] * ratio_T,
                               c_wgt.pwgt[2] * ratio_T]
                    wgt_virt.append(new_wgt)

                if '_tree' in type_nlo:
                    new_wgt = [c_wgt.pwgt[0] * ratio_T,
                               c_wgt.pwgt[1] * ratio_T,
                               c_wgt.pwgt[2] * ratio_T]
                    wgt_tree.append(new_wgt)
                    
                base_wgt.append(c_wgt.pwgt[:3])
        
        #change the ordering to the fortran one:
        scales2 = self.invert_momenta(scales2)
        pdg = self.invert_momenta(pdg)
        bjx = self.invert_momenta(bjx)
        # re-compute original weight to reduce numerical inacurracy
        base_wgt = self.invert_momenta(base_wgt)
        
        orig_wgt_check, partial_check = self.combine_wgt(scales2, pdg, bjx, base_wgt, gs, qcdpower, 1., 1.)
        
        if '_nlo' in type_nlo:
            wgt = self.invert_momenta(wgt_virt)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                new_out, partial = self.combine_wgt(scales2, pdg, bjx, wgt, gs, qcdpower, 1., 1.)
            # try to correct for precision issue
            avg = [partial_check[i]/ref_wgts[i] for i in range(len(ref_wgts))]
            out = sum(partial[i]/avg[i] if 0.85<avg[i]<1.15 else 0 \
                          for i in range(len(avg)))
            final_weight['_nlo'] = out/orig_wgt*event.wgt

            
        if '_tree' in type_nlo:
            wgt = self.invert_momenta(wgt_tree)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                out, partial = self.combine_wgt(scales2, pdg, bjx, wgt, gs, qcdpower, 1., 1.)
            # try to correct for precision issue
            avg = [partial_check[i]/ref_wgts[i] for i in range(len(ref_wgts))]
            new_out = sum(partial[i]/avg[i] if 0.85<avg[i]<1.15 else partial[i] \
                          for i in range(len(avg)))
            final_weight['_tree'] = new_out/orig_wgt*event.wgt    
                  
             
        if '_lo' in type_nlo:
            w_orig = self.calculate_matrix_element(event, 0)
            w_new =  self.calculate_matrix_element(event, 1)            
            final_weight['_lo'] = w_new/w_orig*event.wgt
            
            
        if self.output_type != 'default' and len(type_nlo)==1 and '_lo' not in type_nlo:
            to_write = [partial[i]/ref_wgts[i]*partial_check[i]
                             if 0.85<avg[i]<1.15 else 0
                              for i in range(len(ref_wgts))]
            for cevent in event.nloweight.cevents:
                for c_wgt in cevent.wgts:
                        c_wgt.ref_wgt = to_write.pop(0)
                        if '_tree' in type_nlo:
                            c_wgt.pwgt = wgt_tree.pop(0)
                        else:
                            c_wgt.pwgt = wgt_virt.pop(0)
            assert not to_write
            assert not wgt_tree
        return final_weight 
        
     
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
    
    def calculate_matrix_element(self, event, hypp_id, scale2=0):
        """routine to return the matrix element"""
        
        if self.has_nlo:
            nb_retry, sleep = 10, 60 
        else:
            nb_retry, sleep = 5, 20 
        
        tag, order = event.get_tag_and_order()
        if isinstance(hypp_id, str) and hypp_id.startswith('V'):
            tag = (tag,'V')
            hypp_id = int(hypp_id[1:])
        #    base = "rw_mevirt"
        #else:
        #    base = "rw_me"
        
        if (not self.second_model and not self.second_process and not self.dedicated_path) or hypp_id==0:
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
        else:
            orig_order, Pdir, hel_dict = self.id_to_path_second[tag] 
            
        base = os.path.basename(os.path.dirname(Pdir))
        if '_second' in base:
            moduletag = (base, 2)
        else:
            moduletag = (base, 2+hypp_id)
        
        module = self.f2pylib[moduletag]

        p = event.get_momenta(orig_order)
        # add helicity information
        
        hel_order = event.get_helicity(orig_order)
        if self.helicity_reweighting and 9 not in hel_order:
            nhel = hel_dict[tuple(hel_order)]                
        else:
            nhel = -1
            
        # For 2>N pass in the center of mass frame
        #   - required for helicity by helicity re-weighitng
        #   - Speed-up loop computation 
        if (hasattr(event[1], 'status') and event[1].status == -1) or \
           (event[1].px == event[1].py == 0.):
            pboost = lhe_parser.FourMomentum(p[0]) + lhe_parser.FourMomentum(p[1])
            for i,thisp in enumerate(p):
                p[i] = lhe_parser.FourMomentum(thisp).zboost(pboost).get_tuple()
            assert p[0][1] == p[0][2] == 0 == p[1][2] == p[1][2] == 0 
        
        pold = list(p)
        p = self.invert_momenta(p)
        pdg = list(orig_order[0])+list(orig_order[1])

        with misc.chdir(Pdir):
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                me_value = module.smatrixhel(pdg,p, event.aqcd, scale2, nhel)
                                
        # for loop we have also the stability status code
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
        if self.exitted:
            return
        self.exitted = True
        
        if 'init' in self.banner:
            cross = 0 
            error = 0
            for line in self.banner['init'].split('\n'):
                split = line.split()
                if len(split) == 4:
                    cross, error = float(split[0]), float(split[1])
                    
        if not self.multicore == 'create':
            # No print of results for the multicore mode for the one printed on screen
            if 'orig' not in self.all_cross_section:
                logger.info('Original cross-section: %s +- %s pb' % (cross, error))
            else: 
                logger.info('Original cross-section: %s +- %s pb (cross-section from sum of weights: %s)' % (cross, error, self.all_cross_section['orig'][0]))
            logger.info('Computed cross-section:')
            keys = self.all_cross_section.keys()
            keys.sort()
            for key in keys:
                if key == 'orig':
                    continue
                logger.info('%s : %s +- %s pb' % (key[0] if not key[1] else '%s%s' % key,
                    self.all_cross_section[key][0],self.all_cross_section[key][1] ))  
        self.terminate_fortran_executables()
    
        if self.rwgt_dir and self.multicore == False:
            self.save_to_pickle()
        
        with misc.stdchannel_redirected(sys.stdout, os.devnull):
            for run_id in self.calculator:
                del self.calculator[run_id]
            del self.calculator
        
            
    def __del__(self):
        self.do_quit('')

    
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        

    @misc.mute_logger()
    def create_standalone_tree_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""
        
        mgcmd = self.mg5cmd         
        path_me = data['path'] 
        # 2. compute the production matrix element -----------------------------
        has_nlo = False  
        mgcmd.exec_cmd("set group_subprocesses False")

        if not second:
            logger.info('generating the square matrix element for reweighting')
        else:
            logger.info('generating the square matrix element for reweighting (second model and/or processes)')
        start = time.time()
        commandline=''
        for i,proc in enumerate(data['processes']):
            if '[' not in proc:
                commandline += "add process %s ;" % proc
            else:
                has_nlo = True
                if self.banner.get('run_card','ickkw') == 3:
                    if len(proc) == min([len(p.strip()) for p in data['processes']]):
                        commandline += self.get_LO_definition_from_NLO(proc, self.model)
                    else:
                        commandline += self.get_LO_definition_from_NLO(proc,
                                                     self.model, real_only=True)
                else:
                    commandline += self.get_LO_definition_from_NLO(proc, self.model)
        
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        try:
            mgcmd.exec_cmd(commandline, precmd=True, errorhandling=False)
        except diagram_generation.NoDiagramException:
            commandline=''
            for proc in data['processes']:
                if '[' not in proc:
                    raise
                # pass to virtsq=
                base, post = proc.split('[',1)
                nlo_order, post = post.split(']',1)
                if '=' not in nlo_order:
                    nlo_order = 'virt=%s' % nlo_order
                elif 'noborn' in nlo_order:
                    nlo_order = nlo_order.replace('noborn', 'virt')
                commandline += "add process %s [%s] %s;" % (base,nlo_order,post)
            commandline = commandline.replace('add process', 'generate',1)
            logger.info("RETRY with %s", commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            has_nlo = False
        except Exception, error:
            raise
        
        commandline = 'output standalone_rw %s --prefix=int' % pjoin(path_me,data['paths'][0])
        mgcmd.exec_cmd(commandline, precmd=True)        
        logger.info('Done %.4g' % (time.time()-start))
        self.has_standalone_dir = True
        

        # 3. Store id to directory information ---------------------------------
        if False:
            # keep this for debugging
            matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
            
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
                    Pdir = pjoin(path_me, data['paths'][0], 'SubProcesses', 
                                      'P%s' % me.get('processes')[0].shell_string())
    
                    if not os.path.exists(Pdir):
                        to_check.append(tag)
                        continue                        
                    if tag in data['id2path']:
                        if not Pdir == data['id2path'][tag][1]:
                            misc.sprint(tag, Pdir, data['id2path'][tag][1])
                            raise self.InvalidCmd, '2 different process have the same final states. This module can not handle such situation'
                        else:
                            continue
                    # build the helicity dictionary
                    hel_nb = 0
                    hel_dict = {9:0} # unknown helicity -> use full ME
                    for helicities in me.get_helicity_matrix():
                        hel_nb +=1 #fortran starts at 1
                        hel_dict[tuple(helicities)] = hel_nb
    
                    data['id2path'][tag] = [order, Pdir, hel_dict]        
     
            for tag in to_check:
                if tag not in self.id_to_path:
                    logger.warning("no valid path for %s" % (tag,))
                    #raise self.InvalidCmd, "no valid path for %s" % (tag,)
        
        # 4. Check MadLoopParam for Loop induced
        if os.path.exists(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat')):
            MLCard = banner.MadLoopParam(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'))
            MLCard.set('WriteOutFilters', False)
            MLCard.set('UseLoopFilter', False)
            MLCard.set("DoubleCheckHelicityFilter", False)
            MLCard.set("HelicityFilterLevel", 0)
            MLCard.write(pjoin(path_me, data['paths'][0], 'SubProcesses', 'MadLoopParams.dat'),
                         pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'), 
                         commentdefault=False)
            
            #if self.multicore == 'create':
            #    print "compile OLP", data['paths'][0]
            #    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][0],'SubProcesses'),
            #                 nb_core=self.mother.options['nb_core'])
        
        if os.path.exists(pjoin(path_me, data['paths'][1], 'Cards', 'MadLoopParams.dat')):
            if self.multicore == 'create':
                print "compile OLP", data['paths'][1]
                # It is potentially unsafe to use several cores, We limit ourself to one for now
                # n_cores = self.mother.options['nb_core']
                n_cores = 1
                misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])
                
        return has_nlo

                
    @misc.mute_logger()
    def create_standalone_virt_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""
                
        mgcmd = self.mg5cmd
        path_me = data['path'] 
        # Do not pass here for LO/NLO_tree
        start = time.time()
        commandline=''
        for proc in data['processes']:
            if '[' not in proc:
                pass
            else:
                proc = proc.replace('[', '[ virt=')
                commandline += "add process %s ;" % proc
        # deactivate golem since it creates troubles
        old_options = dict(mgcmd.options)
        if mgcmd.options['golem'] or mgcmd.options['pjfry']:
            logger.info(" When doing NLO reweighting, MG5aMC cannot use the loop reduction algorithms Golem and/or PJFry++")
        mgcmd.options['golem'] = None            
        mgcmd.options['pjfry'] = None 
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output standalone_rw %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
        mgcmd.exec_cmd(commandline, precmd=True) 
        
        #put back golem to original value
        mgcmd.options['golem'] = old_options['golem']
        mgcmd.options['pjfry'] = old_options['pjfry']
        # update make_opts
        m_opts = {}
        if mgcmd.options['lhapdf']:
            #lhapdfversion = subprocess.Popen([mgcmd.options['lhapdf'], '--version'], 
            #        stdout = subprocess.PIPE).stdout.read().strip()[0]
            m_opts['lhapdf'] = True
            m_opts['f2pymode'] = True
            m_opts['lhapdfversion'] = 5 # 6 always fail on my computer since 5 is compatible but slower always use 5
            m_opts['llhapdf'] = self.mother.get_lhapdf_libdir()                       
        else:
            raise Exception, "NLO reweighting requires LHAPDF to work correctly"

        path = pjoin(path_me,data['paths'][1], 'Source', 'make_opts')             
        common_run_interface.CommonRunCmd.update_make_opts_full(path, m_opts)      
        logger.info('Done %.4g' % (time.time()-start))


        # Download LHAPDF SET
        common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
            mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())
        
        # now store the id information             
        if False:
            # keep it for debugging purposes
            matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()            
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
                    Pdir = pjoin(path_me, data['paths'][1], 'SubProcesses', 
                                      'P%s' % me.get('processes')[0].shell_string())
                    assert os.path.exists(Pdir), "Pdir %s do not exists" % Pdir                        
                    if (tag,'V') in data['id2path']:
                        if not Pdir == data['id2path'][(tag,'V')][1]:
                            misc.sprint(tag, Pdir, self.id_to_path[(tag,'V')][1])
                            raise self.InvalidCmd, '2 different process have the same final states. This module can not handle such situation'
                        else:
                            continue
                    # build the helicity dictionary
                    hel_nb = 0
                    hel_dict = {9:0} # unknown helicity -> use full ME
                    for helicities in me.get_helicity_matrix():
                        hel_nb +=1 #fortran starts at 1
                        hel_dict[tuple(helicities)] = hel_nb
    
                    data['id2path'][(tag,'V')] = [order, Pdir, hel_dict]


    @misc.mute_logger()
    def create_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""
                
        data={}
        if not second:
            data['paths'] = ['rw_me', 'rw_mevirt']
            # model
            info = self.banner.get('proc_card', 'full_model_line')
            if '-modelname' in info:
                data['mg_names'] = False
            else:
                data['mg_names'] = True
            data['model_name'] = self.banner.get('proc_card', 'model')
            #processes
            data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
            data['processes'] += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search('^\s*add\s+process', line)]  
            #object_collector
            #self.id_to_path = {}
            #data['id2path'] = self.id_to_path
        else:
            data['paths'] = ['rw_me_second', 'rw_mevirt_second']
            # model
            if self.second_model:
                data['mg_names'] = True
                if ' ' in self.second_model:
                    args = self.second_model.split()
                    if '--modelname' in args:
                        data['mg_names'] = False
                    data['model_name'] = args[0]
                else:
                    data['model_name'] = self.second_model
            else:
                data['model_name'] = None
            #processes
            if self.second_process:
                data['processes'] = self.second_process
            else:
                data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                                 if line.startswith('generate')]
                data['processes'] += [' '.join(line.split()[2:]) 
                                      for line in self.banner.proc_card
                                      if re.search('^\s*add\s+process', line)]
            #object_collector
            #self.id_to_path_second = {}   
            #data['id2path'] = self.id_to_path_second 
        
        # 0. clean previous run ------------------------------------------------
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        data['path'] = path_me
        try:
            shutil.rmtree(pjoin(path_me,data['paths'][0]))
        except Exception: 
            pass
        try:
            shutil.rmtree(pjoin(path_me, data['paths'][1]))
        except Exception: 
            pass

        # 1. prepare the interface----------------------------------------------
        mgcmd = self.mg5cmd
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
            elif line.startswith('define'):
                try:
                    mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except Exception:
                    pass 
                          
        # 1. Load model---------------------------------------------------------  
        if  not data['model_name'] and not second:
            raise self.InvalidCmd('Only UFO model can be loaded in this module.')
        elif data['model_name']:
            self.load_model(data['model_name'], data['mg_names'], complex_mass)
            modelpath = self.model.get('modelpath')
            if os.path.basename(modelpath) != mgcmd._curr_model['name']:
                name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
                if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                    modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])
                
            commandline="import model %s " % modelpath
            if not data['mg_names']:
                commandline += ' -modelname '
            mgcmd.exec_cmd(commandline)
            
            #multiparticles
            for name, content in self.banner.get('proc_card', 'multiparticles'):
                mgcmd.exec_cmd("define %s = %s" % (name, content))
        
        if  second and 'tree_path' in self.dedicated_path:
            files.ln(self.dedicated_path['tree_path'], path_me,name=data['paths'][0])
            if 'virtual_path' in self.dedicated_path:
                has_nlo=True
            else:
                has_nlo=False
        else:
            has_nlo = self.create_standalone_tree_directory(data, second)

        
        # 5. create the virtual for NLO reweighting  ---------------------------
        if second and 'virtual_path' in self.dedicated_path:
            files.ln(self.dedicated_path['virtual_path'], path_me, name=data['paths'][1])
        elif has_nlo and 'NLO' in self.rwgt_mode:
            self.create_standalone_virt_directory(data, second)
            
            if not second:
                #compile the module to combine the weight
                misc.compile(cwd=pjoin(path_me, data['paths'][1], 'Source'))
                #link it 
                if path_me not in sys.path:
                    sys.path.insert(0, os.path.realpath(path_me))
                with misc.chdir(pjoin(path_me)):
                    mymod = __import__('%s.Source.rwgt2py' % data['paths'][1], globals(), locals(), [],-1)
                    mymod =  mymod.Source.rwgt2py
                    with misc.stdchannel_redirected(sys.stdout, os.devnull):
                        mymod.initialise([self.banner.run_card['lpp1'], 
                                      self.banner.run_card['lpp2']],
                                     self.banner.run_card.get_lhapdf_id())
                    self.combine_wgt = mymod.get_wgt

            if self.multicore == 'create':
                print "compile OLP", data['paths'][1]
                try:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])
                except:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=1)
        elif has_nlo and not second and self.rwgt_mode == ['NLO_tree']:
            # We do not have any virtual reweighting to do but we still have to
            #combine the weights.
            #Idea:create a fake directory.
            start = time.time()
            commandline='import model loop_sm;generate g g > e+ ve [virt=QCD]'
            # deactivate golem since it creates troubles
            old_options = dict(mgcmd.options)
            mgcmd.options['golem'] = None            
            mgcmd.options['pjfry'] = None 
            commandline = commandline.replace('add process', 'generate',1)
            logger.info(commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            commandline = 'output standalone_rw %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
            mgcmd.exec_cmd(commandline, precmd=True)    
            #put back golem to original value
            mgcmd.options['golem'] = old_options['golem']
            mgcmd.options['pjfry'] = old_options['pjfry']
            # update make_opts
            m_opts = {}
            if mgcmd.options['lhapdf']:
                #lhapdfversion = subprocess.Popen([mgcmd.options['lhapdf'], '--version'], 
                #        stdout = subprocess.PIPE).stdout.read().strip()[0]
                m_opts['lhapdf'] = True
                m_opts['f2pymode'] = True
                m_opts['lhapdfversion'] = 5 # 6 always fail on my computer since 5 is compatible but slower always use 5
                m_opts['llhapdf'] = self.mother.get_lhapdf_libdir()                        
            else:
                raise Exception, "NLO_tree reweighting requires LHAPDF to work correctly"
 
            path = pjoin(path_me,data['paths'][1], 'Source', 'make_opts')             
            common_run_interface.CommonRunCmd.update_make_opts_full(path, m_opts)      
            logger.info('Done %.4g' % (time.time()-start))

            # Download LHAPDF SET
            common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
                mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())
            
            #compile the module to combine the weight
            misc.compile(cwd=pjoin(path_me, data['paths'][1], 'Source'))
            #link it 
            with misc.chdir(pjoin(path_me)):
                if path_me not in sys.path:
                    sys.path.insert(0, path_me)
                mymod = __import__('%s.Source.rwgt2py' % data['paths'][1], globals(), locals(), [],-1)
                mymod =  mymod.Source.rwgt2py
                with misc.stdchannel_redirected(sys.stdout, os.devnull):
                    mymod.initialise([self.banner.run_card['lpp1'], 
                                  self.banner.run_card['lpp2']],
                                 self.banner.run_card.get_lhapdf_id())
                self.combine_wgt = mymod.get_wgt
            
             
        # 6. If we need a new model/process-------------------------------------
        if (self.second_model or self.second_process or self.dedicated_path) and not second :
            self.create_standalone_directory(second=True)    

        if not second:
            self.has_nlo = has_nlo
            
        
        
    def compile(self):
        """compile the code"""
        
        if self.multicore=='wait':
            return
        
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        for onedir in self.rwgt_dir_possibility:
            if not os.path.isdir(pjoin(path_me,onedir)):
                continue
            pdir = pjoin(path_me, onedir, 'SubProcesses')
            if self.mother:
                nb_core = self.mother.options['nb_core'] if self.mother.options['run_mode'] !=0 else 1
            else:
                nb_core = 1
            os.environ['MENUM'] = '2'
            misc.compile(['allmatrix2py.so'], cwd=pdir, nb_core=nb_core)
            if not (self.second_model or self.second_process or self.dedicated_path):
                os.environ['MENUM'] = '3'
                misc.compile(['allmatrix3py.so'], cwd=pdir, nb_core=nb_core)

    def load_module(self, metag=1):
        """load the various module and load the associate information"""
        
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir        
        
        self.id_to_path = {}
        self.id_to_path_second = {}
        for onedir in self.rwgt_dir_possibility:
            if not os.path.exists(pjoin(path_me,onedir)):
                continue 
            pdir = pjoin(path_me, onedir, 'SubProcesses')
            for tag in [2*metag,2*metag+1]:
                with misc.TMP_variable(sys, 'path', [pjoin(path_me)]+sys.path):      
                    mod_name = '%s.SubProcesses.allmatrix%spy' % (onedir, tag)
                    #mymod = __import__('%s.SubProcesses.allmatrix%spy' % (onedir, tag), globals(), locals(), [],-1)
                    if mod_name in sys.modules.keys():
                        del sys.modules[mod_name]
                        tmp_mod_name = mod_name
                        while '.' in tmp_mod_name:
                            tmp_mod_name = tmp_mod_name.rsplit('.',1)[0]
                            del sys.modules[tmp_mod_name]
                        mymod = __import__(mod_name, globals(), locals(), [],-1)  
                    else:
                        mymod = __import__(mod_name, globals(), locals(), [],-1)    
                    
                    S = mymod.SubProcesses
                    mymod = getattr(S, 'allmatrix%spy' % tag)
                
                # Param card not available -> no initialisation
                self.f2pylib[(onedir,tag)] = mymod
                if hasattr(mymod, 'set_madloop_path'):
                    mymod.set_madloop_path(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources'))
                if (self.second_model or self.second_process or self.dedicated_path):
                    break

            data = self.id_to_path
            if '_second' in onedir:
                data = self.id_to_path_second

            # get all the information
            all_pdgs = mymod.get_pdg_order()
            all_pdgs = [[pdg for pdg in pdgs if pdg!=0] for pdgs in  mymod.get_pdg_order()]
            all_prefix = [''.join(j).strip().lower() for j in mymod.get_prefix()]
            prefix_set = set(all_prefix)


            hel_dict={}
            for prefix in prefix_set:
                if hasattr(mymod,'%sprocess_nhel' % prefix):
                    nhel = getattr(mymod, '%sprocess_nhel' % prefix).nhel    
                    hel_dict[prefix] = {}
                    for i, onehel in enumerate(zip(*nhel)):
                        hel_dict[prefix][tuple(onehel)] = i+1
                elif hasattr(mymod, 'set_madloop_path') and \
                     os.path.exists(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper())):
                    hel_dict[prefix] = {}
                    for i,line in enumerate(open(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper()))):
                        onehel = [int(h) for h in line.split()]
                        hel_dict[prefix][tuple(onehel)] = i+1
                else:
                    misc.sprint(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper() ))
                    misc.sprint(os.path.exists(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper())))
                    continue

            for i,pdg in enumerate(all_pdgs):
                if self.is_decay:
                    incoming = [pdg[0]]
                    outgoing = pdg[1:]
                else:
                    incoming = pdg[0:2]
                    outgoing = pdg[2:]
                order = (list(incoming), list(outgoing))
                incoming.sort()
                outgoing.sort()
                tag = (tuple(incoming), tuple(outgoing))
                if 'virt' in onedir:
                    tag = (tag, 'V')
                prefix = all_prefix[i]
                hel = hel_dict[prefix]
                if tag in data:
                    oldpdg = data[tag][0][0]+data[tag][0][1]
                    if all_prefix[all_pdgs.index(pdg)] == all_prefix[all_pdgs.index(oldpdg)]:
                        for i in range(len(pdg)):
                            if pdg[i] == oldpdg[i]:
                                continue
                            if not self.model or not hasattr(self.model, 'get_mass'):
                                continue
                            if self.model.get_mass(int(pdg[i])) == self.model.get_mass(int(oldpdg[i])):
                                continue
                            misc.sprint(tag, onedir)
                            misc.sprint(data[tag][:-1])
                            misc.sprint(order, pdir,)
                            raise Exception                                
                    else: 
                        misc.sprint(tag, onedir)
                        misc.sprint(data[tag][:-1])
                        misc.sprint(order, pdir,)
                        raise Exception

                data[tag] = order, pdir, hel
             
             
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
        to_save['has_nlo'] = self.has_nlo
        to_save['rwgt_mode'] = self.rwgt_mode
        to_save['rwgt_name'] = self.options['rwgt_name']

        name = pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl')
        save_load_object.save_to_file(name, to_save)


    def load_from_pickle(self, keep_name=False):
        import madgraph.iolibs.save_load_object as save_load_object
        
        obj = save_load_object.load_from_file( pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl'))
        
        self.has_standalone_dir = True
        self.options = {'curr_dir': os.path.realpath(os.getcwd()),
                        'rwgt_name': None}
        if keep_name:
            self.options['rwgt_name'] = obj['rwgt_name']
        
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
        self.has_nlo = obj['has_nlo']
        if not self.rwgt_mode:
            self.rwgt_mode = obj['rwgt_mode']
            logger.info("mode set to %s" % self.rwgt_mode)
        if self.has_nlo and 'NLO' in self.rwgt_mode:
            path = pjoin(obj['rwgt_dir'], 'rw_mevirt','Source')
            sys.path.insert(0, path)
            try:
                mymod = __import__('rwgt2py', globals(), locals())
            except ImportError:
                misc.compile(['rwgt2py.so'], cwd=path)
                mymod = __import__('rwgt2py', globals(), locals())
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                mymod.initialise([self.banner.run_card['lpp1'], 
                              self.banner.run_card['lpp2']],
                             self.banner.run_card.get_lhapdf_id())
            self.combine_wgt = mymod.get_wgt
                    
        
        
        





        
