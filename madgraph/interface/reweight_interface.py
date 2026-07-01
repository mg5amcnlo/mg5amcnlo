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
from __future__ import absolute_import
import difflib
import logging
import copy
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

import madgraph
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

#lhapdf = None


class ReweightInterface(extended_cmd.Cmd):
    """Basic interface for reweighting operation"""

    prompt = 'Reweight>'
    debug_output = 'Reweight_debug'
    sa_class = 'standalone_rw'
    nb_rw=0
    
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
                        'rwgt_name':None,
                        "allow_missing_finalstate":False,
                        "identical_particle_in_prod_and_decay": "average"}

        self.events_file = None
        self.processes = {}
        self.f2pylib = {}
        self.second_model = None
        self.second_process = None
        self.nb_library = 1
        self.dedicated_path = {}
        self.soft_threshold = None
        self.systematics = False # allow to run systematics in ouput2.0 mode
        self.boost_event = False
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
        self.keep_ordering = False
        self.use_eventid = False
        self.inc_sudakov = False
        self.event_path = event_path
        self.path2prefix = {} # store the f2pyprefix associated to a library 
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path, allow_madspin=allow_madspin)
            
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        
        #all the cross-section for convenience
        self.all_cross_section = {}

        #If we are using the DensityInterface
        self.flag_density_matrix = False
            
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

            if self.mother and 'OLP' in self.mother.options:
                if self.mother.options['OLP'].lower() != 'madloop':
                    logger.warning("Accurate NLO mode only works for OLP=MadLoop not for OLP=%s. An approximate (LO) reweighting will be performed instead")
                    self.rwgt_mode = 'LO'
            
            if self.mother and 'lhapdf' in self.mother.options and not self.mother.options['lhapdf']:
                logger.warning('NLO accurate reweighting requires lhapdf to be installed. Pass in approximate LO mode.')
                self.rwgt_mode = 'LO'
        else:
            self.rwgt_mode = 'LO'

        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception(msg)
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        self.is_decay = len(process.split('>',1)[0].split()) == 1 
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)

    @staticmethod
    def get_LO_definition_from_NLO(proc, model, real_only=False, ewsudakov=False):
        """return the LO definitions of the process corresponding to the born/real"""
        
        # split the line definition with the part before and after the NLO tag
        process, order, final = re.split(r'\[\s*(.*)\s*\]', proc)
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
            for pdg in pert[:]:
                if pdg in model.merged_particles:
                    pert.remove(pdg)
                    pert += model.merged_particles[pdg]
                elif -pdg in model.merged_particles:
                    pert.remove(pdg)
                    pert += [-i for i in model.merged_particles[-pdg]]
            pert.sort()                     
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
            result = re.split(r'([/$@]|\w+(?:^2)?(?:=|<=|>)+\w+)', process, 1)                    
            if len(result) ==3:
                process, split, rest = result
                commandline+="add process %s pert_%s %s%s %s --no_warning=duplicate;" % (process, order.replace(' ','') ,split, rest, final)
            else:
                commandline +='add process %s pert_%s %s --no_warning=duplicate;' % (process,order.replace(' ',''), final)
            if ewsudakov:
                # EW sudakov reweight
                # this is a NLO-type generation, so [LOonly=QCD] must be added, toghether
                # with the proper flag for the EW sudakov.
                # Also, --no_warning=duplicate can be removed
                commandline = commandline.replace("--no_warning=duplicate", "[LOonly=QCD] --ewsudakov")
        elif order.startswith(('noborn')):
            # pass in sqrvirt=
            return "add process %s [%s] %s;" % (process, order.replace('noborn', 'sqrvirt'), final)
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
        
        bannerfile = banner.Banner(self.lhe_input.banner)
        start = time.time()
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')

            try:
                event.check() #check 4 momenta/...
            except Exception as error:
                print(event)
                raise error
            
            # check that event does not have more energy than the beam
            if event[1].status == -1:
                if event[0].E > bannerfile.get('run_card', 'ebeam1'):
                    print(event)
                    raise Exception("Event %s has more energy than the beam" % event_nb)
                if event[1].E > bannerfile.get('run_card', 'ebeam2'):
                    print(event)
                    raise Exception("Event %s has more energy than the beam" % event_nb)

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

        print("change model X :use model X for the reweighting")
        print("change process p p > e+ e-: use a new process for the reweighting")
        print("change process p p > mu+ mu- --add : add one new process to existing ones")
        print("change output [default|2.0|unweight]:")
        print("               default: add weight(s) to the current file")    

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
        elif args[0] in ["keep_ordering", "use_eventid"]:
            setattr(self, args[0], banner.ConfigFile.format_variable(args[1], bool, args[0]))
        elif args[0] == "allow_missing_finalstate":
            self.options["allow_missing_finalstate"] = banner.ConfigFile.format_variable(args[1], bool, "allow_missing_finalstate")
        elif args[0] == "process":
            nb_f2py_module += 1
            if self.has_standalone_dir:
                self.terminate_fortran_executables()
                self.has_standalone_dir = False
            if args[-1] == "--add":
                self.second_process.append(" ".join(args[1:-1]))
            else:
                self.second_process = [" ".join(args[1:])]
        elif args[0] == "boost":
            self.boost_event = eval(' '.join(args[1:]))
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
                except Exception as error:
                    self.systematics = args[1:]
            else:
                self.systematics = args[1:]
        elif args[0] == 'soft_threshold':
            self.soft_threshold = banner.ConfigFile.format_variable(args[1], float, 'soft_threshold')
        elif args[0] == 'multicore':
            pass 
            # this line is meant to be parsed by common_run_interface and change the way this class is called.
            #It has no direct impact on this class.
        elif args[0] == "identical_particle_in_prod_and_decay":
            if args[1].lower() not in ['average', 'max', 'crash']:
                raise Exception("option identical_particle_in_prod_and_decay can only be one of the following ['average', 'max', 'crash']")
            self.options[args[0]] = args[1].lower()
        elif args[0] == 'include_sudakov':
            if args[1] == 'True':
                self.inc_sudakov = True
                self.rwgt_mode = 'LO'
        else:
            logger.critical("unknown option! %s.  Discard line." % args[0])
        
            
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.lhe_input:
            if isinstance(self.lhe_input, lhe_parser.EventFile):
                self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
            else:
                raise self.InvalidCmd("No events files defined.")
            
        opts = {'rwgt_name':None, 'rwgt_info':None}
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
        mgcmd = self.mg5cmd
        if opts['rwgt_name']:
            self.options['rwgt_name'] = opts['rwgt_name']
        if opts['rwgt_info']:
            self.options['rwgt_info'] = opts['rwgt_info']
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

        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
        
        scale_rwgt_info=[]
        if 'initrwgt' in self.banner and self.output_type == 'default': 
            for i in self.banner['initrwgt'].split('\n'):
                if "weight id" in i:
                    start = i.find('weight')
                    scale_rwgt_info.append(i[start+11:start+15])
                    
        if self.inc_sudakov:
            type_rwgt=[]
            for tag in scale_rwgt_info:
                tag_strip=tag[1:]
                type_rwgt.append('2'+tag_strip)


            
        # get iterator over param_card and the name associated to the current reweighting.
        param_card_iterator, tag_name = self.handle_param_card(model_line, args, type_rwgt)

        return self.launch_actual_reweighting(param_card_iterator, 
                                              tag_name,
                                              type_rwgt,
                                              path_me)



    def launch_actual_reweighting(self, param_card_iterator, 
                                              tag_name,
                                              type_rwgt,
                                              path_me):
       
        if self.inc_sudakov:
            tag_name = ''
            import importlib
            import numpy as np
            rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
            for onedir in rwgt_dir_possibility:
                if not os.path.exists(pjoin(path_me,onedir)):
                    continue 

                sys.path.insert(0, path_me)
                sud_mod = importlib.import_module('%s.bin.internal.ewsud_pydispatcher' % onedir)
            logger.info('EW Sudakov reweight module imported')

        if type_rwgt==[]:
            type_rwgt=['2001']

        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')

         
        start = time.time()
        # initialize the collector for the various re-weighting
        cross, ratio, ratio_square,error = {},{},{}, {}
        for name in type_rwgt + ['orig']:
            cross[name], error[name] = 0.,0.
            ratio[name],ratio_square[name] = 0., 0.# to compute the variance and associate error

        if self.output_type == "default":
            output = open( self.lhe_input.path +'rw', 'w')
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
        count_errors = 0
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
            if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')

            if self.inc_sudakov:
                weight = self.calculate_weight(event, sud_mod)
            else:
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
                if tag in event.reweight_order:
                    logger.critical('This is a reweighted event file! Do not reweight with ewsudakov twice')
                    return
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
        if self.run_card and 'event_norm' in self.run_card:
            if self.run_card['event_norm'] in ['average','bias']:
                for key, value in cross.items():
                    cross[key] = value / (event_nb+1)
                
        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))     
        if self.inc_sudakov:
            logger.info('Number of events thrown away due to large Sudakov: %s' % str(count_errors))   
        
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
                #output[key].write('</LesHouchesEvents>\n')
                #output.close()
                lhe = lhe_parser.EventFile(output[key].name)
                nb_event = lhe.unweight(target)
                if self.mother and  hasattr(self.mother, 'results'):
                    results = self.mother.results
                    results.add_detail('nb_event', nb_event)
                    results.current.parton.append('lhe')
                logger.info('Event %s is now unweighted under the new theory: %s(%s)' % (lhe.name, target, nb_event))                
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
            if self.options['rwgt_name']:
                reweight_name = self.options['rwgt_name'].rsplit('_',1)[0] # to avoid side effect during the scan
            else:
                reweight_name = None
            for i,card in enumerate(param_card_iterator):
                if reweight_name:
                    self.options['rwgt_name'] = '%s_%s' % (reweight_name, i+1)
                self.new_param_card = card
                #card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                self.exec_cmd("launch --keep_card", printcmd=False, precmd=True)
        
        self.options['rwgt_name'] = None

    def setup_f2py_interface(self):
        """ ensure that all f2py interface are ready/loaded/compiled
            if this function does not return None then nothing is executed after this
            usefull for some plugin
        """

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



    def handle_param_card(self, model_line, args, type_rwgt):
        

        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')
        if not '--keep_card' in args:
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_me', 'rw_mevirt')
            with open(pjoin(rw_dir, 'Cards', 'param_card.dat'), 'w') as fsock:
                fsock.write(self.banner['slha']) 
            out, cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'],
                                ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line,
                                write_file=False, return_instance=True
                                )
            self.stored_line = None
            card = cmd.param_card
            new_card = card.write()
        elif self.new_param_card:
            new_card = self.new_param_card.write()
        else:
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()
        
        # check for potential scan in the new card 
        pattern_scan = re.compile(r'''^(decay)?[\s\d]*scan''', re.I+re.M) 
        param_card_iterator = []
        if pattern_scan.search(new_card):
            import madgraph.interface.extended_cmd as extended_cmd
            try:
                import internal.extended_cmd as extended_internal
                Shell_internal = extended_internal.CmdShell
            except:
                Shell_internal = None
            if not isinstance(self.mother, (extended_cmd.CmdShell, Shell_internal)): 
                raise Exception("scan are not allowed on the Web")
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)
            if self.options['rwgt_name']:
                self.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            self.new_param_card = first_card
            #first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))  

        # check if "Auto" is present for a width parameter)
        if 'block' not in new_card.lower():
            raise Exception(str(new_card))
        tmp_card = new_card.lower().split('block',1)[1]
        if "auto" in tmp_card: 
            if param_card_iterator:
                first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            else:
                ff = open(pjoin(rw_dir, 'Cards', 'param_card.dat'),'w')
                ff.write(new_card)
                ff.close()
                
            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()

        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner and self.output_type == 'default':
            if 'name=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup name=\'mg_reweighting\'\s*weight_name_strategy=\'includeIdInWeightName\'>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'(?:rwgt_(?P<id>\\d+)|(?P<id2>[_\\w\\-\\.]+))(?P<rwgttype>\\s*|_\\w+)\'>(?P<info>.*?)</weight>', re.S+re.I+re.M)
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
        self.orig_param_card_text = s_orig
        s_new = new_card
        if self.flag_density_matrix: #for the density mode we don't use rw_me/Cards/param_card.dat
            self.new_param_card = check_param_card.ParamCard(s_orig.splitlines())
        else:
            self.new_param_card = check_param_card.ParamCard(s_new.splitlines())
        #define tag for the run
        if self.options['rwgt_name']:
            tag = self.options['rwgt_name']
        else:
            tag = str(rewgtid)

        if 'rwgt_info' in self.options and self.options['rwgt_info']:
            card_diff = self.options['rwgt_info']
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, self.options['rwgt_info']))
        elif not self.second_model and not self.dedicated_path:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  self.new_param_card
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process and not self.flag_density_matrix: #if we are in the density mode, the param_card are not modified, warning useless
                    logger.warning(' REWEIGHTING: original card and new card are identical.')
            try:
                if old_param['sminputs'].get(3)- new_param['sminputs'].get(3) > 1e-3 * new_param['sminputs'].get(3):
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception as error:
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
            if self.inc_sudakov:
                try:
                    sud_order = int(rwgttype[-1]) -1
                    sud_order = '10' +rwgttype[-2:]
                    self.banner['initrwgt'] += '<weight id=\'%s\'>%sscale_%s_sud</weight>\n' % \
                            (rwgttype, diff, sud_order)
                except IndexError:
                    logger.critical('This is a reweighted event file! Do not reweight with ewsudakov twice')
                    sys.exit(1)
            else:
                if tag.isdigit():
                    self.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                    (tag, rwgttype, diff)
                else:
                    self.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                    (tag, rwgttype, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')

        if self.flag_density_matrix:
            logger.info('starts to compute density matrices for events with the inputs from the reweight_card:')
        else:
            logger.info('starts to compute weight for events with the following modification to the param_card:')
            logger.info(card_diff.replace('\n','\nKEEP:'))
        try:
            self.run_card = banner.Banner(self.banner).charge_card('run_card')
        except Exception:
            logger.debug('no run card found -- reweight interface')
            self.run_card = None

        if self.options['rwgt_name']:
            tag_name = self.options['rwgt_name']
        else:
            tag_name = 'rwgt_%s' % rewgtid

        #initialise module.
        for (path,tag), module in self.f2pylib.items():

            with misc.chdir(pjoin(os.path.dirname(rw_dir), path)):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):                    
                    if 'rw_me_' in path or tag == 3:
                        param_card = self.new_param_card
                    else:
                        param_card = check_param_card.ParamCard(self.orig_param_card_text)
                    module.initialise('../Cards/param_card.dat')
                    for block in param_card:
                        if block.lower() == 'qnumbers':
                            continue
                        for param   in param_card[block]:
                            lhacode = param.lhacode
                            value = param.value
                            name = '%s_%s' % (block.upper(), '_'.join([str(i) for i in lhacode]))
                            module.change_para(name, value)
                        if param_card[block].scale:
                            name = "mdl__%s__scale" % block.upper()
                            module.change_para(name, param_card[block].scale)

                    #check for running attribute
                    update_running_info = False
                    if tag == 2:
                        if not self.model:
                            update_running_info = True
                        elif  self.model["running_elements"]:
                            update_running_info = True
                    elif self.second_model:
                        if self.second_model["running_elements"]:
                            update_running_info = True
                    elif  not self.model:
                        update_running_info = True
                    elif self.model["running_elements"]:
                        update_running_info = True
                    if update_running_info:
                        try:
                            run_card = banner.RunCard(self.banner.get('run_card'))
                            module.set_fixed_extra_scale(run_card['fixed_extra_scale'])
                            module.set_mue_over_ref(run_card['mue_over_ref'])
                            module.set_mue_ref_fixed(run_card['mue_ref_fixed'])
                            module.set_maxjetflavor(run_card['maxjetflavor'])
                            module.set_asmz(param_card.get('sminputs').get((3,)).value)
                            module.set_nloop(2)
                        except Exception:
                            if self.model:
                                raise
                    module.update_all_coup()
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


    def calculate_weight(self, event, sud_mod=None):
        """space defines where to find the calculator (in multicore)"""
        
        if not self.inc_sudakov:
            if self.has_nlo and self.rwgt_mode != "LO":
                if not hasattr(self,'pdf'):
                    lhapdf = misc.import_python_lhapdf(self.mg5cmd.options['lhapdf'])
                    self.pdf = lhapdf.mkPDF(self.banner.run_card.get_lhapdf_id())
                
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
                raise Exception("Invalid matrix element for original computation (weight=0)")

            return {'orig': orig_wgt, '': w_new/w_orig*orig_wgt*jac}
        else:
            buff_event=copy.deepcopy(event)
            orig_wgt = event.wgt
            w_orig= event.wgt
            pi=3.141592653589
            mW = 80.3

            mgcmd = self.mg5cmd
            import importlib
            import numpy as np

            # identify the process
            try:
                process = self.banner.get_detail('proc_card', 'generate')
            except KeyError:
                process = self.banner.get_detail('proc_card', 'add process')
            process, opts = mg_interface.MadGraphCmd.split_process_line(process)
            nexternal = len([p for p in process.split() if p != '>'])

            # Remove all propagator particles from the event to be passed to Sud module
            for ip,part in enumerate(list(buff_event)):
                if (abs(part.status) != 1):
                    buff_event.pop(ip)

            x = 1.0
            sud_cut= x*(mW**2)
            min_inv=1000000000.0   # dummy variable, used to start the loop of min_inv finding
            inv_dict={}

            if (len(buff_event) == nexternal +1): # is an H-event
                for ievt,evt in enumerate(buff_event):
                    # Find the smallest abs(inv) and the corresponding pair
                    if (ievt <= 1):
                        sign1 = 1.0
                    else:
                        sign1 = -1.0
                    for ievt2,evt2 in enumerate(buff_event):
                        if (ievt2 <= 1):
                            sign2 = 1.0
                        else:
                            sign2 = -1.0
                        if (ievt2 > ievt):
                            inv = (sign1*evt.E+sign2*evt2.E)**2-(sign1*evt.px+sign2*evt2.px)**2\
                                            - (sign1*evt.py+sign2*evt2.py)**2-(sign1*evt.pz+sign2*evt2.pz)**2
                            inv_dict[(ievt,ievt2)] = abs(inv)
                            if (abs(inv) < min_inv):
                                min_inv=abs(inv)
                                min_i=ievt
                                min_j=ievt2
                            
                # Below finds the current process tag and tries to recombine the min_i and min_j
                tag, order = buff_event.get_tag_and_order()
                matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()

                ij_comb= []
                if min_i <= 1:
                    state = False
                else:
                    state = True
                comb_i = fks_common.FKSLeg({'id': buff_event[min_i].pid,'number': min_i+1,'state': state})
                if min_j<= 1:
                    state = False
                else:
                    state = True
                comb_j = fks_common.FKSLeg({'id': buff_event[min_j].pid,'number': min_j+1,'state': state})
                ij_comb =fks_common.combine_ij(comb_i,comb_j, self.model, dict={},pert='QCD')
                if ij_comb == []:
                    ij_comb =fks_common.combine_ij(comb_j,comb_i, self.model, dict={},pert='QCD')
        
                # For n+1-body reweighting
                if min_inv > sud_cut:
                    event_to_sud = buff_event
                    n_part = nexternal+1
                    mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
                    type = 1  #### H1 type

                # For n-body reweighting
                else:
                    # If no reasonable recbination found, still use the n+1-body kinematics for sudakov
                    if ij_comb == []:
                        event_to_sud = buff_event
                        n_part = nexternal+1
                        mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
                        #### H1 type
                        type = 1
                    else:
                        buff_event.merge_particles_kinematics(min_i,min_j,ij_comb)
                        event_to_sud = buff_event
                        n_part = nexternal 
                        mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
                        # map to n+1 body if recoil does not exist at Born level among processes
                        if mapped_tag not in sud_mod.pdg2ewsud_dict.keys():
                            event_to_sud = buff_event
                            n_part = nexternal+1
                            mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
                            type = 1   #### H1 type
                        else:
                            type = 2    #### H2 type                 
            elif (len(buff_event) == nexternal): # is an S-event
                    event_to_sud = buff_event
                    n_part = nexternal 
                    mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
                    type = 0   ### S type
            else:
                logger.critical('ERROR: neither H nor S event!')
                logger.critical(buff_event)
                sys.exit(2)
                
            # Boost to partonic CM frame if not already in one for the momentum reshuffling 
            p = lhe_parser.FourMomentum()
            for i,particle in enumerate(event_to_sud):
                    if particle.status == -1:
                        p += particle
            in_part_mom = p
            if not ((abs(p.px) < 1e-6 * p.E) and (abs(p.py) < 1e-6 * p.E) and (abs(p.pz) < 1e-6 * p.E)):
                event_to_sud.boost(in_part_mom)

            # Rotate system to a partonic CM along z-axis
            initial = copy.deepcopy(event_to_sud[0])
            if not ((abs(initial.px) < 1e-6 * initial.E) and (abs(initial.py) < 1e-6 * initial.E)):
                for p in event_to_sud:
                    p.set_momentum(lhe_parser.FourMomentum(p).rotate_to_z(prot=lhe_parser.FourMomentum(initial)))
            
            # Set all light quarks and lepton masses to zero in event file
            #self.set_final_jet_mass_to_zero(event_to_sud)
            event_to_sud.set_final_jet_mass_to_zero()
            # Set finally all initial masses to zero rather than the masses in the event files
            event_to_sud.set_initial_mass_to_zero()
            event_to_sud.check_kinematics_only()

            gstr = math.sqrt(4*pi*event.aqcd)
            sorted_tag = (tuple(mapped_order[0]),tuple(sorted(mapped_order[1])))

            # Read in the event momenta into np array in the order which is defined in the process directory
            perm = []
            i = 1
            for r in mapped_order[1]:
                if not list(sud_mod.original_pdg_list_dict[sorted_tag][1]).index(r) in perm:
                    perm.append(list(sud_mod.original_pdg_list_dict[sorted_tag][1]).index(r))
                else:
                    perm.append(list(sud_mod.original_pdg_list_dict[sorted_tag][1]).index(r)+i)
                    i += 1
            order = dict((i,j) for i,j in enumerate(perm))
            event_to_sud_order = copy.deepcopy(event_to_sud)
            event_to_sud_order[:len(mapped_tag[0])] =  event_to_sud[:len(mapped_tag[0])]
            for r in order.keys():
                event_to_sud_order[r+len(mapped_tag[0])] = event_to_sud[order[r]+len(mapped_tag[0])]
            p_in = np.zeros(shape=(n_part, 4))
            for i,el in enumerate(event_to_sud_order):
                p_in[i] = [float(el.E),float(el.px),float(el.py),float(el.pz)]
            mapped_tag, mapped_order = event_to_sud_order.get_tag_and_order()
            if list(sud_mod.original_pdg_list_dict[sorted_tag][1]) != mapped_order[1]:
                logger.critical('ERROR: order in particle momenta does not match MG convention!')
                sys.exit(3)

            # compute the actual Sudakov weight
            res = sud_mod.ewsudakov(sorted_tag, p_in, gstr)

            # Do the reewightings
            sudrat0 = 1. + res[1]/res[0] # SUD0            s_to_rij ON  rij_ge_mw ON
            sudrat1 = 1. + res[2]/res[0] # SUD1 (SKD_weak) s_to_rij ON  rij_ge_mw ON
            sudrat2 = 1. + res[3]/res[0] # SUD1 (SKD_weak) s_to_rij OFF  rij_ge_mw ON
            sudrat3 = 1. + res[4]/res[0] # SUD1 (SKD_weak) s_to_rij OFF  rij_ge_mw OFF
            sudrat4 = 1. + res[5]/res[0] # SUD1 (SKD_weak) s_to_rij ON  rij_ge_mw OFF

            # Damp when the Sudakov weights are too large (in abs value)
            large_sud_error=False
            if abs(sudrat1) > 200:
                logger.info('ERROR: event will not be reweighted because Sudakov ratio is too large: %s ' %sudrat1)
                logger.info(buff_event)
                sudrat0 = 1. 
                sudrat1 = 1.
                sudrat2 = 1.
                sudrat3 = 1. 
                sudrat4 = 1.
                large_sud_error = True

            # Dummy step: needed to read in the parse_reweight function
            event.rescale_weights(1.)

            rwgt_dict = copy.deepcopy(event.parse_reweight())
            if rwgt_dict=={}:
                rwgt_dict['1001'] =  orig_wgt
            rwgt_dict_new = {}
            rwgt_dict_new['orig'] = orig_wgt

            for el in rwgt_dict:
                ending = el[-2:]
                tag = '20' + ending
                rwgt_dict_new[tag] = rwgt_dict[el]*sudrat1  # use SDK_weak! 
            
            return rwgt_dict_new
    
    def get_pdg_tuple(self, pdgs, nincoming):
        """write a tuple of 2 tuple, with the incoming particles unsorted
        and the outgoing ones sorted
        """
        incoming = pdgs[:nincoming]
        outgoing = pdgs[nincoming:]
        return (tuple(incoming), tuple(sorted(outgoing)))

    def calculate_nlo_weight(self, event):

        type_nlo = self.get_weight_names()
        final_weight = {'orig': event.wgt}
            
        event.parse_reweight()
        event.parse_nlo_weight(threshold=self.soft_threshold) 
        if not event.nloweight.ispureqcd():
            raise Exception('NLO reweighting does not support mixed expansion mode. Only LO accurate mode is allowed.')
        
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
                w_origV = self.calculate_matrix_element(cevent, 'V0', scale2=scale2**2)
                w_newV =  self.calculate_matrix_element(cevent, 'V1', scale2=scale2**2)                    
                ratio_BV = (w_newV + w_new) / (w_origV + w_orig)
                ratio_V = w_newV/w_origV if w_origV else  "should not be used"
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
        
        
        orig_wgt_check, partial_check = self.combine_wgt_local(scales2, pdg, bjx, base_wgt, gs, qcdpower, self.pdf)
        #change the ordering to the fortran one: 
        #scales2_i = self.invert_momenta(scales2)
        #pdg_i = self.invert_momenta(pdg)
        #bjx_i = self.invert_momenta(bjx)
        # re-compute original weight to reduce numerical inacurracy
        #base_wgt_i = self.invert_momenta(base_wgt)
        #orig_wgt_check, partial_check = self.combine_wgt(scales2_i, pdg_i, bjx_i, base_wgt_i, gs, qcdpower, 1., 1.)
        
        if '_nlo' in type_nlo:
            #wgt = self.invert_momenta(wgt_virt)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                new_out, partial = self.combine_wgt_local(scales2, pdg, bjx, wgt_virt, gs, qcdpower, self.pdf)
            # try to correct for precision issue
            avg = [partial_check[i]/ref_wgts[i] for i in range(len(ref_wgts))]
            out = sum(partial[i]/avg[i] if 0.85<avg[i]<1.15 else 0 \
                        for i in range(len(avg)))
            final_weight['_nlo'] = out/orig_wgt*event.wgt

            
        if '_tree' in type_nlo:
            #wgt = self.invert_momenta(wgt_tree)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                out, partial = self.combine_wgt_local(scales2, pdg, bjx, wgt_tree, gs, qcdpower, self.pdf)
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


    def combine_wgt_local(self, scale2s, pdgs, bjxs, base_wgts, gss, qcdpowers, pdf):

        wgt = 0.
        wgts = []
        for (scale2, pdg, bjx, base_wgt, gs, qcdpower) in   zip(scale2s, pdgs, bjxs, base_wgts, gss, qcdpowers):
            Q2, mur2, muf2 = scale2 #Q2 is Ellis-Sexton scale
            #misc.sprint(Q2, mur2, muf2, base_wgt, gs, qcdpower)
            pdf1 = pdf.xfxQ2(pdg[0], bjx[0], muf2)/bjx[0]
            pdf2 = pdf.xfxQ2(pdg[1], bjx[1], muf2)/bjx[1]
            alphas = pdf.alphasQ2(mur2)
            tmp = base_wgt[0] + base_wgt[1] * math.log(mur2/Q2) + base_wgt[2] * math.log(muf2/Q2)
            tmp *= gs**qcdpower*pdf1*pdf2
            wgt += tmp
            wgts.append(tmp)
        return wgt, wgts
        

    
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

    def _get_revert_merged_for(self, model):
        """Return the {individual_pdg: merged_pdg} mapping for a given model.

        The merged PDG codes (81 for jets, 82 for charged leptons, ...) are
        defined inside the model object itself (model['merged_particles'])
        when apply_flavor_grouping is on. For models where flavor grouping
        is not applied -- automatically the case for any loop / perturbative
        model, and also the case for every reweight model load since the
        reweight forces apply_flavor_grouping=False -- the dict is empty
        and we return None, so callers treat event PDGs as-is (the
        pre-flavor-grouping behavior)."""
        if model is None:
            return None
        merged = model.get('merged_particles')
        if not merged:
            return None
        rm = {}
        for key, value in merged.items():
            for val in value:
                rm[val] = key
        return rm

    def calculate_matrix_element(self, event, hypp_id, scale2=0):
        """routine to return the matrix element"""

        if self.has_nlo:
            nb_retry, sleep = 10, 60
        else:
            nb_retry, sleep = 5, 20

        # The merged-particle mapping is taken from the model that built the
        # id_to_path being looked up: id_to_path is from the original model
        # (self.original_model), id_to_path_second from self.model (the
        # second-model load overwrote self.model). When apply_flavor_grouping
        # is off (the default in reweight contexts, and the automatic case
        # for any loop/perturbative model), merged_particles is empty for
        # both and revert_merged stays None -- giving the pre-flavor-grouping
        # behavior (raw event PDGs everywhere).
        is_virtual_tag = isinstance(hypp_id, str) and hypp_id.startswith('V')
        use_original = (not self.second_model and not self.second_process and
                        not self.dedicated_path) or hypp_id == 0 or is_virtual_tag
        if use_original:
            relevant_model = getattr(self, 'original_model', None) or self.model
        else:
            relevant_model = self.model
        if relevant_model:
            self.revert_merged = self._get_revert_merged_for(relevant_model)

        tag_orig, order = event.get_tag_and_order(None)
        tag, order = event.get_tag_and_order(self.revert_merged)
        if self.keep_ordering:
            old_tag = tuple(tag)
            tag = (tag[0], tuple(order[1]))

        if is_virtual_tag:
            tag = (tag,'V')
            hypp_id = int(hypp_id[1:])
        #    base = "rw_mevirt"
        #else:
        #    base = "rw_me"

        if (not self.second_model and not self.second_process and not self.dedicated_path) or hypp_id==0:
            if tag in self.id_to_path:
                orig_order, Pdir, hel_dict = self.id_to_path[tag]
            else:
                cross_tag = self.get_crossing_tag(tag)
                orig_order, Pdir, hel_dict = self.id_to_path[cross_tag]
        else:
            try:
                orig_order, Pdir, hel_dict = self.id_to_path_second[tag]
            except KeyError:
                cross_tag = self.get_crossing_tag(tag)
                if cross_tag:
                    orig_order, Pdir, hel_dict = self.id_to_path[cross_tag]
                elif self.options['allow_missing_finalstate']:
                    return 0.0
                else:
                    logger.critical('The following initial/final state %s can not be found in the new model/process. If you want to set the weights of such events to zero use "change allow_missing_finalstate False"', tag)
                    raise Exception


        base = os.path.basename(os.path.dirname(Pdir))
        if base == 'rw_me':
            moduletag = (base, 2+hypp_id)
        else:
            moduletag = (base, 2)
        
        module = self.f2pylib[moduletag]
        if self.keep_ordering:
            all_p = [event.get_momenta(orig_order, merged_map=self.revert_merged)]
        else:
            all_p = event.get_all_momenta(orig_order, merged_map=self.revert_merged)
            if len(all_p) >1:
                if self.helicity_reweighting:
                    logger.warning("due to ordering ambiguity, we flip off helicity per helicity reweighting.")
                self.helicity_reweighting = False

        # add helicity information
        event_pos2order, orderevent_2pos = event.get_mapping(orig_order, merged_map=self.revert_merged)
        hel_order = event.get_helicity(orig_order, merged_map=self.revert_merged)
        if self.helicity_reweighting and 9 not in hel_order:
            nhel = hel_dict[tuple(hel_order)]

        else:
            nhel = -1
        pdg = list(orig_order[0])+list(orig_order[1])
        relevant_merged = relevant_model.get('merged_particles') if relevant_model else self.merged_particles
        if relevant_merged and any(p in relevant_merged for p in pdg):
            pdg = event.get_pdg(all_p[0])

        #boosting the event
        all_p = self.method_boost_event(event, all_p, orig_order, hypp_id)
        
        if self.options['identical_particle_in_prod_and_decay'] == 'crash':
            if len(all_p) > 1:
                raise Exception("Ambiguous particle in production and decay. crash as requested by \'identical_particle_in_prod_and_decay\'")

        me_value = 0
        for p in all_p:
            pold = list(p)
                                   
            p = self.invert_momenta(p)
            try:
                pid = event.ievent
            except AttributeError:
                pid = -1
            if not self.use_eventid:
                pid = -1
            
            if not scale2: 
                if hasattr(event, 'scale'):
                    scale2 = event.scale**2
                else:
                    scale2 = 0

            with misc.chdir(Pdir):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):
                    #misc.sprint(pdg, pid, p, event.aqcd, scale2, nhel)
                    new_value = module.smatrixhel(pdg, pid, p, event.aqcd, scale2, nhel)
                    #misc.sprint(new_value)
                    if new_value == 0:
                        raise Exception("Invalid matrix element")
            # for loop we have also the stability status code
            if isinstance(new_value, tuple):
                new_value, code = new_value
                #if code points unstability -> returns 0
                hundred_value = (code % 1000) //100
                if hundred_value in [4]:
                    new_value = 0.
            if self.options["identical_particle_in_prod_and_decay"] == "average":
                me_value += new_value
            elif self.options["identical_particle_in_prod_and_decay"] == "max":
                if abs(new_value) > abs(me_value):
                    me_value = new_value
            else: 
                raise Exception("not valid option")

        if self.options["identical_particle_in_prod_and_decay"] == "average":
            return me_value / len(all_p)        
        else:
            return me_value
        

    def get_crossing_tag(self,tag):
        """find if using crossing symmetry allow to find the correct tag and return the assoicated tag"""

        # get list of possible crossing tag
        crossing_tag = [tuple([int(x) for x in sorted(list(t[0])+list(t[1]))]) for t in self.id_to_path.keys()]

        mytag = list(tag[0])+list(tag[1])
        if self.revert_merged:
            for i in range(len(mytag)):
                if mytag[i] in self.revert_merged:
                    mytag[i] = self.revert_merged[mytag[i]]
                if -mytag[i] in self.revert_merged:
                    mytag[i] = -self.revert_merged[-mytag[i]]
        mytag.sort()
        mytag=tuple(mytag)
        nb_found = crossing_tag.count(mytag)
        if nb_found == 0 :
            return None
        elif nb_found > 1:
            raise Exception('more than one cross-matrix element found')
        else:
            index = crossing_tag.index(mytag)
        return list(self.id_to_path.keys())[index]




    def method_boost_event(self, event, all_p, orig_order, hypp_id):
        # For 2>N pass in the center of mass frame
        #   - required for helicity by helicity re-weighitng
        #   - Speed-up loop computation 
        
        if (hypp_id == 0 and ('frame_id' in self.banner.run_card and self.banner.run_card['frame_id'] !=6)):
            import copy
            new_event = copy.deepcopy(event)
            pboost = FourMomenta()
            to_inc = bin(self.banner.run_card['frame_id'])[2:]
            to_inc.reverse()
            nb_ext = 0
            for p in new_event:
                if p.status in [-1,1]:
                    nb_ext += 1
                    if to_inc[nb_ext]:
                        pboost += p                    
            new_event.boost(pboost)
            if self.keep_ordering:
                new_all_p = [new_event.get_momenta(orig_order)]
            else:
                new_all_p = new_event.get_all_momenta(orig_order)
            if len(new_all_p) > 1:
                logger.critical("due to ordering ambiguity, the boost used might not be consistent. please ensure that this is not an issue")
                
            return new_all_p

        elif (hypp_id == 1 and self.boost_event):
            if self.boost_event is not True:
                new_event = copy.deepcopy(event)
                new_event.boost(self.boost_event)
                if self.keep_ordering:
                    new_all_p = [new_event.get_momenta(orig_order)]
                else:     
                    new_all_p = new_event.get_all_momenta(orig_order)

                return new_all_p
            return all_p #if we arrive here, we should return the input no ?

        elif (hasattr(event[1], 'status') and event[1].status == -1) or \
        (event[1].px == event[1].py == 0.):
            p = all_p[0]
            pboost = lhe_parser.FourMomentum(p[0]) + lhe_parser.FourMomentum(p[1])
            for p in all_p:
                for i,thisp in enumerate(p):
                    p[i] = lhe_parser.FourMomentum(thisp).zboost(pboost).get_tuple()
                assert p[0][1] == p[0][2] == 0 == p[1][2] == p[1][2] == 0 
            
            return all_p
        
        else:
            return all_p


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
            if self.flag_density_matrix:
                import madgraph.various.Density_functions as dens
                logger.info("Cross-section: %s +- %s pb" % (cross, error))
            else:
                if 'orig' not in self.all_cross_section:
                    logger.info('Original cross-section: %s +- %s pb' % (cross, error))
                else: 
                    logger.info('Original cross-section: %s +- %s pb (cross-section from sum of weights: %s)' % (cross, error, self.all_cross_section['orig'][0]))
                logger.info('Computed cross-section:')
                keys = list(self.all_cross_section.keys())
                keys.sort(key=lambda x: str(x))
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
                        commandline += self.get_LO_definition_from_NLO(proc, self.model, ewsudakov=self.inc_sudakov)
                    else:
                        commandline += self.get_LO_definition_from_NLO(proc,
                                                    self.model, real_only=True, ewsudakov=self.inc_sudakov)
                else:
                    commandline += self.get_LO_definition_from_NLO(proc, self.model, ewsudakov=self.inc_sudakov)
        # --no_crossing skips the generation of crossed subprocesses (e.g.
        # u~ g > h u~ when u g > h u is already there). That's fine when
        # flavor grouping is on, because the merged matrix element handles
        # all signs internally. Without flavor grouping, however, the
        # crossed subprocesses must be generated as separate entries --
        # otherwise antiparticle events have nothing to match against in
        # id_to_path. Only emit --no_crossing when both conditions hold.
        if not self.keep_ordering and self._reweight_use_flavor_grouping():
            commandline = commandline.replace('add process', 'add process --no_crossing')
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
            if commandline:
                logger.info("RETRY with %s", commandline)
                mgcmd.exec_cmd(commandline, precmd=True)
                has_nlo = False
        except Exception as error:
            misc.sprint(type(error))
            raise
        
        commandline = 'output %s %s --prefix=int --prefixf2py=%s' % (self.sa_class, pjoin(path_me,data['paths'][0]), self.nb_rw)
        self.path2prefix[pjoin(path_me,data['paths'][0])] = self.nb_rw
        self.nb_rw += 1
        commandline = 'output %s %s --prefix=int' % (self.sa_class, pjoin(path_me,data['paths'][0]))
        if self.inc_sudakov:
            # in this case, the sudakov output format has to be changed
            commandline = 'output ewsudakovsa %s --prefix=int' % pjoin(path_me,data['paths'][0])
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
                            raise self.InvalidCmd('2 different process have the same final states. This module can not handle such situation')
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
                print("compile OLP", data['paths'][1])
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
        commandline = re.sub(r'@\s*\d+', '', commandline)
        # deactivate golem since it creates troubles
        old_options = dict(mgcmd.options)
        if mgcmd.options['golem']:
            logger.info(" When doing NLO reweighting, MG5aMC cannot use the loop reduction algorithms Golem")
        mgcmd.options['golem'] = None            
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output standalone_rw %s --prefix=int -f --prefixf2py=%i ' % (pjoin(path_me, data['paths'][1]), self.nb_rw)
        self.path2prefix[pjoin(path_me,data['paths'][1])] = self.nb_rw
        self.nb_rw += 1
        mgcmd.exec_cmd(commandline, precmd=True) 
        
        #put back golem to original value
        mgcmd.options['golem'] = old_options['golem']
        # update make_opts

        if not mgcmd.options['lhapdf']:
            raise Exception("NLO reweighting requires LHAPDF to work correctly")

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
                            raise self.InvalidCmd('2 different process have the same final states. This module can not handle such situation')
                        else:
                            continue
                    # build the helicity dictionary
                    hel_nb = 0
                    hel_dict = {9:0} # unknown helicity -> use full ME
                    for helicities in me.get_helicity_matrix():
                        hel_nb +=1 #fortran starts at 1
                        hel_dict[tuple(helicities)] = hel_nb

                    data['id2path'][(tag,'V')] = [order, Pdir, hel_dict]

    def load_interface_model(self, second=False):
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
                        if re.search(r'^\s*add\s+process', line)]  
            #object_collector
            #self.id_to_path = {}
            #data['id2path'] = self.id_to_path
        else:
            for key in list(self.f2pylib.keys()):
                if 'rw_me_%s' % self.nb_library in key[0]:
                    del self.f2pylib[key]
                
            self.nb_library += 1
            data['paths'] = ['rw_me_%s' % self.nb_library, 'rw_mevirt_%s' % self.nb_library]


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
                                        if re.search(r'^\s*add\s+process', line)]
            #object_collector
            #self.id_to_path_second = {}   
            #data['id2path'] = self.id_to_path_second 

        #if not self.keep_ordering:
        #    for i,line in enumerate(data['processes']):
        #        data['processes'][i] = '%s --no_crossing' % line
            

        # 0. clean previous run ------------------------------------------------
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        data['path'] = path_me

        for i in range(2):
            pdir = pjoin(path_me,data['paths'][i])
            if os.path.exists(pdir):
                try:
                    shutil.rmtree(pdir)
                except Exception as error:
                    misc.sprint('fail to rm rwgt dir:', error) 
                    pass 

        # 1. prepare the interface----------------------------------------------
        mgcmd = self.mg5cmd
        complex_mass = False  
        ew_scheme = None 
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        has_ew = re.compile(r'''set\s+EWscheme\s*(\w*)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
                if has_ew.search(line, re.I):
                    ew_scheme = has_ew.search(line).group(1)
            elif line.startswith('define'):
                try:
                    mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except madgraph.InvalidCmd:
                    pass 
                        
        # 1. Load model---------------------------------------------------------  
        if  not data['model_name'] and not second:
            raise self.InvalidCmd('Only UFO model can be loaded in this module.')
        elif data['model_name']:
            self.load_model(data['model_name'], data['mg_names'], complex_mass, ew_scheme)
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
                try:
                    mgcmd.exec_cmd("define %s = %s" % (name, content))
                except madgraph.InvalidCmd:
                    pass
        return path_me, data, mgcmd

    @misc.mute_logger()
    def create_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""
    ############################## def load_interface            
        path_me, data, mgcmd = self.load_interface_model(second)
        if  second and 'tree_path' in self.dedicated_path:
            files.ln(self.dedicated_path['tree_path'], path_me,name=data['paths'][0])
            if 'virtual_path' in self.dedicated_path:
                has_nlo=True
            else:
                has_nlo=False
        else:
            has_nlo = self.create_standalone_tree_directory(data, second)

        if has_nlo and not self.rwgt_mode:
            self.rwgt_mode = ['NLO']

        # 5. create the virtual for NLO reweighting  ---------------------------
        if second and 'virtual_path' in self.dedicated_path:
            files.ln(self.dedicated_path['virtual_path'], path_me, name=data['paths'][1])
        elif has_nlo and 'NLO' in self.rwgt_mode:
            self.create_standalone_virt_directory(data, second)
            
            if self.multicore == 'create':
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
            commandline = commandline.replace('add process', 'generate',1)
            logger.info(commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            commandline = 'output standalone_rw %s --prefix=int -f --prefixf2py=%i' % (pjoin(path_me, data['paths'][1]), self.nb_rw)
            self.path2prefix[pjoin(path_me,data['paths'][1])] = self.nb_rw
            self.nb_rw+=1
            mgcmd.exec_cmd(commandline, precmd=True)    
            #put back golem to original value
            mgcmd.options['golem'] = old_options['golem']
            # update make_opts
            if not mgcmd.options['lhapdf']:
                raise Exception("NLO_tree reweighting requires LHAPDF to work correctly")
            
            # Download LHAPDF SET
            common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
                mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())
            
                
            
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
        
        if self.inc_sudakov:
            # The case of EW sudakov is a bit different
            # first, copy the run/param cards in the reweight process folder
            sarw_path = pjoin(path_me, 'rw_me')
            logger.info('Splitting the banner in %s' % os.path.join(sarw_path, 'Cards'))
            self.banner.split(sarw_path)

            logger.info('Compiling reweight Source dir')
            sourcedir = pjoin(sarw_path, 'Source') 
            # set the environmental variable ewsudsa in make_opts
            common_run_interface.CommonRunCmd.update_make_opts_full(pjoin(sourcedir, 'make_opts'), {'ewsudsa': 'True'})
            misc.compile(cwd=sourcedir)
            logger.info('Compiling reweight P* dirs')
            p_dirs = [d for d in \
                open(pjoin(sarw_path, 'SubProcesses', 'subproc.mg')).read().split('\n') if d]
            # determine the number of core to use for compilation
            try:
                import multiprocessing
                try:
                    nb_core = int(self.options['nb_core'])
                except (TypeError, KeyError):
                    nb_core = multiprocessing.cpu_count()
            except ImportError: 
                nb_core = 1

            compile_options = copy.copy(self.options)
            compile_options['nb_core'] = nb_core
            compile_cluster = cluster.MultiCore(**compile_options)
            logger.info('Compiling on %d cores...' % nb_core)

            update_status = lambda i, r, f: (i,r,f) 
            for p_dir in p_dirs:
                compile_cluster.submit(prog = misc.compile, 
                               argument = [['libsudpy'], pjoin(sarw_path, 'SubProcesses', p_dir) ])
            try:
                compile_cluster.wait(self.me_dir, update_status)
            except Exception as  error:
                logger.warning("Compilation of the Subprocesses failed")
                if __debug__:
                    raise
                compile_cluster.remove()
                self.do_quit('')
            logger.info('...done')

        else:
            rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
            for onedir in rwgt_dir_possibility:
                if not os.path.isdir(pjoin(path_me,onedir)):
                    continue
                pdir = pjoin(path_me, onedir, 'SubProcesses')
                self.compile_SubProcess_dir(pdir)


    def compile_SubProcess_dir(self, Sdir):
        """compile a full Subprocess directory"""

        if self.mother:
            nb_core = self.mother.options['nb_core'] if self.mother.options['run_mode'] !=0 else 1
        else:
            nb_core = 1
        os.environ['MENUM'] = '2'
        try: 
            misc.compile(['all_matrix2py.so'], cwd=Sdir, nb_core=nb_core)
        except Exception as e:
            misc.compile(['all_matrix2py.so'], cwd=Sdir, nb_core=1)

        if not (self.second_model or self.second_process or self.dedicated_path):
            os.environ['MENUM'] = '3'
            try:
                misc.compile(['all_matrix3py.so'], cwd=Sdir, nb_core=nb_core)
            except Exception as e:
                misc.compile(['all_matrix3py.so'], cwd=Sdir, nb_core=1)
                

    def load_module(self, metag=1):
        """load the various module and load the associate information"""
        
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir       

        self.id_to_path = {}
        self.id_to_path_second = {}
        rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
        fprefix = ''
        for onedir in rwgt_dir_possibility:
            if pjoin(path_me,onedir) in self.path2prefix:
                fprefix = self.path2prefix[pjoin(path_me,onedir)]
            if not os.path.exists(pjoin(path_me,onedir)):
                continue 
            if self.inc_sudakov:
                return
            pdir = pjoin(path_me, onedir, 'SubProcesses')
            for tag in [2*metag,2*metag+1]:
                with misc.TMP_variable(sys, 'path', [pjoin(path_me), pjoin(path_me,onedir, 'SubProcesses')]+sys.path): 
                    tmp = sys.path[0]
                    import ctypes
                    alllib = pjoin(sys.path[0], ('liball%s_%sme.so' % (onedir, tag)))
                    if os.path.exists(alllib):
                            #os.environ['LD_PRELOAD'] = pjoin(pdir, 'liballme%s' % ext) + os.pathsep + os.environ.get('LD_PRELOAD','')
                            #if ext == '.dylib':
                            #    mode=os.RTLD_LOCAL
                            #else:
                            mode=os.RTLD_GLOBAL | os.RTLD_DEEPBIND
                            try:
                                ctypes.CDLL(alllib, mode=mode)
                            except Exception as err:
                                logger.debug('ctypes trick fail for module')
                            break
                    mod_name = '%s.SubProcesses.all_matrix%spy' % (onedir, tag)
                    #mymod = __import__('%s.SubProcesses.allmatrix%spy' % (onedir, tag), globals(), locals(), [],-1)
                    if mod_name in list(sys.modules.keys()):
                        del sys.modules[mod_name]
                        tmp_mod_name = mod_name
                        while '.' in tmp_mod_name:
                            tmp_mod_name = tmp_mod_name.rsplit('.',1)[0]
                            del sys.modules[tmp_mod_name]
                        import importlib
                        mymod = importlib.import_module(mod_name,)
                        mymod = importlib.reload(mymod)
                        #mymod = __import__(mod_name, globals(), locals(), [])
                    else:
                        import importlib
                        mymod = importlib.import_module(mod_name,)
                        #mymod = __import__(mod_name, globals(), locals(), [])
                    
                if fprefix != '':
                    fprefix = 'f%i_' % fprefix
                    for attr in dir(mymod):
                        if attr.startswith(fprefix):
                            setattr(mymod, attr[len(fprefix):], getattr(mymod, attr)    )
                elif any(attr.startswith('f') and attr[1:].split('_')[0].isdigit() for attr in dir(mymod)):
                    fprefix = [attr for attr in dir(mymod) if attr.startswith('f') and attr[1:].split('_')[0].isdigit()][0].split('_')[0] + '_'
                    for attr in dir(mymod):
                        if attr.startswith(fprefix):
                            setattr(mymod, attr[len(fprefix):], getattr(mymod, attr))
                else:
                    logger.debug("Could not find the fortran prefix in module %s", mod_name)
                fprefix = ''
                # Param card not available -> no initialisation
                self.f2pylib[(onedir,tag)] = mymod
                if hasattr(mymod, 'set_madloop_path'):
                    mymod.set_madloop_path(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources'))
                if (self.second_model or self.second_process or self.dedicated_path):
                    break



            data = self.id_to_path
            if onedir not in ["rw_me",  "rw_mevirt"]:
                data = self.id_to_path_second

            # get all the information

            allids, all_pids = getattr(mymod, 'get_pdg_order')()
            all_pdgs = [[pdg for pdg in pdgs if pdg!=0] for pdgs in  allids]
            all_prefix = [bytes(j).decode(errors="ignore").strip().lower() for j in mymod.get_prefix()]
            prefix_set = set(all_prefix)
            hel_dict={}
            for prefix in prefix_set:
                if hasattr(mymod,'%s%sprocess_nhel' % (fprefix,prefix)):
                    #transer nhel information from fortran to wrapper
                    getattr(mymod, '%sget_nhel_entry' % prefix)()
                    #transer now to python dictionary
                    nhel = getattr(getattr(mymod, '%sprocess_nhel' % prefix), '%snhel' %prefix)
                    hel_dict[prefix] = {}
                    for i, onehel in enumerate(zip(*nhel)):
                        hel_dict[prefix][tuple(onehel)] = i+1
                elif hasattr(mymod, '%sset_madloop_path' % fprefix) or  hasattr(mymod, 'set_madloop_path') and \
                     os.path.exists(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper())):
                    hel_dict[prefix] = {}
                    for i,line in enumerate(open(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper()))):
                        onehel = [int(h) for h in line.split()]
                        hel_dict[prefix][tuple(onehel)] = i+1
                else:
                    misc.sprint(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper()))
                    misc.sprint(dir(mymod))
                    raise Exception
                    continue
            if not hel_dict:
                raise Exception("No helicity information found for reweighting ME in %s" % pdir)    
            for i,(pdg,pid) in enumerate(zip(all_pdgs,all_pids)):
                if self.is_decay:
                    incoming = [pdg[0]]
                    outgoing = pdg[1:]
                else:
                    incoming = pdg[0:2]
                    outgoing = pdg[2:]
                order = (list(incoming), list(outgoing))
                incoming.sort()
                if not self.keep_ordering:
                    outgoing.sort()
                tag = (tuple(incoming), tuple(outgoing))
                if 'virt' in onedir:
                    tag = (tag, 'V')
                prefix = all_prefix[i]
                if prefix in hel_dict:
                    hel = hel_dict[prefix]
                else:
                    hel = {}
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
                        misc.sprint(all_prefix[all_pdgs.index(pdg)])
                        misc.sprint(all_prefix[all_pdgs.index(oldpdg)])
                        misc.sprint(tag, onedir)
                        misc.sprint(data[tag][:-1])
                        misc.sprint(order, pdir,)
                        raise Exception( "two different matrix-element have the same initial/final state. Leading to an ambiguity. If your events are ALWAYS written in the correct-order (look at the numbering in the Feynman Diagram). Then you can add inside your reweight_card the line 'change keep_ordering True'." )
                data[tag] = order, pdir, hel
             
             
    def load_model(self, name, use_mg_default, complex_mass=False, ew_scheme=None):
        """load the model"""

        loop = False

        logger.info('detected model: %s. Loading...' % name)
        model_path = name

        # Decide whether to ask for flavor grouping. We prefer it on
        # (matches the rest of MG5) but the loop machinery does not
        # support it yet, so for any reweight that touches a loop
        # process we force it off for every model loaded -- otherwise
        # the original (LO) model would end up with merged PDGs (e.g.
        # 81 for jets) in id_to_path while the loop model's
        # id_to_path_second would have raw PDGs, and the two would not
        # agree.
        apply_flavor_grouping = self._reweight_use_flavor_grouping()
        import_options = {'apply_flavor_grouping': apply_flavor_grouping}
        # Import model
        base_model = import_ufo.import_model(name, decay=False,
                                               complex_mass_scheme=complex_mass,
                                               options=import_options)

        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()

        # Keep a handle on the first (original) model loaded. self.model
        # gets overwritten when a second model is loaded for
        # second_model / second_process / dedicated_path; we still want
        # to consult the original model's merged_particles when looking
        # up id_to_path (built from the original model).
        if getattr(self, 'original_model', None) is None:
            self.original_model = base_model
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        # Propagate the flavor-grouping decision to the main MG5
        # interface so the subsequent `import model` command (issued
        # from create_standalone_directory) uses the same convention.
        try:
            self.mg5cmd.options['apply_flavor_grouping'] = apply_flavor_grouping
        except Exception:
            pass
        if ew_scheme:
            self.model.change_electroweak_mode(ew_scheme)
        self.mg5cmd.process_model()

    def _reweight_use_flavor_grouping(self):
        """Return whether flavor grouping should be applied when loading
        models for this reweight. We default to the user's MG5 setting
        (True unless changed) but force it off as soon as we see any
        signal that the reweight involves a loop / perturbative
        computation -- since the loop output cannot use merged particles
        and a mixed convention between id_to_path (original model) and
        id_to_path_second (second model) breaks the tag lookup."""
        try:
            default = bool(self.mg5cmd.options.get('apply_flavor_grouping', True))
        except Exception:
            default = True
        if not default:
            return False

        def _looks_like_loop(text):
            return text and ('[' in text)

        if getattr(self, 'has_nlo', False):
            return False
        if self.second_process:
            if any(_looks_like_loop(p) for p in self.second_process):
                return False
        proc_card = self.banner.get('proc_card') if self.banner else None
        if proc_card:
            for line in proc_card:
                stripped = line.strip()
                if (stripped.startswith('generate') or
                        re.search(r'^\s*add\s+process', stripped)):
                    if _looks_like_loop(stripped):
                        return False
        return default
        

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
        to_save['merged_map'] = self._get_revert_merged_for(self.model)
        to_save['merged_particles'] = self.model.get('merged_particles') if self.model else None
        if self.second_model:
            to_save['second_model'] =True
        else:
            to_save['second_model'] = None
        to_save['rwgt_dir'] = self.rwgt_dir
        to_save['has_nlo'] = self.has_nlo
        to_save['rwgt_mode'] = self.rwgt_mode
        to_save['rwgt_name'] = self.options['rwgt_name']
        to_save['allow_missing_finalstate'] = self.options['allow_missing_finalstate']
        to_save['identical_particle_in_prod_and_decay'] = self.options['identical_particle_in_prod_and_decay']
        to_save['nb_library'] = self.nb_library

        name = pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl')
        save_load_object.save_to_file(name, to_save)


    def load_from_pickle(self, keep_name=False):
        import madgraph.iolibs.save_load_object as save_load_object
        
        obj = save_load_object.load_from_file( pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl'))
        
        self.has_standalone_dir = True
        previous_options = dict(self.options)
        self.options = dict(previous_options)
        self.options.update({'curr_dir': os.path.realpath(os.getcwd()),
                             'rwgt_name': None})
        
        if keep_name:
            self.options['rwgt_name'] = (
                previous_options['rwgt_name']
                if previous_options.get('rwgt_name') is not None
                else obj['rwgt_name']
            )


        self.options['allow_missing_finalstate'] = obj['allow_missing_finalstate']
        self.options['identical_particle_in_prod_and_decay'] = obj['identical_particle_in_prod_and_decay']
        old_rwgt = obj['rwgt_dir']
        self.revert_merged = obj['merged_map']
        self.merged_particles = obj['merged_particles']
        if not self.revert_merged:
            self.revert_merged = self._get_revert_merged_for(self.model) 
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
        self.nb_library = obj['nb_library']
        if not self.rwgt_mode:
            self.rwgt_mode = obj['rwgt_mode']
            logger.info("mode set to %s" % self.rwgt_mode)
        if False:#self.has_nlo and 'NLO' in self.rwgt_mode:
            #use python version
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
                    
        
        
        





        
class DensityInterface(ReweightInterface):
    """Basic interface for computing density matrix"""

    def __init__(self, *args, **opts):
        """init the class"""

        logger.info('Using density mode for reweighting')
        
        self.flag_particle_in_density_matrix = False

        self.helicity_direction = [[0], '', []] #pid of the particle chosen as reference for the helicity frame
        self.particle_in_density_matrix = None #pid of the particles selected for the study
        self.momenta_boost = [[0], '', []] #pid of the particles in whose center of mass frame the system will be boosted
        self.allowed_helicities = [0] #basis of helicities
        self.axis_referential = [0]
        self.symmetrise_initial_state = False
        self.spins = None 
        self.number_changing_helicities = None
        self.number_combinations = None
        self.new_param_card = False #Needed to not call ask_edit_card_static
        self.average_rho = 0
        self.total_wgt = 0
        
        ReweightInterface.__init__(self, *args, **opts)
        self.flag_density_matrix = True
        self.has_run = False

        #This block imports the model, because I need it before do_launch() starts
        mgcmd = self.mg5cmd
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
        data = {}
        data['model_name'] = self.banner.get('proc_card', 'model')

        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            data['mg_names'] = False
        else:
            data['mg_names'] = True
        super().load_model(data['model_name'], data['mg_names'], complex_mass)

    def _reweight_use_flavor_grouping(self):
        """Density-matrix computation is per-flavor: merging quark flavors
        into a single generic tag would sum subprocesses and break the
        density matrix interpretation. Disable flavor grouping in this
        mode regardless of the user's global setting."""
        return False

    def do_change(self, line):
        """Method called to read the reweight card, redirects to the correct do_change_ method"""
        keyword = line.split()[0]

        if hasattr(self, 'do_change_%s' % keyword):
            return getattr(self, 'do_change_%s' % keyword)(line.split()[1:])
        
        return super().do_change(line)
        

    def find_arrays(self, input_text):
        pattern = re.compile(r"\[[^\]]*\]", re.IGNORECASE)
        return pattern.findall(input_text)

    def find_observable(self, input_text): #we accepect input as "observable" or "observable_name"
        # pattern = re.compile(r"lambda p: p\.[A-Za-z0-9]+", re.IGNORECASE)
        # output = pattern.findall(input_text)
        # if output == []:
        pattern = re.compile(r"[A-Za-z]+", re.IGNORECASE)
        output = pattern.findall(input_text)
        if len(output) > 1:
            pattern = re.compile(r"[A-Za-z]+_[A-Za-z]+", re.IGNORECASE)
            output = pattern.findall(input_text)
        return pattern.findall(input_text)

    def do_change_helicity_direction(self, line):
        """Change the reference particle for the helicity frame, returns a list of pdg-codes.
        The structure accepted is change helicitty_direction [list of pdg-codes] observable [order of observable values]
        """
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)

        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (helicity referential direction option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes):
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the helicity referential direction. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.

        self.helicity_direction = (pdg_codes, lambda_function, order_particles)


    def do_change_boost_choice(self, line):
        """change the momenta reference for the boost, returns a list of pdg-codes"""
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)
        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (boost option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes): #The code should work even if the two length are different but it is more clear like that for the user too
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the boost. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.
        
        self.momenta_boost = (pdg_codes, lambda_function, order_particles)



    def do_change_order_helicities(self, line):
        """Change the order of the basis of helicities. It accepts inputs for density matrices full and partial"""

        if len(line) == 1 and line[0] == '[0]': # if order_helicitites [0], we take the default value
            return
        
        for i in range(len(line)):
            aux = line[i].strip("[],()")
            if aux != 'None':
                line[i] = int(aux)
            else:
                return #if "change order_helicities None" is the option, we do not change the default value 

        #Let the user enter the allowed_helicities in the complex form ie. [+1, +1, +1, -1, -1, +1, -1, -1] for 2 qubits for instance
        if len(line) == self.number_changing_helicities * self.number_combinations:
            self.allowed_helicities = line
        else: #this part deals with input of the form [basis for particle1] [basis for particle2] 
            cutted_line = []
            counter = 0
            for i in range(len(self.spins)):
                cutted_line.append(line[counter: counter + self.spins[i]])
                counter += self.spins[i]
            
            allowed_hel = []
            for i in range(len(cutted_line[0])):
                for j in range(len(cutted_line[1])):
                    allowed_hel.append(cutted_line[0][i])
                    allowed_hel.append(cutted_line[1][j])
                self.allowed_helicities = allowed_hel
    
    def do_change_symmetrise_initial_state(self, line):
        """
        Chooses whether the initial state should be symmetrised according to 2307.09675. For each event the production matrix calculated is
        R = R(theta) + R(theta + pi)
        """
        for i in range(len(line)): 
            line[i] = line[i].strip("[],()") 
        if line[0] == 'True':
            self.symmetrise_initial_state = True
        elif line[0] == 'False':
            self.symmetrise_initial_state = False
        else:
            logger.warning('Option symmetrise_initial_state not understood, set it to False. Please use the syntax: change symmetrise_initial_state True if you want to enable it.')
            self.symmetrise_initial_state = False
        

    def do_change_axis_referential(self, line):
        """
        Choses a particle in the initial state that is used as referential to define the production angle theta
        It can be useful for non-symetric initial states like u u~.
        It does accept only one pdg-code
        """
        for i in range(len(line)):
            aux = line[i].strip("[],()")
            if aux != 'None':
                line[i] = int(aux)
            else:
                return #if "change axis_referential None" is the option, we do not change the default value 
        self.axis_referential = line


    def do_change_particle_in_density_matrix(self, line):
        """change the particle in the density matrix, calculates the number of particles changes,
           their spins and the number of combinations"""
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)
        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (particle in density matrix option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes):
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the particles in density matrix. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.

        self.particle_in_density_matrix = (pdg_codes, lambda_function, order_particles) 

        self.number_changing_helicities = len(pdg_codes)

        particles = self.model['particles']

        self.spins = [] #list of spins degrees of freedom for each particle studied
        for particle_id in pdg_codes: #list on the pdg-code of the particles that we study
            for n_particles_model in range(len(particles)):
                if particles[n_particles_model]['pdg_code'] == particle_id or particles[n_particles_model]['pdg_code'] == -particle_id:
                    if particles[n_particles_model]['spin'] == 3 and particles[n_particles_model]['mass'] == 'ZERO': #if the particle is a massless boson, we set the spin d.o.f. to 2. Can it be problemtaic for some gauges ?
                        self.spins.append(particles[n_particles_model]['spin'] - 1)
                    else: #if the boson has a mass, we keep the 3 spin d.o.f.
                        self.spins.append(particles[n_particles_model]['spin'])

        #Calculation of the number of helicity combinations
        n_comb = 1
        for i in range(len(self.spins)):
            n_comb *= self.spins[i]
        self.number_combinations = n_comb

        #if the user didn't use the option or if it has not been read yet, fill it automatically here
        if self.allowed_helicities == None or self.allowed_helicities == [0]:
            if self.number_combinations == 2:
                self.allowed_helicities = [+1, -1]
            elif self.number_combinations == 3:
                self.allowed_helicities = [+1, 0, -1]
            elif self.number_combinations == 4:
                self.allowed_helicities = [+1, +1, +1, -1, -1, +1, -1, -1]
            elif self.number_combinations == 6 and self.spins[0] == 2:
                self.allowed_helicities = [+1, +1, +1, 0, +1, -1, -1, +1, -1, 0, -1, -1]
            elif self.number_combinations == 6 and self.spins[0] == 3:
                self.allowed_helicities = [+1, +1, +1, -1, 0, +1, 0, -1, -1, +1, -1, -1]
            elif self.number_combinations == 9:
                self.allowed_helicities = [+1, +1, +1, 0, +1, -1, 0, +1, 0, 0, 0, -1, -1, +1, -1, 0, -1, -1]
            else:
                logger.error("Tried to use density mode selecting more than 2 particles or selecting a spin 0 or spin > 1 particle")
        
        self.flag_particle_in_density_matrix = True


    def do_quit(self, line):
        """exit the reweighting module"""
        if self.has_run:
            return super().do_quit(line)
        
        if self.particle_in_density_matrix == None:
            logger.error("You have not chosen which particle to put in the density matrix, the density matrix computation can not be done. The command to specify the particles to take is 'change particle_in_density_matrix'.")

        logger.info("helicity_direction = \t" + str(self.helicity_direction))
        logger.info("particle_in_density_matrix = \t" + str(self.particle_in_density_matrix))
        logger.info("momenta_boost = \t" + str(self.momenta_boost))
        logger.info("allowed_helicities = \t" + str(self.allowed_helicities))
        logger.info("spins = \t" + str(self.spins))
        logger.info("number_changing_helicities = \t" + str(self.number_changing_helicities))
        logger.info("number_combinations = \t" + str(self.number_combinations))
        logger.info("axis_referential = \t" + str(self.axis_referential))
        logger.info("symmetrise_initial_state = \t" + str(self.symmetrise_initial_state))

        if self.flag_particle_in_density_matrix == False:
            logger.error("Error: the reweight_card contains no option for the density mode")

        self.has_run = True
        self.run_cmd('launch --keep_card') #calls the function do_launch()



    def launch_actual_reweighting(self, param_card_iterator, 
                                              tag_name,
                                              type_rwgt,
                                              path_me):
        """
            This method overwrites the one in the parent class ReweightInterface when we want to do density matrix computation.
            It launches the computation of the density matrix for each event and computes the average density matrix.
        """
       
        import madgraph.various.Density_functions as dens
         
        start = time.time()
        # initialize the collector for the various re-weighting
        cross, ratio, ratio_square,error = {},{},{}, {}
        for name in type_rwgt + ['orig']:
            cross[name], error[name] = 0.,0.
            ratio[name],ratio_square[name] = 0., 0.# to compute the variance and associate error
        
        self.banner['MGDensity'] = 'helicity_direction = ' + str(self.helicity_direction) + '\n' + \
                                    'particle_in_density_matrix = ' + str(self.particle_in_density_matrix) + '\n' + \
                                    'momenta_boost = ' + str(self.momenta_boost) + '\n' + \
                                    'allowed_helicities = ' + str(self.allowed_helicities) + '\n' + \
                                    'number_changing_helicities = ' + str(self.number_changing_helicities) + '\n' + \
                                    'number_combinations = ' + str(self.number_combinations) + '\n' + \
                                    'axis_referential = ' + str(self.axis_referential) + '\n' + \
                                    'symmetrise_initial_state = ' + str(self.symmetrise_initial_state)
        self.banner.pop('initrwgt') #we remove the reweight header because it does not correspond to the operations done
        output = open( self.lhe_input.path +'rw', 'w')
        #write the banner to the output file
        self.banner.write(output, close_tag=False)
            
        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

        self.lhe_input.seek(0)
        count_errors = 0
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
            if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')
            weight = self.calculate_weight(event)
            rho_temp = dens.DensityMatrixObservables(weight['orig'])
            event.density = rho_temp.get_rho_normalised().tolist()

            if not isinstance(weight, dict):
                weight = {'':weight}
                avg_rho_instance = dens.DensityMatrixObservables(weight[''])
                self.average_rho += avg_rho_instance.get_rho_normalised() * event.wgt
                self.total_wgt += event.wgt
            else:
                avg_rho_instance = dens.DensityMatrixObservables(weight['orig'])
                self.average_rho += avg_rho_instance.get_rho_normalised() * event.wgt # weighted sum of the density matrices for the total density matrix
                self.total_wgt += event.wgt


            output.write(str(event))
                
        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))     
        
        # Compute the average density matrix on write it on a .txt file
        rho_avg = [0 for i in range(len(self.average_rho))]
        for i in range(len(rho_avg)):
            rho_avg[i] = self.average_rho[i] / self.total_wgt
        rho_avg_instance = dens.DensityMatrixObservables(rho_avg)
        rho_avg_square = rho_avg_instance.square_matrix()

        logger.info("Average density matrix:")
        for i in range(len(rho_avg_square)):
            print("\t",list(rho_avg_square[i]))
        file_density = open(pjoin(os.path.dirname(self.event_path), f"Average_density_matrix_{os.path.basename(self.lhe_input.name)[:-4]}.txt"), 'w')
        file_density.write(f'Average density matrix of LHE file {os.path.basename(self.lhe_input.name)[:-4]}:\n')
        # Cast each entry to a plain Python ``complex`` so that the file is
        # written in the legacy ``(re+imj)`` repr regardless of the underlying
        # numpy dtype (newer numpy prints np.complex64 values with a
        # ``np.complex64(...)`` wrapper which the consumer parser cannot read).
        for i in range(len(rho_avg_square)):
                row = [complex(v) for v in rho_avg_square[i]]
                file_density.write('\t' + str(row) + '\n')
        file_density.close()


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
        else:
            raise ValueError("Only the 'default' output_type is available for the density mode.")
        

        self.terminate_fortran_executables(new_card_only=True)

        #store result
        for name in cross:
            if name == 'orig':
                self.all_cross_section[name] = (cross[name], error[name])
            else:
                self.all_cross_section[(tag_name,name)] = (cross[name], error[name])


    def calculate_matrix_element(self, event, hypp_id, scale2=0):
        """ This method overwrites the method of the same name in the class ReweightInterface.
            It computes the production matrix R given the user's inputs.
            Output: new_value (the production matrix R for a single event)
        """
        import madgraph.various.Density_functions as dens

        relevant_model = getattr(self, 'original_model', None) or self.model
        if relevant_model:
            self.revert_merged = self._get_revert_merged_for(relevant_model)

        tag, order = event.get_tag_and_order(self.revert_merged)
        if self.keep_ordering:
            old_tag = tuple(tag)
            tag = (tag[0], tuple(order[1]))

        if tag in self.id_to_path:
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
        else:
            cross_tag = self.get_crossing_tag(tag)
            try:
                orig_order, Pdir, hel_dict = self.id_to_path[cross_tag]
            except KeyError:
                misc.sprint(tag)
                misc.sprint(self.id_to_path)
                raise KeyError('Try to fix it')

        base = os.path.basename(os.path.dirname(Pdir))

        if base == 'rw_me':
            moduletag = (base, 2+hypp_id)
        else:
            moduletag = (base, 2)

        module = self.f2pylib[moduletag]

        if self.keep_ordering:
            all_p = [event.get_momenta(orig_order, merged_map=self.revert_merged)]
        else:
            all_p = event.get_all_momenta(orig_order, merged_map=self.revert_merged)

            if len(all_p) >1:
                if self.helicity_reweighting:
                    logger.warning("due to ordering ambiguity, we flip off helicity per helicity reweighting.")
                self.helicity_reweighting = False

        # add helicity information
        hel_order = event.get_helicity(orig_order, merged_map=self.revert_merged)
        if self.helicity_reweighting and 9 not in hel_order:
            nhel = hel_dict[tuple(hel_order)]
        else:
            nhel = -1

        pdg = list(orig_order[0])+list(orig_order[1])
        relevant_merged = relevant_model.get('merged_particles') if relevant_model else self.merged_particles
        if relevant_merged and any(p in relevant_merged for p in pdg):
            pdg = event.get_pdg(all_p[0])

        #list_properties is the list of properties of the class FourMomentum that we can use to rank particles
        list_properties = [p for p in dir(lhe_parser.FourMomentum) if isinstance(getattr(lhe_parser.FourMomentum,p),property)]
        
        
        boost_corrected = self.chose_particle_user_input(event, pdg, list_properties, orig_order, self.momenta_boost, 'momenta_boost', fortran_format = False)
        all_p = self.method_boost_event(event, all_p, orig_order, hypp_id, boost_corrected)
        
        refChoice_corrected = self.chose_particle_user_input(event, pdg, list_properties, orig_order, self.helicity_direction, 'helicity_direction', fortran_format = True)
        phi, theta = self.calculate_angles_rotation(refChoice_corrected, all_p, module)
        
        for i in range(len(all_p)):
            #This block allows to choose which initial state particle is chosen as reference to define theta.
            #If its pz is > 0 the default definition is correct, if it is < 0, then we need to add pi
            if 0 not in self.axis_referential:
                for k in range(len(self.axis_referential)):
                    if self.axis_referential[k] in orig_order[0]: #check whether the pdg is in the initial state
                        for j in range(len(orig_order[0])):
                            if self.axis_referential[k] == orig_order[0][j]:
                                pz_axis_referential = all_p[i][j][3]
                                break #we quit the loop once we found which particle in the initial state is in axis_referential
                        if pz_axis_referential < 0:
                            theta[i] += math.pi

            if self.symmetrise_initial_state: # if we want to calculate R(theta) + R(theta + pi)
                import copy
                theta_bis = [elem + math.pi for elem in theta]
                all_p_bis = copy.deepcopy(all_p)
                all_p_bis = self.rotation_density(module, all_p_bis, phi, theta_bis)

        all_p = self.rotation_density(module, all_p, phi, theta)

        if self.options['identical_particle_in_prod_and_decay'] == 'crash':
            if len(all_p) > 1:
                raise Exception("Ambiguous particle in production and decay. crash as requested by \'identical_particle_in_prod_and_decay\'")


        pos_corrected = self.chose_particle_user_input(event, pdg, list_properties, orig_order, self.particle_in_density_matrix, 'particle_in_density_matrix', fortran_format = True)
        
        status = []
        for particle in event:
            status.append(int(particle.status))

        PDGs, _ = module.get_pdg_order()
        PREFIX = module.get_prefix()
        prefix_cor = []
        All_PDGs = []
        prefix_unique = []
            
        #Bloc to determine which sets of pdg-codes corresponds to which prefix
        for i in range(len(PREFIX)):
            prefix_cor.append(PREFIX[i].decode('UTF-8').strip().lower())
            if prefix_cor[i] not in prefix_unique:
                prefix_unique.append(prefix_cor[i])
        for i in range(len(PDGs)):
            All_PDGs.append(self.permutations_PGD(PDGs[i], status))

        #We take the card in the general folder, not in the reweight folder
        Card_dir = os.path.join(self.me_dir, "Cards", "param_card.dat")

        # Initialisation of the Fortran scripts with param_card.dat
        Initialise_allmatrix = getattr(module, 'initialise')
        Initialise_allmatrix(Card_dir)
        #for i in range(len(prefix_unique)):
        #    InitialiseMatrix = getattr(module, prefix_unique[i] + 'initialisemodel')
        #    InitialiseMatrix(Card_dir)   

        #The prefix is defined for a given event
        for k in range(len(All_PDGs)):
                if pdg in All_PDGs[k]:
                    prefix = prefix_cor[k]

        me_value = 0
        get_density = lambda *args: module.py_get_density(pdg, *args)
        for i in range(len(all_p)):
            pinv = self.invert_momenta(all_p[i])
            production_matrix = get_density(-1, pinv, pos_corrected, #self.number_changing_helicities,
                                            self.allowed_helicities, self.number_combinations,
                                            event.aqcd)    
            if self.symmetrise_initial_state:
                pinv_bis = self.invert_momenta(all_p_bis[i])
                production_matrix_bis = get_density(-1, pinv_bis, pos_corrected, self.allowed_helicities, self.number_combinations,
                                                    event.aqcd) #event.aqcd can be also fixed.

            if self.symmetrise_initial_state:
                rho_instance = dens.DensityMatrixObservables(production_matrix + production_matrix_bis, self.number_combinations * (self.number_combinations + 1) / 2)
                new_value = rho_instance.density_matrix
            else:
                rho_instance = dens.DensityMatrixObservables(production_matrix, self.number_combinations * (self.number_combinations + 1) / 2)
                new_value = rho_instance.density_matrix

        return new_value



    def calculate_weight(self, event, sud_mod=None):
        """ This method overwrites the method of the same name in the class ReweightInterface.
            For this mode, it does not do a lot.
            Output: {'orig': w_orig} (dictionnary with the production matrix as value)
        """
        w_orig = self.calculate_matrix_element(event, 0)
        return {'orig': w_orig}


    def method_boost_event(self, event, all_p, orig_order, hypp_id, boost_corrected):
        """ This method overwrites the method of the same name in the class ReweightInterface.
            Output: new_all_p (all the boosted momenta of a given event)
        """

        if 0 in self.momenta_boost[0]: #if we don't want to boost the system
            return all_p
        
        import copy
        new_event = copy.deepcopy(event)
        nb_ext = 0
        pboost = lhe_parser.FourMomentum()
        for p in new_event: 
            for j in range(len(boost_corrected)):
                if nb_ext == boost_corrected[j]:
                    pboost += p
            nb_ext += 1


        if abs(pboost.px/pboost.E) < 1e-10 and abs(pboost.py/pboost.E) < 1e-10 and abs(pboost.pz/pboost.E) < 1e-10:
            #if we try to boost with with a 4-momentum like [M, 0, 0, 0], we return the momenta without any boost
            return all_p
                
        if abs(pboost.px/pboost.E) < 1e-10:
            pboost.px = 0.
        if abs(pboost.py/pboost.E) < 1e-10:
            pboost.py = 0.
        if abs(pboost.pz/pboost.E) < 1e-10:
            pboost.pz = 0.

        new_event.boost(pboost)
        if self.keep_ordering:
            new_all_p = [new_event.get_momenta(orig_order, merged_map=self.revert_merged)]
        else:
            new_all_p = new_event.get_all_momenta(orig_order, merged_map=self.revert_merged)
        if len(new_all_p) > 1:
            logger.critical("due to ordering ambiguity, the boost used might not be consistent. please ensure that this is not an issue")

        return new_all_p



    def chose_particle_user_input(self, event, pdg, list_properties, orig_order, user_input, name_input, fortran_format = False):
        """
        This function transforms the user_input for a given name_input into the position of particles in the original order.
        The position of the particles can then be used to boost, rotate the event, etc.
        fortran_format = True, means that we use the Fortran format for indices, so lists begin at 1, else we use Python format.
        Output: position_particles
        """
        if 0 in user_input[0]: # if the user does not want to user this input
            return [-1]
        
        if user_input[1] == '':
            position_particles = self.find_position_particles_default_order(orig_order, user_input, name_input, fortran_format) #if the user does not give an observable to rank the particles
            return position_particles

        else:
            found_property = False
            for prop in list_properties: #finding the observable given by the user
                if prop == user_input[1]:
                    found_property = True
                    observable_values = []
                    original_order = [i for i in range(len(event))]
                    for i, p in enumerate(event):
                        if pdg[i] in user_input[0]:
                            correct_p_rot = lhe_parser.FourMomentum(p)
                            observable_values.append(getattr(correct_p_rot, prop))
                        else:
                            observable_values.append(float('NaN'))

                    # if several particles of same pdg have the same value of the observable, we can not rank them so we use the default order.
                    # we do not crash the code because it can happen randomly for an event, even if the user_input is correct
                    if len(set(observable_values)) != len(observable_values) and len(set(user_input[0])) != len(user_input[0]):
                        logger.warning(f"Some particles in {name_input} have the same value for the observable given. For this event the order of the observable is not taken into account.")                        
                        position_particles = self.find_position_particles_default_order(orig_order, user_input, name_input, fortran_format)
                        return position_particles
                    
                    observable_values_sorted, new_order = zip(*sorted(zip(observable_values, original_order), reverse=True)) #ranking the particles via the observable's value

                    if len(user_input[2]) > 0: # if the user gives a ranking to use for the observable, use it
                        position_particles = self.find_position_particles_with_observable(pdg, observable_values_sorted, new_order, original_order, user_input, name_input, fortran_format)
                        return position_particles

                    else: #else they are ranked in decreasing order
                        position_particles = self.find_position_particles_new_order(pdg, user_input, new_order, fortran_format)
                        return position_particles

            if not found_property:
                raise ValueError(f'The observable {user_input[1]} is not recognised. Observables are defined in the class FourMomentum of lhe_parser.')


    def find_position_particles_with_observable(self, pdg, observable_values_sorted, new_order, original_order, user_input, name_input, fortran_format):
        """
        This function transforms the user_input for a given name_input into the position of particles in the original order specifically if the user gives an observable.
        The position of the particles can then be used to boost, rotate the event, etc.
        fortran_format = True, means that we use the Fortran format for indices, so lists begin at 1, else we use Python format.
        Output: position_particles
        """
        if len(user_input[0]) != len(user_input[2]):
            raise ValueError(f'The number of particle in {name_input}[0] and the number of ranks in {name_input}[2] do not match.')

        pdg_new = [0] * len(pdg)
        new_order_corrected = [0] * len(pdg)
        for i in range(len(pdg)):
            if observable_values_sorted[i] == observable_values_sorted[i]: #if the value of observable is not a NaN (it is in the density matrix)
                pdg_new[i] = pdg[new_order[i]]
                new_order_corrected[i] = new_order[i]
            else: #if it is a NaN we keep them in the original order
                pdg_new[i] = pdg[i]
                new_order_corrected[i] = original_order[i]

        # dic_rank_particles keys are the particles chosen by the user and the values are their wanted rank in the observable's order
        dic_rank_particles = {}
        for i in range(len(user_input[0])):
            if user_input[0][i] not in dic_rank_particles.keys():
                dic_rank_particles[user_input[0][i]] = []

        for i in range(len(user_input[0])):
            if user_input[0][i] in dic_rank_particles.keys():
                dic_rank_particles[user_input[0][i]].append(user_input[2][i])

        # dic_values_observable keys are the particles chosen by the user and is filled with their value in the chosen observable
        # dic_postion_new_order keys are the particles chosen by the user and is filled with their position in new_order
        dic_values_observable, dic_postion_new_order = {}, {}
        for key in user_input[0]:
            dic_values_observable[key], dic_postion_new_order[key] = [], []

        for key in dic_rank_particles.keys():
            for j in range(len(pdg_new)):
                if pdg_new[j] == int(key):
                    dic_values_observable[key].append(observable_values_sorted[j])
            for k in range(len(dic_rank_particles[key])):
                for l in range(len(observable_values_sorted)):
                    try:
                        #this if statement represents which index of dic_values_observable[key] we want to keep
                        if observable_values_sorted[l] == dic_values_observable[key][dic_rank_particles[key][k]] and pdg[new_order[l]] == key:
                            dic_postion_new_order[key].append(l)
                    except:
                        raise ValueError(f'There are not enough identical particles for the rank you chose in {name_input}. Please change your input')

        position_particles = []
        for key in dic_postion_new_order.keys(): # here we convert the position in new_order to the position in the original order
            for j in range(len(dic_postion_new_order[key])):
                if fortran_format:
                    position_particles.append(new_order[dic_postion_new_order[key][j]] + 1) #python format begins integers at 0 so we do not need to add +1
                else:
                    position_particles.append(new_order[dic_postion_new_order[key][j]]) #python format begins integers at 0 so we do not need to add +1
        return position_particles

    def find_position_particles_default_order(self, orig_order, user_input, name_input, fortran_format):
        """
        This function transforms the user_input for a given name_input into the position of particles in the original order specifically if the user does not give any additional information.
        The position of the particles can then be used to boost, rotate the event, etc.
        fortran_format = True, means that we use the Fortran format for indices, so lists begin at 1, else we use Python format.
        Output: position_particles
        """
        pdg_to_chose = user_input[0]
        position_particles = []
        orig_order_concatenated = orig_order[0] + orig_order[1]
        particle_in_process_already_chosen = [False for i in range(len(orig_order_concatenated))]
        particle_in_user_input_already_found = [False for i in range(len(pdg_to_chose))]
        
        if 0 in pdg_to_chose:
            return [-1]
        else:
            for i in range(len(orig_order_concatenated)):
                if len(position_particles) != len(user_input[0]):
                    for j in range(len(pdg_to_chose)):
                        if pdg_to_chose[j] == orig_order_concatenated[i] and not particle_in_process_already_chosen[i] and not particle_in_user_input_already_found[j]: # if the particles is still available and that its pdg code corresponds to the user input
                            if fortran_format:
                                position_particles.append(i + 1) #if fortran_format = True, we add +1 because Fortran indices begin at 1 instead of 0 as in Python
                            else:
                                position_particles.append(i) # position of the particle in the original order
                            particle_in_process_already_chosen[i] = True
                            particle_in_user_input_already_found[j] = True

                            break

            if len(position_particles) != len(user_input[0]):
                logger.error(f'The pdg inputs for {name_input} are not correct. At least one pdg is not present in the process.')

            return position_particles


    def find_position_particles_new_order(self, pdg, user_input, new_order, fortran_format):
        """
        This function transforms the user_input for a given name_input into the position of particles in the original order specifically if the user gives 
        an observable but does not specify the rank in which to order them. In this case, they are ranked in decreasing order.
        The position of the particles can then be used to boost, rotate the event, etc.
        fortran_format = True, means that we use the Fortran format for indices, so lists begin at 1, else we use Python format.
        Output: position_particles
        """
        if -1 not in user_input[0]:
            position_particles = [0] * len(user_input[0])
            is_particle_taken = [0] * len(pdg)
            compteur = 0
            for i in range(len(position_particles)):
                for j in range(len(pdg)):
                    if pdg[new_order[j]] == user_input[0][i] and is_particle_taken[j] == 0:
                        if fortran_format:
                            position_particles[compteur] = new_order[j] + 1 #if fortran_format = True, we add +1 because Fortran indices begin at 1 instead of 0 as in Python
                        else:
                            position_particles[compteur] = new_order[j]
                        is_particle_taken[j] = 1
                        compteur += 1
                        break                    
        else:
            position_particles = [-1]
        
        return position_particles

    def calculate_angles_rotation(self, position_particles, all_p, module):
        """ Compute the angles theta and phi given the user's inputs.
            Output: phi and theta for each element of all_p
        """
        if -1 not in position_particles:
            pref = [0, 0, 0, 0]
            phi, theta = [0] * len(all_p), [0] * len(all_p)
            for i in range(len(all_p)):
                for j in range(len(position_particles)):
                    for k in range(len(pref)):
                        pref[k] += all_p[i][position_particles[j] - 1][k]
                phi[i], theta[i] = module.refchoicep(pref)
        else:
            phi, theta = [0] * len(all_p), [0] * len(all_p)

        return phi, theta
    
    def rotation_density(self, module, all_p, phi, theta):
        nexternal = len(all_p[0])
        for i in range(len(all_p)):
                all_p[i] = self.invert_momenta(all_p[i]) #put in fortran format
                all_p[i] = module.rotationp(all_p[i], phi[i], theta[i], nexternal)
                all_p[i] = self.invert_momenta(all_p[i]) #put back into python format

                for j in range(len(all_p[i])):
                    all_p[i][j] = tuple(all_p[i][j])
        return all_p
    
    def permutations_PGD(self, PDG: list[int], status: list[int])-> list[list[int]]:
        """
        Input: a list of PDGs + a list of status
        Output: all the possible PDGs permutations keeping incoming and outcoming particles separate
        """
        from itertools import permutations
        nincoming, noutcoming = 0, 0
        End = []

        for i in range(len(status)):
            if status[i] == -1:
                nincoming += 1
            elif status[i] == +1:
                noutcoming += 1
            elif status[i] == 2 or status[i] == -2: #if the particle is an intermediate particle, we keep them in the final state, it is to the user to not put them in the density matrix
                noutcoming += 1
            else:
                raise ValueError("Status not recognised.")

        InitialState = PDG[0:nincoming]
        FinalState = PDG[nincoming:]
        All_InitialState = list(set(list(permutations(InitialState))))
        All_FinalState = list(set(list(permutations(FinalState))))
        
        list_initial_states = [list(All_InitialState[i]) for i in range(len(All_InitialState))]
        list_final_states = [list(All_FinalState[i]) for i in range(len(All_FinalState))]
        
        for i in range(len(list_initial_states)):
                for j in range(len(list_final_states)):
                        End.append(list_initial_states[i] + list_final_states[j])

        return End