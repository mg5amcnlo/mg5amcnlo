################################################################################
#
# Copyright (c) 2014 The MadGraph5_aMC@NLO Development team and Contributors
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
""" A python file to replace the fortran script gen_ximprove.
    This script analyses the result of the survey/ previous refine and 
    creates the jobs for the following script.
"""
from __future__ import division

import collections
import os
import glob
import logging
import math

try:
    import madgraph
except ImportError:
    MADEVENT = True
    import internal.sum_html as sum_html
    import internal.banner as bannermod
    import internal.misc as misc
else:
    MADEVENT= False
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.banner as bannermod
    import madgraph.various.misc as misc

logger = logging.getLogger('madgraph.madevent.gen_ximprove')
pjoin = os.path.join

class gen_ximprove(object):
    """the class for collecting data and create the job needed for the next run"""
    
    #convenient shortcut for the formatting of variable
    @ staticmethod
    def format_variable(*args):
        return bannermod.ConfigFile.format_variable(*args)
    
    # some hardcoded value which impact the generation
    gen_events_security = 1.2 # multiply the number of requested event by this number for security
    combining_job = 0         # allow to run multiple channel in sequence
    max_request_event = 1000          # split jobs if a channel if it needs more than that 
    max_event_in_iter = 5000
    min_event_in_iter = 1000
    max_splitting = 130       # maximum duplication of a given channel 
    min_iter = 3    
    max_iter = 9

    def __init__(self, cmd, opt=None):
        
        self.cmd = cmd
        self.run_card = cmd.run_card
        run_card = self.run_card
        self.me_dir = cmd.me_dir
        
        #extract from the run_card the information that we need.
        self.gridpack = run_card['gridpack']
        self.nhel = run_card['nhel']
        if "nhel_refine" in run_card:
            self.nhel = run_card["nhel_refine"]
        
        # Default option for the run
        self.gen_events = True
        self.min_iter = 3
        self.parralel = False
        # parameter which was input for the normal gen_ximprove run
        self.err_goal = 0.01
        self.max_np = 9
        self.split_channels = False
        # parameter for the gridpack run
        self.nreq = 2000
        self.iseed = 4321
        self.ngran = 1
        
        # parameter for 


        if isinstance(opt, dict):
            self.configure(opt)
        elif isinstance(opt, bannermod.GridpackCard):
            self.configure_gridpack(opt)
          
        #automatically launch the code
        self.launch()
        
        
    def launch(self):
        """running """  
        
        #start the run
        self.handle_seed()
        
        self.results = sum_html.collect_result(self.cmd, None)
        
        if self.gen_events:
            # We run to provide a given number of events
            self.get_job_for_event()
        else:
            # We run to achieve a given precision
            self.get_job_for_precision()
            
    
    def configure(self, opt):
        """Defines some parameter of the run"""
        
        for key, value in opt.items():
            if key in self.__dict__:
                targettype = type(getattr(self, key))
                setattr(self, key, self.format_variable(value, targettype, key))
            else:
                raise Exception, '%s not define' % key
                        
            
        # special treatment always do outside the loop to avoid side effect
        if 'err_goal' in opt:
            if self.err_goal < 1:
                logger.info("running for accuracy %s%" % (self.err_goal*100))
                self.gen_events = False
            elif self.err_goal >= 1:
                logger.info("Generating %s unweigthed events." % self.err_goal)
                self.gen_events = True
                self.err_goal = self.err_goal * self.gen_events_security # security
                
            
        
    def handle_seed(self):
        """not sure"""
        logger.critical("""handling of random number bypass here!""")
    
    def reset_multijob(self):

        for path in glob.glob(pjoin(self.me_dir, 'Subprocesses', '*', 
                                                           '*','multijob.dat')):
            open(path,'w').write('0\n')
            
    def write_multijob(self, Channel, nb_split):
        """ """
        if nb_split <=1:
            return
        f = open(pjoin(self.me_dir, 'SubProcesses', Channel.get('name'), 'multijob.dat'), 'w')
        f.write('%i\n' % nb_split)
        f.close()
      
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    def get_job_for_event(self):
        """generate the script in order to generate a given number of event"""
        # correspond to write_gen in the fortran version
        
        
        assert self.err_goal >=1
        self.err_goal = int(self.err_goal)
        logger.info("Working on creating %s events" % int(self.err_goal))
        
        goal_lum = self.err_goal/(self.results.xsec)    #pb^-1 
        logger.info('Effective Luminosity %s pb^-1', goal_lum)
        misc.sprint("use %s for cross-section" % self.results.xsec)
        misc.sprint(open(pjoin(self.me_dir,'SubProcesses/results.dat')).read())
        
        #reset the potential multijob of previous run
        self.reset_multijob()
        
        all_channels = sum([list(P) for P in self.results],[])
        all_channels.sort(cmp= lambda x,y: 1 if y.get('luminosity') - \
                                                x.get('luminosity') > 0 else -1) 
                          
        misc.sprint([P.get('luminosity') for P in all_channels])
        
        to_refine = []
        for C in all_channels:
            if goal_lum/C.get('luminosity') >=1:
                to_refine.append(C)
            elif C.get('xerr') > max(C.get('xsec'), 0.01*all_channels[0].get('xsec')):
                to_refine.append(C)

                    
        logger.info('need to improve %s channels' % len(to_refine))

        
        jobs = [] # list of the refine if some job are split is list of
                  # dict with the parameter of the run.

        # try to have a smart load on the cluster (not really important actually)
        if self.combining_job >1:
            # add a nice ordering for the jobs
            new_order = []
            if self.combining_job % 2 == 0:
                for i in range(len(to_refine) //2):
                    new_order.append(to_refine[i])
                    new_order.append(to_refine[-i-1])
                if len(to_refine) % 2:
                    new_order.append(to_refine[i+1])
            else:
                for i in range(len(to_refine) //3):
                    new_order.append(to_refine[i])
                    new_order.append(to_refine[-2*i-1])                    
                    new_order.append(to_refine[-2*i-2])
                if len(to_refine) % 3 == 1:
                    new_order.append(to_refine[i+1])                                        
                elif len(to_refine) % 3 == 2:
                    new_order.append(to_refine[i+2])  
            #ensure that the reordering is done nicely
            assert set([id(C) for C in to_refine]) == set([id(C) for C in new_order])
            to_refine = new_order      
            
                                                
        # loop over the channel to refine
        for C in to_refine:
            #1. Compute the number of points are needed to reach target
            needed_event = goal_lum*C.get('xsec')
            nb_split = int(max(0,((needed_event-1)// self.max_request_event) +1))            
            if not self.split_channels:
                nb_split = 1
            if nb_split > self.max_splitting:
                nb_split = self.max_splitting
            self.write_multijob(C, nb_split)
            #2. estimate how many points we need in each iteration
            nevents =  needed_event / nb_split * (C.get('nevents') / C.get('nunwgt'))
            #split by iter
            nevents = int(nevents / (2**self.min_iter-1))
            # forbid too low/too large value
            nevents = min(self.min_event_in_iter, max(self.max_event_in_iter, nevents))

            #create the  info dict  assume no splitting for the default
            info = {'name': self.cmd.results.current['run_name'],
                    'script_name': 'unknown',
                    'directory': C.name,    # need to be change for splitted job
                    'P_dir': C.parent_name, 
                    'offset': 1,            # need to be change for splitted job
                    'nevents': nevents,
                    'maxiter': self.max_iter,
                    'miniter': self.min_iter,
                    'precision': -goal_lum/nb_split,
                    'nhel': self.run_card['nhel'],
                    'channel': C.name.replace('G','')
                    }

            if nb_split == 1:
                jobs.append(info)
            else:
                for i in range(nb_split):
                    new_info = dict(info)
                    new_info['offset'] = i+1
                    new_info['directory'] += self.alphabet[i % 26] + str((i+1)//26)
                    jobs.append(new_info)
            
        self.create_ajob(pjoin(self.me_dir, 'SubProcesses', 'refine.sh'), jobs)    
                

    def create_ajob(self, template, jobs):
        """create the ajob"""
        
        if not jobs:
            return
        
        #filter the job according to their SubProcess directory # no mix submition
        P2job= collections.defaultdict(list)
        for j in jobs:
            P2job[j['P_dir']].append(j)
        if len(P2job) >1:
            for P in P2job.values():
                self.create_ajob(template, P)
            return
        
        #Here we can assume that all job are for the same directory.
        path = pjoin(self.me_dir, 'SubProcesses' ,jobs[0]['P_dir'])
        
        template_text = open(template, 'r').read()
        # special treatment if needed to combine the script
        # computes how many submition miss one job
        if self.combining_job > 1:
            skip1=0
            n_channels = len(jobs)
            nb_sub = n_channels // self.combining_job
            nb_job_in_last = n_channels % self.combining_job
            if nb_job_in_last:
                nb_sub +=1
                skip1 = self.combining_job - nb_job_in_last
                if skip1 > nb_sub:
                    self.combining_job -=1
                    return self.create_ajob(template, jobs)
            combining_job = self.combining_job
        else:
            #define the variable for combining jobs even in not combine mode
            #such that we can use the same routine
            skip1=0
            combining_job =1
            nb_sub = len(jobs)
            
            
        nb_use = 0
        for i in range(nb_sub):
            script_number = i+1
            if i < skip1:
                nb_job = combining_job -1
            else:
                nb_job = combining_job
            fsock = open(pjoin(path, 'ajob%i' % script_number), 'w')
            for j in range(nb_use, nb_use + nb_job):
                if j> len(jobs):
                    break
                info = jobs[j]
                info['script_name'] = 'ajob%i' % script_number
                fsock.write(template_text % info)
            nb_use += nb_job 
        
        
class gen_ximprove_gridpack(gen_ximprove):
    """a special case for the gridpack"""
    
    def configure(self, gridpack):
        """ """
           
        self.gen_events = True
        self.split_channels = False
        self.min_iter = 1
        if isinstance(gridpack, bannermod.GridpackCard):
            self.configure(gridpack)
        else:
            gridpack = bannermod.GridpackCard(gridpack)
            self.configure(gridpack)
        if self.ngran == -1:
            self.ngran = 1
        logger.info("Running on Grid to generate %s events with granularity %s" %(self.nreq, self.ngran))
                
    
    def get_job_for_precision(self):
        """ Should not happen in gridpack mode"""
        raise Exception
    
    def get_job_for_event(self):
        """ see what to run for getting a given number of event """
        
        raise NotImplemented





