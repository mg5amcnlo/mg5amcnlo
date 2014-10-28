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
        
        # placeholder for information
        self.results = 0 #updated in launch/update_html


        if isinstance(opt, dict):
            self.configure(opt)
        elif isinstance(opt, bannermod.GridpackCard):
            self.configure_gridpack(opt)
          
    def __call__(self):
        return self.launch()
        
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
                logger.info("running for accuracy %s%%" % (self.err_goal*100))
                self.gen_events = False
            elif self.err_goal >= 1:
                logger.info("Generating %s unweigthed events." % self.err_goal)
                self.gen_events = True
                self.err_goal = self.err_goal * self.gen_events_security # security
                
            
        
    def handle_seed(self):
        """not needed but for gridpack --which is not handle here for the moment"""
        return
    
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
    
    def increase_precision(self):
        
        self.max_event_in_iter = 20000
        self.min_events = 7500
        self.gen_events_security = 1.4
    
        
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    def get_job_for_event(self):
        """generate the script in order to generate a given number of event"""
        # correspond to write_gen in the fortran version
        
        
        assert self.err_goal >=1
        self.err_goal = int(self.err_goal)
        
        goal_lum = self.err_goal/(self.results.xsec)    #pb^-1 
        logger.info('Effective Luminosity %s pb^-1', goal_lum)

        
        #reset the potential multijob of previous run
        self.reset_multijob()
        
        all_channels = sum([list(P) for P in self.results],[])
        all_channels.sort(cmp= lambda x,y: 1 if y.get('luminosity') - \
                                                x.get('luminosity') > 0 else -1) 
                          
        
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
                    'channel': C.name.replace('G',''),
                    'grid_refinment' : 0    #no refinment of the grid
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

    def get_job_for_precision(self):
        """create the ajob to achieve a give precision on the total cross-section"""

        
        assert self.err_goal <=1
        xtot = self.results.xsec
        logger.info("Working on precision:  %s %%" %(100*self.err_goal))
        all_channels = sum([list(P) for P in self.results if P.mfactor],[])
        limit = self.err_goal * xtot / len(all_channels)
                
        to_refine = []
        rerr = 0 #error of the job not directly selected
        for C in all_channels:
            cerr = C.mfactor*(C.xerru+C.xerrc**2)
            if  cerr > abs(limit):
                to_refine.append(C)
            else:
                rerr += cerr
        
        if not len(to_refine):
            return
        
        # change limit since most don't contribute 
        limit = math.sqrt((self.err_goal * xtot)**2 - rerr/math.sqrt(len(to_refine)))
        for C in to_refine[:]:
            cerr = C.mfactor*(C.xerru+C.xerrc**2)
            if cerr < limit:
                to_refine.remove(C)
            
        # all the channel are now selected. create the channel information
        logger.info('need to improve %s channels' % len(to_refine))

        
        jobs = [] # list of the refine if some job are split is list of
                  # dict with the parameter of the run.

        # loop over the channel to refine
        for C in to_refine:
            
            #1. Determine how many events we need in each iteration
            yerr = C.get('xsec') + C.mfactor*(C.xerru+C.xerrc**2)
            nevents = 0.2*C.nevents*(yerr/limit)**2
            
            nb_split = int((nevents*(C.nunwgt/C.nevents)/self.max_request_event/ (2**self.min_iter-1))**(2/3))
                           # **(2/3) to slow down the increase in number of jobs            
            if nb_split > self.max_splitting:
                nb_split = self.max_splitting
                
            if nb_split >1:
                nevents = nevents / nb_split
                self.write_multijob(C, nb_split)
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
                    'precision': max(limit/(C.get('xsec')+ yerr), 1e-4),
                    'nhel': self.run_card['nhel'],
                    'channel': C.name.replace('G',''),
                    'grid_refinment' : 1
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
        
    def update_html(self):
        """update the html from this object since it contains all the information"""
        

        run = self.cmd.results.current['run_name']
        if not os.path.exists(pjoin(self.cmd.me_dir, 'HTML', run)):
            os.mkdir(pjoin(self.cmd.me_dir, 'HTML', run))
        
        unit = self.cmd.results.unit
        P_text = "" 
        if self.results:     
            Presults = self.results 
        else:
            self.results = sum_html.collect_result(self.cmd, None)
            Presults = self.results
                
        for P_comb in Presults:
            P_text += P_comb.get_html(run, unit, self.cmd.me_dir) 
        
        Presults.write_results_dat(pjoin(self.cmd.me_dir,'SubProcesses', 'results.dat'))   
        
        fsock = open(pjoin(self.cmd.me_dir, 'HTML', run, 'results.html'),'w')
        fsock.write(sum_html.results_header)
        fsock.write('%s <dl>' % Presults.get_html(run, unit, self.cmd.me_dir))
        fsock.write('%s </dl></body>' % P_text)         
        
        self.cmd.results.add_detail('cross', Presults.xsec)
        self.cmd.results.add_detail('error', Presults.xerru) 
        
        return Presults.xsec, Presults.xerru          


class gen_ximprove_loop_induced(gen_ximprove):
    """Since loop induce is much smaller splits much more the generation."""
    
    
    # some hardcoded value which impact the generation
    gen_events_security = 1.1 # multiply the number of requested event by this number for security
    combining_job = 0         # allow to run multiple channel in sequence
    max_request_event = 400   # split jobs if a channel if it needs more than that 
    max_event_in_iter = 500
    min_event_in_iter = 250
    max_splitting = 260       # maximum duplication of a given channel 
    min_iter = 3    
    max_iter = 6

    def increase_parralelization(self):
        self.max_request_event = 200
        self.max_splitting = 300

def get_ximprove(cmd, opt):
    """Factory Determine the appropriate class and returns it"""
    
    if cmd.proc_characteristics['loop_induced']:
        if cmd.proc_characteristics['nexternal'] <=2:
            return gen_ximprove_loop_induced(cmd, opt)
        else:
            out = gen_ximprove_loop_induced(cmd, opt)
            out.increase_parralelization()
            return out
    elif gen_ximprove.format_variable(cmd.run_card['gridpack'], bool):
        raise Exception, "Not implemented"
    else:
        out = gen_ximprove_loop_induced(cmd, opt)
        if cmd.opts['accuracy'] != cmd._survey_options['accuracy'][1]:
            out.increase_precision()
            
        return out
    


