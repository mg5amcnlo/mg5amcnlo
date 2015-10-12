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
import re
import subprocess
import shutil

try:
    import madgraph
except ImportError:
    MADEVENT = True
    import internal.sum_html as sum_html
    import internal.banner as bannermod
    import internal.misc as misc
    import internal.files as files
    import internal.cluster as cluster
    import internal.combine_grid as combine_grid
    import internal.combine_runs as combine_runs
    import internal.lhe_parser as lhe_parser
else:
    MADEVENT= False
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.banner as bannermod
    import madgraph.various.misc as misc
    import madgraph.iolibs.files as files
    import madgraph.various.cluster as cluster
    import madgraph.madevent.combine_grid as combine_grid
    import madgraph.madevent.combine_runs as combine_runs
    import madgraph.various.lhe_parser as lhe_parser

logger = logging.getLogger('madgraph.madevent.gen_ximprove')
pjoin = os.path.join

class gensym(object):
    """a class to call the fortran gensym executable and handle it's output
    in order to create the various job that are needed for the survey"""
    
    #convenient shortcut for the formatting of variable
    @ staticmethod
    def format_variable(*args):
        return bannermod.ConfigFile.format_variable(*args)
    
    combining_job = 2 # number of channel by ajob
    splitted_grid = False 
    min_iterations = 3
    mode= "survey"
    

    def __init__(self, cmd, opt=None):
        
        try:
            super(gensym, self).__init__(cmd, opt)
        except TypeError:
            pass
        
        # Run statistics, a dictionary of RunStatistics(), with 
        self.run_statistics = {}
        
        self.cmd = cmd
        self.run_card = cmd.run_card
        self.me_dir = cmd.me_dir
        
        
        # dictionary to keep track of the precision when combining iteration
        self.cross = collections.defaultdict(int)
        self.abscross = collections.defaultdict(int)
        self.sigma = collections.defaultdict(int)
        self.chi2 = collections.defaultdict(int)
        
        self.splitted_grid = False
        if self.cmd.proc_characteristics['loop_induced']:
            nexternal = self.cmd.proc_characteristics['nexternal']
            self.splitted_grid = max(2, (nexternal-2)**2)
            if hasattr(self.cmd, "opts") and self.cmd.opts['accuracy'] == 0.1:
                self.cmd.opts['accuracy'] = 0.02
        
        if isinstance(cmd.cluster, cluster.MultiCore) and self.splitted_grid > 1:
            self.splitted_grid = int(cmd.cluster.nb_core**0.5)
            if self.splitted_grid == 1 and cmd.cluster.nb_core >1:
                self.splitted_grid = 2
        
        #if the user defines it in the run_card:
        if self.run_card['survey_splitting'] != -1:
            self.splitted_grid = self.run_card['survey_splitting']
        
        self.splitted_Pdir = {}
        self.splitted_for_dir = lambda x,y: self.splitted_grid
        self.combining_job_for_Pdir = lambda x: self.combining_job
        self.lastoffset = {}
    
    def launch(self):
        """ """
        
        self.subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        subproc = self.subproc
        
        P_zero_result = [] # check the number of times where they are no phase-space
        
        nb_tot_proc = len(subproc)
        for nb_proc,subdir in enumerate(subproc):
            job_list = {}
            self.cmd.update_status('Compiling for process %s/%s. <br> (previous processes already running)' % \
                               (nb_proc+1,nb_tot_proc), level=None)

            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            logger.info('    %s ' % subdir)
            
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(match)
            for match in glob.glob(pjoin(Pdir, 'G*')):
                if os.path.exists(pjoin(match,'results.dat')):
                    os.remove(pjoin(match, 'results.dat')) 
                if os.path.exists(pjoin(match, 'ftn25')):
                    os.remove(pjoin(match, 'ftn25')) 
                    
            #compile gensym
            self.cmd.compile(['gensym'], cwd=Pdir)
            if not os.path.exists(pjoin(Pdir, 'gensym')):
                raise Exception, 'Error make gensym not successful'  
            
            # Launch gensym
            p = misc.Popen(['./gensym'], stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, cwd=Pdir)
            #sym_input = "%(points)d %(iterations)d %(accuracy)f \n" % self.opts
            (stdout, _) = p.communicate('')
            
            if os.path.exists(pjoin(self.me_dir,'error')):
                files.mv(pjoin(self.me_dir,'error'), pjoin(Pdir,'ajob.no_ps.log'))
                P_zero_result.append(subdir)
                continue            
            
            jobs = stdout.split()
            job_list[Pdir] = jobs
            try:
                # check that all input are valid
                [float(s) for s in jobs]
            except Exception:
                logger.debug("unformated string found in gensym. Please check:\n %s" % stdout)
                job_list[Pdir] = []
                for s in jobs:
                    try:
                        float(s)
                    except:
                        continue
                    else:
                        job_list[Pdir].append(s)        
                
            self.cmd.compile(['madevent'], cwd=Pdir)
            self.submit_to_cluster(job_list)
        return job_list, P_zero_result
            
            
    def submit_to_cluster(self, job_list):
        """ """

        if self.run_card['job_strategy'] > 0:
            if len(job_list) >1:
                for path, dirs in job_list.items():
                    self.submit_to_cluster({path:dirs})
                return
            path, value = job_list.items()[0]
            nexternal = self.cmd.proc_characteristics['nexternal']
            current = open(pjoin(path, "nexternal.inc")).read()
            ext = re.search(r"PARAMETER \(NEXTERNAL=(\d+)\)", current).group(1)
            
            if self.run_card['job_strategy'] == 2:
                self.splitted_grid = 2
                if nexternal == int(ext):
                    to_split = 2
                else:
                    to_split = 0
                if hasattr(self, 'splitted_Pdir'):
                    self.splitted_Pdir[path] = to_split
                else:
                    self.splitted_Pdir = {path: to_split}
                    self.splitted_for_dir = lambda x,y : self.splitted_Pdir[x]
            elif self.run_card['job_strategy'] == 1:
                if nexternal == int(ext):
                    combine = 1
                else:
                    combine = self.combining_job
                if hasattr(self, 'splitted_Pdir'):
                    self.splitted_Pdir[path] = combine
                else:
                    self.splitted_Pdir = {path: combine}
                    self.combining_job_for_Pdir = lambda x : self.splitted_Pdir[x]

        if not self.splitted_grid:
            return self.submit_to_cluster_no_splitting(job_list)
        elif self.cmd.cluster_mode == 0:
            return self.submit_to_cluster_no_splitting(job_list)
        elif self.cmd.cluster_mode == 2 and self.cmd.options['nb_core'] == 1:
            return self.submit_to_cluster_no_splitting(job_list)
        else:
            return self.submit_to_cluster_splitted(job_list)
        
    
    def submit_to_cluster_no_splitting(self, job_list):
        """submit the survey without the parralelization.
           This is the old mode which is still usefull in single core"""
     
        # write the template file for the parameter file   
        self.write_parameter(parralelization=False, Pdirs=job_list.keys())
        
        
        # launch the job with the appropriate grouping
        for Pdir, jobs in job_list.items():   
            jobs = list(jobs)
            i=0
            while jobs:
                i+=1
                to_submit = ['0'] # the first entry is actually the offset
                for _ in range(self.combining_job_for_Pdir(Pdir)):
                    if jobs:
                        to_submit.append(jobs.pop(0))
                        
                self.cmd.launch_job(pjoin(self.me_dir, 'SubProcesses', 'survey.sh'),
                                    argument=to_submit,
                                    cwd=pjoin(self.me_dir,'SubProcesses' , Pdir))

                        
    def create_resubmit_one_iter(self, Pdir, G, submit_ps, nb_job, step=0):
        """prepare the input_file for submitting the channel"""

        
        if 'SubProcesses' not in Pdir:
            Pdir = pjoin(self.me_dir, 'SubProcesses', Pdir)

        #keep track of how many job are sended
        self.splitted_Pdir[(Pdir, G)] = int(nb_job)


        # 1. write the new input_app.txt 
        run_card = self.cmd.run_card        
        options = {'event' : submit_ps,
                   'maxiter': 1,
                   'miniter': 1,
                   'accuracy': self.cmd.opts['accuracy'],
                   'helicity': run_card['nhel_survey'] if 'nhel_survey' in run_card \
                            else run_card['nhel'],
                   'gridmode': -2,
                   'channel' : G
                  } 
        
        Gdir = pjoin(Pdir, 'G%s' % G)
        self.write_parameter_file(pjoin(Gdir, 'input_app.txt'), options)   
        
        # 2. check that ftn25 exists.
        assert os.path.exists(pjoin(Gdir, "ftn25"))    
        
        
        # 3. Submit the new jobs
        #call back function
        packet = cluster.Packet((Pdir, G, step+1), 
                                self.combine_iteration,
                                (Pdir, G, step+1))
        
        if step ==0:
            self.lastoffset[(Pdir, G)] = 0 
        
        # resubmit the new jobs            
        for i in xrange(int(nb_job)):
            name = "G%s_%s" % (G,i+1)
            self.lastoffset[(Pdir, G)] += 1
            offset = self.lastoffset[(Pdir, G)]            
            self.cmd.launch_job(pjoin(self.me_dir, 'SubProcesses', 'refine_splitted.sh'),
                                argument=[name, 'G%s'%G, offset],
                                cwd= Pdir,
                                packet_member=packet)


    def submit_to_cluster_splitted(self, job_list):
        """ submit the version of the survey with splitted grid creation 
        """ 
        
        #if self.splitted_grid <= 1:
        #    return self.submit_to_cluster_no_splitting(job_list)

        for Pdir, jobs in job_list.items():
            if self.splitted_for_dir(Pdir, jobs[0]) <= 1:
                return self.submit_to_cluster_no_splitting({Pdir:jobs})

            self.write_parameter(parralelization=True, Pdirs=[Pdir])
            # launch the job with the appropriate grouping

            for job in jobs:
                packet = cluster.Packet((Pdir, job, 1), self.combine_iteration, (Pdir, job, 1))
                for i in range(self.splitted_for_dir(Pdir, job)):    
                    self.cmd.launch_job(pjoin(self.me_dir, 'SubProcesses', 'survey.sh'),
                                    argument=[i+1, job],
                                    cwd=pjoin(self.me_dir,'SubProcesses' , Pdir),
                                    packet_member=packet)

    def combine_iteration(self, Pdir, G, step):

        grid_calculator, cross, error = self.combine_grid(Pdir, G, step)
        
        # Compute the number of events used for this run.                      
        nb_events = grid_calculator.target_evt

        Gdirs = [] #build the the list of directory
        for i in range(self.splitted_for_dir(Pdir, G)):
            path = pjoin(Pdir, "G%s_%s" % (G, i+1))
            Gdirs.append(path)
        
        # 4. make the submission of the next iteration
        #   Three cases - less than 3 iteration -> continue
        #               - more than 3 and less than 5 -> check error
        #               - more than 5 -> prepare info for refine
        need_submit = False
        if step < self.min_iterations and cross != 0:
            if step == 1:
                need_submit = True
            else:
                across = self.abscross[(Pdir,G)]/(self.sigma[(Pdir,G)]+1e-99)
                tot_across = self.get_current_axsec()
                if across / tot_across < 1e-6:
                    need_submit = False
                elif error <  self.cmd.opts['accuracy'] / 100:
                    need_submit = False
                else:
                    need_submit = True
                    
        elif step >= self.cmd.opts['iterations']:
            need_submit = False
        elif self.cmd.opts['accuracy'] < 0:
            #check for luminosity
            raise Exception, "Not Implemented"
        elif self.abscross[(Pdir,G)] == 0:
            need_submit = False 
        else:   
            across = self.abscross[(Pdir,G)]/(self.sigma[(Pdir,G)]+1e-99)
            tot_across = self.get_current_axsec()
            if across == 0:
                need_submit = False
            elif across / tot_across < 1e-5:
                need_submit = False
            elif error >  self.cmd.opts['accuracy']:
                need_submit = True
            else:
                need_submit = False
        
        
        if cross:
            grid_calculator.write_grid_for_submission(Pdir,G,
                        self.splitted_for_dir(Pdir, G),
                        nb_events,mode=self.mode,
                        conservative_factor=5.0)
        
        xsec_format = '.%ig'%(max(3,int(math.log10(1.0/float(error)))+2) 
                              if float(cross)!=0.0 and float(error)!=0.0 else 8)        
        if need_submit:
            message = "%%s/G%%s is at %%%s +- %%.3g pb. Now submitting iteration #%s."%(xsec_format, step+1)
            logger.info(message%\
                        (os.path.basename(Pdir), G, float(cross), 
                                                     float(error)*float(cross)))
            self.resubmit_survey(Pdir,G, Gdirs, step)
        elif cross:
            logger.info("Survey finished for %s/G%s at %s"%(
                    os.path.basename(Pdir),G,('%%%s +- %%.3g pb'%xsec_format))%
                                      (float(cross), float(error)*float(cross)))
            # prepare information for refine
            newGpath = pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G)
            if not os.path.exists(newGpath):
                os.mkdir(newGpath)
                
            # copy the new grid:
            files.cp(pjoin(Gdirs[0], 'ftn25'), 
                         pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G, 'ftn26'))
                        
            # copy the events
            fsock = open(pjoin(newGpath, 'events.lhe'), 'w')
            for Gdir in Gdirs:
                fsock.write(open(pjoin(Gdir, 'events.lhe')).read()) 
            
            # copy one log
            files.cp(pjoin(Gdirs[0], 'log.txt'), 
                         pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G))
            
                               
            # create the appropriate results.dat
            self.write_results(grid_calculator, cross, error, Pdir, G, step)
        else:
            logger.info("Survey finished for %s/G%s [0 cross]", os.path.basename(Pdir),G)
            
            Gdir = pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G)
            if not os.path.exists(Gdir):
                os.mkdir(Gdir)
            # copy one log
            files.cp(pjoin(Gdirs[0], 'log.txt'), Gdir)
            # create the appropriate results.dat
            self.write_results(grid_calculator, cross, error, Pdir, G, step)
            
        return 0

    def combine_grid(self, Pdir, G, step, exclude_sub_jobs=[]):
        """ exclude_sub_jobs is to remove some of the subjobs if a numerical
            issue is detected in one of them. Warning is issue when this occurs.
        """
        
        # 1. create an object to combine the grid information and fill it
        grid_calculator = combine_grid.grid_information(self.run_card['nhel'])
        
        for i in range(self.splitted_for_dir(Pdir, G)):
            if i in exclude_sub_jobs:
                    continue
            path = pjoin(Pdir, "G%s_%s" % (G, i+1)) 
            fsock  = misc.mult_try_open(pjoin(path, 'results.dat'))
            one_result = grid_calculator.add_results_information(fsock)
            fsock.close()
            if one_result.axsec == 0:
                grid_calculator.onefail = True
                continue # grid_information might not exists
            fsock  = misc.mult_try_open(pjoin(path, 'grid_information'))
            grid_calculator.add_one_grid_information(fsock)
            fsock.close()
            os.remove(pjoin(path, 'results.dat'))
            #os.remove(pjoin(path, 'grid_information'))
            
            
             
        #2. combine the information about the total crossection / error
        # start by keep the interation in memory
        cross, across, sigma = grid_calculator.get_cross_section()

        #3. Try to avoid one single PS point which ruins the integration
        #   Should be related to loop evaluation instability.
        maxwgt = grid_calculator.get_max_wgt(0.01)
        if maxwgt:
            nunwgt = grid_calculator.get_nunwgt(maxwgt)
        # Make sure not to apply the security below during the first step of the
        # survey. Also, disregard channels with a contribution relative to the 
        # total cross-section smaller than 1e-8 since in this case it is unlikely
        # that this channel will need more than 1 event anyway.
        apply_instability_security = False
        rel_contrib                = 0.0
        if (self.__class__ != gensym or step > 1):            
            Pdir_across = 0.0
            Gdir_across = 0.0
            for (mPdir,mG) in self.abscross.keys():
                if mPdir == Pdir:
                    Pdir_across += (self.abscross[(mPdir,mG)]/
                                                   (self.sigma[(mPdir,mG)]+1e-99))
                    if mG == G:
                        Gdir_across += (self.abscross[(mPdir,mG)]/
                                                   (self.sigma[(mPdir,mG)]+1e-99)) 
            rel_contrib = abs(Gdir_across/(Pdir_across+1e-99))
            if rel_contrib > (1.0e-8) and \
                                nunwgt < 2 and len(grid_calculator.results) > 1:
                apply_instability_security = True

        if apply_instability_security:
            # check the ratio between the different submit
            th_maxwgt = [(r.th_maxwgt,i) for i,r in enumerate(grid_calculator.results)]
            th_maxwgt.sort()
            ratio = th_maxwgt[-1][0]/th_maxwgt[-2][0]
            if ratio > 1e4:
                logger.warning(
""""One Event with large weight have been found (ratio = %.3g) in channel G%s (with rel.contrib=%.3g).
This is likely due to numerical instabilities. The associated job is discarded to recover.
For offline investigation, the problematic discarded events are stored in:
%s"""%(ratio,G,rel_contrib,pjoin(Pdir,'DiscardedUnstableEvents')))
                exclude_sub_jobs = list(exclude_sub_jobs)
                exclude_sub_jobs.append(th_maxwgt[-1][1])
                grid_calculator.results.run_statistics['skipped_subchannel'] += 1
                
                # Add some monitoring of the problematic events
                gPath = pjoin(Pdir, "G%s_%s" % (G, th_maxwgt[-1][1]+1)) 
                if os.path.isfile(pjoin(gPath,'events.lhe')):
                    lhe_file = lhe_parser.EventFile(pjoin(gPath,'events.lhe'))
                    discardedPath = pjoin(Pdir,'DiscardedUnstableEvents')
                    if not os.path.exists(discardedPath):
                        os.mkdir(discardedPath)    
                    if os.path.isdir(discardedPath):
                        # Keep only the event with a maximum weight, as it surely
                        # is the problematic one.
                        evtRecord = open(pjoin(discardedPath,'discarded_G%s.dat'%G),'a')
                        lhe_file.seek(0) #rewind the file
                        try:
                            evtRecord.write('\n'+str(max(lhe_file,key=lambda evt:abs(evt.wgt))))
                        except Exception:
                            #something wrong write the full file.
                            lhe_file.close()
                            evtRecord.write(pjoin(gPath,'events.lhe').read())
                        evtRecord.close()
                
                return self.combine_grid(Pdir, G, step, exclude_sub_jobs)

        
        if across !=0:
            if sigma != 0:
                self.cross[(Pdir,G)] += cross**3/sigma**2
                self.abscross[(Pdir,G)] += across * cross**2/sigma**2
                self.sigma[(Pdir,G)] += cross**2/ sigma**2
                self.chi2[(Pdir,G)] += cross**4/sigma**2
                # and use those iteration to get the current estimator
                cross = self.cross[(Pdir,G)]/self.sigma[(Pdir,G)]
                if step > 1:
                    error = math.sqrt(abs((self.chi2[(Pdir,G)]/cross**2 - \
                                 self.sigma[(Pdir,G)])/(step-1))/self.sigma[(Pdir,G)])
                else:
                    error = sigma/cross
            else:
                self.cross[(Pdir,G)] = cross
                self.abscross[(Pdir,G)] = across
                self.sigma[(Pdir,G)] = 0
                self.chi2[(Pdir,G)] = 0
                cross = self.cross[(Pdir,G)]
                error = 0
                
        else:
            error = 0
 
        grid_calculator.results.compute_values(update_statistics=True)
        if (str(os.path.basename(Pdir)), G) in self.run_statistics:
            self.run_statistics[(str(os.path.basename(Pdir)), G)]\
                   .aggregate_statistics(grid_calculator.results.run_statistics)
        else:
            self.run_statistics[(str(os.path.basename(Pdir)), G)] = \
                                          grid_calculator.results.run_statistics
    
        self.warnings_from_statistics(G, grid_calculator.results.run_statistics) 
        stats_msg = grid_calculator.results.run_statistics.nice_output(
                                     '/'.join([os.path.basename(Pdir),'G%s'%G]))

        if stats_msg:
            logger.log(5, stats_msg)

        # Clean up grid_information to avoid border effects in case of a crash
        for i in range(self.splitted_for_dir(Pdir, G)):
            path = pjoin(Pdir, "G%s_%s" % (G, i+1))
            try: 
                os.remove(pjoin(path, 'grid_information'))
            except OSError, oneerror:
                if oneerror.errno != 2:
                    raise
        return grid_calculator, cross, error

    def warnings_from_statistics(self,G,stats):
        """Possible warn user for worrying MadLoop stats for this channel."""

        if stats['n_madloop_calls']==0:
            return

        EPS_fraction = float(stats['exceptional_points'])/stats['n_madloop_calls']
        
        msg =  "Channel %s has encountered a fraction of %.3g\n"+ \
         "of numerically unstable loop matrix element computations\n"+\
         "(which could not be rescued using quadruple precision).\n"+\
         "The results might not be trusted."

        if 0.01 > EPS_fraction > 0.001:
             logger.warning(msg%(G,EPS_fraction))
        elif EPS_fraction > 0.01:
             logger.critical((msg%(G,EPS_fraction)).replace('might', 'can'))
             raise Exception, (msg%(G,EPS_fraction)).replace('might', 'can')
    
    def get_current_axsec(self):
        
        across = 0
        for (Pdir,G) in self.abscross:
            across += self.abscross[(Pdir,G)]/(self.sigma[(Pdir,G)]+1e-99)
        return across
    
    def write_results(self, grid_calculator, cross, error, Pdir, G, step):
        
        #compute the value
        if cross == 0:
            abscross,nw, luminosity = 0, 0, 0
            wgt, maxit,nunwgt, wgt, nevents = 0,0,0,0,0
            maxwgt = 0
            error = 0
        else:
            grid_calculator.results.compute_values()
            abscross = self.abscross[(Pdir,G)]/self.sigma[(Pdir,G)]
            nw = grid_calculator.results.nw
            wgt = grid_calculator.results.wgt
            maxit = step
            wgt = 0
            nevents = grid_calculator.results.nevents
            maxwgt = grid_calculator.get_max_wgt()
            nunwgt = grid_calculator.get_nunwgt()
            luminosity = nunwgt/cross
            
        #format the results.dat
        def fstr(nb):
            data = '%E' % nb
            nb, power = data.split('E')
            nb = float(nb) /10
            power = int(power) + 1
            return '%.5fE%+03i' %(nb,power)
        line = '%s %s %s %i %i %i %i %s %s %s %s 0.0 0\n' % \
            (fstr(cross), fstr(error*cross), fstr(error*cross), 
             nevents, nw, maxit,nunwgt,
             fstr(luminosity), fstr(wgt), fstr(abscross), fstr(maxwgt))
                    
        fsock = open(pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G,
                       'results.dat'),'w') 
        fsock.writelines(line)
        fsock.close()
     
    def resubmit_survey(self, Pdir, G, Gdirs, step):
        """submit the next iteration of the survey"""

        # 1. write the new input_app.txt to double the number of points
        run_card = self.cmd.run_card        
        options = {'event' : 2**(step) * self.cmd.opts['points'] / self.splitted_grid,
               'maxiter': 1,
               'miniter': 1,
               'accuracy': self.cmd.opts['accuracy'],
               'helicity': run_card['nhel_survey'] if 'nhel_survey' in run_card \
                            else run_card['nhel'],
               'gridmode': -2,
               'channel' : ''
               } 
        
        if int(options['helicity']) == 1:
            options['event'] = options['event'] * 2**(self.cmd.proc_characteristics['nexternal']//3)
            
        for Gdir in Gdirs:
            self.write_parameter_file(pjoin(Gdir, 'input_app.txt'), options)   
            
        
        #2. resubmit the new jobs
        packet = cluster.Packet((Pdir, G, step+1), self.combine_iteration, \
                                (Pdir, G, step+1))            
        nb_step = len(Gdirs) * (step+1)
        for i,subdir in enumerate(Gdirs):
            subdir = subdir.rsplit('_',1)[1]
            subdir = int(subdir)
            offset = nb_step+i+1
            offset=str(offset)
            tag = "%s.%s" % (subdir, offset)
            
            self.cmd.launch_job(pjoin(self.me_dir, 'SubProcesses', 'survey.sh'),
                                argument=[tag, G],
                                cwd=pjoin(self.me_dir,'SubProcesses' , Pdir),
                                packet_member=packet)
    



    def write_parameter_file(self, path, options):
        """ """
        
        template ="""         %(event)s         %(maxiter)s           %(miniter)s      !Number of events and max and min iterations
  %(accuracy)s    !Accuracy
  %(gridmode)s       !Grid Adjustment 0=none, 2=adjust
  1       !Suppress Amplitude 1=yes
  %(helicity)s        !Helicity Sum/event 0=exact
  %(channel)s      """        
        options['event'] = int(options['event'])
        open(path, 'w').write(template % options)

    
    
    def write_parameter(self, parralelization, Pdirs=None):
        """Write the parameter of the survey run"""

        run_card = self.cmd.run_card
        
        options = {'event' : self.cmd.opts['points'],
                   'maxiter': self.cmd.opts['iterations'],
                   'miniter': self.min_iterations,
                   'accuracy': self.cmd.opts['accuracy'],
                   'helicity': run_card['nhel_survey'] if 'nhel_survey' in run_card \
                                else run_card['nhel'],
                   'gridmode': 2,
                   'channel': ''
                   }
        
        if int(options['helicity'])== 1:
            options['event'] = options['event'] * 2**(self.cmd.proc_characteristics['nexternal']//3)
        
        if parralelization:
            options['gridmode'] = -2
            options['maxiter'] = 1 #this is automatic in dsample anyway
            options['miniter'] = 1 #this is automatic in dsample anyway
            options['event'] /= self.splitted_grid
        
        if not Pdirs:
            Pdirs = self.subproc
               
        for Pdir in Pdirs:
            path =pjoin(Pdir, 'input_app.txt') 
            self.write_parameter_file(path, options)

        
        
class gen_ximprove(object):  
    
    
    # some hardcoded value which impact the generation
    gen_events_security = 1.2 # multiply the number of requested event by this number for security
    combining_job = 0         # allow to run multiple channel in sequence
    max_request_event = 1000          # split jobs if a channel if it needs more than that 
    max_event_in_iter = 5000
    min_event_in_iter = 1000
    max_splitting = 130       # maximum duplication of a given channel 
    min_iter = 3    
    max_iter = 9
    keep_grid_for_refine = False        # only apply if needed to split the job

    #convenient shortcut for the formatting of variable
    @ staticmethod
    def format_variable(*args):
        return bannermod.ConfigFile.format_variable(*args)


    def __new__(cls, cmd, opt):
        """Choose in which type of refine we want to be"""
        
        if cmd.proc_characteristics['loop_induced']:
            return super(gen_ximprove, cls).__new__(gen_ximprove_share, cmd, opt)
        elif gen_ximprove.format_variable(cmd.run_card['gridpack'], bool):
            raise Exception, "Not implemented"
        elif cmd.run_card["job_strategy"] == 2:
            return super(gen_ximprove, cls).__new__(gen_ximprove_share, cmd, opt)
        else:
            return super(gen_ximprove, cls).__new__(gen_ximprove_v4, cmd, opt)
            
            
    def __init__(self, cmd, opt=None):
        
        try:
            super(gen_ximprove, self).__init__(cmd, opt)
        except TypeError:
            pass
        
        self.run_statistics = {}
        self.cmd = cmd
        self.run_card = cmd.run_card
        run_card = self.run_card
        self.me_dir = cmd.me_dir
        
        #extract from the run_card the information that we need.
        self.gridpack = run_card['gridpack']
        self.nhel = run_card['nhel']
        if "nhel_refine" in run_card:
            self.nhel = run_card["nhel_refine"]
        
        if self.run_card['refine_evt_by_job'] != -1:
            self.max_request_event = run_card['refine_evt_by_job']
            
                
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
                    
    
    def find_job_for_event(self):
        """return the list of channel that need to be improved"""
    
        assert self.err_goal >=1
        self.err_goal = int(self.err_goal)
        
        goal_lum = self.err_goal/(self.results.axsec+1e-99)    #pb^-1 
        logger.info('Effective Luminosity %s pb^-1', goal_lum)
        
        all_channels = sum([list(P) for P in self.results],[])
        all_channels.sort(cmp= lambda x,y: 1 if y.get('luminosity') - \
                                                x.get('luminosity') > 0 else -1) 
                          
        to_refine = []
        for C in all_channels:
            if C.get('axsec') == 0:
                continue
            if goal_lum/(C.get('luminosity')+1e-99) >= 1 + (self.gen_events_security-1)/2:
                logger.debug("channel %s is at %s (%s) (%s pb)", C.name,  C.get('luminosity'), goal_lum/(C.get('luminosity')+1e-99), C.get('xsec'))
                to_refine.append(C)
            elif C.get('xerr') > max(C.get('axsec'),
              (1/(100*math.sqrt(self.err_goal)))*all_channels[-1].get('axsec')):
                to_refine.append(C)
         
        logger.info('need to improve %s channels' % len(to_refine))        
        return goal_lum, to_refine

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

    
class gen_ximprove_v4(gen_ximprove):
    
    # some hardcoded value which impact the generation
    gen_events_security = 1.2 # multiply the number of requested event by this number for security
    combining_job = 0         # allow to run multiple channel in sequence
    max_request_event = 1000          # split jobs if a channel if it needs more than that 
    max_event_in_iter = 5000
    min_event_in_iter = 1000
    max_splitting = 130       # maximum duplication of a given channel 
    min_iter = 3    
    max_iter = 9
    keep_grid_for_refine = False        # only apply if needed to split the job



    def __init__(self, cmd, opt=None):     
              
        super(gen_ximprove_v4, self).__init__(cmd, opt)
        
        if cmd.opts['accuracy'] < cmd._survey_options['accuracy'][1]:
            self.increase_precision()

    def reset_multijob(self):

        for path in glob.glob(pjoin(self.me_dir, 'SubProcesses', '*', 
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
        if int(self.nhel) == 1:
            self.min_event_in_iter *= 2**(self.cmd.proc_characteristics['nexternal']//3)
            self.max_event_in_iter *= 2**(self.cmd.proc_characteristics['nexternal']//2)

        self.gen_events_security = 1.3
            
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    def get_job_for_event(self):
        """generate the script in order to generate a given number of event"""
        # correspond to write_gen in the fortran version
        
        
        goal_lum, to_refine = self.find_job_for_event()

        #reset the potential multijob of previous run
        self.reset_multijob()
        
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
            needed_event = goal_lum*C.get('axsec')
            nb_split = int(max(1,((needed_event-1)// self.max_request_event) +1))
            if not self.split_channels:
                nb_split = 1
            if nb_split > self.max_splitting:
                nb_split = self.max_splitting
            nb_split=max(1, nb_split)

            
            #2. estimate how many points we need in each iteration
            if C.get('nunwgt') > 0:
                nevents =  needed_event / nb_split * (C.get('nevents') / C.get('nunwgt'))
                #split by iter
                nevents = int(nevents / (2**self.min_iter-1))
            else:
                nevents = self.max_event_in_iter

            if nevents < self.min_event_in_iter:
                nb_split = int(nb_split * nevents / self.min_event_in_iter) + 1
                nevents = self.min_event_in_iter
            #
            # forbid too low/too large value
            nevents = max(self.min_event_in_iter, min(self.max_event_in_iter, nevents))
            logger.debug("%s : need %s event. Need %s split job of %s points", C.name, needed_event, nb_split, nevents)

            
            # write the multi-job information
            self.write_multijob(C, nb_split)
            
            packet = cluster.Packet((C.parent_name, C.name),
                                    combine_runs.CombineRuns,
                                    (pjoin(self.me_dir, 'SubProcesses', C.parent_name)),
                                    {"subproc": C.name, "nb_split":nb_split})
                                     
            
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
                    'grid_refinment' : 0,    #no refinment of the grid
                    'base_directory': '',   #should be change in splitted job if want to keep the grid
                    'packet': packet, 
                    }

            if nb_split == 1:
                jobs.append(info)
            else:
                for i in range(nb_split):
                    new_info = dict(info)
                    new_info['offset'] = i+1
                    new_info['directory'] += self.alphabet[i % 26] + str((i+1)//26)
                    if self.keep_grid_for_refine:
                        new_info['base_directory'] = info['directory']
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
                if "base_directory" not in info:
                    info["base_directory"] = "./"
                fsock.write(template_text % info)
            nb_use += nb_job 

    def get_job_for_precision(self):
        """create the ajob to achieve a give precision on the total cross-section"""

        
        assert self.err_goal <=1
        xtot = abs(self.results.xsec)
        logger.info("Working on precision:  %s %%" %(100*self.err_goal))
        all_channels = sum([list(P) for P in self.results if P.mfactor],[])
        limit = self.err_goal * xtot / len(all_channels)
        to_refine = []
        rerr = 0 #error of the job not directly selected
        for C in all_channels:
            cerr = C.mfactor*(C.xerru + len(all_channels)*C.xerrc)
            if  cerr > abs(limit):
                to_refine.append(C)
            else:
                rerr += cerr
        rerr *=rerr
        if not len(to_refine):
            return
        
        # change limit since most don't contribute 
        limit = math.sqrt((self.err_goal * xtot)**2 - rerr/math.sqrt(len(to_refine)))
        for C in to_refine[:]:
            cerr = C.mfactor*(C.xerru + len(to_refine)*C.xerrc)
            if cerr < limit:
                to_refine.remove(C)
            
        # all the channel are now selected. create the channel information
        logger.info('need to improve %s channels' % len(to_refine))

        
        jobs = [] # list of the refine if some job are split is list of
                  # dict with the parameter of the run.

        # loop over the channel to refine
        for C in to_refine:
            
            #1. Determine how many events we need in each iteration
            yerr = C.mfactor*(C.xerru+len(to_refine)*C.xerrc)
            nevents = 0.2*C.nevents*(yerr/limit)**2
            
            nb_split = int((nevents*(C.nunwgt/C.nevents)/self.max_request_event/ (2**self.min_iter-1))**(2/3))
            nb_split = max(nb_split, 1)
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
                    'precision': yerr/math.sqrt(nb_split)/(C.get('xsec')+ yerr),
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




class gen_ximprove_v4_nogridupdate(gen_ximprove_v4):

    # some hardcoded value which impact the generation
    gen_events_security = 1.1 # multiply the number of requested event by this number for security
    combining_job = 0         # allow to run multiple channel in sequence
    max_request_event = 400   # split jobs if a channel if it needs more than that 
    max_event_in_iter = 500
    min_event_in_iter = 250
    max_splitting = 260       # maximum duplication of a given channel 
    min_iter = 2    
    max_iter = 6
    keep_grid_for_refine = True


    def __init__(self, cmd, opt=None):     
              
        gen_ximprove.__init__(cmd, opt)
        
        if cmd.proc_characteristics['loopinduced'] and \
           cmd.proc_characteristics['nexternal']  > 2:
            self.increase_parralelization(cmd.proc_characteristics['nexternal'])
            
    def increase_parralelization(self, nexternal):

        self.max_splitting = 1000   
             
        if self.run_card['refine_evt_by_job'] != -1:
            pass
        elif nexternal == 3:
            self.max_request_event = 200
        elif nexternal == 4:
            self.max_request_event = 100
        elif nexternal >= 5:
            self.max_request_event = 50
            self.min_event_in_iter = 125
            self.max_iter = 5

class gen_ximprove_share(gen_ximprove, gensym):
    """Doing the refine in multicore. Each core handle a couple of PS point."""

    nb_ps_by_job = 2000 
    mode = "refine"
    gen_events_security = 1.15
    # Note the real security is lower since we stop the jobs if they are at 96%
    # of this target.

    def __init__(self, *args, **opts):
        
        super(gen_ximprove_share, self).__init__(*args, **opts)
        self.generated_events = {}
        self.splitted_for_dir = lambda x,y : self.splitted_Pdir[(x,y)]
        

    def get_job_for_event(self):
        """generate the script in order to generate a given number of event"""
        # correspond to write_gen in the fortran version
        

        goal_lum, to_refine = self.find_job_for_event()
        self.goal_lum = goal_lum
        
        # loop over the channel to refine to find the number of PS point to launch
        total_ps_points = 0
        channel_to_ps_point = []
        for C in to_refine:
            #0. remove previous events files
            try:
                os.remove(pjoin(self.me_dir, "SubProcesses",C.parent_name, C.name, "events.lhe"))
            except:
                pass
            
            #1. Compute the number of points are needed to reach target
            needed_event = goal_lum*C.get('axsec')
            if needed_event == 0:
                continue
            #2. estimate how many points we need in each iteration
            if C.get('nunwgt') > 0:
                nevents =  needed_event * (C.get('nevents') / C.get('nunwgt'))
                #split by iter
                nevents = int(nevents / (2**self.min_iter-1))
            else:
                nb_split = int(max(1,((needed_event-1)// self.max_request_event) +1))
                if not self.split_channels:
                    nb_split = 1
                if nb_split > self.max_splitting:
                    nb_split = self.max_splitting
                    nevents = self.max_event_in_iter * self.max_splitting          
                else:
                    nevents = self.max_event_in_iter * nb_split

            if nevents > self.max_splitting*self.max_event_in_iter:
                logger.warning("Channel %s/%s has a very low efficiency of unweighting. Might not be possible to reach target" % \
                                                (C.name, C.parent_name))
                nevents = self.max_event_in_iter * self.max_splitting 
                
            total_ps_points += nevents 
            channel_to_ps_point.append((C, nevents)) 
        
        if self.cmd.options["run_mode"] == 1:
            if self.cmd.options["cluster_size"]:
                nb_ps_by_job = total_ps_points /int(self.cmd.options["cluster_size"])
            else:
                nb_ps_by_job = self.nb_ps_by_job
        elif self.cmd.options["run_mode"] == 2:
            remain = total_ps_points % self.cmd.options["nb_core"]
            if remain:
                nb_ps_by_job = 1 + (total_ps_points - remain) / self.cmd.options["nb_core"]
            else:
                nb_ps_by_job = total_ps_points / self.cmd.options["nb_core"]
        else:
            nb_ps_by_job = self.nb_ps_by_job
            
        nb_ps_by_job = int(max(nb_ps_by_job, 500))

        for C, nevents in channel_to_ps_point:
            if nevents % nb_ps_by_job:
                nb_job = 1 + int(nevents // nb_ps_by_job)
            else:
                nb_job = int(nevents // nb_ps_by_job)
            submit_ps = min(nevents, nb_ps_by_job)
            if nb_job == 1:
                submit_ps = max(submit_ps, self.min_event_in_iter)
            self.create_resubmit_one_iter(C.parent_name, C.name[1:], submit_ps, nb_job, step=0)
            needed_event = goal_lum*C.get('xsec')
            logger.debug("%s/%s : need %s event. Need %s split job of %s points", C.parent_name, C.name, needed_event, nb_job, submit_ps)
            
        
    def combine_iteration(self, Pdir, G, step):
        
        grid_calculator, cross, error = self.combine_grid(Pdir, G, step)
        
        # collect all the generated_event
        Gdirs = [] #build the the list of directory
        for i in range(self.splitted_for_dir(Pdir, G)):
            path = pjoin(Pdir, "G%s_%s" % (G, i+1))
            Gdirs.append(path)
        assert len(grid_calculator.results) == len(Gdirs) == self.splitted_for_dir(Pdir, G)
        
                
        # Check how many events are going to be kept after un-weighting.
        needed_event = cross * self.goal_lum
        if needed_event == 0:
            return 0
        # check that the number of events requested is not higher than the actual
        #  total number of events to generate.
        if self.err_goal >=1:
            if needed_event > self.gen_events_security * self.err_goal:
                needed_event = int(self.gen_events_security * self.err_goal)
        
        if (Pdir, G) in self.generated_events:
            old_nunwgt, old_maxwgt = self.generated_events[(Pdir, G)]
        else:
            old_nunwgt, old_maxwgt = 0, 0
        
        if old_nunwgt == 0 and os.path.exists(pjoin(Pdir,"G%s" % G, "events.lhe")):
            # possible for second refine.
            lhe = lhe_parser.EventFile(pjoin(Pdir,"G%s" % G, "events.lhe"))
            old_nunwgt = lhe.unweight(None, trunc_error=0.005, log_level=0)
            old_maxwgt = lhe.max_wgt
            
              

        maxwgt = max(grid_calculator.get_max_wgt(), old_maxwgt)
        new_evt = grid_calculator.get_nunwgt(maxwgt)
        efficiency = new_evt / sum([R.nevents for R in grid_calculator.results])
        nunwgt = old_nunwgt * old_maxwgt / maxwgt
        nunwgt += new_evt

        # check the number of event for this iteration alone
        one_iter_nb_event = grid_calculator.get_nunwgt()
        drop_previous_iteration = False
        # compare the number of events to generate if we discard the previous iteration
        n_target_one_iter = (needed_event-one_iter_nb_event) / ( one_iter_nb_event/ sum([R.nevents for R in grid_calculator.results])) 
        n_target_combined = (needed_event-nunwgt) / efficiency
        if n_target_one_iter < n_target_combined:
            # the last iteration alone has more event that the combine iteration.
            # it is therefore interesting to drop previous iteration.          
            drop_previous_iteration = True
            nunwgt = one_iter_nb_event
            maxwgt = grid_calculator.get_max_wgt()
            new_evt = nunwgt
            efficiency = ( one_iter_nb_event/ sum([R.nevents for R in grid_calculator.results])) 
            
        try:
            if drop_previous_iteration:
                raise IOError
            output_file = open(pjoin(Pdir,"G%s" % G, "events.lhe"), 'a')
        except IOError:
            output_file = open(pjoin(Pdir,"G%s" % G, "events.lhe"), 'w')
                
        misc.call(["cat"] + [pjoin(d, "events.lhe") for d in Gdirs],
                  stdout=output_file)
        output_file.close()
        # For large number of iteration. check the number of event by doing the
        # real unweighting.
        if nunwgt < 0.6 * needed_event and step > self.min_iter:            
            lhe = lhe_parser.EventFile(output_file.name)
            old_nunwgt =nunwgt
            nunwgt = lhe.unweight(None, trunc_error=0.01, log_level=0)
        
    
        self.generated_events[(Pdir, G)] = (nunwgt, maxwgt)

        # misc.sprint("Adding %s event to %s. Currently at %s" % (new_evt, G, nunwgt))
        # check what to do
        if nunwgt >= int(0.96*needed_event)+1: # 0.96*1.15=1.10 =real security
            # We did it.
            logger.info("found enough event for %s/G%s" % (os.path.basename(Pdir), G))
            self.write_results(grid_calculator, cross, error, Pdir, G, step, efficiency)
            return 0
        elif step >= self.max_iter:
            logger.debug("fail to find enough event")
            self.write_results(grid_calculator, cross, error, Pdir, G, step, efficiency)
            return 0

        nb_split_before = len(grid_calculator.results)
        nevents = grid_calculator.results[0].nevents
        if nevents == 0: # possible if some integral returns 0
            nevents = max(g.nevents for g in grid_calculator.results)
        
        need_ps_point = (needed_event - nunwgt)/(efficiency+1e-99)
        need_job = need_ps_point // nevents + 1        
        
        if step < self.min_iter:
            # This is normal but check if we are on the good track
            job_at_first_iter = nb_split_before/2**(step-1) 
            expected_total_job = job_at_first_iter * (2**self.min_iter-1)
            done_job = job_at_first_iter * (2**step-1)
            expected_remaining_job = expected_total_job - done_job

            logger.debug("efficiency status (smaller is better): %s", need_job/expected_remaining_job)            
            # increase if needed but not too much
            need_job = min(need_job, expected_remaining_job*1.25)
            
            nb_job = (need_job-0.5)//(2**(self.min_iter-step)-1) + 1
            nb_job = max(1, nb_job)
            grid_calculator.write_grid_for_submission(Pdir,G,
                self.splitted_for_dir(Pdir, G), nb_job*nevents ,mode=self.mode,
                                              conservative_factor=self.max_iter)
            logger.info("%s/G%s is at %i/%i (%.2g%%) event. Resubmit %i job at iteration %i." \
                 % (os.path.basename(Pdir), G, int(nunwgt),int(needed_event)+1,
                 (float(nunwgt)/needed_event)*100.0 if needed_event>0.0 else 0.0,
                                                                  nb_job, step))
            self.create_resubmit_one_iter(Pdir, G, nevents, nb_job, step)
            #self.create_job(Pdir, G, nb_job, nevents, step)
        
        elif step < self.max_iter:
            if step + 1 == self.max_iter:
                need_job = 1.20 * need_job # avoid to have just too few event.

            nb_job = int(min(need_job, nb_split_before*1.5))
            grid_calculator.write_grid_for_submission(Pdir,G,
                self.splitted_for_dir(Pdir, G), nb_job*nevents ,mode=self.mode,
                                              conservative_factor=self.max_iter)
            
            
            logger.info("%s/G%s is at %i/%i ('%.2g%%') event. Resubmit %i job at iteration %i." \
              % (os.path.basename(Pdir), G, int(nunwgt),int(needed_event)+1,
                 (float(nunwgt)/needed_event)*100.0 if needed_event>0.0 else 0.0,
                                                                  nb_job, step))
            self.create_resubmit_one_iter(Pdir, G, nevents, nb_job, step)
            
            

        return 0
    
        
    def write_results(self, grid_calculator, cross, error, Pdir, G, step, efficiency):
        
        #compute the value
        if cross == 0:
            abscross,nw, luminosity = 0, 0, 0
            wgt, maxit,nunwgt, wgt, nevents = 0,0,0,0,0
            error = 0
        else:
            grid_calculator.results.compute_values()
            abscross = self.abscross[(Pdir,G)]/self.sigma[(Pdir,G)]
            nunwgt, wgt = self.generated_events[(Pdir, G)]
            nw = int(nunwgt / efficiency)
            nunwgt = int(nunwgt)
            maxit = step
            nevents = nunwgt
            # make the unweighting to compute the number of events:
            luminosity = nunwgt/cross
                      
        #format the results.dat
        def fstr(nb):
            data = '%E' % nb
            nb, power = data.split('E')
            nb = float(nb) /10
            power = int(power) + 1
            return '%.5fE%+03i' %(nb,power)
        line = '%s %s %s %i %i %i %i %s %s %s 0.0 0.0 0\n' % \
            (fstr(cross), fstr(error*cross), fstr(error*cross), 
             nevents, nw, maxit,nunwgt,
             fstr(luminosity), fstr(wgt), fstr(abscross))
                    
        fsock = open(pjoin(self.me_dir,'SubProcesses' , Pdir, 'G%s' % G,
                       'results.dat'),'w') 
        fsock.writelines(line)
        fsock.close()

    
    
    
    
        
