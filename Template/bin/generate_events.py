#! /usr/bin/env python
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
""" This is the main script in order to generate events in MadEvent """

import logging
import os
import re
import shutil
import subprocess
import sys
import time

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
pjoin = os.path.join
sys.path.append(pjoin(root_path,'bin','internal'))

import check_param_card

logger = logging.getLogger('madevent')

class MERunError(Exception):
    pass

class MadEventLauncher(object):
    
    # Store the path of the important executables
    main = root_path
    # Strip off last directory
    bin = pjoin(main, 'bin')
    dirbin = pjoin(main, 'bin', 'internal')
    web = 0
    
    ############################################################################
    @classmethod
    def find_main_executables(cls):
        """ find the path of the main executables files.
            1) read from file ./executables_path.dat
            2) try to use default
            3) store None
        """
    
        # Define default
        mainpar = os.path.split(root_path)[0] # MG_ME directory
        cls.pydir = cls.check_dir(pjoin(mainpar, 'pythia-pgs','src'))
        cls.pgsdir = cls.pydir
        cls.delphesdir = cls.check_dir(pjoin(mainpar, 'Delphes'))
        cls.eradir = cls.check_dir(pjoin(mainpar, 'ExRootAnalysis'))
        cls.madir = cls.check_dir(pjoin(mainpar, 'MadAnalysis'))
        cls.td = cls.check_exec(pjoin(mainpar, 'td'))
        if cls.td:
            cls.td = os.path.dirname(cls.td)
            
        # read file ./executables_path.dat to overwrite default
        if not os.path.exists(pjoin(cls.main,'executables_path.dat')):
            return
        for line in file(pjoin(cls.main,'executables_path.dat')):
            line = line.strip()
            line = line.split('#')[0]
            line = line.split()
            if len(line) != 3 or line[1] != '=':
                continue # wrongly formatted line
            if line[0] == 'pythia-pgs':
                # check absolute relative and relative to current dir
                cls.pydir = cls.check_dir(line[3], cls.pydir)
                # Try path from maindir
                path = pjoin(cls.main, line[3])
                cls.pydir = cls.check_dir(path, cls.pydir)
                cls.pgsdir = cls.pydir
            
            elif line[0] == 'delphes':
                cls.delphesdir = cls.check_dir(line[3], cls.delphesdir)
                # Try path from maindir
                path = pjoin(cls.main, line[3])
                cls.delphesdir = cls.check_dir(path, cls.delphesdir)
            
            elif line[0] == 'exrootanalysis':
                cls.eradir = cls.check_dir(line[3], cls.eradir)
                # Try path from maindir
                path = pjoin(cls.main, line[3])
                cls.eradir = cls.check_dir(path, cls.eradir)
                
            elif line[0] == 'madanalysis':
                cls.madir = cls.check_dir(line[3], cls.madir)
                # Try path from maindir
                path = pjoin(cls.main, line[3])
                cls.madir = cls.check_dir(path, cls.madir)
            
            elif line[0] == 'td':
                cls.td = cls.check_dir(line[3], cls.td)
                # Try path from maindir
                path = pjoin(cls.main, line[3])
                cls.td = cls.check_dir(path, cls.td)
                if cls.td:
                    cls.td = os.path.dirname(cls.td)
            else:
                logger.warning('''file executables_path.dat contains configuration for
                %s which is not supported''' % line[0])    
    
    ############################################################################                        
    @staticmethod
    def check_exec(path, default=''):
        """check if the executable exists. if so return the path otherwise the default"""
        
        if is_executable(path):
            return path
        else:
            return default
    
    ############################################################################
    @staticmethod
    def check_dir(path, default=''):
        """check if the directory exists. if so return the path otherwise the default"""
         
        if os.path.isdir(path):
            return path
        else:
            return default  

    ############################################################################
    @classmethod
    def pass_in_web_mode(cls):
        """ reconfigure path for the web"""
        
        cls.web=1
        mg_base = os.environ['MADGRAPH_BASE']
        cls.webbin="%s/MG_ME/WebBin" % mg_base
        cls.pydir="%s/pythia-pgs" % cls.webbin
        cls.pgsdir = cls.pydir
        cls.delphesdir="%s/Delphes" % cls.webbin
        cls.eradir="%s/MG_ME/ExRootAnalysis" % mg_base
        cls.madir="%s/MG_ME/MadAnalysis" % mg_base
        cls.td="%s/MG_ME/td" % mg_base
      
    ############################################################################
    ##  INSTANCE METHOD
    ############################################################################
    def __init__(self, cluster_mode, name='run', cluster_queue='madgraph', nb_core=1):
        """ launch generate_events"""
        
        self.find_main_executables()
        # Store the shell id
        os.system('echo $$ > %s' % pjoin(self.main, 'myprocid'))
        
        self.name = name
        self.cluster_mode = cluster_mode
        self.cluster_queue = cluster_queue
        self.nb_core = nb_core
    
    ############################################################################
    def launch(self, param_card='./Cards/param_card.dat', 
                     run_card='./Cards/run_card.dat'):
        """ """
        
        self.status = pjoin(self.main, 'status')
        self.error =  pjoin(self.main, 'error')
        
        if self.web:
            os.system('touch Online')

        # Check if we need the MSSM special treatment
        model = self.find_model_name()
        if model == 'mssm' or model.startswith('mssm-'):
            param_card = pjoin(self.main, 'Cards','param_card.dat')
            mg5_param = pjoin(self.main, 'Source', 'MODEL', 'MG5_param.dat')
            check_param_card.convert_to_mg5card(param_card, mg5_param)
            check_param_card.check_valid_param_card(mg5_param)

        # check some special information
        run_data = self.read_run_card(run_card)
        
        # limit the number of event to 100k
        nb_event = self.check_nb_events(run_card, run_data)
        logger.info("Generating %s events" % nb_event)
        
        if run_data['gridpack'] in ['T','.true.',True,'true']:
            logger.info("Generating GridPack")

        #  Check if run already exists. If so, store run w/ new name
        #  and remove old run before starting.
        if os.path.exists(self.status):
            os.remove(self.status)
        if os.path.exists(self.error):
            os.remove(self.error)
        os.system('touch RunWeb')
        logger.info('Cleaning directories')
        os.system('echo \"Cleaning directories\" > %s' % self.status)

        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        os.system('%s/clean' % (self.dirbin))
        #
        # LHAPDF INTERFACE
        #
        if run_data['pdlabel'] == "'lhapdf'":
            os.environ['lhapdf'] = True
        elif 'lhapdf' in os.environ.keys():
            del os.environ['lhapdf']
        #
        # Compile Source
        #   
        self.compile_source()
        #
        # ICKKW=2
        #
        if run_data['ickkw'] == 2:
            logger.info('Running with CKKW matching')
            self.treat_CKKW_matching()
        #
        # SURVEY / REFINE
        #
        self.survey()
        if not run_data['gridpack'] in ['T','.true.',True,'true']:
            self.refine(run_data)
        #
        #  Collect the events
        #  
        logger.info('Combining Events')
        self.combine_events()
        #
        #  do the banner
        #
        logger.info('putting the banner')
        self.put_the_banner()

        #
        #  Create root file/plot
        #
        if is_executable(pjoin(self.eradir,'ExRootLHEFConverter'))  and\
           os.path.exists(pjoin(self.main, 'Events', 'unweighted_events.lhe')):
                logger.info('Creating Root File')
                self.create_root_file()
                
        if self.madir and self.td and \
            os.path.exists(pjoin(self.main, 'Events', 'unweighted_events.lhe')) and \
            os.path.exists(pjoin(self.main, 'Cards', 'plot_card.dat')):
                logger.info('Creating Plots')
                self.create_plot()
        #
        # STORE
        #
        subprocess.call(['%s/store' % self.dirbin, self.name],
                            cwd=pjoin(self.main, 'Events'))
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        shutil.copy(pjoin(self.main, 'Events', self.name+'_banner.txt'),
                    pjoin(self.main, 'Events', 'banner.txt'))
        #
        # finish GridPack
        #
        if run_data['gridpack'] in  ['T','.true.',True,'true']:
            self.finalize_grid_pack() 
        #
        #  Run Pythia 
        #
        if is_executable(pjoin(self.pydir, 'pythia')) and \
               os.path.exists(pjoin(self.main, 'Cards', 'pythia_card.dat')):
                logger.info('Running Pythia')
                self.run_pythia(run_data)
        #
        #  Run PGS/Delphes 
        #        
        if self.pgsdir and os.path.exists(pjoin(self.main, 'Cards', 'pgs_card.dat')):
                logger.info('Running PGS')
                self.run_pgs()     
        elif is_executable(pjoin(self.delphesdir, 'Delphes')) and\
               os.path.exists(pjoin(self.main, 'Cards', 'delphes_card.dat')):
                logger.info('Running Delphes')
                self.run_delphes()         
        #
        #  Store Events
        #
        logger.info("Storing Events")
        open(self.status, 'w').writelines("Storing Events")
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        subprocess.call(['%s/store' % self.dirbin, self.name],
                            cwd=self.main)
        #
        # FINALIZE
        #
        os.remove(pjoin(self.main, 'RunWeb'))
        open(self.status, 'w').writelines(" ")
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        os.system('%s/gen_cardhtml-pl' % (self.dirbin))
        logger.info(time.ctime())

    ############################################################################                       
    def survey(self):
        """ make the survey """
        
        os.system('touch %s' % pjoin(self.main, 'survey'))
        os.system('echo \"Starting jobs \" > %s' % self.status)
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        
        if self.cluster_mode == 0:
            args = "0 %s" % self.name
        elif self.cluster_mode == 1:
            args = "1 %s %s " % (self.cluster_queue, self.name) 
        elif self.cluster_mode == 2:
            args = "2 %s %s " % (self.nb_core, self.name) 
        
        os.system('%s/survey %s ' % (self.bin, args))

        if os.path.exists(self.error):
            logger.error(open(self.error).read())
            logger.info(time.ctime())
            os.remove(pjoin(self.main, 'survey'))
            os.remove(pjoin(self.main, 'RunWeb'))
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            os.system('%s/gen_cardhtml-pl' % (self.dirbin))
        
        os.remove(pjoin(self.main, 'survey'))
            
    ############################################################################
    def refine(self, run_data):
        """ make the refine """
        
        mode = self.find_madevent_mode()
        if mode == 'group':
            nb_loop = 1
        else:
            nb_loop = 5
        
        os.system('touch %s' % pjoin(self.main, 'refine'))
        
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        
        nb_event = run_data['nevents']
        if self.cluster_mode == 0:
            args = "%s 0 %s" % (nb_event, self.name)
        elif self.cluster_mode == 1:
            args = "%s 1 %s %s %s " % (nb_event,self.cluster_queue, nb_loop, self.name) 
        elif self.cluster_mode == 2:
            args = "%s 2 %s %s %s " % (nb_event, self.nb_core, nb_loop, self.name) 
        
        os.system('%s/refine %s ' % (self.bin, args))

        shutil.move(pjoin(self.main, 'refine'),
                    pjoin(self.main, 'refine2'))
                
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        os.system('%s/refine %s ' % (self.bin, args))
        os.remove(pjoin(self.main, 'refine2'))
   
    ############################################################################
    def compile_source(self):
        """Compile the Source directory and check that all compilation suceed"""
          
        os.system("echo \"Cleaning directories\" >& %s" % self.status)  
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        
        
        p = subprocess.Popen([pjoin(self.dirbin, 'compile_Source')], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, cwd=self.main)
        p.wait()
        if p.returncode:
            raise MERunError, 'Impossible to Compile Source directory' 
        
    ############################################################################
    def combine_events(self):
        """Combine the events """
    
        os.system("echo \"Combining Events\" >& %s" % self.status)
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        subprocess.call(['make','../bin/internal/combine_events'],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = os.open(os.devnull, os.O_RDWR),
                            cwd=pjoin(self.main, 'Source'))
        subprocess.call(['%s/run_combine' % self.dirbin, str(self.cluster_mode)],
                            cwd=pjoin(self.main, 'SubProcesses'))

        shutil.move(pjoin(self.main, 'SubProcesses', 'events.lhe'),
                    pjoin(self.main, 'Events', 'events.lhe'))
        shutil.move(pjoin(self.main, 'SubProcesses', 'unweighted_events.lhe'),
                    pjoin(self.main, 'Events', 'unweighted_events.lhe'))        

    ############################################################################
    def put_the_banner(self):
        """ write the banner at the top of the event file """
        
        subprocess.call(['%s/put_banner' % self.dirbin, 'events.lhe'],
                            cwd=pjoin(self.main, 'Events'))
        subprocess.call(['%s/put_banner'% self.dirbin, 'unweighted_events.lhe'],
                            cwd=pjoin(self.main, 'Events'))
        
        if os.path.exists(pjoin(self.main, 'Events', 'unweighted_events.lhe')):
            subprocess.call(['%s/extract_banner-pl' % self.dirbin, 
                             'unweighted_events.lhe', 'banner.txt'],
                            cwd=pjoin(self.main, 'Events'))
        
    ############################################################################
    def create_root_file(self):
        """create the LHE root file """
        
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        subprocess.call(['%s/ExRootLHEFConverter' % self.eradir, 
                             'unweighted_events.lhe', 'unweighted_events.root'],
                            cwd=pjoin(self.main, 'Events'))
        
    ############################################################################
    def create_plot(self):
        """create the plot""" 
        plot_dir = pjoin(self.main, 'Events', self.name)
        os.system("echo \"Creating Plots\" >& %s" % self.status)
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir) 
            
        os.system("echo \"../unweighted_events.lhe\" > %s" % pjoin(plot_dir, 'events.list'))
        subprocess.call(['%s/plot' % self.dirbin, self.madir, self.td],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir)
    
        subprocess.call(['%s/plot_page-pl' % self.dirbin, self.name, 'parton'],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.main, 'Events'))
       
        shutil.move(pjoin(self.main, 'Events', 'plots.html'),
                   pjoin(self.main, 'Events', '%s_plots.html' % self.name))   

    ############################################################################
    def finalize_grid_pack(self):
        """Finalize the grid pack""" 
    
        os.system('sed -i.bak "s/\s*.false.*=.*GridRun/  .true.  =  GridRun/g" %s' 
                  % pjoin(self.main, 'Cards','grid_card.dat'))
            
        subprocess.call(['%s/restore_data' % self.dirbin, self.name],
                            cwd=self.main)
        subprocess.call(['%s/store4grid' % self.dirbin, 'default'],
                            cwd=self.main)
        subprocess.call(['%s/clean' % self.dirbin],
                            cwd=self.main) 
        subprocess.call(['make', 'gridpack.tar.gz'],
                            cwd=self.main)                        

    ############################################################################
    def run_pythia(self, run_data):
        """ Run pythia and make the associate plot """
                        
        open(self.status,'w').writelines('Running Pythia')
        os.system("gunzip -c %(path)s/%(name)s_unweighted_events.lhe.gz > %(path)s/unweighted_events.lhe"\
                   % {'path': pjoin(self.main,'Events') ,'name':self.name})
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        # shower and hadronize event through Pythia
        subprocess.call([self.dirbin+'/run_pythia', self.pydir, str(self.cluster_mode)],
                            cwd=pjoin(self.main,'Events'))       

        if not os.path.exists(pjoin(self.main,'Events','pythia_events.hep')):
            logger.warning('Fail to produce pythia output')
            return
        
        # Update the banner with the pythia card
        banner = open(pjoin(self.main,'Events','banner.txt'),'a')
        banner.writelines('<MGPythiaCard>')
        banner.writelines(open(pjoin(self.main, 'Cards','pythia_card.dat')).read())
        banner.writelines('</MGPythiaCard>')
        banner.close()
        
        # Creating LHE file
        if is_executable(pjoin(self.pydir, 'hep2lhe')):
            open(self.status,'w').writelines('Creating Pythia LHE File')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            subprocess.call([self.dirbin+'/run_hep2lhe', self.pydir, str(self.cluster_mode)],
                            cwd=pjoin(self.main,'Events'))       

        # Creating ROOT file
        if is_executable(pjoin(self.eradir, 'ExRootLHEFConverter')):
            open(self.status,'w').writelines('Creating Pythia LHE Root File')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            subprocess.call([self.erabin+'/ExRootLHEFConverter', 
                             'pythia_events.lhe', 'pythia_lhe_events.root'],
                            cwd=pjoin(self.main,'Events')) 

        if int(run_data['ickkw']):
            logger.info('Create matching plots for Pythia')
            subprocess.call([self.dirbin+'/create_matching_plot.sh', self.name],
                            cwd=pjoin(self.main,'Events')) 

        # Plot for pythia
        if is_executable(pjoin(self.madir, 'plot_events')) and self.td:
            logger.info('Creating Plots for Pythia')
            open(self.status,'w').writelines('Creating Plots for Pythia')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            plot_dir = pjoin(self.main,'Events',self.name+'_pythia') 
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            open(pjoin(plot_dir, 'events.list'),'w').writelines('../pythia_events.lhe\n')
            subprocess.call([self.dirbin+'/plot', self.madir, self.td],
                            stdout = open(pjoin(plot_dir,'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir) 
            
            subprocess.call([self.dirbin+'/plot_page-pl', self.name+'_pythia', 'Pythia'],                            
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.main,'Events')) 
            shutil.move(pjoin(self.main, 'Events', 'plots.html'), 
                    pjoin(self.main, 'Events', self.name+'_plots_pythia.html'))
            
    ############################################################################
    def run_pgs(self):
        """ Run pgs and make associate root file/plot"""
        
        # Compile pgs if not there        
        if not is_executable(pjoin(self.pgsdir, 'pgs')):
            logger.info('No PGS executable -- running make')
            subprocess.call(['make'], cwd=self.pgsdir) 

        if not is_executable(pjoin(self.pgsdir, 'pgs')):
            logger.error('Fail to compile PGS')
            return
        open(self.status,'w').writelines('Running PGS')
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        # now pass the event to a detector simulator and reconstruct objects
        subprocess.call([self.dirbin+'/run_pgs', self.pgsdir, str(self.cluster_mode)],
                            cwd=pjoin(self.main,'Events')) 

        if not os.path.exists(pjoin(self.main, 'Events', 'pgs_events.lhco')):
            logger.error('Fail to create LHCO events')
            return 
        
        # Creating Root file
        if is_executable(pjoin(self.eradir, 'ExRootLHCOlympicsConverter')):
            open(self.status,'w').writelines('Creating PGS Root File')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            subprocess.call([self.erabin+'/ExRootLHCOlympicsConverter', 
                             'pgs_events.lhco','pgs_events.root'],
                            cwd=pjoin(self.main,'Events')) 

        
        # Creating plots
        if is_executable(pjoin(self.madir, 'plot_events')) and self.td:
            logger.info("Creating Plots for PGS")
            open(self.status,'w').writelines('Creating Plots for PGS')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            plot_dir = pjoin(self.main,'Events',self.name+'_pgs') 
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            open(pjoin(plot_dir, 'events.list'),'w').writelines('../pgs_events.lhco\n')
            subprocess.call([self.dirbin+'/plot', self.madir, self.td],
                            stdout = open(pjoin(plot_dir,'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir) 
            subprocess.call([self.dirbin+'/plot_page-pl', self.name+'_pgs', 'PGS'],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.main,'Events')) 
            shutil.move(pjoin(self.main, 'Events', 'plots.html'), 
                    pjoin(self.main, 'Events', self.name+'_plots_pgs.html'))

    ############################################################################
    def run_delphes(self):
        """ run dlephes and make associate root file/plot """
        
        open(self.status,'w').writelines('Running Delphes')
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        subprocess.call([self.dirbin+'/run_delphes', self.pgsdir, str(self.cluster_mode)],
                            cwd=self.main) 
        
        if not os.path.exists(pjoin(self.main, 'Events', 'delphes_events.lhco')):
            logger.error('Fail to create LHCO events from DELPHES')
            return 
        
        # Creating plots
        if is_executable(pjoin(self.madir, 'plot_events')) and self.td:
            logger.info("Creating Plots for Delphes")
            open(self.status,'w').writelines('Creating Plots for Delphes')
            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
            plot_dir = pjoin(self.main,'Events',self.name+'_delphes') 
            os.mkdirs(plot_dir)
            open(pjoin(plot_dir, 'events.list'),'w').writelines('../delphes_events.lhco\n')
            subprocess.call([self.dirbin+'/plot', self.madir, self.td],
                            stdout = open(pjoin(plot_dir,'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir) 
            subprocess.call([self.dirbin+'/plot_page-pl', self.name+'_delphes', 'Delphes'],
                            cwd=pjoin(self.main,'Events')) 
            shutil.move(pjoin(self.main, 'Events', 'plots.html'), 
                    pjoin(self.main, 'Events', 'plots_delphes.html'))
            
    ############################################################################
    ##  HELPING ROUTINE
    ############################################################################
    def read_run_card(self, run_card):
        """ """
        output={}
        for line in file(run_card,'r'):
            line = line.split('#')[0]
            line = line.split('!')[0]
            line = line.split('=')
            if len(line) != 2:
                continue
            output[line[1].strip()] = line[0].strip()
        return output
    
    ############################################################################
    def check_nb_events(self,path, data=None):
        """Find the number of event in the run_card, and check that this is not 
        too large"""

        if not data:
            data = self.read_run_card(path)
        
        nb_event = int(data['nevents'])
        if nb_event > 100000:
            logger.warning("Attempting to generate more than 100K events")
            logger.warning("Limiting number to 100K. Use multi_run for larger statistics.")
            os.system(r"""perl -p -i.bak -e "s/\d+\s*=\s*nevents/100000 = nevents/" %s""" \
                                                                         % path)
            data['nevents'] = 100000
        
        return data['nevents']
       
    ############################################################################
    def find_model_name(self):
        """ return the model name """
        if hasattr(self, 'model_name'):
            return self.model_name
        
        model = 'sm'
        for line in file(os.path.join(self.main,'Cards','proc_card_mg5.dat'),'r'):
            line = line.split('#')[0]
            line = line.split('=')[0]
            if line.startswith('import') and 'model' in line:
                model = line.split()[2]       
       
        self.model = model
        return model
    
    
    ############################################################################
    def find_madevent_mode(self):
        """Find if Madevent is in Group mode or not"""
        
        # The strategy is too look in the files Source/run_configs.inc
        # if we found: ChanPerJob=3 then it's a group mode.
        
        file_path = pjoin(self.main, 'Source', 'run_config.inc')
        text = open(file_path).read()
        if re.search(r'''s*parameter\s+\(ChanPerJob=2\)''', text, re.I+re.M):
            return 'group'
        else:
            return 'v4'
       
    ############################################################################
    def treat_ckkw_matching(self, run_data):
        """check for ckkw"""
        
        lpp1 = run_data['lpp1']
        lpp2 = run_data['lpp2']
        e1 = run_data['ebeam1']
        e2 = run_data['ebeam2']
        pd = run_data['pdlabel']
        lha = run_data['lhaid']
        xq = run_data['xqcut']
        translation = {'e1': e1, 'e2':e2, 'pd':pd, 
                       'lha':lha, 'xq':xq}

        if lpp1 or lpp2:
            # Remove ':s from pd          
            if pd.startswith("'"):
                pd = pd[1:]
            if pd.endswith("'"):
                pd = pd[:-1]                

            if xq >2 or xq ==2:
                xq = 2
            
            # find data file
            if pd == "lhapdf":
                issudfile = 'lib/issudgrid-%(e1)s-%(e2)s-%(pd)s-%(lha)s-%(xq)s.dat.gz'
            else:
                issudfile = 'lib/issudgrid-%(e1)s-%(e2)s-%(pd)s-%(xq)s.dat.gz'
            if self.web:
                issudfile = pjoin(self.webbin, issudfile % translation)
            else:
                issudfile = pjoin(self.main, issudfile % translation)
            
            logger.info('Sudakov grid file: %s' % issudfile)
            
            # check that filepath exist
            if os.path.exists(issudfile):
                path = pjoin(self.main, 'lib', 'issudgrid.dat')
                try:
                    os.remove(path)
                except:
                    pass
                os.system('gunzip -c %s > %s' % (issudfile, path))
            else:
                error_msg = 'No sudakov grid file for parameter choice. Please generate a sudakov grid file and restart.'
                logger.error(error_msg)
                os.system("echo %s > %s" % (error_msg, self.error))
                shutil.copy(self.error, self.status)
                os.remove(pjoin(self.main, 'RunWeb'))
                os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
                os.system('%s/gen_cardhtml-pl' % self.dirbin)
 

    def close_on_error(self):
        """Close nicely the run"""
        
        shutil.move(self.error, self.status)
        try:
            os.remove(pjoin(self.main,'refine'))
            os.remove(pjoin(self.main,'refine2'))
        except:
            pass
        os.remove(pjoin(self.main, 'RunWeb'))
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
        os.system('%s/gen_cardhtml-pl' % self.dirbin)
         
 
def is_executable(path):
    """ check if a path is executable"""
    try: 
        return os.access(path, os.X_OK)
    except:
        return False
        
################################################################################  
##   EXECUTABLE
################################################################################                                
if '__main__' == __name__:
    
    # Check that python version is valid
    if not (sys.version_info[0] == 2 or sys.version_info[1] > 4):
        sys.exit('MadEvent works with python 2.5 or higher (but not python 3.X).\n\
               Please upgrate your version of python.')
    
    # MadEvent is sensitive to the initial directory.
    # So go to the main directory
    os.chdir(root_path)
 
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madevent').setLevel(logging.INFO)
 
    argument = sys.argv
    try:
        mode = int(argument[1])
    except:
        mode = int(raw_input('Enter 2 for multi-core, 1 for parallel, 0 for serial run\n'))
    if mode == 0:
        try:
            name = argument[2]
        except:
            name = raw_input('Enter run name\n')
        ME = MadEventLauncher(cluster_mode=0,name=name)
    else:
        try:
            opt = argument[2]
        except:
            if mode == 1: 
                opt = raw_input('Enter name for jobs on pbs queue\n')
            else:
                opt = int(raw_input('Enter number of cores\n'))
        
        try:
            name = argument[3]
        except:
            name = raw_input('enter run name\n')

        if mode == 1:
            ME = MadEventLauncher(1, name=name, cluster_queue=opt)
        else:
            ME =  MadEventLauncher(2, name=name, nb_core=opt)
    
    # reconfigure path for the web 
    if len(argument) == 5:
        ME.pass_in_web_mode()
    
    try:
        ME.launch()
    except MERunError:
        ME.close_on_error()
             
        

        
    
    
    
    
    
    
    
