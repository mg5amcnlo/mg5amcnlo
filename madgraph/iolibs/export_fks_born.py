################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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
"""Methods and classes to export matrix elements to fks format."""

import fractions
import glob
import logging
import os
import re
import shutil
import subprocess
import string

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.fks.fks_born as fks
import madgraph.iolibs.files as files
import madgraph.iolibs.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4

import aloha.create_aloha as create_aloha

import models.sm.write_param_card as write_param_card
from madgraph import MadGraph5Error, MG5DIR
from madgraph.iolibs.files import cp, ln, mv
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_fks')

class ProcessExporterFortranFKS(export_v4.ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""
    
    def __init__(self, mgme_dir = "", dir_path = "", clean = False):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = mgme_dir
        self.dir_path = dir_path
        self.clean = clean
#===============================================================================
# copy the Template in a new directory.
#===============================================================================
    def copy_fkstemplate(self):
        """create the directory run_name as a copy of the MadEvent
        Template, and clean the directory
        For now it is just the same as copy_v4template, but it will be modified
        """
        mgme_dir = self.mgme_dir
        dir_path = self.dir_path
        clean =self.clean
        
        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(dir_path):
            if not mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(dir_path))
            print os.path.join(mgme_dir, 'TemplateFKS')
            shutil.copytree(os.path.join(mgme_dir, 'TemplateFKS'), dir_path, True)
        elif not os.path.isfile(os.path.join(dir_path, 'TemplateVersion.txt')):
            if not mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
        try:
            shutil.copy(os.path.join(mgme_dir, 'MGMEVersion.txt'), dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])
        
        #Ensure that the Template is clean
        if clean:
            logger.info('remove old information in %s' % os.path.basename(dir_path))
            if os.environ.has_key('MADGRAPH_BASE'):
                subprocess.call([os.path.join('bin', 'clean_template'), '--web'], \
                                                                       cwd=dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'clean_template')], \
                                                                       cwd=dir_path)
                except Exception, why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(dir_path),why))
            
            #Write version info
            MG_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])
            
    #===============================================================================
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===============================================================================
    def write_procdef_mg5(self, file_pos, modelname, process_str):
        """ write an equivalent of the MG4 proc_card in order that all the Madevent
        Perl script of MadEvent4 are still working properly for pure MG5 run."""
        
        proc_card_template = template_files.mg4_proc_card.mg4_template
        process_template = template_files.mg4_proc_card.process_template
        process_text = ''
        coupling = ''
        new_process_content = []
        
        
        # First find the coupling and suppress the coupling from process_str
        #But first ensure that coupling are define whithout spaces:
        process_str = process_str.replace(' =', '=')
        process_str = process_str.replace('= ', '=')
        process_str = process_str.replace(',',' , ')
        #now loop on the element and treat all the coupling
        for info in process_str.split():
            if '=' in info:
                coupling += info + '\n'
            else:
                new_process_content.append(info)
        # Recombine the process_str (which is the input process_str without coupling
        #info)
        process_str = ' '.join(new_process_content)
        
        #format the SubProcess
        process_text += process_template.substitute({'process': process_str, \
                                                            'coupling': coupling})
        
        text = proc_card_template.substitute({'process': process_text,
                                            'model': modelname,
                                            'multiparticle':''})
        ff = open(file_pos, 'w')
        ff.write(text)
        ff.close()
        
        
    #===============================================================================
    # write fks.inc file in SubProcesses, to be copied in the various FKS dirs    
    #===============================================================================
    def fks_inc_content(self,fks_proc):
        """composes the content of the fks_inc file"""
        
        replace_dict = {'fksconfigs': fks_proc.ndirs}
        
        file = "\
          integer fks_configs, ipos, jpos \n \
          data fks_configs /  %(fksconfigs)d  / \n \
          integer fks_i( %(fksconfigs)d ), fks_j( %(fksconfigs)d ) \n\
    \n \
          integer fks_ipos(0:nexternal) \n\
          integer fks_j_from_i(nexternal, 0:nexternal) \n\
          integer particle_type(nexternal), PDG_type(nexternal) \n\
          """ % replace_dict +\
          fks_proc.fks_config_string
    
        file += "\n \n \
          data (fks_ipos(ipos),ipos=0,  %d  ) " % len(fks_proc.fks_ipos)
        file +="\
      / %d,  " % len(fks_proc.fks_ipos) + \
         ', '.join([ "%d" % pdg for pdg in fks_proc.fks_ipos]) + " / "
        
        file += "\n\n"
        
        for i in fks_proc.fks_j_from_i.keys():
            file += " data (fks_j_from_i( %d, jpos), jpos=0, %d) " %\
          (i, len(fks_proc.fks_j_from_i[i]) )
            file +="\
      / %d,  " % len(fks_proc.fks_j_from_i[i]) + \
         ', '.join([ "%d" % ii for ii in fks_proc.fks_j_from_i[i]]) + " / \n"
          
        file += "\n\
C \n\
C  Particle type: \n\
C   octet = 8, triplet = 3, singlet = 1 \n\
          data (particle_type(ipos), ipos=1, nexternal) \
      / " + ', '.join([ "%d" % col for col in fks_proc.colors]) + " / "
    
        file += "\n \n\
C \n\
C  Particle type according to PDG: \n\
C   \n\
          data (PDG_type(ipos), ipos=1, nexternal) \
      / " + ', '.join([ "%d" % pdg for pdg in fks_proc.pdg_codes]) + " / "
    
        return file
    
        
        
        
    #===============================================================================
    # generate_subprocess_directory_fks
    #===============================================================================
    def generate_subprocess_directories_fks(self, matrix_element,
                                                  fortran_model,
                                                  me_number,
                                                  path=os.getcwd()):
        """Generate the Pxxxxx_i directories for a subprocess in MadFKS,
        including the necessary matrix.f and various helper files"""
        print "GENERATING FKS SUBDIRS"
        proc = matrix_element['processes'][0]
        print proc.input_string()
        
    
        cwd = os.getcwd()
        try:
            os.chdir(path)
        except OSError, error:
            error_msg = "The directory %s should exist in order to be able " % path + \
                        "to \"export\" in it. If you see this error message by " + \
                        "typing the command \"export\" please consider to use " + \
                        "instead the command \"output\". "
            raise MadGraph5Error, error_msg 
        
        this_fks_process = fks.FKSProcess(proc)
        
        calls = 0
        
    #write fks.inc in SubProcesses, it willbe copied later in the subdirs
        fks_inc = self.fks_inc_content(this_fks_process)
        
        for nfks, fksdir in enumerate(this_fks_process.fks_dirs):
                calls += self.generate_subprocess_directory_fks(nfks, fksdir, fks_inc,
                                                  matrix_element,
                                                  fortran_model,
                                                  me_number,
                                                  path)
            
        os.chdir(cwd)
        return calls
    
            
 