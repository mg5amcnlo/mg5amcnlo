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
"""Methods and classes to export matrix elements to v4 format."""

import copy
from cStringIO import StringIO
from distutils import dir_util
import itertools
import fractions
import glob
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import traceback

import aloha

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod
import madgraph.various.process_checks as process_checks
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import aloha.create_aloha as create_aloha
import models.import_ufo as import_ufo
import models.write_param_card as param_writer
import models.check_param_card as check_param_card


from madgraph import MadGraph5Error, MG5DIR, ReadWrite
from madgraph.iolibs.files import cp, ln, mv

from madgraph import InvalidCmd

pjoin = os.path.join

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_v4')

default_compiler= {'fortran': 'gfortran',
                       'f2py': 'f2py',
                       'cpp':'g++'}


class VirtualExporter(object):
    
    #exporter variable who modified the way madgraph interacts with this class
    
    grouped_mode = 'madevent'  
    # This variable changes the type of object called within 'generate_subprocess_directory'
    #functions. 
    # False to avoid grouping (only identical matrix element are merged)
    # 'madevent' group the massless quark and massless lepton
    # 'madweight' group the gluon with the massless quark
    sa_symmetry = False
    # If no grouped_mode=False, uu~ and u~u will be called independently. 
    #Putting sa_symmetry generates only one of the two matrix-element.
    check = True
    # Ask madgraph to check if the directory already exists and propose to the user to 
    #remove it first if this is the case
    output = 'Template'
    # [Template, None, dir]
    #    - Template, madgraph will call copy_template
    #    - dir, madgraph will just create an empty directory for initialisation
    #    - None, madgraph do nothing for initialisation
    exporter = 'v4'
    # language of the output 'v4' for Fortran output
    #                        'cpp' for C++ output 
    
    
    def __init__(self, dir_path = "", opt=None):
        return

    def copy_template(self, model):
        return

    def generate_subprocess_directory(self, subproc_group, helicity_model, me=None):
    #    generate_subprocess_directory(self, matrix_element, helicity_model, me_number) [for ungrouped]
        return 0 # return an integer stating the number of call to helicity routine
    
    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[]):
        return
    
    def finalize(self,matrix_element, cmdhistory, MG5options, outputflag):
        return
    
    
    def pass_information_from_cmd(self, cmd):
        """pass information from the command interface to the exporter.
           Please do not modify any object of the interface from the exporter.
        """
        return
    
    def modify_grouping(self, matrix_element):
        return False, matrix_element
           
    def export_model_files(self, model_v4_path):
        raise Exception, "V4 model not supported by this type of exporter. Please use UFO model"
        return
    
    def export_helas(self, HELAS_PATH):
        raise Exception, "V4 model not supported by this type of exporter. Please use UFO model"
        return

#===============================================================================
# ProcessExporterFortran
#===============================================================================
class ProcessExporterFortran(VirtualExporter):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True
                        }
    grouped_mode = False

    def __init__(self,  dir_path = "", opt=None):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = MG5DIR
        self.dir_path = dir_path
        self.model = None

        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)
        
        #place holder to pass information to the run_interface
        self.proc_characteristic = banner_mod.ProcCharacteristic()
        
        
    #===========================================================================
    # process exporter fortran switch between group and not grouped
    #===========================================================================
    def export_processes(self, matrix_elements, fortran_model):
        """Make the switch between grouped and not grouped output"""
        
        calls = 0
        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
            for (group_number, me_group) in enumerate(matrix_elements):
                calls = calls + self.generate_subprocess_directory(\
                                          me_group, fortran_model, group_number)
        else:
            for me_number, me in enumerate(matrix_elements.get_matrix_elements()):
                calls = calls + self.generate_subprocess_directory(\
                                                   me, fortran_model, me_number)    
                        
        return calls    
        

    #===========================================================================
    #  create the run_card 
    #===========================================================================
    def create_run_card(self, matrix_elements, history):
        """ """


        # bypass this for the loop-check
        import madgraph.loop.loop_helas_objects as loop_helas_objects
        if isinstance(matrix_elements, loop_helas_objects.LoopHelasMatrixElement):
            matrix_elements = None

        run_card = banner_mod.RunCard()
        
        
        default=True
        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):            
            processes = [me.get('processes')  for megroup in matrix_elements 
                                        for me in megroup['matrix_elements']]
        elif matrix_elements:
            processes = [me.get('processes') 
                                 for me in matrix_elements['matrix_elements']]
        else:
            default =False
    
        if default:
            run_card.create_default_for_process(self.proc_characteristic, 
                                            history,
                                            processes)
          
    
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'))
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card.dat'))
        
        
    #===========================================================================
    # copy the Template in a new directory.
    #===========================================================================
    def copy_template(self, model):
        """create the directory run_name as a copy of the MadEvent
        Template, and clean the directory
        """

        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(self.dir_path):
            assert self.mgme_dir, \
                     "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(self.dir_path))
            shutil.copytree(pjoin(self.mgme_dir, 'Template/LO'),
                            self.dir_path, True)
            # distutils.dir_util.copy_tree since dir_path already exists
            dir_util.copy_tree(pjoin(self.mgme_dir, 'Template/Common'), 
                               self.dir_path)
            # copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards',card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to copy " + card + ".dat to default")
        elif os.getcwd() == os.path.realpath(self.dir_path):
            logger.info('working in local directory: %s' % \
                                                os.path.realpath(self.dir_path))
            # distutils.dir_util.copy_tree since dir_path already exists
            dir_util.copy_tree(pjoin(self.mgme_dir, 'Template/LO'), 
                               self.dir_path)
#            for name in misc.glob('Template/LO/*', self.mgme_dir):
#                name = os.path.basename(name)
#                filname = pjoin(self.mgme_dir, 'Template','LO',name)
#                if os.path.isfile(filename):
#                    files.cp(filename, pjoin(self.dir_path,name))
#                elif os.path.isdir(filename):
#                     shutil.copytree(filename, pjoin(self.dir_path,name), True)
            # distutils.dir_util.copy_tree since dir_path already exists
            dir_util.copy_tree(pjoin(self.mgme_dir, 'Template/Common'), 
                               self.dir_path)
            # Copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards', card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to copy " + card + ".dat to default")            
        elif not os.path.isfile(pjoin(self.dir_path, 'TemplateVersion.txt')):
            assert self.mgme_dir, \
                      "No valid MG_ME path given for MG4 run directory creation."
        try:
            shutil.copy(pjoin(self.mgme_dir, 'MGMEVersion.txt'), self.dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'MGMEVersion.txt'), 'w').write(MG5_version['version'])

        #Ensure that the Template is clean
        if self.opt['clean']:
            logger.info('remove old information in %s' % \
                                                  os.path.basename(self.dir_path))
            if os.environ.has_key('MADGRAPH_BASE'):
                misc.call([pjoin('bin', 'internal', 'clean_template'),
                                 '--web'], cwd=self.dir_path)
            else:
                try:
                    misc.call([pjoin('bin', 'internal', 'clean_template')], \
                                                                       cwd=self.dir_path)
                except Exception, why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(self.dir_path),why))

            #Write version info
            MG_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])

        # add the makefile in Source directory 
        filename = pjoin(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FileWriter(filename))
        
        # add the DiscreteSampler information
        files.cp(pjoin(MG5DIR,'vendor', 'DiscreteSampler', 'DiscreteSampler.f'), 
                 pjoin(self.dir_path, 'Source'))
        files.cp(pjoin(MG5DIR,'vendor', 'DiscreteSampler', 'StringCast.f'), 
                 pjoin(self.dir_path, 'Source'))
        
        # We need to create the correct open_data for the pdf
        self.write_pdf_opendata()
        
        
    #===========================================================================
    # Call MadAnalysis5 to generate the default cards for this process
    #=========================================================================== 
    def create_default_madanalysis5_cards(self, history, proc_defs, processes,
                            ma5_path, output_dir, levels = ['parton','hadron']):
        """ Call MA5 so that it writes default cards for both parton and
        post-shower levels, tailored for this particular process."""
        
        if len(levels)==0:
            return
        
        logger.info('Generating MadAnalysis5 default cards tailored to this process')
        try:
            MA5_interpreter = common_run_interface.CommonRunCmd.\
                          get_MadAnalysis5_interpreter(MG5DIR,ma5_path,loglevel=100)
        except (Exception, SystemExit) as e:
            logger.warning('Fail to create a MadAnalysis5 instance. Therefore the default analysis with MadAnalysis5 will be empty.')
            return
        if MA5_interpreter is None:
            return

        MA5_main = MA5_interpreter.main
       
        for lvl in ['parton','hadron']:
            if lvl in levels:
                card_to_generate = pjoin(output_dir,'madanalysis5_%s_card_default.dat'%lvl)
                try:
                    text = MA5_main.madgraph.generate_card(history, proc_defs, processes,lvl)
                except (Exception, SystemExit) as e:
                    # keep the default card (skip only)
                    logger.warning('MadAnalysis5 failed to write a %s-level'%lvl+
                                                  ' default analysis card for this process.')
                    logger.warning('Therefore, %s-level default analysis with MadAnalysis5 will be empty.'%lvl)
                    error=StringIO()
                    traceback.print_exc(file=error)
                    logger.debug('MadAnalysis5 error was:')
                    logger.debug('-'*60)
                    logger.debug(error.getvalue()[:-1])
                    logger.debug('-'*60)
                else:
                    open(card_to_generate,'w').write(text)

    #===========================================================================
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===========================================================================
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
        replace_dict = {'process': process_str,
                        'coupling': coupling}
        process_text += process_template.substitute(replace_dict)
        
        replace_dict = {'process': process_text,
                                            'model': modelname,
                                            'multiparticle':''}
        text = proc_card_template.substitute(replace_dict)
        
        if file_pos:
            ff = open(file_pos, 'w')
            ff.write(text)
            ff.close()
        else:
            return replace_dict


    def pass_information_from_cmd(self, cmd):
        """Pass information for MA5"""
        
        self.proc_defs = cmd._curr_proc_defs

    #===========================================================================
    # Create jpeg diagrams, html pages,proc_card_mg5.dat and madevent.tar.gz
    #===========================================================================
    def finalize(self, matrix_elements, history='', mg5options={}, flaglist=[]):
        """Function to finalize v4 directory, for inheritance.""" 
        
        self.create_run_card(matrix_elements, history)
        self.create_MA5_cards(matrix_elements, history)
    
    def create_MA5_cards(self,matrix_elements,history):
        """ A wrapper around the creation of the MA5 cards so that it can be 
        bypassed by daughter classes (i.e. in standalone)."""
        if 'madanalysis5_path' in self.opt and not \
                self.opt['madanalysis5_path'] is None and not self.proc_defs is None:
            processes = None
            if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):            
                processes = [me.get('processes')  for megroup in matrix_elements 
                                        for me in megroup['matrix_elements']]
            elif matrix_elements:
                processes = [me.get('processes') 
                                 for me in matrix_elements['matrix_elements']]
            
            self.create_default_madanalysis5_cards(
                history, self.proc_defs, processes,
                self.opt['madanalysis5_path'], pjoin(self.dir_path,'Cards'),
                levels = ['hadron','parton'])
            
            for level in ['hadron','parton']:
                # Copying these cards turn on the use of MadAnalysis5 by default.
                if os.path.isfile(pjoin(self.dir_path,'Cards','madanalysis5_%s_card_default.dat'%level)):
                    shutil.copy(pjoin(self.dir_path,'Cards','madanalysis5_%s_card_default.dat'%level),
                                pjoin(self.dir_path,'Cards','madanalysis5_%s_card.dat'%level))

    #===========================================================================
    # Create the proc_characteristic file passing information to the run_interface
    #===========================================================================
    def create_proc_charac(self, matrix_elements=None, history="", **opts):
        
        self.proc_characteristic.write(pjoin(self.dir_path, 'SubProcesses', 'proc_characteristics'))

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self):
        """Function to write a matrix.f file, for inheritance.
        """
        pass

    #===========================================================================
    # write_pdf_opendata
    #===========================================================================
    def write_pdf_opendata(self):
        """ modify the pdf opendata file, to allow direct access to cluster node
        repository if configure"""
        
        if not self.opt["cluster_local_path"]:
            changer = {"pdf_systemwide": ""}
        else: 
            to_add = """
            tempname='%(path)s'//Tablefile
            open(IU,file=tempname,status='old',ERR=1)
            return
 1          tempname='%(path)s/Pdfdata/'//Tablefile
            open(IU,file=tempname,status='old',ERR=2)
            return
 2          tempname='%(path)s/lhapdf'//Tablefile
            open(IU,file=tempname,status='old',ERR=3)
            return            
 3          tempname='%(path)s/../lhapdf/pdfsets/'//Tablefile
            open(IU,file=tempname,status='old',ERR=4)
            return              
 4          tempname='%(path)s/../lhapdf/pdfsets/6.1/'//Tablefile
            open(IU,file=tempname,status='old',ERR=5)
            return  
            """ % {"path" : self.opt["cluster_local_path"]}
            
            changer = {"pdf_systemwide": to_add}


        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "opendata.f"))        
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_opendata.f"),"r").read()
        ff.writelines(template % changer)

        # Do the same for lhapdf set
        if not self.opt["cluster_local_path"]:
            changer = {"cluster_specific_path": ""}
        else:
            to_add="""
         LHAPath='%(path)s/PDFsets'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return        
         LHAPath='%(path)s/../lhapdf/pdfsets/6.1/'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return
         LHAPath='%(path)s/../lhapdf/pdfsets/'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return  
         LHAPath='./PDFsets'            
         """ % {"path" : self.opt["cluster_local_path"]}
            changer = {"cluster_specific_path": to_add}

        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"))        
        #ff = open(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"),"w")
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_wrap_lhapdf.f"),"r").read()
        ff.writelines(template % changer)
        
        
        return



    #===========================================================================
    # write_maxparticles_file
    #===========================================================================
    def write_maxparticles_file(self, writer, matrix_elements):
        """Write the maxparticles.inc file for MadEvent"""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            maxparticles = max([me.get_nexternal_ninitial()[0] for me in \
                              matrix_elements.get('matrix_elements')])
        else:
            maxparticles = max([me.get_nexternal_ninitial()[0] \
                              for me in matrix_elements])

        lines = "integer max_particles\n"
        lines += "parameter(max_particles=%d)" % maxparticles

        # Write the file
        writer.writelines(lines)

        return True

    
    #===========================================================================
    # export the model
    #===========================================================================
    def export_model_files(self, model_path):
        """Configure the files/link of the process according to the model"""

        # Import the model
        for file in os.listdir(model_path):
            if os.path.isfile(pjoin(model_path, file)):
                shutil.copy2(pjoin(model_path, file), \
                                     pjoin(self.dir_path, 'Source', 'MODEL'))


    def make_model_symbolic_link(self):
        """Make the copy/symbolic links"""
        model_path = self.dir_path + '/Source/MODEL/'
        if os.path.exists(pjoin(model_path, 'ident_card.dat')):
            mv(model_path + '/ident_card.dat', self.dir_path + '/Cards')
        if os.path.exists(pjoin(model_path, 'particles.dat')):
            ln(model_path + '/particles.dat', self.dir_path + '/SubProcesses')
            ln(model_path + '/interactions.dat', self.dir_path + '/SubProcesses')
        cp(model_path + '/param_card.dat', self.dir_path + '/Cards')
        mv(model_path + '/param_card.dat', self.dir_path + '/Cards/param_card_default.dat')
        ln(model_path + '/coupl.inc', self.dir_path + '/Source')
        ln(model_path + '/coupl.inc', self.dir_path + '/SubProcesses')
        self.make_source_links()
        
    def make_source_links(self):
        """ Create the links from the files in sources """

        ln(self.dir_path + '/Source/run.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxparticles.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/run_config.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/lhe_event_infos.inc', self.dir_path + '/SubProcesses', log=False)
        

    #===========================================================================
    # export the helas routine
    #===========================================================================
    def export_helas(self, helas_path):
        """Configure the files/link of the process according to the model"""

        # Import helas routine
        for filename in os.listdir(helas_path):
            filepos = pjoin(helas_path, filename)
            if os.path.isfile(filepos):
                if filepos.endswith('Makefile.template'):
                    cp(filepos, self.dir_path + '/Source/DHELAS/Makefile')
                elif filepos.endswith('Makefile'):
                    pass
                else:
                    cp(filepos, self.dir_path + '/Source/DHELAS')
    # following lines do the same but whithout symbolic link
    # 
    #def export_helas(mgme_dir, dir_path):
    #
    #        # Copy the HELAS directory
    #        helas_dir = pjoin(mgme_dir, 'HELAS')
    #        for filename in os.listdir(helas_dir): 
    #            if os.path.isfile(pjoin(helas_dir, filename)):
    #                shutil.copy2(pjoin(helas_dir, filename),
    #                            pjoin(dir_path, 'Source', 'DHELAS'))
    #        shutil.move(pjoin(dir_path, 'Source', 'DHELAS', 'Makefile.template'),
    #                    pjoin(dir_path, 'Source', 'DHELAS', 'Makefile'))
    #  

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,
                                         me_number):
        """Routine to generate a subprocess directory (for inheritance)"""

        pass

    #===========================================================================
    # get_source_libraries_list
    #===========================================================================
    def get_source_libraries_list(self):
        """ Returns the list of libraries to be compiling when compiling the
        SOURCE directory. It is different for loop_induced processes and 
        also depends on the value of the 'output_dependencies' option"""
        
        return ['$(LIBDIR)libdhelas.$(libext)',
                '$(LIBDIR)libpdf.$(libext)',
                '$(LIBDIR)libmodel.$(libext)',
                '$(LIBDIR)libcernlib.$(libext)',
                '$(LIBDIR)libbias.$(libext)']

    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = ' '.join(['$(LIBRARIES)']+self.get_source_libraries_list())
        if self.opt['model'] == 'mssm' or self.opt['model'].startswith('mssm-'):
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc\n\tcd MODEL; make
MODEL/MG5_param.dat: ../Cards/param_card.dat\n\t../bin/madevent treatcards param
param_card.inc: MODEL/MG5_param.dat\n\t../bin/madevent treatcards param\n'''
        else:
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc\n\tcd MODEL; make    
param_card.inc: ../Cards/param_card.dat\n\t../bin/madevent treatcards param\n'''
        text = open(path).read() % {'libraries': set_of_lib, 'model':model_line} 
        writer.write(text)
        
        return True

    #===========================================================================
    # write_nexternal_madspin
    #===========================================================================
    def write_nexternal_madspin(self, writer, nexternal, ninitial):
        """Write the nexternal_prod.inc file for madspin"""

        replace_dict = {}

        replace_dict['nexternal'] = nexternal
        replace_dict['ninitial'] = ninitial

        file = """ \
          integer    nexternal_prod
          parameter (nexternal_prod=%(nexternal)d)
          integer    nincoming_prod
          parameter (nincoming_prod=%(ninitial)d)""" % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict

    #===========================================================================
    # write_helamp_madspin
    #===========================================================================
    def write_helamp_madspin(self, writer, ncomb):
        """Write the helamp.inc file for madspin"""

        replace_dict = {}

        replace_dict['ncomb'] = ncomb

        file = """ \
          integer    ncomb1
          parameter (ncomb1=%(ncomb)d)
          double precision helamp(ncomb1)    
          common /to_helamp/helamp """ % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict



    #===========================================================================
    # write_nexternal_file
    #===========================================================================
    def write_nexternal_file(self, writer, nexternal, ninitial):
        """Write the nexternal.inc file for MG4"""

        replace_dict = {}

        replace_dict['nexternal'] = nexternal
        replace_dict['ninitial'] = ninitial

        file = """ \
          integer    nexternal
          parameter (nexternal=%(nexternal)d)
          integer    nincoming
          parameter (nincoming=%(ninitial)d)""" % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict
    #===========================================================================
    # write_pmass_file
    #===========================================================================
    def write_pmass_file(self, writer, matrix_element):
        """Write the pmass.inc file for MG4"""

        model = matrix_element.get('processes')[0].get('model')
        
        lines = []
        for wf in matrix_element.get_external_wavefunctions():
            mass = model.get('particle_dict')[wf.get('pdg_code')].get('mass')
            if mass.lower() != "zero":
                mass = "abs(%s)" % mass

            lines.append("pmass(%d)=%s" % \
                         (wf.get('number_external'), mass))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_ngraphs_file
    #===========================================================================
    def write_ngraphs_file(self, writer, nconfigs):
        """Write the ngraphs.inc file for MG4. Needs input from
        write_configs_file."""

        file = "       integer    n_max_cg\n"
        file = file + "parameter (n_max_cg=%d)" % nconfigs

        # Write the file
        writer.writelines(file)

        return True

    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, matrix_element):
        """Write the leshouche.inc file for MG4"""

        # Write the file
        writer.writelines(self.get_leshouche_lines(matrix_element, 0))

        return True

    #===========================================================================
    # get_leshouche_lines
    #===========================================================================
    def get_leshouche_lines(self, matrix_element, numproc):
        """Write the leshouche.inc file for MG4"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        lines = []
        for iproc, proc in enumerate(matrix_element.get('processes')):
            legs = proc.get_legs_with_decays()
            lines.append("DATA (IDUP(i,%d,%d),i=1,%d)/%s/" % \
                         (iproc + 1, numproc+1, nexternal,
                          ",".join([str(l.get('id')) for l in legs])))
            if iproc == 0 and numproc == 0:
                for i in [1, 2]:
                    lines.append("DATA (MOTHUP(%d,i),i=1,%2r)/%s/" % \
                             (i, nexternal,
                              ",".join([ "%3r" % 0 ] * ninitial + \
                                       [ "%3r" % i ] * (nexternal - ninitial))))

            # Here goes the color connections corresponding to the JAMPs
            # Only one output, for the first subproc!
            if iproc == 0:
                # If no color basis, just output trivial color flow
                if not matrix_element.get('color_basis'):
                    for i in [1, 2]:
                        lines.append("DATA (ICOLUP(%d,i,1,%d),i=1,%2r)/%s/" % \
                                 (i, numproc+1,nexternal,
                                  ",".join([ "%3r" % 0 ] * nexternal)))

                else:
                    # First build a color representation dictionnary
                    repr_dict = {}
                    for l in legs:
                        repr_dict[l.get('number')] = \
                            proc.get('model').get_particle(l.get('id')).get_color()\
                            * (-1)**(1+l.get('state'))
                    # Get the list of color flows
                    color_flow_list = \
                        matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                                   ninitial)
                    # And output them properly
                    for cf_i, color_flow_dict in enumerate(color_flow_list):
                        for i in [0, 1]:
                            lines.append("DATA (ICOLUP(%d,i,%d,%d),i=1,%2r)/%s/" % \
                                 (i + 1, cf_i + 1, numproc+1, nexternal,
                                  ",".join(["%3r" % color_flow_dict[l.get('number')][i] \
                                            for l in legs])))

        return lines




    #===========================================================================
    # write_maxamps_file
    #===========================================================================
    def write_maxamps_file(self, writer, maxamps, maxflows,
                           maxproc,maxsproc):
        """Write the maxamps.inc file for MG4."""

        file = "       integer    maxamps, maxflow, maxproc, maxsproc\n"
        file = file + "parameter (maxamps=%d, maxflow=%d)\n" % \
               (maxamps, maxflows)
        file = file + "parameter (maxproc=%d, maxsproc=%d)" % \
               (maxproc, maxsproc)

        # Write the file
        writer.writelines(file)

        return True


    #===========================================================================
    # Routines to output UFO models in MG4 format
    #===========================================================================

    def convert_model(self, model, wanted_lorentz = [],
                             wanted_couplings = []):
        """ Create a full valid MG4 model from a MG5 model (coming from UFO)"""

        # Make sure aloha is in quadruple precision if needed
        old_aloha_mp=aloha.mp_precision
        aloha.mp_precision=self.opt['mp']

        # create the MODEL
        write_dir=pjoin(self.dir_path, 'Source', 'MODEL')
        model_builder = UFO_model_to_mg4(model, write_dir, self.opt + self.proc_characteristic)
        model_builder.build(wanted_couplings)

        # Backup the loop mode, because it can be changed in what follows.
        old_loop_mode = aloha.loop_mode

        # Create the aloha model or use the existing one (for loop exporters
        # this is useful as the aloha model will be used again in the 
        # LoopHelasMatrixElements generated). We do not save the model generated
        # here if it didn't exist already because it would be a waste of
        # memory for tree level applications since aloha is only needed at the
        # time of creating the aloha fortran subroutines.
        if hasattr(self, 'aloha_model'):
            aloha_model = self.aloha_model
        else:
            aloha_model = create_aloha.AbstractALOHAModel(os.path.basename(model.get('modelpath')))
        aloha_model.add_Lorentz_object(model.get('lorentz'))

        # Compute the subroutines
        if wanted_lorentz:
            aloha_model.compute_subset(wanted_lorentz)
        else:
            aloha_model.compute_all(save=False)

        # Write them out
        write_dir=pjoin(self.dir_path, 'Source', 'DHELAS')
        aloha_model.write(write_dir, 'Fortran')

        # Revert the original aloha loop mode
        aloha.loop_mode = old_loop_mode

        #copy Helas Template
        cp(MG5DIR + '/aloha/template_files/Makefile_F', write_dir+'/makefile')
        if any([any(['L' in tag for tag in d[1]]) for d in wanted_lorentz]):
            cp(MG5DIR + '/aloha/template_files/aloha_functions_loop.f', 
                                                 write_dir+'/aloha_functions.f')
            aloha_model.loop_mode = False
        else:
            cp(MG5DIR + '/aloha/template_files/aloha_functions.f', 
                                                 write_dir+'/aloha_functions.f')
        create_aloha.write_aloha_file_inc(write_dir, '.f', '.o')

        # Make final link in the Process
        self.make_model_symbolic_link()
    
        # Re-establish original aloha mode
        aloha.mp_precision=old_aloha_mp
    

    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""

        return False, matrix_element
        
    #===========================================================================
    # Helper functions
    #===========================================================================
    def get_mg5_info_lines(self):
        """Return info lines for MG5, suitable to place at beginning of
        Fortran files"""

        info = misc.get_pkg_info()
        info_lines = ""
        if info and info.has_key('version') and  info.has_key('date'):
            info_lines = "#  Generated by MadGraph5_aMC@NLO v. %s, %s\n" % \
                         (info['version'], info['date'])
            info_lines = info_lines + \
                         "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"
        else:
            info_lines = "#  Generated by MadGraph5_aMC@NLO\n" + \
                         "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"        

        return info_lines

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "C " + process.nice_string().replace('\n', '\nC * ') \
                         for process in matrix_element.get('processes')])


    def get_helicity_lines(self, matrix_element,array_name='NHEL'):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line_list = []
        i = 0
        for helicities in matrix_element.get_helicity_matrix():
            i = i + 1
            int_list = [i, len(helicities)]
            int_list.extend(helicities)
            helicity_line_list.append(\
                ("DATA ("+array_name+"(I,%4r),I=1,%d) /" + \
                 ",".join(['%2r'] * len(helicities)) + "/") % tuple(int_list))

        return "\n".join(helicity_line_list)

    def get_ic_line(self, matrix_element):
        """Return the IC definition line coming after helicities, required by
        switchmom in madevent"""

        nexternal = matrix_element.get_nexternal_ninitial()[0]
        int_list = range(1, nexternal + 1)

        return "DATA (IC(I,1),I=1,%i) /%s/" % (nexternal,
                                                     ",".join([str(i) for \
                                                               i in int_list]))

    def set_chosen_SO_index(self, process, squared_orders):
        """ From the squared order constraints set by the user, this function
        finds what indices of the squared_orders list the user intends to pick.
        It returns this as a string of comma-separated successive '.true.' or 
        '.false.' for each index."""
        
        user_squared_orders = process.get('squared_orders')
        split_orders = process.get('split_orders')
        
        if len(user_squared_orders)==0:
            return ','.join(['.true.']*len(squared_orders))
        
        res = []
        for sqsos in squared_orders:
            is_a_match = True
            for user_sqso, value in user_squared_orders.items():
                if (process.get_squared_order_type(user_sqso) =='==' and \
                        value!=sqsos[split_orders.index(user_sqso)]) or \
                   (process.get_squared_order_type(user_sqso) in ['<=','='] and \
                                value<sqsos[split_orders.index(user_sqso)]) or \
                   (process.get_squared_order_type(user_sqso) == '>' and \
                                value>=sqsos[split_orders.index(user_sqso)]):
                    is_a_match = False
                    break
            res.append('.true.' if is_a_match else '.false.')
            
        return ','.join(res)

    def get_split_orders_lines(self, orders, array_name, n=5):
        """ Return the split orders definition as defined in the list orders and
        for the name of the array 'array_name'. Split rows in chunks of size n."""
        
        ret_list = []  
        for index, order in enumerate(orders):      
            for k in xrange(0, len(order), n):
                ret_list.append("DATA (%s(%3r,i),i=%3r,%3r) /%s/" % \
                  (array_name,index + 1, k + 1, min(k + n, len(order)),
                              ','.join(["%5r" % i for i in order[k:k + n]])))
        return ret_list
    
    def format_integer_list(self, list, name, n=5):
        """ Return an initialization of the python list in argument following 
        the fortran syntax using the data keyword assignment, filling an array 
        of name 'name'. It splits rows in chunks of size n."""
        
        ret_list = []
        for k in xrange(0, len(list), n):
            ret_list.append("DATA (%s(i),i=%3r,%3r) /%s/" % \
                  (name, k + 1, min(k + n, len(list)),
                                  ','.join(["%5r" % i for i in list[k:k + n]])))
        return ret_list

    def get_color_data_lines(self, matrix_element, n=6):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return ["DATA Denom(1)/1/", "DATA (CF(i,1),i=1,1) /1/"]
        else:
            ret_list = []
            my_cs = color.ColorString()
            for index, denominator in \
                enumerate(matrix_element.get('color_matrix').\
                                                 get_line_denominators()):
                # First write the common denominator for this color matrix line
                ret_list.append("DATA Denom(%i)/%i/" % (index + 1, denominator))
                # Then write the numerators for the matrix elements
                num_list = matrix_element.get('color_matrix').\
                                            get_line_numerators(index, denominator)

                for k in xrange(0, len(num_list), n):
                    ret_list.append("DATA (CF(i,%3r),i=%3r,%3r) /%s/" % \
                                    (index + 1, k + 1, min(k + n, len(num_list)),
                                     ','.join(["%5r" % i for i in num_list[k:k + n]])))
                my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[index])
                ret_list.append("C %s" % repr(my_cs))
            return ret_list


    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""

        return "DATA IDEN/%2r/" % \
               matrix_element.get_denominator_factor()

    def get_icolamp_lines(self, mapconfigs, matrix_element, num_matrix_element):
        """Return the ICOLAMP matrix, showing which JAMPs contribute to
        which configs (diagrams)."""

        ret_list = []

        booldict = {False: ".false.", True: ".true."}

        if not matrix_element.get('color_basis'):
            # No color, so only one color factor. Simply write a ".true." 
            # for each config (i.e., each diagram with only 3 particle
            # vertices
            configs = len(mapconfigs)
            ret_list.append("DATA(icolamp(1,i,%d),i=1,%d)/%s/" % \
                            (num_matrix_element, configs,
                             ','.join([".true." for i in range(configs)])))
            return ret_list

        # There is a color basis - create a list showing which JAMPs have
        # contributions to which configs

        # Only want to include leading color flows, so find max_Nc
        color_basis = matrix_element.get('color_basis')
        
        # We don't want to include the power of Nc's which come from the potential
        # loop color trace (i.e. in the case of a closed fermion loop for example)
        # so we subtract it here when computing max_Nc
        max_Nc = max(sum([[(v[4]-v[5]) for v in val] for val in 
                                                      color_basis.values()],[]))

        # Crate dictionary between diagram number and JAMP number
        diag_jamp = {}
        for ijamp, col_basis_elem in \
                enumerate(sorted(matrix_element.get('color_basis').keys())):
            for diag_tuple in matrix_element.get('color_basis')[col_basis_elem]:
                # Only use color flows with Nc == max_Nc. However, notice that
                # we don't want to include the Nc power coming from the loop
                # in this counting.
                if (diag_tuple[4]-diag_tuple[5]) == max_Nc:
                    diag_num = diag_tuple[0] + 1
                    # Add this JAMP number to this diag_num
                    diag_jamp[diag_num] = diag_jamp.setdefault(diag_num, []) + \
                                          [ijamp+1]

        colamps = ijamp + 1
        for iconfig, num_diag in enumerate(mapconfigs):        
            if num_diag == 0:
                continue

            # List of True or False 
            bool_list = [(i + 1 in diag_jamp[num_diag]) for i in range(colamps)]
            # Add line
            ret_list.append("DATA(icolamp(i,%d,%d),i=1,%d)/%s/" % \
                                (iconfig+1, num_matrix_element, colamps,
                                 ','.join(["%s" % booldict[b] for b in \
                                           bool_list])))

        return ret_list

    def get_amp2_lines(self, matrix_element, config_map = []):
        """Return the amp2(i) = sum(amp for diag(i))^2 lines"""

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        # Get minimum legs in a vertex
        vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
       matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        ret_lines = []
        if config_map:
            # In this case, we need to sum up all amplitudes that have
            # identical topologies, as given by the config_map (which
            # gives the topology/config for each of the diagrams
            diagrams = matrix_element.get('diagrams')
            # Combine the diagrams with identical topologies
            config_to_diag_dict = {}
            for idiag, diag in enumerate(matrix_element.get('diagrams')):
                if config_map[idiag] == 0:
                    continue
                try:
                    config_to_diag_dict[config_map[idiag]].append(idiag)
                except KeyError:
                    config_to_diag_dict[config_map[idiag]] = [idiag]
            # Write out the AMP2s summing squares of amplitudes belonging
            # to eiher the same diagram or different diagrams with
            # identical propagator properties.  Note that we need to use
            # AMP2 number corresponding to the first diagram number used
            # for that AMP2.
            for config in sorted(config_to_diag_dict.keys()):

                line = "AMP2(%(num)d)=AMP2(%(num)d)+" % \
                       {"num": (config_to_diag_dict[config][0] + 1)}

                amp = "+".join(["AMP(%(num)d)" % {"num": a.get('number')} for a in \
                                  sum([diagrams[idiag].get('amplitudes') for \
                                       idiag in config_to_diag_dict[config]], [])])
                
                # Not using \sum |M|^2 anymore since this creates troubles
                # when ckm is not diagonal due to the JIM mechanism.
                if '+' in amp:
                    line += "(%s)*dconjg(%s)" % (amp, amp)
                else:
                    line += "%s*dconjg(%s)" % (amp, amp)
                ret_lines.append(line)
        else:
            for idiag, diag in enumerate(matrix_element.get('diagrams')):
                # Ignore any diagrams with 4-particle vertices.
                if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                    continue
                # Now write out the expression for AMP2, meaning the sum of
                # squared amplitudes belonging to the same diagram
                line = "AMP2(%(num)d)=AMP2(%(num)d)+" % {"num": (idiag + 1)}
                line += "+".join(["AMP(%(num)d)*dconjg(AMP(%(num)d))" % \
                                  {"num": a.get('number')} for a in \
                                  diag.get('amplitudes')])
                ret_lines.append(line)

        return ret_lines

    #===========================================================================
    # Returns the data statements initializing the coeffictients for the JAMP
    # decomposition. It is used when the JAMP initialization is decided to be 
    # done through big arrays containing the projection coefficients.
    #===========================================================================    
    def get_JAMP_coefs(self, color_amplitudes, color_basis=None, tag_letter="",\
                       n=50, Nc_value=3):
        """This functions return the lines defining the DATA statement setting
        the coefficients building the JAMPS out of the AMPS. Split rows in
        bunches of size n.
        One can specify the color_basis from which the color amplitudes originates
        so that there are commentaries telling what color structure each JAMP
        corresponds to."""
        
        if(not isinstance(color_amplitudes,list) or 
           not (color_amplitudes and isinstance(color_amplitudes[0],list))):
                raise MadGraph5Error, "Incorrect col_amps argument passed to get_JAMP_coefs"

        res_list = []
        my_cs = color.ColorString()
        for index, coeff_list in enumerate(color_amplitudes):
            # Create the list of the complete numerical coefficient.
            coefs_list=[coefficient[0][0]*coefficient[0][1]*\
                        (fractions.Fraction(Nc_value)**coefficient[0][3]) for \
                        coefficient in coeff_list]
            # Create the list of the numbers of the contributing amplitudes.
            # Mutliply by -1 for those which have an imaginary coefficient.
            ampnumbers_list=[coefficient[1]*(-1 if coefficient[0][2] else 1) \
                              for coefficient in coeff_list]
            # Find the common denominator.      
            commondenom=abs(reduce(fractions.gcd, coefs_list).denominator)
            num_list=[(coefficient*commondenom).numerator \
                      for coefficient in coefs_list]
            res_list.append("DATA NCONTRIBAMPS%s(%i)/%i/"%(tag_letter,\
                                                         index+1,len(num_list)))
            res_list.append("DATA DENOMCCOEF%s(%i)/%i/"%(tag_letter,\
                                                         index+1,commondenom))
            if color_basis:
                my_cs.from_immutable(sorted(color_basis.keys())[index])
                res_list.append("C %s" % repr(my_cs))
            for k in xrange(0, len(num_list), n):
                res_list.append("DATA (NUMCCOEF%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in num_list[k:k + n]])))
                res_list.append("DATA (AMPNUMBERS%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in ampnumbers_list[k:k + n]])))
                pass
        return res_list


    def get_JAMP_lines_split_order(self, col_amps, split_order_amps, 
          split_order_names=None, JAMP_format="JAMP(%s)", AMP_format="AMP(%s)"):
        """Return the JAMP = sum(fermionfactor * AMP(i)) lines from col_amps 
        defined as a matrix element or directly as a color_amplitudes dictionary.
        The split_order_amps specifies the group of amplitudes sharing the same
        amplitude orders which should be put in together in a given set of JAMPS.
        The split_order_amps is supposed to have the format of the second output 
        of the function get_split_orders_mapping function in helas_objects.py.
        The split_order_names is optional (it should correspond to the process
        'split_orders' attribute) and only present to provide comments in the
        JAMP definitions in the code."""

        # Let the user call get_JAMP_lines_split_order directly from a 
        error_msg="Malformed '%s' argument passed to the "+\
                 "get_JAMP_lines_split_order function: %s"%str(split_order_amps)
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            color_amplitudes=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error, error_msg%'col_amps'
        else:
            raise MadGraph5Error, error_msg%'col_amps'
        
        # Verify the sanity of the split_order_amps and split_order_names args
        if isinstance(split_order_amps,list):
            for elem in split_order_amps:
                if len(elem)!=2:
                    raise MadGraph5Error, error_msg%'split_order_amps'
                # Check the first element of the two lists to make sure they are
                # integers, although in principle they should all be integers.
                if not isinstance(elem[0],tuple) or \
                   not isinstance(elem[1],tuple) or \
                   not isinstance(elem[0][0],int) or \
                   not isinstance(elem[1][0],int):
                    raise MadGraph5Error, error_msg%'split_order_amps'
        else:
            raise MadGraph5Error, error_msg%'split_order_amps'
        
        if not split_order_names is None:
            if isinstance(split_order_names,list):
                # Should specify the same number of names as there are elements
                # in the key of the split_order_amps.
                if len(split_order_names)!=len(split_order_amps[0][0]):
                    raise MadGraph5Error, error_msg%'split_order_names'
                # Check the first element of the list to be a string
                if not isinstance(split_order_names[0],str):
                    raise MadGraph5Error, error_msg%'split_order_names'                    
            else:
                raise MadGraph5Error, error_msg%'split_order_names'                
        
        # Now scan all contributing orders to be individually computed and 
        # construct the list of color_amplitudes for JAMP to be constructed
        # accordingly.
        res_list=[]
        for i, amp_order in enumerate(split_order_amps):
            col_amps_order = []
            for jamp in color_amplitudes:
                col_amps_order.append(filter(lambda col_amp: 
                                               col_amp[1] in amp_order[1],jamp))
            if split_order_names:
                res_list.append('C JAMPs contributing to orders '+' '.join(
                              ['%s=%i'%order for order in zip(split_order_names,
                                                                amp_order[0])]))
            if self.opt['export_format'] in ['madloop_matchbox']:
                res_list.extend(self.get_JAMP_lines(col_amps_order,
                                   JAMP_format="JAMP(%s,{0})".format(str(i+1)),
                                   JAMP_formatLC="LNJAMP(%s,{0})".format(str(i+1))))
            else:
                res_list.extend(self.get_JAMP_lines(col_amps_order,
                                   JAMP_format="JAMP(%s,{0})".format(str(i+1))))         

        return res_list


    def get_JAMP_lines(self, col_amps, JAMP_format="JAMP(%s)", AMP_format="AMP(%s)", 
                       split=-1):
        """Return the JAMP = sum(fermionfactor * AMP(i)) lines from col_amps 
        defined as a matrix element or directly as a color_amplitudes dictionary,
        Jamp_formatLC should be define to allow to add LeadingColor computation 
        (usefull for MatchBox)
        The split argument defines how the JAMP lines should be split in order
        not to be too long."""

        # Let the user call get_JAMP_lines directly from a MatrixElement or from
        # the color amplitudes lists.
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            color_amplitudes=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error, "Incorrect col_amps argument passed to get_JAMP_lines"
        else:
            raise MadGraph5Error, "Incorrect col_amps argument passed to get_JAMP_lines"


        res_list = []
        for i, coeff_list in enumerate(color_amplitudes):
            # It might happen that coeff_list is empty if this function was
            # called from get_JAMP_lines_split_order (i.e. if some color flow
            # does not contribute at all for a given order).
            # In this case we simply set it to 0.
            if coeff_list==[]:
                res_list.append(((JAMP_format+"=0D0") % str(i + 1)))
                continue
            # Break the JAMP definition into 'n=split' pieces to avoid having
            # arbitrarly long lines.
            first=True
            n = (len(coeff_list)+1 if split<=0 else split) 
            while coeff_list!=[]:
                coefs=coeff_list[:n]
                coeff_list=coeff_list[n:]
                res = ((JAMP_format+"=") % str(i + 1)) + \
                      ((JAMP_format % str(i + 1)) if not first and split>0 else '')

                first=False
                # Optimization: if all contributions to that color basis element have
                # the same coefficient (up to a sign), put it in front
                list_fracs = [abs(coefficient[0][1]) for coefficient in coefs]
                common_factor = False
                diff_fracs = list(set(list_fracs))
                if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                    common_factor = True
                    global_factor = diff_fracs[0]
                    res = res + '%s(' % self.coeff(1, global_factor, False, 0)
                
                # loop for JAMP
                for (coefficient, amp_number) in coefs:
                    if not coefficient:
                        continue
                    if common_factor:
                        res = (res + "%s" + AMP_format) % \
                                                   (self.coeff(coefficient[0],
                                                   coefficient[1] / abs(coefficient[1]),
                                                   coefficient[2],
                                                   coefficient[3]),
                                                   str(amp_number))
                    else:
                        res = (res + "%s" + AMP_format) % (self.coeff(coefficient[0],
                                                   coefficient[1],
                                                   coefficient[2],
                                                   coefficient[3]),
                                                   str(amp_number))
    
                if common_factor:
                    res = res + ')'
    
                res_list.append(res)
                    
        return res_list

    def get_pdf_lines(self, matrix_element, ninitial, subproc_group = False):
        """Generate the PDF lines for the auto_dsig.f file"""

        processes = matrix_element.get('processes')
        model = processes[0].get('model')

        pdf_definition_lines = ""
        pdf_data_lines = ""
        pdf_lines = ""

        if ninitial == 1:
            pdf_lines = "PD(0) = 0d0\nIPROC = 0\n"
            for i, proc in enumerate(processes):
                process_line = proc.base_string()
                pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                pdf_lines = pdf_lines + "\nPD(IPROC)=1d0\n"
                pdf_lines = pdf_lines + "\nPD(0)=PD(0)+PD(IPROC)\n"
        else:
            # Pick out all initial state particles for the two beams
            initial_states = [sorted(list(set([p.get_initial_pdg(1) for \
                                               p in processes]))),
                              sorted(list(set([p.get_initial_pdg(2) for \
                                               p in processes])))]

            # Prepare all variable names
            pdf_codes = dict([(p, model.get_particle(p).get_name()) for p in \
                              sum(initial_states,[])])
            for key,val in pdf_codes.items():
                pdf_codes[key] = val.replace('~','x').replace('+','p').replace('-','m')

            # Set conversion from PDG code to number used in PDF calls
            pdgtopdf = {21: 0, 22: 7}

            # Fill in missing entries of pdgtopdf
            for pdg in sum(initial_states,[]):
                if not pdg in pdgtopdf and not pdg in pdgtopdf.values():
                    pdgtopdf[pdg] = pdg
                elif pdg not in pdgtopdf and pdg in pdgtopdf.values():
                    # If any particle has pdg code 7, we need to use something else
                    pdgtopdf[pdg] = 6000000 + pdg
                    
            # Get PDF variable declarations for all initial states
            for i in [0,1]:
                pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in \
                                                 initial_states[i]]) + \
                                                 "\n"

            # Get PDF data lines for all initial states
            for i in [0,1]:
                pdf_data_lines += "DATA " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in initial_states[i]]) + \
                                                 "/%d*1D0/" % len(initial_states[i]) + \
                                                 "\n"

            # Get PDF lines for all different initial states
            for i, init_states in enumerate(initial_states):
                if subproc_group:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(IB(%d))).GE.1) THEN\nLP=SIGN(1,LPP(IB(%d)))\n" \
                                 % (i + 1, i + 1)
                else:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(%d)) .GE. 1) THEN\nLP=SIGN(1,LPP(%d))\n" \
                                 % (i + 1, i + 1)

                for initial_state in init_states:
                    if initial_state in pdf_codes.keys():
                        if subproc_group:
                            pdf_lines = pdf_lines + \
                                        ("%s%d=PDG2PDF(ABS(LPP(IB(%d))),%d*LP," + \
                                         "XBK(IB(%d)),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, i + 1, pdgtopdf[initial_state],
                                          i + 1, i + 1)
                        else:
                            pdf_lines = pdf_lines + \
                                        ("%s%d=PDG2PDF(ABS(LPP(%d)),%d*LP," + \
                                         "XBK(%d),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, i + 1, pdgtopdf[initial_state],
                                          i + 1, i + 1)
                pdf_lines = pdf_lines + "ENDIF\n"

            # Add up PDFs for the different initial state particles
            pdf_lines = pdf_lines + "PD(0) = 0d0\nIPROC = 0\n"
            for proc in processes:
                process_line = proc.base_string()
                pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                pdf_lines = pdf_lines + "\nPD(IPROC)="
                for ibeam in [1, 2]:
                    initial_state = proc.get_initial_pdg(ibeam)
                    if initial_state in pdf_codes.keys():
                        pdf_lines = pdf_lines + "%s%d*" % \
                                    (pdf_codes[initial_state], ibeam)
                    else:
                        pdf_lines = pdf_lines + "1d0*"
                # Remove last "*" from pdf_lines
                pdf_lines = pdf_lines[:-1] + "\n"
                pdf_lines = pdf_lines + "PD(0)=PD(0)+DABS(PD(IPROC))\n"

        # Remove last line break from the return variables
        return pdf_definition_lines[:-1], pdf_data_lines[:-1], pdf_lines[:-1]

    #===========================================================================
    # write_props_file
    #===========================================================================
    def write_props_file(self, writer, matrix_element, s_and_t_channels):
        """Write the props.inc file for MadEvent. Needs input from
        write_configs_file."""

        lines = []

        particle_dict = matrix_element.get('processes')[0].get('model').\
                        get('particle_dict')

        for iconf, configs in enumerate(s_and_t_channels):
            for vertex in configs[0] + configs[1][:-1]:
                leg = vertex.get('legs')[-1]
                if leg.get('id') not in particle_dict:
                    # Fake propagator used in multiparticle vertices
                    mass = 'zero'
                    width = 'zero'
                    pow_part = 0
                else:
                    particle = particle_dict[leg.get('id')]
                    # Get mass
                    if particle.get('mass').lower() == 'zero':
                        mass = particle.get('mass')
                    else:
                        mass = "abs(%s)" % particle.get('mass')
                    # Get width
                    if particle.get('width').lower() == 'zero':
                        width = particle.get('width')
                    else:
                        width = "abs(%s)" % particle.get('width')

                    pow_part = 1 + int(particle.is_boson())

                lines.append("prmass(%d,%d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines.append("prwidth(%d,%d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines.append("pow(%d,%d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        configs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        model = matrix_element.get('processes')[0].get('model')
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,
                                                            model)

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""

        lines = []

        s_and_t_channels = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
            for config in configs if [d for d in config if d][0].\
                                             get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any(vert > minvert for vert in [d for d in helas_diags if d]\
              [0].get_vertex_leg_numbers()) :
                # Only 3-vertices allowed in configs.inc except for vertices
                # which originate from a shrunk loop.
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial, model, new_pdg))
                else:
                    stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = zip(*[s for s,t in stchannels])
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("data mapconfig(%d)/%d/" % (nconfigs,
                                                     mapconfigs[iconfig]))

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), nconfigs, len(daughters),
                              ",".join([str(d) for d in daughters])))
                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join([str(d) for d in pdgs])))
                    lines.append("data tprid(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels[:-1]:
                    lines.append("data tprid(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  abs(last_leg.get('id'))))
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%d/" % nconfigs)

        # Write the file
        writer.writelines(lines)

        return s_and_t_channels

    #===========================================================================
    # Global helper methods
    #===========================================================================

    def coeff(self, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""

        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

        if total_coeff == 1:
            if is_imaginary:
                return '+imag1*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-imag1*'
            else:
                return '-'

        res_str = '%+iD0' % total_coeff.numerator

        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%iD0' % total_coeff.denominator

        if is_imaginary:
            res_str = res_str + '*imag1'

        return res_str + '*'


    def set_fortran_compiler(self, default_compiler, force=False):
        """Set compiler based on what's available on the system"""
               
        # Check for compiler
        if default_compiler['fortran'] and misc.which(default_compiler['fortran']):
            f77_compiler = default_compiler['fortran']
        elif misc.which('gfortran'):
            f77_compiler = 'gfortran'
        elif misc.which('g77'):
            f77_compiler = 'g77'
        elif misc.which('f77'):
            f77_compiler = 'f77'
        elif default_compiler['fortran']:
            logger.warning('No Fortran Compiler detected! Please install one')
            f77_compiler = default_compiler['fortran'] # maybe misc fail so try with it
        else:
            raise MadGraph5Error, 'No Fortran Compiler detected! Please install one'
        logger.info('Use Fortran compiler ' + f77_compiler)
        
        
        # Check for compiler. 1. set default.
        if default_compiler['f2py']:
            f2py_compiler = default_compiler['f2py']
        else:
            f2py_compiler = ''
        # Try to find the correct one.
        if default_compiler['f2py'] and misc.which(default_compiler['f2py']):
            f2py_compiler = default_compiler
        elif misc.which('f2py'):
            f2py_compiler = 'f2py'
        elif sys.version_info[1] == 6:
            if misc.which('f2py-2.6'):
                f2py_compiler = 'f2py-2.6'
            elif misc.which('f2py2.6'):
                f2py_compiler = 'f2py2.6'
        elif sys.version_info[1] == 7:
            if misc.which('f2py-2.7'):
                f2py_compiler = 'f2py-2.7'
            elif misc.which('f2py2.7'):
                f2py_compiler = 'f2py2.7'            
        
        to_replace = {'fortran': f77_compiler, 'f2py': f2py_compiler}
        
        
        self.replace_make_opt_f_compiler(to_replace)
        # Replace also for Template but not for cluster
        if not os.environ.has_key('MADGRAPH_DATA') and ReadWrite:
            self.replace_make_opt_f_compiler(to_replace, pjoin(MG5DIR, 'Template', 'LO'))
        
        return f77_compiler

    # an alias for backward compatibility
    set_compiler = set_fortran_compiler


    def set_cpp_compiler(self, default_compiler, force=False):
        """Set compiler based on what's available on the system"""
                
        # Check for compiler
        if default_compiler and misc.which(default_compiler):
            compiler = default_compiler
        elif misc.which('g++'):
            #check if clang version
            p = misc.Popen(['g++', '--version'], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) 
            out, _ = p.communicate()
            if 'clang' in out and  misc.which('clang'):
                compiler = 'clang'
            else:
                compiler = 'g++'
        elif misc.which('c++'):
            compiler = 'c++'
        elif misc.which('clang'):
            compiler = 'clang'
        elif default_compiler:
            logger.warning('No c++ Compiler detected! Please install one')
            compiler = default_compiler # maybe misc fail so try with it
        else:
            raise MadGraph5Error, 'No c++ Compiler detected! Please install one'
        logger.info('Use c++ compiler ' + compiler)
        self.replace_make_opt_c_compiler(compiler)
        # Replace also for Template but not for cluster
        if not os.environ.has_key('MADGRAPH_DATA') and ReadWrite and \
           not __debug__ and not os.path.exists(pjoin(MG5DIR,'bin','create_release.py')):
            self.replace_make_opt_c_compiler(compiler, pjoin(MG5DIR, 'Template', 'LO'))
        
        return compiler


    def replace_make_opt_f_compiler(self, compilers, root_dir = ""):
        """Set FC=compiler in Source/make_opts"""

        assert isinstance(compilers, dict)
        
        mod = False #avoid to rewrite the file if not needed
        if not root_dir:
            root_dir = self.dir_path
            
        compiler= compilers['fortran']
        f2py_compiler = compilers['f2py']
        if not f2py_compiler:
            f2py_compiler = 'f2py'
        for_update= {'DEFAULT_F_COMPILER':compiler,
                     'DEFAULT_F2PY_COMPILER':f2py_compiler}
        make_opts = pjoin(root_dir, 'Source', 'make_opts')

        try:
            common_run_interface.CommonRunCmd.update_make_opts_full(
                            make_opts, for_update)
        except IOError:
            if root_dir == self.dir_path:
                logger.info('Fail to set compiler. Trying to continue anyway.')            

    def replace_make_opt_c_compiler(self, compiler, root_dir = ""):
        """Set CXX=compiler in Source/make_opts.
        The version is also checked, in order to set some extra flags
        if the compiler is clang (on MACOS)"""
       
        is_clang = misc.detect_if_cpp_compiler_is_clang(compiler)
        is_lc    = misc.detect_cpp_std_lib_dependence(compiler) == '-lc++'

        # list of the variable to set in the make_opts file
        for_update= {'DEFAULT_CPP_COMPILER':compiler,
                     'MACFLAG':'-mmacosx-version-min=10.7' if is_clang and is_lc else '',
                     'STDLIB': '-lc++' if is_lc else '-lstdc++',
                     'STDLIB_FLAG': '-stdlib=libc++' if is_lc and is_clang else ''
                     }
        
        if not root_dir:
            root_dir = self.dir_path
        make_opts = pjoin(root_dir, 'Source', 'make_opts')

        try:
            common_run_interface.CommonRunCmd.update_make_opts_full(
                            make_opts, for_update)
        except IOError:
            if root_dir == self.dir_path:
                logger.info('Fail to set compiler. Trying to continue anyway.')  
    
        return

#===============================================================================
# ProcessExporterFortranSA
#===============================================================================
class ProcessExporterFortranSA(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 StandAlone format."""

    matrix_template = "matrix_standalone_v4.inc"

    def __init__(self, *args, **opts):
        """add the format information compare to standard init"""
        
        if 'format' in opts:
            self.format = opts['format']
            del opts['format']
        else:
            self.format = 'standalone'
        ProcessExporterFortran.__init__(self, *args, **opts)

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        #First copy the full template tree if dir_path doesn't exit
        if os.path.isdir(self.dir_path):
            return
        
        logger.info('initialize a new standalone directory: %s' % \
                        os.path.basename(self.dir_path))
        temp_dir = pjoin(self.mgme_dir, 'Template/LO')
        
        # Create the directory structure
        os.mkdir(self.dir_path)
        os.mkdir(pjoin(self.dir_path, 'Source'))
        os.mkdir(pjoin(self.dir_path, 'Source', 'MODEL'))
        os.mkdir(pjoin(self.dir_path, 'Source', 'DHELAS'))
        os.mkdir(pjoin(self.dir_path, 'SubProcesses'))
        os.mkdir(pjoin(self.dir_path, 'bin'))
        os.mkdir(pjoin(self.dir_path, 'bin', 'internal'))
        os.mkdir(pjoin(self.dir_path, 'lib'))
        os.mkdir(pjoin(self.dir_path, 'Cards'))
        
        # Information at top-level
        #Write version info
        shutil.copy(pjoin(temp_dir, 'TemplateVersion.txt'), self.dir_path)
        try:
            shutil.copy(pjoin(self.mgme_dir, 'MGMEVersion.txt'), self.dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])
        
        
        # Add file in SubProcesses
        shutil.copy(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f_sp'), 
                    pjoin(self.dir_path, 'SubProcesses', 'makefile'))
        
        if self.format == 'standalone':
            shutil.copy(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'check_sa.f'), 
                    pjoin(self.dir_path, 'SubProcesses', 'check_sa.f'))
                        
        # Add file in Source
        shutil.copy(pjoin(temp_dir, 'Source', 'make_opts'), 
                    pjoin(self.dir_path, 'Source'))        
        # add the makefile 
        filename = pjoin(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FileWriter(filename))          
        
    #===========================================================================
    # export model files
    #=========================================================================== 
    def export_model_files(self, model_path):
        """export the model dependent files for V4 model"""

        super(ProcessExporterFortranSA,self).export_model_files(model_path)
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO  
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(os.path.join(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()        
        
        text = open(pjoin(self.dir_path,'SubProcesses','check_sa.f')).read()
        text = text.replace('call setpara(\'param_card.dat\')', 'call setpara(\'param_card.dat\', .true.)')
        fsock = open(pjoin(self.dir_path,'SubProcesses','check_sa.f'), 'w')
        fsock.write(text)
        fsock.close()
        
        self.make_model_symbolic_link()

    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS and MODEL directories, to set up
        everything for running standalone
        """

        source_dir = pjoin(self.dir_path, "Source")
        logger.info("Running make for Helas")
        misc.compile(arg=['../lib/libdhelas.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for Model")
        misc.compile(arg=['../lib/libmodel.a'], cwd=source_dir, mode='fortran')

    #===========================================================================
    # Create proc_card_mg5.dat for Standalone directory
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """Finalize Standalone MG4 directory by generation proc_card_mg5.dat"""
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}

        self.compiler_choice(compiler)
        self.make()

        # Write command history as proc_card_mg5
        if history and os.path.isdir(pjoin(self.dir_path, 'Cards')):
            output_file = pjoin(self.dir_path, 'Cards', 'proc_card_mg5.dat')
            history.write(output_file)
        
        ProcessExporterFortran.finalize(self, matrix_elements, 
                                             history, mg5options, flaglist)
        open(pjoin(self.dir_path,'__init__.py'),'w')
        open(pjoin(self.dir_path,'SubProcesses','__init__.py'),'w')

        if 'mode' in self.opt and self.opt['mode'] == "reweight":
            #add the module to hande the NLO weight
            files.copytree(pjoin(MG5DIR, 'Template', 'RWGTNLO'),
                          pjoin(self.dir_path, 'Source'))
            files.copytree(pjoin(MG5DIR, 'Template', 'NLO', 'Source', 'PDF'),
                           pjoin(self.dir_path, 'Source', 'PDF'))
            self.write_pdf_opendata()

    def create_MA5_cards(self,*args,**opts):
        """ Overload the function of the mother so as to bypass this in StandAlone."""
        pass

    def compiler_choice(self, compiler):
        """ Different daughter classes might want different compilers.
        So this function is meant to be overloaded if desired."""
        
        self.set_compiler(compiler)

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model, number):
        """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
        including the necessary matrix.f and nexternal.inc files"""

        cwd = os.getcwd()

        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = pjoin(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

        if self.opt['sa_symmetry']:
            # avoid symmetric output
            for i,proc in enumerate(matrix_element.get('processes')):
                        
                initial = []    #filled in the next line
                final = [l.get('id') for l in proc.get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
                decay_finals = proc.get_final_ids_after_decay()
                decay_finals.sort()
                tag = (tuple(initial), tuple(decay_finals))
                legs = proc.get('legs')[:]
                leg0 = proc.get('legs')[0]
                leg1 = proc.get('legs')[1]
                if not leg1.get('state'):
                    proc.get('legs')[0] = leg1
                    proc.get('legs')[1] = leg0
                    flegs = proc.get('legs')[2:]
                    for perm in itertools.permutations(flegs):
                        for i,p in enumerate(perm):
                            proc.get('legs')[i+2] = p
                        dirpath2 =  pjoin(self.dir_path, 'SubProcesses', \
                               "P%s" % proc.shell_string())
                        #restore original order
                        proc.get('legs')[2:] = legs[2:]              
                        if os.path.exists(dirpath2):
                            proc.get('legs')[:] = legs
                            return 0
                proc.get('legs')[:] = legs

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        #try:
        #    os.chdir(dirpath)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % dirpath)
        #    return 0

        logger.info('Creating files in directory %s' % dirpath)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Create the matrix.f file and the nexternal.inc file
        if self.opt['export_format']=='standalone_msP':
            filename = pjoin(dirpath, 'matrix_prod.f')
        else:
            filename = pjoin(dirpath, 'matrix.f')
        calls = self.write_matrix_element_v4(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model)

        if self.opt['export_format'] == 'standalone_msP':
            filename =  pjoin(dirpath,'configs_production.inc')
            mapconfigs, s_and_t_channels = self.write_configs_file(\
                writers.FortranWriter(filename),
                matrix_element)

            filename =  pjoin(dirpath,'props_production.inc')
            self.write_props_file(writers.FortranWriter(filename),
                             matrix_element,
                             s_and_t_channels)

            filename =  pjoin(dirpath,'nexternal_prod.inc')
            self.write_nexternal_madspin(writers.FortranWriter(filename),
                             nexternal, ninitial)

        if self.opt['export_format']=='standalone_msF':
            filename = pjoin(dirpath, 'helamp.inc')
            ncomb=matrix_element.get_helicity_combinations()
            self.write_helamp_madspin(writers.FortranWriter(filename),
                             ncomb)
            
        filename = pjoin(dirpath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(dirpath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(dirpath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        # Generate diagrams
        filename = pjoin(dirpath, "matrix.ps")
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                             get('diagrams'),
                                          filename,
                                          model=matrix_element.get('processes')[0].\
                                             get('model'),
                                          amplitude=True)
        logger.info("Generating Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        linkfiles = ['check_sa.f', 'coupl.inc', 'makefile']

        for file in linkfiles:
            ln('../%s' % file, cwd=dirpath)

        # Return to original PWD
        #os.chdir(cwd)

        if not calls:
            calls = 0
        return calls


    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = '$(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)'
        model_line='''$(LIBDIR)libmodel.$(libext): MODEL\n\t cd MODEL; make\n'''
        text = open(path).read() % {'libraries': set_of_lib, 'model':model_line} 
        writer.write(text)
        
        return True

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                write=True, proc_prefix=''):
        """Export a matrix element to a matrix.f file in MG4 standalone format
        if write is on False, just return the replace_dict and not write anything."""


        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        if writer:
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter but %s" % type(writer))
            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False

            
        if not self.opt.has_key('sa_symmetry'):
            self.opt['sa_symmetry']=False



        # The proc_id is for MadEvent grouping which is never used in SA.
        replace_dict = {'global_variable':'', 'amp2_lines':'',
                                       'proc_prefix':proc_prefix, 'proc_id':''}

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)

        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal
        replace_dict['nincoming'] = ninitial

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        replace_dict['den_factor_line'] = self.get_den_factor_line(matrix_element)

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor

        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        replace_dict['beamone_helavgfactor'], replace_dict['beamtwo_helavgfactor'] =\
                                       matrix_element.get_beams_hel_avg_factor()

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        if self.opt['export_format']=='standalone_msP':
        # For MadSpin need to return the AMP2
            amp2_lines = self.get_amp2_lines(matrix_element, [] )
            replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
            replace_dict['global_variable'] = \
         "       Double Precision amp2(NGRAPHS)\n       common/to_amps/  amp2\n"

        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')

        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines = self.get_JAMP_lines(matrix_element)
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            replace_dict['nSqAmpSplitOrders']=1
            replace_dict['split_order_str_list']=''
        else:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            replace_dict['split_order_str_list']=str(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            replace_dict['sqsplitorders']='\n'.join(sqamp_so)           
            jamp_lines = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders)
            
            # Now setup the array specifying what squared split order is chosen
            replace_dict['chosen_so_configs']=self.set_chosen_SO_index(
                              matrix_element.get('processes')[0],squared_orders)
            
            # For convenience we also write the driver check_sa_splitOrders.f
            # that explicitely writes out the contribution from each squared order.
            # The original driver still works and is compiled with 'make' while
            # the splitOrders one is compiled with 'make check_sa_born_splitOrders'
            check_sa_writer=writers.FortranWriter('check_sa_born_splitOrders.f')
            self.write_check_sa_splitOrders(squared_orders,split_orders,
              nexternal,ninitial,proc_prefix,check_sa_writer)

        if write:
            writers.FortranWriter('nsqso_born.inc').writelines(
                """INTEGER NSQSO_BORN
                   PARAMETER (NSQSO_BORN=%d)"""%replace_dict['nSqAmpSplitOrders'])

        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)    

        matrix_template = self.matrix_template
        if self.opt['export_format']=='standalone_msP' :
            matrix_template = 'matrix_standalone_msP_v4.inc'
        elif self.opt['export_format']=='standalone_msF':
            matrix_template = 'matrix_standalone_msF_v4.inc'
        elif self.opt['export_format']=='matchbox':
            replace_dict["proc_prefix"] = 'MG5_%i_' % matrix_element.get('processes')[0].get('id')
            replace_dict["color_information"] = self.get_color_string_lines(matrix_element)

        if len(split_orders)>0:
            if self.opt['export_format'] in ['standalone_msP', 'standalone_msF']:
                logger.debug("Warning: The export format %s is not "+\
                  " available for individual ME evaluation of given coupl. orders."+\
                  " Only the total ME will be computed.", self.opt['export_format'])
            elif  self.opt['export_format'] in ['madloop_matchbox']:
                replace_dict["color_information"] = self.get_color_string_lines(matrix_element)
                matrix_template = "matrix_standalone_matchbox_splitOrders_v4.inc"
            else:
                matrix_template = "matrix_standalone_splitOrders_v4.inc"

        replace_dict['template_file'] = pjoin(_file_path, 'iolibs', 'template_files', matrix_template)
        replace_dict['template_file2'] = pjoin(_file_path, \
                                   'iolibs/template_files/split_orders_helping_functions.inc')
        if write and writer:
            path = replace_dict['template_file']
            content = open(path).read()
            content = content % replace_dict
            # Write the file
            writer.writelines(content)
            # Add the helper functions.
            if len(split_orders)>0:
                content = '\n' + open(replace_dict['template_file2'])\
                                   .read()%replace_dict
                writer.writelines(content)
            return len(filter(lambda call: call.find('#') != 0, helas_calls))
        else:
            replace_dict['return_value'] = len(filter(lambda call: call.find('#') != 0, helas_calls))
            return replace_dict # for subclass update

    def write_check_sa_splitOrders(self,squared_orders, split_orders, nexternal,
                                                nincoming, proc_prefix, writer):
        """ Write out a more advanced version of the check_sa drivers that
        individually returns the matrix element for each contributing squared
        order."""
        
        check_sa_content = open(pjoin(self.mgme_dir, 'madgraph', 'iolibs', \
                             'template_files', 'check_sa_splitOrders.f')).read()
        printout_sq_orders=[]
        for i, squared_order in enumerate(squared_orders):
            sq_orders=[]
            for j, sqo in enumerate(squared_order):
                sq_orders.append('%s=%d'%(split_orders[j],sqo))
            printout_sq_orders.append(\
                    "write(*,*) '%d) Matrix element for (%s) = ',MATELEMS(%d)"\
                                                 %(i+1,' '.join(sq_orders),i+1))
        printout_sq_orders='\n'.join(printout_sq_orders)
        replace_dict = {'printout_sqorders':printout_sq_orders, 
                        'nSplitOrders':len(squared_orders),
                        'nexternal':nexternal,
                        'nincoming':nincoming,
                        'proc_prefix':proc_prefix}
        
        if writer:
            writer.writelines(check_sa_content % replace_dict)
        else:
            return replace_dict

class ProcessExporterFortranMatchBox(ProcessExporterFortranSA):
    """class to take care of exporting a set of matrix element for the Matchbox
    code in the case of Born only routine"""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'matchbox', 'mp': False,
                        'sa_symmetry': True}

    #specific template of the born
           

    matrix_template = "matrix_standalone_matchbox.inc"
    
    @staticmethod    
    def get_color_string_lines(matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["out = 1"])
        
        #start the real work
        color_denominators = matrix_element.get('color_matrix').\
                                                         get_line_denominators()
        matrix_strings = []
        my_cs = color.ColorString()
        for i_color in xrange(len(color_denominators)):
            # Then write the numerators for the matrix elements
            my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[i_color])
            t_str=repr(my_cs)
            t_match=re.compile(r"(\w+)\(([\s\d+\,]*)\)")
            # from '1 T(2,4,1) Tr(4,5,6) Epsilon(5,3,2,1) T(1,2)' returns with findall:
            # [('T', '2,4,1'), ('Tr', '4,5,6'), ('Epsilon', '5,3,2,1'), ('T', '1,2')]
            all_matches = t_match.findall(t_str)
            output = {}
            arg=[]
            for match in all_matches:
                ctype, tmparg = match[0], [m.strip() for m in match[1].split(',')]
                if ctype in ['ColorOne' ]:
                    continue
                if ctype not in ['T', 'Tr' ]:
                    raise MadGraph5Error, 'Color Structure not handled by Matchbox: %s'  % ctype
                tmparg += ['0']
                arg +=tmparg
            for j, v in enumerate(arg):
                    output[(i_color,j)] = v

            for key in output:
                if matrix_strings == []:
                    #first entry
                    matrix_strings.append(""" 
                    if (in1.eq.%s.and.in2.eq.%s)then
                    out = %s
                    """  % (key[0], key[1], output[key]))
                else:
                    #not first entry
                    matrix_strings.append(""" 
                    elseif (in1.eq.%s.and.in2.eq.%s)then
                    out = %s
                    """  % (key[0], key[1], output[key]))
        if len(matrix_strings):                
            matrix_strings.append(" else \n out = - 1 \n endif")
        else: 
            return "\n out = - 1 \n "
        return "\n".join(matrix_strings)
    
    def make(self,*args,**opts):
        pass

    def get_JAMP_lines(self, col_amps, JAMP_format="JAMP(%s)", AMP_format="AMP(%s)", split=-1,
                       JAMP_formatLC=None):
    
        """Adding leading color part of the colorflow"""
        
        if not JAMP_formatLC:
            JAMP_formatLC= "LN%s" % JAMP_format

        error_msg="Malformed '%s' argument passed to the get_JAMP_lines"
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            col_amps=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                col_amps=col_amps
            else:
                raise MadGraph5Error, error_msg % 'col_amps'
        else:
            raise MadGraph5Error, error_msg % 'col_amps'

        text = super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(col_amps,
                                            JAMP_format=JAMP_format,
                                            AMP_format=AMP_format,
                                            split=-1)
        
        
        # Filter the col_ampls to generate only those without any 1/NC terms
        
        LC_col_amps = []
        for coeff_list in col_amps:
            to_add = []
            for (coefficient, amp_number) in coeff_list:
                if coefficient[3]==0:
                    to_add.append( (coefficient, amp_number) )
            LC_col_amps.append(to_add)
           
        text += super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(LC_col_amps,
                                            JAMP_format=JAMP_formatLC,
                                            AMP_format=AMP_format,
                                            split=-1)
        
        return text




#===============================================================================
# ProcessExporterFortranMW
#===============================================================================
class ProcessExporterFortranMW(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 - MadWeight format."""

    matrix_file="matrix_standalone_v4.inc"

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranMW, self).copy_template(model)        

        # Add the MW specific file
        shutil.copytree(pjoin(MG5DIR,'Template','MadWeight'),
                               pjoin(self.dir_path, 'Source','MadWeight'), True)        
        shutil.copytree(pjoin(MG5DIR,'madgraph','madweight'),
                        pjoin(self.dir_path, 'bin','internal','madweight'), True) 
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','setrun.f'),
                                      pjoin(self.dir_path, 'Source','setrun.f'))
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','run.inc'),
                                      pjoin(self.dir_path, 'Source','run.inc'))
        # File created from Template (Different in some child class)
        filename = os.path.join(self.dir_path,'Source','run_config.inc')
        self.write_run_config_file(writers.FortranWriter(filename))

        try:
            subprocess.call([os.path.join(self.dir_path, 'Source','MadWeight','bin','internal','pass_to_madweight')],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = os.open(os.devnull, os.O_RDWR),
                            cwd=self.dir_path)
        except OSError:
            # Probably madweight already called
            pass
        
        # Copy the different python file in the Template
        self.copy_python_file()
        # create the appropriate cuts.f
        self.get_mw_cuts_version()

        # add the makefile in Source directory 
        filename = os.path.join(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FortranWriter(filename))



        
    #===========================================================================
    # convert_model
    #===========================================================================    
    def convert_model(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
         
        super(ProcessExporterFortranMW,self).convert_model(model, 
                                               wanted_lorentz, wanted_couplings)
         
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
        # This is not safe if there is a '##' or '-' in the path.
        shutil.copytree(model_path, 
                               pjoin(self.dir_path,'bin','internal','ufomodel'),
                               ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
        if hasattr(model, 'restrict_card'):
            out_path = pjoin(self.dir_path, 'bin', 'internal','ufomodel',
                                                         'restrict_default.dat')
            if isinstance(model.restrict_card, check_param_card.ParamCard):
                model.restrict_card.write(out_path)
            else:
                files.cp(model.restrict_card, out_path)

    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================        
    def copy_python_file(self):
        """copy the python file require for the Template"""

        # madevent interface
        cp(_file_path+'/interface/madweight_interface.py',
                            self.dir_path+'/bin/internal/madweight_interface.py')
        cp(_file_path+'/interface/extended_cmd.py',
                                  self.dir_path+'/bin/internal/extended_cmd.py')
        cp(_file_path+'/interface/common_run_interface.py',
                            self.dir_path+'/bin/internal/common_run_interface.py')
        cp(_file_path+'/various/misc.py', self.dir_path+'/bin/internal/misc.py')        
        cp(_file_path+'/iolibs/files.py', self.dir_path+'/bin/internal/files.py')
        #cp(_file_path+'/iolibs/save_load_object.py', 
        #                      self.dir_path+'/bin/internal/save_load_object.py') 
        cp(_file_path+'/iolibs/file_writers.py', 
                              self.dir_path+'/bin/internal/file_writers.py')
        #model file                        
        cp(_file_path+'../models/check_param_card.py', 
                              self.dir_path+'/bin/internal/check_param_card.py')   
                
        #madevent file
        cp(_file_path+'/__init__.py', self.dir_path+'/bin/internal/__init__.py')
        cp(_file_path+'/various/lhe_parser.py', 
                                self.dir_path+'/bin/internal/lhe_parser.py')         

        cp(_file_path+'/various/banner.py', 
                                   self.dir_path+'/bin/internal/banner.py')
        cp(_file_path+'/various/shower_card.py', 
                                   self.dir_path+'/bin/internal/shower_card.py')
        cp(_file_path+'/various/cluster.py', 
                                       self.dir_path+'/bin/internal/cluster.py') 
        
        # logging configuration
        cp(_file_path+'/interface/.mg5_logging.conf', 
                                 self.dir_path+'/bin/internal/me5_logging.conf') 
        cp(_file_path+'/interface/coloring_logging.py', 
                                 self.dir_path+'/bin/internal/coloring_logging.py')


    #===========================================================================
    # Change the version of cuts.f to the one compatible with MW
    #===========================================================================    
    def get_mw_cuts_version(self, outpath=None):
        """create the appropriate cuts.f
        This is based on the one associated to ME output but:
        1) No clustering (=> remove initcluster/setclscales)
        2) Adding the definition of cut_bw at the file.
        """
        
        template = open(pjoin(MG5DIR,'Template','LO','SubProcesses','cuts.f'))
        
        text = StringIO()
        #1) remove all dependencies in ickkw >1:
        nb_if = 0
        for line in template:
            if 'if(xqcut.gt.0d0' in line:
                nb_if = 1
            if nb_if == 0:
                text.write(line)
                continue
            if re.search(r'if\(.*\)\s*then', line):
                nb_if += 1
            elif 'endif' in line:
                nb_if -= 1
            
        #2) add fake cut_bw (have to put the true one later)
        text.write("""
      logical function cut_bw(p)
      include 'madweight_param.inc'
      double precision p(*)
      if (bw_cut) then
          cut_bw = .true.
      else
          stop 1
      endif
      return
      end
        """)
            
        final = text.getvalue()
        #3) remove the call to initcluster:
        template = final.replace('call initcluster', '! Remove for MW!call initcluster')
        template = template.replace('genps.inc', 'maxparticles.inc')
        #Now we can write it
        if not outpath:
            fsock =  open(pjoin(self.dir_path, 'SubProcesses', 'cuts.f'), 'w')
        elif isinstance(outpath, str):
            fsock = open(outpath, 'w')
        else:
            fsock = outpath
        fsock.write(template)
        
        
        
    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS, MODEL, PDF and CERNLIB directories, to set up
        everything for running madweight
        """

        source_dir = os.path.join(self.dir_path, "Source")
        logger.info("Running make for Helas")
        misc.compile(arg=['../lib/libdhelas.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for Model")
        misc.compile(arg=['../lib/libmodel.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for PDF")
        misc.compile(arg=['../lib/libpdf.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for CERNLIB")
        misc.compile(arg=['../lib/libcernlib.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for GENERIC")
        misc.compile(arg=['../lib/libgeneric.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for blocks")
        misc.compile(arg=['../lib/libblocks.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for tools")
        misc.compile(arg=['../lib/libtools.a'], cwd=source_dir, mode='fortran')

    #===========================================================================
    # Create proc_card_mg5.dat for MadWeight directory
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """Finalize Standalone MG4 directory by generation proc_card_mg5.dat"""
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}



        #proc_charac
        self.create_proc_charac()

        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements)
        ln(pjoin(self.dir_path, 'Source', 'maxparticles.inc'),
           pjoin(self.dir_path, 'Source','MadWeight','blocks'))
        ln(pjoin(self.dir_path, 'Source', 'maxparticles.inc'),
           pjoin(self.dir_path, 'Source','MadWeight','tools'))
        
        self.set_compiler(compiler)
        self.make()
        
        # Write command history as proc_card_mg5
        if os.path.isdir(os.path.join(self.dir_path, 'Cards')):
            output_file = os.path.join(self.dir_path, 'Cards', 'proc_card_mg5.dat')
            history.write(output_file)

        ProcessExporterFortran.finalize(self, matrix_elements,
                                             history, mg5options, flaglist)



    #===========================================================================
    # create the run_card for MW
    #=========================================================================== 
    def create_run_card(self, matrix_elements, history):
        """ """
 
        run_card = banner_mod.RunCard()
    
        # pass to default for MW
        run_card["run_tag"] = "\'not_use\'"
        run_card["fixed_ren_scale"] = "T"
        run_card["fixed_fac_scale"] = "T"
        run_card.remove_all_cut()
                  
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'),
                       template=pjoin(MG5DIR, 'Template', 'MadWeight', 'Cards', 'run_card.dat'),
                       python_template=True)
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card.dat'),
                       template=pjoin(MG5DIR, 'Template', 'MadWeight', 'Cards', 'run_card.dat'),
                       python_template=True)

    #===========================================================================
    # export model files
    #=========================================================================== 
    def export_model_files(self, model_path):
        """export the model dependent files for V4 model"""
        
        super(ProcessExporterFortranMW,self).export_model_files(model_path)
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO  
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(os.path.join(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()

        # Modify setrun.f
        text = open(os.path.join(self.dir_path,'Source','setrun.f')).read()
        text = text.replace('call setpara(param_card_name)', 'call setpara(param_card_name, .true.)')
        fsock = open(os.path.join(self.dir_path,'Source','setrun.f'), 'w')
        fsock.write(text)
        fsock.close()

        # Modify initialization.f
        text = open(os.path.join(self.dir_path,'SubProcesses','initialization.f')).read()
        text = text.replace('call setpara(param_name)', 'call setpara(param_name, .true.)')
        fsock = open(os.path.join(self.dir_path,'SubProcesses','initialization.f'), 'w')
        fsock.write(text)
        fsock.close()
                
                
        self.make_model_symbolic_link()

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,number):
        """Generate the Pxxxxx directory for a subprocess in MG4 MadWeight format,
        including the necessary matrix.f and nexternal.inc files"""

        cwd = os.getcwd()
        misc.sprint(type(matrix_element))
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = os.path.join(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        #try:
        #    os.chdir(dirpath)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % dirpath)
        #    return 0

        logger.info('Creating files in directory %s' % dirpath)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Create the matrix.f file and the nexternal.inc file
        filename = pjoin(dirpath,'matrix.f')
        calls,ncolor = self.write_matrix_element_v4(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model)

        filename = pjoin(dirpath, 'auto_dsig.f')
        self.write_auto_dsig_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(dirpath, 'configs.inc')
        mapconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            matrix_element)

        filename = pjoin(dirpath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(dirpath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(dirpath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        filename = pjoin(dirpath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(dirpath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        filename = pjoin(dirpath, 'maxamps.inc')
        self.write_maxamps_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           ncolor,
                           len(matrix_element.get('processes')),
                           1)

        filename = pjoin(dirpath, 'phasespace.inc')
        self.write_phasespace_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           )

        # Generate diagrams
        filename = pjoin(dirpath, "matrix.ps")
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                             get('diagrams'),
                                          filename,
                                          model=matrix_element.get('processes')[0].\
                                             get('model'),
                                          amplitude='')
        logger.info("Generating Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        #import genps.inc and maxconfigs.inc into Subprocesses
        ln(self.dir_path + '/Source/genps.inc', self.dir_path + '/SubProcesses', log=False)
        #ln(self.dir_path + '/Source/maxconfigs.inc', self.dir_path + '/SubProcesses', log=False)

        linkfiles = ['driver.f', 'cuts.f', 'initialization.f','gen_ps.f', 'makefile', 'coupl.inc','madweight_param.inc', 'run.inc', 'setscales.f', 'genps.inc']

        for file in linkfiles:
            ln('../%s' % file, starting_dir=cwd)
            
        ln('nexternal.inc', '../../Source', log=False, cwd=dirpath)
        ln('leshouche.inc', '../../Source', log=False, cwd=dirpath)
        ln('maxamps.inc', '../../Source', log=False, cwd=dirpath)
        ln('phasespace.inc', '../', log=True, cwd=dirpath)
        # Return to original PWD
        #os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,proc_id = "", config_map = []):
        """Export a matrix element to a matrix.f file in MG4 MadWeight format"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if writer:
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")

            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)

        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        
        replace_dict['template_file'] =  os.path.join(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = ''
        
        if writer:
            file = open(replace_dict['template_file']).read()
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
            return len(filter(lambda call: call.find('#') != 0, helas_calls)),ncolor
        else:
            replace_dict['return_value'] = (len(filter(lambda call: call.find('#') != 0, helas_calls)),ncolor)
            
    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer):
        """Write the nexternal.inc file for madweight"""


        path = os.path.join(_file_path,'iolibs','template_files','madweight_makefile_source')
        set_of_lib = '$(LIBRARIES) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext) $(LIBDIR)libtf.$(libext)'
        text = open(path).read() % {'libraries': set_of_lib}
        writer.write(text)

        return True

    def write_phasespace_file(self, writer, nb_diag):
        """ """
        
        template = """      include 'maxparticles.inc' 
      integer max_branches
      parameter (max_branches=max_particles-1)
      integer max_configs
      parameter (max_configs=%(nb_diag)s)

c     channel position
      integer config_pos,perm_pos
      common /to_config/config_pos,perm_pos
        
        """

        writer.write(template % {'nb_diag': nb_diag})
        

    #===========================================================================
    # write_auto_dsig_file
    #===========================================================================
    def write_auto_dsig_file(self, writer, matrix_element, proc_id = ""):
        """Write the auto_dsig.f file for the differential cross section
        calculation, includes pdf call information (MadWeight format)"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()

        if ninitial < 1 or ninitial > 2:
            raise writers.FortranWriter.FortranWriterError, \
                  """Need ninitial = 1 or 2 to write auto_dsig file"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id
        replace_dict['numproc'] = 1

        # Set dsig_line
        if ninitial == 1:
            # No conversion, since result of decay should be given in GeV
            dsig_line = "pd(0)*dsiguu"
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(0)*conv*dsiguu"

        replace_dict['dsig_line'] = dsig_line

        # Extract pdf lines
        pdf_vars, pdf_data, pdf_lines = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines

        # Lines that differ between subprocess group and regular
        if proc_id:
            replace_dict['numproc'] = int(proc_id)
            replace_dict['passcuts_begin'] = "" 
            replace_dict['passcuts_end'] = "" 
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_subdiag_lines'] = \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
        else:
            replace_dict['passcuts_begin'] = "IF (PASSCUTS(PP)) THEN"
            replace_dict['passcuts_end'] = "ENDIF"
            replace_dict['define_subdiag_lines'] = "" 

        if writer:
            file = open(os.path.join(_file_path, \
                          'iolibs/template_files/auto_dsig_mw.inc')).read()
        
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        configs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        model = matrix_element.get('processes')[0].get('model')
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,matrix_element, model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadWeight"""

        path = os.path.join(_file_path,'iolibs','template_files','madweight_run_config.inc')
        text = open(path).read() % {'chanperjob':'5'}
        writer.write(text)
        return True

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, matrix_element, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""

        lines = []

        particle_dict = matrix_element.get('processes')[0].get('model').\
                        get('particle_dict')

        s_and_t_channels = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
                       for config in configs if [d for d in config if d][0].\
                                                  get_vertex_leg_numbers()!=[]]
        
        minvert = min(vert_list) if vert_list!=[] else 0
        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any([vert > minvert for vert in
                    [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
                # Only 3-vertices allowed in configs.inc
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial,model,new_pdg))
                else:
                    stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = zip(*[s for s,t in stchannels])
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            #lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("*     %d       %d " % (nconfigs,
                                                     mapconfigs[iconfig]))

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                line=str(last_leg.get('number'))+" "+str(daughters[0])+"  "+str(daughters[1])
#                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
#                             (last_leg.get('number'), nconfigs, len(daughters),
#                              ",".join([str(d) for d in daughters])))

                if last_leg.get('id') == 21 and 21 not in particle_dict:
                    # Fake propagator used in multiparticle vertices
                    mass = 'zero'
                    width = 'zero'
                    pow_part = 0
                else:
                    if (last_leg.get('id')!=7):
                      particle = particle_dict[last_leg.get('id')]
                      # Get mass
                      mass = particle.get('mass')
                      # Get width
                      width = particle.get('width')
                    else : # fake propagator used in multiparticle vertices
                      mass= 'zero'
                      width= 'zero'

                line=line+"   "+mass+"  "+width+"   "

                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append(line+" S "+str(last_leg.get('id')))
#                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
#                                 (last_leg.get('number'), nconfigs, nsubprocs,
#                                  ",".join([str(d) for d in pdgs])))
#                    lines.append("data tprid(%d,%d)/0/" % \
#                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels[:-1]:
                    lines.append(line+" T "+str(last_leg.get('id')))
#		    lines.append("data tprid(%d,%d)/%d/" % \
#                                 (last_leg.get('number'), nconfigs,
#                                  abs(last_leg.get('id'))))
#                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
#                                 (last_leg.get('number'), nconfigs, nsubprocs,
#                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
#        lines.append("# Number of configs")
#        lines.append("data mapconfig(0)/%d/" % nconfigs)
        lines.append(" *    ")  # a line with just a star indicates this is the end of file
        # Write the file
        writer.writelines(lines)

        return s_and_t_channels


#===============================================================================
# ProcessExporterFortranME
#===============================================================================
class ProcessExporterFortranME(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadEvent format."""

    matrix_file = "matrix_madevent_v4.inc"

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranME, self).copy_template(model)
        
        # File created from Template (Different in some child class)
        filename = pjoin(self.dir_path,'Source','run_config.inc')
        self.write_run_config_file(writers.FortranWriter(filename))
        
        # The next file are model dependant (due to SLAH convention)
        self.model_name = model.get('name')
        # Add the symmetry.f 
        filename = pjoin(self.dir_path,'SubProcesses','symmetry.f')
        self.write_symmetry(writers.FortranWriter(filename))
        #
        filename = pjoin(self.dir_path,'SubProcesses','addmothers.f')
        self.write_addmothers(writers.FortranWriter(filename))
        # Copy the different python file in the Template
        self.copy_python_file()
        
        



    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================        
    def copy_python_file(self):
        """copy the python file require for the Template"""

        # madevent interface
        cp(_file_path+'/interface/madevent_interface.py',
                            self.dir_path+'/bin/internal/madevent_interface.py')
        cp(_file_path+'/interface/extended_cmd.py',
                                  self.dir_path+'/bin/internal/extended_cmd.py')
        cp(_file_path+'/interface/common_run_interface.py',
                            self.dir_path+'/bin/internal/common_run_interface.py')
        cp(_file_path+'/various/misc.py', self.dir_path+'/bin/internal/misc.py')        
        cp(_file_path+'/iolibs/files.py', self.dir_path+'/bin/internal/files.py')
        cp(_file_path+'/iolibs/save_load_object.py', 
                              self.dir_path+'/bin/internal/save_load_object.py') 
        cp(_file_path+'/iolibs/file_writers.py', 
                              self.dir_path+'/bin/internal/file_writers.py')
        #model file                        
        cp(_file_path+'../models/check_param_card.py', 
                              self.dir_path+'/bin/internal/check_param_card.py')   
        
        #copy all the file present in madevent directory
        for name in os.listdir(pjoin(_file_path, 'madevent')):
            if name not in ['__init__.py'] and name.endswith('.py'):
                cp(_file_path+'/madevent/'+name, self.dir_path+'/bin/internal/')
        
        #madevent file
        cp(_file_path+'/__init__.py', self.dir_path+'/bin/internal/__init__.py')
        cp(_file_path+'/various/lhe_parser.py', 
                                self.dir_path+'/bin/internal/lhe_parser.py')                        
        cp(_file_path+'/various/banner.py', 
                                   self.dir_path+'/bin/internal/banner.py')
        cp(_file_path+'/various/histograms.py', 
                                   self.dir_path+'/bin/internal/histograms.py')
        cp(_file_path+'/various/plot_djrs.py', 
                                   self.dir_path+'/bin/internal/plot_djrs.py')
        cp(_file_path+'/various/systematics.py', self.dir_path+'/bin/internal/systematics.py')        

        cp(_file_path+'/various/cluster.py', 
                                       self.dir_path+'/bin/internal/cluster.py') 
        cp(_file_path+'/madevent/combine_runs.py', 
                                       self.dir_path+'/bin/internal/combine_runs.py')
        # logging configuration
        cp(_file_path+'/interface/.mg5_logging.conf', 
                                 self.dir_path+'/bin/internal/me5_logging.conf') 
        cp(_file_path+'/interface/coloring_logging.py', 
                                 self.dir_path+'/bin/internal/coloring_logging.py')
        # shower card and FO_analyse_card. 
        #  Although not needed, it is imported by banner.py
        cp(_file_path+'/various/shower_card.py', 
                                 self.dir_path+'/bin/internal/shower_card.py') 
        cp(_file_path+'/various/FO_analyse_card.py', 
                                 self.dir_path+'/bin/internal/FO_analyse_card.py') 
 
 
    def convert_model(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
         
        super(ProcessExporterFortranME,self).convert_model(model, 
                                               wanted_lorentz, wanted_couplings)
         
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
        # This is not safe if there is a '##' or '-' in the path.
        shutil.copytree(model_path, 
                               pjoin(self.dir_path,'bin','internal','ufomodel'),
                               ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
        if hasattr(model, 'restrict_card'):
            out_path = pjoin(self.dir_path, 'bin', 'internal','ufomodel',
                                                         'restrict_default.dat')
            if isinstance(model.restrict_card, check_param_card.ParamCard):
                model.restrict_card.write(out_path)
            else:
                files.cp(model.restrict_card, out_path)
                
    #===========================================================================
    # export model files
    #=========================================================================== 
    def export_model_files(self, model_path):
        """export the model dependent files"""

        super(ProcessExporterFortranME,self).export_model_files(model_path)
        
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO 
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(pjoin(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()
                
        # Add the symmetry.f 
        filename = pjoin(self.dir_path,'SubProcesses','symmetry.f')
        self.write_symmetry(writers.FortranWriter(filename), v5=False)
        
        # Modify setrun.f
        text = open(pjoin(self.dir_path,'Source','setrun.f')).read()
        text = text.replace('call setpara(param_card_name)', 'call setpara(param_card_name, .true.)')
        fsock = open(pjoin(self.dir_path,'Source','setrun.f'), 'w')
        fsock.write(text)
        fsock.close()
        
        self.make_model_symbolic_link()

    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,
                                         me_number):
        """Generate the Pxxxxx directory for a subprocess in MG4 madevent,
        including the necessary matrix.f and various helper files"""

        cwd = os.getcwd()
        path = pjoin(self.dir_path, 'SubProcesses')


        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')



        #os.chdir(path)
        # Create the directory PN_xx_xxxxx in the specified path
        subprocdir = "P%s" % matrix_element.get('processes')[0].shell_string()
        try:
            os.mkdir(pjoin(path,subprocdir))
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)

        #try:
        #    os.chdir(subprocdir)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % subprocdir)
        #    return 0

        logger.info('Creating files in directory %s' % subprocdir)
        Ppath = pjoin(path, subprocdir)
        
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Add the driver.f 
        ncomb = matrix_element.get_helicity_combinations()
        filename = pjoin(Ppath,'driver.f')
        self.write_driver(writers.FortranWriter(filename),ncomb,n_grouped_proc=1,
                          v5=self.opt['v5_model'])

        # Create the matrix.f file, auto_dsig.f file and all inc files
        filename = pjoin(Ppath, 'matrix.f')
        calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename),
                      matrix_element, fortran_model, subproc_number = me_number)

        filename = pjoin(Ppath, 'auto_dsig.f')
        self.write_auto_dsig_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(Ppath, 'configs.inc')
        mapconfigs, (s_and_t_channels, nqcd_list) = self.write_configs_file(\
            writers.FortranWriter(filename),
            matrix_element)

        filename = pjoin(Ppath, 'config_nqcd.inc')
        self.write_config_nqcd_file(writers.FortranWriter(filename),
                               nqcd_list)

        filename = pjoin(Ppath, 'config_subproc_map.inc')
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           s_and_t_channels)

        filename = pjoin(Ppath, 'coloramps.inc')
        self.write_coloramps_file(writers.FortranWriter(filename),
                             mapconfigs,
                             matrix_element)

        filename = pjoin(Ppath, 'get_color.f')
        self.write_colors_file(writers.FortranWriter(filename),
                               matrix_element)

        filename = pjoin(Ppath, 'decayBW.inc')
        self.write_decayBW_file(writers.FortranWriter(filename),
                           s_and_t_channels)

        filename = pjoin(Ppath, 'dname.mg')
        self.write_dname_file(writers.FileWriter(filename),
                         "P"+matrix_element.get('processes')[0].shell_string())

        filename = pjoin(Ppath, 'iproc.dat')
        self.write_iproc_file(writers.FortranWriter(filename),
                         me_number)

        filename = pjoin(Ppath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(Ppath, 'maxamps.inc')
        self.write_maxamps_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           ncolor,
                           len(matrix_element.get('processes')),
                           1)

        filename = pjoin(Ppath, 'mg.sym')
        self.write_mg_sym_file(writers.FortranWriter(filename),
                          matrix_element)

        filename = pjoin(Ppath, 'ncombs.inc')
        self.write_ncombs_file(writers.FortranWriter(filename),
                          nexternal)

        filename = pjoin(Ppath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(Ppath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(mapconfigs))


        filename = pjoin(Ppath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(Ppath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        # Find config symmetries and permutations
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(matrix_element)

        filename = pjoin(Ppath, 'symswap.inc')
        self.write_symswap_file(writers.FortranWriter(filename),
                                ident_perms)

        filename = pjoin(Ppath, 'symfact_orig.dat')
        self.write_symfact_file(open(filename, 'w'), symmetry)

        # Generate diagrams
        filename = pjoin(Ppath, "matrix.ps")
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                             get('diagrams'),
                                          filename,
                                          model=matrix_element.get('processes')[0].\
                                             get('model'),
                                          amplitude=True)
        logger.info("Generating Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        self.link_files_in_SubProcess(Ppath)

        #import nexternal/leshouche in Source
        ln(pjoin(Ppath,'nexternal.inc'), pjoin(self.dir_path,'Source'), log=False)
        ln(pjoin(Ppath,'leshouche.inc'),  pjoin(self.dir_path,'Source'), log=False)
        ln(pjoin(Ppath,'maxamps.inc'),  pjoin(self.dir_path,'Source'), log=False)
        # Return to SubProcesses dir
        #os.chdir(os.path.pardir)

        # Add subprocess to subproc.mg
        filename = pjoin(path, 'subproc.mg')
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)

        # Return to original dir
        #os.chdir(cwd)

        # Generate info page
        gen_infohtml.make_info_html(self.dir_path)


        if not calls:
            calls = 0
        return calls

    def link_files_in_SubProcess(self, Ppath):
        """ Create the necessary links in the P* directory path Ppath"""
        
        #import genps.inc and maxconfigs.inc into Subprocesses
        ln(self.dir_path + '/Source/genps.inc', 
                                     self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxconfigs.inc',
                                     self.dir_path + '/SubProcesses', log=False)

        linkfiles = ['addmothers.f',
                     'cluster.f',
                     'cluster.inc',
                     'coupl.inc',
                     'cuts.f',
                     'cuts.inc',
                     'genps.f',
                     'genps.inc',
                     'idenparts.f',
                     'initcluster.f',
                     'makefile',
                     'message.inc',
                     'myamp.f',
                     'reweight.f',
                     'run.inc',
                     'maxconfigs.inc',
                     'maxparticles.inc',
                     'run_config.inc',
                     'lhe_event_infos.inc',
                     'setcuts.f',
                     'setscales.f',
                     'sudakov.inc',
                     'symmetry.f',
                     'unwgt.f']

        for file in linkfiles:
            ln('../' + file , cwd=Ppath)    


    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """Finalize ME v4 directory by creating jpeg diagrams, html
        pages,proc_card_mg5.dat and madevent.tar.gz."""
        
        if 'nojpeg' in flaglist:
            makejpg = False
        else:
            makejpg = True
        if 'online' in flaglist:
            online = True
        else:
            online = False
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}

        # indicate that the output type is not grouped
        if  not isinstance(self, ProcessExporterFortranMEGroup):
            self.proc_characteristic['grouped_matrix'] = False
        self.proc_characteristic['complex_mass_scheme'] = mg5options['complex_mass_scheme']

        modelname = self.opt['model']
        if modelname == 'mssm' or modelname.startswith('mssm-'):
            param_card = pjoin(self.dir_path, 'Cards','param_card.dat')
            mg5_param = pjoin(self.dir_path, 'Source', 'MODEL', 'MG5_param.dat')
            check_param_card.convert_to_mg5card(param_card, mg5_param)
            check_param_card.check_valid_param_card(mg5_param)

        # Add the combine_events.f modify param_card path/number of @X
        filename = pjoin(self.dir_path,'Source','combine_events.f')
        try:
            nb_proc =[p.get('id') for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes')]
        except AttributeError:
            nb_proc =[p.get('id') for m in matrix_elements.get('matrix_elements') for p in m.get('processes')]
        nb_proc = len(set(nb_proc))
        self.write_combine_events(writers.FortranWriter(filename), nb_proc) # already formatted
        # Write maxconfigs.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxconfigs.inc')
        self.write_maxconfigs_file(writers.FortranWriter(filename),
                                   matrix_elements)
        
        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements)
        
        # Touch "done" file
        os.system('touch %s/done' % pjoin(self.dir_path,'SubProcesses'))

        # Check for compiler
        self.set_compiler(compiler)
        self.set_cpp_compiler(compiler['cpp'])
        

        old_pos = os.getcwd()
        subpath = pjoin(self.dir_path, 'SubProcesses')

        P_dir_list = [proc for proc in os.listdir(subpath) 
                      if os.path.isdir(pjoin(subpath,proc)) and proc[0] == 'P']

        devnull = os.open(os.devnull, os.O_RDWR)
        # Convert the poscript in jpg files (if authorize)
        if makejpg:
            try:
                os.remove(pjoin(self.dir_path,'HTML','card.jpg'))
            except Exception, error:
                pass
            logger.info("Generate jpeg diagrams")
            for Pdir in P_dir_list:
                misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_jpeg-pl')],
                                stdout = devnull, cwd=pjoin(subpath, Pdir))

        logger.info("Generate web pages")
        # Create the WebPage using perl script

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')], \
                                      stdout = devnull,cwd=pjoin(self.dir_path))

        #os.chdir(os.path.pardir)

        obj = gen_infohtml.make_info_html(self.dir_path)
              
        if online:
            nb_channel = obj.rep_rule['nb_gen_diag']
            open(pjoin(self.dir_path, 'Online'),'w').write(str(nb_channel))
        #add the information to proc_charac
        self.proc_characteristic['nb_channel'] = obj.rep_rule['nb_gen_diag']
        
        # Write command history as proc_card_mg5
        if os.path.isdir(pjoin(self.dir_path,'Cards')):
            output_file = pjoin(self.dir_path,'Cards', 'proc_card_mg5.dat')
            history.write(output_file)

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull)

        #crate the proc_characteristic file 
        self.create_proc_charac(matrix_elements, history)

        # create the run_card
        ProcessExporterFortran.finalize(self, matrix_elements, history, mg5options, flaglist)

        # Run "make" to generate madevent.tar.gz file
        if os.path.exists(pjoin(self.dir_path,'SubProcesses', 'subproc.mg')):
            if os.path.exists(pjoin(self.dir_path,'madevent.tar.gz')):
                os.remove(pjoin(self.dir_path,'madevent.tar.gz'))
            misc.call([os.path.join(self.dir_path, 'bin', 'internal', 'make_madevent_tar')],
                        stdout = devnull, cwd=self.dir_path)

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull, cwd=self.dir_path)






        #return to the initial dir
        #os.chdir(old_pos)               

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                           proc_id = "", config_map = [], subproc_number = ""):
        """Export a matrix element to a matrix.f file in MG4 madevent format"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if writer: 
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False

        # The proc prefix is not used for MadEvent output so it can safely be set
        # to an empty string.
        replace_dict = {'proc_prefix':''}

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)

        replace_dict['helas_calls'] = "\n".join(helas_calls)


        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract IC line
        ic_line = self.get_ic_line(matrix_element)
        replace_dict['ic_line'] = ic_line

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract ndiags
        ndiags = len(matrix_element.get('diagrams'))
        replace_dict['ndiags'] = ndiags

        # Set define_iconfigs_lines
        replace_dict['define_iconfigs_lines'] = \
             """INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
             COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG"""

        if proc_id:
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_iconfigs_lines'] += \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
            # Set set_amp2_line
            replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(SUBDIAG(%s))/XTOT" % \
                                            proc_id
        else:
            # Standard running
            # Set set_amp2_line
            replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT"

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)


        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 18
        else:
            replace_dict['wavefunctionsize'] = 6

        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element, config_map)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)

        # The JAMP definition depends on the splitting order
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)>0:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['chosen_so_configs']=self.set_chosen_SO_index(
                              matrix_element.get('processes')[0],squared_orders)
        else:
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            
        replace_dict['nAmpSplitOrders']=len(amp_orders)
        replace_dict['nSqAmpSplitOrders']=len(squared_orders)
        replace_dict['split_order_str_list']=str(split_orders)
        replace_dict['nSplitOrders']=max(len(split_orders),1)
        amp_so = self.get_split_orders_lines(
                [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
        sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
        replace_dict['ampsplitorders']='\n'.join(amp_so)
        replace_dict['sqsplitorders']='\n'.join(sqamp_so)
        
        
        # Extract JAMP lines
        # If no split_orders then artificiall add one entry called 'ALL_ORDERS'
        jamp_lines = self.get_JAMP_lines_split_order(\
                             matrix_element,amp_orders,split_order_names=
                        split_orders if len(split_orders)>0 else ['ALL_ORDERS'])
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        replace_dict['template_file'] = pjoin(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = pjoin(_file_path, \
                          'iolibs/template_files/split_orders_helping_functions.inc')      
        if writer:
            file = open(replace_dict['template_file']).read()
            file = file % replace_dict
            # Add the split orders helper functions.
            file = file + '\n' + open(replace_dict['template_file2'])\
                                                            .read()%replace_dict
            # Write the file
            writer.writelines(file)
            return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor
        else:
            replace_dict['return_value'] = (len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor)
            return replace_dict
        
    #===========================================================================
    # write_auto_dsig_file
    #===========================================================================
    def write_auto_dsig_file(self, writer, matrix_element, proc_id = ""):
        """Write the auto_dsig.f file for the differential cross section
        calculation, includes pdf call information"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        self.proc_characteristic['ninitial'] = ninitial
        self.proc_characteristic['nexternal'] = max(self.proc_characteristic['nexternal'], nexternal)

        # Add information relevant for MLM matching:
        # Maximum QCD power in all the contributions
        max_qcd_order = 0
        for diag in matrix_element.get('diagrams'):
            orders = diag.calculate_orders()
            if 'QCD' in orders:
                max_qcd_order = max(max_qcd_order,orders['QCD'])
        max_n_light_final_partons = max(len([1 for id in proc.get_final_ids() 
            if proc.get('model').get_particle(id).get('mass')=='ZERO' and
               proc.get('model').get_particle(id).get('color')>1])
                                    for proc in matrix_element.get('processes'))
        # Maximum number of final state light jets to be matched
        self.proc_characteristic['max_n_matched_jets'] = max(
                               self.proc_characteristic['max_n_matched_jets'],
                                   min(max_qcd_order,max_n_light_final_partons))

        # List of default pdgs to be considered for the CKKWl merging cut
        self.proc_characteristic['colored_pdgs'] = \
          sorted(list(set([abs(p.get('pdg_code')) for p in
            matrix_element.get('processes')[0].get('model').get('particles') if
                                                           p.get('color')>1])))

        if ninitial < 1 or ninitial > 2:
            raise writers.FortranWriter.FortranWriterError, \
                  """Need ninitial = 1 or 2 to write auto_dsig file"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id
        replace_dict['numproc'] = 1

        # Set dsig_line
        if ninitial == 1:
            # No conversion, since result of decay should be given in GeV
            dsig_line = "pd(0)*dsiguu"
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(0)*conv*dsiguu"

        replace_dict['dsig_line'] = dsig_line

        # Extract pdf lines
        pdf_vars, pdf_data, pdf_lines = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines

        # Lines that differ between subprocess group and regular
        if proc_id:
            replace_dict['numproc'] = int(proc_id)
            replace_dict['passcuts_begin'] = ""
            replace_dict['passcuts_end'] = ""
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_subdiag_lines'] = \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
            replace_dict['cutsdone'] = ""
        else:
            replace_dict['passcuts_begin'] = "IF (PASSCUTS(PP)) THEN"
            replace_dict['passcuts_end'] = "ENDIF"
            replace_dict['define_subdiag_lines'] = ""
            replace_dict['cutsdone'] = "      cutsdone=.false.\n       cutspassed=.false."

        if not isinstance(self, ProcessExporterFortranMEGroup):
            ncomb=matrix_element.get_helicity_combinations()
            replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)
        else:
            replace_dict['read_write_good_hel'] = ""
        
        context = {'read_write_good_hel':True}
        
        if writer:
            file = open(pjoin(_file_path, \
                          'iolibs/template_files/auto_dsig_v4.inc')).read()
            file = file % replace_dict

            # Write the file
            writer.writelines(file, context=context)
        else:
            return replace_dict, context
    #===========================================================================
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, mapconfigs, matrix_element):
        """Write the coloramps.inc file for MadEvent"""

        lines = self.get_icolamp_lines(mapconfigs, matrix_element, 1)
        lines.insert(0, "logical icolamp(%d,%d,1)" % \
                        (max(len(matrix_element.get('color_basis').keys()), 1),
                         len(mapconfigs)))


        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_colors_file
    #===========================================================================
    def write_colors_file(self, writer, matrix_elements):
        """Write the get_color.f file for MadEvent, which returns color
        for all particles used in the matrix element."""

        if isinstance(matrix_elements, helas_objects.HelasMatrixElement):
            matrix_elements = [matrix_elements]

        model = matrix_elements[0].get('processes')[0].get('model')

        # We need the both particle and antiparticle wf_ids, since the identity
        # depends on the direction of the wf.
        wf_ids = set(sum([sum([sum([[wf.get_pdg_code(),wf.get_anti_pdg_code()] \
                                    for wf in d.get('wavefunctions')],[]) \
                               for d in me.get('diagrams')], []) \
                          for me in matrix_elements], []))

        leg_ids = set(sum([sum([sum([[l.get('id'), 
                          model.get_particle(l.get('id')).get_anti_pdg_code()] \
                                  for l in p.get_legs_with_decays()], []) \
                                for p in me.get('processes')], []) \
                           for me in matrix_elements], []))
        particle_ids = sorted(list(wf_ids.union(leg_ids)))

        lines = """function get_color(ipdg)
        implicit none
        integer get_color, ipdg

        if(ipdg.eq.%d)then
        get_color=%d
        return
        """ % (particle_ids[0], model.get_particle(particle_ids[0]).get_color())

        for part_id in particle_ids[1:]:
            lines += """else if(ipdg.eq.%d)then
            get_color=%d
            return
            """ % (part_id, model.get_particle(part_id).get_color())
        # Dummy particle for multiparticle vertices with pdg given by
        # first code not in the model
        lines += """else if(ipdg.eq.%d)then
c           This is dummy particle used in multiparticle vertices
            get_color=2
            return
            """ % model.get_first_non_pdg()
        lines += """else
        write(*,*)'Error: No color given for pdg ',ipdg
        get_color=0        
        return
        endif
        end
        """
        
        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_config_nqcd_file
    #===========================================================================
    def write_config_nqcd_file(self, writer, nqcd_list):
        """Write the config_nqcd.inc with the number of QCD couplings
        for each config"""

        lines = []
        for iconf, n in enumerate(nqcd_list):
            lines.append("data nqcd(%d)/%d/" % (iconf+1, n))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_maxconfigs_file
    #===========================================================================
    def write_maxconfigs_file(self, writer, matrix_elements):
        """Write the maxconfigs.inc file for MadEvent"""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            maxconfigs = max([me.get_num_configs() for me in \
                              matrix_elements.get('matrix_elements')])
        else:
            maxconfigs = max([me.get_num_configs() for me in matrix_elements])

        lines = "integer lmaxconfigs\n"
        lines += "parameter(lmaxconfigs=%d)" % maxconfigs

        # Write the file
        writer.writelines(lines)

        return True
    
    #===========================================================================
    # read_write_good_hel
    #===========================================================================
    def read_write_good_hel(self, ncomb):
        """return the code to read/write the good_hel common_block"""    

        convert = {'ncomb' : ncomb}
        output = """
        subroutine write_good_hel(stream_id)
        implicit none
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB)
        INTEGER NTRY
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        write(stream_id,*) GOODHEL
        return
        end
        
        
        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB)
        INTEGER NTRY
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        read(stream_id,*) GOODHEL
        NTRY = MAXTRIES + 1
        return
        end 
        
        subroutine init_good_hel()
        implicit none
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB)        
        INTEGER NTRY
        INTEGER I
        
        do i=1,NCOMB
            GOODHEL(I) = .false.
        enddo
        NTRY = 0
        end
        
        integer function get_maxsproc()
        implicit none
        get_maxsproc = 1
        return 
        end
        
        """ % convert
        
        return output
                                
    #===========================================================================
    # write_config_subproc_map_file
    #===========================================================================
    def write_config_subproc_map_file(self, writer, s_and_t_channels):
        """Write a dummy config_subproc.inc file for MadEvent"""

        lines = []

        for iconfig in range(len(s_and_t_channels)):
            lines.append("DATA CONFSUB(1,%d)/1/" % \
                         (iconfig + 1))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        model = matrix_element.get('processes')[0].get('model')
        configs = [(i+1, d) for (i, d) in \
                       enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,
                                                            model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_run_config.inc')
        
        if self.proc_characteristic['loop_induced']:
            job_per_chan = 1
        else: 
           job_per_chan = 5
        
        if writer:
            text = open(path).read() % {'chanperjob': job_per_chan} 
            writer.write(text)
            return True
        else:
            return {'chanperjob': job_per_chan} 

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""

        lines = []

        s_and_t_channels = []

        nqcd_list = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
                       for config in configs if [d for d in config if d][0].\
                                                  get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any([vert > minvert for vert in
                    [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
                # Only 3-vertices allowed in configs.inc
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial, model,
                                                           new_pdg))
                else:
                    stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = zip(*[s for s,t in stchannels])
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("data mapconfig(%d)/%d/" % (nconfigs,
                                                     mapconfigs[iconfig]))
            # Number of QCD couplings in this diagram
            nqcd = 0
            for h in helas_diags:
                if h:
                    try:
                        nqcd = h.calculate_orders()['QCD']
                    except KeyError:
                        pass
                    break
                else:
                    continue

            nqcd_list.append(nqcd)

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), nconfigs, len(daughters),
                              ",".join([str(d) for d in daughters])))
                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join([str(d) for d in pdgs])))
                    lines.append("data tprid(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels[:-1]:
                    lines.append("data tprid(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  abs(last_leg.get('id'))))
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%d/" % nconfigs)

        # Write the file
        writer.writelines(lines)

        return s_and_t_channels, nqcd_list

    #===========================================================================
    # write_decayBW_file
    #===========================================================================
    def write_decayBW_file(self, writer, s_and_t_channels):
        """Write the decayBW.inc file for MadEvent"""

        lines = []

        booldict = {None: "0", True: "1", False: "2"}

        for iconf, config in enumerate(s_and_t_channels):
            schannels = config[0]
            for vertex in schannels:
                # For the resulting leg, pick out whether it comes from
                # decay or not, as given by the onshell flag
                leg = vertex.get('legs')[-1]
                lines.append("data gForceBW(%d,%d)/%s/" % \
                             (leg.get('number'), iconf + 1,
                              booldict[leg.get('onshell')]))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_dname_file
    #===========================================================================
    def write_dname_file(self, writer, dir_name):
        """Write the dname.mg file for MG4"""

        line = "DIRNAME=%s" % dir_name

        # Write the file
        writer.write(line + "\n")

        return True

    #===========================================================================
    # write_driver
    #===========================================================================
    def write_driver(self, writer, ncomb, n_grouped_proc, v5=True):
        """Write the SubProcess/driver.f file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_driver.f')
        
        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat'
        # Requiring each helicity configuration to be probed by 10 points for 
        # matrix element before using the resulting grid for MC over helicity
        # sampling.
        # We multiply this by 2 because each grouped subprocess is called at most
        # twice for each IMIRROR.
        replace_dict = {'param_card_name':card, 
                        'ncomb':ncomb,
                        'hel_init_points':n_grouped_proc*10*2}
        if not v5:
            replace_dict['secondparam']=',.true.'
        else:
            replace_dict['secondparam']=''            

        if writer:
            text = open(path).read() % replace_dict
            writer.write(text)
            return True
        else:
            return replace_dict

    #===========================================================================
    # write_addmothers
    #===========================================================================
    def write_addmothers(self, writer):
        """Write the SubProcess/addmothers.f"""

        path = pjoin(_file_path,'iolibs','template_files','addmothers.f')

        text = open(path).read() % {'iconfig': 'diag_number'}
        writer.write(text)
        
        return True


    #===========================================================================
    # write_combine_events
    #===========================================================================
    def write_combine_events(self, writer, nb_proc=100):
        """Write the SubProcess/driver.f file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_combine_events.f')
        
        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat' 
        
        #set maxpup (number of @X in the process card)
            
        text = open(path).read() % {'param_card_name':card, 'maxpup':nb_proc+1}
        #the +1 is just a security. This is not needed but I feel(OM) safer with it.
        writer.write(text)
        
        return True


    #===========================================================================
    # write_symmetry
    #===========================================================================
    def write_symmetry(self, writer, v5=True):
        """Write the SubProcess/driver.f file for ME"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_symmetry.f')
        
        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat' 
        
        if v5:
            replace_dict = {'param_card_name':card, 'setparasecondarg':''}      
        else:
            replace_dict= {'param_card_name':card, 'setparasecondarg':',.true.'} 
        
        if writer:
            text = open(path).read() 
            text = text % replace_dict
            writer.write(text)
            return True
        else:
            return replace_dict



    #===========================================================================
    # write_iproc_file
    #===========================================================================
    def write_iproc_file(self, writer, me_number):
        """Write the iproc.dat file for MG4"""
        line = "%d" % (me_number + 1)

        # Write the file
        for line_to_write in writer.write_line(line):
            writer.write(line_to_write)
        return True

    #===========================================================================
    # write_mg_sym_file
    #===========================================================================
    def write_mg_sym_file(self, writer, matrix_element):
        """Write the mg.sym file for MadEvent."""

        lines = []

        # Extract process with all decays included
        final_legs = filter(lambda leg: leg.get('state') == True,
                       matrix_element.get('processes')[0].get_legs_with_decays())

        ninitial = len(filter(lambda leg: leg.get('state') == False,
                              matrix_element.get('processes')[0].get('legs')))

        identical_indices = {}

        # Extract identical particle info
        for i, leg in enumerate(final_legs):
            if leg.get('id') in identical_indices:
                identical_indices[leg.get('id')].append(\
                                    i + ninitial + 1)
            else:
                identical_indices[leg.get('id')] = [i + ninitial + 1]

        # Remove keys which have only one particle
        for key in identical_indices.keys():
            if len(identical_indices[key]) < 2:
                del identical_indices[key]

        # Write mg.sym file
        lines.append(str(len(identical_indices.keys())))
        for key in identical_indices.keys():
            lines.append(str(len(identical_indices[key])))
            for number in identical_indices[key]:
                lines.append(str(number))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_mg_sym_file
    #===========================================================================
    def write_default_mg_sym_file(self, writer):
        """Write the mg.sym file for MadEvent."""

        lines = "0"

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_ncombs_file
    #===========================================================================
    def write_ncombs_file(self, writer, nexternal):
        """Write the ncombs.inc file for MadEvent."""

        # ncomb (used for clustering) is 2^nexternal
        file = "       integer    n_max_cl\n"
        file = file + "parameter (n_max_cl=%d)" % (2 ** nexternal)

        # Write the file
        writer.writelines(file)

        return True

    #===========================================================================
    # write_processes_file
    #===========================================================================
    def write_processes_file(self, writer, subproc_group):
        """Write the processes.dat file with info about the subprocesses
        in this group."""

        lines = []

        for ime, me in \
            enumerate(subproc_group.get('matrix_elements')):
            lines.append("%s %s" % (str(ime+1) + " " * (7-len(str(ime+1))),
                                    ",".join(p.base_string() for p in \
                                             me.get('processes'))))
            if me.get('has_mirror_process'):
                mirror_procs = [copy.copy(p) for p in me.get('processes')]
                for proc in mirror_procs:
                    legs = copy.copy(proc.get('legs_with_decays'))
                    legs.insert(0, legs.pop(1))
                    proc.set("legs_with_decays", legs)
                lines.append("mirror  %s" % ",".join(p.base_string() for p in \
                                                     mirror_procs))
            else:
                lines.append("mirror  none")

        # Write the file
        writer.write("\n".join(lines))

        return True

    #===========================================================================
    # write_symswap_file
    #===========================================================================
    def write_symswap_file(self, writer, ident_perms):
        """Write the file symswap.inc for MG4 by comparing diagrams using
        the internal matrix element value functionality."""

        lines = []

        # Write out lines for symswap.inc file (used to permute the
        # external leg momenta
        for iperm, perm in enumerate(ident_perms):
            lines.append("data (isym(i,%d),i=1,nexternal)/%s/" % \
                         (iperm+1, ",".join([str(i+1) for i in perm])))
        lines.append("data nsym/%d/" % len(ident_perms))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_symfact_file
    #===========================================================================
    def write_symfact_file(self, writer, symmetry):
        """Write the files symfact.dat for MG4 by comparing diagrams using
        the internal matrix element value functionality."""

        pos = max(2, int(math.ceil(math.log10(len(symmetry)))))
        form = "%"+str(pos)+"r %"+str(pos+1)+"r"
        # Write out lines for symswap.inc file (used to permute the
        # external leg momenta
        lines = [ form %(i+1, s) for i,s in enumerate(symmetry) if s != 0] 
        # Write the file
        writer.write('\n'.join(lines))
        writer.write('\n')

        return True

    #===========================================================================
    # write_symperms_file
    #===========================================================================
    def write_symperms_file(self, writer, perms):
        """Write the symperms.inc file for subprocess group, used for
        symmetric configurations"""

        lines = []
        for iperm, perm in enumerate(perms):
            lines.append("data (perms(i,%d),i=1,nexternal)/%s/" % \
                         (iperm+1, ",".join([str(i+1) for i in perm])))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_subproc
    #===========================================================================
    def write_subproc(self, writer, subprocdir):
        """Append this subprocess to the subproc.mg file for MG4"""

        # Write line to file
        writer.write(subprocdir + "\n")

        return True

#===============================================================================
# ProcessExporterFortranMEGroup
#===============================================================================
class ProcessExporterFortranMEGroup(ProcessExporterFortranME):
    """Class to take care of exporting a set of matrix elements to
    MadEvent subprocess group format."""

    matrix_file = "matrix_madevent_group_v4.inc"
    grouped_mode = 'madevent'
    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model,
                                         group_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files."""

        assert isinstance(subproc_group, group_subprocs.SubProcessGroup), \
                                      "subproc_group object not SubProcessGroup"
        
        if not self.model:
            self.model = subproc_group.get('matrix_elements')[0].\
                         get('processes')[0].get('model')

        cwd = os.getcwd()
        path = pjoin(self.dir_path, 'SubProcesses')
        
        os.chdir(path)
        pathdir = os.getcwd()

        # Create the directory PN in the specified path
        subprocdir = "P%d_%s" % (subproc_group.get('number'),
                                 subproc_group.get('name'))
        try:
            os.mkdir(subprocdir)
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)

        try:
            os.chdir(subprocdir)
        except os.error:
            logger.error('Could not cd to directory %s' % subprocdir)
            return 0

        logger.info('Creating files in directory %s' % subprocdir)

        # Create the matrix.f files, auto_dsig.f files and all inc files
        # for all subprocesses in the group

        maxamps = 0
        maxflows = 0
        tot_calls = 0

        matrix_elements = subproc_group.get('matrix_elements')

        # Add the driver.f, all grouped ME's must share the same number of 
        # helicity configuration
        ncomb = matrix_elements[0].get_helicity_combinations()
        for me in matrix_elements[1:]:
            if ncomb!=me.get_helicity_combinations():
                raise MadGraph5Error, "All grouped processes must share the "+\
                                       "same number of helicity configurations."                

        filename = 'driver.f'
        self.write_driver(writers.FortranWriter(filename),ncomb,
                                  n_grouped_proc=len(matrix_elements), v5=self.opt['v5_model'])

        for ime, matrix_element in \
                enumerate(matrix_elements):
            filename = 'matrix%d.f' % (ime+1)
            calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename), 
                            matrix_element,
                            fortran_model,
                            proc_id=str(ime+1),
                            config_map=subproc_group.get('diagram_maps')[ime],
                            subproc_number=group_number)

            filename = 'auto_dsig%d.f' % (ime+1)
            self.write_auto_dsig_file(writers.FortranWriter(filename),
                                 matrix_element,
                                 str(ime+1))

            # Keep track of needed quantities
            tot_calls += int(calls)
            maxflows = max(maxflows, ncolor)
            maxamps = max(maxamps, len(matrix_element.get('diagrams')))

            # Draw diagrams
            filename = "matrix%d.ps" % (ime+1)
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                                    get('diagrams'),
                                              filename,
                                              model = \
                                                matrix_element.get('processes')[0].\
                                                                       get('model'),
                                              amplitude=True)
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Generate a list of diagrams corresponding to each configuration
        # [[d1, d2, ...,dn],...] where 1,2,...,n is the subprocess number
        # If a subprocess has no diagrams for this config, the number is 0
        subproc_diagrams_for_config = subproc_group.get('diagrams_for_configs')

        filename = 'auto_dsig.f'
        self.write_super_auto_dsig_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = 'coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                                   subproc_diagrams_for_config,
                                   maxflows,
                                   matrix_elements)

        filename = 'get_color.f'
        self.write_colors_file(writers.FortranWriter(filename),
                               matrix_elements)

        filename = 'config_subproc_map.inc'
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           subproc_diagrams_for_config)

        filename = 'configs.inc'
        nconfigs, (s_and_t_channels, nqcd_list) = self.write_configs_file(\
            writers.FortranWriter(filename),
            subproc_group,
            subproc_diagrams_for_config)

        filename = 'config_nqcd.inc'
        self.write_config_nqcd_file(writers.FortranWriter(filename),
                                    nqcd_list)

        filename = 'decayBW.inc'
        self.write_decayBW_file(writers.FortranWriter(filename),
                           s_and_t_channels)

        filename = 'dname.mg'
        self.write_dname_file(writers.FortranWriter(filename),
                         subprocdir)

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                         group_number)

        filename = 'leshouche.inc'
        self.write_leshouche_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = 'maxamps.inc'
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           max([len(me.get('processes')) for me in \
                                matrix_elements]),
                           len(matrix_elements))

        # Note that mg.sym is not relevant for this case
        filename = 'mg.sym'
        self.write_default_mg_sym_file(writers.FortranWriter(filename))

        filename = 'mirrorprocs.inc'
        self.write_mirrorprocs(writers.FortranWriter(filename),
                          subproc_group)

        filename = 'ncombs.inc'
        self.write_ncombs_file(writers.FortranWriter(filename),
                          nexternal)

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           nconfigs)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'props.inc'
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        filename = 'processes.dat'
        files.write_to_file(filename,
                            self.write_processes_file,
                            subproc_group)

        # Find config symmetries and permutations
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(subproc_group)

        filename = 'symswap.inc'
        self.write_symswap_file(writers.FortranWriter(filename),
                                ident_perms)

        filename = 'symfact_orig.dat'
        self.write_symfact_file(open(filename, 'w'), symmetry)

        filename = 'symperms.inc'
        self.write_symperms_file(writers.FortranWriter(filename),
                           perms)

        # Generate jpgs -> pass in make_html
        #os.system(pjoin('..', '..', 'bin', 'gen_jpeg-pl'))

        self.link_files_in_SubProcess(pjoin(pathdir,subprocdir))

        #import nexternal/leshouch in Source
        ln('nexternal.inc', '../../Source', log=False)
        ln('leshouche.inc', '../../Source', log=False)
        ln('maxamps.inc', '../../Source', log=False)

        # Return to SubProcesses dir)
        os.chdir(pathdir)

        # Add subprocess to subproc.mg
        filename = 'subproc.mg'
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)
                
        # Return to original dir
        os.chdir(cwd)

        if not tot_calls:
            tot_calls = 0
        return tot_calls

    #===========================================================================
    # write_super_auto_dsig_file
    #===========================================================================
    def write_super_auto_dsig_file(self, writer, subproc_group):
        """Write the auto_dsig.f file selecting between the subprocesses
        in subprocess group mode"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        matrix_elements = subproc_group.get('matrix_elements')

        # Extract process info lines
        process_lines = '\n'.join([self.get_process_info_lines(me) for me in \
                                   matrix_elements])
        replace_dict['process_lines'] = process_lines

        nexternal, ninitial = matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        replace_dict['nsprocs'] = 2*len(matrix_elements)

        # Generate dsig definition line
        dsig_def_line = "DOUBLE PRECISION " + \
                        ",".join(["DSIG%d" % (iproc + 1) for iproc in \
                                  range(len(matrix_elements))])
        replace_dict["dsig_def_line"] = dsig_def_line

        # Generate dsig process lines
        call_dsig_proc_lines = []
        for iproc in range(len(matrix_elements)):
            call_dsig_proc_lines.append(\
                "IF(IPROC.EQ.%(num)d) DSIGPROC=DSIG%(num)d(P1,WGT,IMODE) ! %(proc)s" % \
                {"num": iproc + 1,
                 "proc": matrix_elements[iproc].get('processes')[0].base_string()})
        replace_dict['call_dsig_proc_lines'] = "\n".join(call_dsig_proc_lines)

        ncomb=matrix_elements[0].get_helicity_combinations()
        replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)
        
        if writer:
            file = open(pjoin(_file_path, \
                       'iolibs/template_files/super_auto_dsig_group_v4.inc')).read()
            file = file % replace_dict

            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
        
    #===========================================================================
    # write_mirrorprocs
    #===========================================================================
    def write_mirrorprocs(self, writer, subproc_group):
        """Write the mirrorprocs.inc file determining which processes have
        IS mirror process in subprocess group mode."""

        lines = []
        bool_dict = {True: '.true.', False: '.false.'}
        matrix_elements = subproc_group.get('matrix_elements')
        lines.append("DATA (MIRRORPROCS(I),I=1,%d)/%s/" % \
                     (len(matrix_elements),
                      ",".join([bool_dict[me.get('has_mirror_process')] for \
                                me in matrix_elements])))
        # Write the file
        writer.writelines(lines)

    #===========================================================================
    # write_addmothers
    #===========================================================================
    def write_addmothers(self, writer):
        """Write the SubProcess/addmothers.f"""

        path = pjoin(_file_path,'iolibs','template_files','addmothers.f')

        text = open(path).read() % {'iconfig': 'lconfig'}
        writer.write(text)
        
        return True


    #===========================================================================
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, diagrams_for_config, maxflows,
                                   matrix_elements):
        """Write the coloramps.inc file for MadEvent in Subprocess group mode"""

        # Create a map from subprocess (matrix element) to a list of
        # the diagrams corresponding to each config

        lines = []

        subproc_to_confdiag = {}
        for config in diagrams_for_config:
            for subproc, diag in enumerate(config):
                try:
                    subproc_to_confdiag[subproc].append(diag)
                except KeyError:
                    subproc_to_confdiag[subproc] = [diag]

        for subproc in sorted(subproc_to_confdiag.keys()):
            lines.extend(self.get_icolamp_lines(subproc_to_confdiag[subproc],
                                           matrix_elements[subproc],
                                           subproc + 1))

        lines.insert(0, "logical icolamp(%d,%d,%d)" % \
                        (maxflows,
                         len(diagrams_for_config),
                         len(matrix_elements)))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_config_subproc_map_file
    #===========================================================================
    def write_config_subproc_map_file(self, writer, config_subproc_map):
        """Write the config_subproc_map.inc file for subprocess groups"""

        lines = []
        # Output only configs that have some corresponding diagrams
        iconfig = 0
        for config in config_subproc_map:
            if set(config) == set([0]):
                continue
            lines.append("DATA (CONFSUB(i,%d),i=1,%d)/%s/" % \
                         (iconfig + 1, len(config),
                          ",".join([str(i) for i in config])))
            iconfig += 1
        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # read_write_good_hel
    #===========================================================================
    def read_write_good_hel(self, ncomb):
        """return the code to read/write the good_hel common_block"""    

        convert = {'ncomb' : ncomb}

        output = """
        subroutine write_good_hel(stream_id)
        implicit none
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, 2)
        INTEGER NTRY(2)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        write(stream_id,*) GOODHEL
        return
        end
        
        
        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, 2)
        INTEGER NTRY(2)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        read(stream_id,*) GOODHEL
        NTRY(1) = MAXTRIES + 1
        NTRY(2) = MAXTRIES + 1
        return
        end 
        
        subroutine init_good_hel()
        implicit none
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, 2)        
        INTEGER NTRY(2)
        INTEGER I
        
        do i=1,NCOMB
            GOODHEL(I,1) = .false.
            GOODHEL(I,2) = .false.
        enddo
        NTRY(1) = 0
        NTRY(2) = 0
        end
        
        integer function get_maxsproc()
        implicit none
        include 'maxamps.inc'
        
        get_maxsproc = maxsproc
        return 
        end
                
        """ % convert
        
        return output
                           


    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, subproc_group, diagrams_for_config):
        """Write the configs.inc file with topology information for a
        subprocess group. Use the first subprocess with a diagram for each
        configuration."""

        matrix_elements = subproc_group.get('matrix_elements')
        model = matrix_elements[0].get('processes')[0].get('model')

        diagrams = []
        config_numbers = []
        for iconfig, config in enumerate(diagrams_for_config):
            # Check if any diagrams correspond to this config
            if set(config) == set([0]):
                continue
            subproc_diags = []
            for s,d in enumerate(config):
                if d:
                    subproc_diags.append(matrix_elements[s].\
                                         get('diagrams')[d-1])
                else:
                    subproc_diags.append(None)
            diagrams.append(subproc_diags)
            config_numbers.append(iconfig + 1)

        # Extract number of external particles
        (nexternal, ninitial) = subproc_group.get_nexternal_ninitial()

        return len(diagrams), \
               self.write_configs_file_from_diagrams(writer, diagrams,
                                                config_numbers,
                                                nexternal, ninitial,
                                                     model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_run_config.inc')
        if self.proc_characteristic['loop_induced']:
            job_per_chan = 1
        else: 
            job_per_chan = 2
        text = open(path).read() % {'chanperjob':job_per_chan} 
        writer.write(text)
        return True


    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, subproc_group):
        """Write the leshouche.inc file for MG4"""

        all_lines = []

        for iproc, matrix_element in \
            enumerate(subproc_group.get('matrix_elements')):
            all_lines.extend(self.get_leshouche_lines(matrix_element,
                                                 iproc))
        # Write the file
        writer.writelines(all_lines)
        return True


    def finalize(self,*args, **opts):

        super(ProcessExporterFortranMEGroup, self).finalize(*args, **opts)
        #ensure that the grouping information is on the correct value
        self.proc_characteristic['grouped_matrix'] = True        

        
#===============================================================================
# UFO_model_to_mg4
#===============================================================================

python_to_fortran = lambda x: parsers.UFOExpressionParserFortran().parse(x)

class UFO_model_to_mg4(object):
    """ A converter of the UFO-MG5 Model to the MG4 format """

    # The list below shows the only variables the user is allowed to change by
    # himself for each PS point. If he changes any other, then calling 
    # UPDATE_AS_PARAM() (or equivalently MP_UPDATE_AS_PARAM()) will not
    # correctly account for the change.
    PS_dependent_key = ['aS','MU_R']
    mp_complex_format = 'complex*32'
    mp_real_format = 'real*16'
    # Warning, it is crucial none of the couplings/parameters of the model
    # starts with this prefix. I should add a check for this.
    # You can change it as the global variable to check_param_card.ParamCard
    mp_prefix = check_param_card.ParamCard.mp_prefix
    
    def __init__(self, model, output_path, opt=None):
        """ initialization of the objects """
        
        self.model = model
        self.model_name = model['name']
        self.dir_path = output_path
        
        self.opt = {'complex_mass': False, 'export_format': 'madevent', 'mp':True,
                        'loop_induced': False}
        if opt:
            self.opt.update(opt)
            
        self.coups_dep = []    # (name, expression, type)
        self.coups_indep = []  # (name, expression, type)
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)
        self.params_ext = []   # external parameter
        self.p_to_f = parsers.UFOExpressionParserFortran()
        self.mp_p_to_f = parsers.UFOExpressionParserMPFortran()            
    
    def pass_parameter_to_case_insensitive(self):
        """modify the parameter if some of them are identical up to the case"""
    
        lower_dict={}
        duplicate = set()
        keys = self.model['parameters'].keys()
        for key in keys:
            for param in self.model['parameters'][key]:
                lower_name = param.name.lower()
                if not lower_name:
                    continue
                try:
                    lower_dict[lower_name].append(param)
                except KeyError,error:
                    lower_dict[lower_name] = [param]
                else:
                    duplicate.add(lower_name)
                    logger.debug('%s is define both as lower case and upper case.' 
                                 % lower_name)
        if not duplicate:
            return
        
        re_expr = r'''\b(%s)\b'''
        to_change = []
        change={}
        for value in duplicate:
            for i, var in enumerate(lower_dict[value]):
                to_change.append(var.name)
                new_name = '%s%s' % (var.name.lower(), 
                                                  ('__%d'%(i+1) if i>0 else ''))
                change[var.name] = new_name
                var.name = new_name
    
        # Apply the modification to the map_CTcoup_CTparam of the model
        # if it has one (giving for each coupling the CT parameters whcih
        # are necessary and which should be exported to the model.
        if hasattr(self.model,'map_CTcoup_CTparam'):
            for coup, ctparams in self.model.map_CTcoup_CTparam:
                for i, ctparam in enumerate(ctparams):
                    try:
                        self.model.map_CTcoup_CTparam[coup][i] = change[ctparam]
                    except KeyError:
                        pass

        replace = lambda match_pattern: change[match_pattern.groups()[0]]
        rep_pattern = re.compile(re_expr % '|'.join(to_change))
        
        # change parameters
        for key in keys:
            if key == ('external',):
                continue
            for param in self.model['parameters'][key]: 
                param.expr = rep_pattern.sub(replace, param.expr)
            
        # change couplings
        for key in self.model['couplings'].keys():
            for coup in self.model['couplings'][key]:
                coup.expr = rep_pattern.sub(replace, coup.expr)
                
        # change mass/width
        for part in self.model['particles']:
            if str(part.get('mass')) in to_change:
                part.set('mass', rep_pattern.sub(replace, str(part.get('mass'))))
            if str(part.get('width')) in to_change:
                part.set('width', rep_pattern.sub(replace, str(part.get('width'))))                
                
    def refactorize(self, wanted_couplings = []):    
        """modify the couplings to fit with MG4 convention """
            
        # Keep only separation in alphaS        
        keys = self.model['parameters'].keys()
        keys.sort(key=len)
        for key in keys:
            to_add = [o for o in self.model['parameters'][key] if o.name]

            if key == ('external',):
                self.params_ext += to_add
            elif any([(k in key) for k in self.PS_dependent_key]):
                self.params_dep += to_add
            else:
                self.params_indep += to_add
        # same for couplings
        keys = self.model['couplings'].keys()
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if any([(k in key) for k in self.PS_dependent_key]):
                self.coups_dep += [c for c in coup_list if
                                   (not wanted_couplings or c.name in \
                                    wanted_couplings)]
            else:
                self.coups_indep += [c for c in coup_list if
                                     (not wanted_couplings or c.name in \
                                      wanted_couplings)]
                
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        if 'G' in self.params_dep:
            index = self.params_dep.index('G')
            G = self.params_dep.pop(index)
        #    G.expr = '2*cmath.sqrt(as*pi)'
        #    self.params_indep.insert(0, self.params_dep.pop(index))
        # No need to add it if not defined   
            
        if 'aS' not in self.params_ext:
            logger.critical('aS not define as external parameter adding it!')
            #self.model['parameters']['aS'] = base_objects.ParamCardVariable('aS', 0.138,'DUMMY',(1,))
            self.params_indep.append( base_objects. ModelVariable('aS', '0.138','real'))
            self.params_indep.append( base_objects. ModelVariable('G', '4.1643','real'))
    def build(self, wanted_couplings = [], full=True):
        """modify the couplings to fit with MG4 convention and creates all the 
        different files"""
        
        self.pass_parameter_to_case_insensitive() 
        self.refactorize(wanted_couplings)

        # write the files
        if full:
            if wanted_couplings:
                # extract the wanted ct parameters
                self.extract_needed_CTparam(wanted_couplings=wanted_couplings)
            self.write_all()
            

    def open(self, name, comment='c', format='default'):
        """ Open the file name in the correct directory and with a valid
        header."""
        
        file_path = pjoin(self.dir_path, name)
        
        if format == 'fortran':
            fsock = writers.FortranWriter(file_path, 'w')
        else:
            fsock = open(file_path, 'w')
        
        file.writelines(fsock, comment * 77 + '\n')
        file.writelines(fsock,'%(comment)s written by the UFO converter\n' % \
                               {'comment': comment + (6 - len(comment)) *  ' '})
        file.writelines(fsock, comment * 77 + '\n\n')
        return fsock       

    
    def write_all(self):
        """ write all the files """
        #write the part related to the external parameter
        self.create_ident_card()
        self.create_param_read()
        
        #write the definition of the parameter
        self.create_input()
        self.create_intparam_def(dp=True,mp=False)
        if self.opt['mp']:
            self.create_intparam_def(dp=False,mp=True)
        
        # definition of the coupling.
        self.create_actualize_mp_ext_param_inc()
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        
        # the makefile
        self.create_makeinc()
        self.create_param_write()

        # The model functions
        self.create_model_functions_inc()
        self.create_model_functions_def()
        
        # The param_card.dat        
        self.create_param_card()
        

        # All the standard files
        self.copy_standard_file()

    ############################################################################
    ##  ROUTINE CREATING THE FILES  ############################################
    ############################################################################

    def copy_standard_file(self):
        """Copy the standard files for the fortran model."""
        
        #copy the library files
        file_to_link = ['formats.inc','printout.f', \
                        'rw_para.f', 'testprog.f']
    
        for filename in file_to_link:
            cp( MG5DIR + '/models/template_files/fortran/' + filename, \
                                                                self.dir_path)
            
        file = open(os.path.join(MG5DIR,\
                              'models/template_files/fortran/rw_para.f')).read()

        includes=["include \'coupl.inc\'","include \'input.inc\'",
                                              "include \'model_functions.inc\'"]
        if self.opt['mp']:
            includes.extend(["include \'mp_coupl.inc\'","include \'mp_input.inc\'"])
        # In standalone and madloop we do no use the compiled param card but
        # still parse the .dat one so we must load it.
        if self.opt['loop_induced']:
            #loop induced follow MadEvent way to handle the card.
            load_card = ''
            lha_read_filename='lha_read.f'            
        elif self.opt['export_format'] in ['madloop','madloop_optimized', 'madloop_matchbox']:
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read_mp.f'
        elif self.opt['export_format'].startswith('standalone') \
            or self.opt['export_format'] in ['madweight', 'plugin']\
            or self.opt['export_format'].startswith('matchbox'):
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read.f'
        else:
            load_card = ''
            lha_read_filename='lha_read.f'
        cp( MG5DIR + '/models/template_files/fortran/' + lha_read_filename, \
                                       os.path.join(self.dir_path,'lha_read.f'))
        
        file=file%{'includes':'\n      '.join(includes),
                   'load_card':load_card}
        writer=open(os.path.join(self.dir_path,'rw_para.f'),'w')
        writer.writelines(file)
        writer.close()

        if self.opt['export_format'] in ['madevent', 'FKS5_default', 'FKS5_optimized'] \
            or self.opt['loop_induced']:
            cp( MG5DIR + '/models/template_files/fortran/makefile_madevent', 
                self.dir_path + '/makefile')
            if self.opt['export_format'] in ['FKS5_default', 'FKS5_optimized']:
                path = pjoin(self.dir_path, 'makefile')
                text = open(path).read()
                text = text.replace('madevent','aMCatNLO')
                open(path, 'w').writelines(text)
        elif self.opt['export_format'] in ['standalone', 'standalone_msP','standalone_msF',
                                  'madloop','madloop_optimized', 'standalone_rw', 
                                  'madweight','matchbox','madloop_matchbox', 'plugin']:
            cp( MG5DIR + '/models/template_files/fortran/makefile_standalone', 
                self.dir_path + '/makefile')
        #elif self.opt['export_format'] in []:
            #pass
        else:
            raise MadGraph5Error('Unknown format')

    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        if self.opt['mp']:
            mp_fsock = self.open('mp_coupl.inc', format='fortran')
            mp_fsock_same_name = self.open('mp_coupl_same_name.inc',\
                                            format='fortran')

        # Write header
        header = """double precision G
                common/strong/ G
                 
                double complex gal(2)
                common/weak/ gal
                
                double precision MU_R
                common/rscale/ MU_R

                double precision Nf
                parameter(Nf=%d)
                """ % self.model.get_nflav()
                
        fsock.writelines(header)
        
        if self.opt['mp']:
            header = """%(real_mp_format)s %(mp_prefix)sG
                    common/MP_strong/ %(mp_prefix)sG
                     
                    %(complex_mp_format)s %(mp_prefix)sgal(2)
                    common/MP_weak/ %(mp_prefix)sgal
                    
                    %(complex_mp_format)s %(mp_prefix)sMU_R
                    common/MP_rscale/ %(mp_prefix)sMU_R

                """




            mp_fsock.writelines(header%{'real_mp_format':self.mp_real_format,
                                  'complex_mp_format':self.mp_complex_format,
                                  'mp_prefix':self.mp_prefix})
            mp_fsock_same_name.writelines(header%{'real_mp_format':self.mp_real_format,
                                  'complex_mp_format':self.mp_complex_format,
                                  'mp_prefix':''})

        # Write the Mass definition/ common block
        masses = set()
        widths = set()
        if self.opt['complex_mass']:
            complex_mass = set()
            
        for particle in self.model.get('particles'):
            #find masses
            one_mass = particle.get('mass')
            if one_mass.lower() != 'zero':
                masses.add(one_mass)
                
            # find width
            one_width = particle.get('width')
            if one_width.lower() != 'zero':
                widths.add(one_width)
                if self.opt['complex_mass'] and one_mass.lower() != 'zero':
                    complex_mass.add('CMASS_%s' % one_mass)
            
        if masses:
            fsock.writelines('double precision '+','.join(masses)+'\n')
            fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_real_format+' '+\
                                                          ','.join(masses)+'\n')
                mp_fsock_same_name.writelines('common/MP_masses/ '+\
                                                        ','.join(masses)+'\n\n')                
                mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                                        self.mp_prefix+m for m in masses])+'\n')
                mp_fsock.writelines('common/MP_masses/ '+\
                            ','.join([self.mp_prefix+m for m in masses])+'\n\n')                

        if widths:
            fsock.writelines('double precision '+','.join(widths)+'\n')
            fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_real_format+' '+\
                                                          ','.join(widths)+'\n')
                mp_fsock_same_name.writelines('common/MP_widths/ '+\
                                                        ','.join(widths)+'\n\n')                
                mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                                        self.mp_prefix+w for w in widths])+'\n')
                mp_fsock.writelines('common/MP_widths/ '+\
                            ','.join([self.mp_prefix+w for w in widths])+'\n\n')
        
        # Write the Couplings
        coupling_list = [coupl.name for coupl in self.coups_dep + self.coups_indep]       
        fsock.writelines('double complex '+', '.join(coupling_list)+'\n')
        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        if self.opt['mp']:
            mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                   ','.join(coupling_list)+'\n')
            mp_fsock_same_name.writelines('common/MP_couplings/ '+\
                                                 ','.join(coupling_list)+'\n\n')                
            mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                 self.mp_prefix+c for c in coupling_list])+'\n')
            mp_fsock.writelines('common/MP_couplings/ '+\
                     ','.join([self.mp_prefix+c for c in coupling_list])+'\n\n')            
        
        # Write complex mass for complex mass scheme (if activated)
        if self.opt['complex_mass'] and complex_mass:
            fsock.writelines('double complex '+', '.join(complex_mass)+'\n')
            fsock.writelines('common/complex_mass/ '+', '.join(complex_mass)+'\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                    ','.join(complex_mass)+'\n')
                mp_fsock_same_name.writelines('common/MP_complex_mass/ '+\
                                                  ','.join(complex_mass)+'\n\n')                
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                self.mp_prefix+cm for cm in complex_mass])+'\n')
                mp_fsock.writelines('common/MP_complex_mass/ '+\
                    ','.join([self.mp_prefix+cm for cm in complex_mass])+'\n\n')                       
        
    def create_write_couplings(self):
        """ write the file coupl_write.inc """
        
        fsock = self.open('coupl_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' Couplings of %s'  
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""" % self.model_name)
        def format(coupl):
            return 'write(*,2) \'%(name)s = \', %(name)s' % {'name': coupl.name}
        
        # Write the Couplings
        lines = [format(coupl) for coupl in self.coups_dep + self.coups_indep]       
        fsock.writelines('\n'.join(lines))
        
        
    def create_input(self):
        """create input.inc containing the definition of the parameters"""
        
        fsock = self.open('input.inc', format='fortran')
        if self.opt['mp']:
            mp_fsock = self.open('mp_input.inc', format='fortran')
                    
        #find mass/ width since they are already define
        already_def = set()
        for particle in self.model.get('particles'):
            already_def.add(particle.get('mass').lower())
            already_def.add(particle.get('width').lower())
            if self.opt['complex_mass']:
                already_def.add('cmass_%s' % particle.get('mass').lower())
        
        is_valid = lambda name: name.lower() not in ['g', 'mu_r', 'zero'] and \
                                                 name.lower() not in already_def
        
        real_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'real'
                            and is_valid(param.name)]

        real_parameters += [param.name for param in self.params_ext 
                            if param.type == 'real'and 
                               is_valid(param.name)]

        # check the parameter is a CT parameter or not
        # if yes, just use the needed ones        
        real_parameters = [param for param in real_parameters \
                                           if self.check_needed_param(param)]

        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        if self.opt['mp']:
            mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                              self.mp_prefix+p for p in real_parameters])+'\n')
            mp_fsock.writelines('common/MP_params_R/ '+','.join([\
                            self.mp_prefix+p for p in real_parameters])+'\n\n')        
        
        complex_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'complex' and
                            is_valid(param.name)]

        # check the parameter is a CT parameter or not
        # if yes, just use the needed ones        
        complex_parameters = [param for param in complex_parameters \
                             if self.check_needed_param(param)]

        if complex_parameters:
            fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
            fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')
            if self.opt['mp']:
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                            self.mp_prefix+p for p in complex_parameters])+'\n')
                mp_fsock.writelines('common/MP_params_C/ '+','.join([\
                          self.mp_prefix+p for p in complex_parameters])+'\n\n')

    def check_needed_param(self, param):
        """ Returns whether the parameter in argument is needed for this 
        specific computation or not."""
    
        # If this is a leading order model or if there was no CT parameter
        # employed in this NLO model, one can directly return that the 
        # parameter is needed since only CTParameters are filtered.
        if not hasattr(self, 'allCTparameters') or \
               self.allCTparameters is None or self.usedCTparameters is None or \
               len(self.allCTparameters)==0:
            return True
         
        # We must allow the conjugate shorthand for the complex parameter as
        # well so we check wether either the parameter name or its name with
        # 'conjg__' substituted with '' is present in the list.
        # This is acceptable even if some parameter had an original name 
        # including 'conjg__' in it, because at worst we export a parameter 
        # was not needed.
        param = param.lower()
        cjg_param = param.replace('conjg__','',1)
                
        # First make sure it is a CTparameter
        if param not in self.allCTparameters and \
           cjg_param not in self.allCTparameters:
            return True
        
        # Now check if it is in the list of CTparameters actually used
        return (param in self.usedCTparameters or \
                                             cjg_param in self.usedCTparameters)
                
    def extract_needed_CTparam(self,wanted_couplings=[]):
        """ Extract what are the needed CT parameters given the wanted_couplings"""
        
        if not hasattr(self.model,'map_CTcoup_CTparam') or not wanted_couplings:
            # Setting these lists to none wil disable the filtering in 
            # check_needed_param
            self.allCTparameters  = None
            self.usedCTparameters = None
            return
        
        # All CTparameters appearin in all CT couplings        
        allCTparameters=self.model.map_CTcoup_CTparam.values()
        # Define in this class the list of all CT parameters
        self.allCTparameters=list(\
                            set(itertools.chain.from_iterable(allCTparameters)))

        # All used CT couplings
        w_coupls = [coupl.lower() for coupl in wanted_couplings]
        allUsedCTCouplings = [coupl for coupl in 
              self.model.map_CTcoup_CTparam.keys() if coupl.lower() in w_coupls]
        
        # Now define the list of all CT parameters that are actually used
        self.usedCTparameters=list(\
          set(itertools.chain.from_iterable([
            self.model.map_CTcoup_CTparam[coupl] for coupl in allUsedCTCouplings
                                                                            ])))       
        
        # Now at last, make these list case insensitive
        self.allCTparameters = [ct.lower() for ct in self.allCTparameters]
        self.usedCTparameters = [ct.lower() for ct in self.usedCTparameters]
        
    def create_intparam_def(self, dp=True, mp=False):
        """ create intparam_definition.inc setting the internal parameters.
        Output the double precision and/or the multiple precision parameters
        depending on the parameters dp and mp. If mp only, then the file names
        get the 'mp_' prefix.
         """

        fsock = self.open('%sintparam_definition.inc'%
                             ('mp_' if mp and not dp else ''), format='fortran')
        
        fsock.write_comments(\
                "Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        if dp:        
            fsock.writelines("G = 2 * DSQRT(AS*PI) ! for the first init\n")
        if mp:
            fsock.writelines("MP__G = 2 * SQRT(MP__AS*MP__PI) ! for the first init\n")
            
        for param in self.params_indep:
            if param.name == 'ZERO':
                continue
            # check whether the parameter is a CT parameter
            # if yes,just used the needed ones
            if not self.check_needed_param(param.name):
                continue
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            if mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))    

        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        if dp:        
            fsock.writelines("aS = G**2/4/pi\n")
        if mp:
            fsock.writelines("MP__aS = MP__G**2/4/MP__PI\n")
        for param in self.params_dep:
            # check whether the parameter is a CT parameter
            # if yes,just used the needed ones
            if not self.check_needed_param(param.name):
                continue
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            elif mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))

        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")
        if ('aEWM1',) in self.model['parameters']:
            if dp:
                fsock.writelines(""" gal(1) = 3.5449077018110318d0 / DSQRT(aEWM1)
                                 gal(2) = 1d0
                         """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 2 * SQRT(MP__PI/MP__aEWM1)
                                 %(mp_prefix)sgal(2) = 1d0 
                                 """ %{'mp_prefix':self.mp_prefix})
                pass
        # in Gmu scheme, aEWM1 is not external but Gf is an exteranl variable
        elif ('Gf',) in self.model['parameters']:
            if dp:
                fsock.writelines(""" gal(1) = 2.378414230005442133435d0*MDL_MW*DSQRT(1D0-MDL_MW**2/MDL_MZ**2)*DSQRT(MDL_Gf)
                                 gal(2) = 1d0
                         """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 2*MP__MDL_MW*SQRT(1e0_16-MP__MDL_MW**2/MP__MDL_MZ**2)*SQRT(SQRT(2e0_16)*MP__MDL_Gf)
                                 %(mp_prefix)sgal(2) = 1d0
                                 """ %{'mp_prefix':self.mp_prefix})
                pass
        else:
            if dp:
                logger.warning('$RED aEWM1 and Gf not define in MODEL. AQED will not be written correcty in LHE FILE')
                fsock.writelines(""" gal(1) = 1d0
                                 gal(2) = 1d0
                             """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 1e0_16
                                 %(mp_prefix)sgal(2) = 1e0_16
                             """%{'mp_prefix':self.mp_prefix})

    
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = 25
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        for i in range(nb_coup_indep):
            # For the independent couplings, we compute the double and multiple
            # precision ones together
            data = self.coups_indep[nb_def_by_file * i: 
                             min(len(self.coups_indep), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data, dp=True, mp=self.opt['mp'])
            
        for i in range(nb_coup_dep):
            # For the dependent couplings, we compute the double and multiple
            # precision ones in separate subroutines.
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                                               dp=True,mp=False)
            if self.opt['mp']:
                self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                                              dp=False,mp=True)
        
        
    def create_couplings_main(self, nb_def_by_file=25):
        """ create couplings.f """

        fsock = self.open('couplings.f', format='fortran')
        
        fsock.writelines("""subroutine coup()

                            implicit none
                            double precision PI, ZERO
                            logical READLHA
                            parameter  (PI=3.141592653589793d0)
                            parameter  (ZERO=0d0)
                            include \'model_functions.inc\'""")
        if self.opt['mp']:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 
        fsock.writelines("""include \'input.inc\'
                            include \'coupl.inc\'
                            READLHA = .true.
                            include \'intparam_definition.inc\'""")
        if self.opt['mp']:
            fsock.writelines("""include \'mp_intparam_definition.inc\'\n""")
        
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        if self.opt['mp']:
            fsock.writelines('\n'.join(\
                    ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param()

                            implicit none
                            double precision PI, ZERO
                            logical READLHA
                            parameter  (PI=3.141592653589793d0)            
                            parameter  (ZERO=0d0)
                            include \'model_functions.inc\'""")
        fsock.writelines("""include \'input.inc\'
                            include \'coupl.inc\'
                            READLHA = .false.""")
        fsock.writelines("""    
                            include \'intparam_definition.inc\'\n
                         """)
            
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
                
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param2(mu_r2,as2)

                            implicit none
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            double precision mu_r2, as2
                            include \'model_functions.inc\'""")
        fsock.writelines("""include \'input.inc\'
                            include \'coupl.inc\'""")
        fsock.writelines("""
                            if (mu_r2.gt.0d0) MU_R = mu_r2
                            G = SQRT(4.0d0*PI*AS2) 
                            AS = as2

                            CALL UPDATE_AS_PARAM()
                         """)
        fsock.writelines('''\n return \n end\n''')

        if self.opt['mp']:
            fsock.writelines("""subroutine mp_update_as_param()
    
                                implicit none
                                logical READLHA
                                include \'model_functions.inc\'""")
            fsock.writelines("""%s MP__PI, MP__ZERO
                                    parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                    parameter (MP__ZERO=0e0_16)
                                    include \'mp_input.inc\'
                                    include \'mp_coupl.inc\'
                            """%self.mp_real_format)
            fsock.writelines("""include \'input.inc\'
                                include \'coupl.inc\'
                                include \'actualize_mp_ext_params.inc\'
                                READLHA = .false.
                                include \'mp_intparam_definition.inc\'\n
                             """)
            
            nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
            nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
                    
            fsock.write_comments('\ncouplings needed to be evaluated points by points\n')
    
            fsock.writelines('\n'.join(\
                        ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                          for i in range(nb_coup_dep)]))
            fsock.writelines('''\n return \n end\n''')

    def create_couplings_part(self, nb_file, data, dp=True, mp=False):
        """ create couplings[nb_file].f containing information coming from data.
        Outputs the computation of the double precision and/or the multiple
        precision couplings depending on the parameters dp and mp.
        If mp is True and dp is False, then the prefix 'MP_' is appended to the
        filename and subroutine name.
        """
        
        fsock = self.open('%scouplings%s.f' %('mp_' if mp and not dp else '',
                                                     nb_file), format='fortran')
        fsock.writelines("""subroutine %scoup%s()
          
          implicit none
          include \'model_functions.inc\'"""%('mp_' if mp and not dp else '',nb_file))
        if dp:
            fsock.writelines("""
              double precision PI, ZERO
              parameter  (PI=3.141592653589793d0)
              parameter  (ZERO=0d0)
              include 'input.inc'
              include 'coupl.inc'""")
        if mp:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 

        for coupling in data:
            if dp:            
                fsock.writelines('%s = %s' % (coupling.name,
                                          self.p_to_f.parse(coupling.expr)))
            if mp:
                fsock.writelines('%s%s = %s' % (self.mp_prefix,coupling.name,
                                          self.mp_p_to_f.parse(coupling.expr)))
        fsock.writelines('end')

    def create_model_functions_inc(self):
        """ Create model_functions.inc which contains the various declarations
        of auxiliary functions which might be used in the couplings expressions
        """

        additional_fct = []
        # check for functions define in the UFO model
        ufo_fct = self.model.get('functions')
        if ufo_fct:
            for fct in ufo_fct:
                # already handle by default
                if fct.name not in ["complexconjugate", "re", "im", "sec", 
                       "csc", "asec", "acsc", "theta_function", "cond", 
                       "condif", "reglogp", "reglogm", "reglog", "recms", "arg", "cot"]:
                    additional_fct.append(fct.name)

        
        fsock = self.open('model_functions.inc', format='fortran')
        fsock.writelines("""double complex cond
          double complex condif
          double complex reglog
          double complex reglogp
          double complex reglogm
          double complex recms
          double complex arg
          %s
          """ % "\n".join(["          double complex %s" % i for i in additional_fct]))

        
        if self.opt['mp']:
            fsock.writelines("""%(complex_mp_format)s mp_cond
          %(complex_mp_format)s mp_condif
          %(complex_mp_format)s mp_reglog
          %(complex_mp_format)s mp_reglogp
          %(complex_mp_format)s mp_reglogm
          %(complex_mp_format)s mp_recms
          %(complex_mp_format)s mp_arg
          %(additional)s
          """ %\
          {"additional": "\n".join(["          %s %s" % (self.mp_complex_format, i) for i in additional_fct]),
           'complex_mp_format':self.mp_complex_format
           }) 

    def create_model_functions_def(self):
        """ Create model_functions.f which contains the various definitions
        of auxiliary functions which might be used in the couplings expressions
        Add the functions.f functions for formfactors support
        """

        fsock = self.open('model_functions.f', format='fortran')
        fsock.writelines("""double complex function cond(condition,truecase,falsecase)
          implicit none
          double complex condition,truecase,falsecase
          if(condition.eq.(0.0d0,0.0d0)) then
             cond=truecase
          else
             cond=falsecase
          endif
          end
          
          double complex function condif(condition,truecase,falsecase)
          implicit none
          logical condition
          double complex truecase,falsecase
          if(condition) then
             condif=truecase
          else
             condif=falsecase
          endif
          end

          double complex function recms(condition,expr)
          implicit none
          logical condition
          double complex expr
          if(condition)then
             recms=expr
          else
             recms=dcmplx(dble(expr))
          endif
          end
          
          double complex function reglog(arg)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg
          if(arg.eq.(0.0d0,0.0d0)) then
             reglog=(0.0d0,0.0d0)
          else
             reglog=log(arg)
          endif
          end

          double complex function reglogp(arg)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg
          if(arg.eq.(0.0d0,0.0d0))then
             reglogp=(0.0d0,0.0d0)
          else
             if(dble(arg).lt.0.0d0.and.dimag(arg).lt.0.0d0)then
                reglogp=log(arg) + TWOPII
             else
                reglogp=log(arg)
             endif
          endif
          end

          double complex function reglogm(arg)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg
          if(arg.eq.(0.0d0,0.0d0))then
             reglogm=(0.0d0,0.0d0)
          else
             if(dble(arg).lt.0.0d0.and.dimag(arg).gt.0.0d0)then
                reglogm=log(arg) - TWOPII
             else
                reglogm=log(arg)
             endif
          endif
          end
          
          double complex function arg(comnum)
          implicit none
          double complex comnum
          double complex iim 
          iim = (0.0d0,1.0d0)
          if(comnum.eq.(0.0d0,0.0d0)) then
             arg=(0.0d0,0.0d0)
          else
             arg=log(comnum/abs(comnum))/iim
          endif
          end""")
        if self.opt['mp']:
            fsock.writelines("""
              
              %(complex_mp_format)s function mp_cond(condition,truecase,falsecase)
              implicit none
              %(complex_mp_format)s condition,truecase,falsecase
              if(condition.eq.(0.0e0_16,0.0e0_16)) then
                 mp_cond=truecase
              else
                 mp_cond=falsecase
              endif
              end
              
              %(complex_mp_format)s function mp_condif(condition,truecase,falsecase)
              implicit none
              logical condition
              %(complex_mp_format)s truecase,falsecase
              if(condition) then
                 mp_condif=truecase
              else
                 mp_condif=falsecase
              endif
              end

              %(complex_mp_format)s function mp_recms(condition,expr)
              implicit none
              logical condition
              %(complex_mp_format)s expr
              if(condition)then
                 mp_recms=expr
              else
                 mp_recms=cmplx(real(expr),kind=16)
              endif
              end
              
              %(complex_mp_format)s function mp_reglog(arg)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg
              if(arg.eq.(0.0e0_16,0.0e0_16)) then
                 mp_reglog=(0.0e0_16,0.0e0_16)
              else
                 mp_reglog=log(arg)
              endif
              end

              %(complex_mp_format)s function mp_reglogp(arg)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg
              if(arg.eq.(0.0e0_16,0.0e0_16))then
                 mp_reglogp=(0.0e0_16,0.0e0_16)
              else
                 if(real(arg,kind=16).lt.0.0e0_16.and.imagpart(arg).lt.0.0e0_16)then
                    mp_reglogp=log(arg) + TWOPII
                 else
                    mp_reglogp=log(arg)
                 endif
              endif
              end
              
              %(complex_mp_format)s function mp_reglogm(arg)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg
              if(arg.eq.(0.0e0_16,0.0e0_16))then
                 mp_reglogm=(0.0e0_16,0.0e0_16)
              else
                 if(real(arg,kind=16).lt.0.0e0_16.and.imagpart(arg).gt.0.0e0_16)then
                    mp_reglogm=log(arg) - TWOPII
                 else
                    mp_reglogm=log(arg)
                 endif 
              endif
              end
              
              %(complex_mp_format)s function mp_arg(comnum)
              implicit none
              %(complex_mp_format)s comnum
              %(complex_mp_format)s imm
              imm = (0.0e0_16,1.0e0_16)
              if(comnum.eq.(0.0e0_16,0.0e0_16)) then
                 mp_arg=(0.0e0_16,0.0e0_16)
              else
                 mp_arg=log(comnum/abs(comnum))/imm
              endif
              end"""%{'complex_mp_format':self.mp_complex_format})


        #check for the file functions.f
        model_path = self.model.get('modelpath')
        if os.path.exists(pjoin(model_path,'Fortran','functions.f')):
            fsock.write_comment_line(' USER DEFINE FUNCTIONS ')
            input = pjoin(model_path,'Fortran','functions.f')
            file.writelines(fsock, open(input).read())
            fsock.write_comment_line(' END USER DEFINE FUNCTIONS ')
            
        # check for functions define in the UFO model
        ufo_fct = self.model.get('functions')
        if ufo_fct:
            fsock.write_comment_line(' START UFO DEFINE FUNCTIONS ')
            for fct in ufo_fct:
                # already handle by default
                if fct.name not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc", "condif",
                                    "theta_function", "cond", "reglog", "reglogp", "reglogm", "recms","arg"]:
                    ufo_fct_template = """
          double complex function %(name)s(%(args)s)
          implicit none
          double complex %(args)s
          %(name)s = %(fct)s

          return
          end
          """
                    text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", ".join(fct.arguments),                
                                'fct': self.p_to_f.parse(fct.expr)
                                 }
                    fsock.writelines(text)
            if self.opt['mp']:
                fsock.write_comment_line(' START UFO DEFINE FUNCTIONS FOR MP')
                for fct in ufo_fct:
                    # already handle by default
                    if fct.name not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc","condif",
                                        "theta_function", "cond", "reglog", "reglogp","reglogm", "recms","arg"]:
                        ufo_fct_template = """
          %(complex_mp_format)s function mp__%(name)s(mp__%(args)s)
          implicit none
          %(complex_mp_format)s mp__%(args)s
          mp__%(name)s = %(fct)s

          return
          end
          """
                        text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", mp__".join(fct.arguments),                
                                'fct': self.mp_p_to_f.parse(fct.expr),
                                'complex_mp_format': self.mp_complex_format
                                 }
                        fsock.writelines(text)


                    
            fsock.write_comment_line(' STOP UFO DEFINE FUNCTIONS ')                    

        

    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o'
        text += ' model_functions.o '
        
        nb_coup_indep = 1 + len(self.coups_dep) // 25 
        nb_coup_dep = 1 + len(self.coups_indep) // 25
        couplings_files=['couplings%s.o' % (i+1) \
                                for i in range(nb_coup_dep + nb_coup_indep) ]
        if self.opt['mp']:
            couplings_files+=['mp_couplings%s.o' % (i+1) for i in \
                               range(nb_coup_dep,nb_coup_dep + nb_coup_indep) ]
        text += ' '.join(couplings_files)
        fsock.writelines(text)
        
    def create_param_write(self):
        """ create param_write """

        fsock = self.open('param_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' External Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")
        def format(name):
            return 'write(*,*) \'%(name)s = \', %(name)s' % {'name': name}
        
        # Write the external parameter
        lines = [format(param.name) for param in self.params_ext]       
        fsock.writelines('\n'.join(lines))        
        
        fsock.writelines("""write(*,*)  ' Internal Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")        
        lines = [format(data.name) for data in self.params_indep 
                  if data.name != 'ZERO' and self.check_needed_param(data.name)]
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data.name) for data in self.params_dep \
                 if self.check_needed_param(data.name)]
        
        fsock.writelines('\n'.join(lines))                
        
 
    
    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock.lower()] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            if not parameter.name:
                return ''
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in self.params_ext]
        fsock.writelines('\n'.join(external_param))

    def create_actualize_mp_ext_param_inc(self):
        """ create the actualize_mp_ext_params.inc code """
        
        # In principle one should actualize all external, but for now, it is
        # hardcoded that only AS and MU_R can by dynamically changed by the user
        # so that we only update those ones.
        # Of course, to be on the safe side, one could decide to update all
        # external parameters.
        update_params_list=[p for p in self.params_ext if p.name in 
                                                          self.PS_dependent_key]
        
        res_strings = ["%(mp_prefix)s%(name)s=%(name)s"\
                        %{'mp_prefix':self.mp_prefix,'name':param.name}\
                                                for param in update_params_list]
        # When read_lha is false, it is G which is taken in input and not AS, so
        # this is what should be reset here too.
        if 'aS' in [param.name for param in update_params_list]:
            res_strings.append("%(mp_prefix)sG=G"%{'mp_prefix':self.mp_prefix})
            
        fsock = self.open('actualize_mp_ext_params.inc', format='fortran')
        fsock.writelines('\n'.join(res_strings))

    def create_param_read(self):    
        """create param_read"""
        
        if self.opt['export_format'] in ['madevent', 'FKS5_default', 'FKS5_optimized'] \
            or self.opt['loop_induced']:
            fsock = self.open('param_read.inc', format='fortran')
            fsock.writelines(' include \'../param_card.inc\'')
            return
    
        def format_line(parameter):
            """return the line for the ident_card corresponding to this 
            parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)""" \
                % {'name': parameter.name,
                   'value': self.p_to_f.parse(str(parameter.value.real))}
            if self.opt['mp']:
                template = template+ \
                ("\n call MP_LHA_get_real(npara,param,value,'%(name)s',"+
                 "%(mp_prefix)s%(name)s,%(value)s)") \
                % {'name': parameter.name,'mp_prefix': self.mp_prefix,
                   'value': self.mp_p_to_f.parse(str(parameter.value.real))}    
            return template        
    
        fsock = self.open('param_read.inc', format='fortran')
        res_strings = [format_line(param) \
                          for param in self.params_ext]
        
        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)        
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                
                res_strings.append('%(width)s = sign(%(width)s,%(mass)s)' % \
                 {'width': particle.get('width'), 'mass': particle.get('mass')})
                if self.opt['mp']:
                    res_strings.append(\
                      ('%(mp_pref)s%(width)s = sign(%(mp_pref)s%(width)s,'+\
                       '%(mp_pref)s%(mass)s)')%{'width': particle.get('width'),\
                       'mass': particle.get('mass'),'mp_pref':self.mp_prefix})

        fsock.writelines('\n'.join(res_strings))


    @staticmethod
    def create_param_card_static(model, output_path, rule_card_path=False,
                                 mssm_convert=True):
        """ create the param_card.dat for a givent model --static method-- """
        #1. Check if a default param_card is present:
        done = False
        if hasattr(model, 'restrict_card') and isinstance(model.restrict_card, str):
            restrict_name = os.path.basename(model.restrict_card)[9:-4]
            model_path = model.get('modelpath')
            if os.path.exists(pjoin(model_path,'paramcard_%s.dat' % restrict_name)):
                done = True
                files.cp(pjoin(model_path,'paramcard_%s.dat' % restrict_name),
                         output_path)
        if not done:
            param_writer.ParamCardWriter(model, output_path)
         
        if rule_card_path:   
            if hasattr(model, 'rule_card'):
                model.rule_card.write_file(rule_card_path)
        
        if mssm_convert:
            model_name = model.get('name')
            # IF MSSM convert the card to SLAH1
            if model_name == 'mssm' or model_name.startswith('mssm-'):
                import models.check_param_card as translator    
                # Check the format of the param_card for Pythia and make it correct
                if rule_card_path:
                    translator.make_valid_param_card(output_path, rule_card_path)
                translator.convert_to_slha1(output_path)        
    
    def create_param_card(self):
        """ create the param_card.dat """

        rule_card = pjoin(self.dir_path, 'param_card_rule.dat')
        if not hasattr(self.model, 'rule_card'):
            rule_card=False
        self.create_param_card_static(self.model, 
                                      output_path=pjoin(self.dir_path, 'param_card.dat'), 
                                      rule_card_path=rule_card, 
                                      mssm_convert=True)
        
def ExportV4Factory(cmd, noclean, output_type='default', group_subprocesses=True):
    """ Determine which Export_v4 class is required. cmd is the command 
        interface containing all potential usefull information.
        The output_type argument specifies from which context the output
        is called. It is 'madloop' for MadLoop5, 'amcatnlo' for FKS5 output
        and 'default' for tree-level outputs."""

    opt = cmd.options

    # ==========================================================================
    # First check whether Ninja must be installed.
    # Ninja would only be required if:
    #  a) Loop optimized output is selected
    #  b) the process gathered from the amplitude generated use loops

    if len(cmd._curr_amps)>0:
        try:
            curr_proc = cmd._curr_amps[0].get('process')
        except base_objects.PhysicsObject.PhysicsObjectError:
            curr_proc = None
    elif hasattr(cmd,'_fks_multi_proc') and \
                          len(cmd._fks_multi_proc.get('process_definitions'))>0:
        curr_proc = cmd._fks_multi_proc.get('process_definitions')[0]
    else:
        curr_proc = None

    requires_reduction_tool = opt['loop_optimized_output'] and \
                (not curr_proc is None) and \
                (curr_proc.get('perturbation_couplings') != [] and \
                not curr_proc.get('NLO_mode') in [None,'real','tree','LO','LOonly'])

    # An installation is required then, but only if the specified path is the
    # default local one and that the Ninja library appears missing.
    if requires_reduction_tool:
        cmd.install_reduction_library()
        
    # ==========================================================================
    # First treat the MadLoop5 standalone case       
    MadLoop_SA_options = {'clean': not noclean, 
      'complex_mass':cmd.options['complex_mass_scheme'],
      'export_format':'madloop', 
      'mp':True,
      'loop_dir': os.path.join(cmd._mgme_dir,'Template','loop_material'),
      'cuttools_dir': cmd._cuttools_dir,
      'iregi_dir':cmd._iregi_dir,
      'pjfry_dir':cmd.options['pjfry'],
      'golem_dir':cmd.options['golem'],
      'samurai_dir':cmd.options['samurai'],
      'ninja_dir':cmd.options['ninja'],
      'collier_dir':cmd.options['collier'],
      'fortran_compiler':cmd.options['fortran_compiler'],
      'f2py_compiler':cmd.options['f2py_compiler'],
      'output_dependencies':cmd.options['output_dependencies'],
      'SubProc_prefix':'P',
      'compute_color_flows':cmd.options['loop_color_flows'],
      'mode': 'reweight' if cmd._export_format == "standalone_rw" else '',
      'cluster_local_path': cmd.options['cluster_local_path']
      }


    if output_type.startswith('madloop'):        
        import madgraph.loop.loop_exporters as loop_exporters
        if os.path.isdir(os.path.join(cmd._mgme_dir, 'Template/loop_material')):
            ExporterClass=None
            if not cmd.options['loop_optimized_output']:
                ExporterClass=loop_exporters.LoopProcessExporterFortranSA
            else:
                if output_type == "madloop":
                    ExporterClass=loop_exporters.LoopProcessOptimizedExporterFortranSA
                    MadLoop_SA_options['export_format'] = 'madloop_optimized'
                elif output_type == "madloop_matchbox":
                    ExporterClass=loop_exporters.LoopProcessExporterFortranMatchBox
                    MadLoop_SA_options['export_format'] = 'madloop_matchbox'
                else:
                    raise Exception, "output_type not recognize %s" % output_type
            return ExporterClass(cmd._export_dir, MadLoop_SA_options)
        else:
            raise MadGraph5Error('MG5_aMC cannot find the \'loop_material\' directory'+\
                                 ' in %s'%str(cmd._mgme_dir))

    # Then treat the aMC@NLO output     
    elif output_type=='amcatnlo':
        import madgraph.iolibs.export_fks as export_fks
        ExporterClass=None
        amcatnlo_options = dict(opt)
        amcatnlo_options.update(MadLoop_SA_options)
        amcatnlo_options['mp'] = len(cmd._fks_multi_proc.get_virt_amplitudes()) > 0
        if not cmd.options['loop_optimized_output']:
            logger.info("Writing out the aMC@NLO code")
            ExporterClass = export_fks.ProcessExporterFortranFKS
            amcatnlo_options['export_format']='FKS5_default'
        else:
            logger.info("Writing out the aMC@NLO code, using optimized Loops")
            ExporterClass = export_fks.ProcessOptimizedExporterFortranFKS
            amcatnlo_options['export_format']='FKS5_optimized'
        return ExporterClass(cmd._export_dir, amcatnlo_options)


    # Then the default tree-level output
    elif output_type=='default':
        assert group_subprocesses in [True, False]
        
        opt = dict(opt)
        opt.update({'clean': not noclean,
               'complex_mass': cmd.options['complex_mass_scheme'],
               'export_format':cmd._export_format,
               'mp': False,  
               'sa_symmetry':False, 
               'model': cmd._curr_model.get('name'),
               'v5_model': False if cmd._model_v4_path else True })

        format = cmd._export_format #shortcut

        if format in ['standalone_msP', 'standalone_msF', 'standalone_rw']:
            opt['sa_symmetry'] = True      
        elif format == 'plugin':
            opt['sa_symmetry'] = cmd._export_plugin.sa_symmetry
    
        loop_induced_opt = dict(opt)
        loop_induced_opt.update(MadLoop_SA_options)
        loop_induced_opt['export_format'] = 'madloop_optimized'
        loop_induced_opt['SubProc_prefix'] = 'PV'
        # For loop_induced output with MadEvent, we must have access to the 
        # color flows.
        loop_induced_opt['compute_color_flows'] = True
        for key in opt:
            if key not in loop_induced_opt:
                loop_induced_opt[key] = opt[key]
    
        # Madevent output supports MadAnalysis5
        if format in ['madevent']:
            opt['madanalysis5'] = cmd.options['madanalysis5_path']
            
        if format == 'matrix' or format.startswith('standalone'):
            return ProcessExporterFortranSA(cmd._export_dir, opt, format=format)
        
        elif format in ['madevent'] and group_subprocesses:
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                import madgraph.loop.loop_exporters as loop_exporters
                return  loop_exporters.LoopInducedExporterMEGroup( 
                                               cmd._export_dir,loop_induced_opt)
            else:
                return  ProcessExporterFortranMEGroup(cmd._export_dir,opt)                
        elif format in ['madevent']:
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                import madgraph.loop.loop_exporters as loop_exporters
                return  loop_exporters.LoopInducedExporterMENoGroup( 
                                               cmd._export_dir,loop_induced_opt)
            else:
                return  ProcessExporterFortranME(cmd._export_dir,opt)
        elif format in ['matchbox']:
            return ProcessExporterFortranMatchBox(cmd._export_dir,opt)
        elif cmd._export_format in ['madweight'] and group_subprocesses:

            return ProcessExporterFortranMWGroup(cmd._export_dir, opt)
        elif cmd._export_format in ['madweight']:
            return ProcessExporterFortranMW(cmd._export_dir, opt)
        elif format == 'plugin':
            return cmd._export_plugin(cmd._export_dir, opt)
        else:
            raise Exception, 'Wrong export_v4 format'
    else:
        raise MadGraph5Error, 'Output type %s not reckognized in ExportV4Factory.'
    
            


#===============================================================================
# ProcessExporterFortranMWGroup
#===============================================================================
class ProcessExporterFortranMWGroup(ProcessExporterFortranMW):
    """Class to take care of exporting a set of matrix elements to
    MadEvent subprocess group format."""

    matrix_file = "matrix_madweight_group_v4.inc"
    grouped_mode = 'madweight'
    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model,
                                         group_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files."""

        if not isinstance(subproc_group, group_subprocs.SubProcessGroup):
            raise base_objects.PhysicsObject.PhysicsObjectError,\
                  "subproc_group object not SubProcessGroup"

        if not self.model:
            self.model = subproc_group.get('matrix_elements')[0].\
                         get('processes')[0].get('model')

        pathdir = os.path.join(self.dir_path, 'SubProcesses')

        # Create the directory PN in the specified path
        subprocdir = "P%d_%s" % (subproc_group.get('number'),
                                 subproc_group.get('name'))
        try:
            os.mkdir(pjoin(pathdir, subprocdir))
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)


        logger.info('Creating files in directory %s' % subprocdir)
        Ppath = pjoin(pathdir, subprocdir)

        # Create the matrix.f files, auto_dsig.f files and all inc files
        # for all subprocesses in the group

        maxamps = 0
        maxflows = 0
        tot_calls = 0

        matrix_elements = subproc_group.get('matrix_elements')

        for ime, matrix_element in \
                enumerate(matrix_elements):
            filename = pjoin(Ppath, 'matrix%d.f' % (ime+1))
            calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename), 
                                                matrix_element,
                                                fortran_model,
                                                str(ime+1),
                                                subproc_group.get('diagram_maps')[\
                                                                              ime])

            filename = pjoin(Ppath, 'auto_dsig%d.f' % (ime+1))
            self.write_auto_dsig_file(writers.FortranWriter(filename),
                                 matrix_element,
                                 str(ime+1))

            # Keep track of needed quantities
            tot_calls += int(calls)
            maxflows = max(maxflows, ncolor)
            maxamps = max(maxamps, len(matrix_element.get('diagrams')))

            # Draw diagrams
            filename = pjoin(Ppath, "matrix%d.ps" % (ime+1))
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                                    get('diagrams'),
                                              filename,
                                              model = \
                                                matrix_element.get('processes')[0].\
                                                                       get('model'),
                                              amplitude=True)
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Generate a list of diagrams corresponding to each configuration
        # [[d1, d2, ...,dn],...] where 1,2,...,n is the subprocess number
        # If a subprocess has no diagrams for this config, the number is 0

        subproc_diagrams_for_config = subproc_group.get('diagrams_for_configs')

        filename = pjoin(Ppath, 'auto_dsig.f')
        self.write_super_auto_dsig_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = pjoin(Ppath,'configs.inc')
        nconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            subproc_group,
            subproc_diagrams_for_config)

        filename = pjoin(Ppath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = pjoin(Ppath, 'phasespace.inc')
        self.write_phasespace_file(writers.FortranWriter(filename),
                           nconfigs)
                           

        filename = pjoin(Ppath, 'maxamps.inc')
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           max([len(me.get('processes')) for me in \
                                matrix_elements]),
                           len(matrix_elements))

        filename = pjoin(Ppath, 'mirrorprocs.inc')
        self.write_mirrorprocs(writers.FortranWriter(filename),
                          subproc_group)

        filename = pjoin(Ppath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(Ppath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(Ppath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

#        filename = pjoin(Ppath, 'processes.dat')
#        files.write_to_file(filename,
#                            self.write_processes_file,
#                            subproc_group)

        # Generate jpgs -> pass in make_html
        #os.system(os.path.join('..', '..', 'bin', 'gen_jpeg-pl'))

        linkfiles = ['driver.f', 'cuts.f', 'initialization.f','gen_ps.f', 'makefile', 'coupl.inc','madweight_param.inc', 'run.inc', 'setscales.f']

        for file in linkfiles:
            ln('../%s' % file, cwd=Ppath)

        ln('nexternal.inc', '../../Source', cwd=Ppath, log=False)
        ln('leshouche.inc', '../../Source', cwd=Ppath, log=False)
        ln('maxamps.inc', '../../Source', cwd=Ppath, log=False)
        ln('../../Source/maxparticles.inc', '.', log=True, cwd=Ppath)
        ln('../../Source/maxparticles.inc', '.', name='genps.inc', log=True, cwd=Ppath)
        ln('phasespace.inc', '../', log=True, cwd=Ppath)
        if not tot_calls:
            tot_calls = 0
        return tot_calls


    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
            
        return True, matrix_element.split_lepton_grouping()
    
    #===========================================================================
    # write_super_auto_dsig_file
    #===========================================================================
    def write_super_auto_dsig_file(self, writer, subproc_group):
        """Write the auto_dsig.f file selecting between the subprocesses
        in subprocess group mode"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        matrix_elements = subproc_group.get('matrix_elements')

        # Extract process info lines
        process_lines = '\n'.join([self.get_process_info_lines(me) for me in \
                                   matrix_elements])
        replace_dict['process_lines'] = process_lines

        nexternal, ninitial = matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        replace_dict['nsprocs'] = 2*len(matrix_elements)

        # Generate dsig definition line
        dsig_def_line = "DOUBLE PRECISION " + \
                        ",".join(["DSIG%d" % (iproc + 1) for iproc in \
                                  range(len(matrix_elements))])
        replace_dict["dsig_def_line"] = dsig_def_line

        # Generate dsig process lines
        call_dsig_proc_lines = []
        for iproc in range(len(matrix_elements)):
            call_dsig_proc_lines.append(\
                "IF(IPROC.EQ.%(num)d) DSIGPROC=DSIG%(num)d(P1,WGT,IMODE) ! %(proc)s" % \
                {"num": iproc + 1,
                 "proc": matrix_elements[iproc].get('processes')[0].base_string()})
        replace_dict['call_dsig_proc_lines'] = "\n".join(call_dsig_proc_lines)

        if writer:
            file = open(os.path.join(_file_path, \
                       'iolibs/template_files/super_auto_dsig_mw_group_v4.inc')).read()
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
        
    #===========================================================================
    # write_mirrorprocs
    #===========================================================================
    def write_mirrorprocs(self, writer, subproc_group):
        """Write the mirrorprocs.inc file determining which processes have
        IS mirror process in subprocess group mode."""

        lines = []
        bool_dict = {True: '.true.', False: '.false.'}
        matrix_elements = subproc_group.get('matrix_elements')
        lines.append("DATA (MIRRORPROCS(I),I=1,%d)/%s/" % \
                     (len(matrix_elements),
                      ",".join([bool_dict[me.get('has_mirror_process')] for \
                                me in matrix_elements])))
        # Write the file
        writer.writelines(lines)

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, subproc_group, diagrams_for_config):
        """Write the configs.inc file with topology information for a
        subprocess group. Use the first subprocess with a diagram for each
        configuration."""

        matrix_elements = subproc_group.get('matrix_elements')
        model = matrix_elements[0].get('processes')[0].get('model')

        diagrams = []
        config_numbers = []
        for iconfig, config in enumerate(diagrams_for_config):
            # Check if any diagrams correspond to this config
            if set(config) == set([0]):
                continue
            subproc_diags = []
            for s,d in enumerate(config):
                if d:
                    subproc_diags.append(matrix_elements[s].\
                                         get('diagrams')[d-1])
                else:
                    subproc_diags.append(None)
            diagrams.append(subproc_diags)
            config_numbers.append(iconfig + 1)

        # Extract number of external particles
        (nexternal, ninitial) = subproc_group.get_nexternal_ninitial()

        return len(diagrams), \
               self.write_configs_file_from_diagrams(writer, diagrams,
                                                config_numbers,
                                                nexternal, ninitial,
                                                matrix_elements[0],model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = os.path.join(_file_path,'iolibs','template_files','madweight_run_config.inc') 
        text = open(path).read() % {'chanperjob':'2'} 
        writer.write(text)
        return True


    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, subproc_group):
        """Write the leshouche.inc file for MG4"""

        all_lines = []

        for iproc, matrix_element in \
            enumerate(subproc_group.get('matrix_elements')):
            all_lines.extend(self.get_leshouche_lines(matrix_element,
                                                 iproc))

        # Write the file
        writer.writelines(all_lines)

        return True


    
