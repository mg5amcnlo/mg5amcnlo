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
from __future__ import absolute_import, division
from madgraph.iolibs.helas_call_writers import HelasCallWriter
from six.moves import range
from six.moves import zip
import six
from madgraph.core import base_objects
"""Methods and classes to export matrix elements to v4 format."""

import copy
import math, cmath
from six import StringIO
import itertools
import fractions
import glob
import logging
import math
import os
import io
import re
import shutil
import subprocess
import sys
import time
import traceback
import  collections

import aloha

import madgraph
import models
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
import madgraph
import aloha.create_aloha as create_aloha
import models.import_ufo as import_ufo
import models.write_param_card as param_writer
import models.check_param_card as check_param_card
from models import UFOError


from madgraph import MadGraph5Error, MG5DIR, ReadWrite
from madgraph.iolibs.files import cp, ln, mv

from madgraph import InvalidCmd


pjoin = os.path.join

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_v4')
if madgraph.ordering:
    set = misc.OrderedSet


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

    default_vector_size = 0
    
    
    def __init__(self, dir_path = "", opt=None):
        # cmd_options is a dictionary with all the optional argurment passed at output time
        
        # Activate some monkey patching for the helas call writer.
        helas_call_writers.HelasCallWriter.customize_argument_for_all_other_helas_object = \
                self.helas_call_writer_custom
        
        self.has_second_exporter = None
        

    # helper function for customise helas writter
    @staticmethod
    def custom_helas_call(call, arg):
        """static method to customise the way aloha function call are written
        call is the default template for the call
        arg are the dictionary used for the call
        """
        return call, arg
    
    helas_call_writer_custom = lambda x,y,z: x.custom_helas_call(y,z)


    def copy_template(self, model):
        return

    def generate_subprocess_directory(self, subproc_group, helicity_model, me=None, **opt):
    #    generate_subprocess_directory(self, matrix_element, helicity_model, me_number) [for ungrouped]
        return 0 # return an integer stating the number of call to helicity routine

    def generate_subprocess_directory_end(self, **opt):
        """ This is called only if the class is used as a second exporter. (like simd plugin)
            in that case opt contains all the local variable defined in the upstream class.
            so if multiple option exists this can lead to variable existing in some setup and not other
        """
        return 

    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[]):
        return
    
    def finalize(self,matrix_element, cmdhistory, MG5options, outputflag, second_exporter=None):
        return
    
    
    def pass_information_from_cmd(self, cmd):
        """pass information from the command interface to the exporter.
           Please do not modify any object of the interface from the exporter.
        """
        return
    
    def modify_grouping(self, matrix_element):
        return False, matrix_element
           
    def export_model_files(self, model_v4_path):
        raise Exception("V4 model not supported by this type of exporter. Please use UFO model")
        return
    
    def export_helas(self, HELAS_PATH):
        raise Exception("V4 model not supported by this type of exporter. Please use UFO model")
        return

#===============================================================================
# ProcessExporterFortran
#===============================================================================
class ProcessExporterFortran(VirtualExporter):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{}
                        }
    grouped_mode = False
    jamp_optim = False
    run_card_class = None

    def __init__(self,  dir_path = "", opt=None):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = MG5DIR
        self.dir_path = dir_path
        self.model = None
        self.beam_polarization = [True,True]
        
        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)
        self.cmd_options = self.opt['output_options']
        
        #place holder to pass information to the run_interface
        self.proc_characteristic = banner_mod.ProcCharacteristic()
        # call mother class
        super(ProcessExporterFortran,self).__init__(dir_path, opt)
        
        
    #===========================================================================
    # process exporter fortran switch between group and not grouped
    #===========================================================================
    def export_processes(self, matrix_elements, fortran_model, second_exporter=None, second_helas=None):
        """Make the switch between grouped and not grouped output"""
        
        calls = 0
        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
            # check handling for the polarization
            for m in matrix_elements:
                for me in m.get('matrix_elements'):
                    for p in me.get('processes'):
                        for beamid in [1,2]:
                            for pid in p.get_initial_ids(beamid):
                                spin = p.get('model').get_particle(pid).get('spin')
                                if spin != 2:
                                    self.beam_polarization[beamid-1] = False
                                    break

            for (group_number, me_group) in enumerate(matrix_elements):
                calls = calls + self.generate_subprocess_directory(\
                                          me_group, fortran_model, group_number,
                                          second_exporter=second_exporter, second_helas=second_helas
                                          )
        else:
             # check handling for the polarization
            self.beam_polarization = [True,True]
            for me in matrix_elements.get_matrix_elements():
                for p in me.get('processes'):
                    for beamid in [1,2]:
                        for pid in p.get_initial_ids(beamid):
                            spin = p.get('model').get_particle(pid).get('spin')
                            if spin != 2:
                                self.beam_polarization[beamid-1] = False
                                break
            for me_number, me in enumerate(matrix_elements.get_matrix_elements()):
                calls = calls + self.generate_subprocess_directory(\
                                                   me, fortran_model, me_number,
                                                   second_exporter=second_exporter, second_helas=second_helas)    

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


        run_card = banner_mod.RunCard(self.run_card_class)
        
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
        shutil.copyfile(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'),
                        pjoin(self.dir_path, 'Cards', 'run_card.dat'))
        
        
    #===========================================================================
    # write_vector_size
    #===========================================================================
    def write_vector_size(self, fsock):
        """Write the vector.inc which indicates how many event are handle in parralel."""

        try:
            vector_size = self.opt['output_options']['vector_size']
        except KeyError:
            vector_size = 1
        vector_size = banner_mod.ConfigFile.format_variable(vector_size, int, name='vector_size')
        vector_size = max(1, vector_size)

        try:
            nb_warp = self.opt['output_options']['nb_warp']
        except KeyError:
            nb_warp = 1
        nb_warp = banner_mod.ConfigFile.format_variable(nb_warp, int, name='nb_warp')
        nb_warp = max(1, nb_warp)

        text=["""C
C If VECSIZE_MEMMAX is greater than 1, a vector API is used:
C this is designed for offloading MEs to GPUs or vectorized C++,
C but it can also be used for computing MEs in Fortran.
C If VECSIZE_MEMMAX equals 1, the old scalar API is used:
C this can only be used for computing MEs in Fortran.
C
C Fortran arrays in the vector API can hold up to VECSIZE_MEMMAX
C events and are statically allocated at compile time.
C The constant value of VECSIZE_MEMMAX is fixed at codegen time
C (output madevent ... --vector_size=<VECSIZE_MEMMAX>).
C
C While the arrays can hold up to VECSIZE_MEMMAX events,
C only VECSIZE_USED (<= VECSIZE_MEMAMX) are used in Fortran loops.
C The value of VECSIZE_USED can be chosen at runtime
C (typically 8k-16k for GPUs, 16-32 for vectorized C++).
C
C The value of VECSIZE_USED represents the number of events
C handled by one call to the Fortran/cudacpp "bridge".
C This is not necessarily the number of events which are
C processed in lockstep within a single SIMD vector on CPUs
C or within a single "warp" of threads on GPUs. These parameters
C are internal to the cudacpp bridge and need not be exposed
C to the Fortran program which calls the cudacpp bridge.
C
C NB: THIS FILE CANNOT CONTAIN #ifdef DIRECTIVES
C BECAUSE IT DOES NOT GO THROUGH THE CPP PREPROCESSOR
C (see https://github.com/madgraph5/madgraph4gpu/issues/458).
C
      INTEGER WARP_SIZE
      PARAMETER (WARP_SIZE=%i)
      INTEGER NB_WARP
      PARAMETER (NB_WARP=%i)
      INTEGER VECSIZE_MEMMAX
      PARAMETER (VECSIZE_MEMMAX=%i)
              
              """ % (vector_size,nb_warp, vector_size*nb_warp)]
        fsock.writelines(text)
        return vector_size        

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
            misc.copytree(pjoin(self.mgme_dir, 'Template/LO'),
                            self.dir_path, True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/Common'), 
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
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/LO'), 
                               self.dir_path)
#            for name in misc.glob('Template/LO/*', self.mgme_dir):
#                name = os.path.basename(name)
#                filname = pjoin(self.mgme_dir, 'Template','LO',name)
#                if os.path.isfile(filename):
#                    files.cp(filename, pjoin(self.dir_path,name))
#                elif os.path.isdir(filename):
#                     misc.copytree(filename, pjoin(self.dir_path,name), True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/Common'), 
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
            if 'MADGRAPH_BASE' in os.environ:
                misc.call([pjoin('bin', 'internal', 'clean_template'),
                                 '--web'], cwd=self.dir_path)
            else:
                try:
                    misc.call([pjoin('bin', 'internal', 'clean_template')], \
                                                                       cwd=self.dir_path)
                except Exception as why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(self.dir_path),why))

            #Write version info
            MG_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])

        # add the makefile in Source directory 
        # now moved to finalize

        self.write_vector_size(writers.FortranWriter(pjoin(self.dir_path, 'Source','vector.inc')))
        
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
        start = time.time()
        logger.info('Generating MadAnalysis5 default cards tailored to this process')
        try:
            MA5_interpreter = common_run_interface.CommonRunCmd.\
                          get_MadAnalysis5_interpreter(MG5DIR,ma5_path,loglevel=100)
        except (Exception, SystemExit) as e:
            logger.warning('Fail to create a MadAnalysis5 instance. Therefore the default analysis with MadAnalysis5 will be empty')
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
        stop = time.time()
        if stop-start >1:
            logger.info('Cards created in %.2fs' % (stop-start))

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
    def finalize(self, matrix_elements, history='', mg5options={}, flaglist=[], second_exporter=None):
        """Function to finalize v4 directory, for inheritance.""" 

        filename = pjoin(self.dir_path,'Source','makefile')
        if not second_exporter:
            self.write_source_makefile(writers.FileWriter(filename), self.model)
        else:
           replace_dict = self.write_source_makefile(None, model=self.model)
           second_exporter.write_source_makefile(writers.FileWriter(filename), model=self.model, default=replace_dict)  

        if second_exporter:
            self.has_second_exporter = second_exporter

        if self.has_second_exporter and hasattr(self.has_second_exporter, 'run_card_class'):
            with misc.TMP_variable(self, 'run_card_class', self.has_second_exporter.run_card_class):
                self.create_run_card(matrix_elements, history)
        else:
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

        # this is for LHAPDF
        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"))        
        #ff = open(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"),"w")
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_wrap_lhapdf.f"),"r").read()
    
        NLO = isinstance(self, madgraph.iolibs.export_fks.ProcessExporterFortranFKS)
        ff.writelines(template % changer, {'LO': not NLO})

        # this is for eMELA
        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_emela.f"))        
        #ff = open(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"),"w")
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_wrap_emela.f"),"r").read()
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

        # add file for EWA 
        template = open(pjoin(MG5DIR,'madgraph/iolibs/template_files/madevent_electroweakFlux.inc')).read()
        fsock = open(pjoin(self.dir_path, 'Source', 'ElectroweakFlux.inc'),'w')
        fsock.write(template % {'MW': 'wmass','MZ':'zmass'})                 
        fsock.close() 
        ln(pjoin(self.dir_path, 'Source', 'ElectroweakFlux.inc'), self.dir_path + '/Source/PDF')


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
                                         me_number, **opt):
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
                '$(LIBDIR)libgammaUPC.$(libext)',
                '$(LIBDIR)libmodel.$(libext)',
                '$(LIBDIR)libcernlib.$(libext)',
                '$(LIBDIR)libbias.$(libext)']

    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model=None):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = ' '.join(['$(LIBRARIES)']+self.get_source_libraries_list())
        if self.opt['model'] == 'mssm' or self.opt['model'].startswith('mssm-'):
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc vector.inc\n\tcd MODEL; make
MODEL/MG5_param.dat: ../Cards/param_card.dat\n\t../bin/madevent treatcards param
param_card.inc: MODEL/MG5_param.dat\n\t../bin/madevent treatcards param\n'''
        else:
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc vector.inc\n\tcd MODEL; make    
param_card.inc: ../Cards/param_card.dat\n\t../bin/madevent treatcards param\n'''
        
        replace_dict= {'libraries': set_of_lib, 
                       'model':model_line,
                       'additional_dsample': '',
                       'additional_dependencies':'',
                       'additional_clean':'',
                       'running': ''} 

        if self.opt['running']:
            replace_dict['running'] ="  $(LIBDIR)librunning.$(libext): RUNNING\n\tcd RUNNING; make"
            replace_dict['libraries'] += " $(LIBDIR)librunning.$(libext) "
        
        if writer:
            text = open(path).read() % replace_dict
            writer.write(text)
            
        return replace_dict

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
        self.model = model
        # create the MODEL
        write_dir=pjoin(self.dir_path, 'Source', 'MODEL')
        self.opt['exporter'] = self.__class__
        if 'vector_size' in self.opt:
            self.opt['output_options']['vector_size'] = self.opt['vector_size']
        if 'vector_size' not in self.opt['output_options']:
            self.opt['output_options']['vector_size'] = self.default_vector_size

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
            try:
                with misc.MuteLogger(['madgraph.models'], [60]):
                    aloha_model = create_aloha.AbstractALOHAModel(os.path.basename(model.get('modelpath')))
            except (ImportError, UFOError):
                aloha_model = create_aloha.AbstractALOHAModel(model.get('modelpath'))
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
        if any([any([tag.startswith('L') for tag in d[1]]) for d in wanted_lorentz]):
            cp(MG5DIR + '/aloha/template_files/aloha_functions_loop.f', 
                                                 write_dir+'/aloha_functions.f')
            aloha_model.loop_mode = False
        else:
            if aloha.unitary_gauge !=3:
                cp(MG5DIR + '/aloha/template_files/aloha_functions.f', 
                                                 write_dir+'/aloha_functions.f')
            else:
                cp(MG5DIR + '/aloha/template_files/aloha_functions_fd.f', 
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
        if info and 'version' in info and  'date' in info:
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


    def get_helicity_lines(self, matrix_element,array_name='NHEL', add_nb_comb=False):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line_list = []
        i = 0            
        if add_nb_comb:
            spins = matrix_element.get_spin_state()
            spins.insert(0, len(spins))
            helicity_line_list.append(\
                ("DATA ("+array_name+"(I,0),I=1,%d) /" + \
                 ",".join(['%2r'] * (len(spins)-1)) + "/") % tuple(spins))
            
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
        int_list = list(range(1, nexternal + 1))

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
                if user_sqso == 'WEIGHTED' :
                    logger.debug('WEIGHTED^2%s%s encoutered. Please check behavior for' + \
                            'https://bazaar.launchpad.net/~maddevelopers/mg5amcnlo/3.0.1/revision/613', \
                            (process.get_squared_order_type(user_sqso), sqsos[split_orders.index(user_sqso)]))
                if user_sqso not in split_orders:
                    is_a_match = False
                elif (process.get_squared_order_type(user_sqso) =='==' and \
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
            for k in range(0, len(order), n):
                ret_list.append("DATA (%s(%3r,i),i=%3r,%3r) /%s/" % \
                  (array_name,index + 1, k + 1, min(k + n, len(order)),
                              ','.join(["%5r" % i for i in order[k:k + n]])))
        return ret_list
    
    def format_integer_list(self, list, name, n=5):
        """ Return an initialization of the python list in argument following 
        the fortran syntax using the data keyword assignment, filling an array 
        of name 'name'. It splits rows in chunks of size n."""
        
        ret_list = []
        for k in range(0, len(list), n):
            ret_list.append("DATA (%s(i),i=%3r,%3r) /%s/" % \
                  (name, k + 1, min(k + n, len(list)),
                                  ','.join(["%5r" % i for i in list[k:k + n]])))
        return ret_list

#    def write_namelist_file(self, matrix_element, dirpath):
#
#        fsock = open(pjoin(dirpath, 'namelist.def'), 'w')
#
#        fsock.write(' &NM_CF\n')
#        if not matrix_element.get('color_matrix'):
#            fsock.write('  CF = 1\n')
#        else:
#            cf = []
#            for index, denominator in \
#                enumerate(matrix_element.get('color_matrix').\
#                                                 get_line_denominators()): 
#                num_list = matrix_element.get('color_matrix').\
#                                            get_line_numerators(index, denominator)
#                num_list[index] /= 2
#                cf += [str(int(2*coeff)) for coeff in num_list[index:]]
#            fsock.write('  CF = %s\n' % (','.join(cf))) 
#            fsock.write(' /\n')



    def get_color_data_lines(self, matrix_element, n=128):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return ["DATA Denom/1/", "DATA CF/1/"]

        ret_list = []
        my_cs = color.ColorString()
        denominator = min(matrix_element.get('color_matrix').get_line_denominators())
        ret_list.append("DATA Denom/%i/" % denominator)

        cf_index = 0
        col_basis = matrix_element.get('color_matrix')._col_basis1
        is_asym = matrix_element.get('color_matrix')._col_basis1 is not matrix_element.get('color_matrix')._col_basis2
        for index in range(len(col_basis)):
            num_list = matrix_element.get('color_matrix').get_line_numerators(index, denominator)
            assert all(int(i) == i for i in num_list)
            if is_asym:
                min_k = 0
            else:
                min_k = index # only include the upper diagonal
            for k in range(min_k, len(num_list), n):
                chunk = num_list[k:k+n]
                if is_asym:
                    ret_list.append("DATA (CF(i,%3r),i=%3r,%3r) /%s/" % \
                                    (index+1, k + 1, k+len(chunk),
                                     ','.join([("%i" % (int(i))) for i in chunk])))  
                else: 
                    ret_list.append("DATA (CF(i),i=%3r,%3r) /%s/" % \
                                    (cf_index+1, cf_index + len(chunk),
                                     ','.join([("%i" % ((1 if (k==index and pos==0) else 2)*int(i))) for pos,i in enumerate(chunk)])))
                cf_index += len(chunk)

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
                else:
                    self.proc_characteristic['single_color'] = False

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

    @staticmethod
    def get_multi_channel_dictionary(diagrams, config_map): 
        """diagrams should be from matrix_element.get('diagrams')"""


        config_to_diag_dict = {}
        if config_map:
            # In this case, we need to sum up all amplitudes that have
            # identical topologies, as given by the config_map (which
            # gives the topology/config for each of the diagrams
            # Combine the diagrams with identical topologies
            for idiag, diag in enumerate(diagrams):
                if config_map[idiag] == 0:
                    continue
                try:
                    config_to_diag_dict[config_map[idiag]].append(idiag)
                except KeyError:
                    config_to_diag_dict[config_map[idiag]] = [idiag]
        else:
            # Get minimum legs in a vertex
            vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
                diagrams if diag.get_vertex_leg_numbers()!=[]]
            minvert = min(vert_list) if vert_list!=[] else 0

            for idiag, diag in enumerate(diagrams):
                # Ignore any diagrams with 4-particle vertices.
                if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                    continue
                config_to_diag_dict[config_map[idiag]] = [idiag]

        return  config_to_diag_dict


    def get_amp2_lines(self, matrix_element, config_map = [], replace_dict=None):
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
            config_to_diag_dict = self.get_multi_channel_dictionary(diagrams, config_map)
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
                    amp = "(%s)*dconjg(%s)" % (amp, amp)
                else:
                    amp = "%s*dconjg(%s)" % (amp, amp)
                
                line =  line + "%s" % (amp)
                #line += " * get_channel_cut(p, %s) " % (config)
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
                raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_coefs")

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
            if six.PY2:    
                commondenom=abs(reduce(fractions.gcd, coefs_list).denominator)
            else:
                commondenom=abs(reduce(math.gcd, coefs_list).denominator)
            num_list=[(coefficient*commondenom).numerator \
                      for coefficient in coefs_list]
            res_list.append("DATA NCONTRIBAMPS%s(%i)/%i/"%(tag_letter,\
                                                         index+1,len(num_list)))
            res_list.append("DATA DENOMCCOEF%s(%i)/%i/"%(tag_letter,\
                                                         index+1,commondenom))
            if color_basis:
                my_cs.from_immutable(sorted(color_basis.keys())[index])
                res_list.append("C %s" % repr(my_cs))
            for k in range(0, len(num_list), n):
                res_list.append("DATA (NUMCCOEF%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in num_list[k:k + n]])))
                res_list.append("DATA (AMPNUMBERS%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in ampnumbers_list[k:k + n]])))
                pass
        return res_list


    def get_JAMP_lines_split_order(self, col_amps, split_order_amps, 
          split_order_names=None, JAMP_format="JAMP(%s,{0})", AMP_format="AMP(%s)"):
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
                raise MadGraph5Error(error_msg%'col_amps')
        else:
            raise MadGraph5Error(error_msg%'col_amps')
        
        # Verify the sanity of the split_order_amps and split_order_names args
        if isinstance(split_order_amps,list):
            for elem in split_order_amps:
                if len(elem)!=2:
                    raise MadGraph5Error(error_msg%'split_order_amps')
                # Check the first element of the two lists to make sure they are
                # integers, although in principle they should all be integers.
                if not isinstance(elem[0],tuple) or \
                   not isinstance(elem[1],tuple) or \
                   not isinstance(elem[0][0],int) or \
                   not isinstance(elem[1][0],int):
                    raise MadGraph5Error(error_msg%'split_order_amps')
        else:
            raise MadGraph5Error(error_msg%'split_order_amps')
        
        if not split_order_names is None:
            if isinstance(split_order_names,list):
                # Should specify the same number of names as there are elements
                # in the key of the split_order_amps.
                if len(split_order_names)!=len(split_order_amps[0][0]):
                    raise MadGraph5Error(error_msg%'split_order_names')
                # Check the first element of the list to be a string
                if not isinstance(split_order_names[0],str):
                    raise MadGraph5Error(error_msg%'split_order_names')                    
            else:
                raise MadGraph5Error(error_msg%'split_order_names')                
        
        # Now scan all contributing orders to be individually computed and 
        # construct the list of color_amplitudes for JAMP to be constructed
        # accordingly.
        res_list=[]
        max_tmp = 0
        for i, amp_order in enumerate(split_order_amps):
            col_amps_order = []
            for jamp in color_amplitudes:
                col_amps_order.append([col_amp for col_amp in jamp if col_amp[1] in amp_order[1]])
            if split_order_names:
                res_list.append('C JAMPs contributing to orders '+' '.join(
                              ['%s=%i'%order for order in zip(split_order_names,
                                                                amp_order[0])]))
            if self.opt['export_format'] in ['madloop_matchbox']:
                res_list.extend(self.get_JAMP_lines(col_amps_order,
                                   JAMP_format=JAMP_format.format(str(i+1)),
                                   JAMP_formatLC="LN"+JAMP_format.format(str(i+1)))[0])
            else:
                toadd, nb_tmp = self.get_JAMP_lines(col_amps_order,
                                   JAMP_format=JAMP_format.format(str(i+1)))
                res_list.extend(toadd)
                max_tmp = max(max_tmp, nb_tmp)         

        return res_list, max_tmp


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
                raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_lines")
        else:
            raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_lines")

        all_element = {}
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
                diff_fracs = misc.make_unique(list_fracs)
                if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                    common_factor = True
                    global_factor = diff_fracs[0]
                    res = res + '%s(' % self.coeff(1, global_factor, False, 0)
                
                # loop for JAMP
                for (coefficient, amp_number) in coefs:
                    if not coefficient:
                        continue
                    value = (1j if coefficient[2] else 1)* coefficient[0] * coefficient[1] * fractions.Fraction(3)**coefficient[3]
                    if (i+1, amp_number) not in all_element:
                        all_element[(i+1, amp_number)] = value
                    else:
                        all_element[(i+1, amp_number)] += value
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
        
        if 'jamp_optim' in self.cmd_options:
            jamp_optim = banner_mod.ConfigFile.format_variable(self.cmd_options['jamp_optim'], bool, 'jamp_optim')
        else:
            # class default
            jamp_optim = self.jamp_optim
                
        if not jamp_optim:
            return res_list, 0
        else:
            saved = list(res_list)
        
        if len(all_element) > 1000:
            logger.info("Computing Color-Flow optimization [%s term]", len(all_element))
            start_time = time.time()
        else: 
            start_time = 0
        
        res_list = []
        
        self.myjamp_count = 0
        for key in all_element:
            all_element[key] = complex(all_element[key])
        new_mat, defs = self.optimise_jamp(all_element)
        if start_time:
            logger.info("Color-Flow passed to %s term in %ss. Introduce %i contraction", len(new_mat), int(time.time()-start_time), len(defs))
        
        
        #misc.sprint("number of iteration", self.myjamp_count)
        def format(frac):
            if isinstance(frac, fractions.Fraction):
                if frac.denominator == 1:
                    return str(frac.numerator)
                else:
                    return "%id0/%id0" % (frac.numerator, frac.denominator)
            elif frac.real == frac:
                #misc.sprint(frac.real, frac)
                return ('%.15e' % frac.real).replace('e','d')
                #str(float(frac.real)).replace('e','d')
            else:
                return ('(%.15e,%.15e)' % (frac.real, frac.imag)).replace('e','d')
                #str(frac).replace('e','d').replace('j','*imag1')
                
        
        
        for i, amp1, amp2, frac, nb in defs:
            if amp1 > 0:
                amp1 = AMP_format % amp1
            else:
                amp1 = "TMP_JAMP(%d)" % -amp1
            if amp2 > 0:
                amp2 = AMP_format % amp2
            else:
                amp2 = "TMP_JAMP(%d)" % -amp2
            
            if frac not in  [1., -1]:
                res_list.append(' TMP_JAMP(%d) = %s + (%s) * %s ! used %d times' % (i,amp1, format(frac), amp2, nb))                
            elif frac == 1.:
                res_list.append(' TMP_JAMP(%d) = %s +  %s ! used %d times' % (i,amp1, amp2, nb))  
            else:
                res_list.append(' TMP_JAMP(%d) = %s - %s ! used %d times' % (i,amp1, amp2, nb))  

        jamp_res = collections.defaultdict(list)
        max_jamp=0
        for (jamp, var), factor in new_mat.items():
            if var > 0:
                name = AMP_format % var
            else:
                name = "TMP_JAMP(%d)" % -var
            if factor not in [1.]:
                jamp_res[jamp].append("(%s)*%s" % (format(factor), name))
            elif factor ==1:
                jamp_res[jamp].append("%s" % (name))
            max_jamp = max(max_jamp, jamp)
        
        
        for i in range(1,max_jamp+1):
            name = JAMP_format % i
            if not jamp_res[i]:
                res_list.append(" %s = 0d0" %(name))
            else:
                res_list.append(" %s = %s" %(name, '+'.join(jamp_res[i])))

        return res_list, len(defs)

    def optimise_jamp(self, all_element, nb_line=0, nb_col=0, added=0):
        """ optimise problem of type Y = A X
                A is a matrix (all_element)
                X is the fortran name of the input.
            The code iteratively add sub-expression jtemp[sub_add]
            and recall itself (this is add to the X size)
        """
        self.myjamp_count +=1

        if not nb_line:
            for i,j in all_element:
                if i+1 > nb_line:
                    nb_line = i+1
                if j+1> nb_col:
                    nb_col = j+1  
            if nb_col > 600 and added==0:
                all_element1, all_element2 = {}, {}
                for (k1,k2) in all_element:
                    if k2 >= nb_col//2:
                        all_element2[(k1,1+k2-(nb_col//2))] = all_element[(k1,k2)]
                    else:
                        all_element1[(k1,k2)] = all_element[(k1,k2)]

                all_element1, newdef1 = self.optimise_jamp(all_element1)
                nb_added1 = len(newdef1)

                all_element2, newdef2 = self.optimise_jamp(all_element2)

                for (k1,k2) in all_element2:
                    if k2 >= 0:
                        all_element1[(k1,k2+(nb_col//2)-1)] = all_element2[(k1,k2)]
                    if k2 < 0: 
                        all_element1[(k1,k2-nb_added1)] = all_element2[(k1,k2)]
                # new_def format: added,j1,j2,R, max_count
                for k, j1,j2, R, c in newdef2:
                    if j2 > 0:
                        k2 = j2+nb_col//2 -1
                    else:
                        k2 = j2-nb_added1 
                    if j1 > 0:
                        k1 = j1+nb_col//2 -1
                    else:
                        k1 = j1-nb_added1
                    newdef1.append((k+nb_added1, k1, k2, R, c))
                if newdef1:
                    all_element, new_def = self.optimise_jamp(all_element1, nb_line=0, nb_col=0, added=len(newdef1))
                    newdef1 = newdef1 + new_def
                return all_element, newdef1

        max_count = 0
        all_index = []
        operation = collections.defaultdict(lambda: collections.defaultdict(int))
        for (i,j1), v1 in all_element.items():
            ratios = [(j2,all_element.get((i,j2), 0)/v1) for j2 in range(j1+1, nb_col) if all_element.get((i,j2), 0)]
            for j2, R in ratios:                   
                operation[(j1,j2)][R] +=1 
                if operation[(j1,j2)][R] > max_count:
                    max_count = operation[(j1,j2)][R]
                    all_index = [(j1,j2, R)]
                elif operation[(j1,j2)][R] == max_count:
                    all_index.append((j1,j2, R))

        if max_count <= 1:
            return all_element, []

        to_add = []
        for index in all_index:
            j1,j2,R = index
            first = True
            for i in range(nb_line):
                v1 = all_element.get((i,j1), 0)
                v2 = all_element.get((i,j2), 0)
                if not v1 or not v2: 
                    continue
                if v2/v1 == R:
                    if first:
                        first = False
                        added +=1
                        to_add.append((added,j1,j2,R, max_count))
                        
                    all_element[(i,-added)] = v1
                    del all_element[(i,j1)] #= 0
                    del all_element[(i,j2)] #= 0 

        logger.log(5,"Define %d new shortcut reused %d times", len(to_add), max_count)
        new_element, new_def =  self.optimise_jamp(all_element, nb_line=nb_line, nb_col=nb_col, added=added)
        for one_def in to_add:
            new_def.insert(0, one_def)
        return new_element, new_def   
           
           
            
            

    def get_pdf_lines(self, matrix_element, ninitial, subproc_group = False, vector=False):
        """Generate the PDF lines for the auto_dsig.f file"""

        processes = matrix_element.get('processes')
        model = processes[0].get('model')

        pdf_definition_lines = ""
        ee_pdf_definition_lines = ""
        pdf_data_lines = ""
        pdf_lines = ""

        if vector:
            pdf_definition_lines_vec = ""
            pdf_data_lines_vec = ""
            pdf_lines = """ NB_WARP_USED = VECSIZE_USED / WARP_SIZE
        IF( NB_WARP_USED * WARP_SIZE .NE. VECSIZE_USED ) THEN
        WRITE(*,*) 'ERROR: NB_WARP_USED * WARP_SIZE .NE. VECSIZE_USED',
     &    NB_WARP_USED, WARP_SIZE, VECSIZE_USED
        STOP
        ENDIF

        DO CURR_WARP=1, NB_WARP_USED
        IF(IMIRROR_VEC(CURR_WARP).EQ.1)THEN
          IB(1) = 1
          IB(2) = 2
        ELSE
          IB(1) = 2
          IB(2) = 1
        ENDIF
        DO IWARP=1, warp_SIZE
          IVEC = (CURR_WARP-1)*WARP_SIZE+IWARP
          """



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

            if tuple(initial_states) in [([-11],[11]), ([11],[-11]), ([-13],[13]),([13],[-13])]:
                dressed_lep = True
            else:
                dressed_lep = False
            ee_pdf_definition_lines += "DOUBLE PRECISION dummy_components(n_ee)\n"

   
            # Prepare all variable names
            pdf_codes = dict([(p, model.get_particle(p).get_name()) for p in \
                              sum(initial_states,[])])
            for key,val in pdf_codes.items():
                pdf_codes[key] = val.replace('~','x').replace('+','p').replace('-','m')

            # Set conversion from PDG code to number used in PDF calls
            pdgtopdf = {21: 0, 22: 7}

            # Fill in missing entries of pdgtopdf
            for pdg in sum(initial_states,[]):
                if not pdg in pdgtopdf and not pdg in list(pdgtopdf.values()):
                    pdgtopdf[pdg] = pdg
                elif pdg not in pdgtopdf and pdg in list(pdgtopdf.values()):
                    # If any particle has pdg code 7, we need to use something else
                    pdgtopdf[pdg] = 6000000 + pdg
                    
            # Get PDF variable declarations for all initial states
            if vector:
                vector_ext1 = '(VECSIZE_MEMMAX)' # pass to an array from a double
                vector_ext2 = ', VECSIZE_MEMMAX' # add a dimenion
            else:
                vector_ext1, vector_ext2 = '',''

            for i in [0,1]:
                pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d%s" % (pdf_codes[pdg],i+1, vector_ext1) \
                                                 for pdg in \
                                                 initial_states[i]]) + \
                                                 "\n"
                
                ee_pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d_components(n_ee %s)" % (pdf_codes[pdg],i+1, vector_ext2) \
                                                 for pdg in \
                                                 initial_states[i] if abs(pdg) in [11,13]]) + \
                                                 "\n"
                

            # Get PDF data lines for all initial states
            for i in [0,1]:
                pdf_data_lines += "DATA " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in initial_states[i]]) + \
                                                 "/%d*1D0/" % len(initial_states[i]) + \
                                                 "\n"
                if vector:
                    pdf_data_lines_vec += "DATA " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in initial_states[i]]) + \
                                                 "/%s/" % ','.join(['VECSIZE_MEMMAX*1D0']* len(initial_states[i])) + \
                                                 "\n"


            # Get PDF lines for UPC (non-factorized PDF)
            if 22 in initial_states[0] and 22 in initial_states[1]:
                if subproc_group:
                    pdf_lines = pdf_lines + \
                        "IF (ABS(LPP(IB(1))).EQ.2.AND.ABS(LPP(IB(2))).EQ.2.AND.(PDLABEL(1:4).EQ.'edff'.OR.PDLABEL(1:4).EQ.'chff'))THEN\n"
                    pdf_lines = pdf_lines + \
                        ("%s%d=PHOTONPDFSQUARE(XBK(IB(1)),XBK(IB(2)))\n%s%d=DSQRT(%s%d)\n%s%d=%s%d\n") % \
                        (pdf_codes[22],1,pdf_codes[22],2,pdf_codes[22],1,pdf_codes[22],1,pdf_codes[22],2)
                else:
                    pdf_lines = pdf_lines + \
                        "IF (ABS(LPP(1)).EQ.2.AND.ABS(LPP(2)).EQ.2.AND.(PDLABEL(1:4).EQ.'edff'.OR.PDLABEL(1:4).EQ.'chff'))THEN\n"
                    pdf_lines = pdf_lines + \
                        ("%s%d=PHOTONPDFSQUARE(XBK(1),XBK(2))\n%s%d=DSQRT(%s%d)\n%s%d=%s%d\n") % \
                        (pdf_codes[22],1,pdf_codes[22],2,pdf_codes[22],1,pdf_codes[22],1,pdf_codes[22],2)
                pdf_lines = pdf_lines + "ELSE\n"

            # Get PDF lines for all different initial states
            for i, init_states in enumerate(initial_states):
                if subproc_group:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(IB(%d))).GE.1) THEN\n!LP=SIGN(1,LPP(IB(%d)))\n" \
                                 % (i + 1, i + 1)
                    if not vector:
                        if i == 0:
                            pdf_lines = pdf_lines + \
                                "if (DSQRT(Q2FACT(IB(1))).eq.0d0) then\n" +\
                                "  qscale=0d0\n"+\
                                "    do i=3,nexternal\n"+\
                                "      Qscale=Qscale+dsqrt(max(0d0,(PP(0,i)+PP(3,i))*(PP(0,i)-PP(3,i))))\n"+\
                                "    enddo\n"+\
                                "   qscale=qscale/2d0\n"+\
                                "else\n"+\
                                "   qscale=DSQRT(Q2FACT(1))\n"+\
                                "endif\n"
                        else:
                            pdf_lines = pdf_lines + \
                                "if (DSQRT(Q2FACT(IB(2))).ne.0d0) then\n" +\
                                "   qscale=DSQRT(Q2FACT(2))\n" +\
                                "endif\n"
                else:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(%d)) .GE. 1) THEN\n!LP=SIGN(1,LPP(%d))\n" \
                                 % (i + 1, i + 1)
                    
                for nbi,initial_state in enumerate(init_states):
                    if initial_state in list(pdf_codes.keys()):

                        data = {'part':pdf_codes[initial_state],
                                'beam' : i+1,
                                'pdg': pdgtopdf[initial_state],
                                'vecid': ''
                            }
                        if vector:
                            data['vecid'] = ', IVEC'

                        if vector and subproc_group:
                            template  = "%(part)s%(beam)d(IVEC)=PDG2PDF(LPP(IB(%(beam)d)),%(pdg)d, IB(%(beam)d)," + \
                                         "ALL_XBK(IB(%(beam)d),IVEC),DSQRT(ALL_Q2FACT(%(beam)d, IVEC)))\n"
                            #if dressed_lep and self.opt['vector_size']:
                            #    logger.warning("vector code for lepton pdf not implemented. We removed the option to run dressed lepton")
                            #    self.proc_characteristic['limitations'].append('dressed_ee')
                            #    dressed_lep = False
                        elif subproc_group:
                            template = "%(part)s%(beam)d=PDG2PDF(LPP(IB(%(beam)d)),%(pdg)d, IB(%(beam)d)," + \
                                         "XBK(IB(%(beam)d)), QSCALE)\n"
                        elif vector:
                            template = "%(part)s%(beam)d(IVEC)=PDG2PDF(LPP(%(beam)d),%(pdg)d, %(beam)d," + \
                                         "ALL_XBK(%(beam)d,IVEC),DSQRT(ALL_Q2FACT(%(beam)d,IVEC)))\n"
                            #if dressed_lep:
                            #    raise Exception("vector code for lepton pdf not implemented")
                        else:
                            template = "%(part)s%(beam)d=PDG2PDF(LPP(%(beam)d),%(pdg)d, %(beam)d," + \
                                         "XBK(%(beam)d),DSQRT(Q2FACT(%(beam)d)))\n"
                        if dressed_lep:
                            template += "IF (PDLABEL.EQ.'dressed') %(part)s%(beam)d_components(1:4 %(vecid)s) = ee_components(1:4)\n"

                        pdf_lines = pdf_lines + template % data

                pdf_lines = pdf_lines + "ENDIF\n"

            if 22 in initial_states[0] and 22 in initial_states[1]:
                pdf_lines = pdf_lines + "ENDIF\n"

            if not vector:
                # Add up PDFs for the different initial state particles

                pdf_lines = pdf_lines + "PD(0) = 0d0\nIPROC = 0\n"
                for proc in processes:
                    process_line = proc.base_string()
                    pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                    pdf_lines = pdf_lines + "\nPD(IPROC)="
                    comp_list = []
                    for ibeam in [1, 2]:
                        initial_state = proc.get_initial_pdg(ibeam)
                        if initial_state in list(pdf_codes.keys()):
                            pdf_lines = pdf_lines + "%s%d*" % \
                                        (pdf_codes[initial_state], ibeam)
                            comp_list.append("%s%d" % (pdf_codes[initial_state], ibeam))
                        else:
                            pdf_lines = pdf_lines + "1d0*"
                            comp_list.append("DUMMY")
                    # Remove last "*" from pdf_lines
                    pdf_lines = pdf_lines[:-1] + "\n"

                    # this is for the lepton collisions with electron luminosity 
                    # put here "%s%d_components(i_ee)*%s%d_components(i_ee)"
                    if dressed_lep:
                        pdf_lines += "if (pdlabel.eq.'dressed')" + \
                             "PD(IPROC)=ee_comp_prod(%s_components,%s_components)\n" % \
                             tuple(comp_list)
                    pdf_lines = pdf_lines + "PD(0)=PD(0)+DABS(PD(IPROC))\n"
                    
                    if not dressed_lep:
                        ee_pdf_definition_lines = ""
            else:
                # Add up PDFs for the different initial state particles
                pdf_lines += "ENDDO ! IWARP LOOP\n"
                pdf_lines += "ENDDO ! CURRWARP LOOP\n"
                pdf_lines = pdf_lines + "ALL_PD(0,:) = 0d0\nIPROC = 0\n"
                for proc in processes:
                    comp_list = []
                    process_line = proc.base_string()
                    pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                    pdf_lines += '\n   DO IVEC=1, VECSIZE_USED'
                    pdf_lines = pdf_lines + "\nALL_PD(IPROC,IVEC)="
                    for ibeam in [1, 2]:
                        initial_state = proc.get_initial_pdg(ibeam)
                        if initial_state in list(pdf_codes.keys()):
                            pdf_lines = pdf_lines + "%s%d(IVEC)*" % \
                                        (pdf_codes[initial_state], ibeam)
                            comp_list.append("%s%d" % (pdf_codes[initial_state], ibeam))
                        else:
                            pdf_lines = pdf_lines + "1d0*"
                            comp_list.append("DUMMY")
                    # Remove last "*" from pdf_lines
                    pdf_lines = pdf_lines[:-1] + "\n"
                    # this is for the lepton collisions with electron luminosity 
                    # put here "%s%d_components(i_ee)*%s%d_components(i_ee)"
                    if dressed_lep:
                        pdf_lines += "if (pdlabel.eq.'dressed')" + \
                             "ALL_PD(IPROC,IVEC)=ee_comp_prod(%s_components(1,IVEC),%s_components(1,IVEC))\n" % \
                             tuple(comp_list)
                    pdf_lines = pdf_lines + "ALL_PD(0,IVEC)=ALL_PD(0,IVEC)+DABS(ALL_PD(IPROC,IVEC))\n"
                    pdf_lines += '\n    ENDDO\n'
                    if not dressed_lep:
                        ee_pdf_definition_lines = ""

        # Remove last line break from the return variables                
        if vector:
            return pdf_definition_lines[:-1], pdf_data_lines_vec[:-1], pdf_lines[:-1], ee_pdf_definition_lines
        else:
            return pdf_definition_lines[:-1], pdf_data_lines[:-1], pdf_lines[:-1], ee_pdf_definition_lines

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
                schannels = list(zip(*[s for s,t in stchannels]))
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
                elif verts in tchannels:
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
            raise MadGraph5Error('No Fortran Compiler detected! Please install one')
        logger.info('Use Fortran compiler ' + f77_compiler)
        
        
        # Check for compiler. 1. set default.
        if default_compiler['f2py']:
            f2py_compiler = default_compiler['f2py']
        else:
            f2py_compiler = ''
        # Try to find the correct one.
        if default_compiler['f2py'] and misc.which(default_compiler['f2py']):
            f2py_compiler = default_compiler['f2py']
        elif misc.which('f2py%d.%d' %(sys.version_info.major, sys.version_info.minor)):
            f2py_compiler = 'f2py%d.%d' %(sys.version_info.major, sys.version_info.minor)
        elif misc.which('f2py%d' %(sys.version_info.major)):
            f2py_compiler = 'f2py%d' %(sys.version_info.major)            
        elif misc.which('f2py'):
            f2py_compiler = 'f2py'


        to_replace = {'fortran': f77_compiler, 'f2py': f2py_compiler}
        
        
        self.replace_make_opt_f_compiler(to_replace)
        # Replace also for Template but not for cluster
        if 'MADGRAPH_DATA' not in os.environ and ReadWrite:
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
            out = out.decode(errors='ignore')
            if 'clang' in str(out) and  misc.which('clang'):
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
            raise MadGraph5Error('No c++ Compiler detected! Please install one')
        logger.info('Use c++ compiler ' + compiler)
        self.replace_make_opt_c_compiler(compiler)
        # Replace also for Template but not for cluster
        if 'MADGRAPH_DATA' not in os.environ and ReadWrite and \
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

        # for MOJAVE remove the MACFLAG:
        if is_clang:
            import platform
            version, _, _ = platform.mac_ver()
            if not version:# not linux 
                majversion = 14 # set version to remove MACFLAG
            else:
                majversion, version = [int(x) for x in version.split('.',3)[:2]]

            if majversion >= 11 or (majversion ==10 and version >= 14):
                for_update['MACFLAG'] = '-mmacosx-version-min=10.8' if is_lc else ''

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
    jamp_optim = True
    default_vector_size = 0

    def __init__(self, *args,**opts):
        """add the format information compare to standard init"""
        
        if 'format' in opts:
            self.format = opts['format']
            del opts['format']
        else:
            self.format = 'standalone'
        
        self.prefix_info = {}
        ProcessExporterFortran.__init__(self, *args, **opts)

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        self.model = model
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
                    pjoin(self.dir_path, 'SubProcesses', 'makefileP'))

        if model['running_elements']:
            fsock = open( pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f_sp'), 'r')
            text = fsock.read()
            fsock.close()
            fsock = open(pjoin(self.dir_path, 'SubProcesses', 'makefileP'),'w')
            text = text.replace('LINKLIBS =  -L../../lib/', 'LINKLIBS =  -L../../lib/ -lrunning')
            text = text.replace('LIBS =', 'LIBS = $(LIBDIR)/librunning.$(libext)')
            fsock.write(text)
            fsock.close()
        else:
            # Add file in SubProcesses
            shutil.copy(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f_sp'), 
                    pjoin(self.dir_path, 'SubProcesses', 'makefileP'))
        
        if self.format == 'standalone':
            shutil.copy(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'check_sa.f'), 
                    pjoin(self.dir_path, 'SubProcesses', 'check_sa.f'))
                        
        # Add file in Source
        shutil.copy(pjoin(temp_dir, 'Source', 'make_opts'), 
                    pjoin(self.dir_path, 'Source'))   

        # add the makefile 
        filename = pjoin(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FileWriter(filename),model)          

        # add default vector.inc for SA code
        #filename = pjoin(self.dir_path, 'Source', 'vector.inc')
        #self.write_vector_inc_for_sa(writers.FileWriter(filename), model)

    #===========================================================================
    # handling vector.inc (needed by the model) for SA (assuming no batch)
    #===========================================================================     
    #def write_vector_inc_for_sa(self, writer, model):
    #    """ """
#
#        text="""
#        INTEGER WARP_SIZE
#       PARAMETER (WARP_SIZE=1)
#       INTEGER NB_WARP
#       PARAMETER (NB_WARP=1)
#       INTEGER VECSIZE_MEMMAX
#       PARAMETER (VECSIZE_MEMMAX=1)
#
#"""
#        writer.write(text)
#        return

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
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===========================================================================
    def write_procdef_mg5(self, file_pos, modelname, process_str):
        """ write an equivalent of the MG4 proc_card in order that all the Madevent
        Perl script of MadEvent4 are still working properly for pure MG5 run.
        Not needed for StandAlone so just return
        """
        
        return


    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS and MODEL directories, to set up
        everything for running standalone
        """

        source_dir = pjoin(self.dir_path, "Source")
        logger.info("Running make for Source directory")
        try:
            misc.compile(cwd=source_dir, mode='fortran')
        except:
            misc.compile(arg=['../lib/libdhelas.a'], cwd=source_dir, mode='fortran')
            misc.compile(arg=['../lib/libmodel.a'], cwd=source_dir, mode='fortran')

    #===========================================================================
    # Create proc_card_mg5.dat for Standalone directory
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize Standalone MG4 directory by 
           generation proc_card_mg5.dat
           generate a global makefile
           """
            
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

        if False:#'mode' in self.opt and self.opt['mode'] == "reweight":
            #add the module to hande the NLO weight
            files.copytree(pjoin(MG5DIR, 'Template', 'RWGTNLO'),
                          pjoin(self.dir_path, 'Source'))
            files.copytree(pjoin(MG5DIR, 'Template', 'NLO', 'Source', 'PDF'),
                           pjoin(self.dir_path, 'Source', 'PDF'))
            self.write_pdf_opendata()
            
        if self.prefix_info: 
            self.write_f2py_splitter()
            self.write_f2py_makefile(self.model)
            self.write_f2py_check_sa(matrix_elements,
                            pjoin(self.dir_path,'SubProcesses','check_sa.py'))
        else:
            # create a single makefile to compile all the subprocesses
            text = '''\n# For python linking (require f2py part of numpy)\nifeq ($(origin MENUM),undefined)\n  MENUM=2\nendif\n''' 
            deppython = ''
            for Pdir in os.listdir(pjoin(self.dir_path,'SubProcesses')):
                if os.path.isdir(pjoin(self.dir_path, 'SubProcesses', Pdir)):
                    text += '%(0)s/matrix$(MENUM)py.so:\n\tcd %(0)s;make matrix$(MENUM)py.so\n'% {'0': Pdir}
                    deppython += ' %(0)s/matrix$(MENUM)py.so ' % {'0': Pdir}
            text+='all: %s\n\techo \'done\'' % deppython
            
            ff = open(pjoin(self.dir_path, 'SubProcesses', 'makefile'),'a')
            ff.write(text)
            ff.close()
                    
    def write_f2py_splitter(self):
        """write a function to call the correct matrix element"""
        
        template = open(self.mgme_dir + '/madgraph/iolibs/template_files/python_all_matrix.f').read()
        
        allids = list(self.prefix_info.keys())
        allprefix = [self.prefix_info[key][0] for key in allids]
        min_nexternal = min([len(ids[0]) for ids in allids])
        max_nexternal = max([len(ids[0]) for ids in allids])

        info = []
        for (key, pid), (prefix, tag) in self.prefix_info.items():
            info.append('#PY %s : %s # %s %s' % (tag, key, prefix, pid))
            

        text = []
        for n_ext in range(min_nexternal, max_nexternal+1):
            current_id = [ids[0] for ids in allids if len(ids[0])==n_ext]
            current_pid = [ids[1] for ids in allids if len(ids[0])==n_ext]
            if not current_id:
                continue
            if min_nexternal != max_nexternal:
                if n_ext == min_nexternal:
                    text.append('       if (npdg.eq.%i)then' % n_ext)
                else:
                    text.append('       else if (npdg.eq.%i)then' % n_ext)
            for ii,pdgs in enumerate(current_id):
                pid = current_pid[ii]
                condition = '.and.'.join(['%i.eq.pdgs(%i)' %(pdg, i+1) for i, pdg in enumerate(pdgs)])
                if ii==0:
                    text.append( ' if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition, pid, ii))
                else:
                    text.append( ' else if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition,pid,ii))
                text.append(' call %ssmatrixhel(p, nhel, ans)' % self.prefix_info[(pdgs,pid)][0])
            text.append(' endif')
        #close the function
        if min_nexternal != max_nexternal:
            text.append('endif')

        params = self.get_model_parameter(self.model)
        parameter_setup =[]
        for key, var in params.items():
            parameter_setup.append('        CASE ("%s")\n          %s = value' 
                                   % (key, var))

        # part for the resetting of the helicity
        helreset_def = []
        helreset_setup = []
        for prefix in set(allprefix):
            helreset_setup.append(' %shelreset = .true. ' % prefix)
            helreset_def.append(' logical %shelreset \n common /%shelreset/ %shelreset' % (prefix, prefix, prefix))
        

        formatting = {'python_information':'\n'.join(info), 
                          'smatrixhel': '\n'.join(text),
                          'maxpart': max_nexternal,
                          'nb_me': len(allids),
                          'pdgs': ','.join(str(pdg[i]) if i<len(pdg) else '0' 
                                           for i in range(max_nexternal) for (pdg,pid) in allids),
                          'prefix':'\',\''.join(allprefix),
                          'pids': ','.join(str(pid) for (pdg,pid) in allids),
                          'parameter_setup': '\n'.join(parameter_setup),
                          'helreset_def' : '\n'.join(helreset_def),
                          'helreset_setup' : '\n'.join(helreset_setup),
                          }
        formatting['lenprefix'] = len(formatting['prefix'])
        text = template % formatting
        fsock = writers.FortranWriter(pjoin(self.dir_path, 'SubProcesses', 'all_matrix.f'),'w')
        fsock.writelines(text)
        fsock.close()
    
    def get_model_parameter(self, model):
        """ returns all the model parameter
        """
        params = {}
        for p in model.get('parameters')[('external',)]:
            name = p.name
            nopref = name[4:] if name.startswith('mdl_') else name
            params[nopref] = name
            
            block = p.lhablock
            lha = '_'.join([str(i) for i in p.lhacode])
            params['%s_%s' % (block.upper(), lha)] = name

        if model['running_elements']:
            add_scale = set()
            for runs in self.model.get('running_elements'):
                for line_run in runs.run_objects:
                    for one_element in line_run:
                        add_scale.add(one_element.lhablock)
            for block in add_scale:
                if block.upper() == "SMINPUTS":
                    continue
                name = block
                params['%s__scale' % (block.upper())] = 'mdl__%s__scale' % (block.upper())
                params['mdl__%s__scale' % (block.upper())] = 'mdl__%s__scale' % (block.upper())

        return params                      
                                        
        
        
    def write_f2py_check_sa(self, matrix_element, writer):
        """ Write the general check_sa.py in SubProcesses that calls all processes successively."""
        # To be implemented. It is just an example file, i.e. not crucial.
        return
    
    def write_f2py_makefile(self, model):
        """ """
        template = pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f2py')
        destination = pjoin(self.dir_path, 'SubProcesses', 'makefile')

        # Add file in SubProcesses
        if model['running_elements']:
            text = open(template,'r').read()
            text = text.replace('LINKLIBS_ME =  -L../lib/', 'LINKLIBS_ME =  -L../lib/ -lrunning ')
            text = text.replace('LINKLIBS_ALL =  -L../lib/', 'LINKLIBS_ALL =  -L../lib/ -lrunning ')
            open(destination, 'w').write(text)
        else:
             shutil.copy(template, destination)

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
                                         fortran_model, number, **opt):
        """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
        including the necessary matrix.f and nexternal.inc files"""

        cwd = os.getcwd()
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = pjoin(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

        if self.opt['sa_symmetry']:
            # avoid symmetric output
            for i,proc in enumerate(matrix_element.get('processes')):
                   
                tag = proc.get_tag()     
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
            
        proc_prefix = ''
        if 'prefix' in self.cmd_options:
            if self.cmd_options['prefix'] == 'int':
                proc_prefix = 'M%s_' % number
            elif self.cmd_options['prefix'] == 'proc':
                proc_prefix = matrix_element.get('processes')[0].shell_string().split('_',1)[1]
            else:
                raise Exception('--prefix options supports only \'int\' and \'proc\'')
            for proc in matrix_element.get('processes'):
                ids = [l.get('id') for l in proc.get('legs_with_decays')]
                self.prefix_info[(tuple(ids), proc.get('id'))] = [proc_prefix, proc.get_tag()] 
                
        calls = self.write_matrix_element_v4(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model,
            proc_prefix=proc_prefix)
        

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
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
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

        linkfiles = ['check_sa.f', 'coupl.inc']

        if proc_prefix and os.path.exists(pjoin(dirpath, '..', 'check_sa.f')):
            text = open(pjoin(dirpath, '..', 'check_sa.f')).read()
            pat = re.compile('smatrix', re.I)
            new_text, n  = re.subn(pat, '%ssmatrix' % proc_prefix, text)
            with open(pjoin(dirpath, 'check_sa.f'),'w') as f:
                f.write(new_text)
            linkfiles.pop(0)

        for file in linkfiles:
            ln('../%s' % file, cwd=dirpath)
        ln('../makefileP', name='makefile', cwd=dirpath)
        # Return to original PWD
        #os.chdir(cwd)

        if not calls:
            calls = 0
        return calls


    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = '$(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)'
        model_line='''$(LIBDIR)libmodel.$(libext): MODEL\n\t cd MODEL; make\n'''

        if model['running_elements']:
            running_line = '''$(LIBDIR)librunning.$(libext): RUNNING\n\t cd RUNNING; make\n'''
            set_of_lib += ' $(LIBDIR)librunning.$(libext) '
        else:
            running_line  = '' 

        replace_dict= {'libraries': set_of_lib, 
                       'model':model_line,
                       'additional_dsample': '',
                       'additional_dependencies':'',
                       'additional_clean':'',
                       'running': running_line} 

        text = open(path).read() % replace_dict
        
        if writer:
            writer.write(text)
        
        return replace_dict

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

            
        if 'sa_symmetry' not in self.opt:
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
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element)
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            replace_dict['nSqAmpSplitOrders']=1
            replace_dict['split_order_str_list']=''
            replace_dict['nb_temp_jamp'] = nb_tmp_jamp

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
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders)
            replace_dict['nb_temp_jamp'] = nb_tmp_jamp
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
            files.cp('nsqso_born.inc', '..')

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
            return len([call for call in helas_calls if call.find('#') != 0])
        else:
            replace_dict['return_value'] = len([call for call in helas_calls if call.find('#') != 0])
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
        for i_color in range(len(color_denominators)):
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
                    raise MadGraph5Error('Color Structure not handled by Matchbox: %s'  % ctype)
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

    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        try:
            misc.compile(cwd=pjoin(self.dir_path,'Source','MODEL'))
        except OSError:
            pass
        return super().finalize(matrix_elements, history, mg5options, flaglist)
    

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
                raise MadGraph5Error(error_msg % 'col_amps')
        else:
            raise MadGraph5Error(error_msg % 'col_amps')

        text, nb = super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(col_amps,
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
           
        text2, nb2 = super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(LC_col_amps,
                                            JAMP_format=JAMP_formatLC,
                                            AMP_format=AMP_format,
                                            split=-1)
        text += text2 
        
        return text, max(nb,nb2)




#===============================================================================
# ProcessExporterFortranMW
#===============================================================================
class ProcessExporterFortranMW(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 - MadWeight format."""

    matrix_file="matrix_standalone_v4.inc"
    jamp_optim = False

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranMW, self).copy_template(model)        

        # Add the MW specific file
        misc.copytree(pjoin(MG5DIR,'Template','MadWeight'),
                               pjoin(self.dir_path, 'Source','MadWeight'), True)        
        misc.copytree(pjoin(MG5DIR,'madgraph','madweight'),
                        pjoin(self.dir_path, 'bin','internal','madweight'), True) 
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','setrun.f'),
                                      pjoin(self.dir_path, 'Source','setrun.f'))
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','run.inc'),
                                      pjoin(self.dir_path, 'Source','run.inc'))
        # File created from Template (Different in some child class)
        filename = os.path.join(self.dir_path,'Source','run_config.inc')
        self.write_run_config_file(writers.FortranWriter(filename))

        self.handle_cuts_inc()

        try:
            subprocess.call([os.path.join(self.dir_path, 'Source','MadWeight','bin','internal','pass_to_madweight')],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = os.open(os.devnull, os.O_RDWR),
                            cwd=self.dir_path)
        except OSError:
            # Probably madweight already called
            pass
        
        ln(pjoin(self.dir_path, 'Source','PDF','eepdf.inc'),pjoin(self.dir_path, 'Source'))

        # Copy the different python file in the Template
        self.copy_python_file()
        # create the appropriate cuts.f
        self.get_mw_cuts_version()

        # add the makefile in Source directory 
        filename = os.path.join(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FortranWriter(filename), self.model)



    def handle_cuts_inc(self):

        text = open(pjoin(self.dir_path, 'Source', 'cuts.inc'),'r').read()
        text = text.replace('maxjetflavor','dummy_maxjetflavor')

        fsock = open(pjoin(self.dir_path, 'Source', 'cuts.inc'),'w')
        fsock.write(text)

        fsock.write('''            
                logical fixed_extra_scale
                integer maxjetflavor
                double precision mue_over_ref
                double precision mue_ref_fixed
                common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
                ''')
        fsock.close()

        
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
        misc.copytree(model_path, 
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
        cp(_file_path+'/iolibs/save_load_object.py', 
                              self.dir_path+'/bin/internal/save_load_object.py') 
        cp(_file_path+'/madevent/gen_crossxhtml.py', 
                              self.dir_path+'/bin/internal/gen_crossxhtml.py')
        cp(_file_path+'/madevent/sum_html.py', 
                              self.dir_path+'/bin/internal/sum_html.py')
        cp(_file_path+'/various/FO_analyse_card.py', 
                              self.dir_path+'/bin/internal/FO_analyse_card.py')                 
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
        logger.info("Running make for gammaUPC")
        misc.compile(arg=['../lib/libgammaUPC.a'], cwd=source_dir, mode='fortran')
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
    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize Standalone MG4 directory by generation proc_card_mg5.dat"""
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}



        #proc_charac
        if hasattr(self, "nlo_mixed_expansion"):
            self.proc_characteristics['nlo_mixed_expansion'] = mg5options['nlo_mixed_expansion']
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
                                         fortran_model,number, **opt):
        """Generate the Pxxxxx directory for a subprocess in MG4 MadWeight format,
        including the necessary matrix.f and nexternal.inc files"""

        cwd = os.getcwd()
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
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
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
        ln('../../Source/vector.inc', log=True, cwd=dirpath)
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
        jamp_lines, nb = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        
        replace_dict['template_file'] =  os.path.join(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = ''
        
        if writer:
            file = open(replace_dict['template_file']).read()
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
            return len([call for call in helas_calls if call.find('#') != 0]),ncolor
        else:
            replace_dict['return_value'] = (len([call for call in helas_calls if call.find('#') != 0]),ncolor)
            
    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model):
        """Write the nexternal.inc file for madweight"""


        path = os.path.join(_file_path,'iolibs','template_files','madweight_makefile_source')
        set_of_lib = '$(LIBRARIES) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libgammaUPC.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext) $(LIBDIR)libtf.$(libext)'
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
            raise writers.FortranWriter.FortranWriterError("""Need ninitial = 1 or 2 to write auto_dsig file""")

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
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines
        replace_dict['ee_comp_vars'] = eepdf_vars



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
                schannels = list(zip(*[s for s,t in stchannels]))
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
                elif verts in tchannels:
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
    done_warning_tchannel = False
    
    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{},
                        'hel_recycling': False
                        }
    jamp_optim = True
    default_vector_size = 1
    

    def __new__(cls, *args, **opts):
        """wrapper needed for some plugin"""

        return super(ProcessExporterFortranME, cls).__new__(cls)


    def __init__(self,  dir_path = "", opt=None):
        
        super(ProcessExporterFortranME, self).__init__(dir_path, opt)
        
        # check and format the hel_recycling options as it should if provided 
        if opt and isinstance(opt['output_options'], dict) and \
                                       'hel_recycling' in opt['output_options']:
            self.opt['hel_recycling'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['hel_recycling'], bool, 'hel_recycling')

        if opt and isinstance(opt['output_options'], dict) and \
                                       't_strategy' in opt['output_options']:
            self.opt['t_strategy'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['t_strategy'], int, 't_strategy')

        if opt and isinstance(opt['output_options'], dict) and \
                                       'vector_size' in opt['output_options']:
            misc.sprint(opt['output_options']['vector_size'])
            self.opt['vector_size'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['vector_size'], int, 'vector_size')
        else:
            self.opt['vector_size'] = 1

        if opt and isinstance(opt['output_options'], dict) and \
                                       'nb_warp' in opt['output_options']:
            self.opt['nb_warp'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['nb_warp'], int, 'nb_warp')
        else:
            self.opt['nb_warp'] = 1

    # helper function for customise helas writter
    @staticmethod
    def custom_helas_call(call, arg):
        if arg['mass'] == '%(M)s,%(W)s,':
            arg['mass'] = '%(M)s, fk_%(W)s,'
        elif '%(W)s' in arg['mass']:
            raise Exception

        arg['coup'] = re.sub('coup(\d+)\)s','coup\g<1>)s%(vec\g<1>)s', arg['coup'])

        return call, arg
    
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

        if model["running_elements"]:
            if not os.path.exists(pjoin(MG5DIR, 'Template',"Running")):
                raise Exception("Library for the running have not been installed. To install them please run \"install RunningCoupling\"")
            misc.copytree(pjoin(MG5DIR, 'Template',"Running"), 
                            pjoin(self.dir_path,'Source','RUNNING'))
        
        

    


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
        misc.copytree(model_path, 
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
                                         me_number, **opt):
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
        if 'hel_recycling' in self.opt and self.opt['hel_recycling']:
            filename = pjoin(Ppath, 'matrix_orig.f')
        else:
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
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
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

    link_Sub_files = ['addmothers.f',
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
                     'unwgt.f',
                     'dummy_fct.f'
                     ]

    def link_files_in_SubProcess(self, Ppath):
        """ Create the necessary links in the P* directory path Ppath"""
        
        #import genps.inc and maxconfigs.inc into Subprocesses
        ln(self.dir_path + '/Source/genps.inc', 
                                     self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxconfigs.inc',
                                     self.dir_path + '/SubProcesses', log=False)

        linkfiles = self.link_Sub_files

        for file in linkfiles:
            ln('../' + file , cwd=Ppath)    


    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize ME v4 directory by creating jpeg diagrams, html
        pages,proc_card_mg5.dat and madevent.tar.gz."""
        

        if second_exporter:
            self.has_second_exporter = second_exporter

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
        self.proc_characteristic['nlo_mixed_expansion'] = mg5options['nlo_mixed_expansion']
        
        self.proc_characteristic['complex_mass_scheme'] = mg5options['complex_mass_scheme']
        self.proc_characteristic['gauge'] = mg5options['gauge']

        # set limitation linked to the model
    
        
        # indicate the PDG of all initial particle
        try:
            pdgs1 = [p.get_initial_pdg(1) for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(1)]
            pdgs2 = [p.get_initial_pdg(2) for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(2)]
        except AttributeError:
            pdgs1 = [p.get_initial_pdg(1) for m in matrix_elements.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(1)]
            pdgs2 = [p.get_initial_pdg(2) for m in matrix_elements.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(2)]
        self.proc_characteristic['pdg_initial1'] = pdgs1
        self.proc_characteristic['pdg_initial2'] = pdgs2
        
        
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
            except Exception as error:
                pass
            
            if misc.which('gs'):
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


        if second_exporter:
            second_exporter.finalize(matrix_elements, history, mg5options, flaglist)



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

        # check if MLM/.../ is supported for this matrix-element and update associate flag
        if self.model and 'MLM' in self.model["limitations"]:
            if 'MLM' not in self.proc_characteristic["limitations"]:
                used_couplings = matrix_element.get_used_couplings(output="set") 
                for vertex in self.model.get('interactions'):
                    particles = [p for p in vertex.get('particles')]
                    if 21 in [p.get('pdg_code') for p in particles]:
                        colors = [par.get('color') for par in particles]
                        if 1 in colors:
                            continue
                        elif 'QCD' not in vertex.get('orders'):
                            for bad_coup in vertex.get('couplings').values():
                                if bad_coup in used_couplings:
                                    self.proc_characteristic["limitations"].append('MLM')
                                    break

        # The proc prefix is not used for MadEvent output so it can safely be set
        # to an empty string.
        replace_dict = {'proc_prefix':'',
                        'set_amp2_line': 'ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT'}
 
 
        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        if fortran_model.width_tchannel_set_tozero and not ProcessExporterFortranME.done_warning_tchannel:
            logger.info("Some T-channel width have been set to zero [new since 2.8.0]\n if you want to keep this width please set \"zerowidth_tchannel\" to False", '$MG:BOLD')
            ProcessExporterFortranME.done_warning_tchannel = True

        replace_dict['helas_calls'] = "\n".join(helas_calls)


        #adding the support for the fake width (forbidding too small width)
        mass_width = matrix_element.get_all_mass_widths()
        mass_width = sorted(list(mass_width))
        width_list = set([e[1] for e in mass_width])
        
        replace_dict['fake_width_declaration'] = \
            ('  double precision fk_%s \n' * len(width_list)) % tuple(width_list)
        replace_dict['fake_width_declaration'] += \
            ('  save fk_%s \n' * len(width_list)) % tuple(width_list)
        fk_w_defs = []
        one_def = ' IF(%(w)s.ne.0d0) fk_%(w)s = SIGN(MAX(ABS(%(w)s), ABS(%(m)s*small_width_treatment)), %(w)s)'     
        for m, w in mass_width:
            if w == 'zero':
                if ' fk_zero = 0d0' not in fk_w_defs: 
                    fk_w_defs.append(' fk_zero = 0d0')
                continue    
            fk_w_defs.append(one_def %{'m':m, 'w':w})
        replace_dict['fake_width_definitions'] = '\n'.join(fk_w_defs)

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

        # if proc_id:
        #     # Set lines for subprocess group version
        #     # Set define_iconfigs_lines
        #     replace_dict['define_iconfigs_lines'] += \
        #          """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
        #          COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
        #     # Set set_amp2_line
        #     replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(SUBDIAG(%s))/XTOT" % \
        #                                     proc_id
        # else:
        #     # Standard running
        #     # Set set_amp2_line
        #     replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT"

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
            if hasattr(self.model, '_curr_gauge') and self.model._curr_gauge == 'FD':
                replace_dict['wavefunctionsize'] = 7

        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element, config_map, replace_dict)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)

        # The JAMP definition depends on the splitting order
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)>0:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['chosen_so_configs']=self.set_chosen_SO_index(
                              matrix_element.get('processes')[0],squared_orders)
            replace_dict['select_configs_if'] = '          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN' % replace_dict
            replace_dict['select_configs_endif'] = ' endif'
        else:
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            # addtionally set the function to NOT be called
            replace_dict['select_configs_if'] = ''
            replace_dict['select_configs_endif'] = ''
            
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
        jamp_lines, nb_temp = self.get_JAMP_lines_split_order(\
                             matrix_element,amp_orders,split_order_names=
                        split_orders if len(split_orders)>0 else ['ALL_ORDERS'])
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_temp

        if self.beam_polarization == [True, True]:
            replace_dict['beam_polarization'] = """
                         DO JJ=1,nincoming
               IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                 T=T*ABS(POL(JJ))
               ELSE IF(POL(JJ).NE.1d0)THEN
                 T=T*(2d0-ABS(POL(JJ)))
               ENDIF
             ENDDO
            """
        else:
            replace_dict['beam_polarization'] = ""
            for i in [0,1]:
                if self.beam_polarization[i]:
                    replace_dict['beam_polarization'] = """
                                   ! handling only one beam polarization here. Second beam can be handle via the pdf.
                                   IF(POL(%(bid)i).NE.1d0.AND.NHEL(%(bid)i,I).EQ.INT(SIGN(1d0,POL(%(bid)i)))) THEN
                 T=T*ABS(POL(%(bid)i))
               ELSE IF(POL(%(bid)i).NE.1d0)THEN
                 T=T*(2d0-ABS(POL(%(bid)i)))
               ENDIF """ % {'bid': i+1}




        replace_dict['template_file'] = pjoin(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = pjoin(_file_path, \
                          'iolibs/template_files/split_orders_helping_functions.inc')      
        
        s1,s2 = matrix_element.get_spin_state_initial()
        replace_dict['nb_spin_state1'] = s1
        replace_dict['nb_spin_state2'] = s2


        if writer:
            file = open(replace_dict['template_file']).read()
            file = file % replace_dict
            # Add the split orders helper functions.
            file = file + '\n' + open(replace_dict['template_file2'])\
                                                            .read()%replace_dict
            # Write the file
            writer.writelines(file)
            return len([call for call in helas_calls if call.find('#') != 0]), ncolor
        else:
            replace_dict['return_value'] = (len([call for call in helas_calls if call.find('#') != 0]), ncolor)
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
            raise writers.FortranWriter.FortranWriterError("""Need ninitial = 1 or 2 to write auto_dsig file""")

        replace_dict = {}
        replace_dict['additional_header'] = ''
        replace_dict['OMP_LIB'] = " USE OMP_LIB"
        replace_dict['OMP_PREFIX'] = "!$OMP PARALLEL\n!$OMP DO"
        replace_dict['OMP_POSTFIX'] = "!$OMP END DO\n!$OMP END PARALLEL"


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
            conv_factor=""
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(0)*conv*dsiguu"
            conv_factor="conv*"

        replace_dict['dsig_line'] = dsig_line
        replace_dict['conv'] = conv_factor
        # Extract pdf lines
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines
        replace_dict['ee_comp_vars'] = eepdf_vars

        # Extract pdf lines vectorised code
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                self.get_pdf_lines(matrix_element, ninitial, proc_id != "", 
                                   vector=max(1,int(self.opt['vector_size'])))
        replace_dict['pdf_vars_vec'] = pdf_vars
        replace_dict['pdf_data_vec'] = pdf_data
        replace_dict['ee_comp_vars_vec'] = eepdf_vars
        replace_dict['pdf_lines_vec'] = pdf_lines


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
            replace_dict['get_channel'] = "SUBDIAG(%s)" % proc_id
            replace_dict['get_channel_vec'] = """
            CHANNELS(IVEC) = CONFSUB(%s,SYMCONF(ICONF_VEC(CURR_WARP)))
            SUBDIAG(%s) = CHANNELS(IVEC) ! only valid if a single process
            channel = SUBDIAG(%s)""" % (proc_id,proc_id, proc_id)
            #SUBDIAG(%s)" % proc_id
            replace_dict['ADDITIONAL_FCT'] = ''
        else:
            replace_dict['passcuts_begin'] = ""#IF (PASSCUTS(PP)) THEN"
            replace_dict['passcuts_end'] = ""#ENDIF"
            replace_dict['define_subdiag_lines'] = "INTEGER IB(2)"
            replace_dict['cutsdone'] = "      cutsdone=.false.\n       cutspassed=.false."
            replace_dict['get_channel'] = "MAPCONFIG(ICONFIG)"
            replace_dict['get_channel_vec'] = " channel  = MAPCONFIG(ICONFIG)"
            # need to extract get_helicities/select color from the group template file
            text = open(pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files', 'super_auto_dsig_group_v4.inc')).read()
            color_hel_text = writers.FortranWriter.get_routine(text, ['select_color', 'get_helicities'])
            #misc.sprint(color_hel_text)
            get_nhel, get_helicity = [],[]
            get_nhel.append("   integer get_nhel" )
            get_helicity.append("   do i=1,nexternal")
            get_helicity.append(
                    "        nhel(i) = get_nhel(ihel,i)")
            get_helicity.append("enddo")
            replace_dict['call_to_local_get_helicities'] = "\n".join(get_helicity)
            replace_dict['definition_of_local_get_nhel'] = "\n".join(get_nhel)


            #raise Exception
            replace_dict['ADDITIONAL_FCT'] = self.get_dummy_grouping()+ '\n'.join(color_hel_text) % replace_dict
        # extract and replace ncombinations, helicity lines
        ncomb=matrix_element.get_helicity_combinations()
        replace_dict['ncomb']= ncomb
        helicity_lines = self.get_helicity_lines(matrix_element, add_nb_comb=True)
        replace_dict['helicity_lines'] = helicity_lines

        context = {'read_write_good_hel':True}        
        if not isinstance(self, ProcessExporterFortranMEGroup):            
            replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)
            context['nogrouping'] = True
        else:
            replace_dict['read_write_good_hel'] = ""
            context['nogrouping'] = False
        
        if writer:
            file = open(pjoin(_file_path, \
                          'iolibs/template_files/auto_dsig_v4.inc')).read()
            file = file % replace_dict

            # Write the file
            writer.writelines(file, context=context)
        else:
            return replace_dict, context

            
    #===========================================================================
    # get_dummy_grouping
    #===========================================================================
    def get_dummy_grouping(self):
        """ return dummy function for 
        prepare_grouping
        select_grouping
        for situation where they are no grouping
        """

        return """
        
        subroutine PREPARE_GROUPING_CHOICE(PP, WGT, INIT)
        double precision PP(*)
        double precision WGT
        logical INIT
        return
        end

        SUBROUTINE SELECT_GROUPING(IMIRROR, IPROC, ICONF, WGT, IWARP)
        Integer imirror
        integer iproc
        integer iconf
        double precision WGT
        integer iwarp
        return 
        end
        
        
        """


    #===========================================================================
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, mapconfigs, matrix_element):
        """Write the coloramps.inc file for MadEvent"""

        lines = self.get_icolamp_lines(mapconfigs, matrix_element, 1)
        lines.insert(0, "logical icolamp(%d,%d,1)" % \
                        (max(len(list(matrix_element.get('color_basis').keys())), 1),
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
                 
            # pass to ping-pong strategy for t-channel for 3 ore more T-channel
            #  this is directly related to change in genps.f
            tstrat = self.opt.get('t_strategy', 0)
            if isinstance(self, madgraph.loop.loop_exporters.LoopInducedExporterMEGroup):
                tstrat = 2
            tchannels, tchannels_strategy = ProcessExporterFortranME.reorder_tchannels(tchannels, tstrat, self.model)
            
            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels, tchannels_strategy])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = list(zip(*[s for s,t in stchannels]))
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
            lines.append("data tstrategy(%d)/%d/" % (nconfigs, tchannels_strategy))
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
                elif verts in tchannels:
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
    # reoder t-channels
    #===========================================================================
    
    #ordering = 0    
    @staticmethod
    def reorder_tchannels(tchannels, tstrat, model):
        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        #ProcessExporterFortranME.ordering +=1

        if len(tchannels) < 3 or tstrat == 2 or not model:
            return tchannels, 2
        elif tstrat == 1:
            return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
        elif tstrat == -2:
            return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
        elif tstrat == -1:
            return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels, 1), -1        
        elif len(tchannels) < 4:
            #
            first = tchannels[0]['legs'][1]['number']
            t1 =  tchannels[0]['legs'][-1]['id']
            last = tchannels[-1]['legs'][1]['number']
            t2 = tchannels[-1]['legs'][0]['id']
            m1  = model.get_particle(t1).get('mass') == 'ZERO'
            m2  = model.get_particle(t2).get('mass') == 'ZERO'
            if m2 and not m1:
                return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
            elif m1 and not m2:
                return tchannels, 2
            elif first < last:
                return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
            else:
                return tchannels, 2 
        else:
            first = tchannels[0]['legs'][1]['number']
            t1 =  tchannels[0]['legs'][-1]['id']
            last = tchannels[-1]['legs'][1]['number']
            t2 = tchannels[-1]['legs'][0]['id']
            m1  = model.get_particle(t1).get('mass') == 'ZERO'
            m2  = model.get_particle(t2).get('mass') == 'ZERO'
            
            t12 =  tchannels[1]['legs'][-1]['id']
            m12 = model.get_particle(t12).get('mass') == 'ZERO'
            t22 = tchannels[-2]['legs'][0]['id']
            m22 = model.get_particle(t22).get('mass') == 'ZERO'
            if m2 and not m1:
                if m22:
                    return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
                else:
                    return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
            elif m1 and not m2:
                if m12:
                    return tchannels, 2
                else:
                    return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
            elif m1 and m2 and  len(tchannels) == 4 and not m12: # 3 T propa
                return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
                # this case seems quite sensitive we tested method 2 specifically and this was not helping in general 
            elif not m1 and not m2 and  len(tchannels) == 4 and m12:
                if first < last:
                    return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
                return tchannels, 2
            else:
                return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2


                

    @staticmethod
    def reorder_tchannels_flipside(tchannels):
        """change the tchannel ordering to pass to a ping-pong strategy.
           assume ninitial == 2
        
        We assume that we receive something like this
        
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-1)
                |
                X --------- -1

                X----------  3
                | 
                | (-N+2)
                |                
                X --------- L
                |
                | (-N+1) 
                |                
        -N ----- X ------- P        
        
        coded as 
        (1 -2 > -X) (-X 4 > -X-1) (-X-1 -1 > -X-2) ...
        ((-N+3) 3 > (-N+2)) ((-n+2) L > (-n+1)) ((-n+1) P > -N)
        
        we want to convert this as:
        -N ----- X ------- -2
                |
                | (-N+1) 
                |
                X -------- 4
                | 
                | (-N+2)
                |
                X --------- -1

                X----------  3
                | 
                | (-X-1)
                |                
                X --------- L
                |
                | (-X) 
                |                
        2 ----- X ------- P          
        
        coded as 
        ( 2 P > -X) (-X L > -X-1) (-X-1 3 > -X-2)... (-X-L -2 > -N)
        """
        
        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        if len(tchannels) < 2:
            return tchannels

        out = []
        oldid2new = {}
        
        # initialisation
        # id of the first T-channel (-X)
        propa_id = tchannels[0]['legs'][-1]['number'] 
        #
        # Setup the last vertex to refenence the second id beam
        # -N (need to setup it to 2.
        initialid = tchannels[-1]['legs'][-1]['number']       
        oldid2new[initialid] = 2
        oldid2new[1] = initialid
            
        i = 0 
        while tchannels:
            old_vert = tchannels.pop()
                
            #copy the vertex /leglist to avoid side effects
            new_vert = copy.copy(old_vert)
            new_vert['legs'] = base_objects.LegList([base_objects.Leg(l) for l in old_vert['legs']])
            # vertex taken from the bottom we have 
            # (-N+1 X > -N) we need to flip to pass to 
            # -N X > -N+1 (and then relabel -N and -N+1  
            legs = new_vert['legs'] # shorcut
            id1 = legs[0]['number']
            id2 = legs[1]['number']
            id3 = legs[2]['number']
            # to be secure  we also support (X -N+1 > -N)
            if id3 == id2 -1 and id1 !=1:
                legs[0], legs[1] = legs[1], legs[0]
            #flipping side
            legs[0], legs[2] = legs[2], legs[0]

            # the only new relabelling is the last element of the list
            # always thanks to the above flipping
            old_propa_id = new_vert['legs'][-1]['number'] 
            oldid2new[old_propa_id] = propa_id

            
            #pass to new convention for leg numbering:
            for l in new_vert['legs']:
                if l['number'] in  oldid2new:
                    l['number'] = oldid2new[l['number']]  
                    
            # new_vert is now ready
            out.append(new_vert)
            # prepare next iteration
            propa_id -=1
            i +=1

        return out
    
    @staticmethod
    def reorder_tchannels_pingpong(tchannels, id=2):
        """change the tchannel ordering to pass to a ping-pong strategy.
           assume ninitial == 2
        
        We assume that we receive something like this
        
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-1)
                |
                X --------- -1

                X----------  3
                | 
                | (-N+2)
                |                
                X --------- L
                |
                | (-N+1) 
                |                
        -N ----- X ------- P        
        
        coded as 
        (1 -2 > -X) (-X 4 > -X-1) (-X-1 -1 > -X-2) ...
        ((-N+3) 3 > (-N+2)) ((-n+2) L > (-n+1)) ((-n+1) P > -N)
        
        we want to convert this as:
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-2)
                |
                X --------- -1

                X----------  3
                | 
                | (-X-3)
                |                
                X --------- L
                |
                | (-X-1) 
                |                
        2 ----- X ------- P          
        
        coded as 
        (1 -2 > -X) (2 P > -X-1) (-X 4 > -X-2) (-X-1 L > -X-3) ...
        """

        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        if len(tchannels) < 2:
            return tchannels

        out = []
        oldid2new = {}
        
        # initialisation
        # id of the first T-channel (-X)
        propa_id = tchannels[0]['legs'][-1]['number'] 
        #
        # Setup the last vertex to refenence the second id beam
        # -N (need to setup it to 2.
        initialid = tchannels[-1]['legs'][-1]['number']       
        oldid2new[initialid] = id


        
        i = 0 
        while tchannels:
            #ping pong by taking first/last element in aternance
            if id ==2:
                if i % 2 == 0:
                    old_vert = tchannels.pop(0)
                else:
                    old_vert = tchannels.pop()
            else:
                if i % 2 != 0:
                    old_vert = tchannels.pop(0)
                else:
                    old_vert = tchannels.pop()
                    
            #copy the vertex /leglist to avoid side effects
            new_vert = base_objects.Vertex(old_vert)
            new_vert['legs'] = base_objects.LegList([base_objects.Leg(l) for l in old_vert['legs']])
            # if vertex taken from the bottom we have 
            # (-N+1 X > -N) we need to flip to pass to 
            # -N X > -N+1 (and then relabel -N and -N+1
            # to be secure  we also support (X -N+1 > -N)
            if (i % 2 ==1 and id ==2) or (i %2 == 0 and id ==1): 
                legs = new_vert['legs'] # shorcut
                id1 = legs[0]['number']
                id2 = legs[1]['number'] 
                if id1 > id2:
                    legs[0], legs[1] = legs[1], legs[0]
                else:
                    legs[0], legs[2] = legs[2], legs[0]
            
            # the only new relabelling is the last element of the list
            # always thanks to the above flipping
            old_propa_id = new_vert['legs'][-1]['number'] 
            oldid2new[old_propa_id] = propa_id

            if i==0 and id==1:
                legs[0]['number'] = 2
            
            #pass to new convention for leg numbering:
            for l in new_vert['legs']:
                if l['number'] in  oldid2new:
                    l['number'] = oldid2new[l['number']]    
            
            # new_vert is now ready
            out.append(new_vert)
            # prepare next iteration
            propa_id -=1
            i +=1

        return out

            
        
        
    
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

        replace_dict['DRIVER_EXTRA_HEADER'] = ""
        replace_dict['DRIVER_EXTRA_INITIALISE'] = ""
        replace_dict['DRIVER_EXTRA_FINALISE'] = ""

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
        final_legs = [leg for leg in matrix_element.get('processes')[0].get_legs_with_decays() if leg.get('state') == True]

        ninitial = len([leg for leg in matrix_element.get('processes')[0].get('legs') if leg.get('state') == False])

        identical_indices = {}

        # Extract identical particle info
        for i, leg in enumerate(final_legs):
            if leg.get('id') in identical_indices:
                identical_indices[leg.get('id')].append(\
                                    i + ninitial + 1)
            else:
                identical_indices[leg.get('id')] = [i + ninitial + 1]

        # Remove keys which have only one particle
        for key in list(identical_indices.keys()):
            if len(identical_indices[key]) < 2:
                del identical_indices[key]

        # Write mg.sym file
        lines.append(str(len(list(identical_indices.keys()))))
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
    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{},
                        'hel_recycling': True
                        }
    
    
    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model,
                                         group_number,
                                         second_exporter=None,
                                         second_helas=None):
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
                raise MadGraph5Error("All grouped processes must share the "+\
                                       "same number of helicity configurations.")                

        filename = 'driver.f'
        self.write_driver(writers.FortranWriter(filename),ncomb,
                                  n_grouped_proc=len(matrix_elements), v5=self.opt['v5_model'])

        try:
            self.proc_characteristic['hel_recycling'] = self.opt['hel_recycling']
        except KeyError:
            self.proc_characteristic['hel_recycling'] = False
            self.opt['hel_recycling'] = False
        for ime, matrix_element in \
                enumerate(matrix_elements):
            if self.opt['hel_recycling']:
                filename = 'matrix%d_orig.f' % (ime+1)
                replace_dict = self.write_matrix_element_v4(None, 
                                matrix_element,
                                fortran_model,
                                proc_id=str(ime+1),
                                config_map=subproc_group.get('diagram_maps')[ime],
                                subproc_number=group_number)
                calls,ncolor = replace_dict['return_value']
                tfile = open(replace_dict['template_file']).read()
                file = tfile % replace_dict
                # Add the split orders helper functions.
                file = file + '\n' + open(replace_dict['template_file2'])\
                                                            .read()%replace_dict
                # Write the file
                writer = writers.FortranWriter(filename)
                writer.writelines(file)
                
                #
                # write the dedicated template for helicity recycling
                #
                tfile = open(replace_dict['template_file'].replace('.inc',"_hel.inc")).read() 
                file = tfile % replace_dict
                # Add the split orders helper functions.
                file = file + '\n' + open(replace_dict['template_file2'])\
                                                            .read()%replace_dict
                # Write the file
                writer = writers.FortranWriter('template_matrix%d.f' % (ime+1))
                writer.uniformcase = False
                writer.writelines(file)
                
                
                
                
            else:
                filename = 'matrix%d.f' % (ime+1)
                calls, ncolor = \
                   self.write_matrix_element_v4(writers.FortranWriter(filename), 
                                matrix_element,
                                fortran_model,
                                proc_id=str(ime+1),
                                config_map=subproc_group.get('diagram_maps')[ime],
                                subproc_number=group_number)

            if second_exporter:
                process_exporter_cpp = second_exporter.oneprocessclass(matrix_element,second_helas, prefix=ime)
                dirpath = '.'
                with misc.chdir(dirpath):
                    logger.info('Creating files in directory %s' % dirpath)
                    process_exporter_cpp.path = dirpath
                    # Create the process .h and .cc files
                    process_exporter_cpp.generate_process_files_madevent(proc_id=str(ime+1),
                                        config_map=subproc_group.get('diagram_maps')[ime], 
                                        subproc_number=group_number)
                    for file in second_exporter.to_link_in_P:
                        ln('../%s' % file)    
                # second_exporter.write_matrix_element_madevent(ime,
                #                 matrix_element,
                #                 second_helas,
                #                 proc_id=str(ime+1),
                #                 config_map=subproc_group.get('diagram_maps')[ime],
                #                 subproc_number=group_number
                # )


            filename = 'auto_dsig%d.f' % (ime+1)
            self.write_auto_dsig_file(writers.FortranWriter(filename),
                                 matrix_element,
                                 str(ime+1))

            # Keep track of needed quantities
            tot_calls += int(calls)
            maxflows = max(maxflows, ncolor)
            maxamps = max(maxamps, len(matrix_element.get('diagrams')))

            # Draw diagrams
            if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
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
        
        # check consistency
        for i, sym_fact in enumerate(symmetry):
            
            if sym_fact >= 0:
                continue
            if nqcd_list[i] != nqcd_list[abs(sym_fact)-1]:
                misc.sprint(i, sym_fact, nqcd_list[i], nqcd_list[abs(sym_fact)])
                raise Exception("identical diagram with different QCD powwer")
        

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

        if second_exporter:
            tmp = locals()
            del tmp['self']
            process_exporter_cpp.generate_subprocess_directory_end(**tmp) 

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
        call_dsig_proc_lines_vec = []
        for iproc in range(len(matrix_elements)):
            data = {"num": iproc + 1,
                 "proc": matrix_elements[iproc].get('processes')[0].base_string()}
            call_dsig_proc_lines.append(\
                "IF(IPROC.EQ.%(num)d) DSIGPROC=DSIG%(num)d(P1,WGT,IMODE) ! %(proc)s" % data
                )
            call_dsig_proc_lines_vec.append(\
                "IF(IPROC.EQ.%(num)d) CALL DSIG%(num)d_VEC(ALL_P1,ALL_XBK,ALL_Q2FACT,ALL_CM_RAP,ALL_WGT,IMODE,ALL_OUT,SYMCONF, CONFSUB,ICONF_VEC,IMIRROR_VEC,VECSIZE_USED) ! %(proc)s" % data
                )

        replace_dict['call_dsig_proc_lines'] = "\n".join(call_dsig_proc_lines)
        replace_dict['call_dsig_proc_lines_vec'] = "\n".join(call_dsig_proc_lines_vec)

        ncomb=matrix_elements[0].get_helicity_combinations()
        replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)

        s1,s2 = matrix_elements[0].get_spin_state_initial()
        replace_dict['nb_spin_state1'] = s1
        replace_dict['nb_spin_state2'] = s2
        
        printzeroamp = []
        for iproc in range(len(matrix_elements)):
            printzeroamp.append(\
                "        call print_zero_amp%i()" % ( iproc + 1))
        replace_dict['print_zero_amp'] = "\n".join(printzeroamp)
        
        
        get_nhel = []
        for iproc in range(len(matrix_elements)):
            get_nhel.append("   integer get_nhel%i   " %(iproc+1) )
            if iproc == 0:
                get_helicity = [' if(iproc.eq.1)then']
            else: 
                get_helicity.append(' elseif(iproc.eq.%s)then' % (iproc+1))
            get_helicity.append("   do i=1,nexternal")
            get_helicity.append(
                "        nhel(i) = get_nhel%i(ihel,i)" % ( iproc + 1))
            get_helicity.append("enddo")
        get_helicity.append(" endif" ) 

        replace_dict['call_to_local_get_helicities'] = "\n".join(get_helicity)
        replace_dict['definition_of_local_get_nhel'] = "\n".join(get_nhel)

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
        subproc_to_confdiag = self.get_confdiag_from_group_mapconfig(diagrams_for_config)
        lines = []

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
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXSPROC)
        INTEGER NTRY(MAXSPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        write(stream_id,*) GOODHEL
        return
        end
        
        
        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXSPROC)
        INTEGER NTRY(MAXSPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        read(stream_id,*) GOODHEL
        NTRY(:) = MAXTRIES + 1
        return
        end 
        
        subroutine init_good_hel()
        implicit none
        include 'maxamps.inc'
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXSPROC)        
        INTEGER NTRY(MAXSPROC)
        INTEGER I,J

        GOODHEL(:,:) = .false.        
        NTRY(:) = 0
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
    @staticmethod
    def get_confdiag_from_group_mapconfig(config_subproc_map, subprocid=None):
            """ This is converting the  mapconfigs generated from the 
                    subproc_group.get('diagrams_for_configs')
                and convert it to a datastructure like expected from the 
                get_icolamp_lines (which does not handle grouping) 
                
                if subproc is None it returns the full output as a dictionary
                with subproc_id as key.
                if provided it returned the associated list for that subproc id.

                Static method since need to be used from cpp case as well.
            """

            subproc_to_confdiag = {}
            for config in config_subproc_map:
                for subproc, diag in enumerate(config):
                    try:
                        subproc_to_confdiag[subproc].append(diag)
                    except KeyError:
                        subproc_to_confdiag[subproc] = [diag]
                        
            if subprocid is None:
                return subproc_to_confdiag
            else:
                return subproc_to_confdiag[subprocid]

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


    def finalize(self,*args, second_exporter=None, **opts):

        if second_exporter:
            self.has_second_exporter = second_exporter
        super(ProcessExporterFortranMEGroup, self).finalize(*args, second_exporter=None, **opts)
        #ensure that the grouping information is on the correct value
        self.proc_characteristic['grouped_matrix'] = True

        filename = pjoin(self.dir_path,'Source','makefile')
        if not second_exporter:
            self.write_source_makefile(writers.FileWriter(filename), model=self.model)
        else:
           replace_dict = self.write_source_makefile(None)
           second_exporter.write_source_makefile(writers.FileWriter(filename), model=self.model, default=replace_dict)  

        if second_exporter:
            second_exporter.finalize(*args, **opts)

        
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
        self.coups_indep_noloop = []  # (name, expression, type)
        self.coups_indep_loop = []  # (name, expression, type)
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)
        self.params_ext = []   # external parameter
        self.p_to_f = parsers.UFOExpressionParserFortran(self.model)
        self.mp_p_to_f = parsers.UFOExpressionParserMPFortran(self.model)   
        try:
            vector_size = self.opt['output_options']['vector_size']
            self.vector_size = banner_mod.ConfigFile.format_variable(vector_size, int, 'vector_size')
        except KeyError as error:
            misc.sprint(error)
            self.vector_size = 0

        try:
            nb_warp = self.opt['output_options']['nb_warp']
        except KeyError:
            nb_warp = 1
        self.nb_warp = max(1, banner_mod.ConfigFile.format_variable(nb_warp, int, 'nb_warp'))
        if self.opt['mp']:
            assert self.vector_size in [0,1]
            self.nb_warp = 1
        self.scales = []
        self.MUE = None # extra parameter loop #2 which is running
        
        if self.model.get('running_elements'):
            all_elements = set()
            add_scale = set()
            for runs in self.model.get('running_elements'):
                for line_run in runs.run_objects:
                    for one_element in line_run:
                        all_elements.add(one_element.name)
                        add_scale.add(one_element.lhablock)
            all_elements.union(set(self.PS_dependent_key))
            self.PS_dependent_key = list(all_elements)
            MUE = [p for p in self.model.get('parameters')[('external',)] if p.lhablock.lower() == 'loop' and tuple(p.lhacode) == (2,)]
            
            if MUE:
                self.MUE = MUE[0]
                self.PS_dependent_key.append(MUE[0].name)
            
            try:
                add_scale.remove('SMINPUTS')
            except Exception:
                pass
            self.scales = add_scale

    
    def pass_parameter_to_case_insensitive(self):
        """modify the parameter if some of them are identical up to the case"""
    
        lower_dict={}
        duplicate = set()
        keys = list(self.model['parameters'].keys())
        keys.sort()
        for key in keys:
            for param in self.model['parameters'][key]:
                lower_name = param.name.lower()
                if not lower_name:
                    continue
                try:
                    lower_dict[lower_name].append(param)
                except KeyError as error:
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
            
        # Keep only separation in alphaS + running one        
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)

        for key in keys:
            to_add = [o for o in self.model['parameters'][key] if o.name]

            if key == ('external',):
                self.params_ext += to_add
            elif any([(k in key) for k in self.PS_dependent_key]):
                self.params_dep += to_add
            else:
                self.params_indep += to_add
                
        # same for couplings + tracking which running happens
        keys = list(self.model['couplings'].keys())
        keys.sort(key=len)
        used_running_key = set()
        for key, coup_list in self.model['couplings'].items():
            if any([(k in key) for k in self.PS_dependent_key]):
                to_add = [c for c in coup_list if
                                   (not wanted_couplings or c.name in \
                                    wanted_couplings)]
                if to_add:
                    self.coups_dep += to_add
                    used_running_key.update(set(key))
            else:
                self.coups_indep_noloop += [c for c in coup_list if
                                     (not wanted_couplings or c.name in \
                                      wanted_couplings) and \
                                      not any([tag in c.name.lower() for tag in ['uv', 'r2']])]
                self.coups_indep_loop += [c for c in coup_list if
                                     (not wanted_couplings or c.name in \
                                      wanted_couplings) and \
                                      any([tag in c.name.lower() for tag in ['uv', 'r2']])]

        # keep track of all couplings (for backward compatibility and/or tests
        self.coups_indep = self.coups_indep_noloop + self.coups_indep_loop
               
        #store the running parameter that are used
        self.used_running_key = used_running_key     
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        if 'G' in self.params_dep:
            index = self.params_dep.index('G')
            G = self.params_dep.pop(index)
        #    G.expr = '2*cmath.sqrt(as*pi)'
        #    self.params_indep.insert(0, self.params_dep.pop(index))
        # No need to add it if not defined   

        if 'aS' not in self.params_ext and 'aS' not in self.params_indep:
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
            write_class = io.FileIO
            
            write_class.writelines(fsock, comment * 77 + '\n')
            write_class.writelines(fsock, '%(comment)s written by the UFO converter\n' % \
                               {'comment': comment + (6 - len(comment)) *  ' '})
            write_class.writelines(fsock, comment * 77 + '\n\n')
        else:
            fsock = open(file_path, 'w')  
            fsock.writelines(comment * 77 + '\n')
            fsock.writelines('%(comment)s written by the UFO converter\n' % \
                                   {'comment': comment + (6 - len(comment)) *  ' '})
            fsock.writelines(comment * 77 + '\n\n')
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
        self.create_ewa()
        
        # definition of the coupling.
        self.create_actualize_mp_ext_param_inc()
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        self.create_printout()
        
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
        file_to_link = ['formats.inc', \
                        'rw_para.f', 'testprog.f']
    
        for filename in file_to_link:
            cp( MG5DIR + '/models/template_files/fortran/' + filename, \
                                                                self.dir_path)
            
        file = open(os.path.join(MG5DIR,\
                              'models/template_files/fortran/rw_para.f')).read()

        if self.vector_size:
            includes=["include \'../vector.inc\'"]
        else:
            includes = []
        
        includes +=["include \'coupl.inc\'",
                  "include \'input.inc\'",
                  "include \'model_functions.inc\'"]
        if self.opt['mp']:
            includes.extend(["include \'mp_coupl.inc\'","include \'mp_input.inc\'"])
        # In standalone and madloop we do no use the compiled param card but
        # still parse the .dat one so we must load it.
        if self.opt['loop_induced']:
            #loop induced follow MadEvent way to handle the card.
            load_card = ''
            lha_read_filename='lha_read.f' 
            updateloop_default = '.true.'           
        elif self.opt['export_format'] in ['madloop','madloop_optimized', 'madloop_matchbox']:
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read_mp.f'
            updateloop_default = '.true.'
        elif self.opt['export_format'].startswith('standalone') \
            or self.opt['export_format'] in ['madweight', 'plugin']\
            or self.opt['export_format'].startswith('matchbox'):
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read.f'
            updateloop_default = '.true.'
        else:
            load_card = ''
            lha_read_filename='lha_read.f'
            updateloop_default = '.false.'
            
        cp( MG5DIR + '/models/template_files/fortran/' + lha_read_filename, \
                                       os.path.join(self.dir_path,'lha_read.f'))
        
        file=file%{'includes':'\n      '.join(includes),
                   'load_card':load_card,
                   'updateloop_default': updateloop_default}
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
                text = text.replace('madevent','aMCatNLO').replace('../vector.inc', '')
                open(path, 'w').writelines(text)
        elif self.opt['export_format'] in ['standalone', 'standalone_msP','standalone_msF',
                                  'madloop','madloop_optimized', 'standalone_rw', 
                                  'madweight','matchbox','madloop_matchbox', 'plugin']:
            cp( MG5DIR + '/models/template_files/fortran/makefile_standalone', 
                self.dir_path + '/makefile')
        else:
            raise MadGraph5Error('Unknown format')

        if self.opt['export_format'].startswith('standalone'):
            cp( MG5DIR + '/Template/LO/Source/alfas_functions.f', 
                self.dir_path)
            cp( MG5DIR + '/Template/LO/Source/alfas.inc', 
                self.dir_path)

            fsock = open(pjoin(self.dir_path, '..', 'cuts.inc'),'w')
            fsock.write('''            
            logical fixed_extra_scale
            integer maxjetflavor
            double precision mue_over_ref
            double precision mue_ref_fixed
            common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
            ''')

            if self.model['running_elements']:
                cp( MG5DIR + '/Template/Running',  pjoin(self.dir_path, '..', 'RUNNING'))



    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        if self.opt['mp']:
            mp_fsock = self.open('mp_coupl.inc', format='fortran')
            mp_fsock_same_name = self.open('mp_coupl_same_name.inc',\
                                            format='fortran')

        # Write header
        header = """C
C NB: VECSIZE_MEMMAX is defined in vector.inc
C NB: vector.inc must be included before coupl.inc
C

                double precision G, all_G%(vec)s
                common/strong/ G, all_G
                 
                double complex gal(2)
                common/weak/ gal
                
                double precision MU_R, all_mu_r%(vec)s
                common/rscale/ MU_R, all_mu_r

                """   % {'vec': ('' if not self.vector_size else '(1)' if self.vector_size<=1 else '(VECSIZE_MEMMAX)')}
                ###   % {'vec': ("(VECSIZE_MEMMAX)" if self.vector_size else '')}
                ###   % {'vec': ("(%i)" % max(1,self.vector_size) if self.vector_size else '')}

        # Nf is the number of light quark flavours
        header = header+"""double precision Nf
                parameter(Nf=%dd0)
                """ % self.model.get_nflav()
        #Nl is the number of massless leptons
        header = header+"""double precision Nl
                parameter(Nl=%dd0)
                """ % self.model.get_nleps()
                
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
                                  'mp_prefix':self.mp_prefix,
                                  'vector_size': '(1)' if self.vector_size else ''})
            mp_fsock_same_name.writelines(header%{'real_mp_format':self.mp_real_format,
                                  'complex_mp_format':self.mp_complex_format,
                                  'mp_prefix':'',
                                  'vector_size': '' if self.vector_size else ''})

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
            masses = sorted(list(masses))
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
            widths = sorted(list(widths))
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
        if self.coups_indep:
            c_list = [coupl.name for coupl in self.coups_indep_noloop + self.coups_indep_loop]  
            if c_list:
                fsock.writelines('double complex '+', '.join(c_list)+'\n') 

        if self.vector_size:
            c_list = ['%s(%s)' %(coupl.name, "VECSIZE_MEMMAX") for coupl in self.coups_dep]
        else:
            c_list = [coupl.name for coupl in self.coups_dep] 
        
        if c_list:
            fsock.writelines('double complex '+', '.join(c_list)+'\n')   
        coupling_list = [coupl.name for coupl in self.coups_dep + self.coups_indep_noloop + self.coups_indep_loop]       

        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        if self.opt['mp']:
            c_list = [coupl.name for coupl in self.coups_indep] 
            if c_list: 
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                   ','.join(c_list)+'\n')
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                 self.mp_prefix+c for c in c_list])+'\n')
            if False: #no vector handling in quadruple for the moment
                c_list = ['%s(%s)' %(coupl.name, "VECSIZE_MEMMAX") for coupl in self.coups_dep]
            else:
                c_list = [coupl.name for coupl in self.coups_dep] 
            if c_list: 
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                   ','.join(c_list)+'\n')
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                 self.mp_prefix+c for c in c_list])+'\n')
            mp_fsock_same_name.writelines('common/MP_couplings/ '+\
                                                 ','.join(coupling_list)+'\n\n')                

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
        lines = [format(coupl) for coupl in self.coups_dep + self.coups_indep_noloop + self.coups_indep_loop]       
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

        real_parameters += ['mdl__%s__scale' % s for s in self.scales]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        if self.opt['mp']:
            mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                              self.mp_prefix+p for p in real_parameters])+'\n')
            mp_fsock.writelines('common/MP_T_params_R/ '+','.join([\
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
            if hasattr(self.model, "notused_ct_params"):
                if param.endswith(('_fin_','_1eps_','_2eps_')):
                    limit = -2
                elif param.endswith(('_1eps','_2eps')):
                    limit =-1
                else:
                    limit = 0
                base = '_'.join(param.split('_')[1:limit])
                if base in self.model.notused_ct_params:
                    return False
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
        allCTparameters=list(self.model.map_CTcoup_CTparam.values())
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
        

    def create_printout(self):
        """create printout.f"""

        replace_dict = {'include_vector': "include '../vector.inc' ! VECSIZE_MEMMAX (needed by coupl.inc)"}

        if not self.vector_size:
            replace_dict['include_vector'] = ''

        fsock = self.open('printout.f', format='fortran')
        text = open(pjoin(MG5DIR , 'models', 'template_files','fortran', 'printout.f')).read()
        text = text % replace_dict
        fsock.write(text)


    def create_ewa(self):
        """create electroweakFlux.inc 
           this file only need the correct name for the mass for the W and Z
        """

        try:
            fsock = self.open(pjoin(self.dir_path,'../PDF/ElectroweakFlux.inc'), format='fortran')
        except:
            logger.debug('No PDF directory do not cfeate ElectroweakFlux.inc')
            return

        masses = {'MZ': '0d0', 'MW': '0d0'}
        count = 0
        for particle in self.model['particles']:
            if particle.get('pdg_code') == 24:
                masses['MW'] = particle.get('mass')
                count += 1
            elif particle.get('pdg_code') == 23:
                masses['MZ'] =  particle.get('mass')
                count += 1
            if count == 2:
                break

        template = open(pjoin(MG5DIR,'madgraph/iolibs/template_files/madevent_electroweakFlux.inc')).read()
        fsock.write(template % masses)                 
        fsock.close()

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

        # these are the parameters needed for the loops
        if hasattr(self, 'allCTparameters') and self.allCTparameters:
            ct_params = [param for param in self.params_dep \
                if self.check_needed_param(param.name) and \
                   param.name.lower() in self.allCTparameters]
        else:
            ct_params = []
        
        for param in self.params_dep:
            # skip the CT parameters, which have already been done before
            if not self.check_needed_param(param.name) or param in ct_params:
                continue
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            elif mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))

        fsock.write_comments('\nParameters that should be updated for the loops.\n')

        # do not skip the evaluation of these parameters in MP
        if not mp and ct_params: fsock.writelines('if (updateloop) then')
        for param in ct_params:
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            elif mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))

        if not mp and ct_params: fsock.writelines('endif')

        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")

        # Let us not necessarily investigate the presence of alpha_EW^-1 of Gf as an external parameter, but also just as a parameter
        if ('aEWM1',) in self.model['parameters'] or \
           any( ('aEWM1'.lower() in [p.name.lower() for p in p_list]) for p_list in self.model['parameters'].values() ):
            if dp:
                fsock.writelines(""" gal(1) = 3.5449077018110318d0 / DSQRT(ABS(aEWM1))
                                 gal(2) = 1d0
                         """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 2 * SQRT(MP__PI/ABS(MP__aEWM1))
                                 %(mp_prefix)sgal(2) = 1d0 
                                 """ %{'mp_prefix':self.mp_prefix})
                pass
        # in Gmu scheme, aEWM1 is not external but Gf is an exteranl variable
        elif ('Gf',) in self.model['parameters']:
            # Make sure to consider complex masses if the complex mass scheme is activated
            if self.opt['complex_mass']:
                mass_prefix = 'CMASS_MDL_'
            else:
                mass_prefix = 'MDL_'

            if dp:
                if self.opt['complex_mass']:
                    fsock.writelines(""" gal(1) = ABS(2.378414230005442133435d0*%(mass_prefix)sMW*SQRT(DCMPLX(1.0D0,0.0d0)-%(mass_prefix)sMW**2/%(mass_prefix)sMZ**2)*DSQRT(MDL_Gf))
                                 gal(2) = 1d0
                         """%{'mass_prefix':mass_prefix})
                else:
                    fsock.writelines(""" gal(1) = 2.378414230005442133435d0*%(mass_prefix)sMW*DSQRT(1D0-%(mass_prefix)sMW**2/%(mass_prefix)sMZ**2)*DSQRT(MDL_Gf)
                                 gal(2) = 1d0
                         """%{'mass_prefix':mass_prefix})
            elif mp:
                if self.opt['complex_mass']:
                    fsock.writelines(""" %(mp_prefix)sgal(1) = ABS(2*%(mp_prefix)s%(mass_prefix)sMW*SQRT(CMPLX(1e0_16,0.0e0_16,KIND=16)-%(mp_prefix)s%(mass_prefix)sMW**2/%(mp_prefix)s%(mass_prefix)sMZ**2)*SQRT(SQRT(2e0_16)*%(mp_prefix)sMDL_Gf))
                                 %(mp_prefix)sgal(2) = 1e0_16
                                 """ %{'mp_prefix':self.mp_prefix,'mass_prefix':mass_prefix})
                else:
                    fsock.writelines(""" %(mp_prefix)sgal(1) = 2*%(mp_prefix)s%(mass_prefix)sMW*SQRT(1e0_16-%(mp_prefix)s%(mass_prefix)sMW**2/%(mp_prefix)s%(mass_prefix)sMZ**2)*SQRT(SQRT(2e0_16)*%(mp_prefix)sMDL_Gf)
                                 %(mp_prefix)sgal(2) = 1e0_16
                                 """ %{'mp_prefix':self.mp_prefix,'mass_prefix':mass_prefix})

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

    nb_def_by_file = 50
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = self.nb_def_by_file
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        for i in range(nb_coup_indep_noloop):
            ##### For the independent couplings, we compute the double and multiple
            ##### precision ones together
            # For the EW sudakov approximation, because of the numerical derivatives
            # we need to separate MP vs DP also here
            data = self.coups_indep_noloop[nb_def_by_file * i: 
                             min(len(self.coups_indep_noloop), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data, dp=True, mp=False)

            if self.opt['mp']:
                self.create_couplings_part( i + 1, data, dp=False,mp=True)

        for i in range(nb_coup_indep_loop):
            ##### For the independent couplings, we compute the double and multiple
            ##### precision ones together
            # For the EW sudakov approximation, because of the numerical derivatives
            # we need to separate MP vs DP also here
            data = self.coups_indep_loop[nb_def_by_file * i: 
                             min(len(self.coups_indep_loop), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1 + nb_coup_indep_noloop, data, dp=True, mp=False)

            if self.opt['mp']:
                self.create_couplings_part( i + 1 + nb_coup_indep_noloop, data, dp=False,mp=True)
            
        for i in range(nb_coup_dep):
            # For the dependent couplings, we compute the double and multiple
            # precision ones in separate subroutines.
            nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                        dp=True, mp=False, vec=self.vector_size*self.nb_warp)
            if self.opt['mp']:
                self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                           dp=False, mp=True, vec=self.vector_size*self.nb_warp)
        
        
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
        if self.vector_size:
            fsock.writelines("include \'../vector.inc\'\n")
        if self.opt['mp']:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 
            
        fsock.writelines("""logical updateloop
                            common /to_updateloop/updateloop
                            include \'input.inc\'
                         """)

        fsock.writelines("""    
                            include \'coupl.inc\'
                            READLHA = .true.
                            include \'intparam_definition.inc\'""")
        if self.opt['mp']:
            fsock.writelines("if (updateloop) then\n")
            fsock.writelines("""include \'mp_intparam_definition.inc\'\n""")
            fsock.writelines("endif\n")
        
        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        


        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep_noloop)]))

        fsock.writelines('if (updateloop) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1 + nb_coup_indep_noloop) for i in range(nb_coup_indep_loop)]))
        fsock.writelines('\nendif\n')
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%(i)s(%(args)s)' %  {'i': nb_coup_indep + i + 1,
                                                    'args':'1' if self.vector_size  else ''} \
                      for i in range(nb_coup_dep)]))

        # the MP-version is there also for those couplings which do not depend 
        #  on the PSP
        if self.opt['mp']:
            fsock.write_comments('\ncouplings in multiple precision\n')

            fsock.writelines('if (updateloop) then\n')

            fsock.writelines('\n'.join(\
                    ['call mp_coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        
            fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

            fsock.writelines('\n'.join(\
                    ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))

            fsock.writelines('\nendif\n')

        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param(%(args)s)

                            implicit none
                            %(args_dep)s
                            double precision PI, ZERO
                            logical READLHA, FIRST
                            data first /.true./
                            save first
                            parameter  (PI=3.141592653589793d0)            
                            parameter  (ZERO=0d0)
                            logical updateloop
                            common /to_updateloop/updateloop
                            include \'model_functions.inc\'
                            double precision Gother
                            
                            double precision model_scale
                            common /model_scale/model_scale
                            """ % \
                            {'args': 'vecid' if (self.vector_size) else '',
                            'args_dep': ' integer vecid' if self.vector_size  else ''}
                         )


        if self.opt['export_format'] in ['madevent']:
            fsock.writelines("""
                            include \'../maxparticles.inc\'
                            include \'../cuts.inc\'
                             """)
            if self.vector_size:
                fsock.writelines("""
                            include \'../vector.inc\'
                                 """)
            fsock.writelines("""            
                            include \'../run.inc\'""")        
        elif self.opt['export_format'] in  ['madloop_optimized']:
            if self.vector_size:
                fsock.writelines("""
                            include \'../vector.inc\'
                                 """)
            fsock.writelines("""
                            include \'../maxparticles.inc\'
                            include \'../cuts.inc\'
                            include \'../run.inc\'""")
        else:
            fsock.writelines("""
                            include \'../cuts.inc\'
                            data maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed /5,.false.,1d0,91.188/
                            include \'../run.inc\'""")
        fsock.writelines("""
                            double precision alphas 
                            external alphas
                            """)

        fsock.writelines("""include \'input.inc\'
                            include \'coupl.inc\'
                            READLHA = .false.""")
        fsock.writelines("""    
                            include \'intparam_definition.inc\'\n
                            
                         """)
        
        if self.model['running_elements']:
            running_block = self.model.get_running(self.used_running_key) 
            if running_block:
                MUE = [p for p in self.model.get('parameters')[('external',)] if p.lhablock.lower() == 'loop' and tuple(p.lhacode) == (2,)]

                
                
                fsock.write_comments('calculate the running parameter')
                fsock.writelines(' if(fixed_extra_scale.and.first) then')
                if self.MUE:
                    fsock.writelines(' %s = mue_ref_fixed' % self.MUE.name)
                fsock.writelines(' Gother = SQRT(4.0D0*PI*ALPHAS(mue_ref_fixed))') 
                fsock.writelines(' first = .false.') 
                for i in range(len(running_block)):
                    fsock.writelines(" call C_RUNNING_%s(Gother) ! %s \n" % (i+1,list(running_block[i])))   
                fsock.writelines(' elseif(.not.fixed_extra_scale) then')
                fsock.writelines(' Gother = G')
                
                if self.MUE:
                    fsock.writelines(' %s = mue_over_ref*model_scale' % self.MUE.name)
                else:
                    misc.sprint('NO MUE')
                    #raise Exception
                
                fsock.writelines(' if(mue_over_ref.ne.1d0)then')
                fsock.writelines('  Gother = SQRT(4.0D0*PI*ALPHAS(mue_over_ref*model_scale))')
                fsock.writelines(' endif')
                
                for i in range(len(running_block)):
                    fsock.writelines(" call C_RUNNING_%s(Gother) ! %s \n" % (i+1,list(running_block[i])))   
                fsock.writelines('endif')

        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
                
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        if self.vector_size:
            fsock.writelines("""     ALL_G(VECID) = G   """)

        fsock.writelines('\n'.join(\
                    ['call coup%(i)s(%(args)s)' %  {"i": nb_coup_indep + i + 1, "args": 'vecid' if self.vector_size  else ''} \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param2(mu_r2,as2 %(args)s)

                            implicit none
                            
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            double precision mu_r2, as2
                            %(args_dep)s
                            include \'model_functions.inc\'"""%
                            {'args': ',vecid' if self.vector_size else '',
                            'args_dep': ' integer vecid' if self.vector_size else ''
                            })
        fsock.writelines("""include \'input.inc\'
                         """)
        if self.vector_size:
            fsock.writelines("       include \'../vector.inc\'\n")
        fsock.writelines("""include \'coupl.inc\'
                            double precision model_scale
                            common /model_scale/model_scale
                            """)
        fsock.writelines("""
                            if (mu_r2.gt.0d0) MU_R = DSQRT(mu_r2)
                            model_scale = DSQRT(mu_r2)
                            G = SQRT(4.0d0*PI*AS2) 
                            AS = as2

                            CALL UPDATE_AS_PARAM(%(args)s)
                         """%
                            {'args': 'vecid' if self.vector_size  else '',
                            'args_dep': ' integer vecid' if self.vector_size else ''
                            }
                            )
                         
                         
        fsock.writelines('''\n return \n end\n''')

        # fsock.writelines("""subroutine update_model_to_scale(scale)
        #                     ! scale in GeV
        #                     implicit none
        #                     double precision scale
        #                     double precision PI
        #                     double precision alphas
        #                     external alphas
        #                     parameter  (PI=3.141592653589793d0)
        #                     double precision mu_r2, as2
        #                     include \'model_functions.inc\'""")
        # fsock.writelines("""include \'input.inc\'
        #                     include \'../vector.inc\'
        #                     include \'coupl.inc\'
        #                     """)
        # fsock.writelines("""
        #                     AS = ALPHAS(scale)
        #                     AS2 = AS*AS
        #                     call update_as_param2(scale**2, AS2)
        #                  """)
        # fsock.writelines('''\n return \n end\n''')




        if self.opt['mp']:
            fsock.writelines("""subroutine mp_update_as_param()
    
                                implicit none
                                logical READLHA
                                include \'model_functions.inc\'""")
            if self.vector_size:
                fsock.writelines("""include \'../vector.inc\'\n""")
            fsock.writelines("""%s MP__PI, MP__ZERO
                                    parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                    parameter (MP__ZERO=0e0_16)
                                    include \'mp_input.inc\'
                                    include \'mp_coupl.inc\'
                            """%self.mp_real_format)
            fsock.writelines("""include \'input.inc\'""")

            fsock.writelines("""include \'coupl.inc\'
                                include \'actualize_mp_ext_params.inc\'
                                READLHA = .false.
                                include \'mp_intparam_definition.inc\'\n
                             """)
            
            nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
            nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
            nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
            nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 

            if self.model['running_elements']:
                #running_block = self.model.get_running(self.used_running_key) 
                if running_block:
                    fsock.write_comments('calculate the running parameter')
                    for i in range(len(running_block)):
                        fsock.writelines(" call MP_C_RUNNING_%s(G) ! %s \n" % (i+1,list(running_block[i])))   
            
                    
            fsock.write_comments('\ncouplings needed to be evaluated points by points\n')
    
            fsock.writelines('\n'.join(\
                        ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                          for i in range(nb_coup_dep)]))
            fsock.writelines('''\n return \n end\n''')
            
        if self.model['running_elements'] and running_block:
            self.write_running_blocks(fsock, running_block)
    
    def write_running_blocks(self, fsock, running_block):
        
        for block_nb, runparams in enumerate(running_block):
            text = self.write_one_running_block(block_nb, runparams)
            fsock.writelines(text)
            
    
    template_running_gs_gs2 = """
                  SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

      include 'input.inc'
      %(mpinput)s

      include '../cuts.inc'
      include '../vector.inc'
      INCLUDE 'coupl.inc'
      double precision GMU


      double complex mat1(%(size)i,%(size)i), mat2(%(size)i,%(size)i), fullmat(%(size)i,%(size)i), matexp(%(size)i,%(size)i)
      data mat2 /%(mat2)s/
      data mat1 /%(mat1)s/
      double precision C0(%(size)i),Cout(%(size)i)
      data C0 /%(size)i * 0d0/
      logical first
      data first /.true./
      integer i,j,k
      double precision G0,beta0, alphas
      external alphas
      data G0 /0d0/
      double precision r1,r2
      if (first) then
         %(initc0)s
         G0 = SQRT(4.0D0*PI*ALPHAS(mdl__%(scale)s__scale))
         %(check_scale)s
         first = .false.
      endif
      beta0 = 11. - 2./3. * maxjetflavor
      r1 = (1/GMU -1/G0)/ beta0
      r2 = DLOG(G0/GMU)/beta0
      do j=1,%(size)i
         do i=1,%(size)i
            fullmat(j,i) = mat1(j,i) *r1 + mat2(j,i)*r2
         enddo
      enddo
      call c8mat_expm1( %(size)i, fullmat, matexp)
      do j=1,%(size)i
         Cout(j) = 0d0
      enddo

      do i=1,%(size)i
         do j=1,%(size)i
            Cout(j) = Cout(j) + matexp(j,i) * c0(i)
         enddo
      enddo

      %(assignc)s

      return
      end
            """
            
    template_running_gs2 = """
                  SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

      include '../cuts.inc'
      include '../vector.inc'
      INCLUDE 'input.inc'
      %(mpinput)s
      INCLUDE 'coupl.inc'
      double precision GMU

      double complex mat2(%(size)i,%(size)i), fullmat(%(size)i,%(size)i), matexp(%(size)i,%(size)i)
      data mat2 /%(mat2)s/
      double precision C0(%(size)i),Cout(%(size)i)
      data C0 /%(size)i * 0d0/
      logical first
      data first /.true./
      integer i,j,k
      double precision G0,beta0, alphas
      external alphas
      data G0 /0d0/
      double precision r1,r2
      if (first) then
         %(initc0)s
         G0 = SQRT(4.0D0*PI*ALPHAS(mdl__%(scale)s__scale))
         %(check_scale)s
         first = .false.
      endif
      beta0 = 11. - 2./3. * maxjetflavor
      r2 = DLOG(G0/GMU) / beta0 
      do j=1,%(size)i
         do i=1,%(size)i
            fullmat(j,i) = mat2(j,i)*r2
         enddo
      enddo
      call c8mat_expm1( %(size)i, fullmat, matexp)
      do j=1,%(size)i
         Cout(j) = 0d0
      enddo

      do i=1,%(size)i
         do j=1,%(size)i
            Cout(j) = Cout(j) + matexp(j,i) * c0(i)
         enddo
      enddo

      %(assignc)s

      return
      end
            """
            
    template_running_x3 = """
    SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

       include '../cuts.inc'
       include '../vector.inc'
      INCLUDE 'input.inc'
      %(mpinput)s
      INCLUDE 'coupl.inc'
      double precision GMU

      double complex mat3
      data mat3 /%(mat3)s/
      double precision C0
      data C0 /0d0/
      logical first
      data first /.true./
      integer i,j,k
      if (first) then
         C0 = %(mp)s%(initc0)s
         first = .false.
         %(check_scale)s
      endif
      
      %(mp)s%(assignc)s =  1/DSQRT( 1/C0/C0 - 2*mat3 *DLOG(MU_R/mdl__%(scale)s__scale))
      
      return
      end
      """
    
    def get_scales(self):

        scales = set()
        
        for elements in self.model["running_elements"]:
            for params in elements.run_objects:
                sparams = [str(p) for p in params]
                if not any(param in runparams for param in sparams):
                    continue
                if 'aS' in sparams or sparams.count('G') == 2:
                    to_update = mat2
                    prefact = 4*math.pi
                    try:
                        sparams.remove('aS')
                    except:
                        sparams.remove('G')
                        sparams.remove('G')
                else:
                    to_update = mat1
                    sparams.remove('G')
                    prefact = 16*math.pi**2
                    
                if len(sparams) == 3:
                    if len(set(sparams)) !=1:
                        raise Exception( "Not supported type of running")
                    mat3 = eval(elements.value)
                    continue
                elif len(sparams) !=2:
                    raise Exception("Not supported type of running")
                id1 = runparams.index(sparams[0])
                id2 = runparams.index(sparams[1])
                assert to_update[id1][id2] == 0
                to_update[id1][id2] = eval(elements.value)*prefact
                for param in params:
                    scales.add(param.lhablock)

        try:
            scales.remove('SMINPUTS')
        except Exception:
            pass

        return scales


    def write_one_running_block(self, block_nb, runparams):
               
        runparams = list(runparams)
        
        size = len(runparams) 
        mat1=[[0]*size for _ in range(size)]
        mat2=[[0]*size for _ in range(size)]
        mat3=0
        scales = set()
        
        for elements in self.model["running_elements"]:
            for params in elements.run_objects:
                sparams = [str(p) for p in params]
                if not any(param in runparams for param in sparams):
                    continue
                if 'aS' in sparams or sparams.count('G') == 2:
                    to_update = mat2
                    prefact = 4*math.pi
                    try:
                        sparams.remove('aS')
                    except:
                        sparams.remove('G')
                        sparams.remove('G')
                else:
                    to_update = mat1
                    sparams.remove('G')
                    prefact = 16*math.pi**2
                    
                if len(sparams) == 3:
                    if len(set(sparams)) !=1:
                        raise Exception( "Not supported type of running")
                    mat3 = eval(elements.value)
                    continue
                elif len(sparams) !=2:
                    raise Exception("Not supported type of running")
                id1 = runparams.index(sparams[0])
                id2 = runparams.index(sparams[1])
                assert to_update[id1][id2] == 0
                try:
                    to_update[id1][id2] = eval(elements.value)*prefact
                except Exception:
                    to_update[id1][id2] = '%s *( %s)' % (prefact, elements.value) 

                for param in params:
                    scales.add(param.lhablock)

        try:
            scales.remove('SMINPUTS')
        except Exception:
            pass
        
        data = {}
        data['block_nb'] = block_nb+1
        data['size'] = size
        data['mp'] = ''
        if mat3:
            template = self.template_running_x3
            data['mat3']
            data['initc0'] = "MDL_%s" % runparams[0]
            data['assignc'] = "MDL_%s" % runparams[0]
            text = template % data
            if self.opt['mp']:
                data['mp'] = 'MP_'
                data['initc0'] = "MP__MDL_%s" % runparams[0]
                data['assignc'] = "MP__MDL_%s" % runparams[0]
                text += template % data 
            return text
        
        data['initc0'] = "\n".join(["c0(%i) = MDL_%s" % (i+1, name)
                                    for i, name in enumerate(runparams)])
        data['assignc'] = "\n".join(["MDL_%s = COUT(%i)" % (name,i+1)
                                    for i, name in enumerate(runparams)])
        data['mp'] = ''
        data['check_scale'] = ''
        
        if len(scales) == 1:
            data['scale'] = scales.pop()
        else:
            one_scale = scales.pop()
            data['scale'] = one_scale
            for scale in scales:
                check_scale = """ if (MDL__%(1)s__SCALE.ne.MDL__%(2)s__SCALE) then
                write(*,*) 'ERROR scale %(1)s and %(2)s need to be equal for the running'
                stop 5
                endif
                """
                data['check_scale'] += check_scale % {'1': one_scale, '2': scale}           

        # need to compute the matrices
        # carefull some component are proportional to aS
        # need to convert those to G^2
        # need to be carefull with prefactor included (none yet)

        
        
        
        
        data['mat1'] = ",".join(["%e" % mat1[j][i] if not isinstance(mat1[j][i], str) else "%e" %0  for i in range(data['size']) for j in range(data['size'])])
        data['mat2'] = ",".join(["%e" % mat2[j][i] if not isinstance(mat2[j][i], str) else "%e" %0 for i in range(data['size']) for j in range(data['size'])])
        
        # add initialization for parameter that have coupling parameter
        for i in range(data['size']):
            for j in range(data['size']):
                if isinstance(mat1[i][j], str):
                    towrite = mat1[i][j].replace('cmath.pi', 'pi')
                    towrite = towrite.replace('cmath.sqrt(', 'SQRT(1d0*')
                    towrite = towrite.replace('math.pi', 'pi')
                    towrite = towrite.replace('math.sqrt(', 'SQRT(1d0*')
                    data['initc0'] += "\n   MAT1(%i,%i) = %s" % (i+1, j+1, towrite)
                if isinstance(mat2[i][j], str):
                    towrite = mat2[i][j].replace('cmath.pi', 'pi')
                    towrite = towrite.replace('cmath.sqrt(', 'SQRT(1d0*')
                    towrite = towrite.replace('math.pi', 'pi')
                    towrite = towrite.replace('math.sqrt(', 'SQRT(1d0*')
                    data['initc0'] += "\n   MAT2(%i,%i) = %s" % (i+1, j+1, towrite)

        data['mpinput'] =''
        if any(mat1[i][j] for i,j in zip(range(size),range(size))):
            template = self.template_running_gs_gs2
        else:
            template = self.template_running_gs2
        
        text = template % data
        if self.opt['mp']:
            data['mp'] = 'MP_'
            data['mpinput']="INCLUDE 'mp_input.inc'"
            data['initc0'] = "\n".join(["c0(%i) = MP__MDL_%s" % (i+1, name)
                                    for i, name in enumerate(runparams)])
            # add initialization for parameter that have coupling parameter
            for i in range(data['size']):
                for j in range(data['size']):
                    if isinstance(mat1[i][j], str):
                        towrite = mat1[i][j].replace('cmath.pi', 'MP__pi')
                        towrite = towrite.replace('cmath.sqrt(', 'SQRT((1_E16*')
                        towrite = towrite.replace('math.pi', 'MP__pi')
                        towrite = towrite.replace('math.sqrt(', 'SQRT(1_E16*')
                        data['initc0'] += "\n   MAT1(%i,%i) = %s" % (i+1, j+1, mat1[i][j].replace('MDL_', 'MP__MDL_'))
                    if isinstance(mat2[i][j], str):
                        towrite = mat2[i][j].replace('cmath.pi', 'MP__pi')
                        towrite = towrite.replace('cmath.sqrt(', 'SQRT((1_E16*')
                        towrite = towrite.replace('math.pi', 'MP__pi')
                        towrite = towrite.replace('math.sqrt(', 'SQRT(1_E16*')
                        data['initc0'] += "\n   MAT2(%i,%i) = %s" % (i+1, j+1, mat2[i][j].replace('MDL_', 'MP__MDL_'))

            data['assignc'] = "\n".join(["MP__MDL_%s = COUT(%i)" % (name,i+1)
                                    for i, name in enumerate(runparams)])
            text += template % data   
            
        return text

    def create_couplings_part(self, nb_file, data, dp=True, mp=False, vec=False):
        """ create couplings[nb_file].f containing information coming from data.
        Outputs the computation of the double precision and/or the multiple
        precision couplings depending on the parameters dp and mp.
        If mp is True and dp is False, then the prefix 'MP_' is appended to the
        filename and subroutine name.
        """
        
        fsock = self.open('%scouplings%s.f' %('mp_' if mp and not dp else '',
                                                     nb_file), format='fortran')
        fsock.writelines("""subroutine %(mp)scoup%(nb_file)s( %(args)s)
          
          implicit none
          %(def_args)s
          include \'model_functions.inc\'"""% {'mp': 'mp_' if mp and not dp else '',
                                               'nb_file': nb_file,
                                               'args': 'vecid' if (vec and not mp) else '',
                                               'def_args': '  integer vecid' if vec else ''})

        if self.vector_size:
            fsock.writelines("""include '../vector.inc'\n""")

        if dp:
            fsock.writelines("""
              double precision PI, ZERO
              parameter  (PI=3.141592653589793d0)
              parameter  (ZERO=0d0)
              include 'input.inc'""")
            fsock.writelines("""include 'coupl.inc'""")
        if mp:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 

        for coupling in data:
            if dp:  

                fsock.writelines('%(name)s%(index)s = %(expr)s' % {'name': coupling.name,
                                          'index': '(vecid)' if vec else '',
                                          'expr': self.p_to_f.parse(coupling.expr)})
            if mp:
                fsock.writelines('%(mp)s%(name)s%(index)s = %(expr)s' % {'mp': self.mp_prefix,
                                          'name': coupling.name,
                                          'index': '', #no vectorization in quadruple
                                          'expr': self.mp_p_to_f.parse(coupling.expr)})
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
                if str(fct.name) not in ["complexconjugate", "re", "im", "sec", 
                       "csc", "asec", "acsc", "theta_function", "cond", 
                       "condif", "reglogp", "reglogm", "reglog", "recms", "arg",
                                    "grreglog","regsqrt","B0F","b0f","sqrt_trajectory",
                                    "log_trajectory"]:
                    additional_fct.append(fct.name)
        # put in lower case and remove duplicate
        additional_fct = list({f.lower():'' for f in additional_fct if f.lower() not in ['condif', 'reglog', 'reglogp', 'reglogm', 'recms', 'arg', 'grreglog', 'regsqrt']}) 
        fsock = self.open('model_functions.inc', format='fortran')
        fsock.writelines("""double complex cond
          double complex condif
          double complex reglog
          double complex reglogp
          double complex reglogm
          double complex regsqrt
          double complex grreglog
          double complex recms
          double complex arg
          double complex B0F
          double complex sqrt_trajectory
          double complex log_trajectory
          %s
          """ % "\n".join(["          double complex %s" % i for i in additional_fct]))

        
        if self.opt['mp']:
            fsock.writelines("""%(complex_mp_format)s mp_cond
          %(complex_mp_format)s mp_condif
          %(complex_mp_format)s mp_reglog
          %(complex_mp_format)s mp_reglogp
          %(complex_mp_format)s mp_reglogm
          %(complex_mp_format)s mp_regsqrt
          %(complex_mp_format)s mp_grreglog
          %(complex_mp_format)s mp_recms
          %(complex_mp_format)s mp_arg
          %(complex_mp_format)s mp_B0F
          %(complex_mp_format)s mp_sqrt_trajectory
          %(complex_mp_format)s mp_log_trajectory
          %(additional)s
          """ %\
          {"additional": "\n".join(["          %s mp_%s" % (self.mp_complex_format, i) for i in additional_fct]),
           'complex_mp_format':self.mp_complex_format
           }) 

    def create_model_functions_def(self):
        """ Create model_functions.f which contains the various definitions
        of auxiliary functions which might be used in the couplings expressions
        Add the functions.f functions for formfactors support
        """

        fsock = self.open('model_functions.f', format='fortran')
        fsock.writelines(r"""double complex function cond(condition,truecase,falsecase)
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

          double complex function reglog(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          if(arg.eq.(0.0d0,0.0d0)) then
             reglog=(0.0d0,0.0d0)
          else
             reglog=log(arg)
          endif
          end

          double complex function reglogp(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
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

          double complex function reglogm(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
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

          double complex function regsqrt(arg_in)
          implicit none
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          regsqrt=sqrt(arg)
          end

          double complex function grreglog(logsw,expr1_in,expr2_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex expr1_in,expr2_in
          double complex expr1,expr2
          double precision logsw
          double precision imagexpr
          logical firstsheet
          expr1=expr1_in
          expr2=expr2_in
          if(dabs(dimag(expr1)).eq.0.0d0)then
             expr1=dcmplx(dble(expr1),0.0d0)
          endif
          if(dabs(dble(expr1)).eq.0.0d0)then
             expr1=dcmplx(0.0d0,dimag(expr1))
          endif
          if(dabs(dimag(expr2)).eq.0.0d0)then
             expr2=dcmplx(dble(expr2),0.0d0)
          endif
          if(dabs(dble(expr2)).eq.0.0d0)then
             expr2=dcmplx(0.0d0,dimag(expr2))
          endif
          if(expr1.eq.(0.0d0,0.0d0))then
             grreglog=(0.0d0,0.0d0)
          else
             imagexpr=dimag(expr1)*dimag(expr2)
             firstsheet=imagexpr.ge.0.0d0
             firstsheet=firstsheet.or.dble(expr1).ge.0.0d0
             firstsheet=firstsheet.or.dble(expr2).ge.0.0d0
             if(firstsheet)then
                grreglog=log(expr1)
             else
                if(dimag(expr1).gt.0.0d0)then
                   grreglog=log(expr1) - logsw*TWOPII
                else
                   grreglog=log(expr1) + logsw*TWOPII
                endif
             endif
          endif
          end

          module b0f_caching

          type b0f_node
          double complex p2,m12,m22
          double complex value
          type(b0f_node),pointer::parent
          type(b0f_node),pointer::left
          type(b0f_node),pointer::right
          end type b0f_node

          contains

          subroutine b0f_search(item, head, find)
          implicit none
          type(b0f_node),pointer,intent(inout)::head,item
          logical,intent(out)::find
          type(b0f_node),pointer::item1
          integer::icomp
          find=.false.
          nullify(item%parent)
          nullify(item%left)
          nullify(item%right)
          if(.not.associated(head))then
             head => item
             return
          endif
          item1 => head
          do
             icomp=b0f_node_compare(item,item1)
             if(icomp.lt.0)then
                if(.not.associated(item1%left))then
                   item1%left => item
                   item%parent => item1
                   exit
                else
                   item1 => item1%left
                endif
             elseif(icomp.gt.0)then
                if(.not.associated(item1%right))then
                   item1%right => item
                   item%parent => item1
                   exit
                else
                   item1 => item1%right
                endif
             else
                find=.true.
                item%value=item1%value
                exit
             endif
          enddo
          return
          end

          integer function b0f_node_compare(item1,item2) result(res)
          implicit none
          type(b0f_node),pointer,intent(in)::item1,item2
          res=complex_compare(item1%p2,item2%p2)
          if(res.ne.0)return
          res=complex_compare(item1%m22,item2%m22)
          if(res.ne.0)return
          res=complex_compare(item1%m12,item2%m12)
          return
          end

          integer function real_compare(r1,r2) result(res)
          implicit none
          double precision r1,r2
          double precision maxr,diff
          double precision tiny
          parameter (tiny=-1d-14)
          maxr=max(abs(r1),abs(r2))
          diff=r1-r2
          if(maxr.le.1d-99.or.abs(diff)/max(maxr,1d-99).le.abs(tiny))then
             res=0
             return
          endif
          if(diff.gt.0d0)then
             res=1
             return
          else
             res=-1
             return
          endif
          end

          integer function complex_compare(c1,c2) result(res)
          implicit none
          double complex c1,c2
          double precision r1,r2
          r1=dble(c1)
          r2=dble(c2)
          res=real_compare(r1,r2)
          if(res.ne.0)return
          r1=dimag(c1)
          r2=dimag(c2)
          res=real_compare(r1,r2)
          return
          end

          end module b0f_caching

          double complex function B0F(p2,m12,m22)
          use b0f_caching
          implicit none
          double complex p2,m12,m22
          double complex zero,TWOPII
          parameter (zero=(0.0d0,0.0d0))
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double precision M,M2,Ga,Ga2
          double precision tiny
          parameter (tiny=-1d-14)
          double complex logterms
          double complex log_trajectory
          logical use_caching
          parameter (use_caching=.true.)
          type(b0f_node),pointer::item
          type(b0f_node),pointer,save::b0f_bt
          integer init
          save init
          data init /0/
          logical find
          IF(m12.eq.zero)THEN
c           it is a special case
c           refer to Eq.(5.48) in arXiv:1804.10017
            M=DBLE(p2) ! M^2
            M2=DBLE(m22) ! M2^2
            IF(M.LT.tiny.OR.M2.LT.tiny)THEN
            WRITE(*,*)'ERROR:B0F is not well defined when M^2,M2^2<0'
            STOP
            ENDIF
            M=DSQRT(DABS(M))
            M2=DSQRT(DABS(M2))
            IF(M.EQ.0d0)THEN
               Ga=0d0
            ELSE
               Ga=-DIMAG(p2)/M
            ENDIF
            IF(M2.EQ.0d0)THEN
               Ga2=0d0
            ELSE
               Ga2=-DIMAG(m22)/M2
            ENDIF
            IF(p2.ne.m22.and.p2.ne.zero.and.m22.ne.zero)THEN
               b0f=(m22-p2)/p2*LOG((m22-p2)/m22)
               IF(M.GT.M2.and.Ga*M2.GT.Ga2*M)THEN
                  b0f=b0f-TWOPII
               ENDIF
               RETURN
            ELSE
                WRITE(*,*)'ERROR:B0F is not supported for a simple form'
                STOP
            ENDIF
          ENDIF
c         the general case
c         trajectory method as advocated in arXiv:1804.10017 (Eq.(E.47))
          if(use_caching)then
             if(init.eq.0)then
                nullify(b0f_bt)
                init=1
             endif
             allocate(item)
             item%p2=p2
             item%m12=m12
             item%m22=m22
             find=.false.
             call b0f_search(item,b0f_bt,find)
             if(find)then
                b0f=item%value
                deallocate(item)
                return
             else
                logterms=log_trajectory(100,p2,m12,m22)
                b0f=-LOG(p2/m22)+logterms
                item%value=b0f
                return
             endif
          else
             logterms=log_trajectory(100,p2,m12,m22)
             b0f=-LOG(p2/m22)+logterms
          endif
          RETURN
          end

          double complex function sqrt_trajectory(n_seg,p2,m12,m22)
c         only needed when p2*m12*m22=\=0
          implicit none
          integer n_seg ! number of segments
          double complex p2,m12,m22
          double complex zero,one
          parameter (zero=(0.0d0,0.0d0),one=(1.0d0,0.0d0))
          double complex gamma0,gamma1
          double precision M,Ga,dGa,Ga_start
          double precision Gai,intersection
          double complex argim1,argi,p2i
          double complex gamma0i,gamma1i
          double precision tiny
          parameter (tiny=-1d-24)
          integer i
          double precision prefactor
          IF(ABS(p2*m12*m22).EQ.0d0)THEN
            WRITE(*,*)'ERROR:sqrt_trajectory works when p2*m12*m22/=0'
            STOP
          ENDIF
          M=DBLE(p2) ! M^2
          M=DSQRT(DABS(M))
          IF(M.EQ.0d0)THEN
             Ga=0d0
          ELSE
             Ga=-DIMAG(p2)/M
          ENDIF
c         Eq.(5.37) in arXiv:1804.10017
          gamma0=one+m12/p2-m22/p2
          gamma1=m12/p2-dcmplx(0d0,1d0)*ABS(tiny)/p2
          IF(ABS(Ga).EQ.0d0)THEN
             sqrt_trajectory=SQRT(gamma0**2-4d0*gamma1)
             RETURN
          ENDIF
c         segments from -DABS(tiny*Ga) to Ga
          Ga_start=-DABS(tiny*Ga)
          dGa=(Ga-Ga_start)/n_seg
          prefactor=1d0
          Gai=Ga_start
          p2i=dcmplx(M**2,-Gai*M)
          gamma0i=one+m12/p2i-m22/p2i
          gamma1i=m12/p2i-dcmplx(0d0,1d0)*ABS(tiny)/p2i
          argim1=gamma0i**2-4d0*gamma1i
          DO i=1,n_seg
             Gai=dGa*i+Ga_start
             p2i=dcmplx(M**2,-Gai*M)
             gamma0i=one+m12/p2i-m22/p2i
             gamma1i=m12/p2i-dcmplx(0d0,1d0)*ABS(tiny)/p2i
             argi=gamma0i**2-4d0*gamma1i
             IF(DIMAG(argi)*DIMAG(argim1).LT.0d0)THEN
                intersection=DIMAG(argim1)*(DBLE(argi)-DBLE(argim1))
                intersection=intersection/(DIMAG(argi)-DIMAG(argim1))
                intersection=intersection-DBLE(argim1)
                IF(intersection.GT.0d0)THEN
                   prefactor=-prefactor
                ENDIF
             ENDIF
             argim1=argi
          ENDDO
          sqrt_trajectory=SQRT(gamma0**2-4d0*gamma1)*prefactor
          RETURN
          end

          double complex function log_trajectory(n_seg,p2,m12,m22)
c         sum of log terms appearing in Eq.(5.35) of arXiv:1804.10017
c         only needed when p2*m12*m22=\=0
          implicit none
c         4 possible logarithms appearing in Eq.(5.35) of arXiv:1804.10017
c         log(arg(i)) with arg(i) for i=1 to 4
c         i=1: (ga_{+}-1)
c         i=2: (ga_{-}-1)
c         i=3: (ga_{+}-1)/ga_{+}
c         i=4: (ga_{-}-1)/ga_{-}
          integer n_seg ! number of segments
          double complex p2,m12,m22
          double complex zero,one,half,TWOPII
          parameter (zero=(0.0d0,0.0d0),one=(1.0d0,0.0d0))
          parameter (half=(0.5d0,0.0d0))
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex gamma0,gammap,gammam,sqrtterm
          double precision M,Ga,dGa,Ga_start
          double precision Gai,intersection
          double complex argim1(4),argi(4),p2i,sqrttermi
          double complex gamma0i,gammapi,gammami
          double precision tiny
          parameter (tiny=-1d-14)
          integer i,j
          double complex addfactor(4)
          double complex sqrt_trajectory
          IF(ABS(p2*m12*m22).EQ.0d0)THEN
            WRITE(*,*)'ERROR:log_trajectory works when p2*m12*m22/=0'
            STOP
          ENDIF
          M=DBLE(p2) ! M^2
          M=DSQRT(DABS(M))
          IF(M.EQ.0d0)THEN
             Ga=0d0
          ELSE
             Ga=-DIMAG(p2)/M
          ENDIF
c         Eq.(5.36-5.38) in arXiv:1804.10017
          sqrtterm=sqrt_trajectory(n_seg,p2,m12,m22)
          gamma0=one+m12/p2-m22/p2
          gammap=half*(gamma0+sqrtterm)
          gammam=half*(gamma0-sqrtterm)
          IF(ABS(Ga).EQ.0d0)THEN
             log_trajectory=-LOG(gammap-one)-LOG(gammam-one)+gammap*LOG((gammap-one)/gammap)+gammam*LOG((gammam-one)/gammam)
             RETURN
          ENDIF
c         segments from -DABS(tiny*Ga) to Ga
          Ga_start=-DABS(tiny*Ga)
          dGa=(Ga-Ga_start)/n_seg
          addfactor(1:4)=zero
          Gai=Ga_start
          p2i=dcmplx(M**2,-Gai*M)
          sqrttermi=sqrt_trajectory(n_seg,p2i,m12,m22)
          gamma0i=one+m12/p2i-m22/p2i
          gammapi=half*(gamma0i+sqrttermi)
          gammami=half*(gamma0i-sqrttermi)
          argim1(1)=gammapi-one
          argim1(2)=gammami-one
          argim1(3)=(gammapi-one)/gammapi
          argim1(4)=(gammami-one)/gammami
          DO i=1,n_seg
             Gai=dGa*i+Ga_start
             p2i=dcmplx(M**2,-Gai*M)
             sqrttermi=sqrt_trajectory(n_seg,p2i,m12,m22)
             gamma0i=one+m12/p2i-m22/p2i
             gammapi=half*(gamma0i+sqrttermi)
             gammami=half*(gamma0i-sqrttermi)
             argi(1)=gammapi-one
             argi(2)=gammami-one
             argi(3)=(gammapi-one)/gammapi
             argi(4)=(gammami-one)/gammami
             DO j=1,4
                IF(DIMAG(argi(j))*DIMAG(argim1(j)).LT.0d0)THEN
                   intersection=DIMAG(argim1(j))*(DBLE(argi(j))-DBLE(argim1(j)))
                   intersection=intersection/(DIMAG(argi(j))-DIMAG(argim1(j)))
                   intersection=intersection-DBLE(argim1(j))
                   IF(intersection.GT.0d0)THEN
                      IF(DIMAG(argim1(j)).LT.0)THEN
                         addfactor(j)=addfactor(j)-TWOPII
                      ELSE
                         addfactor(j)=addfactor(j)+TWOPII
                      ENDIF
                   ENDIF
                ENDIF
                argim1(j)=argi(j)
              ENDDO
          ENDDO
          log_trajectory=-(LOG(gammap-one)+addfactor(1))-(LOG(gammam-one)+addfactor(2))
          log_trajectory=log_trajectory+gammap*(LOG((gammap-one)/gammap)+addfactor(3))
          log_trajectory=log_trajectory+gammam*(LOG((gammam-one)/gammam)+addfactor(4))
          RETURN
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

              
              %(complex_mp_format)s function mp_reglog(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              if(arg.eq.(0.0e0_16,0.0e0_16)) then
                 mp_reglog=(0.0e0_16,0.0e0_16)
              else
                 mp_reglog=log(arg)
              endif
              end

              %(complex_mp_format)s function mp_reglogp(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
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
              
              %(complex_mp_format)s function mp_reglogm(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
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

              %(complex_mp_format)s function mp_regsqrt(arg_in)
              implicit none
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              mp_regsqrt=sqrt(arg)
              end

              %(complex_mp_format)s function mp_grreglog(logsw,expr1_in,expr2_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s expr1_in,expr2_in
              %(complex_mp_format)s expr1,expr2
              %(real_mp_format)s logsw
              %(real_mp_format)s imagexpr
              logical firstsheet
              expr1=expr1_in
              expr2=expr2_in
              if(abs(imagpart(expr1)).eq.0.0e0_16)then
                 expr1=cmplx(real(expr1,kind=16),0.0e0_16)
              endif
              if(abs(real(expr1,kind=16)).eq.0.0e0_16)then
                 expr1=cmplx(0.0e0_16,imagpart(expr1))
              endif
              if(abs(imagpart(expr2)).eq.0.0e0_16)then
                 expr2=cmplx(real(expr2,kind=16),0.0e0_16)
              endif
              if(abs(real(expr2,kind=16)).eq.0.0e0_16)then
                 expr2=cmplx(0.0e0_16,imagpart(expr2))
              endif
              if(expr1.eq.(0.0e0_16,0.0e0_16))then
                 mp_grreglog=(0.0e0_16,0.0e0_16)
              else
                 imagexpr=imagpart(expr1)*imagpart(expr2)
                 firstsheet=imagexpr.ge.0.0e0_16
                 firstsheet=firstsheet.or.real(expr1,kind=16).ge.0.0e0_16
                 firstsheet=firstsheet.or.real(expr2,kind=16).ge.0.0e0_16
                 if(firstsheet)then
                    mp_grreglog=log(expr1)
                 else
                    if(imagpart(expr1).gt.0.0e0_16)then
                       mp_grreglog=log(expr1) - logsw*TWOPII
                    else
                       mp_grreglog=log(expr1) + logsw*TWOPII
                    endif
                 endif
              endif
              end

              module mp_b0f_caching

              type mp_b0f_node
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s value
              type(mp_b0f_node),pointer::parent
              type(mp_b0f_node),pointer::left
              type(mp_b0f_node),pointer::right
              end type mp_b0f_node

              contains

              subroutine mp_b0f_search(item, head, find)
              implicit none
              type(mp_b0f_node),pointer,intent(inout)::head,item
              logical,intent(out)::find
              type(mp_b0f_node),pointer::item1
              integer::icomp
              find=.false.
              nullify(item%%parent)
              nullify(item%%left)
              nullify(item%%right)
              if(.not.associated(head))then
                 head => item
                 return
              endif
              item1 => head
              do
                 icomp=mp_b0f_node_compare(item,item1)
                 if(icomp.lt.0)then
                    if(.not.associated(item1%%left))then
                       item1%%left => item
                       item%%parent => item1
                       exit
                    else
                       item1 => item1%%left
                    endif
                 elseif(icomp.gt.0)then
                    if(.not.associated(item1%%right))then
                       item1%%right => item
                       item%%parent => item1
                       exit
                     else
                       item1 => item1%%right
                     endif
                 else
                     find=.true.
                     item%%value=item1%%value
                     exit
                 endif
              enddo
              return
              end

              integer function mp_b0f_node_compare(item1,item2) result(res)
              implicit none
              type(mp_b0f_node),pointer,intent(in)::item1,item2
              res=mp_complex_compare(item1%%p2,item2%%p2)
              if(res.ne.0)return
              res=mp_complex_compare(item1%%m22,item2%%m22)
              if(res.ne.0)return
              res=mp_complex_compare(item1%%m12,item2%%m12)
              return
              end

              integer function mp_real_compare(r1,r2) result(res)
              implicit none
              %(real_mp_format)s r1,r2
              %(real_mp_format)s maxr,diff
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              maxr=max(abs(r1),abs(r2))
              diff=r1-r2
              if(maxr.le.1.0e-99_16.or.abs(diff)/max(maxr,1.0e-99_16).le.abs(tiny))then
                 res=0
                 return
              endif
              if(diff.gt.0.0e0_16)then
                 res=1
                 return
              else
                 res=-1
                 return
              endif
              end

              integer function mp_complex_compare(c1,c2) result(res)
              implicit none
              %(complex_mp_format)s c1,c2
              %(real_mp_format)s r1,r2
              r1=real(c1,kind=16)
              r2=real(c2,kind=16)
              res=mp_real_compare(r1,r2)
              if(res.ne.0)return
              r1=imagpart(c1)
              r2=imagpart(c2)
              res=mp_real_compare(r1,r2)
              return
              end

              end module mp_b0f_caching

              %(complex_mp_format)s function mp_b0f(p2,m12,m22)
              use mp_b0f_caching
              implicit none
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,TWOPII
              parameter (zero=(0.0e0_16,0.0e0_16))
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(real_mp_format)s M,M2,Ga,Ga2
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              %(complex_mp_format)s logterms
              %(complex_mp_format)s mp_log_trajectory
              logical use_caching
              parameter (use_caching=.true.)
              type(mp_b0f_node),pointer::item
              type(mp_b0f_node),pointer,save::b0f_bt
              integer init
              save init
              data init /0/
              logical find
              IF(m12.eq.zero)THEN
                 M=real(p2,kind=16)
                 M2=real(m22,kind=16)
                 IF(M.LT.tiny.OR.M2.LT.tiny)THEN
                 WRITE(*,*)'ERROR:MP_B0F is not well defined when M^2,M2^2<0'
                 STOP
                 ENDIF
                 M=sqrt(abs(M))
                 M2=sqrt(abs(M2))
                 IF(M.EQ.0.0e0_16)THEN
                    Ga=0.0e0_16
                 ELSE
                    Ga=-imagpart(p2)/M
                 ENDIF
                 IF(M2.EQ.0.0e0_16)THEN
                    Ga2=0.0e0_16
                 ELSE
                    Ga2=-imagpart(m22)/M2
                 ENDIF
                 IF(p2.NE.m22.AND.p2.NE.zero.AND.m22.NE.zero)THEN
                    mp_b0f=(m22-p2)/p2*log((m22-p2)/m22)
                    IF(M.GT.M2.AND.Ga*M2.GT.Ga2*M)THEN
                       mp_b0f=mp_b0f-TWOPII
                    ENDIF
                    RETURN
                 ELSE
                    WRITE(*,*)'ERROR:MP_B0F is not supported for a simple form'
                    STOP
                 ENDIF
              ENDIF
              if(use_caching)then
                 if(init.eq.0)then
                    nullify(b0f_bt)
                    init=1
                 endif
                 allocate(item)
                 item%%p2=p2
                 item%%m12=m12
                 item%%m22=m22
                 find=.false.
                 call mp_b0f_search(item, b0f_bt, find)
                 if(find)then
                    mp_b0f=item%%value
                    deallocate(item)
                    return
                 else
                    logterms=mp_log_trajectory(100,p2,m12,m22)
                    mp_b0f=-LOG(p2/m22)+logterms
                    item%%value=mp_b0f
                    return
                 endif
              else
                 logterms=mp_log_trajectory(100,p2,m12,m22)
                 mp_b0f=-LOG(p2/m22)+logterms
              endif
              RETURN
              end

              %(complex_mp_format)s function mp_sqrt_trajectory(n_seg,p2,m12,m22)
              implicit none
              integer n_seg
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,one
              parameter (zero=(0.0e0_16,0.0e0_16),one=(1.0e0_16,0.0e0_16))
              %(complex_mp_format)s gamma0,gamma1
              %(real_mp_format)s M,Ga,dGa,Ga_start
              %(real_mp_format)s Gai,intersection
              %(complex_mp_format)s argim1,argi,p2i
              %(complex_mp_format)s gamma0i,gamma1i
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-24_16)
              integer i
              %(real_mp_format)s prefactor
              IF(ABS(p2*m12*m22).EQ.0.0e0_16)THEN
              WRITE(*,*)'ERROR:mp_sqrt_trajectory works when p2*m12*m22/=0'
              STOP
              ENDIF
              M=real(p2,kind=16)
              M=sqrt(abs(M))
              IF(M.EQ.0.0e0_16)THEN
                 Ga=0.0e0_16
              ELSE
                 Ga=-imagpart(p2)/M
              ENDIF
              gamma0=one+m12/p2-m22/p2
              gamma1=m12/p2-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2
              IF(abs(Ga).EQ.0.0e0_16)THEN
                mp_sqrt_trajectory=sqrt(gamma0**2-4.0e0_16*gamma1)
                RETURN
              ENDIF
              Ga_start=-abs(tiny*Ga)
              dGa=(Ga-Ga_start)/n_seg
              prefactor=1.0e0_16
              Gai=Ga_start
              p2i=cmplx(M**2,-Gai*M)
              gamma0i=one+m12/p2i-m22/p2i
              gamma1i=m12/p2i-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2i
              argim1=gamma0i**2-4.0e0_16*gamma1i
              DO i=1,n_seg
                 Gai=dGa*i+Ga_start
                 p2i=cmplx(M**2,-Gai*M)
                 gamma0i=one+m12/p2i-m22/p2i
                 gamma1i=m12/p2i-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2i
                 argi=gamma0i**2-4.0e0_16*gamma1i
                 IF(imagpart(argi)*imagpart(argim1).LT.0.0e0_16)THEN
                   intersection=imagpart(argim1)*(real(argi,kind=16)-real(argim1,kind=16))
                   intersection=intersection/(imagpart(argi)-imagpart(argim1))
                   intersection=intersection-real(argim1,kind=16)
                   IF(intersection.GT.0.0e0_16)THEN
                      prefactor=-prefactor
                   ENDIF
                 ENDIF
                 argim1=argi
              ENDDO
              mp_sqrt_trajectory=sqrt(gamma0**2-4.0e0_16*gamma1)*prefactor
              RETURN
              end

              %(complex_mp_format)s function mp_log_trajectory(n_seg,p2,m12,m22)
              implicit none
              integer n_seg
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,one,half,TWOPII
              parameter (zero=(0.0e0_16,0.0e0_16),one=(1.0e0_16,0.0e0_16))
              parameter (half=(0.5e0_16,0.0e0_16))
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s gamma0,gammap,gammam,sqrtterm
              %(real_mp_format)s M,Ga,dGa,Ga_start
              %(real_mp_format)s Gai,intersection
              %(complex_mp_format)s argim1(4),argi(4),p2i,sqrttermi
              %(complex_mp_format)s gamma0i,gammapi,gammami
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              integer i,j
              %(complex_mp_format)s addfactor(4)
              %(complex_mp_format)s mp_sqrt_trajectory
              IF(abs(p2*m12*m22).eq.0.0e0_16)THEN
              WRITE(*,*)'ERROR:mp_log_trajectory works when p2*m12*m22/=0'
              STOP
              ENDIF
              M=real(p2,kind=16)
              M=sqrt(abs(M))
              IF(M.eq.0.0e0_16)THEN
                 Ga=0.0e0_16
              ELSE
                 Ga=-imagpart(p2)/M
              ENDIF
              sqrtterm=mp_sqrt_trajectory(n_seg,p2,m12,m22)
              gamma0=one+m12/p2-m22/p2
              gammap=half*(gamma0+sqrtterm)
              gammam=half*(gamma0-sqrtterm)
              IF(abs(Ga).EQ.0.0e0_16)THEN
                 mp_log_trajectory=-LOG(gammap-one)-LOG(gammam-one)+gammap*LOG((gammap-one)/gammap)+gammam*LOG((gammam-one)/gammam)
                 RETURN
              ENDIF
              Ga_start=-abs(tiny*Ga)
              dGa=(Ga-Ga_start)/n_seg
              addfactor(1:4)=zero
              Gai=Ga_start
              p2i=cmplx(M**2,-Gai*M)
              sqrttermi=mp_sqrt_trajectory(n_seg,p2i,m12,m22)
              gamma0i=one+m12/p2i-m22/p2i
              gammapi=half*(gamma0i+sqrttermi)
              gammami=half*(gamma0i-sqrttermi)
              argim1(1)=gammapi-one
              argim1(2)=gammami-one
              argim1(3)=(gammapi-one)/gammapi
              argim1(4)=(gammami-one)/gammami
              DO i=1,n_seg
                 Gai=dGa*i+Ga_start
                 p2i=cmplx(M**2,-Gai*M)
                 sqrttermi=mp_sqrt_trajectory(n_seg,p2i,m12,m22)
                 gamma0i=one+m12/p2i-m22/p2i
                 gammapi=half*(gamma0i+sqrttermi)
                 gammami=half*(gamma0i-sqrttermi)
                 argi(1)=gammapi-one
                 argi(2)=gammami-one
                 argi(3)=(gammapi-one)/gammapi
                 argi(4)=(gammami-one)/gammami
                 DO j=1,4
                    IF(imagpart(argi(j))*imagpart(argim1(j)).LT.0.0e0_16)THEN
                       intersection=imagpart(argim1(j))*(real(argi(j),kind=16)-real(argim1(j),kind=16))
                       intersection=intersection/(imagpart(argi(j))-imagpart(argim1(j)))
                       intersection=intersection-real(argim1(j),kind=16)
                       IF(intersection.GT.0.0e0_16)THEN
                          IF(imagpart(argim1(j)).LT.0.0e0_16)THEN
                             addfactor(j)=addfactor(j)-TWOPII
                          ELSE
                             addfactor(j)=addfactor(j)+TWOPII
                          ENDIF
                       ENDIF
                    ENDIF
                    argim1(j)=argi(j)
                 ENDDO
              ENDDO
              mp_log_trajectory=-(LOG(gammap-one)+addfactor(1))-(LOG(gammam-one)+addfactor(2))
              mp_log_trajectory=mp_log_trajectory+gammap*(LOG((gammap-one)/gammap)+addfactor(3))
              mp_log_trajectory=mp_log_trajectory+gammam*(LOG((gammam-one)/gammam)+addfactor(4))
              RETURN
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
              end"""%{'complex_mp_format':self.mp_complex_format,'real_mp_format':self.mp_real_format})


        #check for the file functions.f
        model_path = self.model.get('modelpath')
        if os.path.exists(pjoin(model_path,'Fortran','functions.f')):
            fsock.write_comment_line(' USER DEFINE FUNCTIONS ')
            input = pjoin(model_path,'Fortran','functions.f')
            fsock.writelines(open(input).read())
            fsock.write_comment_line(' END USER DEFINE FUNCTIONS ')
            
        # check for functions define in the UFO model
        ufo_fct = self.model.get('functions')
        if ufo_fct:
            fsock.write_comment_line(' START UFO DEFINE FUNCTIONS ')
            done = []
            for fct in ufo_fct:
                # already handle by default
                if str(fct.name.lower()) not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc", "condif",
                                    "theta_function", "cond", "reglog", "reglogp", "reglogm", "recms","arg",
                                    "grreglog","regsqrt","B0F","b0f","sqrt_trajectory","log_trajectory"]:

                    ufo_fct_template = """
          double complex function %(name)s(%(args)s)
          implicit none
          double complex %(args)s
          %(definitions)s
          %(name)s = %(fct)s

          return
          end
          """
                    str_fct = self.p_to_f.parse(fct.expr)
                    if not self.p_to_f.to_define:
                        definitions = []
                    else:
                        definitions=[]
                        for d in self.p_to_f.to_define:
                            if d == 'pi':
                                definitions.append(' double precision pi')
                                definitions.append(' data pi /3.1415926535897932d0/')
                            else:
                                definitions.append(' double complex %s' % d)
                                
                    text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", ".join(fct.arguments),                
                                'fct': str_fct,
                                'definitions': '\n'.join(definitions)
                                 }

                    fsock.writelines(text)
            if self.opt['mp']:
                fsock.write_comment_line(' START UFO DEFINE FUNCTIONS FOR MP')
                for fct in ufo_fct:
                    # already handle by default
                    if fct.name not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc","condif",
                                        "theta_function", "cond", "reglog", "reglogp","reglogm", "recms","arg",
                                        "grreglog","regsqrt","B0F","b0f","sqrt_trajectory","log_trajectory"]:

                        ufo_fct_template = """
          %(complex_mp_format)s function mp_%(name)s(mp__%(args)s)
          implicit none
          %(complex_mp_format)s mp__%(args)s
          %(definitions)s
          mp_%(name)s = %(fct)s

          return
          end
          """
                        str_fct = self.mp_p_to_f.parse(fct.expr)
                        if not self.mp_p_to_f.to_define:
                            definitions = []
                        else:
                            definitions=[]
                            for d in self.mp_p_to_f.to_define:
                                if d == 'pi': 
                                    definitions.append(' %s mp__pi' % self.mp_real_format)
                                    definitions.append(' data mp__pi /3.141592653589793238462643383279502884197e+00_16/')
                                else:   
                                    definitions.append(' %s mp_%s' % (self.mp_complex_format,d))
                        text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", mp__".join(fct.arguments),                
                                'fct': str_fct,
                                'definitions': '\n'.join(definitions),
                                'complex_mp_format': self.mp_complex_format
                                 }
                        fsock.writelines(text)


                    
            fsock.write_comment_line(' STOP UFO DEFINE FUNCTIONS ')                    

        

    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o'
        text += ' model_functions.o '
        
        if self.opt['export_format'].startswith('standalone'):
            text += ' alfas_functions.o '

        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // self.nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // self.nb_def_by_file
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // self.nb_def_by_file
        couplings_files=['couplings%s.o' % (i+1) \
                                for i in range(nb_coup_dep + nb_coup_indep) ]
        if self.opt['mp']:
            # this part changed to include also the couplings which do not 
            # depend on the PSP
            couplings_files+=['mp_couplings%s.o' % (i+1) \
                                for i in range(nb_coup_dep + nb_coup_indep) ]
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
        # order them in a smart way
        self.params_ext.sort(key=models.write_param_card.cmp_to_key(models.write_param_card.ParamCardWriter.order_param))

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
        if self.model['running_elements']:
            scales = set()
            
            for elements in self.model["running_elements"]:
                for params in elements.run_objects:
                    for param in params:
                        scales.add(param.lhablock)

            try:
                scales.remove('SMINPUTS')
            except Exception:
                pass
            #entry should be a parameter ... not a string
            for b in scales:
                param = base_objects.ParamCardVariable(
                    'mdl__%s__scale' % b.lower(),
                     91.188, b, [0])
                external_param.append(format(param))

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

            if parameter.lhablock.lower() == 'loop':
                template = template.replace('LHA_get_real', 'LHA_get_real_silent') 
                
            return template        
    
        fsock = self.open('param_read.inc', format='fortran')
        res_strings = [format_line(param) \
                          for param in self.params_ext]
        
        if self.model['running_elements']:
            scales = set()
            
            for elements in self.model["running_elements"]:
                for params in elements.run_objects:
                    for param in params:
                        scales.add(param.lhablock)

            try:
                scales.remove('SMINPUTS')
            except Exception:
                pass
            #entry should be a parameter ... not a string
            for b in scales:
                param = base_objects.ParamCardVariable(
                    'mdl__%s__scale' % b,
                     91.188, b, 0)
                res_strings.append(format_line(param))
        
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
                                 mssm_convert=True, write_special=True):
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
            param_writer.ParamCardWriter(model, output_path, write_special=write_special)
         
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
    
    def create_param_card(self, write_special=True):
        """ create the param_card.dat """

        rule_card = pjoin(self.dir_path, 'param_card_rule.dat')
        if not hasattr(self.model, 'rule_card'):
            rule_card=False
        write_special = True
        if 'exporter' in self.opt:
            import madgraph.loop.loop_exporters as loop_exporters
            import madgraph.iolibs.export_fks as export_fks
            write_special = False
            if  issubclass(self.opt['exporter'], loop_exporters.LoopProcessExporterFortranSA):
                write_special = True
                if issubclass(self.opt['exporter'],(loop_exporters.LoopInducedExporterME,export_fks.ProcessExporterFortranFKS)):
                     write_special = False
                        
        self.create_param_card_static(self.model, 
                                      output_path=pjoin(self.dir_path, 'param_card.dat'), 
                                      rule_card_path=rule_card, 
                                      mssm_convert=True,
                                      write_special=write_special)
        
def ExportV4Factory(cmd, noclean, output_type='default', group_subprocesses=True, cmd_options={}):
    """ Determine which Export_v4 class is required. cmd is the command 
        interface containing all potential usefull information.
        The output_type argument specifies from which context the output
        is called. It is 'madloop' for MadLoop5, 'amcatnlo' for FKS5 output
        and 'default' for tree-level outputs."""

    opt = dict(cmd.options)
    opt['output_options'] = cmd_options

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
      'cluster_local_path': cmd.options['cluster_local_path'],
      'output_options': cmd_options
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
                    raise Exception("output_type not recognize %s" % output_type)
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
        amcatnlo_options['running'] = cmd._curr_model.get('running_elements')
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

    # Then treat the EW sudakov Standalone output     
    elif output_type=='ewsudsa':
        import madgraph.iolibs.export_fks as export_fks
        ExporterClass=None
        amcatnlo_options = dict(opt)
        amcatnlo_options.update(MadLoop_SA_options)
        amcatnlo_options['mp'] = False
        logger.info("Writing out the EW Sudakov approximation in a standalone format")
        ExporterClass = export_fks.ProcessExporterEWSudakovSA
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
               'v5_model': False if cmd._model_v4_path else True,
               'running': cmd._curr_model.get('running_elements'),
                })

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
            elif cmd._export_plugin:
                return cmd._export_plugin(cmd._export_dir,opt) 
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
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                return cmd._export_plugin(cmd._export_dir, loop_induced_opt)
            else:
                return cmd._export_plugin(cmd._export_dir, opt)

        else:
            raise Exception('Wrong export_v4 format')
    else:
        raise MadGraph5Error('Output type %s not reckognized in ExportV4Factory.')
    
            


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
                                         group_number, **opt):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files."""

        if not isinstance(subproc_group, group_subprocs.SubProcessGroup):
            raise base_objects.PhysicsObject.PhysicsObjectError("subproc_group object not SubProcessGroup")

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

        linkfiles = ['driver.f', 'cuts.f', 'initialization.f','gen_ps.f', 'makefile', 'coupl.inc','madweight_param.inc', 'run.inc', 'setscales.f', 'dummy_fct.f']

        for file in linkfiles:
            ln('../%s' % file, cwd=Ppath)

        ln('nexternal.inc', '../../Source', cwd=Ppath, log=False)
        ln('leshouche.inc', '../../Source', cwd=Ppath, log=False)
        ln('maxamps.inc', '../../Source', cwd=Ppath, log=False)
        ln('../../Source/vector.inc', cwd=Ppath, log=False)
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


    
