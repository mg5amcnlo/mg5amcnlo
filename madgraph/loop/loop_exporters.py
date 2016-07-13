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
import fractions
import glob
import logging
import os
import stat
import sys
import re
import shutil
import subprocess
import itertools
import time
import datetime


import aloha

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.various.misc as misc
import madgraph.various.q_polynomial as q_polynomial
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import madgraph.various.progressbar as pbar
import madgraph.various.q_polynomial as q_polynomial
import madgraph.core.color_amp as color_amp
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.check_param_card as check_param_card
from madgraph.loop.loop_base_objects import LoopDiagram
from madgraph.loop.MadLoopBannerStyles import MadLoopBannerStyles

import madgraph.various.banner as banner_mod

pjoin = os.path.join

import aloha.create_aloha as create_aloha
import models.write_param_card as param_writer
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
from madgraph.iolibs.files import cp, ln, mv
pjoin = os.path.join
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.loop_exporter')

#===============================================================================
# LoopExporterFortran
#===============================================================================
class LoopExporterFortran(object):
    """ Class to define general helper functions to the different 
        loop fortran exporters (ME, SA, MEGroup, etc..) which will inherit both 
        from this class AND from the corresponding ProcessExporterFortran(ME,SA,...).
        It plays the same role as ProcessExporterFrotran and simply defines here
        loop-specific helpers functions necessary for all loop exporters.
        Notice that we do not have LoopExporterFortran inheriting from 
        ProcessExporterFortran but give access to arguments like dir_path and
        clean using options. This avoids method resolution object ambiguity"""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madloop', 'mp':True,
                        'loop_dir':'', 'cuttools_dir':'', 
                        'fortran_compiler':'gfortran',
                        'SubProc_prefix': 'P',
                        'output_dependencies': 'external',
                        'compute_color_flows': False,
                        'mode':''}

    include_names    = {'ninja' : 'mninja.mod',
                        'golem' : 'generic_function_1p.mod',
                        'samurai':'msamurai.mod'}

    def __init__(self, mgme_dir="", dir_path = "", opt=None):
        """Initiate the LoopExporterFortran with directory information on where
        to find all the loop-related source files, like CutTools"""


        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)

        self.SubProc_prefix = self.opt['SubProc_prefix']
        self.loop_dir = self.opt['loop_dir']
        self.cuttools_dir = self.opt['cuttools_dir']
        self.fortran_compiler = self.opt['fortran_compiler']
        self.dependencies = self.opt['output_dependencies']
        self.compute_color_flows = self.opt['compute_color_flows']

        super(LoopExporterFortran,self).__init__(mgme_dir, dir_path, self.opt)        


    def link_CutTools(self, targetPath):
        """Link the CutTools source directory inside the target path given
        in argument"""
                
        if self.dependencies=='internal':
            new_CT_path = pjoin(targetPath,'Source','CutTools')
            shutil.copytree(self.cuttools_dir, new_CT_path, symlinks=True)
            
            current = misc.detect_current_compiler(os.path.join(new_CT_path,
                                                                    'makefile'))
            new = 'gfortran' if self.fortran_compiler is None else \
                                                          self.fortran_compiler
            if current != new:
                misc.mod_compilator(new_CT_path, new, current)
            
            # Create the links to the lib folder
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(pjoin(targetPath,'Source','CutTools','includects',file), 
                                                        pjoin(targetPath,'lib'))
            # Make sure it is recompiled at least once. Because for centralized
            # MG5_aMC installations, it might be that compiler differs.
            # Not necessary anymore because I check the compiler version from
            # the log compiler_version.log generated during CT compilation
            # misc.compile(['cleanCT'], cwd = pjoin(targetPath,'Source'))
 
        if self.dependencies=='external':
            if not os.path.exists(os.path.join(self.cuttools_dir,'includects','libcts.a')):
                logger.info('Compiling CutTools. This has to be done only once and'+\
                                  ' can take a couple of minutes.','$MG:color:BLACK')
                current = misc.detect_current_compiler(os.path.join(\
                                                  self.cuttools_dir,'makefile'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                          self.fortran_compiler
                if current != new:
                    misc.mod_compilator(self.cuttools_dir, new, current)
                misc.compile(cwd=self.cuttools_dir, job_specs = False)
                
                if not os.path.exists(os.path.join(self.cuttools_dir,
                                                      'includects','libcts.a')):            
                    raise MadGraph5Error,"CutTools could not be correctly compiled."
    
            # Create the links to the lib folder
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(os.path.join(self.cuttools_dir,'includects',file),
                                    os.path.join(targetPath,'lib'),abspath=True)

        elif self.dependencies=='environment_paths':
            # Here the user chose to define the dependencies path in one of 
            # his environmental paths
            CTlib = misc.which_lib('libcts.a')
            CTmod = misc.which_lib('mpmodule.mod')
            if not CTlib is None and not CTmod is None:
                logger.info('MG5_aMC is using CutTools installation found at %s.'%\
                                                         os.path.dirname(CTlib)) 
                ln(os.path.join(CTlib),os.path.join(targetPath,'lib'),abspath=True)
                ln(os.path.join(CTmod),os.path.join(targetPath,'lib'),abspath=True)
            else:
                raise InvalidCmd("Could not find the location of the files"+\
                    " libcts.a and mp_module.mod in you environment paths.")
    
    def get_aloha_model(self, model):
        """ Caches the aloha model created here as an attribute of the loop 
        exporter so that it can later be used in the LoopHelasMatrixElement
        in the function compute_all_analytic_information for recycling aloha 
        computations across different LoopHelasMatrixElements steered by the
        same loop exporter.
        """
        if not hasattr(self, 'aloha_model'):
            self.aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
        return self.aloha_model

    #===========================================================================
    # write the multiple-precision header files
    #===========================================================================
    def write_mp_files(self, writer_mprec, writer_mpc):
        """Write the cts_mprec.h and cts_mpc.h"""

        file = open(os.path.join(self.cuttools_dir, 'src/cts/cts_mprec.h')).read()
        writer_mprec.writelines(file)

        file = open(os.path.join(self.cuttools_dir, 'src/cts/cts_mpc.h')).read()
        file = file.replace('&','')
        writer_mpc.writelines(file)

        return True
        
#===============================================================================
# LoopProcessExporterFortranSA
#===============================================================================
class LoopProcessExporterFortranSA(LoopExporterFortran,
                                   export_v4.ProcessExporterFortranSA):
                                   
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format."""
       
    template_dir=os.path.join(_file_path,'iolibs/template_files/loop')
    madloop_makefile_name = 'makefile'

    MadLoop_banner = MadLoopBannerStyles.get_MadLoop_Banner(
               style='classic2', color='green', 
               top_frame_char = '=', bottom_frame_char = '=',
               left_frame_char = '{',right_frame_char = '}',
               print_frame=True, side_margin = 7, up_margin = 1)

    def copy_v4template(self, modelname = ''):
        """Additional actions needed to setup the Template.
        """
        super(LoopProcessExporterFortranSA, self).copy_v4template(modelname)

        self.loop_additional_template_setup()
    
    def loop_additional_template_setup(self, copy_Source_makefile = True):
        """ Perform additional actions specific for this class when setting
        up the template with the copy_v4template function."""

        # We must change some files to their version for NLO computations
        cpfiles= ["Cards/MadLoopParams.dat",
                  "SubProcesses/MadLoopParamReader.f",
                  "SubProcesses/MadLoopParams.inc"]
        if copy_Source_makefile:
            cpfiles.append("Source/makefile")
        
        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))
        
        # Also put a copy of MadLoopParams.dat into MadLoopParams_default.dat
        shutil.copy(pjoin(self.dir_path, 'Cards','MadLoopParams.dat'),
                      pjoin(self.dir_path, 'Cards','MadLoopParams_default.dat'))

        self.MadLoopparam = banner_mod.MadLoopParam(pjoin(self.loop_dir,'StandAlone',
                                                  'Cards', 'MadLoopParams.dat'))
        # write the output file
        self.MadLoopparam.write(pjoin(self.dir_path,"SubProcesses",
                                                           "MadLoopParams.dat"))

        # We might need to give a different name to the MadLoop makefile\
        shutil.copy(pjoin(self.loop_dir,'StandAlone','SubProcesses','makefile'),
                pjoin(self.dir_path, 'SubProcesses',self.madloop_makefile_name))

        # Write SubProcesses/MadLoop_makefile_definitions with dummy variables
        # for the non-optimized output
        link_tir_libs=[]
        tir_libs=[]

        filePath = pjoin(self.dir_path, 'SubProcesses',
                                                 'MadLoop_makefile_definitions')
        calls = self.write_loop_makefile_definitions(
                        writers.MakefileWriter(filePath),link_tir_libs,tir_libs)

            
        # We need minimal editing of MadLoopCommons.f
        MadLoopCommon = open(os.path.join(self.loop_dir,'StandAlone', 
                                    "SubProcesses","MadLoopCommons.inc")).read()
        writer = writers.FortranWriter(os.path.join(self.dir_path, 
                                             "SubProcesses","MadLoopCommons.f"))
        writer.writelines(MadLoopCommon%{
                                   'print_banner_commands':self.MadLoop_banner})
        writer.close()
        
        
        # Copy the whole MadLoop5_resources directory (empty at this stage)
        if not os.path.exists(pjoin(self.dir_path,'SubProcesses',
                                                        'MadLoop5_resources')):
            cp(pjoin(self.loop_dir,'StandAlone','SubProcesses',
                    'MadLoop5_resources'),pjoin(self.dir_path,'SubProcesses'))

        # Link relevant cards from Cards inside the MadLoop5_resources
        ln(pjoin(self.dir_path,'SubProcesses','MadLoopParams.dat'), 
                      pjoin(self.dir_path,'SubProcesses','MadLoop5_resources'))
        ln(pjoin(self.dir_path,'Cards','param_card.dat'),
                      pjoin(self.dir_path,'SubProcesses','MadLoop5_resources'))
        ln(pjoin(self.dir_path,'Cards','ident_card.dat'), 
                      pjoin(self.dir_path,'SubProcesses','MadLoop5_resources'))

        # And remove check_sa in the SubProcess folder since now there is a
        # check_sa tailored to each subprocess.
        if os.path.isfile(pjoin(self.dir_path,'SubProcesses','check_sa.f')):
            os.remove(pjoin(self.dir_path,'SubProcesses','check_sa.f'))

        cwd = os.getcwd()
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0
                     
        # Write the cts_mpc.h and cts_mprec.h files imported from CutTools
        self.write_mp_files(writers.FortranWriter('cts_mprec.h'),\
                                            writers.FortranWriter('cts_mpc.h'))

        # Return to original PWD
        os.chdir(cwd)

        # We must link the CutTools to the Library folder of the active Template
        super(LoopProcessExporterFortranSA, self).link_CutTools(self.dir_path)

    # This function is placed here and not in optimized exporterd,
    # because the same makefile.inc should be used in all cases.
    def write_loop_makefile_definitions(self, writer, link_tir_libs,
                                                       tir_libs,tir_include=[]):
        """ Create the file makefile which links to the TIR libraries."""
            
        file = open(os.path.join(self.loop_dir,'StandAlone',
                      'SubProcesses','MadLoop_makefile_definitions.inc')).read()  
        replace_dict={}
        replace_dict['link_tir_libs']=' '.join(link_tir_libs)
        replace_dict['tir_libs']=' '.join(tir_libs)
        replace_dict['dotf']='%.f'
        replace_dict['prefix']= self.SubProc_prefix
        replace_dict['doto']='%.o'
        replace_dict['tir_include']=' '.join(tir_include)
        file=file%replace_dict
        if writer:
            writer.writelines(file)
        else:
            return file
        
    def convert_model_to_mg4(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
        """ Caches the aloha model created here when writing out the aloha 
        fortran subroutine.
        """
        self.get_aloha_model(model)
        super(LoopProcessExporterFortranSA, self).convert_model_to_mg4(model,
           wanted_lorentz = wanted_lorentz, wanted_couplings = wanted_couplings)

    def get_ME_identifier(self, matrix_element, 
                                 group_number = None, group_elem_number = None):
        """ A function returning a string uniquely identifying the matrix 
        element given in argument so that it can be used as a prefix to all
        MadLoop5 subroutines and common blocks related to it. This allows
        to compile several processes into one library as requested by the 
        BLHA (Binoth LesHouches Accord) guidelines.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent."""

        # When disabling the loop grouping in the LoopInduced MadEvent output,
        # we have only the group_number set and the proc_id set to None. In this
        # case we don't print the proc_id.
        if (not group_number is None) and group_elem_number is None:
            return 'ML5_%d_%s_'%(matrix_element.get('processes')[0].get('id'),
                                                                   group_number)            
        elif group_number is None or group_elem_number is None:
            return 'ML5_%d_'%matrix_element.get('processes')[0].get('id') 
        else:
            return 'ML5_%d_%s_%s_'%(matrix_element.get('processes')[0].get('id'),
                                                group_number, group_elem_number)

    def get_SubProc_folder_name(self, process, 
                                 group_number = None, group_elem_number = None):
        """Returns the name of the SubProcess directory, which can contain
        the process goup and group element number for the case of loop-induced
        integration with MadEvent."""
        
        # When disabling the loop grouping in the LoopInduced MadEvent output,
        # we have only the group_number set and the proc_id set to None. In this
        # case we don't print the proc_id.
        if not group_number is None and group_elem_number is None:
            return "%s%d_%s_%s"%(self.SubProc_prefix, process.get('id'), 
                              group_number,process.shell_string(print_id=False))
        elif group_number is None or group_elem_number is None:
            return "%s%s" %(self.SubProc_prefix,process.shell_string())
        else:
            return "%s%d_%s_%s_%s"%(self.SubProc_prefix, process.get('id'), 
           group_number, group_elem_number,process.shell_string(print_id=False))

    #===========================================================================
    # Set the compiler to be gfortran for the loop processes.
    #===========================================================================
    def compiler_choice(self, compiler=export_v4.default_compiler):
        """ Different daughter classes might want different compilers.
        Here, the gfortran compiler is used throughout the compilation 
        (mandatory for CutTools written in f90) """
        if isinstance(compiler, str):
            fortran_compiler = compiler
            compiler = export_v4.default_compiler
            compiler['fortran'] = fortran_compiler
        
        if not compiler['fortran'] is None and not \
                       any([name in compiler['fortran'] for name in \
                                                         ['gfortran','ifort']]):
            logger.info('For loop processes, the compiler must be fortran90'+\
                        'compatible, like gfortran.')
            compiler['fortran'] = 'gfortran'
            self.set_compiler(compiler,True)
        else:
            self.set_compiler(compiler)
        
        self.set_cpp_compiler(compiler['cpp'])
    
    def turn_to_mp_calls(self, helas_calls_list):
        # Prepend 'MP_' to all the helas calls in helas_calls_list.
        # Might look like a brutal unsafe implementation, but it is not as 
        # these calls are built from the properties of the HELAS objects and
        # whether they are evaluated in double or quad precision is none of 
        # their business but only relevant to the output algorithm.
        # Also the cast to complex masses DCMPLX(*) must be replaced by
        # CMPLX(*,KIND=16)
        MP=re.compile(r"(?P<toSub>^.*CALL\s+)",re.IGNORECASE | re.MULTILINE)
        
        def replaceWith(match_obj):
            return match_obj.group('toSub')+'MP_'

        DCMPLX=re.compile(r"DCMPLX\((?P<toSub>([^\)]*))\)",\
                                                   re.IGNORECASE | re.MULTILINE)
        
        for i, helas_call in enumerate(helas_calls_list):
            new_helas_call=MP.sub(replaceWith,helas_call)
            helas_calls_list[i]=DCMPLX.sub(r"CMPLX(\g<toSub>,KIND=16)",\
                                                                 new_helas_call)

    def make_source_links(self):
        """ In the loop output, we don't need the files from the Source folder """
        pass

    def make_model_symbolic_link(self):
        """ Add the linking of the additional model files for multiple precision
        """
        super(LoopProcessExporterFortranSA, self).make_model_symbolic_link()
        model_path = self.dir_path + '/Source/MODEL/'
        ln(model_path + '/mp_coupl.inc', self.dir_path + '/SubProcesses')
        ln(model_path + '/mp_coupl_same_name.inc', self.dir_path + '/SubProcesses')
    
    def make(self):
        """ Compiles the additional dependences for loop (such as CutTools)."""
        super(LoopProcessExporterFortranSA, self).make()
        
        # make CutTools (only necessary with MG option output_dependencies='internal')
        libdir = os.path.join(self.dir_path,'lib')
        sourcedir = os.path.join(self.dir_path,'Source')
        if self.dependencies=='internal':
            if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
                if os.path.exists(pjoin(sourcedir,'CutTools')):
                    logger.info('Compiling CutTools (can take a couple of minutes) ...')
                    misc.compile(['CutTools'], cwd = sourcedir)
                    logger.info('          ...done.')
                else:
                    raise MadGraph5Error('Could not compile CutTools because its'+\
                   ' source directory could not be found in the SOURCE folder.')
        if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
            raise MadGraph5Error('CutTools compilation failed.')
        
        # Verify compatibility between current compiler and the one which was
        # used when last compiling CutTools (if specified).
        compiler_log_path = pjoin(os.path.dirname((os.path.realpath(pjoin(
                                  libdir, 'libcts.a')))),'compiler_version.log')
        if os.path.exists(compiler_log_path):
            compiler_version_used = open(compiler_log_path,'r').read()
            if not str(misc.get_gfortran_version(misc.detect_current_compiler(\
                       pjoin(sourcedir,'make_opts')))) in compiler_version_used:
                if os.path.exists(pjoin(sourcedir,'CutTools')):
                    logger.info('CutTools was compiled with a different fortran'+\
                                            ' compiler. Re-compiling it now...')
                    misc.compile(['cleanCT'], cwd = sourcedir)
                    misc.compile(['CutTools'], cwd = sourcedir)
                    logger.info('          ...done.')
                else:
                    raise MadGraph5Error("CutTools installation in %s"\
                                 %os.path.realpath(pjoin(libdir, 'libcts.a'))+\
                 " seems to have been compiled with a different compiler than"+\
                    " the one specified in MG5_aMC. Please recompile CutTools.")
    
    def cat_coeff(self, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Concatenate the coefficient information to reduce it to 
        (fraction, is_imaginary) """

        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

        return (total_coeff, is_imaginary)       

    def get_amp_to_jamp_map(self, col_amps, n_amps):
        """ Returns a list with element 'i' being a list of tuples corresponding
        to all apparition of amplitude number 'i' in the jamp number 'j'
        with coeff 'coeff_j'. The format of each tuple describing an apparition 
        is (j, coeff_j). where coeff_j is of the form (Fraction, is_imag)."""

        if(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error, "Incorrect col_amps argument passed to get_amp_to_jamp_map"
        else:
            raise MadGraph5Error, "Incorrect col_amps argument passed to get_amp_to_jamp_map"
        
        # To store the result
        res_list = [[] for i in range(n_amps)]
        for i, coeff_list in enumerate(color_amplitudes):
                for (coefficient, amp_number) in coeff_list:
                    res_list[amp_number-1].append((i,self.cat_coeff(\
                      coefficient[0],coefficient[1],coefficient[2],coefficient[3])))

        return res_list

    def get_color_matrix(self, matrix_element):
        """Return the color matrix definition lines. This color matrix is of size
        NLOOPAMPSxNBORNAMPS and allows for squaring individually each Loop and Born
        amplitude."""

        logger.info('Computing diagram color coefficients')

        # The two lists have a list of tuples at element 'i' which correspond
        # to all apparitions of loop amplitude number 'i' in the jampl number 'j'
        # with coeff 'coeffj'. The format of each tuple describing an apparition 
        # is (j, coeffj).
        ampl_to_jampl=self.get_amp_to_jamp_map(\
          matrix_element.get_loop_color_amplitudes(),
          matrix_element.get_number_of_loop_amplitudes())
        if matrix_element.get('processes')[0].get('has_born'):
            ampb_to_jampb=self.get_amp_to_jamp_map(\
          matrix_element.get_born_color_amplitudes(),
          matrix_element.get_number_of_born_amplitudes())
        else:
            ampb_to_jampb=ampl_to_jampl
        # Below is the original color matrix multiplying the JAMPS
        if matrix_element.get('color_matrix'):
            ColorMatrixDenom = \
              matrix_element.get('color_matrix').get_line_denominators()
            ColorMatrixNum = [ matrix_element.get('color_matrix').\
                               get_line_numerators(index, denominator) for
                               (index, denominator) in enumerate(ColorMatrixDenom) ]
        else:
            ColorMatrixDenom= [1]
            ColorMatrixNum = [[1]]
            
        # Below is the final color matrix output
        ColorMatrixNumOutput=[]
        ColorMatrixDenomOutput=[]
        
        # Now we construct the color factors between each born and loop amplitude
        # by scanning their contributions to the different jamps.
        start = time.time()
        progress_bar = None
        time_info = False
        for i, jampl_list in enumerate(ampl_to_jampl):
            # This can be pretty long for processes with many color flows.
            # So, if necessary (i.e. for more than 15s), we tell the user the
            # estimated time for the processing.
            if i==5:
                elapsed_time = time.time()-start
                t = len(ampl_to_jampl)*(elapsed_time/5.0)
                if t > 10.0:
                    time_info = True
                    logger.info('The color factors computation will take '+\
                      ' about %s to run. '%str(datetime.timedelta(seconds=int(t)))+\
                      'Started on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))
                    if logger.getEffectiveLevel()<logging.WARNING:
                        widgets = ['Color computation:', pbar.Percentage(), ' ', 
                                                pbar.Bar(),' ', pbar.ETA(), ' ']
                        progress_bar = pbar.ProgressBar(widgets=widgets, 
                                       maxval=len(ampl_to_jampl), fd=sys.stdout)
            
            if not progress_bar is None:
                progress_bar.update(i+1)
                # Flush to force the printout of the progress_bar to be updated
                sys.stdout.flush()

            line_num=[]
            line_denom=[]

            # Treat the special case where this specific amplitude contributes to no
            # color flow at all. So it is zero because of color but not even due to
            # an accidental cancellation among color flows, but simply because of its
            # projection to each individual color flow is zero. In such case, the 
            # corresponding jampl_list is empty and all color coefficients must then
            # be zero. This happens for example in the Higgs Effective Theory model
            # for the bubble made of a 4-gluon vertex and the effective ggH vertex.
            if len(jampl_list)==0:
                line_num=[0]*len(ampb_to_jampb)
                line_denom=[1]*len(ampb_to_jampb)
                ColorMatrixNumOutput.append(line_num)
                ColorMatrixDenomOutput.append(line_denom)
                continue

            for jampb_list in ampb_to_jampb:
                real_num=0
                imag_num=0
                common_denom=color_amp.ColorMatrix.lcmm(*[abs(ColorMatrixDenom[jampl]*
                    ampl_coeff[0].denominator*ampb_coeff[0].denominator) for 
                    ((jampl, ampl_coeff),(jampb,ampb_coeff)) in 
                    itertools.product(jampl_list,jampb_list)])
                for ((jampl, ampl_coeff),(jampb, ampb_coeff)) in \
                                       itertools.product(jampl_list,jampb_list):
                    # take the numerator and multiply by lcm/denominator
                    # as we will later divide by the lcm.
                    buff_num=ampl_coeff[0].numerator*\
                        ampb_coeff[0].numerator*ColorMatrixNum[jampl][jampb]*\
                        abs(common_denom)/(ampl_coeff[0].denominator*\
                        ampb_coeff[0].denominator*ColorMatrixDenom[jampl])
                    # Remember that we must take the complex conjugate of
                    # the born jamp color coefficient because we will compute
                    # the square with 2 Re(LoopAmp x BornAmp*)
                    if ampl_coeff[1] and ampb_coeff[1]:
                        real_num=real_num+buff_num
                    elif not ampl_coeff[1] and not ampb_coeff[1]:
                        real_num=real_num+buff_num
                    elif not ampl_coeff[1] and ampb_coeff[1]:
                        imag_num=imag_num-buff_num
                    else:
                        imag_num=imag_num+buff_num
                assert not (real_num!=0 and imag_num!=0), "MadGraph5_aMC@NLO found a "+\
                  "color matrix element which has both a real and imaginary part."
                if imag_num!=0:
                    res=fractions.Fraction(imag_num,common_denom)
                    line_num.append(res.numerator)
                    # Negative denominator means imaginary color coef of the
                    # final color matrix
                    line_denom.append(res.denominator*-1)
                else:
                    res=fractions.Fraction(real_num,common_denom)
                    line_num.append(res.numerator)
                    # Positive denominator means real color coef of the final color matrix
                    line_denom.append(res.denominator)

            ColorMatrixNumOutput.append(line_num)
            ColorMatrixDenomOutput.append(line_denom)

        if time_info:
            logger.info('Finished on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))            
        if progress_bar!=None:
            progress_bar.finish()

        return (ColorMatrixNumOutput,ColorMatrixDenomOutput)

    def get_context(self,matrix_element):
        """ Returns the contextual variables which need to be set when
        pre-processing the template files."""

        # The nSquaredSO entry of the general replace dictionary should have
        # been set in write_loopmatrix prior to the first call to this function
        # However, for cases where the TIRCaching contextual variable is 
        # irrelevant (like in the default output), this might not be the case
        # so we set it to 1. 
        try:
            n_squared_split_orders = matrix_element.rep_dict['nSquaredSO']
        except (KeyError, AttributeError):
            n_squared_split_orders = 1

        LoopInduced = not matrix_element.get('processes')[0].get('has_born')
        
        # Force the computation of loop color flows for loop_induced processes
        ComputeColorFlows = self.compute_color_flows or LoopInduced
        # The variable AmplitudeReduction is just to make the contextual
        # conditions more readable in the include files.
        AmplitudeReduction = LoopInduced or ComputeColorFlows
        # Even when not reducing at the amplitude level, the TIR caching
        # is useful when there is more than one squared split order config.
        TIRCaching = AmplitudeReduction or n_squared_split_orders>1
        MadEventOutput = False

        return {'LoopInduced': LoopInduced,
                'ComputeColorFlows': ComputeColorFlows,
                'AmplitudeReduction': AmplitudeReduction,
                'TIRCaching': TIRCaching,
                'MadEventOutput': MadEventOutput}

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    def generate_loop_subprocess(self, matrix_element, fortran_model,
                          group_number = None, proc_id = None, config_map=None):
        """Generate the Pxxxxx directory for a loop subprocess in MG4 standalone,
        including the necessary loop_matrix.f, born_matrix.f and include files.
        Notice that this is too different from generate_subprocess_directory_v4
        so that there is no point reusing this mother function.
        The 'group_number' and 'proc_id' options are only used for the LoopInduced
        MadEvent output and only to specify the ME_identifier and the P* 
        SubProcess directory name."""

        cwd = os.getcwd()
        proc_dir_name = self.get_SubProc_folder_name(
                        matrix_element.get('processes')[0],group_number,proc_id)
        dirpath = os.path.join(self.dir_path, 'SubProcesses', proc_dir_name)
        
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0

        logger.info('Creating files in directory %s' % dirpath)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        calls=self.write_loop_matrix_element_v4(None,matrix_element,
                         fortran_model, group_number = group_number, 
                                     proc_id = proc_id, config_map = config_map)

        # We assume here that all processes must share the same property of 
        # having a born or not, which must be true anyway since these are two
        # definite different classes of processes which can never be treated on
        # the same footing.
        if matrix_element.get('processes')[0].get('has_born'):
            filename = 'born_matrix.f'
            calls = self.write_bornmatrix(
                writers.FortranWriter(filename),
                matrix_element,
                fortran_model)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        # Do not draw the loop diagrams if they are too many.
        # The user can always decide to do it manually, if really needed
        loop_diags = [loop_diag for loop_diag in\
             matrix_element.get('base_amplitude').get('loop_diagrams')\
             if isinstance(loop_diag,LoopDiagram) and loop_diag.get('type') > 0]
        if len(loop_diags)>5000:
            logger.info("There are more than 5000 loop diagrams."+\
                                              "Only the first 5000 are drawn.")
        filename = "loop_matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(
            loop_diags[:5000]),filename,
            model=matrix_element.get('processes')[0].get('model'),amplitude='')
        logger.info("Drawing loop Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        if matrix_element.get('processes')[0].get('has_born'):   
            filename = "born_matrix.ps"
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('born_diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude='')
            logger.info("Generating born Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string(\
                                                          print_weighted=False))
            plot.draw()

        self.link_files_from_Subprocesses(self.get_SubProc_folder_name(
                       matrix_element.get('processes')[0],group_number,proc_id))
        
        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    def link_files_from_Subprocesses(self,proc_name):
        """ To link required files from the Subprocesses directory to the
        different P* ones"""
        
        linkfiles = ['coupl.inc',
                     'cts_mprec.h', 'cts_mpc.h', 'mp_coupl.inc', 
                     'mp_coupl_same_name.inc',
                     'MadLoopParamReader.f','MadLoopCommons.f',
                     'MadLoopParams.inc']
        
        for file in linkfiles:
            ln('../%s' % file)
        
        ln('../%s'%self.madloop_makefile_name, name='makefile')

        # The mp module
        ln('../../lib/mpmodule.mod')
            
        # Also like the whole MadLoop5_files directory
        ln('../MadLoop5_resources')

    def generate_general_replace_dict(self,matrix_element,
                                           group_number = None, proc_id = None):
        """Generates the entries for the general replacement dictionary used
        for the different output codes for this exporter.The arguments 
        group_number and proc_id are just for the LoopInduced output with MadEvent."""
        
        dict={}
        # A general process prefix which appears in front of all MadLooop
        # subroutines and common block so that several processes can be compiled
        # together into one library, as necessary to follow BLHA guidelines.
        
        dict['proc_prefix'] = self.get_ME_identifier(matrix_element,
                       group_number = group_number, group_elem_number = proc_id)

        # The proc_id is used for MadEvent grouping, so none of our concern here
        # and it is simply set to an empty string.        
        dict['proc_id'] = ''
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        dict['info_lines'] = info_lines
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        dict['process_lines'] = process_lines
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        dict['nexternal'] = nexternal
        dict['nincoming'] = ninitial
        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        dict['ncomb'] = ncomb
        # Extract nloopamps
        nloopamps = matrix_element.get_number_of_loop_amplitudes()
        dict['nloopamps'] = nloopamps
        # Extract nloopdiags
        nloopdiags = len(matrix_element.get('diagrams'))
        dict['nloopdiags'] = nloopdiags
        # Extract nctamps
        nctamps = matrix_element.get_number_of_CT_amplitudes()
        dict['nctamps'] = nctamps
        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_external_wavefunctions()
        dict['nwavefuncs'] = nwavefuncs
        # Set format of the double precision
        dict['real_dp_format']='real*8'
        dict['real_mp_format']='real*16'
        # Set format of the complex
        dict['complex_dp_format']='complex*16'
        dict['complex_mp_format']='complex*32'
        # Set format of the masses
        dict['mass_dp_format'] = dict['complex_dp_format']
        dict['mass_mp_format'] = dict['complex_mp_format']
        # Fill in default values for the placeholders for the madevent 
        # loop-induced output
        dict['nmultichannels'] = 0
        dict['nmultichannel_configs'] = 0
        dict['config_map_definition'] = ''
        dict['config_index_map_definition'] = ''        
        # Color matrix size
        # For loop induced processes it is NLOOPAMPSxNLOOPAMPS and otherwise
        # it is NLOOPAMPSxNBORNAMPS
        # Also, how to access the number of Born squared order contributions

        if matrix_element.get('processes')[0].get('has_born'):
            dict['color_matrix_size'] = 'nbornamps'
            dict['get_nsqso_born']=\
                           "include 'nsqso_born.inc'"
        else:
            dict['get_nsqso_born']="""INTEGER NSQSO_BORN
            PARAMETER (NSQSO_BORN=0)
            """
            dict['color_matrix_size'] = 'nloopamps'    
        
        # These placeholders help to have as many common templates for the
        # output of the loop induced processes and those with a born 
        # contribution.
        if matrix_element.get('processes')[0].get('has_born'):
            # Extract nbornamps
            nbornamps = matrix_element.get_number_of_born_amplitudes()
            dict['nbornamps'] = nbornamps
            dict['ncomb_helas_objs'] = ',ncomb'
            dict['nbornamps_decl'] = \
              """INTEGER NBORNAMPS
                 PARAMETER (NBORNAMPS=%d)"""%nbornamps
            dict['nBornAmps'] = nbornamps
                 
        else:
            dict['ncomb_helas_objs'] = ''  
            dict['dp_born_amps_decl'] = ''
            dict['dp_born_amps_decl_in_mp'] = ''
            dict['copy_mp_to_dp_born_amps'] = ''
            dict['mp_born_amps_decl'] = ''
            dict['nbornamps_decl'] = ''
            dict['nbornamps'] = 0
            dict['nBornAmps'] = 0
        
        return dict
    
    def write_loop_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        group_number = None, proc_id = None, config_map = None):
        """ Writes loop_matrix.f, CT_interface.f, loop_num.f and
        mp_born_amps_and_wfs.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent and only used in get_ME_identifier.
        """
        
        # Create the necessary files for the loop matrix element subroutine
        
        if config_map:
            raise MadGraph5Error, 'The default loop output cannot be used with'+\
              'MadEvent and cannot compute the AMP2 for multi-channeling.'

        if not isinstance(fortran_model,\
          helas_call_writers.FortranUFOHelasCallWriter):
            raise MadGraph5Error, 'The loop fortran output can only'+\
              ' work with a UFO Fortran model'
        
        LoopFortranModel = helas_call_writers.FortranUFOHelasCallWriter(
                     argument=fortran_model.get('model'),
                     hel_sum=matrix_element.get('processes')[0].get('has_born'))

        # Compute the analytical information of the loop wavefunctions in the
        # loop helas matrix elements using the cached aloha model to reuse
        # as much as possible the aloha computations already performed for
        # writing out the aloha fortran subroutines.
        matrix_element.compute_all_analytic_information(
          self.get_aloha_model(matrix_element.get('processes')[0].get('model')))

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        matrix_element.rep_dict = self.generate_general_replace_dict(
                 matrix_element, group_number = group_number, proc_id = proc_id)                                 
        
        # Extract max number of loop couplings (specific to this output type)
        matrix_element.rep_dict['maxlcouplings']= \
                                         matrix_element.find_max_loop_coupling()
        # The born amp declaration suited for also outputing the loop-induced
        # processes as well.
        if matrix_element.get('processes')[0].get('has_born'):
            matrix_element.rep_dict['dp_born_amps_decl_in_mp'] = \
                  matrix_element.rep_dict['complex_dp_format']+" DPAMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sAMPS/DPAMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['dp_born_amps_decl'] = \
                  matrix_element.rep_dict['complex_dp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sAMPS/AMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['mp_born_amps_decl'] = \
                  matrix_element.rep_dict['complex_mp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sMP_AMPS/AMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['copy_mp_to_dp_born_amps'] = \
                   '\n'.join(['DO I=1,NBORNAMPS','DPAMP(I,H)=AMP(I,H)','ENDDO'])
        
        if writer:
            raise MadGraph5Error, 'Matrix output mode no longer supported.'
        
        filename = 'loop_matrix.f'
        calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                      matrix_element,
                                      LoopFortranModel)       

        # Write out the proc_prefix in a file, this is quite handy
        proc_prefix_writer = writers.FortranWriter('proc_prefix.txt','w')
        proc_prefix_writer.write(matrix_element.rep_dict['proc_prefix'])
        proc_prefix_writer.close()
                    
        filename = 'check_sa.f'
        self.write_check_sa(writers.FortranWriter(filename),matrix_element)
        
        filename = 'CT_interface.f'
        self.write_CT_interface(writers.FortranWriter(filename),\
                                matrix_element)
        
        
        
        filename = 'improve_ps.f'
        calls = self.write_improve_ps(writers.FortranWriter(filename),
                                                             matrix_element)
        
        filename = 'loop_num.f'
        self.write_loop_num(writers.FortranWriter(filename),\
                                matrix_element,LoopFortranModel)
        
        filename = 'mp_born_amps_and_wfs.f'
        self.write_born_amps_and_wfs(writers.FortranWriter(filename),\
                                     matrix_element,LoopFortranModel)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                                                            nexternal, ninitial)

        filename = 'process_info.inc'        
        self.write_process_info_file(writers.FortranWriter(filename),
                                                                 matrix_element)
        return calls

    def write_process_info_file(self, writer, matrix_element):
        """A small structural function to write the include file specifying some
        process characteristics."""

        model = matrix_element.get('processes')[0].get('model')
        process_info = {}
        # The maximum spin of any particle connected (or directly running in) 
        # any loop of this matrix element. This is important because there is
        # some limitation in the stability tests that can be performed when this
        # maximum spin is above 3 (vectors). Also CutTools has limitations in 
        # this regard.
        process_info['max_spin_connected_to_loop']=\
                                 matrix_element.get_max_spin_connected_to_loop()
        
        process_info['max_spin_external_particle']= max(
                model.get_particle(l.get('id')).get('spin') for l in 
                                 matrix_element.get('processes')[0].get('legs'))

        proc_include = \
"""
INTEGER MAX_SPIN_CONNECTED_TO_LOOP
PARAMETER(MAX_SPIN_CONNECTED_TO_LOOP=%(max_spin_connected_to_loop)d)
INTEGER MAX_SPIN_EXTERNAL_PARTICLE
PARAMETER(MAX_SPIN_EXTERNAL_PARTICLE=%(max_spin_external_particle)d)
"""%process_info

        writer.writelines(proc_include)
                                
    def generate_subprocess_directory_v4(self, matrix_element, fortran_model):
        """ To overload the default name for this function such that the correct
        function is used when called from the command interface """
        
        return self.generate_loop_subprocess(matrix_element,fortran_model)

    def write_check_sa(self, writer, matrix_element):
        """Writes out the steering code check_sa. In the optimized output mode,
        All the necessary entries in the replace_dictionary have already been 
        set in write_loopmatrix because it is only there that one has access to
        the information about split orders."""        
        replace_dict = copy.copy(matrix_element.rep_dict)     
        for key in ['print_so_born_results','print_so_loop_results',
            'write_so_born_results','write_so_loop_results','set_coupling_target']:
            if key not in replace_dict.keys():
                replace_dict[key]=''
        
        if matrix_element.get('processes')[0].get('has_born'):
            file = open(os.path.join(self.template_dir,'check_sa.inc')).read()
        else:
            file = open(os.path.join(self.template_dir,\
                                          'check_sa_loop_induced.inc')).read()
        file=file%replace_dict
        writer.writelines(file)
         
        # We can always write the f2py wrapper if present (in loop optimized mode, it is)
        if not os.path.isfile(pjoin(self.template_dir,'check_py.f.inc')):
            return
        file = open(os.path.join(self.template_dir,\
                                      'check_py.f.inc')).read()
        file=file%replace_dict
        new_path = writer.name.replace('check_sa.f', 'f2py_wrapper.f')
        new_writer = writer.__class__(new_path, 'w')
        new_writer.writelines(file)

        file = open(os.path.join(self.template_dir,\
                                      'check_sa.py.inc')).read()
        # For now just put in an empty PS point but in the future, maybe generate
        # a valid one already here by default
        curr_proc = matrix_element.get('processes')[0]
        random_PSpoint_python_formatted = \
"""# Specify your chosen PS point below. If you leave it filled with None, then the script will attempt to read it from the file PS.input.
p= [[None,]*4]*%d"""%len(curr_proc.get('legs'))

        process_definition_string = curr_proc.nice_string().replace('Process:','')
        file=file.format(random_PSpoint_python_formatted,process_definition_string)
        new_path = writer.name.replace('check_sa.f', 'check_sa.py')
        new_writer = open(new_path, 'w')
        new_writer.writelines(file)
        # Make it executable
        os.chmod(new_path, os.stat(new_path).st_mode | stat.S_IEXEC)

    def write_improve_ps(self, writer, matrix_element):
        """ Write out the improve_ps subroutines which modify the PS point
        given in input and slightly deform it to achieve exact onshellness on
        all external particles as well as perfect energy-momentum conservation""" 
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        (nexternal,ninitial)=matrix_element.get_nexternal_ninitial()
        replace_dict['ninitial']=ninitial
        mass_list=matrix_element.get_external_masses()[:-2]
        mp_variable_prefix = check_param_card.ParamCard.mp_prefix

        # Write the quadruple precision version of this routine only.
        replace_dict['real_format']=replace_dict['real_mp_format']
        replace_dict['mp_prefix']='MP_'
        replace_dict['exp_letter']='e'
        replace_dict['mp_specifier']='_16'
        replace_dict['coupl_inc_name']='mp_coupl.inc'
        replace_dict['masses_def']='\n'.join(['MASSES(%(i)d)=%(prefix)s%(m)s'\
                            %{'i':i+1,'m':m, 'prefix':mp_variable_prefix} for \
                                                  i, m in enumerate(mass_list)])
        file_mp = open(os.path.join(self.template_dir,'improve_ps.inc')).read()
        file_mp=file_mp%replace_dict
        #
        writer.writelines(file_mp)

    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        file = open(os.path.join(self.template_dir,'loop_num.inc')).read()
        
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        loop_helas_calls=fortran_model.get_loop_amplitude_helas_calls(matrix_element)
        replace_dict['maxlcouplings']=matrix_element.find_max_loop_coupling()
        replace_dict['loop_helas_calls'] = "\n".join(loop_helas_calls) 
       
        # The squaring is only necessary for the processes with born where the 
        # sum over helicities is done before sending the numerator to CT.
        dp_squaring_lines=['DO I=1,NBORNAMPS',
            'CFTOT=DCMPLX(CF_N(AMPLNUM,I)/DBLE(ABS(CF_D(AMPLNUM,I))),0.0d0)',
            'IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1',
            'RES=RES+CFTOT*BUFF*DCONJG(AMP(I,H))','ENDDO']
        mp_squaring_lines=['DO I=1,NBORNAMPS',
'CFTOT=CMPLX(CF_N(AMPLNUM,I)/(1.0E0_16*ABS(CF_D(AMPLNUM,I))),0.0E0_16,KIND=16)',
            'IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1',
            'QPRES=QPRES+CFTOT*BUFF*CONJG(AMP(I,H))','ENDDO']
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['dp_squaring']='\n'.join(dp_squaring_lines)
            replace_dict['mp_squaring']='\n'.join(mp_squaring_lines)
        else:
            replace_dict['dp_squaring']='RES=BUFF'
            replace_dict['mp_squaring']='QPRES=BUFF'                       

        # Prepend MP_ to all helas calls.
        self.turn_to_mp_calls(loop_helas_calls)
        replace_dict['mp_loop_helas_calls'] = "\n".join(loop_helas_calls)
        
        file=file%replace_dict
        
        if writer:
            writer.writelines(file)
        else:
            return file

    def write_CT_interface(self, writer, matrix_element, optimized_output=False):
        """ Create the file CT_interface.f which contains the subroutine defining
         the loop HELAS-like calls along with the general interfacing subroutine.
         It is used to interface against any OPP tool, including Samurai and Ninja."""

        files=[]

        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(matrix_element.rep_dict)
        
        # We finalize CT result differently wether we used the built-in 
        # squaring against the born.
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['finalize_CT']='\n'.join([\
         'RES(%d)=NORMALIZATION*2.0d0*DBLE(RES(%d))'%(i,i) for i in range(1,4)])
        else:
            replace_dict['finalize_CT']='\n'.join([\
                     'RES(%d)=NORMALIZATION*RES(%d)'%(i,i) for i in range(1,4)])
        
        file = open(os.path.join(self.template_dir,'CT_interface.inc')).read()  

        file = file % replace_dict
        files.append(file)
        
        # Now collect the different kind of subroutines needed for the
        # loop HELAS-like calls.
        HelasLoopAmpsCallKeys=matrix_element.get_used_helas_loop_amps()

        for callkey in HelasLoopAmpsCallKeys:
            replace_dict=copy.copy(matrix_element.rep_dict)
            # Add to this dictionary all other attribute common to all
            # HELAS-like loop subroutines.
            if matrix_element.get('processes')[0].get('has_born'):
                replace_dict['validh_or_nothing']=',validh'
            else:
                replace_dict['validh_or_nothing']=''
            # In the optimized output, the number of couplings in the loop is
            # not specified so we only treat it here if necessary:
            if len(callkey)>2:
                replace_dict['ncplsargs']=callkey[2]
                cplsargs="".join(["C%d,MP_C%d, "%(i,i) for i in range(1,callkey[2]+1)])
                replace_dict['cplsargs']=cplsargs
                cplsdecl="".join(["C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
                replace_dict['cplsdecl']=cplsdecl
                mp_cplsdecl="".join(["MP_C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
                replace_dict['mp_cplsdecl']=mp_cplsdecl
                cplset="\n".join(["\n".join(["LC(%d)=C%d"%(i,i),\
                                         "MP_LC(%d)=MP_C%d"%(i,i)])\
                              for i in range(1,callkey[2]+1)])
                replace_dict['cplset']=cplset
            
            replace_dict['nloopline']=callkey[0]
            wfsargs="".join(["W%d, "%i for i in range(1,callkey[1]+1)])
            replace_dict['wfsargs']=wfsargs
            # We don't pass the multiple precision mass in the optimized_output
            if not optimized_output:
                margs="".join(["M%d,MP_M%d, "%(i,i) for i in range(1,callkey[0]+1)])
            else:
                margs="".join(["M%d, "%i for i in range(1,callkey[0]+1)])
            replace_dict['margs']=margs                
            wfsargsdecl="".join([("W%d, "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['wfsargsdecl']=wfsargsdecl
            margsdecl="".join(["M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['margsdecl']=margsdecl
            mp_margsdecl="".join(["MP_M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['mp_margsdecl']=mp_margsdecl
            weset="\n".join([("WE("+str(i)+")=W"+str(i)) for \
                             i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            weset="\n".join([("WE(%d)=W%d"%(i,i)) for i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            msetlines=["M2L(1)=M%d**2"%(callkey[0]),]
            mset="\n".join(msetlines+["M2L(%d)=M%d**2"%(i,i-1) for \
                             i in range(2,callkey[0]+1)])
            replace_dict['mset']=mset            
            mset2lines=["ML(1)=M%d"%(callkey[0]),"ML(2)=M%d"%(callkey[0]),
                  "MP_ML(1)=MP_M%d"%(callkey[0]),"MP_ML(2)=MP_M%d"%(callkey[0])]
            mset2="\n".join(mset2lines+["\n".join(["ML(%d)=M%d"%(i,i-2),
                                               "MP_ML(%d)=MP_M%d"%(i,i-2)]) for \
                                        i in range(3,callkey[0]+3)])
            replace_dict['mset2']=mset2           
            replace_dict['nwfsargs'] = callkey[1]
            if callkey[0]==callkey[1]:
                replace_dict['nwfsargs_header'] = ""
                replace_dict['pairingargs']=""
                replace_dict['pairingdecl']=""
                pairingset="""DO I=1,NLOOPLINE
                                PAIRING(I)=1
                              ENDDO
                           """
                replace_dict['pairingset']=pairingset               
            else:
                replace_dict['nwfsargs_header'] = '_%d'%callkey[1]
                pairingargs="".join([("P"+str(i)+", ") for i in \
                                                         range(1,callkey[0]+1)])
                replace_dict['pairingargs']=pairingargs
                pairingdecl="integer "+"".join([("P"+str(i)+", ") for i in \
                                                    range(1,callkey[0]+1)])[:-2]
                replace_dict['pairingdecl']=pairingdecl
                pairingset="\n".join([("PAIRING("+str(i)+")=P"+str(i)) for \
                             i in range(1,callkey[0]+1)])
                replace_dict['pairingset']=pairingset
            
            file = open(os.path.join(self.template_dir,\
                                             'helas_loop_amplitude.inc')).read()
            file = file % replace_dict
            files.append(file)   
        
        file="\n".join(files)
        
        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    # Helper function to split HELAS CALLS in dedicated subroutines placed
    # in different files.
    def split_HELASCALLS(self, writer, replace_dict, template_name, masterfile, \
                         helas_calls, entry_name, bunch_name,n_helas=2000,
                         required_so_broadcaster = 'LOOP_REQ_SO_DONE',
                         continue_label = 1000, momenta_array_name='P',
                         context={}):
        """ Finish the code generation with splitting.         
        Split the helas calls in the argument helas_calls into bunches of 
        size n_helas and place them in dedicated subroutine with name 
        <bunch_name>_i. Also setup the corresponding calls to these subroutine 
        in the replace_dict dictionary under the entry entry_name.
        The context specified will be forwarded to the the fileWriter."""
        helascalls_replace_dict=copy.copy(replace_dict)
        helascalls_replace_dict['bunch_name']=bunch_name
        helascalls_files=[]
        for i, k in enumerate(range(0, len(helas_calls), n_helas)):
            helascalls_replace_dict['bunch_number']=i+1                
            helascalls_replace_dict['helas_calls']=\
                                           '\n'.join(helas_calls[k:k + n_helas])
            helascalls_replace_dict['required_so_broadcaster']=\
                                                         required_so_broadcaster
            helascalls_replace_dict['continue_label']=continue_label
            new_helascalls_file = open(os.path.join(self.template_dir,\
                                                          template_name)).read()
            new_helascalls_file = new_helascalls_file % helascalls_replace_dict
            helascalls_files.append(new_helascalls_file)
        # Setup the call to these HELASCALLS subroutines in loop_matrix.f
        helascalls_calls = [ "CALL %s%s_%d(%s,NHEL,H,IC)"%\
              (replace_dict['proc_prefix'] ,bunch_name,a+1,momenta_array_name) \
                                          for a in range(len(helascalls_files))]
        replace_dict[entry_name]='\n'.join(helascalls_calls)
        if writer:
            for i, helascalls_file in enumerate(helascalls_files):
                filename = '%s_%d.f'%(bunch_name,i+1)
                writers.FortranWriter(filename).writelines(helascalls_file,
                                                                context=context)
        else:
                masterfile='\n'.join([masterfile,]+helascalls_files)                

        return masterfile

    def write_loopmatrix(self, writer, matrix_element, fortran_model,
                                                                 noSplit=False):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line
        # When the user asks for the polarized matrix element we must 
        # multiply back by the helicity averaging factor
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()

        # These entries are specific for the output for loop-induced processes
        # Also sets here the details of the squaring of the loop ampltiudes
        # with the born or the loop ones.
        if not matrix_element.get('processes')[0].get('has_born'):
            replace_dict['compute_born']=\
"""C There is of course no born for loop induced processes
ANS(0)=0.0d0
"""
            replace_dict['set_reference']='\n'.join([
              'C For loop-induced, the reference for comparison is set later'+\
              ' from the total contribution of the previous PS point considered.',
              'C But you can edit here the value to be used for the first PS point.',
                'if (NPSPOINTS.eq.0) then','ref=1.0d-50','else',
                'ref=nextRef/DBLE(NPSPOINTS)','endif'])
            replace_dict['loop_induced_setup'] = '\n'.join([
              'HELPICKED_BU=HELPICKED','HELPICKED=H','MP_DONE=.FALSE.',
              'IF(SKIPLOOPEVAL) THEN','GOTO 1227','ENDIF'])
            replace_dict['loop_induced_finalize'] = \
            ("""DO I=NCTAMPS+1,NLOOPAMPS
               IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN
                 WRITE(*,*) '##W03 WARNING Contribution ',I
                 WRITE(*,*) ' is unstable for helicity ',H
               ENDIF
C                IF(.NOT.%(proc_prefix)sISZERO(ABS(AMPL(2,I))+ABS(AMPL(3,I)),REF,-1,H)) THEN
C                  WRITE(*,*) '##W04 WARNING Contribution ',I,' for helicity ',H,' has a contribution to the poles.'
C                  WRITE(*,*) 'Finite contribution         = ',AMPL(1,I)
C                  WRITE(*,*) 'single pole contribution    = ',AMPL(2,I)
C                  WRITE(*,*) 'double pole contribution    = ',AMPL(3,I)
C                ENDIF
               ENDDO
               1227 CONTINUE
               HELPICKED=HELPICKED_BU""")%replace_dict
            replace_dict['loop_helas_calls']=""
            replace_dict['nctamps_or_nloopamps']='nloopamps'
            replace_dict['nbornamps_or_nloopamps']='nloopamps'
            replace_dict['squaring']=\
                    """ANS(1)=ANS(1)+DBLE(CFTOT*AMPL(1,I)*DCONJG(AMPL(1,J)))
                       IF (J.EQ.1) THEN
                         ANS(2)=ANS(2)+DBLE(CFTOT*AMPL(2,I))+DIMAG(CFTOT*AMPL(2,I))
                         ANS(3)=ANS(3)+DBLE(CFTOT*AMPL(3,I))+DIMAG(CFTOT*AMPL(3,I))                         
                       ENDIF"""      
        else: 
            replace_dict['compute_born']=\
"""C Compute the born, for a specific helicity if asked so.
call %(proc_prefix)ssmatrixhel(P_USER,USERHEL,ANS(0))
"""%matrix_element.rep_dict
            replace_dict['set_reference']=\
"""C We chose to use the born evaluation for the reference
call %(proc_prefix)ssmatrix(p,ref)"""%matrix_element.rep_dict
            replace_dict['loop_induced_helas_calls'] = ""
            replace_dict['loop_induced_finalize'] = ""
            replace_dict['loop_induced_setup'] = ""
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nbornamps'
            replace_dict['squaring']='\n'.join(['DO K=1,3',
                   'ANS(K)=ANS(K)+2.0d0*DBLE(CFTOT*AMPL(K,I)*DCONJG(AMP(J,H)))',
                                                                       'ENDDO'])

        # Write a dummy nsquaredSO.inc which is used in the default
        # loop_matrix.f code (even though it does not support split orders evals)
        # just to comply with the syntax expected from the external code using MadLoop.
        writers.FortranWriter('nsquaredSO.inc').writelines(
"""INTEGER NSQUAREDSO
PARAMETER (NSQUAREDSO=0)""")

        # Actualize results from the loops computed. Only necessary for
        # processes with a born.
        actualize_ans=[]
        if matrix_element.get('processes')[0].get('has_born'):
            actualize_ans.append("DO I=NCTAMPS+1,NLOOPAMPS")
            actualize_ans.extend("ANS(%d)=ANS(%d)+AMPL(%d,I)"%(i,i,i) for i \
                                                                  in range(1,4)) 
            actualize_ans.append(\
               "IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN")
            actualize_ans.append(\
                   "WRITE(*,*) '##W03 WARNING Contribution ',I,' is unstable.'")
            actualize_ans.extend(["ENDIF","ENDDO"])
            replace_dict['actualize_ans']='\n'.join(actualize_ans)
        else:
            replace_dict['actualize_ans']=\
            ("""C We add five powers to the reference value to loosen a bit the vanishing pole check.
C               IF(.NOT.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)).AND..NOT.%(proc_prefix)sISZERO(ABS(ANS(2))+ABS(ANS(3)),ABS(ANS(1))*(10.0d0**5),-1,H)) THEN
C                 WRITE(*,*) '##W05 WARNING Found a PS point with a contribution to the single pole.'
C                 WRITE(*,*) 'Finite contribution         = ',ANS(1)
C                 WRITE(*,*) 'single pole contribution    = ',ANS(2)
C                 WRITE(*,*) 'double pole contribution    = ',ANS(3)
C               ENDIF""")%replace_dict
        
        # Write out the color matrix
        (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
        CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorNumFactors.dat'%matrix_element.rep_dict),'w')
        for ColorLine in CMNum:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        CMWriter=open(pjoin('..','MadLoop5_resources',
          '%(proc_prefix)sColorDenomFactors.dat'%matrix_element.rep_dict),'w')
        for ColorLine in CMDenom:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        
        # Write out the helicity configurations
        HelConfigs=matrix_element.get_helicity_matrix()
        HelConfigWriter=open(pjoin('..','MadLoop5_resources',
                 '%(proc_prefix)sHelConfigs.dat'%matrix_element.rep_dict),'w')
        for HelConfig in HelConfigs:
            HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
        HelConfigWriter.close()
        
        # Extract helas calls
        loop_amp_helas_calls = fortran_model.get_loop_amp_helas_calls(\
                                                                 matrix_element)
        # The proc_prefix must be replaced
        loop_amp_helas_calls = [lc % matrix_element.rep_dict 
                                                 for lc in loop_amp_helas_calls]
        
        born_ct_helas_calls, UVCT_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element)
        # In the default output, we do not need to separate these two kind of
        # contributions
        born_ct_helas_calls = born_ct_helas_calls + UVCT_helas_calls
        file = open(os.path.join(self.template_dir,\
        
                        'loop_matrix_standalone.inc')).read()
        
        if matrix_element.get('processes')[0].get('has_born'):
            toBeRepaced='loop_helas_calls'
        else:
            toBeRepaced='loop_induced_helas_calls'

        # Decide here wether we need to split the loop_matrix.f file or not.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>1000)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'helas_calls_split.inc',file,born_ct_helas_calls,\
                            'born_ct_helas_calls','helas_calls_ampb')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'helas_calls_split.inc',file,loop_amp_helas_calls,\
                    toBeRepaced,'helas_calls_ampl')
        else:
            replace_dict['born_ct_helas_calls']='\n'.join(born_ct_helas_calls)
            replace_dict[toBeRepaced]='\n'.join(loop_amp_helas_calls)
        
        file = file % replace_dict

        loop_calls_finder = re.compile(r'^\s*CALL\S*LOOP\S*')
        n_loop_calls = len(filter(lambda call: 
               not loop_calls_finder.match(call) is None, loop_amp_helas_calls))
        if writer:
            # Write the file
            writer.writelines(file)  
            return n_loop_calls
        else:
            # Return it to be written along with the others
            return n_loop_calls, file
                  
    def write_bornmatrix(self, writer, matrix_element, fortran_model):
        """Create the born_matrix.f file for the born process as for a standard
        tree-level computation."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")

        # For now, we can use the exact same treatment as for tree-level
        # computations by redefining here a regular HelasMatrixElementf or the
        # born process.
        # It is important to make a deepcopy, as we don't want any possible 
        # treatment on the objects of the bornME to have border effects on
        # the content of the LoopHelasMatrixElement object.
        bornME = helas_objects.HelasMatrixElement()
        for prop in bornME.keys():
            bornME.set(prop,copy.deepcopy(matrix_element.get(prop)))
        bornME.set('base_amplitude',None,force=True)
        bornME.set('diagrams',copy.deepcopy(\
                                            matrix_element.get_born_diagrams()))        
        bornME.set('color_basis',copy.deepcopy(\
                                        matrix_element.get('born_color_basis')))
        bornME.set('color_matrix',copy.deepcopy(\
                              color_amp.ColorMatrix(bornME.get('color_basis'))))
        # This is to decide wether once to reuse old wavefunction to store new
        # ones (provided they are not used further in the code.)
        bornME.optimization = True
        return super(LoopProcessExporterFortranSA,self).write_matrix_element_v4(
                                                  writer, bornME, fortran_model, 
                           proc_prefix=matrix_element.rep_dict['proc_prefix'])

    def write_born_amps_and_wfs(self, writer, matrix_element, fortran_model,
                                                                 noSplit=False): 
        """ Writes out the code for the subroutine MP_BORN_AMPS_AND_WFS which 
        computes just the external wavefunction and born amplitudes in 
        multiple precision. """

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        replace_dict = copy.copy(matrix_element.rep_dict)

        # For the wavefunction copy, check what suffix is needed for the W array
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['h_w_suffix']=',H'
        else:
            replace_dict['h_w_suffix']=''            

        # Extract helas calls
        born_amps_and_wfs_calls , uvct_amp_calls = \
          fortran_model.get_born_ct_helas_calls(matrix_element, include_CT=True)
        # In the default output, these two kind of contributions do not need to
        # be differentiated
        born_amps_and_wfs_calls = born_amps_and_wfs_calls + uvct_amp_calls
        
        # Turn these HELAS calls to the multiple-precision version of the HELAS
        # subroutines.
        self.turn_to_mp_calls(born_amps_and_wfs_calls)

        file = open(os.path.join(self.template_dir,\
                        'mp_born_amps_and_wfs.inc')).read()   
        # Decide here wether we need to split the loop_matrix.f file or not.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>2000)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'mp_helas_calls_split.inc',file,\
                            born_amps_and_wfs_calls,'born_amps_and_wfs_calls',\
                            'mp_helas_calls')
        else:
            replace_dict['born_amps_and_wfs_calls']=\
                                            '\n'.join(born_amps_and_wfs_calls)
        
        file = file % replace_dict
        if writer:
            # Write the file
            writer.writelines(file)  
        else:
            # Return it to be written along with the others
            return file

#===============================================================================
# LoopProcessOptimizedExporterFortranSA
#===============================================================================

class LoopProcessOptimizedExporterFortranSA(LoopProcessExporterFortranSA):
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format which exploits the Pozzorini method of representing
       the loop numerators as polynomial to render its evaluations faster."""

    template_dir=os.path.join(_file_path,'iolibs/template_files/loop_optimized')
    # The option below controls wether one wants to group together in one single
    # CutTools/TIR call the loops with same denominator structure
    forbid_loop_grouping = False
    
    # List of potential TIR library one wants to link to.
    # Golem and Samurai will typically get obtained from gosam_contrib
    # which might also contain a version of ninja. We must therefore
    # make sure that ninja appears first in the list of -L because 
    # it is the tool for which the user is most susceptible of 
    # using a standalone verison independent of gosam_contrib
    all_tir=['pjfry','iregi','ninja','golem','samurai']
    
    def __init__(self, mgme_dir="", dir_path = "", opt=None):
        """Initiate the LoopProcessOptimizedExporterFortranSA with directory 
        information on where to find all the loop-related source files, 
        like CutTools and TIR"""

        super(LoopProcessOptimizedExporterFortranSA,self).__init__(mgme_dir, 
                                                                   dir_path, opt)

        # TIR available ones
        self.tir_available_dict={'pjfry':True,'iregi':True,'golem':True,
                                 'samurai':True,'ninja':True}

        for tir in self.all_tir:
            tir_dir="%s_dir"%tir
            if tir_dir in self.opt and not self.opt[tir_dir] is None:
                # Make sure to defer the 'local path' to the current MG5aMC root.
                tir_path = self.opt[tir_dir].strip()
                if tir_path.startswith('.'):
                    tir_path = os.path.abspath(pjoin(MG5DIR,tir_path))
                setattr(self,tir_dir,tir_path)
            else:
                setattr(self,tir_dir,'')

    def copy_v4template(self, modelname = ''):
        """Additional actions needed to setup the Template. 
        """
        
        super(LoopProcessOptimizedExporterFortranSA, self).copy_v4template(
                                                                      modelname)
        
        self.loop_optimized_additional_template_setup()

    def get_context(self,matrix_element, **opts):
        """ Additional contextual information which needs to be created for
        the optimized output."""
        
        context = LoopProcessExporterFortranSA.get_context(self, matrix_element, 
                                                                         **opts)

        # For now assume Ninja always supports quadruple precision
        try:
            context['ninja_supports_quad_prec'] = \
                     misc.get_ninja_quad_prec_support(getattr(self,'ninja_dir'))
        except AttributeError:
            context['ninja_supports_quad_prec'] = False

        for tir in self.all_tir:
            context['%s_available'%tir]=self.tir_available_dict[tir]
            # safety check
            if tir not in ['golem','pjfry','iregi','samurai','ninja']:
                raise MadGraph5Error,"%s was not a TIR currently interfaced."%tir_name

        return context

    def loop_optimized_additional_template_setup(self):
        """ Perform additional actions specific for this class when setting
        up the template with the copy_v4template function."""
        
        # We must link the TIR to the Library folder of the active Template
        link_tir_libs=[]
        tir_libs=[]
        tir_include=[]
        
        for tir in self.all_tir:
            tir_dir="%s_dir"%tir
            libpath=getattr(self,tir_dir)
            libname="lib%s.a"%tir
            tir_name=tir
            libpath = self.link_TIR(os.path.join(self.dir_path, 'lib'),
                                              libpath,libname,tir_name=tir_name)
            setattr(self,tir_dir,libpath)
            if libpath != "":
                if tir in ['ninja','pjfry','golem','samurai']:
                    # It is cleaner to use the original location of the libraries
                    link_tir_libs.append('-L%s/ -l%s'%(libpath,tir))
                    tir_libs.append('%s/lib%s.$(libext)'%(libpath,tir))
                    if tir in ['ninja','golem', 'samurai']:
                        trgt_path = pjoin(os.path.dirname(libpath),'include')
                        to_include = misc.find_includes_path(trgt_path,
                                                        self.include_names[tir])
                        if to_include is None:
                            logger.error(
'Could not find the include directory for %s, looking in %s.\n' % (tir, str(trgt_path))+
'Generation carries on but you will need to edit the include path by hand in the makefiles.')
                            to_include = '<Not_found_define_it_yourself>'                
                        tir_include.append('-I %s'%str(to_include))
                        # To be able to easily compile a MadLoop library using
                        # makefiles built outside of the MG5_aMC framework
                        # (such as what is done with the Sherpa interface), we
                        # place here an easy handle on the golem includes
                        name_map = {'golem':'golem95','samurai':'samurai',
                                    'ninja':'ninja'}
                        ln(to_include, starting_dir=pjoin(self.dir_path,'lib'),
                                   name='%s_include'%name_map[tir],abspath=True)
                        ln(libpath, starting_dir=pjoin(self.dir_path,'lib'),
                                       name='%s_lib'%name_map[tir],abspath=True)
                else :
                    link_tir_libs.append('-l%s'%tir)
                    tir_libs.append('$(LIBDIR)lib%s.$(libext)'%tir)

        MadLoop_makefile_definitions = pjoin(self.dir_path,'SubProcesses',
                                                 'MadLoop_makefile_definitions')
        if os.path.isfile(MadLoop_makefile_definitions):
            os.remove(MadLoop_makefile_definitions)

        calls = self.write_loop_makefile_definitions(
                        writers.MakefileWriter(MadLoop_makefile_definitions),
                                link_tir_libs,tir_libs, tir_include=tir_include)

    def link_files_from_Subprocesses(self,proc_name):
        """ Does the same as the mother routine except that it also links
        coef_specs.inc in the HELAS folder."""

        LoopProcessExporterFortranSA.link_files_from_Subprocesses(self,proc_name)
        
        # Link the coef_specs.inc for aloha to define the coefficient
        # general properties (of course necessary in the optimized mode only)
        ln(os.path.join(self.dir_path,'Source','DHELAS','coef_specs.inc'),
           os.path.join(self.dir_path, 'SubProcesses', proc_name),
           abspath=False, cwd=None)


    def link_TIR(self, targetPath,libpath,libname,tir_name='TIR'):
        """Link the TIR source directory inside the target path given
        in argument"""
        
        if tir_name in ['pjfry','golem','samurai','ninja']:
            # not self-contained libraries
            if (not isinstance(libpath,str)) or (not os.path.exists(libpath)) \
            or (not os.path.isfile(pjoin(libpath,libname))):
                if isinstance(libpath,str) and libpath != '' and \
                (not os.path.isfile(pjoin(libpath,libname))):
                    # WARNING ONLY appears when the libpath is a wrong specific path.
                    logger.warning("The %s reduction library could not be found"%tir_name\
                                   +" with PATH:%s specified in mg5_configuration.txt."%libpath\
                                   +" It will not be available.")
                self.tir_available_dict[tir_name]=False
                return ""
            # Check the version of the tool, if the library was found
            if tir_name in ['ninja','samurai'] and self.tir_available_dict[tir_name]:
                # Make sure the librry was indeed installed in the source directory
                # of the tool, of course this check doesn't make sense.
                if os.path.isfile(pjoin(libpath,os.pardir,'AUTHORS')):
                    try:
                        version = open(pjoin(libpath,os.pardir,'VERSION'),'r').read()
                    except IOError:
                        version = None
                    if version is None :
                        logger.warning(
"Your version of '%s' in \n  %s\nseems too old %sto be compatible with MG5_aMC."
%(tir_name, libpath ,'' if not version else '(v%s) '%version)+
("\nConsider updating it by hand or using the 'install' function of MG5_aMC." if tir_name!='samurai'
 else "\nAsk the authors for the latest version compatible with MG5_aMC."))
        else:
            # self-contained libraries
            if (not isinstance(libpath,str)) or (not os.path.exists(libpath)):
                # WARNING ONLY appears when the libpath is a wrong specific path.
                logger.warning("The %s reduction library could not be found"%tir_name\
                                   +" with PATH:%s specified in mg5_configuration.txt."%libpath\
                                   +" It will not be available.")
                self.tir_available_dict[tir_name]=False
                return ""
       
        if self.dependencies=='internal':
            if tir_name in ['pjfry','golem','samurai','ninja']:
                self.tir_available_dict[tir_name]=False
                logger.info("When using the 'output_dependencies=internal' "+\
" MG5_aMC option, the (optional) reduction library %s cannot be employed because"%tir_name+\
" it is not distributed with the MG5_aMC code so that it cannot be copied locally.")
                return ""
            elif tir_name == "iregi":
                # This is the right paths for IREGI
                new_iregi_path = pjoin(targetPath,os.path.pardir,'Source','IREGI')
                shutil.copytree(pjoin(libpath,os.path.pardir), new_iregi_path, 
                                                                  symlinks=True)
                
                current = misc.detect_current_compiler(
                                 pjoin(new_iregi_path,'src','makefile_ML5_lib'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                        self.fortran_compiler
                if current != new:
                    misc.mod_compilator(pjoin(new_iregi_path,'src'), new,current)
                    misc.mod_compilator(pjoin(new_iregi_path,'src','oneloop'), 
                                                                   new, current)

                # Create the links to the lib folder
                ln(pjoin(targetPath,os.path.pardir,'Source','IREGI','src',
                                                            libname),targetPath)
            else:
                logger.info("Tensor integral reduction library "+\
                                            "%s not implemented yet."%tir_name)
            return libpath
 
        elif self.dependencies=='external':
            if not os.path.exists(pjoin(libpath,libname)) and tir_name=='iregi':
                logger.info('Compiling IREGI. This has to be done only once and'+\
                             ' can take a couple of minutes.','$MG:color:BLACK')
                
                current = misc.detect_current_compiler(os.path.join(\
                                                    libpath,'makefile_ML5_lib'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                        self.fortran_compiler
                if current != new:
                    misc.mod_compilator(libpath, new,current)
                    misc.mod_compilator(pjoin(libpath,'oneloop'), new, current)

                misc.compile(cwd=libpath, job_specs = False)

                if not os.path.exists(pjoin(libpath,libname)):            
                    logger.warning("IREGI could not be compiled. Check"+\
                      "the compilation errors at %s. The related "%libpath+\
                                              "functionalities are turned off.")
                    self.tir_available_dict[tir_name]=False
                    return ""
            # We link the tools below directly to the lib directory of the output 
            if not tir_name in ['pjfry','golem','samurai','ninja']:
                ln(os.path.join(libpath,libname),targetPath,abspath=True)

        elif self.dependencies=='environment_paths':
            # Here the user chose to define the dependencies path in one of 
            # his environmental paths
            newlibpath = misc.which_lib(libname)
            if not newlibpath is None:
                logger.info('MG5_aMC is using %s installation found at %s.'%\
                                                          (tir_name,newlibpath)) 
                # We link the tools below directly to directly where the library is detected
                if not tir_name in ['pjfry','golem','samurai','ninja']:
                    ln(newlibpath,targetPath,abspath=True)
                self.tir_available_dict[tir_name]=True
                return os.path.dirname(newlibpath)
            else:
                logger.warning("Could not find the location of the file"+\
                  " %s in you environment paths. The related "%libname+\
                                             "functionalities are turned off.")
                self.tir_available_dict[tir_name]=False
                return ""
            
        self.tir_available_dict[tir_name]=True
        return libpath
    
    def set_group_loops(self, matrix_element):
        """ Decides whether we must group loops or not for this matrix element"""

        # Decide if loops sharing same denominator structures have to be grouped
        # together or not.
        if self.forbid_loop_grouping:
            self.group_loops = False
        else:
            self.group_loops = (not self.get_context(matrix_element)['ComputeColorFlows'])\
                          and matrix_element.get('processes')[0].get('has_born')
        
        return self.group_loops
    
    def write_loop_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        group_number = None, proc_id = None, config_map = None):
        """ Writes loop_matrix.f, CT_interface.f,TIR_interface.f,GOLEM_inteface.f 
        and loop_num.f only but with the optimized FortranModel.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent and only used in get_ME_identifier."""
                
        # Warn the user that the 'matrix' output where all relevant code is
        # put together in a single file is not supported in this loop output.
        if writer:
            raise MadGraph5Error, 'Matrix output mode no longer supported.'
        
        if not isinstance(fortran_model,\
          helas_call_writers.FortranUFOHelasCallWriter):
            raise MadGraph5Error, 'The optimized loop fortran output can only'+\
              ' work with a UFO Fortran model'
        OptimizedFortranModel=\
          helas_call_writers.FortranUFOHelasCallWriterOptimized(\
          fortran_model.get('model'),False)


        if not matrix_element.get('processes')[0].get('has_born') and \
                                                   not self.compute_color_flows:
            logger.debug("Color flows will be employed despite the option"+\
              " 'loop_color_flows' being set to False because it is necessary"+\
                                                          " for optimizations.")

        # Compute the analytical information of the loop wavefunctions in the
        # loop helas matrix elements using the cached aloha model to reuse
        # as much as possible the aloha computations already performed for
        # writing out the aloha fortran subroutines.
        matrix_element.compute_all_analytic_information(
          self.get_aloha_model(matrix_element.get('processes')[0].get('model')))

        self.set_group_loops(matrix_element)

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        matrix_element.rep_dict = LoopProcessExporterFortranSA.\
                        generate_general_replace_dict(self, matrix_element, 
                                 group_number = group_number, proc_id = proc_id)

        # and those specific to the optimized output
        self.set_optimized_output_specific_replace_dict_entries(matrix_element)

        # Create the necessary files for the loop matrix element subroutine      
        proc_prefix_writer = writers.FortranWriter('proc_prefix.txt','w')
        proc_prefix_writer.write(matrix_element.rep_dict['proc_prefix'])
        proc_prefix_writer.close()
                    
        filename = 'loop_matrix.f'
        calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                      matrix_element,
                                      OptimizedFortranModel)    
        
        filename = 'check_sa.f'
        self.write_check_sa(writers.FortranWriter(filename),matrix_element)
        
        filename = 'polynomial.f'
        calls = self.write_polynomial_subroutines(
                                      writers.FortranWriter(filename),
                                      matrix_element)
        
        filename = 'improve_ps.f'
        calls = self.write_improve_ps(writers.FortranWriter(filename),
                                                                 matrix_element)
        
        filename = 'CT_interface.f'
        self.write_CT_interface(writers.FortranWriter(filename),\
                                matrix_element)
        
        filename = 'TIR_interface.f'
        self.write_TIR_interface(writers.FortranWriter(filename),
                                matrix_element)
        
        if 'golem' in self.tir_available_dict and self.tir_available_dict['golem']:
            filename = 'GOLEM_interface.f'
            self.write_GOLEM_interface(writers.FortranWriter(filename),
                                       matrix_element)

        filename = 'loop_num.f'
        self.write_loop_num(writers.FortranWriter(filename),\
                                           matrix_element,OptimizedFortranModel)
        
        filename = 'mp_compute_loop_coefs.f'
        self.write_mp_compute_loop_coefs(writers.FortranWriter(filename),\
                                     matrix_element,OptimizedFortranModel)

        if self.get_context(matrix_element)['ComputeColorFlows']:
            filename = 'compute_color_flows.f'
            self.write_compute_color_flows(writers.FortranWriter(filename),
                                        matrix_element, config_map = config_map)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                                                            nexternal, ninitial)
        
        # Write general process information                        
        filename = 'process_info.inc'
        self.write_process_info_file(writers.FortranWriter(filename),
                                                                 matrix_element)

        if self.get_context(matrix_element)['TIRCaching']:
            filename = 'tir_cache_size.inc'
            self.write_tir_cache_size_include(writers.FortranWriter(filename))

        return calls

    def set_optimized_output_specific_replace_dict_entries(self, matrix_element):
        """ Specify the entries of the replacement dictionary which are specific
        to the optimized output and only relevant to it (the more general entries
        are set in the the mother class LoopProcessExporterFortranSA."""
        
        max_loop_rank=matrix_element.get_max_loop_rank()
        matrix_element.rep_dict['maxrank']=max_loop_rank
        matrix_element.rep_dict['loop_max_coefs']=\
                        q_polynomial.get_number_of_coefs_for_rank(max_loop_rank)
        max_loop_vertex_rank=matrix_element.get_max_loop_vertex_rank()
        matrix_element.rep_dict['vertex_max_coefs']=\
                 q_polynomial.get_number_of_coefs_for_rank(max_loop_vertex_rank)
                 
        matrix_element.rep_dict['nloopwavefuncs']=\
                               matrix_element.get_number_of_loop_wavefunctions()
        max_spin=matrix_element.get_max_loop_particle_spin()

        matrix_element.rep_dict['max_lwf_size']= 4 if max_spin <=3 else 16
        matrix_element.rep_dict['nloops']=len(\
                        [1 for ldiag in matrix_element.get_loop_diagrams() for \
                                           lamp in ldiag.get_loop_amplitudes()])
        
        if self.set_group_loops(matrix_element):
            matrix_element.rep_dict['nloop_groups']=\
                                          len(matrix_element.get('loop_groups'))
        else:
            matrix_element.rep_dict['nloop_groups']=\
                                              matrix_element.rep_dict['nloops']
    
    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        replace_dict=copy.copy(matrix_element.rep_dict)

        file = open(os.path.join(self.template_dir,'loop_num.inc')).read()  
        file = file % replace_dict
        writer.writelines(file,context=self.get_context(matrix_element))

    def write_CT_interface(self, writer, matrix_element):
        """ We can re-use the mother one for the loop optimized output."""
        LoopProcessExporterFortranSA.write_CT_interface(\
                            self, writer, matrix_element,optimized_output=True)

    def write_TIR_interface(self, writer, matrix_element):
        """ Create the file TIR_interface.f which does NOT contain the subroutine
         defining the loop HELAS-like calls along with the general interfacing 
         subroutine. """

        # First write TIR_interface which interfaces MG5 with TIR.
        replace_dict=copy.copy(matrix_element.rep_dict)
            
        file = open(os.path.join(self.template_dir,'TIR_interface.inc')).read()  

        # Check which loops have an Higgs effective vertex so as to correctly 
        # implement CutTools limitation
        loop_groups = matrix_element.get('loop_groups')
        has_HEFT_vertex = [False]*len(loop_groups)
        for i, (denom_structure, loop_amp_list) in enumerate(loop_groups):
            for lamp in loop_amp_list:
                final_lwf = lamp.get_final_loop_wavefunction()
                while not final_lwf is None:
                    # We define here an HEFT vertex as any vertex built up from
                    # only massless vectors and scalars (at least one of each)
                    scalars = len([1 for wf in final_lwf.get('mothers') if 
                                                             wf.get('spin')==1])
                    vectors = len([1 for wf in final_lwf.get('mothers') if 
                                  wf.get('spin')==3 and wf.get('mass')=='ZERO'])
                    if scalars>=1 and vectors>=1 and \
                               scalars+vectors == len(final_lwf.get('mothers')):
                        has_HEFT_vertex[i] = True
                        break
                    final_lwf = final_lwf.get_loop_mother()
                else:
                    continue
                break

        has_HEFT_list = []
        chunk_size = 9
        for k in xrange(0, len(has_HEFT_vertex), chunk_size):
            has_HEFT_list.append("DATA (HAS_AN_HEFT_VERTEX(I),I=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(has_HEFT_vertex)),
                     ','.join(['.TRUE.' if l else '.FALSE.' for l in 
                                           has_HEFT_vertex[k:k + chunk_size]])))
        replace_dict['has_HEFT_list'] = '\n'.join(has_HEFT_list)

        file = file % replace_dict
        
        FPR = q_polynomial.FortranPolynomialRoutines(
        replace_dict['maxrank'],coef_format=replace_dict['complex_dp_format'],\
                                         sub_prefix=replace_dict['proc_prefix'])
        if self.tir_available_dict['pjfry']:
            file += '\n\n'+FPR.write_pjfry_mapping()
        if self.tir_available_dict['iregi']:
            file += '\n\n'+FPR.write_iregi_mapping()

        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    def write_GOLEM_interface(self, writer, matrix_element):
        """ Create the file GOLEM_interface.f which does NOT contain the subroutine
         defining the loop HELAS-like calls along with the general interfacing 
         subroutine. """

        # First write GOLEM_interface which interfaces MG5 with TIR.
        replace_dict=copy.copy(matrix_element.rep_dict)
        
        # We finalize TIR result differently wether we used the built-in 
        # squaring against the born.
        if not self.get_context(matrix_element)['AmplitudeReduction']:
            replace_dict['loop_induced_sqsoindex']=',SQSOINDEX'
        else:
            replace_dict['loop_induced_sqsoindex']=''
            
        file = open(os.path.join(self.template_dir,'GOLEM_interface.inc')).read()
 
        file = file % replace_dict

        FPR = q_polynomial.FortranPolynomialRoutines(replace_dict['maxrank'],\
                                                    coef_format=replace_dict['complex_dp_format'],\
                                                    sub_prefix=replace_dict['proc_prefix'])
        
        file += '\n\n'+FPR.write_golem95_mapping()
        
        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    def write_polynomial_subroutines(self,writer,matrix_element):
        """ Subroutine to create all the subroutines relevant for handling
        the polynomials representing the loop numerator """
        
        # First create 'loop_max_coefs.inc'
        IncWriter=writers.FortranWriter('loop_max_coefs.inc','w')
        IncWriter.writelines("""INTEGER LOOPMAXCOEFS
                           PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)"""
                                                       %matrix_element.rep_dict)
        
        # Then coef_specs directly in DHELAS if it does not exist already
        # 'coef_specs.inc'. If several processes exported different files there,
        # it is fine because the overall maximum value will overwrite it in the
        # end
        coef_specs_path = pjoin(self.dir_path, 'Source','DHELAS','coef_specs.inc')
        if not os.path.isfile(coef_specs_path):
            IncWriter=writers.FortranWriter(coef_specs_path,'w')
            IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %matrix_element.rep_dict)
            IncWriter.close()
        
        # List of all subroutines to place there
        subroutines=[]
        
        # Start from the routine in the template
        replace_dict = copy.copy(matrix_element.rep_dict)
                
        dp_routine = open(os.path.join(self.template_dir,'polynomial.inc')).read()
        mp_routine = open(os.path.join(self.template_dir,'polynomial.inc')).read()
        # The double precision version of the basic polynomial routines, such as
        # create_loop_coefs
        replace_dict['complex_format'] = replace_dict['complex_dp_format']
        replace_dict['real_format'] = replace_dict['real_dp_format']
        replace_dict['mp_prefix'] = ''
        replace_dict['kind'] = 8
        replace_dict['zero_def'] = '0.0d0'
        replace_dict['one_def'] = '1.0d0'
        dp_routine = dp_routine % replace_dict 
        # The quadruple precision version of the basic polynomial routines
        replace_dict['complex_format'] = replace_dict['complex_mp_format']
        replace_dict['real_format'] = replace_dict['real_mp_format']
        replace_dict['mp_prefix'] = 'MP_'
        replace_dict['kind'] = 16
        replace_dict['zero_def'] = '0.0e0_16'
        replace_dict['one_def'] = '1.0e0_16'
        mp_routine = mp_routine % replace_dict
        subroutines.append(dp_routine)
        subroutines.append(mp_routine)        

        # Initialize the polynomial routine writer
        poly_writer=q_polynomial.FortranPolynomialRoutines(
            matrix_element.get_max_loop_rank(),
            updater_max_rank = matrix_element.get_max_loop_vertex_rank(), 
            sub_prefix=replace_dict['proc_prefix'],
            proc_prefix=replace_dict['proc_prefix'],
            mp_prefix='')
        # Write the polynomial constant module common to all
        writer.writelines(poly_writer.write_polynomial_constant_module()+'\n')

        mp_poly_writer=q_polynomial.FortranPolynomialRoutines(
            matrix_element.get_max_loop_rank(),
            updater_max_rank = matrix_element.get_max_loop_vertex_rank(),        
            coef_format='complex*32', sub_prefix='MP_'+replace_dict['proc_prefix'],
            proc_prefix=replace_dict['proc_prefix'], mp_prefix='MP_')
        # The eval subroutine
        subroutines.append(poly_writer.write_polynomial_evaluator())
        subroutines.append(mp_poly_writer.write_polynomial_evaluator())
        # The add coefs subroutine
        subroutines.append(poly_writer.write_add_coefs())
        subroutines.append(mp_poly_writer.write_add_coefs())        
        # The merging one for creating the loop coefficients
        subroutines.append(poly_writer.write_wl_merger())
        subroutines.append(mp_poly_writer.write_wl_merger())
        for wl_update in matrix_element.get_used_wl_updates():
            # We pick here the most appropriate way of computing the 
            # tensor product depending on the rank of the two tensors.
            # The various choices below come out from a careful comparison of
            # the different methods using the valgrind profiler
            if wl_update[0]==wl_update[1]==1 or wl_update[0]==0 or wl_update[1]==0:
                # If any of the rank is 0, or if they are both equal to 1, 
                # then we are better off using the full expanded polynomial, 
                # and let the compiler optimize it.
                subroutines.append(poly_writer.write_expanded_wl_updater(\
                                                     wl_update[0],wl_update[1]))
                subroutines.append(mp_poly_writer.write_expanded_wl_updater(\
                                                     wl_update[0],wl_update[1]))
            elif wl_update[0] >= wl_update[1]:
                # If the loop polynomial is larger then we will filter and loop
                # over the vertex coefficients first. The smallest product for
                # which the routines below could be used is then 
                # loop_rank_2 x vertex_rank_1
                subroutines.append(poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=True))
                subroutines.append(mp_poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=True))
            else:
                # This happens only when the rank of the updater (vertex coef)
                # is larger than the one of the loop coef and none of them is
                # zero. This never happens in renormalizable theories but it
                # can happen in the HEFT ones or other effective ones. In this
                # case the typicaly use of this routine if for the product
                # loop_rank_1 x vertex_rank_2
                subroutines.append(poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=False))
                subroutines.append(mp_poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=False))            
                
        writer.writelines('\n\n'.join(subroutines),
                                       context=self.get_context(matrix_element))

    def write_mp_compute_loop_coefs(self, writer, matrix_element, fortran_model):
        """Create the write_mp_compute_loop_coefs.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(matrix_element.rep_dict)                 

        # Extract helas calls
        squared_orders = matrix_element.get_squared_order_contribs()
        split_orders = matrix_element.get('processes')[0].get('split_orders')
        
        born_ct_helas_calls , uvct_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element,
                       squared_orders=squared_orders, split_orders=split_orders)
        self.turn_to_mp_calls(born_ct_helas_calls)
        self.turn_to_mp_calls(uvct_helas_calls)
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops,
                        squared_orders=squared_orders,split_orders=split_orders)
        # The proc_prefix must be replaced
        coef_construction = [c % matrix_element.rep_dict for c 
                                                           in coef_construction]
        self.turn_to_mp_calls(coef_construction)
        self.turn_to_mp_calls(coef_merging)        
                                         
        file = open(os.path.join(self.template_dir,\
                                           'mp_compute_loop_coefs.inc')).read()

        # Setup the contextual environment which is used in the splitting
        # functions below
        context = self.get_context(matrix_element)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'mp_helas_calls_split.inc',file,born_ct_helas_calls,\
                        'mp_born_ct_helas_calls','mp_helas_calls_ampb',
                        required_so_broadcaster = 'MP_CT_REQ_SO_DONE',
                        continue_label = 2000,
                        momenta_array_name = 'MP_P',
                        context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'mp_helas_calls_split.inc',file,uvct_helas_calls,\
                        'mp_uvct_helas_calls','mp_helas_calls_uvct',
                        required_so_broadcaster = 'MP_UVCT_REQ_SO_DONE',
                        continue_label = 3000,
                        momenta_array_name = 'MP_P',
                        context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                'mp_helas_calls_split.inc',file,coef_construction,\
                'mp_coef_construction','mp_coef_construction',
                required_so_broadcaster = 'MP_LOOP_REQ_SO_DONE',
                continue_label = 4000,
                momenta_array_name = 'MP_P',
                context=context)

        replace_dict['mp_coef_merging']='\n'.join(coef_merging)
                    
        file = file % replace_dict
 
        # Write the file
        writer.writelines(file,context=context)

    def write_color_matrix_data_file(self, writer, col_matrix):
        """Writes out the files (Loop|Born)ColorFlowMatrix.dat corresponding
        to the color coefficients for JAMP(L|B)*JAMP(L|B)."""
        
        res = []
        for line in range(len(col_matrix._col_basis1)):
            numerators = []
            denominators = []
            for row in range(len(col_matrix._col_basis2)):
                coeff = col_matrix.col_matrix_fixed_Nc[(line,row)]
                numerators.append('%6r'%coeff[0].numerator)
                denominators.append('%6r'%(
                                  coeff[0].denominator*(-1 if coeff[1] else 1)))
            res.append(' '.join(numerators))
            res.append(' '.join(denominators))            
        
        res.append('EOF')
        
        writer.writelines('\n'.join(res))
    
    def write_color_flow_coefs_data_file(self, writer, color_amplitudes, 
                                                                   color_basis):
        """ Writes the file '(Loop|Born)ColorFlowCoefs.dat using the coefficients
        list of the color_amplitudes in the argument of this function."""

        my_cs = color.ColorString()        
        
        res = []

        for jamp_number, coeff_list in enumerate(color_amplitudes):
            my_cs.from_immutable(sorted(color_basis.keys())[jamp_number])
            # Order the ColorString so that its ordering is canonical.
            ordered_cs = color.ColorFactor([my_cs]).full_simplify()[0]    
            res.append('%d # Coefficient for flow number %d with expr. %s'\
                            %(len(coeff_list), jamp_number+1, repr(ordered_cs)))
            # A line element is a tuple (numerator, denominator, amplitude_id)
            line_element = []

            for (coefficient, amp_number) in coeff_list:
                coef = self.cat_coeff(\
                    coefficient[0],coefficient[1],coefficient[2],coefficient[3])
                line_element.append((coef[0].numerator,
                         coef[0].denominator*(-1 if coef[1] else 1),amp_number))
            # Sort them by growing amplitude number
            line_element.sort(key=lambda el:el[2])

            for i in range(3):
                res.append(' '.join('%6r'%elem[i] for elem in line_element))
        
        res.append('EOF')
        writer.writelines('\n'.join(res))
    
    def write_compute_color_flows(self, writer, matrix_element, config_map):
        """Writes the file compute_color_flows.f which uses the AMPL results
        from a common block to project them onto the color flow space so as 
        to compute the JAMP quantities. For loop induced processes, this file
        will also contain a subroutine computing AMPL**2 for madevent
        multichanneling."""
        
        loop_col_amps = matrix_element.get_loop_color_amplitudes()
        matrix_element.rep_dict['nLoopFlows'] = len(loop_col_amps)
        
        dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sLoopColorFlowCoefs.dat'
                                                %matrix_element.rep_dict),'w')
        self.write_color_flow_coefs_data_file(dat_writer,
                        loop_col_amps, matrix_element.get('loop_color_basis'))
        dat_writer.close()

        dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sLoopColorFlowMatrix.dat'
                                                %matrix_element.rep_dict),'w')
        self.write_color_matrix_data_file(dat_writer,
                                             matrix_element.get('color_matrix'))
        dat_writer.close() 

        if matrix_element.get('processes')[0].get('has_born'):
            born_col_amps = matrix_element.get_born_color_amplitudes()
            matrix_element.rep_dict['nBornFlows'] = len(born_col_amps)
            dat_writer = open(pjoin('..','MadLoop5_resources',
                                      '%(proc_prefix)sBornColorFlowCoefs.dat'
                                                %matrix_element.rep_dict),'w')
            self.write_color_flow_coefs_data_file(dat_writer,
                          born_col_amps, matrix_element.get('loop_color_basis'))
            dat_writer.close()
            
            dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sBornColorFlowMatrix.dat'
                                                %matrix_element.rep_dict),'w')
            self.write_color_matrix_data_file(dat_writer,
                  color_amp.ColorMatrix(matrix_element.get('born_color_basis')))
            dat_writer.close()
        else:
            matrix_element.rep_dict['nBornFlows'] = 0

        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # The following variables only have to be defined for the LoopInduced
        # output for madevent.
        if self.get_context(matrix_element)['MadEventOutput']:
            self.get_amp2_lines(matrix_element, replace_dict, config_map)
        else:
            replace_dict['config_map_definition'] = ''
            replace_dict['config_index_map_definition'] = ''            
            replace_dict['nmultichannels'] = 0
            replace_dict['nmultichannel_configs'] = 0
            
        # The nmultichannels entry will be used in the matrix<i> wrappers as 
        # well, so we add it to the general_replace_dict too.
        matrix_element.rep_dict['nmultichannels'] = \
                                                  replace_dict['nmultichannels']
        matrix_element.rep_dict['nmultichannel_configs'] = \
                                           replace_dict['nmultichannel_configs']        
        
        
        file = open(os.path.join(self.template_dir,\
                                 'compute_color_flows.inc')).read()%replace_dict

        writer.writelines(file,context=self.get_context(matrix_element))
    
    def fix_coef_specs(self, overall_max_lwf_spin, overall_max_loop_vert_rank):
        """ If processes with different maximum loop wavefunction size or
        different maximum loop vertex rank have to be output together, then
        the file 'coef.inc' in the HELAS Source folder must contain the overall
        maximum of these quantities. It is not safe though, and the user has 
        been appropriatly warned at the output stage """
        
        # Remove the existing link
        coef_specs_path=os.path.join(self.dir_path,'Source','DHELAS',\
                                                               'coef_specs.inc')
        os.remove(coef_specs_path)
       
        spin_to_wf_size = {1:4,2:4,3:4,4:16,5:16}
        overall_max_lwf_size = spin_to_wf_size[overall_max_lwf_spin]
        overall_max_loop_vert_coefs = q_polynomial.get_number_of_coefs_for_rank(
                                                     overall_max_loop_vert_rank)
        # Replace it by the appropriate value
        IncWriter=writers.FortranWriter(coef_specs_path,'w')
        IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %{'max_lwf_size':overall_max_lwf_size,
                             'vertex_max_coefs':overall_max_loop_vert_coefs})
        IncWriter.close()

    def setup_check_sa_replacement_dictionary(self, matrix_element, \
                                       split_orders,squared_orders,amps_orders):
        """ Sets up the replacement dictionary for the writeout of the steering
        file check_sa.f"""
        if len(squared_orders)<1:
            matrix_element.rep_dict['print_so_loop_results']=\
                                         "write(*,*) 'No split orders defined.'"
        elif len(squared_orders)==1:
            matrix_element.rep_dict['set_coupling_target']=''
            matrix_element.rep_dict['print_so_loop_results']=\
              "write(*,*) 'All loop contributions are of split orders (%s)'"%(
                      ' '.join(['%s=%d'%(split_orders[i],squared_orders[0][i]) \
                                            for i in range(len(split_orders))]))
        else:
            matrix_element.rep_dict['set_coupling_target']='\n'.join([
'# Here we leave the default target squared split order to -1, meaning that we'+
' aim at computing all individual contributions. You can choose otherwise.',
'call %(proc_prefix)sSET_COUPLINGORDERS_TARGET(-1)'%matrix_element.rep_dict])
            matrix_element.rep_dict['print_so_loop_results'] = '\n'.join([
              '\n'.join(["write(*,*) '%dL) Loop ME for orders (%s) :'"%((j+1),(' '.join(
          ['%s=%d'%(split_orders[i],so[i]) for i in range(len(split_orders))]))),
              "IF (PREC_FOUND(%d).NE.-1.0d0) THEN"%(j+1),
              "write(*,*) ' > accuracy = ',PREC_FOUND(%d)"%(j+1),
              "ELSE",
              "write(*,*) ' > accuracy =   NA'",              
              "ENDIF",
              "write(*,*) ' > finite   = ',MATELEM(1,%d)"%(j+1),
              "write(*,*) ' > 1eps     = ',MATELEM(2,%d)"%(j+1),
              "write(*,*) ' > 2eps     = ',MATELEM(3,%d)"%(j+1)
              ]) for j, so in enumerate(squared_orders)])
        matrix_element.rep_dict['write_so_loop_results'] = '\n'.join(
          ["write (69,*) 'Split_Orders_Names %s'"%(' '.join(split_orders))]+
          ['\n'.join([
          "write (69,*) 'Loop_SO_Results %s'"%(' '.join(
                                           ['%d'%so_value for so_value in so])),
          "write (69,*) 'SO_Loop ACC  ',PREC_FOUND(%d)"%(j+1),
          "write (69,*) 'SO_Loop FIN  ',MATELEM(1,%d)"%(j+1),
          "write (69,*) 'SO_Loop 1EPS ',MATELEM(2,%d)"%(j+1),
          "write (69,*) 'SO_Loop 2EPS ',MATELEM(3,%d)"%(j+1),
          ]) for j, so in enumerate(squared_orders)])

        # We must reconstruct here the born squared orders.
        squared_born_so_orders = []
        for i, amp_order in enumerate(amps_orders['born_amp_orders']):
            for j in range(0,i+1):
                key = tuple([ord1 + ord2 for ord1,ord2 in \
                        zip(amp_order[0],amps_orders['born_amp_orders'][j][0])])
                if not key in squared_born_so_orders:
                    squared_born_so_orders.append(key)
        if len(squared_born_so_orders)<1:
            matrix_element.rep_dict['print_so_born_results'] = ''
        elif len(squared_born_so_orders)==1: 
            matrix_element.rep_dict['print_so_born_results'] = \
              "write(*,*) 'All Born contributions are of split orders (%s)'"%(
                ' '.join(['%s=%d'%(split_orders[i],squared_born_so_orders[0][i]) 
                                            for i in range(len(split_orders))]))
        else:
            matrix_element.rep_dict['print_so_born_results'] = '\n'.join([
          "write(*,*) '%dB) Born ME for orders (%s) = ',MATELEM(0,%d)"%(j+1,' '.join(
       ['%s=%d'%(split_orders[i],so[i]) for i in range(len(split_orders))]),j+1)
                                for j, so in enumerate(squared_born_so_orders)])
        matrix_element.rep_dict['write_so_born_results'] = '\n'.join(
          ['\n'.join([
          "write (69,*) 'Born_SO_Results %s'"%(' '.join(
                                           ['%d'%so_value for so_value in so])),
          "write (69,*) 'SO_Born BORN ',MATELEM(0,%d)"%(j+1),
          ]) for j, so in enumerate(squared_born_so_orders)])
        
        # Add a bottom bar to both print_so_[loop|born]_results 
        matrix_element.rep_dict['print_so_born_results'] += \
                             '\nwrite (*,*) "---------------------------------"'
        matrix_element.rep_dict['print_so_loop_results'] += \
                             '\nwrite (*,*) "---------------------------------"'
                             
    def write_tir_cache_size_include(self, writer):
        """Write the file 'tir_cache_size.inc' which sets the size of the TIR
        cache the the user wishes to employ and the default value for it.
        This can have an impact on MadLoop speed when using stability checks
        but also impacts in a non-negligible way MadLoop's memory footprint.
        It is therefore important that the user can chose its size."""

        # For the standalone optimized output, a size of one is necessary.
        # The MadLoop+MadEvent output sets it to 2 because it can gain further
        # speed increase with a TIR cache of size 2 due to the structure of the
        # calls to MadLoop there.
        tir_cach_size = "parameter(TIR_CACHE_SIZE=1)"
        writer.writelines(tir_cach_size)

    def write_loopmatrix(self, writer, matrix_element, fortran_model, \
                                                   write_auxiliary_files=True,):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        # Starting off with the treatment of the split_orders since some 
        # of the information extracted there will come into the 
        # general_replace_dict. Split orders are abbreviated SO in all the 
        # keys of the replacement dictionaries.
        
        # Take care of the split_orders
        squared_orders, amps_orders = matrix_element.get_split_orders_mapping()
        # Creating here a temporary list containing only the information of 
        # what are the different squared split orders contributing
        # (i.e. not using max_contrib_amp_number and max_contrib_ref_amp_number)
        sqso_contribs = [sqso[0] for sqso in squared_orders]
        split_orders = matrix_element.get('processes')[0].get('split_orders')
        # The entries set in the function below are only for check_sa written
        # out in write_loop__matrix_element_v4 (it is however placed here because the
        # split order information is only available here).
        self.setup_check_sa_replacement_dictionary(matrix_element,
                                         split_orders,sqso_contribs,amps_orders)
        
        # Now recast the split order basis for the loop, born and counterterm
        # amplitude into one single splitorderbasis.
        overall_so_basis = list(set(
            [born_so[0] for born_so in amps_orders['born_amp_orders']]+
            [born_so[0] for born_so in amps_orders['loop_amp_orders']]))
        # We must re-sort it to make sure it follows an increasing WEIGHT order
        order_hierarchy = matrix_element.get('processes')[0]\
                                            .get('model').get('order_hierarchy')
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            overall_so_basis.sort(key= lambda so: 
                         sum([order_hierarchy[split_orders[i]]*order_power for \
                                              i, order_power in enumerate(so)]))

        # Those are additional entries used throughout the different files of
        # MadLoop5
        matrix_element.rep_dict['split_order_str_list'] = str(split_orders)
        matrix_element.rep_dict['nSO'] = len(split_orders)
        matrix_element.rep_dict['nSquaredSO'] = len(sqso_contribs)
        matrix_element.rep_dict['nAmpSO'] = len(overall_so_basis)

        writers.FortranWriter('nsquaredSO.inc').writelines(
"""INTEGER NSQUAREDSO
PARAMETER (NSQUAREDSO=%d)"""%matrix_element.rep_dict['nSquaredSO'])
        
        replace_dict = copy.copy(matrix_element.rep_dict)
        # Build the general array mapping the split orders indices to their
        # definition
        replace_dict['ampsplitorders'] = '\n'.join(self.get_split_orders_lines(\
                                             overall_so_basis,'AMPSPLITORDERS'))
        replace_dict['SquaredSO'] = '\n'.join(self.get_split_orders_lines(\
                                                  sqso_contribs,'SQPLITORDERS'))
        
        # Specify what are the squared split orders selected by the proc def.
        replace_dict['chosen_so_configs'] = self.set_chosen_SO_index(
                               matrix_element.get('processes')[0],sqso_contribs)
        
        # Now we build the different arrays storing the split_orders ID of each
        # amp.
        ampSO_list=[-1]*sum(len(el[1]) for el in amps_orders['loop_amp_orders'])
        for SO in amps_orders['loop_amp_orders']:
            for amp_number in SO[1]:
                ampSO_list[amp_number-1]=overall_so_basis.index(SO[0])+1

        replace_dict['loopAmpSO'] = '\n'.join(self.format_integer_list(
                                                    ampSO_list,'LOOPAMPORDERS'))
        ampSO_list=[-1]*sum(len(el[1]) for el in amps_orders['born_amp_orders'])
        for SO in amps_orders['born_amp_orders']:
            for amp_number in SO[1]:
                ampSO_list[amp_number-1]=overall_so_basis.index(SO[0])+1
        replace_dict['BornAmpSO'] = '\n'.join(self.format_integer_list(
                                                    ampSO_list,'BORNAMPORDERS'))

        # We then go to the TIR setup
        # The first entry is the CutTools, we make sure it is available
        looplibs_av=['.TRUE.']
        # one should be careful about the order in the following as it must match
        # the ordering in MadLoopParamsCard.
        for tir_lib in ['pjfry','iregi','golem','samurai','ninja']:
            looplibs_av.append('.TRUE.' if tir_lib in self.all_tir and \
                                self.tir_available_dict[tir_lib] else '.FALSE.')
        replace_dict['data_looplibs_av']=','.join(looplibs_av)

        # Helicity offset convention
        # For a given helicity, the attached integer 'i' means
        # 'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, 
        #                             to helicity number abs(i+HELOFFSET)
        # 'i' == -HELOFFSET        -> Helicity is analytically zero
        # 'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'.
        #                             If it is zero, it is skipped.
        # Typically, the hel_offset is 10000
        replace_dict['hel_offset'] = 10000

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line                  

        # When the user asks for the polarized matrix element we must 
        # multiply back by the helicity averaging factor
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        
        if write_auxiliary_files:
            # Write out the color matrix
            (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
            CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorNumFactors.dat'%matrix_element.rep_dict),'w')
            for ColorLine in CMNum:
                CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
            CMWriter.close()
            CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorDenomFactors.dat'%matrix_element.rep_dict),'w')
            for ColorLine in CMDenom:
                CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
            CMWriter.close()
            
            # Write out the helicity configurations
            HelConfigs=matrix_element.get_helicity_matrix()
            HelConfigWriter=open(pjoin('..','MadLoop5_resources',
                 '%(proc_prefix)sHelConfigs.dat'%matrix_element.rep_dict),'w')
            for HelConfig in HelConfigs:
                HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
            HelConfigWriter.close()
        
        # Extract helas calls
        born_ct_helas_calls, uvct_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element,
                        squared_orders=squared_orders,split_orders=split_orders)
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops,
                        squared_orders=squared_orders,split_orders=split_orders)

        loop_CT_calls = fortran_model.get_loop_CT_calls(matrix_element,\
                       group_loops=self.group_loops,
                       squared_orders=squared_orders, split_orders=split_orders)
        # The proc_prefix must be replaced
        coef_construction = [c % matrix_element.rep_dict for c 
                                                           in coef_construction]
        loop_CT_calls = [lc % matrix_element.rep_dict for lc in loop_CT_calls]
        
        file = open(os.path.join(self.template_dir,\
                                           'loop_matrix_standalone.inc')).read()

        # Setup the contextual environment which is used in the splitting
        # functions below
        context = self.get_context(matrix_element)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'helas_calls_split.inc',file,born_ct_helas_calls,\
                        'born_ct_helas_calls','helas_calls_ampb',
                        required_so_broadcaster = 'CT_REQ_SO_DONE',
                        continue_label = 2000, context = context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'helas_calls_split.inc',file,uvct_helas_calls,\
                        'uvct_helas_calls','helas_calls_uvct',
                        required_so_broadcaster = 'UVCT_REQ_SO_DONE',
                        continue_label = 3000, context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                'helas_calls_split.inc',file,coef_construction,\
                'coef_construction','coef_construction',
                required_so_broadcaster = 'LOOP_REQ_SO_DONE',
                continue_label = 4000, context=context)    
        file=self.split_HELASCALLS(writer,replace_dict,\
                'helas_calls_split.inc',file,loop_CT_calls,\
                'loop_CT_calls','loop_CT_calls',
                required_so_broadcaster = 'CTCALL_REQ_SO_DONE',
                continue_label = 5000, context=context)
       
        # Add the entries above to the general_replace_dict so that it can be 
        # used by write_mp_compute_loop_coefs later
        matrix_element.rep_dict['loop_CT_calls']=replace_dict['loop_CT_calls']            
        matrix_element.rep_dict['born_ct_helas_calls']=replace_dict['born_ct_helas_calls']            
        matrix_element.rep_dict['uvct_helas_calls']=replace_dict['uvct_helas_calls']            
        matrix_element.rep_dict['coef_construction']=replace_dict['coef_construction']            
        
        replace_dict['coef_merging']='\n'.join(coef_merging)

        file = file % replace_dict
        number_of_calls = len(filter(lambda call: call.find('CALL LOOP') != 0, \
                                                                 loop_CT_calls))   
        if writer:
            # Write the file
            writer.writelines(file,context=context)
            return number_of_calls
        else:
            # Return it to be written along with the others
            return number_of_calls, file

#===============================================================================
# LoopProcessExporterFortranSA
#===============================================================================
class LoopProcessExporterFortranMatchBox(LoopProcessOptimizedExporterFortranSA,
                                      export_v4.ProcessExporterFortranMatchBox):                                  
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format."""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madloop_matchbox', 'mp':True,
                        'loop_dir':'', 'cuttools_dir':'', 
                        'fortran_compiler':'gfortran',
                        'output_dependencies':'external',
                        'sa_symmetry':True}



    def get_color_string_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        return export_v4.ProcessExporterFortranMatchBox.get_color_string_lines(matrix_element)


    def get_JAMP_lines(self, *args, **opts):
        """Adding leading color part of the colorflow"""
            
        return export_v4.ProcessExporterFortranMatchBox.get_JAMP_lines(self, *args, **opts)
      
    def get_ME_identifier(self, matrix_element, group_number = None, group_elem_number = None):
        """ To not mix notations between borns and virtuals we call it here also MG5 """
        return 'MG5_%d_'%matrix_element.get('processes')[0].get('id')         
      

#===============================================================================
# LoopInducedExporter
#===============================================================================
class LoopInducedExporterME(LoopProcessOptimizedExporterFortranSA):
    """ A class to specify all the functions common to LoopInducedExporterMEGroup
    and LoopInducedExporterMENoGroup (but not relevant for the original
    Madevent exporters)"""

    madloop_makefile_name = 'makefile_MadLoop'
    
    
    def __init__(self, *args, **opts):
        """ Initialize the process, setting the proc characteristics."""
        super(LoopInducedExporterME, self).__init__(*args, **opts)
        self.proc_characteristic['loop_induced'] = True
    
    def get_context(self,*args,**opts):
        """ Make sure that the contextual variable MadEventOutput is set to
        True for this exporter"""
        
        context = super(LoopInducedExporterME,self).get_context(*args,**opts)
        context['MadEventOutput'] = True
        return context
        
    
    def get_source_libraries_list(self):
        """ Returns the list of libraries to be compiling when compiling the
        SOURCE directory. It is different for loop_induced processes and 
        also depends on the value of the 'output_dependencies' option"""
        
        libraries_list = super(LoopInducedExporterME,self).\
                                                     get_source_libraries_list()

        if self.dependencies=='internal':
            libraries_list.append('$(LIBDIR)libcts.$(libext)')
            libraries_list.append('$(LIBDIR)libiregi.$(libext)')

        return libraries_list

    def link_files_in_SubProcess(self, Ppath):
        """ Add the loop-induced related links to the P* directory Ppath"""
        
        super(LoopInducedExporterME,self).link_files_in_SubProcess(Ppath)
        
        ln(pjoin('../MadLoop5_resources') , cwd=Ppath)

    def copy_v4template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        LoopProcessExporterFortranSA.loop_additional_template_setup(self,
                                                     copy_Source_makefile=False)

        LoopProcessOptimizedExporterFortranSA.\
                                  loop_optimized_additional_template_setup(self)
                                  
    
    #===========================================================================
    # Create jpeg diagrams, html pages,proc_card_mg5.dat and madevent.tar.gz
    #===========================================================================
    def finalize_v4_directory(self, matrix_elements, history = "", makejpg = False, 
                              online = False, compiler='g77'):
        """Function to finalize v4 directory, for inheritance.
        """
        
        self.proc_characteristic['loop_induced'] = True
        
        # This can be uncommented if one desires to have the MadLoop
        # initialization performed at the end of the output phase.
        # Alternatively, one can simply execute the command 'initMadLoop' in
        # the madevent interactive interface after the output.
        # from madgraph.interface.madevent_interface import MadLoopInitializer
        # MadLoopInitializer.init_MadLoop(self.dir_path,
        #                   subproc_prefix=self.SubProc_prefix, MG_options=None)

    def write_tir_cache_size_include(self, writer):
        """Write the file 'tir_cache_size.inc' which sets the size of the TIR
        cache the the user wishes to employ and the default value for it.
        This can have an impact on MadLoop speed when using stability checks
        but also impacts in a non-negligible way MadLoop's memory footprint.
        It is therefore important that the user can chose its size."""

        # In this case of MadLoop+MadEvent output, we set it to 2 because we
        # gain further speed increase with a TIR cache of size 2 due to the 
        # the fact that we call MadLoop once per helicity configuration in this 
        # case.
        tir_cach_size = "parameter(TIR_CACHE_SIZE=2)"
        writer.writelines(tir_cach_size)

    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        proc_id = None, config_map = [], subproc_number = None):
        """ Write it the wrapper to call the ML5 subroutine in the library.""" 
        
        # Generating the MadEvent wrapping ME's routines
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
        
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        # It can be set to None when write_matrix_element_v4 is called without
        # grouping. In this case the subroutine SMATRIX should take an empty
        # suffix.
        if proc_id is None:
            replace_dict['proc_id'] = ''
        else:
            replace_dict['proc_id'] = proc_id
        
        #set the average over the number of initial helicities
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        
        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines
        
        
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
            replace_dict['configID_in_matrix'] = "SUBDIAG(%s)"%proc_id
        else:
            # Standard running
            # Set set_amp2_line
            replace_dict['configID_in_matrix'] = "MAPCONFIG(ICONFIG)"
        
        # If group_numer
        replace_dict['ml_prefix'] = \
                 self.get_ME_identifier(matrix_element, subproc_number, proc_id)
        
        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor
        
        n_tot_diags = len(matrix_element.get_loop_diagrams())
        replace_dict['n_tot_diags'] = n_tot_diags

        file = open(pjoin(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)

        return 0, ncolor

    def get_amp2_lines(self, *args, **opts):
        """Make sure the function is implemented in the daughters"""

        raise NotImplemented, 'The function get_amp2_lines must be called in '+\
                                       ' the daugthers of LoopInducedExporterME'

#===============================================================================
# LoopInducedExporterMEGroup
#===============================================================================
class LoopInducedExporterMEGroup(LoopInducedExporterME,
                                       export_v4.ProcessExporterFortranMEGroup):
    """Class to take care of exporting a set of grouped loop induced matrix 
    elements"""
    
    matrix_file = "matrix_loop_induced_madevent_group.inc"

    def make_source_links(self,*args, **opts):
        """ In the loop-induced output with MadEvent, we need the files from the 
        Source folder """
        export_v4.ProcessExporterFortranMEGroup.make_source_links(
                                                            self, *args, **opts)

    def write_source_makefile(self, *args, **opts):
        """Pick the correct write_source_makefile function from 
        ProcessExporterFortranMEGroup"""
        
        export_v4.ProcessExporterFortranMEGroup.write_source_makefile(self,
                                                                  *args, **opts)

    def copy_v4template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        
        # Start witht the MadEvent one
        export_v4.ProcessExporterFortranMEGroup.copy_v4template(self,*args,**opts)

        # Then the MadLoop-standalone related one
        LoopInducedExporterME.copy_v4template(self, *args, **opts)

    def finalize_v4_directory(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically what finalize_v4_directory must be used, so that the
        # MRO doesn't interfere.

        self.proc_characteristic['loop_induced'] = True
        
        export_v4.ProcessExporterFortranMEGroup.finalize_v4_directory(
                                                              self,*args,**opts)
        
        # And the finilize_v4 from LoopInducedExporterME which essentially takes
        # care of MadLoop virtuals initialization
        LoopInducedExporterME.finalize_v4_directory(self,*args,**opts)
        
    def generate_subprocess_directory_v4(self, subproc_group,
                                                    fortran_model,group_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files"""
            
        # Generate the MadLoop files
        calls = 0
        matrix_elements = subproc_group.get('matrix_elements')
        for ime, matrix_element in enumerate(matrix_elements):
            calls += self.generate_loop_subprocess(matrix_element,fortran_model,
          group_number = group_number, proc_id = str(ime+1),
#          group_number = str(subproc_group.get('number')), proc_id = str(ime+1),
          config_map = subproc_group.get('diagram_maps')[ime])
        
        # Then generate the MadEvent files
        export_v4.ProcessExporterFortranMEGroup.generate_subprocess_directory_v4(
                                 self, subproc_group,fortran_model,group_number)
        
        return calls
    
    def get_amp2_lines(self, matrix_element, replace_dict, config_map):
        """Return the various replacement dictionary inputs necessary for the 
        multichanneling amp2 definition for the loop-induced MadEvent output.
        """

        if not config_map:
            raise MadGraph5Error, 'A multi-channeling configuration map is '+\
              ' necessary for the MadEvent Loop-induced output with grouping.'

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()

        ret_lines = []
        # In this case, we need to sum up all amplitudes that have
        # identical topologies, as given by the config_map (which
        # gives the topology/config for each of the diagrams
        diagrams = matrix_element.get('diagrams')
                
        # Note that we need to use AMP2 number corresponding to the first 
        # diagram number used for that AMP2.
        # The dictionary below maps the config ID to this corresponding first 
        # diagram number
        config_index_map = {}
        # For each diagram number, the dictionary below gives the config_id it
        # belongs to or 0 if it doesn't belong to any.
        loop_amp_ID_to_config = {}
        
        # Combine the diagrams with identical topologies
        config_to_diag_dict = {}
        for idiag, diag in enumerate(matrix_element.get('diagrams')):
            try:
                config_to_diag_dict[config_map[idiag]].append(idiag)
            except KeyError:
                config_to_diag_dict[config_map[idiag]] = [idiag]

        for config in sorted(config_to_diag_dict.keys()):
            config_index_map[config] = (config_to_diag_dict[config][0] + 1)
                   
            # First add the UV and R2 counterterm amplitudes of each selected
            # diagram for the multichannel config
            CT_amp_numbers = [a.get('number') for a in \
                                    sum([diagrams[idiag].get_ct_amplitudes() for \
                                     idiag in config_to_diag_dict[config]], [])]
            
            for CT_amp_number in CT_amp_numbers:
                loop_amp_ID_to_config[CT_amp_number] = config 

            # Now add here the loop amplitudes.
            loop_amp_numbers = [a.get('amplitudes')[0].get('number')
                       for a in sum([diagrams[idiag].get_loop_amplitudes() for \
                                     idiag in config_to_diag_dict[config]], [])]
            
            for loop_amp_number in loop_amp_numbers:
                loop_amp_ID_to_config[loop_amp_number] = config
                
        # Notice that the config_id's are not necessarily sequential here, so
        # the size of the config_index_map array has to be the maximum over all
        # config_ids.
        # config_index_map should never be empty unless there was no diagram,
        # so the expression below is ok.
        n_configs = max(config_index_map.keys())
        replace_dict['nmultichannel_configs'] = n_configs
                
        # We must fill the empty entries of the map with the dummy amplitude 
        # number 0.
        conf_list = [(config_index_map[i] if i in config_index_map else 0) \
                                                  for i in range(1,n_configs+1)]
        # Now the placeholder 'nmultichannels' refers to the number of 
        # multi-channels which are contributing, so we must filter out zeros. 
        replace_dict['nmultichannels'] = len([_ for _ in conf_list if _!=0])
        
        # Now write the amp2 related inputs in the replacement dictionary
        res_list = []
        chunk_size = 6
        for k in xrange(0, len(conf_list), chunk_size):
            res_list.append("DATA (config_index_map(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(conf_list)),
                    ','.join(["%6r" % i for i in conf_list[k:k + chunk_size]])))

        replace_dict['config_index_map_definition'] = '\n'.join(res_list)

        res_list = []
        n_loop_amps = max(loop_amp_ID_to_config.keys())
        amp_list = [loop_amp_ID_to_config[i] for i in \
                                   sorted(loop_amp_ID_to_config.keys()) if i!=0]
        chunk_size = 6
        for k in xrange(0, len(amp_list), chunk_size):
            res_list.append("DATA (CONFIG_MAP(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(amp_list)),
                    ','.join(["%6r" % i for i in amp_list[k:k + chunk_size]])))

        replace_dict['config_map_definition'] = '\n'.join(res_list)

        return
    
#===============================================================================
# LoopInducedExporterMENoGroup
#===============================================================================
class LoopInducedExporterMENoGroup(LoopInducedExporterME,
                                            export_v4.ProcessExporterFortranME):
    """Class to take care of exporting a set of individual loop induced matrix 
    elements"""

    matrix_file = "matrix_loop_induced_madevent.inc"

    def make_source_links(self,*args, **opts):
        """ In the loop-induced output with MadEvent, we need the files from the 
        Source folder """
        super(export_v4.ProcessExporterFortranME,self).\
                                                make_source_links(*args, **opts)

    def write_source_makefile(self, *args, **opts):
        """Pick the correct write_source_makefile function from 
        ProcessExporterFortran"""
        
        super(export_v4.ProcessExporterFortranME,self).\
                                            write_source_makefile(*args, **opts)

    def copy_v4template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        
        # Start witht the MadEvent one
        export_v4.ProcessExporterFortranME.copy_v4template(self,*args,**opts)

        # Then the MadLoop-standalone related one
        LoopInducedExporterME.copy_v4template(self, *args, **opts)

    def finalize_v4_directory(self, *args, **opts):
        """Pick the right mother functions
        """
        
        self.proc_characteristic['loop_induced'] = True
        # Call specifically what finalize_v4_directory must be used, so that the
        # MRO doesn't interfere.
        export_v4.ProcessExporterFortranME.finalize_v4_directory(
                                                              self,*args,**opts)

        # And the finilize_v4 from LoopInducedExporterME which essentially takes
        # care of MadLoop virtuals initialization
        LoopInducedExporterME.finalize_v4_directory(self,*args,**opts)

    def generate_subprocess_directory_v4(self, matrix_element, fortran_model, me_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files"""
    
        # Then generate the MadLoop files
        calls = self.generate_loop_subprocess(matrix_element,fortran_model,                           
                                                       group_number = me_number)
        
        
        # First generate the MadEvent files
        calls += export_v4.ProcessExporterFortranME.generate_subprocess_directory_v4(
                                 self, matrix_element, fortran_model, me_number)
        return calls

    def get_amp2_lines(self, matrix_element, replace_dict, config_map):
        """Return the amp2(i) = sum(amp for diag(i))^2 lines"""

        if config_map:
            raise MadGraph5Error, 'A configuration map should not be specified'+\
                              ' for the Loop induced exporter without grouping.'

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        # Get minimum legs in a vertex
        vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
        matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Note that we need to use AMP2 number corresponding to the first 
        # diagram number used for that AMP2.
        # The dictionary below maps the config ID to this corresponding first 
        # diagram number
        config_index_map = {}
        # For each diagram number, the dictionary below gives the config_id it
        # belongs to or 0 if it doesn't belong to any.
        loop_amp_ID_to_config = {}

        n_configs = 0
        for idiag, diag in enumerate(matrix_element.get('diagrams')):
            # Ignore any diagrams with 4-particle vertices.
            use_for_multichanneling = True
            if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                use_for_multichanneling = False
                curr_config = 0
            else:
                n_configs += 1
                curr_config = n_configs

            if not use_for_multichanneling:
                if 0 not in config_index_map: 
                    config_index_map[0] = idiag + 1
            else:
                config_index_map[curr_config] = idiag + 1
                   
            CT_amps = [ a.get('number') for a in diag.get_ct_amplitudes()]
            for CT_amp in CT_amps:
                loop_amp_ID_to_config[CT_amp] = curr_config
                            
            Loop_amps = [a.get('amplitudes')[0].get('number')
                                            for a in diag.get_loop_amplitudes()]
            for Loop_amp in Loop_amps:
                loop_amp_ID_to_config[Loop_amp] = curr_config
 
        # Now write the amp2 related inputs in the replacement dictionary
        n_configs = len([k for k in config_index_map.keys() if k!=0])
        replace_dict['nmultichannel_configs'] = n_configs
        # Now the placeholder 'nmultichannels' refers to the number of 
        # multi-channels which are contributing which, in the non-grouped case
        # is always equal to the total number of multi-channels. 
        replace_dict['nmultichannels'] = n_configs
        
        res_list = []
        conf_list = [config_index_map[i] for i in sorted(config_index_map.keys())
                                                                        if i!=0]
        chunk_size = 6
        for k in xrange(0, len(conf_list), chunk_size):
            res_list.append("DATA (config_index_map(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(conf_list)),
                    ','.join(["%6r" % i for i in conf_list[k:k + chunk_size]])))

        replace_dict['config_index_map_definition'] = '\n'.join(res_list)

        res_list = []
        n_loop_amps = max(loop_amp_ID_to_config.keys())
        amp_list = [loop_amp_ID_to_config[i] for i in \
                                   sorted(loop_amp_ID_to_config.keys()) if i!=0]
        chunk_size = 6
        for k in xrange(0, len(amp_list), chunk_size):
            res_list.append("DATA (CONFIG_MAP(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(amp_list)),
                    ','.join(["%6r" % i for i in amp_list[k:k + chunk_size]])))

        replace_dict['config_map_definition'] = '\n'.join(res_list)
