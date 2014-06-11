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
import madgraph.core.color_amp as color_amp
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.check_param_card as check_param_card

pjoin = os.path.join

import aloha.create_aloha as create_aloha
import models.write_param_card as param_writer
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
from madgraph.iolibs.files import cp, ln, mv
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

    def __init__(self, mgme_dir="", dir_path = "", opt=None):
        """Initiate the LoopExporterFortran with directory information on where
        to find all the loop-related source files, like CutTools"""

        if opt:
            self.opt = opt
        else: 
            self.opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madloop', 'mp':True,
                        'loop_dir':'', 'cuttools_dir':'', 
                        'fortran_compiler':'gfortran',
                        'output_dependencies':'external'}

        self.loop_dir = self.opt['loop_dir']
        self.cuttools_dir = self.opt['cuttools_dir']
        self.fortran_compiler = self.opt['fortran_compiler']
        self.dependencies = self.opt['output_dependencies']

        super(LoopExporterFortran,self).__init__(mgme_dir, dir_path, self.opt)
        
    def link_CutTools(self, targetPath):
        """Link the CutTools source directory inside the target path given
        in argument"""
                
        if self.dependencies=='internal':
            shutil.copytree(self.cuttools_dir, 
                           pjoin(targetPath,'Source','CutTools'), symlinks=True)
            # Create the links to the lib folder
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(os.path.join(os.path.pardir,'Source','CutTools','includects',file),
                                                 os.path.join(targetPath,'lib'))
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
                                                 os.path.join(targetPath,'lib'))

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

    def copy_v4template(self, modelname):
        """Additional actions needed for setup of Template
        """
        super(LoopProcessExporterFortranSA, self).copy_v4template(modelname)
        
        # We must change some files to their version for NLO computations
        cpfiles= ["Source/makefile","SubProcesses/makefile",\
                  "SubProcesses/check_sa.f","SubProcesses/MadLoopParamReader.f",
                  "Cards/MadLoopParams.dat",
                  "SubProcesses/MadLoopParams.inc"]
        
        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))

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
        
    def convert_model_to_mg4(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
        """ Caches the aloha model created here when writing out the aloha 
        fortran subroutine.
        """
        self.get_aloha_model(model)
        super(LoopProcessExporterFortranSA, self).convert_model_to_mg4(model,
           wanted_lorentz = wanted_lorentz, wanted_couplings = wanted_couplings)

    #===========================================================================
    # Set the compiler to be gfortran for the loop processes.
    #===========================================================================
    def compiler_choice(self, compiler):
        """ Different daughter classes might want different compilers.
        Here, the gfortran compiler is used throughout the compilation 
        (mandatory for CutTools written in f90) """
        if not compiler is None and not any([name in compiler for name in \
                                                         ['gfortran','ifort']]):
            logger.info('For loop processes, the compiler must be fortran90'+\
                        'compatible, like gfortran.')
            self.set_compiler('gfortran',True)
        else:
            self.set_compiler(compiler)

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
            if i==10:
                elapsed_time = time.time()-start
                t = int(len(ampl_to_jampl)*(elapsed_time/10.0))
                if t > 20:
                    time_info = True
                    logger.info('The color factors computation will take '+\
                      ' about %s to run. '%str(datetime.timedelta(seconds=t))+\
                      'Started on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))
                    if logger.getEffectiveLevel()<logging.WARNING:
                        widgets = ['Color computation:', pbar.Percentage(), ' ', 
                                                pbar.Bar(),' ', pbar.ETA(), ' ']
                        progress_bar = pbar.ProgressBar(widgets=widgets, 
                                        maxval=len(ampl_to_jampl),fd=sys.stdout)
            if progress_bar!=None:
                progress_bar.update(i+1)
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

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    
    def generate_loop_subprocess(self, matrix_element, fortran_model):
        """Generate the Pxxxxx directory for a loop subprocess in MG4 standalone,
        including the necessary loop_matrix.f, born_matrix.f and include files.
        Notice that this is too different from generate_subprocess_directory_v4
        so that there is no point reusing this mother function."""

        cwd = os.getcwd()
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = os.path.join(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

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

        calls=self.write_matrix_element_v4(None,matrix_element,fortran_model)

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

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             (nexternal-2), ninitial)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        # Do not draw the loop diagrams if they are too many.
        # The user can always decide to do it manually, if really needed
        if (len(matrix_element.get('base_amplitude').get('loop_diagrams'))>1000):
            logger.info("There are more than 1000 loop diagrams."+\
                                               "Only the first 1000 are drawn.")
        filename = "loop_matrix.ps"
        writers.FortranWriter(filename).writelines("""C Post-helas generation loop-drawing is not ready yet.""")
        plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(
              matrix_element.get('base_amplitude').get('loop_diagrams')[:1000]),
              filename,
              model=matrix_element.get('processes')[0].get('model'),
              amplitude='')
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

        if not matrix_element.get('processes')[0].get('has_born'):
            # There is a specific check_sa.f for loop induced processes
            shutil.copy(os.path.join(self.loop_dir,'StandAlone','SubProcesses',
                                     'check_sa_loop_induced.f'),
                        os.path.join(self.dir_path, 'SubProcesses','check_sa.f'))

        self.link_files_from_Subprocesses(proc_name=\
                              matrix_element.get('processes')[0].shell_string())
        
        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    def link_files_from_Subprocesses(self,proc_name=""):
        """ To link required files from the Subprocesses directory to the
        different P* ones"""
        
        linkfiles = ['check_sa.f', 'coupl.inc', 'makefile',
                     'cts_mprec.h', 'cts_mpc.h', 'mp_coupl.inc', 
                     'mp_coupl_same_name.inc',
                     'MadLoopParamReader.f',
                     'MadLoopParams.inc']
        
        for file in linkfiles:
            ln('../%s' % file)

        # The mp module
        ln('../../lib/mpmodule.mod')
            
        # For convenience also link the madloop param card
        ln('../../Cards/MadLoopParams.dat')

    def generate_general_replace_dict(self,matrix_element):
        """Generates the entries for the general replacement dictionary used
        for the different output codes for this exporter"""
        
        dict={}
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        dict['info_lines'] = info_lines
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        dict['process_lines'] = process_lines
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        dict['nexternal'] = nexternal-2
        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        dict['ncomb'] = ncomb
        # Extract nloopamps
        nloopamps = matrix_element.get_number_of_loop_amplitudes()
        dict['nloopamps'] = nloopamps
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
        # Color matrix size
        # For loop induced processes it is NLOOPAMPSxNLOOPAMPS and otherwise
        # it is NLOOPAMPSxNBORNAMPS
        if matrix_element.get('processes')[0].get('has_born'):
            dict['color_matrix_size'] = 'nbornamps'
        else:
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
        else:
            dict['ncomb_helas_objs'] = ''  
            dict['dp_born_amps_decl'] = ''
            dict['dp_born_amps_decl_in_mp'] = ''
            dict['copy_mp_to_dp_born_amps'] = ''
            dict['mp_born_amps_decl'] = ''
            dict['nbornamps_decl'] = ''
        
        return dict
    
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                proc_id = "", config_map = []):
        """ Writes loop_matrix.f, CT_interface.f and loop_num.f only"""
        
        # Create the necessary files for the loop matrix element subroutine

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
        self.general_replace_dict=\
                              self.generate_general_replace_dict(matrix_element)
        # Extract max number of loop couplings (specific to this output type)
        self.general_replace_dict['maxlcouplings']= \
                                         matrix_element.find_max_loop_coupling()
        # The born amp declaration suited for also outputing the loop-induced
        # processes as well.
        if matrix_element.get('processes')[0].get('has_born'):
            self.general_replace_dict['dp_born_amps_decl_in_mp'] = \
                  self.general_replace_dict['complex_dp_format']+" DPAMP(NBORNAMPS,NCOMB)"+\
                  "\n common/AMPS/DPAMP"
            self.general_replace_dict['dp_born_amps_decl'] = \
                  self.general_replace_dict['complex_dp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/AMPS/AMP"
            self.general_replace_dict['mp_born_amps_decl'] = \
                  self.general_replace_dict['complex_mp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/MP_AMPS/AMP"
            self.general_replace_dict['copy_mp_to_dp_born_amps'] = \
                   '\n'.join(['DO I=1,NBORNAMPS','DPAMP(I,H)=AMP(I,H)','ENDDO'])
        
        if writer:
            raise MadGraph5Error, 'Matrix output mode no longer supported.'
        
        else:
            filename = 'loop_matrix.f'
            calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                          matrix_element,
                                          LoopFortranModel) 
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

            return calls

    def generate_subprocess_directory_v4(self, matrix_element,
                                         fortran_model):
        """ To overload the default name for this function such that the correct
        function is used when called from the command interface """
        
        return self.generate_loop_subprocess(matrix_element,fortran_model)


    def write_improve_ps(self, writer, matrix_element):
        """ Write out the improve_ps subroutines which modify the PS point
        given in input and slightly deform it to achieve exact onshellness on
        all external particles as well as perfect energy-momentum conservation""" 
        replace_dict = copy.copy(self.general_replace_dict)
        
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
        
        replace_dict = copy.copy(self.general_replace_dict)
        
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
        the loop HELAS-like calls along with the general interfacing subroutine. """

        files=[]

        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(self.general_replace_dict)
        
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
            replace_dict=copy.copy(self.general_replace_dict)
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
            writer.writelines(file)
        else:
            return file

    # Helper function to split HELAS CALLS in dedicated subroutines placed
    # in different files.
    def split_HELASCALLS(self, writer, replace_dict, template_name, masterfile, \
                         helas_calls, entry_name, bunch_name,n_helas=2000):
        """ Finish the code generation with splitting.         
        Split the helas calls in the argument helas_calls into bunches of 
        size n_helas and place them in dedicated subroutine with name 
        <bunch_name>_i. Also setup the corresponding calls to these subroutine 
        in the replace_dict dictionary under the entry entry_name. """
        helascalls_replace_dict=copy.copy(replace_dict)
        helascalls_replace_dict['bunch_name']=bunch_name
        helascalls_files=[]
        for i, k in enumerate(range(0, len(helas_calls), n_helas)):
            helascalls_replace_dict['bunch_number']=i+1                
            helascalls_replace_dict['helas_calls']=\
                                           '\n'.join(helas_calls[k:k + n_helas])
            new_helascalls_file = open(os.path.join(self.template_dir,\
                                            template_name)).read()
            new_helascalls_file = new_helascalls_file % helascalls_replace_dict
            helascalls_files.append(new_helascalls_file)
        # Setup the call to these HELASCALLS subroutines in loop_matrix.f
        helascalls_calls = [ "CALL %s_%d(P,NHEL,H,IC)"%(bunch_name,a+1) \
                            for a in range(len(helascalls_files))]
        replace_dict[entry_name]='\n'.join(helascalls_calls)
        if writer:
            for i, helascalls_file in enumerate(helascalls_files):
                filename = '%s_%d.f'%(bunch_name,i+1)
                writers.FortranWriter(filename).writelines(helascalls_file)
        else:
                masterfile='\n'.join([masterfile,]+helascalls_files)                

        return masterfile

    def write_loopmatrix(self, writer, matrix_element, fortran_model, \
                         noSplit=False):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(self.general_replace_dict)
        
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
            """HELPICKED=HELPICKED_BU
               DO I=NCTAMPS+1,NLOOPAMPS
               IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN
                 WRITE(*,*) '##W03 WARNING Contribution ',I
                 WRITE(*,*) ' is unstable for helicity ',H
               ENDIF
C                IF(.NOT.ISZERO(ABS(AMPL(2,I))+ABS(AMPL(3,I)),REF,-1,H)) THEN
C                  WRITE(*,*) '##W04 WARNING Contribution ',I,' for helicity ',H,' has a contribution to the poles.'
C                  WRITE(*,*) 'Finite contribution         = ',AMPL(1,I)
C                  WRITE(*,*) 'single pole contribution    = ',AMPL(2,I)
C                  WRITE(*,*) 'double pole contribution    = ',AMPL(3,I)
C                ENDIF
               ENDDO
               1227 CONTINUE"""
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
            replace_dict['set_reference']='call smatrix(p,ref)'
            replace_dict['loop_induced_helas_calls'] = ""
            replace_dict['loop_induced_finalize'] = ""
            replace_dict['loop_induced_setup'] = ""
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nbornamps'
            replace_dict['squaring']='\n'.join(['DO K=1,3',
                   'ANS(K)=ANS(K)+2.0d0*DBLE(CFTOT*AMPL(K,I)*DCONJG(AMP(J,H)))',
                                                                       'ENDDO'])

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
            """IF(.NOT.ISZERO(ABS(ANS(2))+ABS(ANS(3)),REF*(10.0d0**-2),-1,H)) THEN
                 WRITE(*,*) '##W05 WARNING Found a PS point with a contribution to the single pole.'
                 WRITE(*,*) 'Finite contribution         = ',ANS(1)
                 WRITE(*,*) 'single pole contribution    = ',ANS(2)
                 WRITE(*,*) 'double pole contribution    = ',ANS(3)
               ENDIF"""
        
        # Write out the color matrix
        (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
        CMWriter=open('ColorNumFactors.dat','w')
        for ColorLine in CMNum:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        CMWriter=open('ColorDenomFactors.dat','w')
        for ColorLine in CMDenom:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        
        # Write out the helicity configurations
        HelConfigs=matrix_element.get_helicity_matrix()
        HelConfigWriter=open('HelConfigs.dat','w')
        for HelConfig in HelConfigs:
            HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
        HelConfigWriter.close()
        
        # Extract helas calls
        loop_amp_helas_calls = fortran_model.get_loop_amp_helas_calls(\
                                                            matrix_element)
        born_ct_helas_calls = fortran_model.get_born_ct_helas_calls(\
                                                            matrix_element)
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
        if writer:
            # Write the file
            writer.writelines(file)  
            return len(filter(lambda call: call.find('CALL LOOP') != 0, \
                              loop_amp_helas_calls))
        else:
            # Return it to be written along with the others
            return len(filter(lambda call: call.find('CALL LOOP') != 0, \
                              loop_amp_helas_calls)), file
                  
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
        
        return super(LoopProcessExporterFortranSA,self).\
          write_matrix_element_v4(writer,bornME,fortran_model)

    def write_born_amps_and_wfs(self, writer, matrix_element, fortran_model,\
                                noSplit=False): 
        """ Writes out the code for the subroutine MP_BORN_AMPS_AND_WFS which 
        computes just the external wavefunction and born amplitudes in 
        multiple precision. """

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        replace_dict = copy.copy(self.general_replace_dict)

        # For the wavefunction copy, check what suffix is needed for the W array
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['h_w_suffix']=',H'
        else:
            replace_dict['h_w_suffix']=''            

        # Extract helas calls
        born_amps_and_wfs_calls = fortran_model.get_born_ct_helas_calls(\
                                                matrix_element, include_CT=True)
        
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
    # CutTools call the loops with same denominator structure
    group_loops=True

    def link_files_from_Subprocesses(self,proc_name=""):
        """ Does the same as the mother routine except that it also links
        coef_specs.inc in the HELAS folder."""

        LoopProcessExporterFortranSA.link_files_from_Subprocesses(self,proc_name)
        
        # Link the coef_specs.inc for aloha to define the coefficient
        # general properties (of course necessary in the optimized mode only)
        ln(os.path.join(self.dir_path, 'SubProcesses', "P%s" % proc_name,
                 'coef_specs.inc'),os.path.join(self.dir_path,'Source/DHELAS/'))

    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                proc_id = "", config_map = []):
        """ Writes loop_matrix.f, CT_interface.f and loop_num.f only but with
        the optimized FortranModel"""
        # Create the necessary files for the loop matrix element subroutine
        
        if not isinstance(fortran_model,\
          helas_call_writers.FortranUFOHelasCallWriter):
            raise MadGraph5Error, 'The optimized loop fortran output can only'+\
              ' work with a UFO Fortran model'
        OptimizedFortranModel=\
          helas_call_writers.FortranUFOHelasCallWriterOptimized(\
          fortran_model.get('model'),False)

        # Compute the analytical information of the loop wavefunctions in the
        # loop helas matrix elements using the cached aloha model to reuse
        # as much as possible the aloha computations already performed for
        # writing out the aloha fortran subroutines.
        matrix_element.compute_all_analytic_information(
          self.get_aloha_model(matrix_element.get('processes')[0].get('model')))

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        self.general_replace_dict=LoopProcessExporterFortranSA.\
                              generate_general_replace_dict(self,matrix_element)
            
        # Now some features specific to the optimized output        
        max_loop_rank=matrix_element.get_max_loop_rank()
        self.general_replace_dict['loop_max_coefs']=\
                        q_polynomial.get_number_of_coefs_for_rank(max_loop_rank)
        max_loop_vertex_rank=matrix_element.get_max_loop_vertex_rank()
        self.general_replace_dict['vertex_max_coefs']=\
                 q_polynomial.get_number_of_coefs_for_rank(max_loop_vertex_rank)
        self.general_replace_dict['nloopwavefuncs']=\
                               matrix_element.get_number_of_loop_wavefunctions()
        max_spin=matrix_element.get_max_loop_particle_spin()
        if max_spin>3:
            raise MadGraph5Error, "ML5 can only handle loop particles with"+\
                                                               " spin 1 at most"
        self.general_replace_dict['max_lwf_size']=4
        self.general_replace_dict['nloops']=len(\
                        [1 for ldiag in matrix_element.get_loop_diagrams() for \
                                           lamp in ldiag.get_loop_amplitudes()])
        if self.group_loops and \
                             matrix_element.get('processes')[0].get('has_born'):
            self.general_replace_dict['nloop_groups']=\
                                          len(matrix_element.get('loop_groups'))
        else:
            self.general_replace_dict['nloop_groups']=\
                                              self.general_replace_dict['nloops']

        # The born amp declaration suited for also outputing the loop-induced
        # processes as well. (not used for now, but later)
        if matrix_element.get('processes')[0].get('has_born'):
            self.general_replace_dict['dp_born_amps_decl'] = \
                  self.general_replace_dict['complex_dp_format']+" AMP(NBORNAMPS)"+\
                  "\n common/AMPS/AMP"
            self.general_replace_dict['mp_born_amps_decl'] = \
                  self.general_replace_dict['complex_mp_format']+" AMP(NBORNAMPS)"+\
                  "\n common/MP_AMPS/AMP"

        if writer:
            raise MadGraph5Error, 'Matrix output mode no longer supported.'

        else:
                        
            filename = 'loop_matrix.f'
            calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                          matrix_element,
                                          OptimizedFortranModel)
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

            filename = 'loop_num.f'
            self.write_loop_num(writers.FortranWriter(filename),\
                                    matrix_element,OptimizedFortranModel)
            
            filename = 'mp_compute_loop_coefs.f'
            self.write_mp_compute_loop_coefs(writers.FortranWriter(filename),\
                                         matrix_element,OptimizedFortranModel)

            return calls

    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        replace_dict=copy.copy(self.general_replace_dict)

        file = open(os.path.join(self.template_dir,'loop_num.inc')).read()  
        file = file % replace_dict
        writer.writelines(file)

    def write_CT_interface(self, writer, matrix_element):
        """ We can re-use the mother one for the loop optimized output."""
        LoopProcessExporterFortranSA.write_CT_interface(\
                            self, writer, matrix_element,optimized_output=True)

    def write_polynomial_subroutines(self,writer,matrix_element):
        """ Subroutine to create all the subroutines relevant for handling
        the polynomials representing the loop numerator """
        
        # First create 'coef_specs.inc'
        IncWriter=writers.FortranWriter('coef_specs.inc','w')
        IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER LOOP_MAXCOEFS
                           PARAMETER (LOOP_MAXCOEFS=%(loop_max_coefs)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %self.general_replace_dict)
        IncWriter.close()
        
        # List of all subroutines to place there
        subroutines=[]
        
        # Start from the routine in the template
        file = open(os.path.join(self.template_dir,'polynomial.inc')).read()  
        file = file % self.general_replace_dict
        subroutines.append(file)
        
        # Initialize the polynomial routine writer
        poly_writer=q_polynomial.FortranPolynomialRoutines(
                                             matrix_element.get_max_loop_rank())
        mp_poly_writer=q_polynomial.FortranPolynomialRoutines(
                    matrix_element.get_max_loop_rank(),coef_format='complex*32',
                                                               sub_prefix='MP_')
        # The eval subroutine
        subroutines.append(poly_writer.write_polynomial_evaluator())
        subroutines.append(mp_poly_writer.write_polynomial_evaluator())
        # The add coefs subroutine
        subroutines.append(poly_writer.write_add_coefs())
        subroutines.append(mp_poly_writer.write_add_coefs())        
        # The merging one for creating the loop coefficients
        subroutines.append(poly_writer.write_wl_merger())
        subroutines.append(mp_poly_writer.write_wl_merger())
        # Now the udpate subroutines
        for wl_update in matrix_element.get_used_wl_updates():
            subroutines.append(poly_writer.write_wl_updater(\
                                                     wl_update[0],wl_update[1]))
            subroutines.append(mp_poly_writer.write_wl_updater(\
                                                     wl_update[0],wl_update[1]))
        writer.writelines('\n\n'.join(subroutines))

    def write_mp_compute_loop_coefs(self, writer, matrix_element, fortran_model, \
                                    noSplit=False):
        """Create the write_mp_compute_loop_coefs.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(self.general_replace_dict)                 

        # These entries are specific for the output for loop-induced processes
        # Also sets here the details of the squaring of the loop ampltiudes
        # with the born or the loop ones.
        if not matrix_element.get('processes')[0].get('has_born'):
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nctamps'
            replace_dict['mp_squaring']=\
          'ANS(1)=ANS(1)+DUMMY*REAL(CFTOT*AMPL(1,I)*CONJG(AMPL(1,J),KIND=16),KIND=16)'        
        else:
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nbornamps'
            replace_dict['mp_squaring']='\n'.join(['DO K=1,3',
                'ANS(K)=ANS(K)+DUMMY*2.0e0_16*REAL(CFTOT*AMPL(K,I)*CONJG(AMP(J))'+\
                                                           ',KIND=16)','ENDDO'])
        
        # Extract helas calls
        born_ct_helas_calls = fortran_model.get_born_ct_helas_calls(\
                                                            matrix_element)
        self.turn_to_mp_calls(born_ct_helas_calls)
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops)
        self.turn_to_mp_calls(coef_construction)
        self.turn_to_mp_calls(coef_merging)        
                                         
        file = open(os.path.join(self.template_dir,\
                                           'mp_compute_loop_coefs.inc')).read()

        # Decide here wether we need to split the loop_matrix.f file or not.
        # 200 is reasonable but feel free to change it.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>200)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'mp_helas_calls_split.inc',file,born_ct_helas_calls,\
                            'mp_born_ct_helas_calls','mp_helas_calls_ampb')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'mp_helas_calls_split.inc',file,coef_construction,\
                    'mp_coef_construction','mp_coef_construction')
        else:
            replace_dict['mp_born_ct_helas_calls']='\n'.join(born_ct_helas_calls)
            replace_dict['mp_coef_construction']='\n'.join(coef_construction)
        
        replace_dict['mp_coef_merging']='\n'.join(coef_merging)
        
        file = file % replace_dict
 
        # Write the file
        writer.writelines(file)  

    def fix_coef_specs(self, overall_max_lwf_size, overall_max_loop_vert_rank):
        """ If processes with different maximum loop wavefunction size or
        different maximum loop vertex rank have to be output together, then
        the file 'coef.inc' in the HELAS Source folder must contain the overall
        maximum of these quantities. It is not safe though, and the user has 
        been appropriatly warned at the output stage """
        
        # Remove the existing link
        coef_specs_path=os.path.join(self.dir_path,'Source','DHELAS',\
                                                               'coef_specs.inc')
        os.remove(coef_specs_path)
        
        # Replace it by the appropriate value
        IncWriter=writers.FortranWriter(coef_specs_path,'w')
        IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %{'max_lwf_size':overall_max_lwf_size,
                             'vertex_max_coefs':overall_max_loop_vert_rank})
        IncWriter.close()

    def write_loopmatrix(self, writer, matrix_element, fortran_model, \
                         noSplit=False):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(self.general_replace_dict)

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

        # These entries are specific for the output for loop-induced processes
        # Also sets here the details of the squaring of the loop ampltiudes
        # with the born or the loop ones.
        if not matrix_element.get('processes')[0].get('has_born'):
            replace_dict['set_reference']='\n'.join([
              'C Chose the arbitrary scale of reference to use for comparisons'+\
              ' for this loop-induced process.','ref=1.0d-50'])
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nctamps'
            replace_dict['squaring']=\
                    'ANS(1)=ANS(1)+DUMMY*DBLE(CFTOT*AMPL(1,I)*DCONJG(AMPL(1,J)))'
                    
        else:
            replace_dict['set_reference']='call smatrix(p,ref)'
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nbornamps'
            replace_dict['squaring']='\n'.join(['DO K=1,3',
                   'ANS(K)=ANS(K)+2.0d0*DUMMY*DBLE(CFTOT*AMPL(K,I)*DCONJG(AMP(J)))',
                                                                       'ENDDO'])

        # Actualize results from the loops computed. Only necessary for
        # processes with a born.
        actualize_ans=[]
        if matrix_element.get('processes')[0].get('has_born'):
            actualize_ans.append("DO I=1,NLOOPGROUPS")
            actualize_ans.extend("ANS(%d)=ANS(%d)+LOOPRES(%d,I)"%(i,i,i) for i \
                                                                  in range(1,4)) 
            actualize_ans.append(\
               "IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN")
            actualize_ans.append(\
                   "WRITE(*,*) '##W03 WARNING Contribution ',I,' is unstable.'")            
            actualize_ans.extend(["ENDIF","ENDDO"])            
        replace_dict['actualize_ans']='\n'.join(actualize_ans)
        
        # Write out the color matrix
        (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
        CMWriter=open('ColorNumFactors.dat','w')
        for ColorLine in CMNum:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        CMWriter=open('ColorDenomFactors.dat','w')
        for ColorLine in CMDenom:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        
        # Write out the helicity configurations
        HelConfigs=matrix_element.get_helicity_matrix()
        HelConfigWriter=open('HelConfigs.dat','w')
        for HelConfig in HelConfigs:
            HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
        HelConfigWriter.close()
        
        # Extract helas calls
        born_ct_helas_calls = fortran_model.get_born_ct_helas_calls(\
                                                            matrix_element)
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops)
        loop_CT_calls = fortran_model.get_loop_CT_calls(\
                                    matrix_element,group_loops=self.group_loops)
        
        file = open(os.path.join(self.template_dir,\
                                           'loop_matrix_standalone.inc')).read()

        # Decide here wether we need to split the loop_matrix.f file or not.
        # 200 is reasonable but feel free to change it.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>200)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'helas_calls_split.inc',file,born_ct_helas_calls,\
                            'born_ct_helas_calls','helas_calls_ampb')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'helas_calls_split.inc',file,coef_construction,\
                    'coef_construction','coef_construction')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'helas_calls_split.inc',file,loop_CT_calls,\
                    'loop_CT_calls','loop_CT_calls')
        else:
            replace_dict['born_ct_helas_calls']='\n'.join(born_ct_helas_calls)
            replace_dict['coef_construction']='\n'.join(coef_construction)
            replace_dict['loop_CT_calls']='\n'.join(loop_CT_calls)
        
        replace_dict['coef_merging']='\n'.join(coef_merging)
        
        file = file % replace_dict
        number_of_calls = len(filter(lambda call: call.find('CALL LOOP') != 0, \
                                                                 loop_CT_calls))   
        if writer:
            # Write the file
            writer.writelines(file)  
            return number_of_calls
        else:
            # Return it to be written along with the others
            return number_of_calls, file
