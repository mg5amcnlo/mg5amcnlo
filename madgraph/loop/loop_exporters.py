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
"""Methods and classes to export matrix elements to v4 format."""

import copy
import fractions
import glob
import logging
import os
import re
import shutil
import subprocess

import aloha

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.various.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import madgraph.core.color_amp as color_amp
import madgraph.iolibs.helas_call_writers as helas_call_writers


import aloha.create_aloha as create_aloha
import models.write_param_card as param_writer
from madgraph import MadGraph5Error, MG5DIR
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
        Notice that we do have LoopExporterFortran inheriting from 
        ProcessExporterFortran hence giving access to arguments like dir_path and
        clean. This creates a diamond inheritance scheme in which we avoid mro
        (method resolution order) ambiguity by using unique method names here."""

    def __init__(self, loop_dir = "", cuttools_dir = "", *args, **kwargs):
        """Initiate the LoopExporterFortran with directory information on where
        to find all the loop-related source files, like CutTools"""

        self.loop_dir = loop_dir
        self.cuttools_dir = cuttools_dir
        super(LoopExporterFortran,self).__init__(*args, **kwargs)
        
    def link_CutTools(self, targetPath):
        """Link the CutTools source directory inside the target path given
        in argument"""

        cwd = os.getcwd()

        try:
            os.chdir(targetPath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0

        if not os.path.exists(os.path.join(self.cuttools_dir,'includects','libcts.a')):
            logger.info('Compiling CutTools')
            misc.compile(cwd=self.cuttools_dir)

        if os.path.exists(os.path.join(self.cuttools_dir,'includects','libcts.a')):            
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(os.path.join(self.cuttools_dir,'includects')+'/%s' % file)
        else:
            raise MadGraph5Error,"CutTools could not be correctly compiled."

        # Return to original PWD
        os.chdir(cwd)

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
class LoopProcessExporterFortranSA(export_v4.ProcessExporterFortranSA,
                                   LoopExporterFortran):
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format."""

    # Init function to initialize both mother classes, depending on the nature
    # of the argument given
    def __init__(self, *args, **kwargs):    
       super(LoopProcessExporterFortranSA, self).__init__(*args, **kwargs)          

    def copy_v4template(self, modelname):
        """Additional actions needed for setup of Template
        """
        
        super(LoopProcessExporterFortranSA, self).copy_v4template(modelname)
        
        # We must link the CutTools to the Library folder of the active Template
        super(LoopProcessExporterFortranSA, self).link_CutTools(
                                    os.path.join(self.dir_path, 'lib'))
        
        # We must change some files to their version for NLO computations
        cpfiles= ["Source/makefile","SubProcesses/makefile",\
                  "SubProcesses/check_sa.f"]
        
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
                            writers.FortranWriter('cts_mpc.h'),)

        # Return to original PWD
        os.chdir(cwd)

    #===========================================================================
    # Set the compiler to be gfortran for the loop processes.
    #===========================================================================
    def compiler_choice(self, compiler):
        """ Different daughter classes might want different compilers.
        Here, the gfortran compiler is used throughout the compilation 
        (mandatory for CutTools written in f90) """
        if compiler not in ['gfortran','ifort']:
            logger.info('For loop processes, the compiler must be fortran90'+\
                        'compatible, like gfortran.')
            self.set_compiler('gfortran',True)
        else:
            self.set_compiler(compiler)

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

        filename = "loop_matrix.ps"
#        writers.FortranWriter(filename).writelines("""C Post-helas generation loop-drawing is not ready yet.""")
#        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
#                                             get('loop_diagrams'),
#                                          filename,
#                                          model=matrix_element.get('processes')[0].\
#                                             get('model'),
#                                          amplitude='')
#        logger.info("Generating loop Feynman diagrams for " + \
#                     matrix_element.get('processes')[0].nice_string())
#        plot.draw()

        if matrix_element.get('processes')[0].get('has_born'):   
            filename = "born_matrix.ps"
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('born_diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude='')
            logger.info("Generating born Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        if not matrix_element.get('processes')[0].get('has_born'):
            # There is a specific check_sa.f for loop induced processes
            shutil.copy(os.path.join(self.loop_dir,'StandAlone','Subprocesses',
                                     'check_sa_loop_induced.f'),
                        os.path.join(self.dir_path, 'Subprocesses','check_sa.f'))

        self.link_files_from_Subprocesses()
        
        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    def link_files_from_Subprocesses(self):
        """ To link required files from the Subprocesses directory to the
        different P* ones"""
        
        linkfiles = ['check_sa.f', 'coupl.inc', 'makefile', \
                     'cts_mprec.h', 'cts_mpc.h']
        for file in linkfiles:
            ln('../%s' % file)

        linkfiles = ['mpmodule.mod']

        for file in linkfiles:
            ln('../../lib/%s' % file)

    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                proc_id = "", config_map = []):
        """ Writes loop_matrix.f, CT_interface.f and loop_num.f only"""
        # Create the necessary files for the loop matrix element subroutine
                
        if writer:
            file1 = self.write_loop_num(None,matrix_element,fortran_model)
            file2 = self.write_CT_interface(None,matrix_element)
            calls, file3 = self.write_loopmatrix(None,matrix_element,fortran_model)               
            file = "\n".join([file1,file2,file3])
            writer.writelines(file)
            return calls
        
        else:
            
            filename = 'loop_matrix.f'
            calls = self.write_loopmatrix(
                                          writers.FortranWriter(filename),
                                          matrix_element,
                                          fortran_model)             
            filename = 'CT_interface.f'
            self.write_CT_interface(writers.FortranWriter(filename),\
                                    matrix_element)
            
            filename = 'loop_num.f'
            self.write_loop_num(writers.FortranWriter(filename),\
                                    matrix_element,fortran_model)
        
            return calls


    def generate_subprocess_directory_v4(self, matrix_element,
                                         fortran_model):
        """ To overload the default name for this function such that the correct
        function is used when called from the command interface """
        
        return self.generate_loop_subprocess(matrix_element,fortran_model)

    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/loop_num.inc')).read()
        
        replace_dict = {}
        
        replace_dict['mass_format']="complex*16"
        replace_dict['nexternal']=(matrix_element.get_nexternal_ninitial()[0]-2)
        loop_helas_calls=fortran_model.get_loop_amplitude_helas_calls(matrix_element)
        replace_dict['maxlcouplings']=matrix_element.find_max_loop_coupling()
        replace_dict['loop_helas_calls'] = "\n".join(loop_helas_calls)

        file=file%replace_dict
        
        if writer:
            writer.writelines(file)
        else:
            return file
        
    def write_CT_interface(self, writer, matrix_element):
        """ Create the file loop_helas.f which contains the subroutine defining
        the loop HELAS-like calls along with the general interfacing subroutine. """

        files=[]

        # Fill here what's common to all files read here.             
        replace_dict_orig={}

        # Extract the number of external legs
        replace_dict_orig['nexternal']=\
          (matrix_element.get_nexternal_ninitial()[0]-2)

        # specify the format of the masses 
        replace_dict_orig['mass_format'] = 'complex*16'

        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(replace_dict_orig)
        file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/CT_interface.inc')).read()
        
        replace_dict['mass_translation'] = 'M2L(I)'
        # specify what scalar loop library must be used.
        # For now we use AVH for both CMS and nonCMS outputs.
        if aloha.complex_mass:
            replace_dict['loop_lib'] = 2
        else:
            replace_dict['loop_lib'] = 2  
        
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines
        
        file = file % replace_dict
        files.append(file)
        
        # Now collect the different kind of subroutines needed for the
        # loop HELAS-like calls.
        CallKeys=[]
        for ldiag in matrix_element.get_loop_diagrams():
            for lamp in ldiag.get_loop_amplitudes():
                if lamp.get_call_key()[1:] not in CallKeys:
                    CallKeys.append(lamp.get_call_key()[1:])
                
        for callkey in CallKeys:
            replace_dict=copy.copy(replace_dict_orig)
            # Add to this dictionary all other attribute common to all
            # HELAS-like loop subroutines.
            replace_dict['maxlcouplings']=matrix_element.find_max_loop_coupling()
            replace_dict['nloopline']=callkey[0]
            wfsargs="".join([("W%d, MP%d, "%(i,i)) for i in range(1,callkey[1]+1)])
            replace_dict['ncplsargs']=callkey[2]
            replace_dict['wfsargs']=wfsargs
            margs="".join([("M"+str(i)+", ") for i in range(1,callkey[0]+1)])
            replace_dict['margs']=margs
            cplsargs="".join([("C"+str(i)+", ") for i in range(1,callkey[2]+1)])
            replace_dict['cplsargs']=cplsargs
            wfsargsdecl="".join([("W%d(20), "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['wfsargsdecl']=wfsargsdecl
            momposdecl="".join([("MP%d, "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['momposdecl']=momposdecl
            margsdecl="".join([("M"+str(i)+", ") for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['margsdecl']=margsdecl
            cplsdecl="".join([("C"+str(i)+", ") for i in range(1,callkey[2]+1)])[:-2]
            replace_dict['cplsdecl']=cplsdecl
            weset="\n".join([("WE(I,"+str(i)+")=W"+str(i)+"(I)") for \
                             i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            momposset="\n".join([("MOMPOS(%d)=MP%d"%(i,i)) for \
                             i in range(1,callkey[1]+1)])
            replace_dict['momposset']=momposset
            msetlines=["M2L(1)=M%d**2"%(callkey[0]),]
            mset="\n".join(msetlines+[("M2L("+str(i)+")=M"+str(i-1)+"**2") for \
                             i in range(2,callkey[0]+1)])
            replace_dict['mset']=mset
            cplset="\n".join([("LC("+str(i)+")=C"+str(i)) for \
                             i in range(1,callkey[2]+1)])
            replace_dict['cplset']=cplset            
            mset2lines=["ML(1)=M%d"%(callkey[0]),"ML(2)=M%d"%(callkey[0])]
            mset2="\n".join(mset2lines+[("ML("+str(i)+")=M"+str(i-2)) for \
                             i in range(3,callkey[0]+2)])
            replace_dict['mset2']=mset2           
            replace_dict['nwfsargs'] = callkey[1]
            if callkey[0]==callkey[1]:
                file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/helas_loop_amplitude.inc')).read()                
            else:
                file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/helas_loop_amplitude_pairing.inc')).read() 
                pairingargs="".join([("P"+str(i)+", ") for i in range(1,callkey[0]+1)])
                replace_dict['pairingargs']=pairingargs
                pairingdecl="".join([("P"+str(i)+", ") for i in range(1,callkey[0]+1)])[:-2]
                replace_dict['pairingdecl']=pairingdecl
                pairingset="\n".join([("PAIRING("+str(i)+")=P"+str(i)) for \
                             i in range(1,callkey[0]+1)])
                replace_dict['pairingset']=pairingset
            file = file % replace_dict
            files.append(file)   
        
        file="\n".join(files)
        
        if writer:
            writer.writelines(file)
        else:
            return file
        
    def write_loopmatrix(self, writer, matrix_element, fortran_model, noSplit=False):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Decide here wether we need to split the loop_matrix.f file or not.
        needSplitting=(not noSplit and \
                       (len(matrix_element.get_all_amplitudes())>1000))
        # If splitting is necessary, one has two choices to treat color:
        # A: splitColor=False
        #       The color info is kept together in loop_matrix.f
        # B: splitColor=True
        #       The JAMP initialization is done as for the born, but they are
        #       split in different jamp_calls_#.f subroutines. 
        splitColor=True

        # Helper function to define the JAMP and HELAS calls directly in the
        # loop_matrix.f file or split in dedicated subroutines split in
        # different files if the output is too large.
        def normal_HELASJAMP():
            """ Finish the loop_matrix.f generation without splitting """
            replace_dict['helas_calls'] = "\n".join(helas_calls)
            # Extract BORNJAMP lines
            born_jamp_lines = self.get_JAMP_lines(matrix_element.get_born_color_amplitudes(),
                                                  "JAMPB(","AMP(",15)
            replace_dict['born_jamp_lines'] = '\n'.join(born_jamp_lines)
            # Extract LOOPJAMP lines
            loop_jamp_lines = self.get_JAMP_lines(matrix_element.get_loop_color_amplitudes(),
                                                  "JAMPL(K,","AMPL(K,",15)
            replace_dict['loop_jamp_lines'] = '\n'.join(loop_jamp_lines) 
            
        def split_HELASJAMP(masterfile):
            """ Finish the loop_matrix.f generation with splitting """         
            # Split the helas calls into bunches of size n_helas calls.
            n_helas=2000
            helascalls_replace_dict={}
            for key in ['nloopamps','nbornamps','nexternal','nwavefuncs','ncomb']:
                helascalls_replace_dict[key]=replace_dict[key]
            helascalls_files=[]
            for i, k in enumerate(range(0, len(helas_calls), n_helas)):
                helascalls_replace_dict['bunch_number']=i+1                
                helascalls_replace_dict['helas_calls']='\n'.join(helas_calls[k:k + n_helas])
                new_helascalls_file = open(os.path.join(_file_path, \
                     'iolibs/template_files/loop/helas_calls_split.inc')).read()
                new_helascalls_file = new_helascalls_file % helascalls_replace_dict
                helascalls_files.append(new_helascalls_file)
            # Setup the call to these HELASCALLS subroutines in loop_matrix.f
            helascalls_calls = [ "CALL HELASCALLS_%d(P,NHEL,H,IC)"%(a+1) for a in \
                                  range(len(helascalls_files))]
            replace_dict['helas_calls']='\n'.join(helascalls_calls)
            if writer:
                for i, helascalls_file in enumerate(helascalls_files):
                    filename = 'helas_calls_%d.f'%(i+1)
                    writers.FortranWriter(filename).writelines(helascalls_file)
            else:
                    masterfile='\n'.join([masterfile,]+helascalls_files)                
            
            if splitColor:
                # Split the jamp definition into bunches of size n_jamp calls.
                n_jamp=250
                # Extract BORNJAMP lines
                born_jamp_lines = self.get_JAMP_lines(matrix_element.get_born_color_amplitudes(),
                                                  "JAMPB(","AMP(",7)
                # Extract LOOPJAMP lines
                loop_jamp_lines = self.get_JAMP_lines(matrix_element.get_loop_color_amplitudes(),
                                                      "JAMPL(K,","AMPL(K,",7)
                replace_dict['loop_jamp_lines'] = '\n'.join(loop_jamp_lines) 
                jampcalls_replace_dict={}
                for key in ['nloopamps','nbornamps','nwavefuncs','ncomb',\
                            'ncolorloop','ncolorborn']:
                    jampcalls_replace_dict[key]=replace_dict[key]
                jampBcalls_files=[]                
                jampLcalls_files=[]
                jampcalls_replace_dict['argument']=""
                jampcalls_replace_dict['tag_letter']="B"
                for i, k in enumerate(range(0, len(born_jamp_lines), n_jamp)):
                    jampcalls_replace_dict['bunch_number']=i+1                
                    jampcalls_replace_dict['jamp_calls']='\n'.join(born_jamp_lines[k:k + n_jamp])
                    new_jampcalls_file = open(os.path.join(_file_path, \
                         'iolibs/template_files/loop/jamp_calls_split.inc')).read()
                    new_jampcalls_file = new_jampcalls_file % jampcalls_replace_dict
                    jampBcalls_files.append(new_jampcalls_file)
                jampcalls_replace_dict['argument']="K"
                jampcalls_replace_dict['tag_letter']="L"
                for i, k in enumerate(range(0, len(loop_jamp_lines), n_jamp)):
                    jampcalls_replace_dict['bunch_number']=i+1                
                    jampcalls_replace_dict['jamp_calls']='\n'.join(loop_jamp_lines[k:k + n_jamp])
                    new_jampcalls_file = open(os.path.join(_file_path, \
                         'iolibs/template_files/loop/jamp_calls_split.inc')).read()
                    new_jampcalls_file = new_jampcalls_file % jampcalls_replace_dict
                    jampLcalls_files.append(new_jampcalls_file)                    
                # Setup the call to these JAMPCALLS subroutines in loop_matrix.f
                jampBcalls_calls = [ "CALL JAMPBCALLS_%d()"%(a+1) for a in \
                                      range(len(jampBcalls_files))]
                # Setup the call to these JAMPCALLS subroutines in loop_matrix.f
                jampLcalls_calls = [ "CALL JAMPLCALLS_%d(K)"%(a+1) for a in \
                                      range(len(jampLcalls_files))]
                replace_dict['born_jamp_lines'] = '\n'.join(jampBcalls_calls)
                replace_dict['loop_jamp_lines'] = '\n'.join(jampLcalls_calls)
                if writer:
                    for i, jampBcalls_file in enumerate(jampBcalls_files):
                        filename = 'jampB_calls_%d.f'%(i+1)
                        writers.FortranWriter(filename).writelines(jampBcalls_file)
                    for i, jampLcalls_file in enumerate(jampLcalls_files):
                        filename = 'jampL_calls_%d.f'%(i+1)
                        writers.FortranWriter(filename).writelines(jampLcalls_file)
                else:
                    masterfile='\n'.join([masterfile,]+jampBcalls_files)
                    masterfile='\n'.join([masterfile,]+jampLcalls_files)                    
            else:
                # Create the born_color.inc include
                replace_dict2 = {}
                born_color_amplitudes=matrix_element.get_born_color_amplitudes()
                replace_dict2['max_ncontrib_born'] =  \
                    max([len(coefs) for coefs in born_color_amplitudes])
                born_jamp_factors = self.get_JAMP_coefs(born_color_amplitudes, \
                   matrix_element.get('born_color_basis'), tag_letter="B")
                replace_dict2['color_coefs_lines'] =  '\n'.join(born_jamp_factors)
                file2 = open(os.path.join(_file_path, \
                         'iolibs/template_files/loop/born_color.inc')).read()
                file2 = file2 % replace_dict2
                # Create the loop_color.inc include
                replace_dict3 = {}
                loop_color_amplitudes=matrix_element.get_loop_color_amplitudes()
                replace_dict3['max_ncontrib_loop'] =  \
                    max([len(coefs) for coefs in loop_color_amplitudes])
                loop_jamp_factors = self.get_JAMP_coefs(loop_color_amplitudes, \
                   matrix_element.get('loop_color_basis'), tag_letter="L")
                replace_dict3['color_coefs_lines'] =  '\n'.join(loop_jamp_factors)
                file3 = open(os.path.join(_file_path, \
                         'iolibs/template_files/loop/loop_color.inc')).read()
                file3 = file3 % replace_dict3
                # Now write these color coefficients as include file only if the 
                # user does not require for a self contained single loop_matrix.f file.
                if writer:
                    # Write the two color include files
                    filename = 'born_color.inc'
                    writers.FortranWriter(filename).writelines(file2)
                    filename = 'loop_color.inc'
                    writers.FortranWriter(filename).writelines(file3)
                    # Write the loop_matrix.f driver file.
                    replace_dict['born_jamps_coefs'] = "include 'born_color.inc'"
                    replace_dict['loop_jamps_coefs'] = "include 'loop_color.inc'"  
                else:
                    # The user wants a unique self-contained loop_matrix.f
                    replace_dict['born_jamps_coefs'] = file2
                    replace_dict['loop_jamps_coefs'] = file3
            return masterfile
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal-2

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

        # Extract nloopamps
        nloopamps = matrix_element.get_number_of_loop_amplitudes()
        replace_dict['nloopamps'] = nloopamps

        # Extract nbronamps
        nbornamps = matrix_element.get_number_of_born_amplitudes()
        replace_dict['nbornamps'] = nbornamps

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_external_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolorloop
        ncolorloop = max(1, len(matrix_element.get('loop_color_basis')))
        replace_dict['ncolorloop'] = ncolorloop

        # Extract ncolorborn
        ncolorborn = max(1, len(matrix_element.get('born_color_basis')))
        replace_dict['ncolorborn'] = ncolorborn

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                                                            matrix_element) 
                       
        # The summing is performed differently depending on wether
        # the process is loop induced or not
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['squaring']="""
          DO K = 1, 3
              DO I = 1, NCOLORLOOP
                  ZTEMP = (0.D0,0.D0)          
                  DO J = 1, NCOLORBORN
                      ZTEMP = ZTEMP + CF(J,I)*JAMPB(J)
                  ENDDO
                  RES(K) = RES(K)+DBLE(ZTEMP*DCONJG(JAMPL(K,I))/DENOM(I)) 
              ENDDO
              DO J = 1, NCOLORBORN
                  ZTEMP = (0.D0,0.D0)
                  DO I = 1, NCOLORLOOP
                      ZTEMP = ZTEMP + CF(J,I)*JAMPL(K,I)/DENOM(I)
                  ENDDO
                  RES(K) = RES(K)+DBLE(ZTEMP*DCONJG(JAMPB(J)))
              ENDDO
          ENDDO
            """
        else:
            replace_dict['squaring']="""
              DO I = 1, NCOLORLOOP
                  ZTEMP = (0.D0,0.D0)          
                  DO J = 1, NCOLORBORN
                      ZTEMP = ZTEMP + CF(J,I)*JAMPL(1,J)
                  ENDDO
                  RES(1) = RES(1)+DBLE(ZTEMP*DCONJG(JAMPL(1,I))/DENOM(I)) 
              ENDDO
            """       

        if not needSplitting:
            file = open(os.path.join(_file_path, \
                'iolibs/template_files/loop/loop_matrix_standalone.inc')).read()
            normal_HELASJAMP()
        else:     
            file = open(os.path.join(_file_path, \
              'iolibs/template_files/loop/loop_matrix_standalone_split.inc'\
              if splitColor else \
              'iolibs/template_files/loop/loop_matrix_standalone_split_helasCallsOnly.inc')).read()
            file=split_HELASJAMP(file)
                                                    
        file = file % replace_dict
        if writer:
            # Write the file
            writer.writelines(file)  
            return len(filter(lambda call: call.find('CALL LOOP') != 0, helas_calls))
        else:
            return len(filter(lambda call: call.find('CALL LOOP') != 0, helas_calls)), file
                  
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
        
        bornME = helas_objects.HelasMatrixElement()
        for prop in bornME.keys():
            bornME.set(prop,matrix_element.get(prop))
        bornME.set('base_amplitude',None,force=True)
        bornME.set('diagrams',matrix_element.get_born_diagrams())        
        bornME.set('color_basis',matrix_element.get('born_color_basis'))
        bornME.set('color_matrix',color_amp.ColorMatrix(bornME.get('color_basis')))
        
        return super(LoopProcessExporterFortranSA,self).\
          write_matrix_element_v4(writer,bornME,fortran_model)

#===============================================================================
# LoopProcessOptimizedExporterFortranSA
#===============================================================================
class LoopProcessOptimizedExporterFortranSA(LoopProcessExporterFortranSA):
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format including many optimization like:
           1. Calling CutTools for each loop amplitude squared against the born
              amplitudes and already summed over the helicity configurations."""

    template_dir=os.path.join(_file_path,\
                      'iolibs/template_files/loop_optimized')

    def turn_to_mp_calls(self, helas_calls_list):
        # Prepend 'MP_' to all the helas calls in helas_calls_list.
        # Might look like a brutal unsafe implementation, but it is not as 
        # these calls are built from the properties of the HELAS objects and
        # wether they are evaluated in double or quad precision is none of 
        # their business but only relevant to the output algorithm.
        # Also the cast to complex masses DCMPLX(*) must be replaced by
        # CMPLX(*,KIND=16)
        MP=re.compile(r"(?P<toSub>^.*CALL\s+)",re.IGNORECASE)
        
        def replaceWith(match_obj):
            return match_obj.group('toSub')+'MP_'

        DCMPLX=re.compile(r"DCMPLX\((?P<toSub>([^\)]*))\)",re.IGNORECASE)
        
        for i, helas_call in enumerate(helas_calls_list):
            new_helas_call=MP.sub(replaceWith,helas_call)
            helas_calls_list[i]=DCMPLX.sub(r"CMPLX(\g<toSub>,KIND=16)",\
                                                                 new_helas_call)

    def copy_v4template(self, modelname,*args, **opts):
        """Additional actions for the optimized output needed for setup of 
        Template
        """
        LoopProcessExporterFortranSA.copy_v4template(self, modelname,\
                                                                 *args, **opts)
        # We add here the user-friendly MadLoop option setter.
        cpfiles= ["SubProcesses/MadLoopParamReader.f",
                  "SubProcesses/MadLoopParams.dat",
                  "SubProcesses/MadLoopParams.inc"]
        
        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))

    def link_files_from_Subprocesses(self):
        """ Links the additional model files for multiples precision """
        
        LoopProcessExporterFortranSA.link_files_from_Subprocesses(self)
        linkfiles = ['mp_coupl.inc', 'mp_coupl_same_name.inc',
                     'MadLoopParamReader.f','MadLoopParams.dat',
                     'MadLoopParams.inc']
        
        for file in linkfiles:
            ln('../%s' % file)       

    def make_model_symbolic_link(self):
        """ Add the linking of the additional model files for multiple precision
        """
        LoopProcessExporterFortranSA.make_model_symbolic_link(self)
        model_path = self.dir_path + '/Source/MODEL/'
        ln(model_path + '/mp_coupl.inc', self.dir_path + '/SubProcesses')
        ln(model_path + '/mp_coupl_same_name.inc', self.dir_path + '/SubProcesses')
    
    def cat_coeff(self, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Concatenate the coefficient information to reduce it to 
        (fraction, is_imaginary) """

        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

        return (total_coeff, is_imaginary)       

    def get_amp_to_jamp_map(self, col_amps, n_amps):
        """ Returns a list with element 'i' being a list of tuples corresponding
        to all appearance of amplitude number 'i' in the jamp number 'j'
        with coeff 'coeff_j'. The format of each tuple describing an appearance 
        is (j, coeff_j)."""

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

        # The two lists have a list of tuples at element 'i' which correspond
        # to all appearance of loop amplitude number 'i' in the jampl number 'j'
        # with coeff 'coeffj'. The format of each tuple describing an appearance 
        # is (j, coeffj).
        ampl_to_jampl=self.get_amp_to_jamp_map(\
          matrix_element.get_loop_color_amplitudes(),
          matrix_element.get_number_of_loop_amplitudes())
        ampb_to_jampb=self.get_amp_to_jamp_map(\
          matrix_element.get_born_color_amplitudes(),
          matrix_element.get_number_of_born_amplitudes())
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
        for jampl_list in ampl_to_jampl:
            line=[]
            line_denom=[]
            for jampb_list in ampb_to_jampb:
                coeff_real=fractions.Fraction(0,1)
                coeff_imag=fractions.Fraction(0,1)
                for (jampl, ampl_coeff) in jampl_list:
                    for (jampb, ampb_coeff) in jampb_list:
                        buff=ampl_coeff[0]*ampb_coeff[0]*\
                          fractions.Fraction(ColorMatrixNum[jampl][jampb],\
                            ColorMatrixDenom[jampl])
                        # Remember that we must take the complex conjugate of
                        # the born jamp color coefficient because we will compute
                        # the square with 2 Re(LoopAmp x BornAmp*)
                        if ampl_coeff[1] and ampb_coeff[1]:
                            coeff_real=coeff_real+buff
                        elif not ampl_coeff[1] and not ampb_coeff[1]:
                            coeff_real=coeff_real+buff
                        elif not ampl_coeff[1] and ampb_coeff[1]:
                            coeff_imag=coeff_imag-buff
                        else:
                            coeff_imag=coeff_imag+buff
                if coeff_real!=0 and coeff_imag!=0:
                    raise MadGraph5Error,"MadGraph5 found a color matrix element"+\
                      " which has both a real and imaginary part."
                elif coeff_imag!=0:
                    line.append(coeff_imag)
                    # Negative denominator means imaginary color coef of the final color matrix
                    line_denom.append(-1)
                else:
                    line.append(coeff_real)
                    # Positive denominator means real color coef of the final color matrix
                    line_denom.append(1)
              
            lcmm = color_amp.ColorMatrix.lcmm(*[coeff.denominator \
                                      for coeff in line])
            # The sign of the denom is already provided, we only set the absolute value here
            line_denom = [denom*lcmm for denom in line_denom]
            ColorMatrixDenomOutput.append(line_denom)
            ColorMatrixNumOutput.append([\
              (coeff.numerator*abs(line_denom[0])/coeff.denominator) for coeff in line])

        return (ColorMatrixNumOutput,ColorMatrixDenomOutput)

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
          fortran_model.get('model'))

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        self.general_replace_dict={}
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        self.general_replace_dict['info_lines'] = info_lines
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        self.general_replace_dict['process_lines'] = process_lines
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        self.general_replace_dict['nexternal'] = nexternal-2
        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        self.general_replace_dict['ncomb'] = ncomb
        # Extract nloopamps
        nloopamps = matrix_element.get_number_of_loop_amplitudes()
        self.general_replace_dict['nloopamps'] = nloopamps
        # Extract nbornamps
        nbornamps = matrix_element.get_number_of_born_amplitudes()
        self.general_replace_dict['nbornamps'] = nbornamps
        # Extract nctamps
        nctamps = matrix_element.get_number_of_CT_amplitudes()
        self.general_replace_dict['nctamps'] = nctamps
        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_external_wavefunctions()
        self.general_replace_dict['nwavefuncs'] = nwavefuncs
        # Extract max number of couplings
        self.general_replace_dict['maxlcouplings']=\
           matrix_element.find_max_loop_coupling()
        # Set format of the double precision
        self.general_replace_dict['real_dp_format']='real*8'
        self.general_replace_dict['real_mp_format']='real*16'
        # Set format of the complex
        self.general_replace_dict['complex_dp_format']='complex*16'
        self.general_replace_dict['complex_mp_format']='complex*32'
        # Set format of the masses
        self.general_replace_dict['mass_dp_format'] = \
                          self.general_replace_dict['complex_dp_format']
        self.general_replace_dict['mass_mp_format'] = \
                          self.general_replace_dict['complex_mp_format']
        
        if writer:
            files=[]
            files.append(self.write_loop_num(None,matrix_element,\
                                                         OptimizedFortranModel))
            files.append(self.write_CT_interface(None,matrix_element))
            calls, loop_matrix = self.write_loopmatrix(None,matrix_element,\
                                                          OptimizedFortranModel)
            files.append(loop_matrix)
            files.append(write_born_amps_and_wfs(None,matrix_element,\
                                                        OptimizedFortranModel))
            file = "\n".join(files)
            writer.writelines(file)
            return calls
        
        else:
                        
            filename = 'loop_matrix.f'
            calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                          matrix_element,
                                          OptimizedFortranModel)             
            filename = 'CT_interface.f'
            self.write_CT_interface(writers.FortranWriter(filename),\
                                    matrix_element)
            
            filename = 'loop_num.f'
            self.write_loop_num(writers.FortranWriter(filename),\
                                    matrix_element,OptimizedFortranModel)
            
            filename = 'mp_born_amps_and_wfs.f'
            self.write_born_amps_and_wfs(writers.FortranWriter(filename),\
                                         matrix_element,OptimizedFortranModel)

            return calls                

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
        
        # Now stuff for the multiple precision numerator function
        (nexternal,ninitial)=matrix_element.get_nexternal_ninitial()
        replace_dict['n_initial']=ninitial
        # The last momenta is fixed by the others and the last two particles
        # are the L-cut ones, so -3.
        mass_list=matrix_element.get_external_masses()[:-3]
        replace_dict['force_onshell']='\n'.join([\
        'P(3,%(i)d)=SIGN(SQRT(P(0,%(i)d)**2-P(1,%(i)d)**2-P(2,%(i)d)**2-%(m)s**2),P(3,%(i)d))'%\
         {'i':i+1,'m':m} for i, m in enumerate(mass_list)])
        # Prepend MP_ to all helas calls.
        self.turn_to_mp_calls(loop_helas_calls)
        replace_dict['mp_loop_helas_calls'] = "\n".join(loop_helas_calls)
        
        file=file%replace_dict
        
        if writer:
            writer.writelines(file)
        else:
            return file
        
    def write_CT_interface(self, writer, matrix_element):
        """ Create the file CT_interface.f which contains the subroutine defining
        the loop HELAS-like calls along with the general interfacing subroutine. """

        files=[]
        
        replace_dict_orig=copy.copy(self.general_replace_dict)

        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(replace_dict_orig)
        file = open(os.path.join(self.template_dir,'CT_interface.inc')).read()  

        file = file % replace_dict
        files.append(file)
        
        # Now collect the different kind of subroutines needed for the
        # loop HELAS-like calls.
        CallKeys=[]
        for ldiag in matrix_element.get_loop_diagrams():
            for lamp in ldiag.get_loop_amplitudes():
                if lamp.get_call_key()[1:] not in CallKeys:
                    CallKeys.append(lamp.get_call_key()[1:])
                
        for callkey in CallKeys:
            replace_dict=copy.copy(replace_dict_orig)
            # Add to this dictionary all other attribute common to all
            # HELAS-like loop subroutines.
            replace_dict['nloopline']=callkey[0]
            wfsargs="".join([("W%d, MP%d, "%(i,i)) for i in range(1,callkey[1]+1)])
            replace_dict['ncplsargs']=callkey[2]
            replace_dict['wfsargs']=wfsargs
            margs="".join(["M%d,MP_M%d, "%(i,i) for i in range(1,callkey[0]+1)])
            replace_dict['margs']=margs
            cplsargs="".join(["C%d,MP_C%d, "%(i,i) for i in range(1,callkey[2]+1)])
            replace_dict['cplsargs']=cplsargs
            wfsargsdecl="".join([("W%d, "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['wfsargsdecl']=wfsargsdecl
            momposdecl="".join([("MP%d, "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['momposdecl']=momposdecl
            margsdecl="".join(["M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['margsdecl']=margsdecl
            mp_margsdecl="".join(["MP_M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['mp_margsdecl']=mp_margsdecl
            cplsdecl="".join(["C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
            replace_dict['cplsdecl']=cplsdecl
            mp_cplsdecl="".join(["MP_C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
            replace_dict['mp_cplsdecl']=mp_cplsdecl
            weset="\n".join([("WE("+str(i)+")=W"+str(i)) for \
                             i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            momposset="\n".join([("MOMPOS(%d)=MP%d"%(i,i)) for \
                             i in range(1,callkey[1]+1)])
            replace_dict['momposset']=momposset
            msetlines=["M2L(1)=M%d**2"%(callkey[0]),]
            mset="\n".join(msetlines+["M2L(%d)=M%d**2"%(i,i-1) for \
                             i in range(2,callkey[0]+1)])
            replace_dict['mset']=mset
            cplset="\n".join(["\n".join(["LC(%d)=C%d"%(i,i),\
                                         "MP_LC(%d)=MP_C%d"%(i,i)])\
                              for i in range(1,callkey[2]+1)])
            replace_dict['cplset']=cplset            
            mset2lines=["ML(1)=M%d"%(callkey[0]),"ML(2)=M%d"%(callkey[0]),
                  "MP_ML(1)=MP_M%d"%(callkey[0]),"MP_ML(2)=MP_M%d"%(callkey[0])]
            mset2="\n".join(mset2lines+["\n".join(["ML(%d)=M%d"%(i,i-2),
                                               "MP_ML(%d)=MP_M%d"%(i,i-2)]) for \
                                        i in range(3,callkey[0]+2)])
            replace_dict['mset2']=mset2           
            replace_dict['nwfsargs'] = callkey[1]
            if callkey[0]==callkey[1]:
                file = open(os.path.join(self.template_dir,\
                                             'helas_loop_amplitude.inc')).read()                
            else:
                file = open(os.path.join(self.template_dir,\
                                     'helas_loop_amplitude_pairing.inc')).read() 
                pairingargs="".join([("P"+str(i)+", ") for i in range(1,callkey[0]+1)])
                replace_dict['pairingargs']=pairingargs
                pairingdecl="".join([("P"+str(i)+", ") for i in range(1,callkey[0]+1)])[:-2]
                replace_dict['pairingdecl']=pairingdecl
                pairingset="\n".join([("PAIRING("+str(i)+")=P"+str(i)) for \
                             i in range(1,callkey[0]+1)])
                replace_dict['pairingset']=pairingset
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
        
        # Decide here wether we need to split the loop_matrix.f file or not.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>2000)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'helas_calls_split.inc',file,born_ct_helas_calls,\
                            'born_ct_helas_calls','helas_calls_ampb')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'helas_calls_split.inc',file,loop_amp_helas_calls,\
                    'loop_helas_calls','helas_calls_ampl')
        else:
            replace_dict['born_ct_helas_calls']='\n'.join(born_ct_helas_calls)
            replace_dict['loop_helas_calls']='\n'.join(loop_amp_helas_calls)
        
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

    def write_born_amps_and_wfs(self, writer, matrix_element, fortran_model,\
                                noSplit=False): 
        """ Writes out the code for the subroutine MP_BORN_AMPS_AND_WFS which 
        computes just the external wavefunction and born amplitudes in 
        multiple precision. """

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        replace_dict = copy.copy(self.general_replace_dict)

        # Extract helas calls
        born_amps_and_wfs_calls = fortran_model.get_born_ct_helas_calls(\
                                               matrix_element, include_CT=False)
        
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