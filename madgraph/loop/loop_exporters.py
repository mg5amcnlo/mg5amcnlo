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
        # The born matrix element, if needed
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

        linkfiles = ['check_sa.f', 'coupl.inc', 'makefile', \
                     'cts_mprec.h', 'cts_mpc.h']

        for file in linkfiles:
            ln('../%s' % file)

        linkfiles = ['mpmodule.mod']

        for file in linkfiles:
            ln('../../lib/%s' % file)

        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

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
        
        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(replace_dict_orig)
        file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/CT_interface.inc')).read()
        
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
        #       The color info is kept together in a big array initialized
        #       through data statements in include files
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
        
        # We assume here that all processes must share the same property of 
        # having a born or not, which must be true anyway since these are two
        # definite different classes of processes which can never be treated on
        # the same footing.
        if not matrix_element.get('processes')[0].get('has_born'):
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
