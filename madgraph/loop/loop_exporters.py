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
import madgraph.iolibs.misc as misc
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

    def __init__(self, loop_dir = "", *args, **kwargs):
        """Initiate the LoopExporterFortran with directory information on where
        to find all the loop-related source files, like CutTools"""
        self.loop_dir = loop_dir
        super(LoopExporterFortran,self).__init__(*args, **kwargs)
        
    def copy_CutTools(self, targetPath):
        """copy the CutTools source directory inside the target path given
        in argument"""
        
        CutTools_path=os.path.join(self.loop_dir, 'CutTools')
        if not os.path.isdir(os.path.join(targetPath, 'CUTTOOLS')):
            if os.path.isdir(CutTools_path):
                shutil.copytree(CutTools_path,os.path.join(targetPath,'CUTTOOLS')\
                                , True)
            else:
                raise MadGraph5Error, \
                      "No valid CutTools path given for processing loops."

    #===========================================================================
    # write the multiple-precision header files
    #===========================================================================
    def write_mp_files(self, writer_mprec, writer_mpc):
        """Write the cts_mprec.h and cts_mpc.h"""

        file = open(os.path.join(self.loop_dir, 'CutTools/src/cts/cts_mprec.h')).read()
        writer_mprec.writelines(file)

        file = open(os.path.join(self.loop_dir, 'CutTools/src/cts/cts_mpc.h')).read()
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

    def copy_v4template(self):
        """Additional actions needed for setup of Template
        """
        
        super(LoopProcessExporterFortranSA, self).copy_v4template()
        
        # We must copy the CutTools to the Source folder of the active Template
        super(LoopProcessExporterFortranSA, self).copy_CutTools(
                                    os.path.join(self.dir_path, 'Source'))
        
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
    def compiler_choice(self):
        """ Different daughter classes might want different compilers.
        Here, the gfortran compiler is used throughout the compilation 
        (mandatory for CutTools written in f90) """
        
        self.set_compiler('gfortran',True)

    #===========================================================================
    # Make the CutTools directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS and MODEL directories, to set up
        everything for running standalone
        """
        
        super(LoopProcessExporterFortranSA, self).make()
        
        source_dir = os.path.join(self.dir_path, "Source")
        logger.info("Running make for CutTools")
        misc.compile(arg=['../../lib/libcts.a'], cwd=source_dir, mode='fortran')
#        shutil.copy(os.path.join(self.dir_path,\
#            'Source/CUTTOOLS/includects/libcts.a'),\
#            os.path.join(self.dir_path,'lib/libcts.a'))
#        # Copy here the two f90 modules for multiple-precision in CutTools.
#        shutil.copy(os.path.join(self.dir_path,\
#            'Source/CUTTOOLS/includects/mpmodule.mod'),\
#            os.path.join(self.dir_path,'lib/mpmodule.mod'))
#        shutil.copy(os.path.join(self.dir_path,\
#            'Source/CUTTOOLS/includects/ddmodule.mod'),\
#            os.path.join(self.dir_path,'lib/ddmodule.mod'))

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    def generate_loop_subprocess(self, matrix_element,
                                         fortran_model):
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

#        Not ready yet
        # Generate diagrams
#        filename = "loop_matrix.ps"
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

        linkfiles = ['mpmodule.mod','ddmodule.mod']

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
            mset="\n".join([("M2L("+str(i)+")=M"+str(i)+"**2") for \
                             i in range(1,callkey[0]+1)])
            replace_dict['mset']=mset
            cplset="\n".join([("LC("+str(i)+")=C"+str(i)) for \
                             i in range(1,callkey[2]+1)])
            replace_dict['cplset']=cplset            
            msetlines=["ML(1)=M%d"%(callkey[0]),"ML(2)=M%d"%(callkey[0])]
            mset2="\n".join(msetlines+[("ML("+str(i)+")=M"+str(i-2)) for \
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
        
    def write_loopmatrix(self, writer, matrix_element, fortran_model):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

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

#        Not ready yet
        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        replace_dict['helas_calls'] = "\n".join(helas_calls)
#        replace_dict['helas_calls'] = "\n".join("Not ready yet")

        # Extract BORNJAMP lines
        born_jamp_lines = self.get_JAMP_lines(matrix_element.get_born_color_amplitudes(),
                                              "JAMPB(","AMP(")
        replace_dict['born_jamp_lines'] = '\n'.join(born_jamp_lines)
        # Extract LOOPJAMP lines
        loop_jamp_lines = self.get_JAMP_lines(matrix_element.get_loop_color_amplitudes(),
                                              "JAMPL(K,","AMPL(K,")
        replace_dict['loop_jamp_lines'] = '\n'.join(loop_jamp_lines)        

        file = open(os.path.join(_file_path, \
                 'iolibs/template_files/loop/loop_matrix_standalone.inc')).read()
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