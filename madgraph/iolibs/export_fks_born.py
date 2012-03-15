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
import copy

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.fks.fks_born as fks
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_exporters as loop_exporters

import aloha.create_aloha as create_aloha

import models.sm.write_param_card as write_param_card
from madgraph import MadGraph5Error, MG5DIR
from madgraph.iolibs.files import cp, ln, mv
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_fks')

class ProcessExporterFortranFKS_born(loop_exporters.LoopProcessExporterFortranSA):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""
    
    def __init__(self, mgme_dir = "", dir_path = "", clean = False, \
                 loop_dir = "", cts_dir = ""):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = mgme_dir
        self.dir_path = dir_path
        self.clean = clean
        self.loop_dir = loop_dir
        self.cuttools_dir = cts_dir

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
            shutil.copytree(os.path.join(mgme_dir, 'TemplateFKSborn'), dir_path, True)
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

        # We must link the CutTools to the Library folder of the active Template
        self.link_CutTools(os.path.join(dir_path, 'lib'))

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
        
        
##    #===============================================================================
##    # write fks.inc file in SubProcesses, to be copied in the various FKS dirs    
##    #===============================================================================
##    def fks_inc_content(self,fks_proc):
##        """composes the content of the fks_inc file"""
##        
##        replace_dict = {'fksconfigs': fks_proc.ndirs}
##        
##        file = "\
##          integer fks_configs, ipos, jpos \n \
##          data fks_configs /  %(fksconfigs)d  / \n \
##          integer fks_i( %(fksconfigs)d ), fks_j( %(fksconfigs)d ) \n\
##    \n \
##          integer fks_ipos(0:nexternal) \n\
##          integer fks_j_from_i(nexternal, 0:nexternal) \n\
##          integer particle_type(nexternal), PDG_type(nexternal) \n\
##          """ % replace_dict +\
##          fks_proc.fks_config_string
##    
##        file += "\n \n \
##          data (fks_ipos(ipos),ipos=0,  %d  ) " % len(fks_proc.fks_ipos)
##        file +="\
##      / %d,  " % len(fks_proc.fks_ipos) + \
##         ', '.join([ "%d" % pdg for pdg in fks_proc.fks_ipos]) + " / "
##        
##        file += "\n\n"
##        
##        for i in fks_proc.fks_j_from_i.keys():
##            file += " data (fks_j_from_i( %d, jpos), jpos=0, %d) " %\
##          (i, len(fks_proc.fks_j_from_i[i]) )
##            file +="\
##      / %d,  " % len(fks_proc.fks_j_from_i[i]) + \
##         ', '.join([ "%d" % ii for ii in fks_proc.fks_j_from_i[i]]) + " / \n"
##          
##        file += "\n\
##C \n\
##C  Particle type: \n\
##C   octet = 8, triplet = 3, singlet = 1 \n\
##          data (particle_type(ipos), ipos=1, nexternal) \
##      / " + ', '.join([ "%d" % col for col in fks_proc.colors]) + " / "
##    
##        file += "\n \n\
##C \n\
##C  Particle type according to PDG: \n\
##C   \n\
##          data (PDG_type(ipos), ipos=1, nexternal) \
##      / " + ', '.join([ "%d" % pdg for pdg in fks_proc.pdg_codes]) + " / "
##    
##        return file
    
        
        
        
    #===============================================================================
    # generate_subprocess_directory_fks
    #===============================================================================
    def generate_real_directories_fks(self, matrix_element,
                                                  fortran_model,
                                                  me_number,
                                                  path=os.getcwd()):
        """Generate the Pxxxxx_i directories for a subprocess in MadFKS,
        including the necessary matrix.f and various helper files"""
        proc = matrix_element.born_matrix_element['processes'][0]
        
        cwd = os.getcwd()
        try:
            os.chdir(path)
        except OSError, error:
            error_msg = "The directory %s should exist in order to be able " % path + \
                        "to \"export\" in it. If you see this error message by " + \
                        "typing the command \"export\" please consider to use " + \
                        "instead the command \"output\". "
            raise MadGraph5Error, error_msg 
        
        calls = 0
        
##    #write fks.inc in SubProcesses, it willbe copied later in the subdirs
##        fks_inc = self.fks_inc_content(this_fks_process)
        self.fksdirs = []
        #first make and cd the direcrory corresponding to the born process:
        borndir = "P%s" % \
        (matrix_element.get('processes')[0].shell_string())
        os.mkdir(borndir)
        os.chdir(borndir)

## write the files corresponding to the born process in the P* directory
        self.generate_born_fks_files(matrix_element,
                fortran_model, me_number, path)

        filename = 'OLE_order.lh'
        self.write_lh_order(filename, matrix_element)
        
        if matrix_element.virt_matrix_element:
                    calls += self.generate_virt_directory( \
                            matrix_element.virt_matrix_element, \
                            fortran_model, \
                            os.path.join(path, borndir))
#write the infortions for the different real emission processes
        self.write_real_matrix_elements(matrix_element, fortran_model)
        self.write_pdf_calls(matrix_element, fortran_model)

#write the wrappers
        filename = 'real_me_chooser.f'
        self.write_real_me_wrapper(writers.FortranWriter(filename), 
                                   matrix_element, 
                                   fortran_model)

        filename = 'parton_lum_chooser.f'
        self.write_pdf_wrapper(writers.FortranWriter(filename), 
                                   matrix_element, 
                                   fortran_model)
#        for nfks, fksreal in enumerate(matrix_element.real_processes):
#                calls += self.generate_subprocess_directory_fks(nfks, fksreal,
#                                                  matrix_element, borndir,
#                                                  fortran_model,
#                                                  me_number,
#                                                  path)
            
        os.chdir(cwd)
        return calls

    def write_real_matrix_elements(self, matrix_element, fortran_model):
        """writes the matrix_i.f files which contain the real matrix elements""" 

        for n, fksreal in enumerate(matrix_element.real_processes):
            filename = 'matrix_%d.f' % (n + 1)
            self.write_matrix_element_fks(writers.FortranWriter(filename),
                                            fksreal.matrix_element, n + 1, 
                                            fortran_model)

    def write_pdf_calls(self, matrix_element, fortran_model):
        """writes the matrix_i.f files which contain the real matrix elements""" 
        for n, fksreal in enumerate(matrix_element.real_processes):
            filename = 'parton_lum_%d.f' % (n + 1)
            self.write_pdf_file(writers.FortranWriter(filename),
                                            fksreal.matrix_element, n + 1, 
                                            fortran_model)


    def write_pdf_wrapper(self, writer, matrix_element, fortran_model):
        """writes the wrapper which allows to chose among the different real matrix elements"""

        file = \
"""double precision function dlum()
double precision lum
integer nfksprocess
common/c_nfksprocess/nfksprocess
"""
        for n in range(len(matrix_element.real_processes)):
            file += \
"""if (nfksprocess.eq.%(n)d) then
lum = dlum_%(n)d()
else""" % {'n': n + 1}
        file += \
"""
write(*,*) 'ERROR: invalid n in dlum :', n
stop
endif

return lum
end
"""
        # Write the file
        writer.writelines(file)
        return 0


    def write_real_me_wrapper(self, writer, matrix_element, fortran_model):
        """writes the wrapper which allows to chose among the different real matrix elements"""

        file = \
"""subroutine smatrix(p, wgt)
include 'nexternal.inc'
double precision p(0:3, nexternal)
double precision wgt
integer nfksprocess
common/c_nfksprocess/nfksprocess
"""
        for n in range(len(matrix_element.real_processes)):
            file += \
"""if (nfksprocess.eq.%(n)d) then
call smatrix_%(n)d(p, wgt)
else""" % {'n': n + 1}
        file += \
"""
write(*,*) 'ERROR: invalid n in real_matrix :', n
stop
endif

return
end
"""
        # Write the file
        writer.writelines(file)
        return 0



    def generate_born_fks_files(self, matrix_element, fortran_model, me_number, path):
        """generates the files needed for the born applitude in the P* directory, which will
        be needed by the P* directories"""
        pathdir = os.getcwd()

        filename = 'born.f'
        calls_born, ncolor_born = \
            self.write_born_fks(writers.FortranWriter(filename),\
                             matrix_element,
                             fortran_model)

        file_dict= {True : '_inverse.inc', False :'.inc'}
        for invert in [True, False]:

            filename = 'born_conf' + file_dict[invert]
            nconfigs, s_and_t_channels = self.write_configs_file(\
                writers.FortranWriter(filename),
                matrix_element.born_matrix_element, 
                invert,
                fortran_model)

            filename = 'born_props' + file_dict[invert]
            self.write_props_file(writers.FortranWriter(filename),
                             matrix_element.born_matrix_element,
                             fortran_model,
                                s_and_t_channels)
        
            filename = 'born_decayBW' + file_dict[invert]
            self.write_decayBW_file(writers.FortranWriter(filename),
                                s_and_t_channels)
    
        filename = 'born_leshouche.inc'
        nflows = self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element.born_matrix_element,
                             fortran_model)
    
        filename = 'born_nhel.inc'
        self.write_born_nhel_file(writers.FortranWriter(filename),
                           matrix_element.born_matrix_element, nflows,
                           fortran_model,
                           ncolor_born)
    
        filename = 'born_ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                            nconfigs)
    
        
        #write the sborn_sf.f and the b_sf_files
        filename = ['sborn_sf.f', 'sborn_sf_dum.f']
        for i, links in enumerate([matrix_element.color_links, []]):
            self.write_sborn_sf(writers.FortranWriter(filename[i]),
                                                links,
                                                fortran_model)
        self.color_link_files = [] 
        for i in range(len(matrix_element.color_links)):
            filename = 'b_sf_%3.3d.f' % (i + 1)              
            self.color_link_files.append(filename)
            self.write_b_sf_fks(writers.FortranWriter(filename),
                         matrix_element, i,
                         fortran_model)


    def generate_virt_directory(self, loop_matrix_element, fortran_model, dir_name):
        """writes the V**** directory inside the P**** directories specified in
        dir_name"""

        cwd = os.getcwd()

        matrix_element = loop_matrix_element

        # Create the directory PN_xx_xxxxx in the specified path
        name = "V%s" % matrix_element.get('processes')[0].shell_string()
        dirpath = os.path.join(dir_name, name)

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0

        logger.info('Creating files in directory %s' % name)

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
#        Not ready yet
        writers.FortranWriter(filename).writelines("""C Post-helas generation loop-drawing is not ready yet.""")
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


        linkfiles = ['coupl.inc', 'cts_mprec.h', 'cts_mpc.h']

        for file in linkfiles:
            ln('../../%s' % file)

        os.system("ln -s ../../check_sa_loop.f check_sa.f")
        os.system("ln -s ../../makefile_loop makefile")

        linkfiles = ['mpmodule.mod']

        for file in linkfiles:
            ln('../../../lib/%s' % file)

        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls



    #===============================================================================
    # write_lh_order
    #===============================================================================
    #test written
    def write_lh_order(self, filename, fksborn):
        """Creates the OLE_order.lh file. This function should be edited according
        to the OLP which is used. NOW FOR NJET"""
        replace_dict = {}
        orders = fksborn.orders 
        replace_dict['mesq'] = 'CHsummed'
        replace_dict['corr'] = 'QCD'
        replace_dict['irreg'] = 'CDR'
        replace_dict['aspow'] = orders['QCD']
        replace_dict['aepow'] = orders['QED']
        replace_dict['pdgs'] = fksborn.get_lh_pdg_string()
        replace_dict['symfin'] = 'Yes'
        content = \
"#OLE_order written by MadGraph 5\n\
\n\
MatrixElementSquareType %(mesq)s\n\
CorrectionType          %(corr)s\n\
IRregularisation        %(irreg)s\n\
AlphasPower             %(aspow)d\n\
AlphaPower              %(aepow)d\n\
NJetSymmetrizeFinal     %(symfin)s\n\
\n\
# process\n\
%(pdgs)s\n\
" % replace_dict 
        
        file = open(filename, 'w')
        file.write(content)
        file.close
        return


    #===============================================================================
    # write_born_fks
    #===============================================================================
    # test written
    def write_born_fks(self, writer, fksborn, fortran_model):
        """Export a matrix element to a born.f file in MadFKS format"""

        matrix_element = fksborn.born_matrix_element
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
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
        #den_factor_line = get_den_factor_line(matrix_element)
    
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
    
        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
    
        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        # Extract glu_ij_lines
        ij_lines = self.get_ij_lines(fksborn)
        replace_dict['ij_lines'] = '\n'.join(ij_lines)

        # Extract den_factor_lines
        den_factor_lines = self.get_den_factor_lines(fksborn)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/born_fks_tilde_from_born.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor

    #===============================================================================
    # write_born_sf_fks
    #===============================================================================
    #test written
    def write_sborn_sf(self, writer, color_links, fortran_model):
        """Creates the sborn_sf.f file, containing the calls to the different 
        color linked borns"""
        
        replace_dict = {}
        nborns = len(color_links)
        ifkss = []
        iborns = []
        mms = []
        nns = [] 
        iflines = "\n"
        
        #header for the sborn_sf.f file 
        file = """subroutine sborn_sf(p_born,m,n,wgt)
          implicit none
          include "nexternal.inc"
          double precision p_born(0:3,nexternal-1),wgt
          double complex wgt1(2)
          integer m,n \n"""
    
        if nborns > 0:

            for i, c_link in enumerate(color_links):
                iborn = i+1
                
                iff = {True : 'if', False : 'elseif'}[i==0]

                m, n = c_link['link']
                
                iflines += \
                "c b_sf_%(iborn)3.3d links partons %(m)d and %(n)d \n\
                    %(iff)s (m.eq.%(m)d .and. n.eq.%(n)d) then \n\
                    call sb_sf_%(iborn)3.3d(p_born,wgt)\n\n" \
                        %{'m':m, 'n': n, 'iff': iff, 'iborn': iborn}
            
            file += iflines + \
            """else
            wgt = 0d0
            endif
            
            return
            end"""        
        elif nborns == 0:
            #write a dummy file
            file+="""
c     This is a dummy function because
c     this subdir has no soft singularities
            wgt = 0d0          
            
            return
            end"""           
        # Write the end of the file
       
        writer.writelines(file)
    
    #===============================================================================
    # write_b_sf_fks
    #===============================================================================
    #test written
    def write_b_sf_fks(self, writer, fksborn, i, fortran_model):
        """Create the b_sf_xxx.f file for the soft linked born in MadFKS format"""

        matrix_element = copy.copy(fksborn.born_matrix_element)

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        iborn = i + 1
        link = fksborn.color_links[i]
    
        replace_dict = {}
        
        replace_dict['iborn'] = iborn
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines 
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines + \
            "\nc spectators: %d %d \n" % tuple(link['link'])
        
    
        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb
    
        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines
    
        # Extract IC line
        ic_line = self.get_ic_line(matrix_element)
        replace_dict['ic_line'] = ic_line

        # Extract den_factor_lines
        den_factor_lines = self.get_den_factor_lines(fksborn)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
    
    
        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs
    
        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs
    
        # Extract ncolor
        ncolor1 = max(1, len(link['orig_basis']))
        replace_dict['ncolor1'] = ncolor1
        ncolor2 = max(1, len(link['link_basis']))
        replace_dict['ncolor2'] = ncolor2
    
        # Extract color data lines
        color_data_lines = self.get_color_data_lines_from_color_matrix(\
                                link['link_matrix'])
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)
    
    
        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
    
        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
        new_jamp_lines = []
        for line in jamp_lines:
            line = string.replace(line, 'JAMP', 'JAMP1')
            new_jamp_lines.append(line)
        replace_dict['jamp1_lines'] = '\n'.join(new_jamp_lines)
    
        
        matrix_element.set('color_basis', link['link_basis'] )
        jamp_lines = self.get_JAMP_lines(matrix_element)
        new_jamp_lines = []
        for line in jamp_lines:
            line = string.replace(line, 'JAMP', 'JAMP2')
            new_jamp_lines.append(line)
        replace_dict['jamp2_lines'] = '\n'.join(new_jamp_lines)
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/b_sf_xxx_fks_from_born.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return 0 , ncolor1
    
    


    #===============================================================================
    # write_born_nhel_file
    #===============================================================================
    #test written
    def write_born_nhel_file(self, writer, matrix_element, nflows, fortran_model, ncolor):
        """Write the born_nhel.inc file for MG4."""
    
        ncomb = matrix_element.get_helicity_combinations()
        file = "       integer    max_bhel, max_bcol \n"
        file = file + "parameter (max_bhel=%d)\nparameter(max_bcol=%d)" % \
               (ncomb, nflows)
    
        # Write the file
        writer.writelines(file)
    
        return True
    

    
    def generate_subprocess_directory_fks(self, nfks, fksreal, matrix_element,
                            born_dir, fortran_model, me_number, path=os.getcwd()):   
        """Generate the Pxxxxx_i directory for a subprocess in MadFKS,
        including the necessary matrix.f and various helper files
        matrix_element is the real emission ME, the information on the 
        reduced process are contained in fksreal
        matrix element is a FKSHelasProcessFromBorn"""      
    
        pathdir = os.getcwd()
    
        # Create the directory PN_xx_xxxxx in the specified path
        subprocdir = "R_%s_i%d_j%d" % \
        (fksreal.matrix_element.get('processes')[0].shell_string()[2:], 
         fksreal.i_fks, fksreal.j_fks)
        self.fksdirs.append(subprocdir)
    #    dirs.append(subprocdir)
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
        calls = 0


        #write the config.fks file, containing only nfks (+1)
        os.system("echo %d > config.fks" % 1 )
        
        bool_dict = {True : 'Y', False : 'N'}
        bool_dict1 = {True : 'I', False : 'E'}
        
        # write nbodyonly.fks
        os.system('echo %s > nbodyonly.fks' % \
                  bool_dict[fksreal.is_nbody_only])
        
        # write integrate.fks
        os.system('echo %s > integrate.fks' % \
                  bool_dict[fksreal.is_to_integrate])
        os.system('echo %s >> integrate.fks' % \
                  bool_dict1[fksreal.need_color_links])

        filename = 'fks.inc'
        self.write_fks_inc(writers.FortranWriter(filename),
                                            fksreal,
                                            fortran_model)


        # Create the matrix.f file, file and all inc files (as for v4)
        real_matrix_element = fksreal.matrix_element
        filename = 'matrix.f'
        calls, ncolor = \
               self.write_matrix_element_fks(writers.FortranWriter(filename),
                                            real_matrix_element, 0,
                                            fortran_model)

    #write auto_dsig (for MadFKS it contains only the parton luminosities
        filename = 'auto_dsig.f'
        self.write_pdf_file(writers.FortranWriter(filename),
                             real_matrix_element, 0,
                             fortran_model) 

        filename = 'den_factor.inc'
        self.write_den_factor_file(writers.FortranWriter(filename),
                             matrix_element.born_matrix_element,
                             real_matrix_element, 
                             fortran_model)

        filename = 'glu_ij.inc'
        self.write_glu_ij_file(writers.FortranWriter(filename),
                            fksreal.ijglu, 
                            fortran_model)

        filename = 'mirrorprocs.inc'
        self.write_mirrorprocs(writers.FortranWriter(filename),
                                            real_matrix_element)
        

        filename = 'coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                             real_matrix_element,
                             fortran_model)
    
        filename = 'configs.inc'
        nconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            real_matrix_element,
            fksreal.j_fks == 2,
            fortran_model)


        filename = 'bornfromreal.inc'
        self.write_bornfromreal_file(writers.FortranWriter(filename),
                             fksreal,
                             fortran_model)
    
        filename = 'props.inc'
        self.write_props_file(writers.FortranWriter(filename),
                         real_matrix_element,
                         fortran_model,
                            s_and_t_channels)
    
        filename = 'decayBW.inc'
        self.write_decayBW_file(writers.FortranWriter(filename),
                            s_and_t_channels)
    
        filename = 'dname.mg'
        self.write_dname_file(writers.FortranWriter(filename),
                         real_matrix_element,
                         fortran_model)
    
        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                         me_number)
    
        filename = 'leshouche.inc'
        self.write_leshouche_file(writers.FortranWriter(filename),
                             real_matrix_element,
                             fortran_model)
    
        filename = 'maxamps.inc'
        self.write_maxamps_file(writers.FortranWriter(filename),
                           real_matrix_element,
                           fortran_model,
                           ncolor)
    
        filename = 'mg.sym'
        self.write_mg_sym_file(writers.FortranWriter(filename),
                          real_matrix_element,
                          fortran_model)
    
        filename = 'ncombs.inc'
        self.write_ncombs_file(writers.FortranWriter(filename),
                          real_matrix_element,
                          fortran_model)
    
        filename = 'nexternal.inc'
        (nexternal, ninitial) = real_matrix_element.get_nexternal_ninitial()
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)
    
        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                            nconfigs)
    
        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                             real_matrix_element)

        linkfiles = ['LesHouchesDummy.f',
                     'MCmasses_HERWIG6.inc',
                     'MCmasses_HERWIGPP.inc',
                     'MCmasses_PYTHIA6Q.inc',
                     'add_write_info.f',
                     'check_dip.f',
                     'cluster.f',
                     'cluster.inc',
                     'coupl.inc',
                     'cuts.f',
                     'cuts.inc',
                     'dbook.inc',
                     'driver_mint.f',
                     'driver_mintMC.f',
                     'driver_vegas.f',
                     'fastjetfortran_madfks.cc',
                     'fks_Sij.f',
                     'fks_nonsingular.f',
                     'fks_powers.inc',
                     'fks_singular.f',
                     'genps.f',
                     'genps.inc',
                     'genps_fks.f',
                     'initcluster.f',
                     'ktclusdble.f',
                     'link_fks.f',
                     'madfks_dbook.f',
                     'madfks_mcatnlo.inc',
                     'madfks_plot.f',
                     'message.inc',
                     'mint-integrator2.f',
                     'mint.inc',
                     'mirror.f',
                     'montecarlocounter.f',
                     'myamp.f',
                     'q_es.inc',
                     'reweight.f',
                     'reweight.inc',
                     'reweight0.inc',
                     'reweight1.inc',
                     'reweight_events.f',
                     'reweight_xsec.f',
                     'run.inc',
                     'setcuts.f',
                     'setscales.f',
                     'sudakov.inc',
                     'symmetry.f',
                     'symmetry_fks_test_MC.f',
                     'symmetry_fks_test_ME.f',
                     'symmetry_fks_test_Sij.f',
                     'symmetry_fks_v3.f',
                     'trapfpe.c',
                     'unwgt.f',
                     'vegas2.for',
                     'write_ajob.f',
                     'write_event.f']

        for file in linkfiles:
            ln('../../' + file , '.')


        #copy the makefile 
        os.system("ln -s ../../makefile_fks_dir ./makefile")

        cpfiles = [ 'props.inc', 'configs.inc']
        for file in cpfiles:
            os.system('cp '+file+' '+file+'.back')
        
        #import nexternal/leshouches in Source
        ln('nexternal.inc', '../../../Source', log=False)
        ln('leshouche.inc', '../../../Source', log=False)

        if fksreal.ijglu > 0:
            os.system('ln -s ../born_intf.f ./born.f')
        else:
            os.system('ln -s ../born.f ./born.f')

        file_dict= {True : '_inverse.inc', False :'.inc'}

        linkfiles_born_invert = \
                         ['born_conf',
                          'born_decayBW',
                          'born_props']

        for file in linkfiles_born_invert:
            os.system('ln -s ../%s ./%s' \
                    % (file + file_dict[fksreal.j_fks == 2],
                       file + file_dict[False]))

        linkfiles_born = \
                         ['born_leshouche.inc',
                          'born_ngraphs.inc',
                          'born_nhel.inc',
                          'OLE_order.lh']


        for file in linkfiles_born:
            ln('../' + file , '.')

        if fksreal.need_color_links:
            for file in self.color_link_files:
                os.system('ln -s ../%s ./%s' % (file, file))
            os.system('ln -s ../sborn_sf.f ./sborn_sf.f')
            os.system('ln -s ../../LesHouches.f ./LesHouches.f')
        else:
            os.system('ln -s ../sborn_sf_dum.f ./sborn_sf.f')
            os.system('ln -s ../../LesHouchesDummy.f ./LesHouches.f')


    

        # Return to original dir
        os.chdir(pathdir)
    
        if not calls:
            calls = 0
        return calls
            

    #===============================================================================
    # write_fks_inc
    #===============================================================================
    def write_fks_inc(self, writer, me, fortran_model): #test_written
        """Writes the content of fks.inc, which lists the various fks configs
        of a FKSHelasProcessFromReal, plus some extra additional ifos"""

        replace_dict = {}
        replace_dict['nconfs'] = 1
        replace_dict['confs_lines'] = self.get_fks_conf_lines(me)
        replace_dict['fks_j_from_i_lines'] = self.get_fks_j_from_i_lines(me)
        replace_dict['pdg_string'] = ", ".join("%d" % leg['id'] \
                for leg in me.matrix_element.get('processes')[0]['legs'])
        replace_dict['col_string'] = ", ".join("%d" % col for col in me.colors)

        content = \
"""      INTEGER FKS_CONFIGS, IPOS, JPOS
      DATA FKS_CONFIGS / %(nconfs)d /
      INTEGER FKS_I(%(nconfs)d), FKS_J(%(nconfs)d)
      INTEGER FKS_J_FROM_I(NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE(NEXTERNAL), PDG_TYPE(NEXTERNAL)

%(confs_lines)s
%(fks_j_from_i_lines)sC     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE(IPOS), IPOS=1, NEXTERNAL) / %(col_string)s /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE(IPOS), IPOS=1, NEXTERNAL) / %(pdg_string)s /
"""   % replace_dict

        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        writer.writelines(content)
    
        return True

    def get_fks_conf_lines(self, me): #test written
        """generate the lines for fks.inc describing the various fks configs"""
        n = 0
        lines = ""
        if not me.isfinite:
            n += 1
            lines += 'c     FKS configuration number  %d\n' % n \
                   + 'DATA FKS_I(%d) / %d /\n' % (n, me.i_fks)\
                   + 'DATA FKS_J(%d) / %d /\n' % (n, me.j_fks)
        else:
            lines += 'c     FKS configuration number  %d\n' % 1 \
                   + 'DATA FKS_I(%d) / %d /\n' % (1,2)\
                   + 'DATA FKS_J(%d) / %d /\n' % (1,1)

        return lines


    def get_fks_j_from_i_lines(self, me): #test written
        """generate the lines for fks.inc describing initializating the
        fks_j_from_i array"""
        lines = ""
        if not me.isfinite:
            for ii, js in me.fks_j_from_i.items():
                if js:
                    lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                             % (ii, len(js), len(js), ', '.join(["%d" % j for j in js]))
        else:
            lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                     % (2, 1, 1, '1')

        return lines

 
    #===============================================================================
    # write_matrix_element_fks
    #===============================================================================
    #test written
    def write_matrix_element_fks(self, writer, matrix_element, n, fortran_model):
        """Export a matrix element to a matrix.f file in MG4 madevent format"""
    
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0,0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
    
        replace_dict = {}
        replace_dict['N_me'] = n
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines
    
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
    
        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
    
        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
    
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/realmatrix_fks_born.inc')).read()

        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor


    #===============================================================================
    # write_auto_dsig_file
    #===============================================================================
    def write_pdf_file(self, writer, matrix_element, n, fortran_model):
        #test written
        """Write the auto_dsig.f file for MadFKS, which contains 
          pdf call information"""
    
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
    
        if ninitial < 1 or ninitial > 2:
            raise writers.FortranWriter.FortranWriterError, \
                  """Need ninitial = 1 or 2 to write auto_dsig file"""
    
        replace_dict = {}

        replace_dict['N_me'] = n
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines
    
        pdf_lines = self.get_pdf_lines_mir(matrix_element, ninitial, False, False)
        replace_dict['pdf_lines'] = pdf_lines

        pdf_lines_mirr = self.get_pdf_lines_mir(matrix_element, ninitial, False, True)
        replace_dict['pdf_lines_mirr'] = pdf_lines_mirr
    
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/parton_lum_n_fks.inc')).read()
        file = file % replace_dict
    
        # Write the file
        writer.writelines(file)


    #===============================================================================
    # get_den_factor_lines
    #===============================================================================
    def get_den_factor_lines(self, fks_born):
        """returns the lines with the information on the particle number of the born 
        that splits"""
    
        lines = []
        lines.append('integer iden_values(%d)' % len(fks_born.real_processes))
        lines.append('data iden_values /' + \
                     ', '.join(['%d' % ( 
                     fks_born.born_matrix_element.get_denominator_factor() / \
                     fks_born.born_matrix_element['identical_particle_factor'] * \
                     real.matrix_element['identical_particle_factor'] ) \
                     for real in fks_born.real_processes]) + '/')

        return lines

    #===============================================================================
    # write_den_factor_file
    #===============================================================================
    #test_written
    def write_den_factor_file(self, writer, matrix_element, real_matrix_element, fortran_model):
        """Write the den_factor.inc file to be included in the born ME, with identical
        particle factor as the real ME"""
    
        lines = ["DATA IDEN/%2r/" % \
               (matrix_element.get_denominator_factor()/\
               matrix_element['identical_particle_factor'] *\
               real_matrix_element['identical_particle_factor'])]

        # Write the file
        writer.writelines(lines)



    #===============================================================================
    # get_ij_lines
    #===============================================================================
    def get_ij_lines(self, fks_born):
        """returns the lines with the information on the particle number of the born 
        that splits"""
    
        lines = []
        lines.append('integer ij_values(%d)' % len(fks_born.real_processes))
        lines.append('data ij_values /' + \
                     ', '.join(['%d' % real.ij for real in fks_born.real_processes]) + '/')

        return lines


    #===============================================================================
    # write_glu_ij_file
    #===============================================================================
    def write_glu_ij_file(self, writer, glu_ij, fortran_model):
        """Write the glu_ij.inc file to be included in the born ME when the leg that splits
        is a gluon"""
    
        lines = ["DATA GLU_IJ/%2r/" % glu_ij]

        # Write the file
        writer.writelines(lines)

    #===============================================================================
    # write_coloramps_file
    #===============================================================================
    #test written
    def write_coloramps_file(self, writer, matrix_element, fortran_model):
        """Write the coloramps.inc file for MadEvent"""
    
        lines = self.get_icolamp_lines(matrix_element)
    
        # Write the file
        writer.writelines(lines)
    
        return True


    #===============================================================================
    # write_mirrorprocs
    #===============================================================================   
    #test_written
    def write_mirrorprocs(self, writer, matrix_element):
        """writes the content of the mirrorprocs.inc file"""
        bool_dict = {True: ".true.", False: ".false."}
        content = \
"logical mirrorproc \n\
data mirrorproc /%s/" % bool_dict[matrix_element.get('has_mirror_process')]
        writer.writelines(content)
        return True


    #===============================================================================
    # write_bornfromreal_file
    #===============================================================================
    def write_bornfromreal_file(self, writer, fksrealproc, fortran_model):
    # test written
        """Write the bornfromreal.inc file, with informations on how to link born
        and real diagrams"""

        lines = []
        b_confs = sorted([l['born_conf']+1 for l in fksrealproc.bornfromreal]) 

        for link in fksrealproc.bornfromreal:
            lines.append('data b_from_r(%d) / %d /' % \
                         (link['real_conf']+1, link['born_conf']+1) )
            lines.append('data r_from_b(%d) / %d /' % \
                         (link['born_conf']+1, link['real_conf']+1) )
        lines.append('integer mapb')
        lines.append('data (mapbconf(mapb), mapb=0, %d) / %d, %s /' % \
                     ( len(fksrealproc.bornfromreal), 
                       len(fksrealproc.bornfromreal),
                       ', '.join(['%d' % c for c in b_confs])))

        # Write the file
        writer.writelines(lines)

    
    #===============================================================================
    # write_configs_file
    #===============================================================================
    #test_written
    def write_configs_file(self, writer, matrix_element, reverse_t_ch, fortran_model):
        """Write the configs.inc file for MadEvent"""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        lines = []
    
        iconfig = 0
    
        s_and_t_channels = []


        model = matrix_element.get('processes')[0].get('model')
#        new_pdg = model.get_first_non_pdg()
    
        base_diagrams = matrix_element.get('base_amplitude').get('diagrams')
        minvert = min([max([len(vert.get('legs')) for vert in \
                            diag.get('vertices')]) for diag in base_diagrams])
    
        for idiag, diag in enumerate(base_diagrams):
            if any([len(vert.get('legs')) > minvert for vert in
                    diag.get('vertices')]):
                # Only 3-vertices allowed in configs.inc
                continue
            iconfig = iconfig + 1
            helas_diag = matrix_element.get('diagrams')[idiag]
            lines.append("# Diagram %d" % \
                         (helas_diag.get('number')))
            # Correspondance between the config and the amplitudes
            lines.append("data mapconfig(%4d)/%4d/" % (iconfig,
                                                     helas_diag.get('number')))
    
            # Need to reorganize the topology so that we start with all
            # final state external particles and work our way inwards
            schannels, tchannels = helas_diag.get('amplitudes')[0].\
                                         get_s_and_t_channels(ninitial, 990, reverse_t_ch)
    
            s_and_t_channels.append([schannels, tchannels])
    
            # Write out propagators for s-channel and t-channel vertices
            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels
    
            for vert in allchannels:
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%3d,%4d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), iconfig, len(daughters),
                              ",".join(["%3d" % d for d in daughters])))
                if vert in schannels:
                    lines.append("data sprop(%4d,%4d)/%8d/" % \
                                 (last_leg.get('number'), iconfig,
                                  last_leg.get('id')))
                elif vert in tchannels[:-1]:
                    lines.append("data tprid(%4d,%4d)/%8d/" % \
                                 (last_leg.get('number'), iconfig,
                                  abs(last_leg.get('id'))))
    
        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%4d/" % iconfig)
    
        # Write the file
        writer.writelines(lines)
    
        return iconfig, s_and_t_channels
    
    #===============================================================================
    # write_decayBW_file
    #===============================================================================
    #test written
    def write_decayBW_file(self, writer, s_and_t_channels):
        """Write the decayBW.inc file for MadEvent"""

        lines = []

        booldict = {False: ".false.", True: ".false."}
        ####Changed by MZ 2011-11-23!!!!

        for iconf, config in enumerate(s_and_t_channels):
            schannels = config[0]
            for vertex in schannels:
                # For the resulting leg, pick out whether it comes from
                # decay or not, as given by the from_group flag
                leg = vertex.get('legs')[-1]
                lines.append("data gForceBW(%d,%d)/%s/" % \
                             (leg.get('number'), iconf + 1,
                              booldict[leg.get('from_group')]))

        # Write the file
        writer.writelines(lines)

        return True
    
    #===============================================================================
    # write_dname_file
    #===============================================================================
    def write_dname_file(self, writer, matrix_element, fortran_model):
        """Write the dname.mg file for MG4"""
    
        line = "DIRNAME=P%s" % \
               matrix_element.get('processes')[0].shell_string()
    
        # Write the file
        writer.write(line + "\n")
    
        return True
    
    #===============================================================================
    # write_iproc_file
    #===============================================================================
    def write_iproc_file(self, writer, me_number):
        """Write the iproc.dat file for MG4"""
    
        line = "%d" % (me_number + 1)
    
        # Write the file
        for line_to_write in writer.write_line(line):
            writer.write(line_to_write)
        return True
    
    #===============================================================================
    # write_leshouche_file
    #===============================================================================
    def write_leshouche_file(self, writer, matrix_element, fortran_model):
        #test written
        """Write the leshouche.inc file for MG4"""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    
        lines = []
        for iproc, proc in enumerate(matrix_element.get('processes')):
            legs = proc.get_legs_with_decays()
            lines.append("DATA (IDUP(i,%d),i=1,%d)/%s/" % \
                         (iproc + 1, nexternal,
                          ",".join([str(l.get('id')) for l in legs])))
            for i in [1, 2]:
                lines.append("DATA (MOTHUP(%d,i,%3r),i=1,%2r)/%s/" % \
                         (i, iproc + 1, nexternal,
                          ",".join([ "%3r" % 0 ] * ninitial + \
                                   [ "%3r" % i ] * (nexternal - ninitial))))
    
            # Here goes the color connections corresponding to the JAMPs
            # Only one output, for the first subproc!
            if iproc == 0:
                # If no color basis, just output trivial color flow
                if not matrix_element.get('color_basis'):
                    for i in [1, 2]:
                        lines.append("DATA (ICOLUP(%d,i,  1),i=1,%2r)/%s/" % \
                                 (i, nexternal,
                                  ",".join([ "%3r" % 0 ] * nexternal)))
                    color_flow_list = []
    
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
                            lines.append("DATA (ICOLUP(%d,i,%3r),i=1,%2r)/%s/" % \
                                 (i + 1, cf_i + 1, nexternal,
                                  ",".join(["%3r" % color_flow_dict[l.get('number')][i] \
                                            for l in legs])))
    
        # Write the file
        writer.writelines(lines)
    
        return len(color_flow_list)
    


    #===============================================================================
    # write_maxamps_file
    #===============================================================================
    def write_maxamps_file(self, writer, matrix_element, fortran_model, ncolor):
        #test written
        """Write the maxamps.inc file for MG4."""
    
        file = "       integer    maxamps, maxflow\n"
        file = file + "parameter (maxamps=%d, maxflow=%d)" % \
               (len(matrix_element.get_all_amplitudes()), ncolor)
    
        # Write the file
        writer.writelines(file)
    
        return True
    
    #===============================================================================
    # write_mg_sym_file
    #===============================================================================
    #test written
    def write_mg_sym_file(self, writer, matrix_element, fortran_model):
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


    #===============================================================================
    # Helper functions
    #===============================================================================


    def get_fks_j_from_i_lines(self, me): #test written
        """generate the lines for fks.inc describing initializating the
        fks_j_from_i array"""
        lines = ""
        if not me.isfinite:
            for ii, js in me.fks_j_from_i.items():
                if js:
                    lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                             % (ii, len(js), len(js), ', '.join(["%d" % j for j in js]))
        else:
            lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                     % (2, 1, 1, '1')

        return lines



    def get_fks_j_from_i_lines(self, me): #test written
        """generate the lines for fks.inc describing initializating the
        fks_j_from_i array"""
        lines = ""
        if not me.isfinite:
            for ii, js in me.fks_j_from_i.items():
                if js:
                    lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                             % (ii, len(js), len(js), ', '.join(["%d" % j for j in js]))
        else:
            lines += 'DATA (FKS_J_FROM_I(%d, JPOS), JPOS = 0, %d)  / %d, %s /\n' \
                     % (2, 1, 1, '1')

        return lines


    def get_pdf_lines_mir(self, matrix_element, ninitial, subproc_group = False,\
                          mirror = False): #test written
        """Generate the PDF lines for the auto_dsig.f file"""

        processes = matrix_element.get('processes')

        pdf_lines = ""

        if ninitial == 1:
            pdf_lines = "PD(0) = 0d0\nIPROC = 0\n"
            for i, proc in enumerate(processes):
                process_line = proc.base_string()
                pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                pdf_lines = pdf_lines + "\nPD(IPROC)=PD(IPROC-1) + 1d0\n"
        else:
            # Set notation for the variables used for different particles
            pdf_codes = {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b',
                         21: 'g', 22: 'a'}
            # Set conversion from PDG code to number used in PDF calls
            pdgtopdf = {21: 0, 22: 7}
            # Fill in missing entries
            for key in pdf_codes.keys():
                if key < 21:
                    pdf_codes[-key] = pdf_codes[key] + 'b'
                    pdgtopdf[key] = key
                    pdgtopdf[-key] = -key

            # Pick out all initial state particles for the two beams
            initial_states = [sorted(list(set([p.get_initial_pdg(1) for \
                                               p in processes]))),
                              sorted(list(set([p.get_initial_pdg(2) for \
                                               p in processes])))]

            # Get PDF values for the different initial states
            for i, init_states in enumerate(initial_states):
                if not mirror:
                    ibeam = i + 1
                else:
                    ibeam = 2 - i
                if subproc_group:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(IB(%d))).GE.1) THEN\nLP=SIGN(1,LPP(IB(%d)))\n" \
                                 % (ibeam, ibeam)
                else:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(%d)) .GE. 1) THEN\nLP=SIGN(1,LPP(%d))\n" \
                                 % (ibeam, ibeam)

                for initial_state in init_states:
                    if initial_state in pdf_codes.keys():
                        if subproc_group:
                            pdf_lines = pdf_lines + \
                                        ("%s%d=PDG2PDF(ABS(LPP(IB(%d))),%d*LP," + \
                                         "XBK(IB(%d)),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, ibeam, pdgtopdf[initial_state],
                                          ibeam, ibeam)
                        else:
                            pdf_lines = pdf_lines + \
                                        ("%s%d=PDG2PDF(ABS(LPP(%d)),%d*LP," + \
                                         "XBK(%d),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, ibeam, pdgtopdf[initial_state],
                                          ibeam, ibeam)
                pdf_lines = pdf_lines + "ENDIF\n"

            # Add up PDFs for the different initial state particles
            pdf_lines = pdf_lines + "PD(0) = 0d0\nIPROC = 0\n"
            for proc in processes:
                process_line = proc.base_string()
                pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                pdf_lines = pdf_lines + "\nPD(IPROC)=PD(IPROC-1) + "
                for ibeam in [1, 2]:
                    initial_state = proc.get_initial_pdg(ibeam)
                    if initial_state in pdf_codes.keys():
                        pdf_lines = pdf_lines + "%s%d*" % \
                                    (pdf_codes[initial_state], ibeam)
                    else:
                        pdf_lines = pdf_lines + "1d0*"
                # Remove last "*" from pdf_lines
                pdf_lines = pdf_lines[:-1] + "\n"

        # Remove last line break from pdf_lines
        return pdf_lines[:-1]


    #test written
    def get_color_data_lines_from_color_matrix(self, color_matrix, n=6):
        """Return the color matrix definition lines for the given color_matrix. Split
        rows in chunks of size n."""
    
        if not color_matrix:
            return ["DATA Denom(1)/1/", "DATA (CF(i,1),i=1,1) /1/"]
        else:
            ret_list = []
            my_cs = color.ColorString()
            for index, denominator in \
                enumerate(color_matrix.get_line_denominators()):
                # First write the common denominator for this color matrix line
                ret_list.append("DATA Denom(%i)/%i/" % (index + 1, denominator))
                # Then write the numerators for the matrix elements
                num_list = color_matrix.get_line_numerators(index, denominator)    
                for k in xrange(0, len(num_list), n):
                    ret_list.append("DATA (CF(i,%3r),i=%3r,%3r) /%s/" % \
                                    (index + 1, k + 1, min(k + n, len(num_list)),
                                     ','.join(["%5r" % i for i in num_list[k:k + n]])))

            return ret_list

    
    def get_den_factor_line(self, matrix_element):
        #test written
        """Return the denominator factor line for this matrix element,
        corrected with the final state symmetry factor of real_matrix_element
        (if given)"""
     
        return "DATA IDEN/%2r/" % \
               matrix_element.get_denominator_factor()
    
    
    def get_icolamp_lines(self, matrix_element): 
        #test written
        """Return the ICOLAMP matrix, showing which AMPs are parts of
        which JAMPs."""
    
        ret_list = []
    
        booldict = {False: ".false.", True: ".true."}
    
        amplitudes = matrix_element.get_all_amplitudes()
    
        color_amplitudes = matrix_element.get_color_amplitudes()
    
        ret_list.append("logical icolamp(%d,%d)" % \
                        (len(amplitudes), len(color_amplitudes)))
    
        for icolor, coeff_list in enumerate(color_amplitudes):
    
            # List of amplitude numbers used in this JAMP
            amp_list = [amp_number for (dummy, amp_number) in coeff_list]
    
            # List of True or False 
            bool_list = [(i + 1 in amp_list) for i in \
                              range(len(amplitudes))]
            # Add line
            ret_list.append("DATA(icolamp(i,%d),i=1,%d)/%s/" % \
                                (icolor + 1, len(bool_list),
                                 ','.join(["%s" % booldict[i] for i in \
                                           bool_list])))
        return ret_list
    

    
        # Write the file
        writer.writelines(lines)
    
        return True
    
    #===============================================================================
    # write_ncombs_file
    #===============================================================================
    def write_ncombs_file(self, writer, matrix_element, fortran_model):
#        #test written
        """Write the ncombs.inc file for MadEvent."""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    
        # ncomb (used for clustering) is 2^(nexternal + 1)
        file = "       integer    n_max_cl\n"
        file = file + "parameter (n_max_cl=%d)" % (2 ** (nexternal + 1))
    
        # Write the file
        writer.writelines(file)
   
        return True
    
    
    #===============================================================================
    # write_props_file
    #===============================================================================
    #test_written
    def write_props_file(self, writer, matrix_element, fortran_model, s_and_t_channels):
        """Write the props.inc file for MadEvent. Needs input from
        write_configs_file. With respect to the parent routine, it has some 
        more specific formats that allow the props.inc file to be read by the 
        link program"""
    
        lines = []
    
        particle_dict = matrix_element.get('processes')[0].get('model').\
                        get('particle_dict')
    
        for iconf, configs in enumerate(s_and_t_channels):
            for vertex in configs[0] + configs[1][:-1]:
                leg = vertex.get('legs')[-1]
                if leg.get('id') == 21 and 21 not in particle_dict:
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
    
                lines.append("pmass(%3d,%4d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines.append("pwidth(%3d,%4d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines.append("pow(%3d,%4d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))
    
        # Write the file
        writer.writelines(lines)
    
        return True
