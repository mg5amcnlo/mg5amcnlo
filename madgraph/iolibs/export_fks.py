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
"""Methods and classes to export matrix elements to fks format."""

from distutils import dir_util
import glob
import logging
import os
import re
import shutil
import subprocess
import string
import copy
import platform

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.base_objects as base_objects
import madgraph.fks.fks_helas_objects as fks_helas_objects
import madgraph.fks.fks_base as fks
import madgraph.fks.fks_common as fks_common
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_exporters as loop_exporters
import madgraph.various.q_polynomial as q_polynomial
import madgraph.various.banner as banner_mod

import aloha.create_aloha as create_aloha

import models.write_param_card as write_param_card
import models.check_param_card as check_param_card
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
from madgraph.iolibs.files import cp, ln, mv

pjoin = os.path.join

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_fks')


def make_jpeg_async(args):
    Pdir = args[0]
    old_pos = args[1]
    dir_path = args[2]
  
    devnull = os.open(os.devnull, os.O_RDWR)
  
    os.chdir(Pdir)
    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'internal', 'gen_jpeg-pl')],
                    stdout = devnull)
    os.chdir(os.path.pardir)  


#=================================================================================
# Class for used of the (non-optimized) Loop process
#=================================================================================
class ProcessExporterFortranFKS(loop_exporters.LoopProcessExporterFortranSA):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

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
        clean =self.opt['clean']
        
        
        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(dir_path):
            if not mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(dir_path))
            shutil.copytree(os.path.join(mgme_dir, 'Template', 'NLO'), dir_path, True)
            # distutils.dir_util.copy_tree since dir_path already exists
            dir_util.copy_tree(pjoin(self.mgme_dir, 'Template', 'Common'),
                               dir_path)
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
                subprocess.call([os.path.join('bin', 'internal', 'clean_template'), 
                                    '--web'],cwd=dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'internal', 'clean_template')], \
                                                                       cwd=dir_path)
                except Exception, why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(dir_path),why))
            #Write version info
            MG_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])

        # We must link the CutTools to the Library folder of the active Template
        self.link_CutTools(dir_path)
        
        link_tir_libs=[]
        tir_libs=[]
        os.remove(os.path.join(self.dir_path,'SubProcesses','makefile_loop.inc'))
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        filename = pjoin(self.dir_path, 'SubProcesses','makefile_loop')
        calls = self.write_makefile_TIR(writers.MakefileWriter(filename),
                                                        link_tir_libs,tir_libs)
        os.remove(os.path.join(self.dir_path,'Source','make_opts.inc'))
        filename = pjoin(self.dir_path, 'Source','make_opts')
        calls = self.write_make_opts(writers.MakefileWriter(filename),
                                                        link_tir_libs,tir_libs)
        
        # Duplicate run_card and FO_analyse_card
        for card in ['FO_analyse_card', 'shower_card']:
            try:
                shutil.copy(pjoin(self.dir_path, 'Cards',
                                         card + '.dat'),
                           pjoin(self.dir_path, 'Cards',
                                        card + '_default.dat'))
            except IOError:
                logger.warning("Failed to copy " + card + ".dat to default")

        cwd = os.getcwd()
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0

        # We add here the user-friendly MadLoop option setter.
        cpfiles= ["SubProcesses/MadLoopParamReader.f",
                  "Cards/MadLoopParams.dat",
                  "SubProcesses/MadLoopParams.inc"]
        
        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))
        
        shutil.copy(pjoin(self.dir_path, 'Cards','MadLoopParams.dat'),
                    pjoin(self.dir_path, 'Cards','MadLoopParams_default.dat'))

        if os.path.exists(pjoin(self.dir_path, 'Cards', 'MadLoopParams.dat')):          
                self.MadLoopparam = banner_mod.MadLoopParam(pjoin(self.dir_path, 
                                                  'Cards', 'MadLoopParams.dat'))
                # write the output file
                self.MadLoopparam.write(pjoin(self.dir_path,"SubProcesses",
                                                           "MadLoopParams.dat"))
                                       
        # We need minimal editing of MadLoopCommons.f
        MadLoopCommon = open(os.path.join(self.loop_dir,'StandAlone', 
                                    "SubProcesses","MadLoopCommons.inc")).read()
        writer = writers.FortranWriter(os.path.join(self.dir_path, 
                                             "SubProcesses","MadLoopCommons.f"))
        writer.writelines(MadLoopCommon%{
                                   'print_banner_commands':self.MadLoop_banner})
        writer.close()
                                       
        # Write the cts_mpc.h and cts_mprec.h files imported from CutTools
        self.write_mp_files(writers.FortranWriter('cts_mprec.h'),\
                                           writers.FortranWriter('cts_mpc.h'))

        
        # Finally make sure to turn off MC over Hel for the default mode.
        FKS_card_path = pjoin(self.dir_path,'Cards','FKS_params.dat')
        FKS_card_file = open(FKS_card_path,'r')
        FKS_card = FKS_card_file.read()
        FKS_card_file.close()
        FKS_card = re.sub(r"#NHelForMCoverHels\n-?\d+",
                                             "#NHelForMCoverHels\n-1", FKS_card)
        FKS_card_file = open(FKS_card_path,'w')
        FKS_card_file.write(FKS_card)
        FKS_card_file.close()

        # Return to original PWD
        os.chdir(cwd)
        # Copy the different python files in the Template
        self.copy_python_files()

        # We need to create the correct open_data for the pdf
        self.write_pdf_opendata()
        
    # I put it here not in optimized one, because I want to use the same makefile_loop.inc
    # Also, we overload this function (i.e. it is already defined in 
    # LoopProcessExporterFortranSA) because the path of the template makefile
    # is different.
    def write_makefile_TIR(self, writer, link_tir_libs,tir_libs,tir_include=[]):
        """ Create the file makefile_loop which links to the TIR libraries."""
            
        file = open(os.path.join(self.mgme_dir,'Template','NLO',
                                 'SubProcesses','makefile_loop.inc')).read()  
        replace_dict={}
        replace_dict['link_tir_libs']=' '.join(link_tir_libs)
        replace_dict['tir_libs']=' '.join(tir_libs)
        replace_dict['dotf']='%.f'
        replace_dict['doto']='%.o'
        replace_dict['tir_include']=' '.join(tir_include)
        file=file%replace_dict
        if writer:
            writer.writelines(file)
        else:
            return file
        
    # I put it here not in optimized one, because I want to use the same make_opts.inc
    def write_make_opts(self, writer, link_tir_libs,tir_libs):
        """ Create the file make_opts which links to the TIR libraries."""
        file = open(os.path.join(self.mgme_dir,'Template','NLO',
                                 'Source','make_opts.inc')).read()  
        replace_dict={}
        replace_dict['link_tir_libs']=' '.join(link_tir_libs)
        replace_dict['tir_libs']=' '.join(tir_libs)
        replace_dict['dotf']='%.f'
        replace_dict['doto']='%.o'
        file=file%replace_dict
        if writer:
            writer.writelines(file)
        else:
            return file

    #===========================================================================
    # copy_python_files 
    #===========================================================================        
    def copy_python_files(self):
        """copy python files required for the Template"""

        files_to_copy = [ \
          pjoin('interface','amcatnlo_run_interface.py'),
          pjoin('interface','extended_cmd.py'),
          pjoin('interface','common_run_interface.py'),
          pjoin('interface','coloring_logging.py'),
          pjoin('various','misc.py'),
          pjoin('various','shower_card.py'),
          pjoin('various','FO_analyse_card.py'),
          pjoin('various','histograms.py'),      
          pjoin('various','banner.py'),          
          pjoin('various','cluster.py'),          
          pjoin('various','lhe_parser.py'),
          pjoin('madevent','sum_html.py'),
          pjoin('madevent','gen_crossxhtml.py'),          
          pjoin('iolibs','files.py'),
          pjoin('iolibs','save_load_object.py'),
          pjoin('iolibs','file_writers.py'),
          pjoin('..','models','check_param_card.py'),
          pjoin('__init__.py')
        ]
        cp(_file_path+'/interface/.mg5_logging.conf', 
                                 self.dir_path+'/bin/internal/me5_logging.conf')
        
        for cp_file in files_to_copy:
            cp(pjoin(_file_path,cp_file),
                pjoin(self.dir_path,'bin','internal',os.path.basename(cp_file)))

    def convert_model_to_mg4(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):

        super(ProcessExporterFortranFKS,self).convert_model_to_mg4(model, 
                                               wanted_lorentz, wanted_couplings)
        
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
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
    # write_maxparticles_file
    #===========================================================================
    def write_maxparticles_file(self, writer, maxparticles):
        """Write the maxparticles.inc file for MadEvent"""

        lines = "integer max_particles, max_branch\n"
        lines += "parameter (max_particles=%d) \n" % maxparticles
        lines += "parameter (max_branch=max_particles-1)"

        # Write the file
        writer.writelines(lines)

        return True


    #===========================================================================
    # write_maxconfigs_file
    #===========================================================================
    def write_maxconfigs_file(self, writer, maxconfigs):
        """Write the maxconfigs.inc file for MadEvent"""

        lines = "integer lmaxconfigs\n"
        lines += "parameter (lmaxconfigs=%d)" % maxconfigs

        # Write the file
        writer.writelines(lines)

        return True


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
    # write a initial states map, useful for the fast PDF NLO interface
    #===============================================================================
    def write_init_map(self, file_pos, initial_states):
        """ Write an initial state process map. Each possible PDF
        combination gets an unique identifier."""
        
        text=''
        for i,e in enumerate(initial_states):
            text=text+str(i+1)+' '+str(len(e))
            for t in e:
                text=text+'   '
                try:
                    for p in t:
                        text=text+' '+str(p)
                except TypeError:
                        text=text+' '+str(t)
            text=text+'\n'
        
        ff = open(file_pos, 'w')
        ff.write(text)
        ff.close()

    def get_ME_identifier(self, matrix_element, *args, **opts):
        """ A function returning a string uniquely identifying the matrix 
        element given in argument so that it can be used as a prefix to all
        MadLoop5 subroutines and common blocks related to it. This allows
        to compile several processes into one library as requested by the 
        BLHA (Binoth LesHouches Accord) guidelines. The MadFKS design
        necessitates that there is no process prefix."""
        
        return ''

    #===============================================================================
    # write_coef_specs
    #===============================================================================
    def write_coef_specs_file(self, virt_me_list):
        """writes the coef_specs.inc in the DHELAS folder. Should not be called in the 
        non-optimized mode"""
        raise fks_common.FKSProcessError(), \
                "write_coef_specs should be called only in the loop-optimized mode"
        
        
    #===============================================================================
    # generate_directories_fks
    #===============================================================================
    def generate_directories_fks(self, matrix_element, fortran_model, me_number,
                                    me_ntot, path=os.getcwd(),OLP='MadLoop'):
        """Generate the Pxxxxx_i directories for a subprocess in MadFKS,
        including the necessary matrix.f and various helper files"""
        proc = matrix_element.born_matrix_element['processes'][0]

        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')
        
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
        
        self.fksdirs = []
        #first make and cd the direcrory corresponding to the born process:
        borndir = "P%s" % \
        (matrix_element.get('processes')[0].shell_string())
        os.mkdir(borndir)
        os.chdir(borndir)
        logger.info('Writing files in %s (%d / %d)' % (borndir, me_number + 1, me_ntot))

## write the files corresponding to the born process in the P* directory
        self.generate_born_fks_files(matrix_element,
                fortran_model, me_number, path)

        # With NJET you want to generate the order file per subprocess and most
        # likely also generate it for each subproc.
        if OLP=='NJET':
            filename = 'OLE_order.lh'
            self.write_lh_order(filename, [matrix_element.born_matrix_element.get('processes')[0]], OLP)
        
        if matrix_element.virt_matrix_element:
                    calls += self.generate_virt_directory( \
                            matrix_element.virt_matrix_element, \
                            fortran_model, \
                            os.path.join(path, borndir))

#write the infortions for the different real emission processes

        self.write_real_matrix_elements(matrix_element, fortran_model)

        self.write_pdf_calls(matrix_element, fortran_model)

        filename = 'nFKSconfigs.inc'
        self.write_nfksconfigs_file(writers.FortranWriter(filename), 
                                    matrix_element, 
                                    fortran_model)

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                              me_number)

        filename = 'fks_info.inc'
        self.write_fks_info_file(writers.FortranWriter(filename), 
                                 matrix_element, 
                                 fortran_model)

        filename = 'leshouche_info.dat'
        nfksconfs,maxproc,maxflow,nexternal=\
                self.write_leshouche_info_file(filename,matrix_element)

        # if no corrections are generated ([LOonly] mode), get 
        # these variables from the born
        if nfksconfs == maxproc == maxflow == 0:
            nfksconfs = 1
            (dummylines, maxproc, maxflow) = self.get_leshouche_lines(
                    matrix_element.born_matrix_element, 1)

        filename = 'leshouche_decl.inc'
        self.write_leshouche_info_declarations(
                              writers.FortranWriter(filename), 
                              nfksconfs,maxproc,maxflow,nexternal,
                              fortran_model)

        filename = 'configs_and_props_info.dat'
        nconfigs,max_leg_number,nfksconfs=self.write_configs_and_props_info_file(
                              filename, 
                              matrix_element)

        filename = 'configs_and_props_decl.inc'
        self.write_configs_and_props_info_declarations(
                              writers.FortranWriter(filename), 
                              nconfigs,max_leg_number,nfksconfs,
                              fortran_model)
        
        filename = 'real_from_born_configs.inc'
        self.write_real_from_born_configs(
                              writers.FortranWriter(filename), 
                              matrix_element,
                              fortran_model)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                            nconfigs)

#write the wrappers
        filename = 'real_me_chooser.f'
        self.write_real_me_wrapper(writers.FortranWriter(filename), 
                                   matrix_element, 
                                   fortran_model)

        filename = 'parton_lum_chooser.f'
        self.write_pdf_wrapper(writers.FortranWriter(filename), 
                                   matrix_element, 
                                   fortran_model)

        filename = 'get_color.f'
        self.write_colors_file(writers.FortranWriter(filename),
                               matrix_element)

        filename = 'nexternal.inc'
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)
        self.proc_characteristic['ninitial'] = ninitial
        self.proc_characteristic['nexternal'] = max(self.proc_characteristic['nexternal'], nexternal)
    
        filename = 'pmass.inc'
        try:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.real_processes[0].matrix_element)
        except IndexError:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.born_matrix_element)

        #draw the diagrams
        self.draw_feynman_diagrams(matrix_element)

        linkfiles = ['BinothLHADummy.f',
                     'check_poles.f',
                     'MCmasses_HERWIG6.inc',
                     'MCmasses_HERWIGPP.inc',
                     'MCmasses_PYTHIA6Q.inc',
                     'MCmasses_PYTHIA6PT.inc',
                     'MCmasses_PYTHIA8.inc',
                     'add_write_info.f',
                     'coupl.inc',
                     'cuts.f',
                     'FKS_params.dat',
                     'initial_states_map.dat',
                     'OLE_order.olc',
                     'FKSParams.inc',
                     'FKSParamReader.f',
                     'cuts.inc',
                     'unlops.inc',
                     'pythia_unlops.f',
                     'driver_mintMC.f',
                     'driver_mintFO.f',
                     'driver_vegas.f',
                     'appl_interface.cc',
                     'appl_interface_dummy.f',
                     'appl_common.inc',
                     'reweight_appl.inc',
                     'driver_reweight.f',
                     'fastjetfortran_madfks_core.cc',
                     'fastjetfortran_madfks_full.cc',
                     'fjcore.cc',
                     'fastjet_wrapper.f',
                     'fjcore.hh',
                     'fks_Sij.f',
                     'fks_powers.inc',
                     'fks_singular.f',
                     'veto_xsec.f',
                     'veto_xsec.inc',
                     'c_weight.inc',
                     'fks_inc_chooser.f',
                     'leshouche_inc_chooser.f',
                     'configs_and_props_inc_chooser.f',
                     'genps.inc',
                     'genps_fks.f',
                     'boostwdir2.f',
                     'madfks_mcatnlo.inc',
                     'open_output_files.f',
                     'open_output_files_dummy.f',
                     'HwU_dummy.f',
                     'madfks_plot.f',
                     'analysis_dummy.f',
                     'mint-integrator2.f',
                     'MC_integer.f',
                     'mint.inc',
                     'montecarlocounter.f',
                     'q_es.inc',
                     'recluster.cc',
                     'Boosts.h',
                     'reweight.inc',
                     'reweight0.inc',
                     'reweight1.inc',
                     'reweightNLO.inc',
                     'reweight_all.inc',
                     'reweight_events.f',
                     'reweight_xsec.f',
                     'reweight_xsec_events.f',
                     'reweight_xsec_events_pdf_dummy.f',
                     'iproc_map.f',
                     'run.inc',
                     'run_card.inc',
                     'setcuts.f',
                     'setscales.f',
                     'symmetry_fks_test_MC.f',
                     'symmetry_fks_test_ME.f',
                     'symmetry_fks_test_Sij.f',
                     'symmetry_fks_v3.f',
                     'trapfpe.c',
                     'vegas2.for',
                     'write_ajob.f',
                     'handling_lhe_events.f',
                     'write_event.f',
                     'fill_MC_mshell.f',
                     'maxparticles.inc',
                     'message.inc',
                     'initcluster.f',
                     'cluster.inc',
                     'cluster.f',
                     'reweight.f',
                     'randinit',
                     'sudakov.inc',
                     'maxconfigs.inc',
                     'timing_variables.inc']

        for file in linkfiles:
            ln('../' + file , '.')
        os.system("ln -s ../../Cards/param_card.dat .")

        #copy the makefile 
        os.system("ln -s ../makefile_fks_dir ./makefile")
        if matrix_element.virt_matrix_element:
            os.system("ln -s ../BinothLHA.f ./BinothLHA.f")
        elif OLP!='MadLoop':
            os.system("ln -s ../BinothLHA_OLP.f ./BinothLHA.f")
        else:
            os.system("ln -s ../BinothLHA_user.f ./BinothLHA.f")


        #import nexternal/leshouches in Source
        ln('nexternal.inc', '../../Source', log=False)
        ln('born_leshouche.inc', '../../Source', log=False)


        # Return to SubProcesses dir
        os.chdir(os.path.pardir)
        # Add subprocess to subproc.mg
        filename = 'subproc.mg'
        files.append_to_file(filename,
                             self.write_subproc,
                             borndir)

            
        os.chdir(cwd)
        # Generate info page
        gen_infohtml.make_info_html_nlo(self.dir_path)


        return calls

    #===========================================================================
    #  create the run_card 
    #===========================================================================
    def create_run_card(self, processes, history):
        """ """
 
        run_card = banner_mod.RunCardNLO()
        
        run_card.create_default_for_process(self.proc_characteristic, 
                                            history,
                                            processes)
        
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'))
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card.dat'))


    def finalize_fks_directory(self, matrix_elements, history, makejpg = False,
            online = False, 
            compiler_dict={'fortran': 'gfortran', 'cpp': 'g++'}, 
            output_dependencies = 'external', MG5DIR = None):
        """Finalize FKS directory by creating jpeg diagrams, html
        pages,proc_card_mg5.dat and madevent.tar.gz."""
        
        self.proc_characteristic['grouped_matrix'] = False
        self.create_proc_charac()

        self.create_run_card(matrix_elements.get_processes(), history)
#        modelname = self.model.get('name')
#        if modelname == 'mssm' or modelname.startswith('mssm-'):
#            param_card = os.path.join(self.dir_path, 'Cards','param_card.dat')
#            mg5_param = os.path.join(self.dir_path, 'Source', 'MODEL', 'MG5_param.dat')
#            check_param_card.convert_to_mg5card(param_card, mg5_param)
#            check_param_card.check_valid_param_card(mg5_param)

#        # write the model functions get_mass/width_from_id
        filename = os.path.join(self.dir_path,'Source','MODEL','get_mass_width_fcts.f')
        makeinc = os.path.join(self.dir_path,'Source','MODEL','makeinc.inc')
        self.write_get_mass_width_file(writers.FortranWriter(filename), makeinc, self.model)

#        # Write maxconfigs.inc based on max of ME's/subprocess groups

        filename = os.path.join(self.dir_path,'Source','maxconfigs.inc')
        self.write_maxconfigs_file(writers.FortranWriter(filename),
                                   matrix_elements.get_max_configs())
        
#        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = os.path.join(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements.get_max_particles())

        # Touch "done" file
        os.system('touch %s/done' % os.path.join(self.dir_path,'SubProcesses'))
        
        # Check for compiler
        fcompiler_chosen = self.set_fortran_compiler(compiler_dict)
        ccompiler_chosen = self.set_cpp_compiler(compiler_dict['cpp'])

        old_pos = os.getcwd()
        os.chdir(os.path.join(self.dir_path, 'SubProcesses'))
        P_dir_list = [proc for proc in os.listdir('.') if os.path.isdir(proc) and \
                                                                    proc[0] == 'P']

        devnull = os.open(os.devnull, os.O_RDWR)
        # Convert the poscript in jpg files (if authorize)
        if makejpg:
            logger.info("Generate jpeg diagrams")
            for Pdir in P_dir_list:
                os.chdir(Pdir)
                subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'internal', 'gen_jpeg-pl')],
                                stdout = devnull)
                os.chdir(os.path.pardir)
#
        logger.info("Generate web pages")
        # Create the WebPage using perl script

        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')], \
                                                                stdout = devnull)

        os.chdir(os.path.pardir)
#
#        obj = gen_infohtml.make_info_html(self.dir_path)
#        [mv(name, './HTML/') for name in os.listdir('.') if \
#                            (name.endswith('.html') or name.endswith('.jpg')) and \
#                            name != 'index.html']               
#        if online:
#            nb_channel = obj.rep_rule['nb_gen_diag']
#            open(os.path.join('./Online'),'w').write(str(nb_channel))
        
        # Write command history as proc_card_mg5
        if os.path.isdir('Cards'):
            output_file = os.path.join('Cards', 'proc_card_mg5.dat')
            history.write(output_file)

        # Duplicate run_card and FO_analyse_card
        for card in ['run_card', 'FO_analyse_card', 'shower_card']:
            try:
                shutil.copy(pjoin(self.dir_path, 'Cards',
                                         card + '.dat'),
                           pjoin(self.dir_path, 'Cards',
                                        card + '_default.dat'))
            except IOError:
                logger.warning("Failed to copy " + card + ".dat to default")


        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull)

        # Run "make" to generate madevent.tar.gz file
        if os.path.exists(pjoin('SubProcesses', 'subproc.mg')):
            if os.path.exists('amcatnlo.tar.gz'):
                os.remove('amcatnlo.tar.gz')
            subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'internal', 'make_amcatnlo_tar')],
                        stdout = devnull)
#
        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull)

        #return to the initial dir
        os.chdir(old_pos)
        
        # Setup stdHep
        # Find the correct fortran compiler
        base_compiler= ['FC=g77','FC=gfortran']
        
        StdHep_path = pjoin(MG5DIR, 'vendor', 'StdHEP')
        
        if output_dependencies == 'external':
            # check if stdhep has to be compiled (only the first time)
            if not os.path.exists(pjoin(MG5DIR, 'vendor', 'StdHEP', 'lib', 'libstdhep.a')) or \
                not os.path.exists(pjoin(MG5DIR, 'vendor', 'StdHEP', 'lib', 'libFmcfio.a')):
                if 'FC' not in os.environ or not os.environ['FC']:
                    path = os.path.join(StdHep_path, 'src', 'make_opts')
                    text = open(path).read()
                    for base in base_compiler:
                        text = text.replace(base,'FC=%s' % fcompiler_chosen)
                    open(path, 'w').writelines(text)

                logger.info('Compiling StdHEP. This has to be done only once.')
                misc.compile(cwd = pjoin(MG5DIR, 'vendor', 'StdHEP'))
                logger.info('Done.')
            #then link the libraries in the exported dir
            files.ln(pjoin(StdHep_path, 'lib', 'libstdhep.a'), \
                                         pjoin(self.dir_path, 'MCatNLO', 'lib'))
            files.ln(pjoin(StdHep_path, 'lib', 'libFmcfio.a'), \
                                         pjoin(self.dir_path, 'MCatNLO', 'lib'))

        elif output_dependencies == 'internal':
            StdHEP_internal_path = pjoin(self.dir_path,'Source','StdHEP')
            shutil.copytree(StdHep_path,StdHEP_internal_path, symlinks=True)
            # Create the links to the lib folder
            linkfiles = ['libstdhep.a', 'libFmcfio.a']
            for file in linkfiles:
                ln(pjoin(os.path.pardir,os.path.pardir,'Source','StdHEP','lib',file),
                                  os.path.join(self.dir_path, 'MCatNLO', 'lib'))
                if 'FC' not in os.environ or not os.environ['FC']:
                    path = pjoin(StdHEP_internal_path, 'src', 'make_opts')
                    text = open(path).read()
                    for base in base_compiler:
                        text = text.replace(base,'FC=%s' % fcompiler_chosen)
                    open(path, 'w').writelines(text)
                # To avoid compiler version conflicts, we force a clean here
                misc.compile(['clean'],cwd = StdHEP_internal_path)
        
        elif output_dependencies == 'environment_paths':
            # Here the user chose to define the dependencies path in one of 
            # his environmental paths
            libStdHep = misc.which_lib('libstdhep.a')
            libFmcfio = misc.which_lib('libFmcfio.a')
            if not libStdHep is None and not libFmcfio is None:
                logger.info('MG5_aMC is using StdHep installation found at %s.'%\
                                                     os.path.dirname(libStdHep)) 
                ln(pjoin(libStdHep),pjoin(self.dir_path, 'MCatNLO', 'lib'),abspath=True)
                ln(pjoin(libFmcfio),pjoin(self.dir_path, 'MCatNLO', 'lib'),abspath=True)
            else:
                raise InvalidCmd("Could not find the location of the files"+\
                    " libstdhep.a and libFmcfio.a in you environment paths.")
            
        else:
            raise MadGraph5Error, 'output_dependencies option %s not recognized'\
                                                            %output_dependencies
           

    def write_real_from_born_configs(self, writer, matrix_element, fortran_model):
        """Writes the real_from_born_configs.inc file that contains
        the mapping to go for a given born configuration (that is used
        e.g. in the multi-channel phase-space integration to the
        corresponding real-emission diagram, i.e. the real emission
        diagram in which the combined ij is split in i_fks and
        j_fks."""
        lines=[]
        lines2=[]
        max_links=0
        born_me=matrix_element.born_matrix_element
        for iFKS, conf in enumerate(matrix_element.get_fks_info_list()):
            iFKS=iFKS+1
            links=conf['fks_info']['rb_links']
            max_links=max(max_links,len(links))
            for i,diags in enumerate(links):
                if not i == diags['born_conf']:
                    print links
                    raise MadGraph5Error, "born_conf should be canonically ordered"
            real_configs=', '.join(['%d' % int(diags['real_conf']+1) for diags in links])
            lines.append("data (real_from_born_conf(irfbc,%d),irfbc=1,%d) /%s/" \
                             % (iFKS,len(links),real_configs))

        lines2.append("integer irfbc")
        lines2.append("integer real_from_born_conf(%d,%d)" \
                         % (max_links,len(matrix_element.get_fks_info_list())))
        # Write the file
        writer.writelines(lines2+lines)


    #===============================================================================
    # write_get_mass_width_file
    #===============================================================================
    #test written
    def write_get_mass_width_file(self, writer, makeinc, model):
        """Write the get_mass_width_file.f file for MG4.
        Also update the makeinc.inc file
        """
        mass_particles = [p for p in model['particles'] if p['mass'].lower() != 'zero'] 
        width_particles = [p for p in model['particles'] if p['width'].lower() != 'zero'] 
        
        iflines_mass = ''
        iflines_width = ''

        for i, part in enumerate(mass_particles):
            if i == 0:
                ifstring = 'if'
            else:
                ifstring = 'else if'
            if part['self_antipart']:
                iflines_mass += '%s (id.eq.%d) then\n' % \
                        (ifstring, part.get_pdg_code())
            else:
                iflines_mass += '%s (id.eq.%d.or.id.eq.%d) then\n' % \
                        (ifstring, part.get_pdg_code(), part.get_anti_pdg_code())
            iflines_mass += 'get_mass_from_id=abs(%s)\n' % part.get('mass')

        for i, part in enumerate(width_particles):
            if i == 0:
                ifstring = 'if'
            else:
                ifstring = 'else if'
            if part['self_antipart']:
                iflines_width += '%s (id.eq.%d) then\n' % \
                        (ifstring, part.get_pdg_code())
            else:
                iflines_width += '%s (id.eq.%d.or.id.eq.%d) then\n' % \
                        (ifstring, part.get_pdg_code(), part.get_anti_pdg_code())
            iflines_width += 'get_width_from_id=abs(%s)\n' % part.get('width')

        replace_dict = {'iflines_mass' : iflines_mass,
                        'iflines_width' : iflines_width}

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/get_mass_width_fcts.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)

        # update the makeinc
        makeinc_content = open(makeinc).read()
        makeinc_content = makeinc_content.replace('MODEL = ', 'MODEL = get_mass_width_fcts.o ')
        open(makeinc, 'w').write(makeinc_content)

        return 


    def write_configs_and_props_info_declarations(self, writer, max_iconfig, max_leg_number, nfksconfs, fortran_model):
        """writes the declarations for the variables relevant for configs_and_props
        """
        lines = []
        lines.append("integer ifr,lmaxconfigs_used,max_branch_used")
        lines.append("parameter (lmaxconfigs_used=%4d)" % max_iconfig)
        lines.append("parameter (max_branch_used =%4d)" % -max_leg_number)
        lines.append("integer mapconfig_d(%3d,0:lmaxconfigs_used)" % nfksconfs)
        lines.append("integer iforest_d(%3d,2,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)
        lines.append("integer sprop_d(%3d,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)
        lines.append("integer tprid_d(%3d,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)
        lines.append("double precision pmass_d(%3d,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)
        lines.append("double precision pwidth_d(%3d,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)
        lines.append("integer pow_d(%3d,-max_branch_used:-1,lmaxconfigs_used)" % nfksconfs)

        writer.writelines(lines)


    def write_configs_and_props_info_file(self, filename, matrix_element):
        """writes the configs_and_props_info.inc file that cointains
        all the (real-emission) configurations (IFOREST) as well as
        the masses and widths of intermediate particles"""
        lines = []
        lines.append("# C -> MAPCONFIG_D") 
        lines.append("# F/D -> IFOREST_D") 
        lines.append("# S -> SPROP_D") 
        lines.append("# T -> TPRID_D") 
        lines.append("# M -> PMASS_D/PWIDTH_D") 
        lines.append("# P -> POW_D") 
        lines2 = []
        nconfs = len(matrix_element.get_fks_info_list())
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        max_iconfig=0
        max_leg_number=0

        for iFKS, conf in enumerate(matrix_element.get_fks_info_list()):
            iFKS=iFKS+1
            iconfig = 0
            s_and_t_channels = []
            mapconfigs = []
            fks_matrix_element=matrix_element.real_processes[conf['n_me'] - 1].matrix_element
            base_diagrams = fks_matrix_element.get('base_amplitude').get('diagrams')
            model = fks_matrix_element.get('base_amplitude').get('process').get('model')
            minvert = min([max([len(vert.get('legs')) for vert in \
                                    diag.get('vertices')]) for diag in base_diagrams])
    
            lines.append("# ")
            lines.append("# nFKSprocess %d" % iFKS)
            for idiag, diag in enumerate(base_diagrams):
                if any([len(vert.get('legs')) > minvert for vert in
                        diag.get('vertices')]):
                # Only 3-vertices allowed in configs.inc
                    continue
                iconfig = iconfig + 1
                helas_diag = fks_matrix_element.get('diagrams')[idiag]
                mapconfigs.append(helas_diag.get('number'))
                lines.append("# Diagram %d for nFKSprocess %d" % \
                                 (helas_diag.get('number'),iFKS))
                # Correspondance between the config and the amplitudes
                lines.append("C   %4d   %4d   %4d " % (iFKS,iconfig,
                                                           helas_diag.get('number')))
    
                # Need to reorganize the topology so that we start with all
                # final state external particles and work our way inwards
                schannels, tchannels = helas_diag.get('amplitudes')[0].\
                    get_s_and_t_channels(ninitial, model, 990)
    
                s_and_t_channels.append([schannels, tchannels])
    
                # Write out propagators for s-channel and t-channel vertices
                allchannels = schannels
                if len(tchannels) > 1:
                    # Write out tchannels only if there are any non-trivial ones
                    allchannels = schannels + tchannels
    
                for vert in allchannels:
                    daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                    last_leg = vert.get('legs')[-1]
                    lines.append("F   %4d   %4d   %4d   %4d" % \
                                     (iFKS,last_leg.get('number'), iconfig, len(daughters)))
                    for d in daughters:
                        lines.append("D   %4d" % d)
                    if vert in schannels:
                        lines.append("S   %4d   %4d   %4d   %10d" % \
                                         (iFKS,last_leg.get('number'), iconfig,
                                          last_leg.get('id')))
                    elif vert in tchannels[:-1]:
                        lines.append("T   %4d   %4d   %4d   %10d" % \
                                         (iFKS,last_leg.get('number'), iconfig,
                                          abs(last_leg.get('id'))))

                # update what the array sizes (mapconfig,iforest,etc) will be
                    max_leg_number = min(max_leg_number,last_leg.get('number'))
                max_iconfig = max(max_iconfig,iconfig)
    
            # Write out number of configs
            lines.append("# Number of configs for nFKSprocess %d" % iFKS)
            lines.append("C   %4d   %4d   %4d" % (iFKS,0,iconfig))
            
            # write the props.inc information
            lines2.append("# ")
            particle_dict = fks_matrix_element.get('processes')[0].get('model').\
                get('particle_dict')
    
            for iconf, configs in enumerate(s_and_t_channels):
                for vertex in configs[0] + configs[1][:-1]:
                    leg = vertex.get('legs')[-1]
                    if leg.get('id') not in particle_dict:
                        # Fake propagator used in multiparticle vertices
                        pow_part = 0
                    else:
                        particle = particle_dict[leg.get('id')]
    
                        pow_part = 1 + int(particle.is_boson())
    
                    lines2.append("M   %4d   %4d   %4d   %10d " % \
                                     (iFKS,leg.get('number'), iconf + 1, leg.get('id')))
                    lines2.append("P   %4d   %4d   %4d   %4d " % \
                                     (iFKS,leg.get('number'), iconf + 1, pow_part))
    
        # Write the file
        open(filename,'w').write('\n'.join(lines+lines2))

        return max_iconfig, max_leg_number, nconfs


    def write_leshouche_info_declarations(self, writer, nfksconfs, 
                                  maxproc, maxflow, nexternal, fortran_model):
        """writes the declarations for the variables relevant for leshouche_info
        """
        lines = []
        lines.append('integer maxproc_used, maxflow_used')
        lines.append('parameter (maxproc_used = %d)' % maxproc)
        lines.append('parameter (maxflow_used = %d)' % maxflow)
        lines.append('integer idup_d(%d,%d,maxproc_used)' % (nfksconfs, nexternal))
        lines.append('integer mothup_d(%d,%d,%d,maxproc_used)' % (nfksconfs, 2, nexternal))
        lines.append('integer icolup_d(%d,%d,%d,maxflow_used)' % (nfksconfs, 2, nexternal))
        lines.append('integer niprocs_d(%d)' % (nfksconfs))

        writer.writelines(lines)


    def write_leshouche_info_file(self, filename, matrix_element):
        """writes the leshouche_info.inc file which contains 
        the LHA informations for all the real emission processes
        """
        lines = []
        lines.append("# I -> IDUP_D")
        lines.append("# M -> MOTHUP_D")
        lines.append("# C -> ICOLUP_D")
        nfksconfs = len(matrix_element.get_fks_info_list())
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        maxproc = 0
        maxflow = 0
        for i, conf in enumerate(matrix_element.get_fks_info_list()):
#        for i, real in enumerate(matrix_element.real_processes):
            (newlines, nprocs, nflows) = self.get_leshouche_lines(
                    matrix_element.real_processes[conf['n_me'] - 1].matrix_element, i + 1)
            lines.extend(newlines)
            maxproc = max(maxproc, nprocs)
            maxflow = max(maxflow, nflows)

        # Write the file
        open(filename,'w').write('\n'.join(lines))

        return nfksconfs, maxproc, maxflow, nexternal


    def write_pdf_wrapper(self, writer, matrix_element, fortran_model):
        """writes the wrapper which allows to chose among the different real matrix elements"""

        file = \
"""double precision function dlum()
implicit none
include 'timing_variables.inc'
integer nfksprocess
common/c_nfksprocess/nfksprocess
call cpu_time(tbefore)
"""
        if matrix_element.real_processes:
            for n, info in enumerate(matrix_element.get_fks_info_list()):
                file += \
"""if (nfksprocess.eq.%(n)d) then
call dlum_%(n_me)d(dlum)
else""" % {'n': n + 1, 'n_me' : info['n_me']}
            file += \
"""
write(*,*) 'ERROR: invalid n in dlum :', nfksprocess
stop
endif
call cpu_time(tAfter)
tPDF = tPDF + (tAfter-tBefore)
return
end
"""
        else:
            file+= \
"""call dlum_0(dlum)
call cpu_time(tAfter)
tPDF = tPDF + (tAfter-tBefore)
return
end
"""

        # Write the file
        writer.writelines(file)
        return 0


    def write_real_me_wrapper(self, writer, matrix_element, fortran_model):
        """writes the wrapper which allows to chose among the different real matrix elements"""

        file = \
"""subroutine smatrix_real(p, wgt)
implicit none
include 'nexternal.inc'
double precision p(0:3, nexternal)
double precision wgt
integer nfksprocess
common/c_nfksprocess/nfksprocess
"""
        for n, info in enumerate(matrix_element.get_fks_info_list()):
            file += \
"""if (nfksprocess.eq.%(n)d) then
call smatrix_%(n_me)d(p, wgt)
else""" % {'n': n + 1, 'n_me' : info['n_me']}

        if matrix_element.real_processes:
            file += \
"""
write(*,*) 'ERROR: invalid n in real_matrix :', nfksprocess
stop
endif
return
end
"""
        else:
            file += \
"""
wgt=0d0
return
end
"""
        # Write the file
        writer.writelines(file)
        return 0


    def draw_feynman_diagrams(self, matrix_element):
        """Create the ps files containing the feynman diagrams for the born process,
        as well as for all the real emission processes"""

        filename = 'born.ps'
        plot = draw.MultiEpsDiagramDrawer(matrix_element.born_matrix_element.\
                                    get('base_amplitude').get('diagrams'),
                                    filename,
                                    model=matrix_element.born_matrix_element.\
                                    get('processes')[0].get('model'),
                                    amplitude=True, diagram_type='born')
        plot.draw()

        for n, fksreal in enumerate(matrix_element.real_processes):
            filename = 'matrix_%d.ps' % (n + 1)
            plot = draw.MultiEpsDiagramDrawer(fksreal.matrix_element.\
                                        get('base_amplitude').get('diagrams'),
                                        filename,
                                        model=fksreal.matrix_element.\
                                        get('processes')[0].get('model'),
                                        amplitude=True, diagram_type='real')
            plot.draw()


    def write_real_matrix_elements(self, matrix_element, fortran_model):
        """writes the matrix_i.f files which contain the real matrix elements""" 



        for n, fksreal in enumerate(matrix_element.real_processes):
            filename = 'matrix_%d.f' % (n + 1)
            self.write_matrix_element_fks(writers.FortranWriter(filename),
                                            fksreal.matrix_element, n + 1, 
                                            fortran_model)

    def write_pdf_calls(self, matrix_element, fortran_model):
        """writes the parton_lum_i.f files which contain the real matrix elements.
        If no real emission existst, write the one for the born""" 

        if matrix_element.real_processes:
            for n, fksreal in enumerate(matrix_element.real_processes):
                filename = 'parton_lum_%d.f' % (n + 1)
                self.write_pdf_file(writers.FortranWriter(filename),
                                                fksreal.matrix_element, n + 1, 
                                                fortran_model)
        else:
                filename = 'parton_lum_0.f'
                self.write_pdf_file(writers.FortranWriter(filename),
                                                matrix_element.born_matrix_element, 0, 
                                                fortran_model)


    def generate_born_fks_files(self, matrix_element, fortran_model, me_number, path):
        """generates the files needed for the born amplitude in the P* directory, which will
        be needed by the P* directories"""
        pathdir = os.getcwd()

        filename = 'born.f'
        calls_born, ncolor_born = \
            self.write_born_fks(writers.FortranWriter(filename),\
                             matrix_element,
                             fortran_model)

        filename = 'born_hel.f'
        self.write_born_hel(writers.FortranWriter(filename),\
                             matrix_element,
                             fortran_model)


        filename = 'born_conf.inc'
        nconfigs, mapconfigs, s_and_t_channels = \
                    self.write_configs_file(
                    writers.FortranWriter(filename),
                    matrix_element.born_matrix_element, 
                    fortran_model)

        filename = 'born_props.inc'
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element.born_matrix_element,
                         fortran_model,
                            s_and_t_channels)
    
        filename = 'born_decayBW.inc'
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
                    matrix_element.born_matrix_element.get_number_of_amplitudes())

        filename = 'ncombs.inc'
        self.write_ncombs_file(writers.FortranWriter(filename),
                               matrix_element.born_matrix_element,
                               fortran_model)

        filename = 'born_maxamps.inc'
        maxamps = len(matrix_element.get('diagrams'))
        maxflows = ncolor_born
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           max([len(matrix_element.get('processes')) for me in \
                                matrix_element.born_matrix_element]),1)

        filename = 'config_subproc_map.inc'
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           s_and_t_channels)

        filename = 'coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                             mapconfigs,
                             matrix_element.born_matrix_element,
                             fortran_model)
        
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


    def generate_virtuals_from_OLP(self,process_list,export_path, OLP):
        """Generates the library for computing the loop matrix elements
        necessary for this process using the OLP specified."""
        
        # Start by writing the BLHA order file
        virtual_path = pjoin(export_path,'OLP_virtuals')
        if not os.path.exists(virtual_path):
            os.makedirs(virtual_path)
        filename = os.path.join(virtual_path,'OLE_order.lh')
        self.write_lh_order(filename, process_list, OLP)

        fail_msg='Generation of the virtuals with %s failed.\n'%OLP+\
            'Please check the virt_generation.log file in %s.'\
                                 %str(pjoin(virtual_path,'virt_generation.log'))

        # Perform some tasks specific to certain OLP's
        if OLP=='GoSam':
            cp(pjoin(self.mgme_dir,'Template','loop_material','OLP_specifics',
                             'GoSam','makevirt'),pjoin(virtual_path,'makevirt'))
            cp(pjoin(self.mgme_dir,'Template','loop_material','OLP_specifics',
                             'GoSam','gosam.rc'),pjoin(virtual_path,'gosam.rc'))
            ln(pjoin(export_path,'Cards','param_card.dat'),virtual_path)
            # Now generate the process
            logger.info('Generating the loop matrix elements with %s...'%OLP)
            virt_generation_log = \
                            open(pjoin(virtual_path,'virt_generation.log'), 'w')
            retcode = subprocess.call(['./makevirt'],cwd=virtual_path, 
                            stdout=virt_generation_log, stderr=virt_generation_log)
            virt_generation_log.close()
            # Check what extension is used for the share libraries on this system
            possible_other_extensions = ['so','dylib']
            shared_lib_ext='so'
            for ext in possible_other_extensions:
                if os.path.isfile(pjoin(virtual_path,'Virtuals','lib',
                                                            'libgolem_olp.'+ext)):
                    shared_lib_ext = ext

            # Now check that everything got correctly generated
            files_to_check = ['olp_module.mod',str(pjoin('lib',
                                                'libgolem_olp.'+shared_lib_ext))]
            if retcode != 0 or any([not os.path.exists(pjoin(virtual_path,
                                       'Virtuals',f)) for f in files_to_check]):
                raise fks_common.FKSProcessError(fail_msg)
            # link the library to the lib folder
            ln(pjoin(virtual_path,'Virtuals','lib','libgolem_olp.'+shared_lib_ext),
                                                       pjoin(export_path,'lib'))
            
        # Specify in make_opts the right library necessitated by the OLP
        make_opts_content=open(pjoin(export_path,'Source','make_opts')).read()
        make_opts=open(pjoin(export_path,'Source','make_opts'),'w')
        if OLP=='GoSam':
            if platform.system().lower()=='darwin':
                # On mac the -rpath is not supported and the path of the dynamic
                # library is automatically wired in the executable
                make_opts_content=make_opts_content.replace('libOLP=',
                                                          'libOLP=-Wl,-lgolem_olp')
            else:
                # On other platforms the option , -rpath= path to libgolem.so is necessary
                # Using a relative path is not ideal because the file libgolem.so is not
                # copied on the worker nodes.
#                make_opts_content=make_opts_content.replace('libOLP=',
#                                      'libOLP=-Wl,-rpath=../$(LIBDIR) -lgolem_olp')
                # Using the absolute path is working in the case where the disk of the 
                # front end machine is mounted on all worker nodes as well.
                make_opts_content=make_opts_content.replace('libOLP=', 
                 'libOLP=-Wl,-rpath='+str(pjoin(export_path,'lib'))+' -lgolem_olp')
            
            
        make_opts.write(make_opts_content)
        make_opts.close()

        # A priori this is generic to all OLP's
        
        # Parse the contract file returned and propagate the process label to
        # the include of the BinothLHA.f file            
        proc_to_label = self.parse_contract_file(
                                            pjoin(virtual_path,'OLE_order.olc'))

        self.write_BinothLHA_inc(process_list,proc_to_label,\
                                              pjoin(export_path,'SubProcesses'))
        
        # Link the contract file to within the SubProcess directory
        ln(pjoin(virtual_path,'OLE_order.olc'),pjoin(export_path,'SubProcesses'))
        
    def write_BinothLHA_inc(self, processes, proc_to_label, SubProcPath):
        """ Write the file Binoth_proc.inc in each SubProcess directory so as 
        to provide the right process_label to use in the OLP call to get the
        loop matrix element evaluation. The proc_to_label is the dictionary of
        the format of the one returned by the function parse_contract_file."""
        
        for proc in processes:
            name = "P%s"%proc.shell_string()
            proc_pdgs=(tuple([leg.get('id') for leg in proc.get('legs') if \
                                                         not leg.get('state')]),
                       tuple([leg.get('id') for leg in proc.get('legs') if \
                                                             leg.get('state')]))                             
            incFile = open(pjoin(SubProcPath, name,'Binoth_proc.inc'),'w')
            try:
                incFile.write(
"""      INTEGER PROC_LABEL
      PARAMETER (PROC_LABEL=%d)"""%(proc_to_label[proc_pdgs]))
            except KeyError:
                raise fks_common.FKSProcessError('Could not found the target'+\
                  ' process %s > %s in '%(str(proc_pdgs[0]),str(proc_pdgs[1]))+\
                          ' the proc_to_label argument in write_BinothLHA_inc.')
            incFile.close()

    def parse_contract_file(self, contract_file_path):
        """ Parses the BLHA contract file, make sure all parameters could be 
        understood by the OLP and return a mapping of the processes (characterized
        by the pdg's of the initial and final state particles) to their process
        label. The format of the mapping is {((in_pdgs),(out_pdgs)):proc_label}.
        """
        
        proc_def_to_label = {}
        
        if not os.path.exists(contract_file_path):
            raise fks_common.FKSProcessError('Could not find the contract file'+\
                                 ' OLE_order.olc in %s.'%str(contract_file_path))

        comment_re=re.compile(r"^\s*#")
        proc_def_re=re.compile(
            r"^(?P<in_pdgs>(\s*-?\d+\s*)+)->(?P<out_pdgs>(\s*-?\d+\s*)+)\|"+
            r"\s*(?P<proc_class>\d+)\s*(?P<proc_label>\d+)\s*$")
        line_OK_re=re.compile(r"^.*\|\s*OK")
        for line in file(contract_file_path):
            # Ignore comments
            if not comment_re.match(line) is None:
                continue
            # Check if it is a proc definition line
            proc_def = proc_def_re.match(line)
            if not proc_def is None:
                if int(proc_def.group('proc_class'))!=1:
                    raise fks_common.FKSProcessError(
'aMCatNLO can only handle loop processes generated by the OLP which have only '+\
' process class attribute. Found %s instead in: \n%s'\
                                           %(proc_def.group('proc_class'),line))
                in_pdgs=tuple([int(in_pdg) for in_pdg in \
                                             proc_def.group('in_pdgs').split()])
                out_pdgs=tuple([int(out_pdg) for out_pdg in \
                                            proc_def.group('out_pdgs').split()])
                proc_def_to_label[(in_pdgs,out_pdgs)]=\
                                               int(proc_def.group('proc_label'))
                continue
            # For the other types of line, just make sure they end with | OK
            if line_OK_re.match(line) is None:
                raise fks_common.FKSProcessError(
                      'The OLP could not process the following line: \n%s'%line)
        
        return proc_def_to_label
            
                                
    def generate_virt_directory(self, loop_matrix_element, fortran_model, dir_name):
        """writes the V**** directory inside the P**** directories specified in
        dir_name"""

        cwd = os.getcwd()

        matrix_element = loop_matrix_element

        # Create the MadLoop5_resources directory if not already existing
        dirpath = os.path.join(dir_name, 'MadLoop5_resources')
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

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

        calls=self.write_loop_matrix_element_v4(None,matrix_element,fortran_model)
        # The born matrix element, if needed
        filename = 'born_matrix.f'
        calls = self.write_bornmatrix(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model)

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        filename = "loop_matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(
              matrix_element.get('base_amplitude').get('loop_diagrams')[:1000]),
              filename,
              model=matrix_element.get('processes')[0].get('model'),
              amplitude='')
        logger.info("Drawing loop Feynman diagrams for " + \
            matrix_element.get('processes')[0].nice_string(print_weighted=False))
        plot.draw()

        filename = "born_matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
            get('born_diagrams'),filename,model=matrix_element.get('processes')[0].\
                                                      get('model'),amplitude='')
        logger.info("Generating born Feynman diagrams for " + \
            matrix_element.get('processes')[0].nice_string(print_weighted=False))
        plot.draw()

        linkfiles = ['coupl.inc', 'mp_coupl.inc', 'mp_coupl_same_name.inc',
                     'cts_mprec.h', 'cts_mpc.h', 'MadLoopParamReader.f',
                     'MadLoopCommons.f','MadLoopParams.inc']

        # We should move to MadLoop5_resources directory from the SubProcesses

        ln(pjoin(os.path.pardir,os.path.pardir,'MadLoopParams.dat'),
                                              pjoin('..','MadLoop5_resources'))

        for file in linkfiles:
            ln('../../%s' % file)

        os.system("ln -s ../../makefile_loop makefile")

        linkfiles = ['mpmodule.mod']

        for file in linkfiles:
            ln('../../../lib/%s' % file)

        linkfiles = ['coef_specs.inc']

        for file in linkfiles:        
            ln('../../../Source/DHELAS/%s' % file)

        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    def get_qed_qcd_orders_from_weighted(self, nexternal, weighted):
        """computes the QED/QCD orders from the knowledge of the n of ext particles
        and of the weighted orders"""
        # n vertices = nexternal - 2 =QED + QCD
        # weighted = 2*QED + QCD
        QED = weighted - nexternal + 2
        QCD = weighted - 2 * QED
        return QED, QCD



    #===============================================================================
    # write_lh_order
    #===============================================================================
    #test written
    def write_lh_order(self, filename, process_list, OLP='MadLoop'):
        """Creates the OLE_order.lh file. This function should be edited according
        to the OLP which is used. For now it is generic."""
        
        
        if len(process_list)==0:
            raise fks_common.FKSProcessError('No matrix elements provided to '+\
                                                 'the function write_lh_order.')
            return
        
        # We assume the orders to be common to all Subprocesses
        
        orders = process_list[0].get('orders') 
        if 'QED' in orders.keys() and 'QCD' in orders.keys():
            QED=orders['QED']
            QCD=orders['QCD']
        elif 'QED' in orders.keys():
            QED=orders['QED']
            QCD=0
        elif 'QCD' in orders.keys():
            QED=0
            QCD=orders['QCD']
        else:
            QED, QCD = self.get_qed_qcd_orders_from_weighted(\
                    len(process_list[0].get('legs')),
                    orders['WEIGHTED'])

        replace_dict = {}
        replace_dict['mesq'] = 'CHaveraged'
        replace_dict['corr'] = ' '.join(process_list[0].\
                                                  get('perturbation_couplings'))
        replace_dict['irreg'] = 'CDR'
        replace_dict['aspow'] = QCD
        replace_dict['aepow'] = QED
        replace_dict['modelfile'] = './param_card.dat'
        replace_dict['params'] = 'alpha_s'
        proc_lines=[]
        for proc in process_list:
            proc_lines.append('%s -> %s' % \
                    (' '.join(str(l['id']) for l in proc['legs'] if not l['state']),
                     ' '.join(str(l['id']) for l in proc['legs'] if l['state'])))
        replace_dict['pdgs'] = '\n'.join(proc_lines)
        replace_dict['symfin'] = 'Yes'
        content = \
"#OLE_order written by MadGraph5_aMC@NLO\n\
\n\
MatrixElementSquareType %(mesq)s\n\
CorrectionType          %(corr)s\n\
IRregularisation        %(irreg)s\n\
AlphasPower             %(aspow)d\n\
AlphaPower              %(aepow)d\n\
NJetSymmetrizeFinal     %(symfin)s\n\
ModelFile               %(modelfile)s\n\
Parameters              %(params)s\n\
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

        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 20
        else:
            replace_dict['wavefunctionsize'] = 8

        # Extract glu_ij_lines
        ij_lines = self.get_ij_lines(fksborn)
        replace_dict['ij_lines'] = '\n'.join(ij_lines)

        # Extract den_factor_lines
        den_factor_lines = self.get_den_factor_lines(fksborn)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
    
        # Extract the number of FKS process
        replace_dict['nconfs'] = max(len(fksborn.get_fks_info_list()),1)

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/born_fks.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor


    def write_born_hel(self, writer, fksborn, fortran_model):
        """Export a matrix element to a born_hel.f file in MadFKS format"""

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
   
        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
    
        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        # Extract den_factor_lines
        den_factor_lines = self.get_den_factor_lines(fksborn)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
    
        # Extract the number of FKS process
        replace_dict['nconfs'] = len(fksborn.get_fks_info_list())

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/born_fks_hel.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return


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
                
                if m != n:
                    iflines += \
                    "c b_sf_%(iborn)3.3d links partons %(m)d and %(n)d \n\
                        %(iff)s ((m.eq.%(m)d .and. n.eq.%(n)d).or.(m.eq.%(n)d .and. n.eq.%(m)d)) then \n\
                        call sb_sf_%(iborn)3.3d(p_born,wgt)\n\n" \
                            %{'m':m, 'n': n, 'iff': iff, 'iborn': iborn}
                else:
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
    
    
        # Extract the number of FKS process
        replace_dict['nconfs'] = len(fksborn.get_fks_info_list())

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/b_sf_xxx_fks.inc')).read()
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
    
    #===============================================================================
    # write_fks_info_file
    #===============================================================================
    def write_nfksconfigs_file(self, writer, fksborn, fortran_model):
        """Writes the content of nFKSconfigs.inc, which just gives the
        total FKS dirs as a parameter.
        nFKSconfigs is always >=1 (use a fake configuration for LOonly)"""
        replace_dict = {}
        replace_dict['nconfs'] = max(len(fksborn.get_fks_info_list()), 1)
        content = \
"""      INTEGER FKS_CONFIGS
      PARAMETER (FKS_CONFIGS=%(nconfs)d)
      
"""   % replace_dict

        writer.writelines(content)

            
    #===============================================================================
    # write_fks_info_file
    #===============================================================================
    def write_fks_info_file(self, writer, fksborn, fortran_model): #test_written
        """Writes the content of fks_info.inc, which lists the informations on the 
        possible splittings of the born ME.
        nconfs is always >=1 (use a fake configuration for LOonly).
        The fake configuration use an 'antigluon' (id -21, color=8) as i_fks and 
        the last colored particle as j_fks."""

        replace_dict = {}
        fks_info_list = fksborn.get_fks_info_list()
        replace_dict['nconfs'] = max(len(fks_info_list), 1)

        # this is for processes with 'real' or 'all' as NLO mode 
        if len(fks_info_list) > 0:
            fks_i_values = ', '.join(['%d' % info['fks_info']['i'] \
                                                     for info in fks_info_list]) 
            fks_j_values = ', '.join(['%d' % info['fks_info']['j'] \
                                                     for info in fks_info_list]) 

            col_lines = []
            pdg_lines = []
            charge_lines = []
            fks_j_from_i_lines = []
            for i, info in enumerate(fks_info_list):
                col_lines.append( \
                    'DATA (PARTICLE_TYPE_D(%d, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                    % (i + 1, ', '.join('%d' % col for col in fksborn.real_processes[info['n_me']-1].colors) ))
                pdg_lines.append( \
                    'DATA (PDG_TYPE_D(%d, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                    % (i + 1, ', '.join('%d' % pdg for pdg in info['pdgs'])))
                charge_lines.append(\
                    'DATA (PARTICLE_CHARGE_D(%d, IPOS), IPOS=1, NEXTERNAL) / %s /'\
                    % (i + 1, ', '.join('%19.15fd0' % charg\
                                        for charg in fksborn.real_processes[info['n_me']-1].charges) ))
                fks_j_from_i_lines.extend(self.get_fks_j_from_i_lines(fksborn.real_processes[info['n_me']-1],\
                                                                       i + 1))
        else:
        # this is for 'LOonly', generate a fake FKS configuration with
        # - i_fks = nexternal, pdg type = -21 and color =8
        # - j_fks = the last colored particle
            bornproc = fksborn.born_matrix_element.get('processes')[0]
            pdgs = [l.get('id') for l in bornproc.get('legs')] + [-21]
            colors = [l.get('color') for l in bornproc.get('legs')] + [8]
            charges = [0.] * len(colors) 

            fks_i = len(colors)
            # use the first colored particle if it exists, or 
            # just the first
            fks_j=1
            for cpos, col in enumerate(colors[:-1]):
                if col != 1:
                    fks_j = cpos+1

            fks_i_values = str(fks_i)
            fks_j_values = str(fks_j)
            col_lines = ['DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join([str(col) for col in colors])]
            pdg_lines = ['DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join([str(pdg) for pdg in pdgs])]
            charge_lines = ['DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join('%19.15fd0' % charg for charg in charges)]
            fks_j_from_i_lines = ['DATA (FKS_J_FROM_I_D(1, %d, JPOS), JPOS = 0, 1)  / 1, %d /' \
                            % (fks_i, fks_j)]
            

        replace_dict['fks_i_line'] = "data fks_i_D / %s /" % fks_i_values
        replace_dict['fks_j_line'] = "data fks_j_D / %s /" % fks_j_values
        replace_dict['col_lines'] = '\n'.join(col_lines)
        replace_dict['pdg_lines'] = '\n'.join(pdg_lines)
        replace_dict['charge_lines'] = '\n'.join(charge_lines)
        replace_dict['fks_j_from_i_lines'] = '\n'.join(fks_j_from_i_lines)

        content = \
"""      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(%(nconfs)d), FKS_J_D(%(nconfs)d)
      INTEGER FKS_J_FROM_I_D(%(nconfs)d, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(%(nconfs)d, NEXTERNAL), PDG_TYPE_D(%(nconfs)d, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(%(nconfs)d, NEXTERNAL)
      
 %(fks_i_line)s 
 %(fks_j_line)s 

%(fks_j_from_i_lines)s

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
%(col_lines)s

C     
C     Particle type according to PDG:
C     
%(pdg_lines)s

C
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
%(charge_lines)s
"""   % replace_dict
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        writer.writelines(content)
    
        return True

 
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
    
        # Extract nwavefuncs (important to place after get_matrix_element_calls
        # so that 'me_id' is set)
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs
    
        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)

        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 20
        else:
            replace_dict['wavefunctionsize'] = 8
    
        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
    
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
    
        realfile = open(os.path.join(_file_path, \
                             'iolibs/template_files/realmatrix_fks.inc')).read()

        realfile = realfile % replace_dict
        
        # Write the file
        writer.writelines(realfile)
    
        return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor


    #===============================================================================
    # write_pdf_file
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
    
        pdf_vars, pdf_data, pdf_lines = \
                self.get_pdf_lines_mir(matrix_element, ninitial, False, False)
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines

        pdf_vars_mirr, pdf_data_mirr, pdf_lines_mirr = \
                self.get_pdf_lines_mir(matrix_element, ninitial, False, True)
        replace_dict['pdf_lines_mirr'] = pdf_lines_mirr
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/parton_lum_n_fks.inc')).read()
        file = file % replace_dict
    
        # Write the file
        writer.writelines(file)



    #===============================================================================
    # write_coloramps_file
    #===============================================================================
    #test written
    def write_coloramps_file(self, writer, mapconfigs, matrix_element, fortran_model):
        """Write the coloramps.inc file for MadEvent"""

        lines = []
        lines.append( "logical icolamp(%d,%d,1)" % \
                        (max(len(matrix_element.get('color_basis').keys()), 1),
                         len(mapconfigs)))
    
        lines += self.get_icolamp_lines(mapconfigs, matrix_element, 1)
    
        # Write the file
        writer.writelines(lines)
    
        return True


    #===============================================================================
    # write_leshouche_file
    #===============================================================================
    #test written
    def write_leshouche_file(self, writer, matrix_element, fortran_model):
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
    # write_configs_file
    #===============================================================================
    #test_written
    def write_configs_file(self, writer, matrix_element, fortran_model):
        """Write the configs.inc file for MadEvent"""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        lines = []
    
        iconfig = 0
    
        s_and_t_channels = []
        mapconfigs = []

        model = matrix_element.get('processes')[0].get('model')
#        new_pdg = model.get_first_non_pdg()
    
        base_diagrams = matrix_element.get('base_amplitude').get('diagrams')
        model = matrix_element.get('base_amplitude').get('process').get('model')
        minvert = min([max([len(vert.get('legs')) for vert in \
                            diag.get('vertices')]) for diag in base_diagrams])
    
        for idiag, diag in enumerate(base_diagrams):
            if any([len(vert.get('legs')) > minvert for vert in
                    diag.get('vertices')]):
                # Only 3-vertices allowed in configs.inc
                continue
            iconfig = iconfig + 1
            helas_diag = matrix_element.get('diagrams')[idiag]
            mapconfigs.append(helas_diag.get('number'))
            lines.append("# Diagram %d, Amplitude %d" % \
                         (helas_diag.get('number'),helas_diag.get('amplitudes')[0]['number']))
            # Correspondance between the config and the amplitudes
            lines.append("data mapconfig(%4d)/%4d/" % (iconfig,
                                                     helas_diag.get('amplitudes')[0]['number']))
    
            # Need to reorganize the topology so that we start with all
            # final state external particles and work our way inwards
            schannels, tchannels = helas_diag.get('amplitudes')[0].\
                                         get_s_and_t_channels(ninitial, model, 990)
    
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
    
        return iconfig, mapconfigs, s_and_t_channels

    
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
    # Helper functions
    #===============================================================================


    #===============================================================================
    # get_fks_j_from_i_lines
    #===============================================================================

    def get_fks_j_from_i_lines(self, me, i = 0): #test written
        """generate the lines for fks.inc describing initializating the
        fks_j_from_i array"""
        lines = []
        if not me.isfinite:
            for ii, js in me.fks_j_from_i.items():
                if js:
                    lines.append('DATA (FKS_J_FROM_I_D(%d, %d, JPOS), JPOS = 0, %d)  / %d, %s /' \
                             % (i, ii, len(js), len(js), ', '.join(["%d" % j for j in js])))
        else:
            lines.append('DATA (FKS_J_FROM_I_D(%d, JPOS), JPOS = 0, %d)  / %d, %s /' \
                     % (2, 1, 1, '1'))
        lines.append('')

        return lines


    #===============================================================================
    # get_leshouche_lines
    #===============================================================================
    def get_leshouche_lines(self, matrix_element, ime):
        #test written
        """Write the leshouche.inc file for MG4"""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    
        lines = []
        for iproc, proc in enumerate(matrix_element.get('processes')):
            legs = proc.get_legs_with_decays()
            lines.append("I   %4d   %4d       %s" % \
                         (ime, iproc + 1,
                          " ".join([str(l.get('id')) for l in legs])))
            for i in [1, 2]:
                lines.append("M   %4d   %4d   %4d      %s" % \
                         (ime, i, iproc + 1,
                          " ".join([ "%3d" % 0 ] * ninitial + \
                                   [ "%3d" % i ] * (nexternal - ninitial))))
    
            # Here goes the color connections corresponding to the JAMPs
            # Only one output, for the first subproc!
            if iproc == 0:
                # If no color basis, just output trivial color flow
                if not matrix_element.get('color_basis'):
                    for i in [1, 2]:
                        lines.append("C   %4d   %4d   1      %s" % \
                                 (ime, i, 
                                  " ".join([ "%3d" % 0 ] * nexternal)))
                    color_flow_list = []
                    nflow = 1
    
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
                            lines.append("C   %4d   %4d   %4d      %s" % \
                                 (ime, i + 1, cf_i + 1,
                                  " ".join(["%3d" % color_flow_dict[l.get('number')][i] \
                                            for l in legs])))

                    nflow = len(color_flow_list)

        nproc = len(matrix_element.get('processes'))
    
        return lines, nproc, nflow


    #===============================================================================
    # get_den_factor_lines
    #===============================================================================
    def get_den_factor_lines(self, fks_born):
        """returns the lines with the information on the denominator keeping care
        of the identical particle factors in the various real emissions"""
    
        lines = []
        info_list = fks_born.get_fks_info_list()
        if info_list:
            # if the reals have been generated, fill with the corresponding average factor
            lines.append('INTEGER IDEN_VALUES(%d)' % len(info_list))
            lines.append('DATA IDEN_VALUES /' + \
                         ', '.join(['%d' % ( 
                         fks_born.born_matrix_element.get_denominator_factor() ) \
                         for info in info_list]) + '/')
        else:
            # otherwise use the born
            lines.append('INTEGER IDEN_VALUES(1)')
            lines.append('DATA IDEN_VALUES / %d /' \
                    % fks_born.born_matrix_element.get_denominator_factor())

        return lines


    #===============================================================================
    # get_ij_lines
    #===============================================================================
    def get_ij_lines(self, fks_born):
        """returns the lines with the information on the particle number of the born 
        that splits"""
        info_list = fks_born.get_fks_info_list()
        lines = []
        if info_list:
            # if the reals have been generated, fill with the corresponding value of ij
            lines.append('INTEGER IJ_VALUES(%d)' % len(info_list))
            lines.append('DATA IJ_VALUES /' + \
                         ', '.join(['%d' % info['fks_info']['ij'] for info in info_list]) + '/')
        else:
            #otherwise just put the first leg
            lines.append('INTEGER IJ_VALUES(1)')
            lines.append('DATA IJ_VALUES / 1 /')

        return lines


    def get_pdf_lines_mir(self, matrix_element, ninitial, subproc_group = False,\
                          mirror = False): #test written
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
                pdf_lines = pdf_lines + "\nPD(IPROC) = 1d0\n"
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
                            if abs(pdgtopdf[initial_state]) <= 7:  
                                pdf_lines = pdf_lines + \
                                    ("%s%d=PDG2PDF(ABS(LPP(IB(%d))),%d*LP," + \
                                         "XBK(IB(%d)),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, ibeam, pdgtopdf[initial_state],
                                          ibeam, ibeam)
                            else:
                                # setting other partons flavours outside quark, gluon, photon to be 0d0
                                pdf_lines = pdf_lines + \
                                    ("c settings other partons flavours outside quark, gluon, photon to 0d0\n" + \
                                     "%s%d=0d0\n") % \
                                         (pdf_codes[initial_state],i + 1)                                
                        else:
                            if abs(pdgtopdf[initial_state]) <= 7:  
                                pdf_lines = pdf_lines + \
                                    ("%s%d=PDG2PDF(ABS(LPP(%d)),%d*LP," + \
                                         "XBK(%d),DSQRT(Q2FACT(%d)))\n") % \
                                         (pdf_codes[initial_state],
                                          i + 1, ibeam, pdgtopdf[initial_state],
                                          ibeam, ibeam)
                            else:
                                # setting other partons flavours outside quark, gluon, photon to be 0d0
                                pdf_lines = pdf_lines + \
                                    ("c settings other partons flavours outside quark, gluon, photon to 0d0\n" + \
                                     "%s%d=0d0\n") % \
                                         (pdf_codes[initial_state],i + 1)                                

                pdf_lines = pdf_lines + "ENDIF\n"

            # Add up PDFs for the different initial state particles
            pdf_lines = pdf_lines + "PD(0) = 0d0\nIPROC = 0\n"
            for proc in processes:
                process_line = proc.base_string()
                pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                pdf_lines = pdf_lines + "\nPD(IPROC) = "
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
        return pdf_definition_lines[:-1], pdf_data_lines[:-1], pdf_lines[:-1]


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

    #===============================================================================
    # write_ncombs_file
    #===============================================================================
    def write_ncombs_file(self, writer, matrix_element, fortran_model):
#        #test written
        """Write the ncombs.inc file for MadEvent."""
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    
        # ncomb (used for clustering) is 2^(nexternal)
        file = "       integer    n_max_cl\n"
        file = file + "parameter (n_max_cl=%d)" % (2 ** (nexternal+1))
    
        # Write the file
        writer.writelines(file)
   
        return True
    
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
    # write_colors_file
    #===========================================================================
    def write_colors_file(self, writer, matrix_element):
        """Write the get_color.f file for MadEvent, which returns color
        for all particles used in the matrix element."""

        try:
            matrix_elements=matrix_element.real_processes[0].matrix_element
        except IndexError:
            matrix_elements=[matrix_element.born_matrix_element]

        if isinstance(matrix_elements, helas_objects.HelasMatrixElement):
            matrix_elements = [matrix_elements]

        model = matrix_elements[0].get('processes')[0].get('model')

        # We need the both particle and antiparticle wf_ids, since the identity
        # depends on the direction of the wf.
        # loop on the real emissions
        wf_ids = set(sum([sum([sum([sum([[wf.get_pdg_code(),wf.get_anti_pdg_code()] \
                              for wf in d.get('wavefunctions')],[]) \
                              for d in me.get('diagrams')],[]) \
                              for me in [real_proc.matrix_element]],[])\
                              for real_proc in matrix_element.real_processes],[]))
        # and also on the born
        wf_ids = wf_ids.union(set(sum([sum([[wf.get_pdg_code(),wf.get_anti_pdg_code()] \
                              for wf in d.get('wavefunctions')],[]) \
                              for d in matrix_element.born_matrix_element.get('diagrams')],[])))

        # loop on the real emissions
        leg_ids = set(sum([sum([sum([[l.get('id') for l in \
                                p.get_legs_with_decays()] for p in \
                                me.get('processes')], []) for me in \
                                [real_proc.matrix_element]], []) for real_proc in \
                                matrix_element.real_processes],[]))
        # and also on the born
        leg_ids = leg_ids.union(set(sum([[l.get('id') for l in \
                                p.get_legs_with_decays()] for p in \
                                matrix_element.born_matrix_element.get('processes')], [])))
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
    
                lines.append("pmass(%3d,%4d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines.append("pwidth(%3d,%4d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines.append("pow(%3d,%4d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))
    
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





#=================================================================================
# Class for using the optimized Loop process
#=================================================================================
class ProcessOptimizedExporterFortranFKS(loop_exporters.LoopProcessOptimizedExporterFortranSA,\
                                         ProcessExporterFortranFKS):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

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
        clean =self.opt['clean']
        
        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(dir_path):
            if not mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(dir_path))
            shutil.copytree(os.path.join(mgme_dir, 'Template', 'NLO'), dir_path, True)
            # distutils.dir_util.copy_tree since dir_path already exists
            dir_util.copy_tree(pjoin(self.mgme_dir, 'Template', 'Common'),
                               dir_path)
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
                subprocess.call([os.path.join('bin', 'internal', 'clean_template'), 
                    '--web'], cwd=dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'internal', 'clean_template')], \
                                                                       cwd=dir_path)
                except Exception, why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(dir_path),why))
            #Write version info
            MG_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])

        # We must link the CutTools to the Library folder of the active Template
        self.link_CutTools(dir_path)
        # We must link the TIR to the Library folder of the active Template
        link_tir_libs=[]
        tir_libs=[]
        tir_include=[]
        for tir in self.all_tir:
            tir_dir="%s_dir"%tir
            libpath=getattr(self,tir_dir)
            libpath = self.link_TIR(os.path.join(self.dir_path, 'lib'),
                                              libpath,"lib%s.a"%tir,tir_name=tir)
            setattr(self,tir_dir,libpath)
            if libpath != "":
                if tir in ['pjfry','ninja','golem', 'samurai']:
                    # We should link dynamically when possible, so we use the original
                    # location of these libraries.
                    link_tir_libs.append('-L%s/ -l%s'%(libpath,tir))
                    tir_libs.append('%s/lib%s.$(libext)'%(libpath,tir))
                    # We must add the corresponding includes for golem, samurai
                    # and ninja
                    if tir in ['golem','samurai','ninja']:
                        trg_path = pjoin(os.path.dirname(libpath),'include')
                        to_include = misc.find_includes_path(trg_path,
                                                        self.include_names[tir])
                        if to_include is None:
                            logger.error(
'Could not find the include directory for %s, looking in %s.\n' % (tir ,str(trg_path))+
'Generation carries on but you will need to edit the include path by hand in the makefiles.')
                            to_include = '<Not_found_define_it_yourself>'
                        tir_include.append('-I %s'%to_include)
                else:
                    link_tir_libs.append('-l%s'%tir)
                    tir_libs.append('$(LIBDIR)lib%s.$(libext)'%tir)
            
        os.remove(os.path.join(self.dir_path,'SubProcesses','makefile_loop.inc'))
        cwd = os.getcwd()
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0
        filename = 'makefile_loop'
        calls = self.write_makefile_TIR(writers.MakefileWriter(filename),
                                 link_tir_libs,tir_libs,tir_include=tir_include)
        os.remove(os.path.join(self.dir_path,'Source','make_opts.inc'))
        dirpath = os.path.join(self.dir_path, 'Source')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0
        filename = 'make_opts'
        calls = self.write_make_opts(writers.MakefileWriter(filename),
                                                        link_tir_libs,tir_libs)
        # Return to original PWD
        os.chdir(cwd)

        cwd = os.getcwd()
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0
                                       
        # We add here the user-friendly MadLoop option setter.
        cpfiles= ["SubProcesses/MadLoopParamReader.f",
                  "Cards/MadLoopParams.dat",
                  "SubProcesses/MadLoopParams.inc"]

        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))
        
        shutil.copy(pjoin(self.dir_path, 'Cards','MadLoopParams.dat'),
                      pjoin(self.dir_path, 'Cards','MadLoopParams_default.dat'))

        
        
        if os.path.exists(pjoin(self.dir_path, 'Cards', 'MadLoopParams.dat')):          
                self.MadLoopparam = banner_mod.MadLoopParam(pjoin(self.dir_path, 
                                                  'Cards', 'MadLoopParams.dat'))
                # write the output file
                self.MadLoopparam.write(pjoin(self.dir_path,"SubProcesses",
                                                           "MadLoopParams.dat"))

        # We need minimal editing of MadLoopCommons.f
        MadLoopCommon = open(os.path.join(self.loop_dir,'StandAlone', 
                                    "SubProcesses","MadLoopCommons.inc")).read()
        writer = writers.FortranWriter(os.path.join(self.dir_path, 
                                             "SubProcesses","MadLoopCommons.f"))
        writer.writelines(MadLoopCommon%{
                                   'print_banner_commands':self.MadLoop_banner})
        writer.close()

        # link the files from the MODEL
        model_path = self.dir_path + '/Source/MODEL/'
        # Note that for the [real=] mode, these files are not present
        if os.path.isfile(os.path.join(model_path,'mp_coupl.inc')):
            ln(model_path + '/mp_coupl.inc', self.dir_path + '/SubProcesses')
        if os.path.isfile(os.path.join(model_path,'mp_coupl_same_name.inc')):
            ln(model_path + '/mp_coupl_same_name.inc', \
                                                self.dir_path + '/SubProcesses')

        # Write the cts_mpc.h and cts_mprec.h files imported from CutTools
        self.write_mp_files(writers.FortranWriter('cts_mprec.h'),\
                            writers.FortranWriter('cts_mpc.h'),)

        self.copy_python_files()


        # We need to create the correct open_data for the pdf
        self.write_pdf_opendata()


        # Return to original PWD
        os.chdir(cwd)
        
    def generate_virt_directory(self, loop_matrix_element, fortran_model, dir_name):
        """writes the V**** directory inside the P**** directories specified in
        dir_name"""

        cwd = os.getcwd()

        matrix_element = loop_matrix_element

        # Create the MadLoop5_resources directory if not already existing
        dirpath = os.path.join(dir_name, 'MadLoop5_resources')
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

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

        calls=self.write_loop_matrix_element_v4(None,matrix_element,fortran_model)
        
        # We need a link to coefs.inc from DHELAS
        ln(pjoin(self.dir_path, 'Source', 'DHELAS', 'coef_specs.inc'),
                                                        abspath=False, cwd=None)
    
        # The born matrix element, if needed
        filename = 'born_matrix.f'
        calls = self.write_bornmatrix(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model)

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        filename = "loop_matrix.ps"
        writers.FortranWriter(filename).writelines("""C Post-helas generation loop-drawing is not ready yet.""")
        plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(
              matrix_element.get('base_amplitude').get('loop_diagrams')[:1000]),
              filename,
              model=matrix_element.get('processes')[0].get('model'),
              amplitude='')
        logger.info("Drawing loop Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string(\
                                                          print_weighted=False))
        plot.draw()

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

        linkfiles = ['coupl.inc', 'mp_coupl.inc', 'mp_coupl_same_name.inc',
                     'cts_mprec.h', 'cts_mpc.h', 'MadLoopParamReader.f',
                     'MadLoopParams.inc','MadLoopCommons.f']

        for file in linkfiles:
            ln('../../%s' % file)


        os.system("ln -s ../../makefile_loop makefile")
        
# We should move to MadLoop5_resources directory from the SubProcesses
        ln(pjoin(os.path.pardir,os.path.pardir,'MadLoopParams.dat'),
                                              pjoin('..','MadLoop5_resources'))        

        linkfiles = ['mpmodule.mod']

        for file in linkfiles:
            ln('../../../lib/%s' % file)

        linkfiles = ['coef_specs.inc']

        for file in linkfiles:        
            ln('../../../Source/DHELAS/%s' % file)

        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls


    #===============================================================================
    # write_coef_specs
    #===============================================================================
    def write_coef_specs_file(self, max_loop_vertex_ranks):
        """ writes the coef_specs.inc in the DHELAS folder. Should not be called in the 
        non-optimized mode"""
        filename = os.path.join(self.dir_path, 'Source', 'DHELAS', 'coef_specs.inc')

        replace_dict = {}
        replace_dict['max_lwf_size'] = 4
        replace_dict['vertex_max_coefs'] = max(\
                [q_polynomial.get_number_of_coefs_for_rank(n) 
                    for n in max_loop_vertex_ranks])
        IncWriter=writers.FortranWriter(filename,'w')
        IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           % replace_dict)
        IncWriter.close()
    


            
