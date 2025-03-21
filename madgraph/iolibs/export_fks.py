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

from __future__ import absolute_import
from __future__ import division
import glob
import logging
import os
import re
import shutil
import subprocess
import string
import copy
import platform

import madgraph
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
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
import madgraph.various.shower_card as shower_mod

import aloha.create_aloha as create_aloha

import models.write_param_card as write_param_card
import models.check_param_card as check_param_card
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
from madgraph.iolibs.files import cp, ln, mv
from six.moves import range
from six.moves import zip

pjoin = os.path.join

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_fks')
if madgraph.ordering:
    set	= misc.OrderedSet

# the base to compute the orders_tag
orderstag_base = 100

def get_orderstag(ords):
    step = 1
    tag = 0
    for o in ords:
        tag += step*o
        step *= orderstag_base
    return tag


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
    def copy_fkstemplate(self, model):
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
                raise MadGraph5Error("No valid MG_ME path given for MG4 run directory creation.")
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(dir_path))
            misc.copytree(os.path.join(mgme_dir, 'Template', 'NLO'), dir_path, True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template', 'Common'),dir_path)
            # Copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards', card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to move " + card + ".dat to default")
            
        elif not os.path.isfile(os.path.join(dir_path, 'TemplateVersion.txt')):
            if not mgme_dir:
                raise MadGraph5Error("No valid MG_ME path given for MG4 run directory creation.")
        try:
            shutil.copy(os.path.join(mgme_dir, 'MGMEVersion.txt'), dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])
        
        #Ensure that the Template is clean
        if clean:
            logger.info('remove old information in %s' % os.path.basename(dir_path))
            if 'MADGRAPH_BASE' in os.environ:
                subprocess.call([os.path.join('bin', 'internal', 'clean_template'), 
                                    '--web'],cwd=dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'internal', 'clean_template')], \
                                                                       cwd=dir_path)
                except Exception as why:
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
        

        # Duplicate FO_analyse_card
        for card in ['FO_analyse_card']:
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

        # Copy the Pythia8 Sudakov tables (needed for MC@NLO-DELTA matching)
        shutil.copy(os.path.join(self.mgme_dir,'vendor','SudGen','sudakov.f'), \
                    os.path.join(self.dir_path,'SubProcesses','sudakov.f'),follow_symlinks=True)

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
                                   'print_banner_commands':self.MadLoop_banner},
                                            context={'collier_available':False})
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
        
        if model["running_elements"]:
            if not os.path.exists(pjoin(MG5DIR, 'Template',"Running")):
                raise Exception("Library for the running have not been installed. To install them please run \"install RunningCoupling\"")
                
            misc.copytree(pjoin(MG5DIR, 'Template',"Running"), 
                            pjoin(self.dir_path,'Source','RUNNING'))
        
        
        
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
          pjoin('various','systematics.py'),          
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

    def convert_model(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):

        super(ProcessExporterFortranFKS,self).convert_model(model, 
                                               wanted_lorentz, wanted_couplings)
        
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
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
    def write_maxproc_files(self, nmaxpdf, subproc_path):
        """write the c++ and fortran header files with the max number of pdf pairs
        """
        # fortran
        content = "      integer mxpdflumi\n      integer max_nproc\n      parameter(mxpdflumi=%d,max_nproc=%d)\n" \
                % (nmaxpdf, nmaxpdf)
        fout = open(pjoin(subproc_path, 'pineappl_maxproc.inc'), 'w')
        fout.write(content)
        fout.close()

        # c++
        content = "#define  __max_nproc__ %d" % (nmaxpdf)
        fout = open(pjoin(subproc_path, 'pineappl_maxproc.h'), 'w')
        fout.write(content)
        fout.close()



    #===============================================================================
    # write a initial states map, useful for the fast PDF NLO interface
    #===============================================================================
    def write_init_map(self, file_pos, initial_states):
        """ Write an initial state process map. Each possible PDF
        combination gets an unique identifier."""
        
        text=''
        i=0
        for i,e in enumerate(initial_states):
            text=text+str(i+1)+' '+str(len(e))
            for t in e:
                if len(t) ==1:
                    t.append(0)
                text=text+'   '
                try:
                    for p in t:
                        if p == None : p = 0
                        text=text+' '+str(p)
                except TypeError:
                        text=text+' '+str(t)
            text=text+'\n'
        
        ff = open(file_pos, 'w')
        ff.write(text)
        ff.close()

        return i+1

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
        raise fks_common.FKSProcessError()("write_coef_specs should be called only in the loop-optimized mode")
        
        
    #===============================================================================
    # generate_directories_fks
    #===============================================================================
    def generate_directories_fks(self, matrix_element, fortran_model, me_number,
                                    me_ntot, path=os.getcwd(),OLP='MadLoop'):
        """Generate the Pxxxxx_i directories for a subprocess in MadFKS,
        including the necessary matrix.f and various helper files"""
        proc = matrix_element.born_me['processes'][0]

        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')
        
        cwd = os.getcwd()
        try:
            os.chdir(path)
        except OSError as error:
            error_msg = "The directory %s should exist in order to be able " % path + \
                        "to \"export\" in it. If you see this error message by " + \
                        "typing the command \"export\" please consider to use " + \
                        "instead the command \"output\". "
            raise MadGraph5Error(error_msg) 
        
        calls = 0
        
        self.fksdirs = []
        #first make and cd the direcrory corresponding to the born process:
        borndir = "P%s" % \
        (matrix_element.born_me.get('processes')[0].shell_string())
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
            self.write_lh_order(filename, [matrix_element.born_me.get('processes')[0]], OLP)
        
        if matrix_element.virt_matrix_element:
                    calls += self.generate_virt_directory( \
                            matrix_element.virt_matrix_element, \
                            fortran_model, \
                            os.path.join(path, borndir))

#write the infortions for the different real emission processes
        sqsorders_list = \
            self.write_real_matrix_elements(matrix_element, fortran_model)

        filename = 'extra_cnt_wrapper.f'
        self.write_extra_cnt_wrapper(writers.FortranWriter(filename),
                                     matrix_element.extra_cnt_me_list, 
                                     fortran_model)
        for i, extra_cnt_me in enumerate(matrix_element.extra_cnt_me_list):
            replace_dict = {}

            den_factor_lines = self.get_den_factor_lines(matrix_element,
                                                         extra_cnt_me)
            replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)

            ij_lines = self.get_ij_lines(matrix_element)
            replace_dict['ij_lines'] = '\n'.join(ij_lines)

            filename = 'born_cnt_%d.f' % (i+1)
            self.write_split_me_fks(writers.FortranWriter(filename),
                                        extra_cnt_me, 
                                        fortran_model, 'cnt', '%d' % (i+1),
                                        replace_dict)

        self.write_pdf_calls(matrix_element, fortran_model)

        filename = 'nFKSconfigs.inc'
        self.write_nfksconfigs_file(writers.FortranWriter(filename), 
                                    matrix_element, 
                                    fortran_model)

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                              me_number)

        filename = 'fks_info.inc'
        # write_fks_info_list returns a set of the splitting types
        split_types = self.write_fks_info_file(writers.FortranWriter(filename), 
                                 matrix_element, 
                                 fortran_model)

        # update the splitting types
        self.proc_characteristic['splitting_types'] = list(\
                set(self.proc_characteristic['splitting_types']).union(\
                    split_types))

        filename = 'leshouche_info.dat'
        nfksconfs,maxproc,maxflow,nexternal=\
                self.write_leshouche_info_file(filename,matrix_element)

        # if no corrections are generated ([LOonly] mode), get 
        # these variables from the born
        if nfksconfs == maxproc == maxflow == 0:
            nfksconfs = 1
            (dummylines, maxproc, maxflow) = self.get_leshouche_lines(
                    matrix_element.born_me, 1)

        filename = 'leshouche_decl.inc'
        self.write_leshouche_info_declarations(
                              writers.FortranWriter(filename), 
                              nfksconfs,maxproc,maxflow,nexternal,
                              fortran_model)
        filename = 'genps.inc'
        ngraphs = matrix_element.born_me.get_number_of_amplitudes()
        ncolor = max(1,len(matrix_element.born_me.get('color_basis')))
        self.write_genps(writers.FortranWriter(filename),maxproc,ngraphs,\
                         ncolor,maxflow,fortran_model)

        filename = 'configs_and_props_info.dat'
        nconfigs,max_leg_number=self.write_configs_and_props_info_file(
                              filename, 
                              matrix_element)

        filename = 'configs_and_props_decl.inc'
        self.write_configs_and_props_info_declarations(
                              writers.FortranWriter(filename), 
                              nconfigs,max_leg_number,nfksconfs,
                              fortran_model)
        
        # For processes with only QCD splittings, write
        # the file with the mapping of born vs real diagrams
        # Otherwise, write a dummy file
        filename = 'real_from_born_configs.inc'
        if self.proc_characteristic['splitting_types'] == ['QCD']:
            self.write_real_from_born_configs(
                              writers.FortranWriter(filename), 
                              matrix_element,
                              fortran_model)
        else:
            self.write_real_from_born_configs_dummy(
                              writers.FortranWriter(filename), 
                              matrix_element,
                              fortran_model)

        filename = 'maxconfigs.inc'
        self.write_maxconfigs_file(writers.FortranWriter(filename),
                max(nconfigs,matrix_element.born_me.get_number_of_amplitudes()))

#write the wrappers for real ME's
        filename_me = 'real_me_chooser.f'
        filename_lum = 'parton_lum_chooser.f'
        self.write_real_wrappers(writers.FortranWriter(filename_me), 
                                 writers.FortranWriter(filename_lum),
                                   matrix_element, sqsorders_list,
                                   fortran_model)

        filename = 'get_color.f'
        self.write_colors_file(writers.FortranWriter(filename),
                               matrix_element)

        filename = 'nexternal.inc'
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'orders.inc'
        amp_split_orders, amp_split_size, amp_split_size_born = \
			   self.write_orders_file(
                            writers.FortranWriter90(filename),
                            matrix_element)

        filename = 'a0Gmuconv.inc'
        startfroma0 = self.write_a0gmuconv_file(
                            writers.FortranWriter(filename),
                            matrix_element)

        filename = 'rescale_alpha_tagged.f'
        self.write_rescale_a0gmu_file(
                            writers.FortranWriter(filename),
                            startfroma0, matrix_element, split_types)

        filename = 'orders.h'
        self.write_orders_c_header_file(
                            writers.CPPWriter(filename),
                            amp_split_size, amp_split_size_born)

        filename = 'amp_split_orders.inc'
        self.write_amp_split_orders_file(
                            writers.FortranWriter(filename),
                            amp_split_orders)
        self.proc_characteristic['ninitial'] = ninitial
        self.proc_characteristic['nexternal'] = max(self.proc_characteristic['nexternal'], nexternal)
        
        filename = 'maxparticles.inc'
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     nexternal)
        
        filename = 'pmass.inc'
        try:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.real_processes[0].matrix_element)
        except IndexError:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.born_me)

        #draw the diagrams
        self.draw_feynman_diagrams(matrix_element)

        linkfiles = ['BinothLHADummy.f',
                     'check_poles.f',
                     'check_sudakov.f',
                     'check_sudakov_angle2.f',
                     'momentum_reshuffling.f',
                     'MCmasses_HERWIG6.inc',
                     'MCmasses_HERWIGPP.inc',
                     'MCmasses_PYTHIA6Q.inc',
                     'MCmasses_PYTHIA6PT.inc',
                     'MCmasses_PYTHIA8.inc',
                     'add_write_info.f',
                     'coupl.inc',
                     'cuts.f',
                     'dummy_fct.f',
                     'FKS_params.dat',
                     'initial_states_map.dat',
                     'OLE_order.olc',
                     'FKSParams.f90',
                     'cuts.inc',
                     'unlops.inc',
                     'pythia_unlops.f',
                     'driver_mintMC.f',
                     'driver_mintFO.f',
                     'pineappl_interface.cc',
                     'pineappl_interface_dummy.f',
                     'pineappl_common.inc',
                     'reweight_pineappl.inc',
                     'fastjetfortran_madfks_core.cc',
                     'fastjetfortran_madfks_full.cc',
                     'fjcore.cc',
                     'fastjet_wrapper.f',
                     'fjcore.hh',
                     'fks_Sij.f',
                     'fks_powers.inc',
                     'fks_singular.f',
                     'splitorders_stuff.f',
                     'orderstags_glob.f',
                     'chooser_functions.f',
                     'veto_xsec.f',
                     'veto_xsec.inc',
                     'weight_lines.f',
                     'genps_fks.f',
                     'boostwdir2.f',
                     'madfks_mcatnlo.inc',
                     'open_output_files.f',
                     'open_output_files_dummy.f',
                     'HwU_dummy.f',
                     'madfks_plot.f',
                     'analysis_dummy.f',
                     'analysis_lhe.f',
                     'mint_module.f90',
                     'MC_integer.f',
                     'mint.inc',
                     'montecarlocounter.f',
                     'q_es.inc',
                     'recluster.cc',
                     'Boosts.h',
                     'reweight_xsec.f',
                     'reweight_xsec_events.f',
                     'reweight_xsec_events_pdf_dummy.f',
                     'iproc_map.f',
                     'run.inc',
                     'eepdf.inc',
                     'run_card.inc',
                     'setcuts.f',
                     'setscales.f',
                     'recmom.f',
                     'test_soft_col_limits.f',
                     'symmetry_fks_v3.f',
                     'vegas2.for',
                     'write_ajob.f',
                     'handling_lhe_events.f',
                     'write_event.f',
                     'fill_MC_mshell.f',
                     'cluster.f',
                     'randinit',
                     'pineappl_maxproc.inc',
                     'pineappl_maxproc.h',
                     'timing_variables.inc',
                     'pythia8_fortran_dummy.cc',
                     'pythia8_fortran.cc',
                     'pythia8_wrapper.cc',
                     'pythia8_control_setup.inc',
                     'pythia8_control.inc',
                     'dire_fortran.cc',
                     'LHAFortran_aMCatNLO.h',
                     'sudakov.f',
                     'hep_event_streams.inc',
                     'orderstag_base.inc',
                     'orderstags_glob.dat',
                     'polfit.f']

        if matrix_element.ewsudakov:
            linkfiles.append('ewsudakov_functions.f')
        else:
            linkfiles.append('ewsudakov_functions_dummy.f')

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

        return calls, amp_split_orders

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

    #===========================================================================
    #  create the run_card 
    #===========================================================================
    def create_shower_card(self, processes, history):
        """ """
 
        shower_card = shower_mod.ShowerCard()
        shower_card.create_default_for_process(self.proc_characteristic, 
                                            history,
                                            processes)
        
        shower_card.write(pjoin(self.dir_path, 'Cards', 'shower_card_default.dat'),
                          template=pjoin(MG5DIR, 'Template', 'NLO', 'Cards', 'shower_card.dat'))
        shower_card.write(pjoin(self.dir_path, 'Cards', 'shower_card.dat'),
                          template=pjoin(MG5DIR, 'Template', 'NLO', 'Cards', 'shower_card.dat'))


    def pass_information_from_cmd(self, cmd):
        """pass information from the command interface to the exporter.
           Please do not modify any object of the interface from the exporter.
        """
        self.proc_defs = cmd._curr_proc_defs
        if hasattr(cmd,'born_processes'):
            self.born_processes = cmd.born_processes
        else:
            self.born_processes = []
        return

    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """Finalize FKS directory by creating jpeg diagrams, html
                pages,proc_card_mg5.dat and madevent.tar.gz and create the MA5 card if
        necessary."""
        
        devnull = os.open(os.devnull, os.O_RDWR)
        try:
            res = misc.call([mg5options['lhapdf'], '--version'], \
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            res = 1
        if res != 0:
            logger.info('The value for lhapdf in the current configuration does not ' + \
                        'correspond to a valid executable.\nPlease set it correctly either in ' + \
                        'input/mg5_configuration or with "set lhapdf /path/to/lhapdf-config" ' + \
                        'and regenrate the process. \nTo avoid regeneration, edit the ' + \
                        ('%s/Cards/amcatnlo_configuration.txt file.\n' % self.dir_path ) + \
                        'Note that you can still compile and run aMC@NLO with the built-in PDFs\n')
        
        compiler_dict = {'fortran':  mg5options['fortran_compiler'],
                             'cpp':  mg5options['cpp_compiler'],
                             'f2py':  mg5options['f2py_compiler']}
        
        if 'nojpeg' in flaglist:
            makejpg = False
        else:
            makejpg = True
        output_dependencies = mg5options['output_dependencies']
        
        self.proc_characteristic['ew_sudakov'] = 'ewsudakov' in matrix_elements.keys() and \
                                                 matrix_elements['ewsudakov']
        
        self.proc_characteristic['grouped_matrix'] = False
        self.proc_characteristic['complex_mass_scheme'] = mg5options['complex_mass_scheme']
        self.proc_characteristic['nlo_mixed_expansion'] = mg5options['nlo_mixed_expansion']
        # determine perturbation order
        perturbation_order = []
        firstprocess = history.get('generate')
        order = re.findall(r"\[(.*)\]", firstprocess)
        if 'QED' in order[0]:
            perturbation_order.append('QED')
        if 'QCD' in order[0]:
            perturbation_order.append('QCD')
        self.proc_characteristic['perturbation_order'] = perturbation_order 
        
        self.create_proc_charac()

        filename = os.path.join(self.dir_path,'SubProcesses','orderstag_base.inc')
        self.write_orderstag_base_file(writers.FortranWriter(filename))

        self.create_run_card(matrix_elements.get_processes(), history)
        self.create_shower_card(matrix_elements.get_processes(), history)
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
            if (not os.path.exists(pjoin(MG5DIR, 'vendor', 'StdHEP', 'lib', 'libstdhep.a')) or \
                not os.path.exists(pjoin(MG5DIR, 'vendor', 'StdHEP', 'lib', 'libFmcfio.a'))) and \
                not os.path.exists(pjoin(MG5DIR, 'vendor', 'StdHEP','fail')):
                if 'FC' not in os.environ or not os.environ['FC']:
                    path = os.path.join(StdHep_path, 'src', 'make_opts')
                    text = open(path).read()
                    for base in base_compiler:
                        text = text.replace(base,'FC=%s' % fcompiler_chosen)
                    open(path, 'w').writelines(text)
                logger.info('Compiling StdHEP. This has to be done only once.')
                try:
                    misc.compile(cwd = pjoin(MG5DIR, 'vendor', 'StdHEP'))
                except Exception as error:
                    logger.debug(str(error))
                    logger.warning("StdHep failed to compiled. This forbids to run NLO+PS with PY6 and Herwig6")
                    logger.info("details on the compilation error are available on %s", pjoin(MG5DIR, 'vendor', 'StdHEP','fail'))
                    logger.info("if you want to retry the compilation automatically, you have to remove that file first")
                    with open(pjoin(MG5DIR, 'vendor', 'StdHEP','fail'),'w') as fsock:
                        fsock.write(str(error))
                else:
                    logger.info('Done.')
            if os.path.exists(pjoin(StdHep_path, 'lib', 'libstdhep.a')):
                #then link the libraries in the exported dir
                files.ln(pjoin(StdHep_path, 'lib', 'libstdhep.a'), \
                                         pjoin(self.dir_path, 'MCatNLO', 'lib'))
                files.ln(pjoin(StdHep_path, 'lib', 'libFmcfio.a'), \
                                         pjoin(self.dir_path, 'MCatNLO', 'lib'))

        elif output_dependencies == 'internal':
            StdHEP_internal_path = pjoin(self.dir_path,'Source','StdHEP')
            misc.copytree(StdHep_path,StdHEP_internal_path, symlinks=True)
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
            raise MadGraph5Error('output_dependencies option %s not recognized'\
                                                            %output_dependencies)
           
        # Create the default MadAnalysis5 cards
        if 'madanalysis5_path' in self.opt and not \
                self.opt['madanalysis5_path'] is None and not self.proc_defs is None:
            # When using 
            processes = sum([me.get('processes') if not isinstance(me, str) else [] \
                                for me in matrix_elements.get('matrix_elements')],[])

            # Try getting the processes from the generation info directly if no ME are
            # available (as it is the case for parallel generation
            if len(processes)==0:
                processes = self.born_processes
            if len(processes)==0:
                logger.warning(
"""MG5aMC could not provide to Madanalysis5 the list of processes generated.
As a result, the default card will not be tailored to the process generated.
This typically happens when using the 'low_mem_multicore_nlo_generation' NLO generation mode.""")
            # For now, simply assign all processes to each proc_defs.
            # That shouldn't really affect the default analysis card created by MA5
            self.create_default_madanalysis5_cards(
                history, self.proc_defs, [processes,]*len(self.proc_defs),
                self.opt['madanalysis5_path'], pjoin(self.dir_path,'Cards'),
                levels =['hadron'])





    def write_real_from_born_configs(self, writer, matrix_element, fortran_model):
        """Writes the real_from_born_configs.inc file that contains
        the mapping to go for a given born configuration (that is used
        e.g. in the multi-channel phase-space integration to the
        corresponding real-emission diagram, i.e. the real emission
        diagram in which the combined ij is split in i_fks and
        j_fks."""
        lines = []
        lines2 = []
        max_links = 0
        born_me = matrix_element.born_me
        for iFKS, conf in enumerate(matrix_element.get_fks_info_list()):
            iFKS = iFKS+1
            links = conf['fks_info']['rb_links']
            max_links = max(max_links,len(links))
            for i,diags in enumerate(links):
                if not i == diags['born_conf']:
                    print(links)
                    raise MadGraph5Error( "born_conf should be canonically ordered")
            real_configs = ', '.join(['%d' % int(diags['real_conf']+1) for diags in links])
            lines.append("data (real_from_born_conf(irfbc,%d),irfbc=1,%d) /%s/" \
                             % (iFKS,len(links),real_configs))

        # this is for 'LOonly' processes; in this case, a fake configuration 
        # with all the born diagrams is written
        if not matrix_element.get_fks_info_list():
            # compute (again) the number of configurations at the born
            base_diagrams = born_me.get('base_amplitude').get('diagrams')
            minvert = min([max([len(vert.get('legs')) for vert in \
                                    diag.get('vertices')]) for diag in base_diagrams])
    
            for idiag, diag in enumerate(base_diagrams):
                if any([len(vert.get('legs')) > minvert for vert in
                        diag.get('vertices')]):
                # Only 3-vertices allowed in configs.inc
                    continue
                max_links = max_links + 1
                
            real_configs=', '.join(['%d' % i for i in range(1, max_links+1)])
            lines.append("data (real_from_born_conf(irfbc,%d),irfbc=1,%d) /%s/" \
                             % (1,max_links,real_configs))

        lines2.append("integer irfbc")
        lines2.append("integer real_from_born_conf(%d,%d)" \
                         % (max_links, max(len(matrix_element.get_fks_info_list()),1)))
        # Write the file
        writer.writelines(lines2+lines)

    def write_real_from_born_configs_dummy(self, writer, matrix_element, fortran_model):
        """write a dummy file"""
        max_links = 10
        lines2 = []
        lines2.append("integer irfbc")
        lines2.append("integer real_from_born_conf(%d,%d)" \
                         % (max_links,len(matrix_element.get_fks_info_list())))
        # Write the file
        writer.writelines(lines2)


    def write_amp_split_orders_file(self, writer, amp_split_orders):
        """ write the include file with the information of the coupling power for the 
        differen entries in the amp_split array"""
        text = "integer iaso, amp_split_orders(%d, nsplitorders)\n" % len(amp_split_orders)

        for i, amp_orders in enumerate(amp_split_orders):
            text+= "data (amp_split_orders(%d, iaso), iaso=1,nsplitorders) / %s /\n" % \
                (i + 1, ', '.join(['%d' % o for o in amp_orders]))

        writer.writelines(text)

    def write_orderstag_file(self, splitorders, outdir):
        outfile = open(os.path.join(outdir, 'SubProcesses', 'orderstags_glob.dat'), 'w')
        outfile.write('%d\n' % len(splitorders))
        tags = ['%d' % get_orderstag(ords) for ords in splitorders]
        outfile.write(' '.join(tags) + '\n')
        outfile.close()

    def write_orderstag_base_file(self, writer):
        """write a small include file containing the 'base'
        to compute the orders_tag"""

        text = "integer orders_tag_base\n"
        text+= "parameter (orders_tag_base=%d)\n" % orderstag_base
        writer.writelines(text)


    def write_a0gmuconv_file(self, writer, matrix_element):
        """writes an include file with the informations about the 
        alpha0 < > gmu conversion, to be used when the process has
        tagged photons
        """

        bool_dict = {True: '.true.', False: '.false.'}
        bornproc = matrix_element.born_me['processes'][0]
        startfromalpha0 = False
        if any([l['is_tagged'] and l['id'] == 22 for l in bornproc['legs']]):
            if 'loop_qcd_qed_sm_a0' in bornproc['model'].get('modelpath'):
                startfromalpha0 = True

        text = 'logical  startfroma0\nparameter (startfroma0=%s)\n' % bool_dict[startfromalpha0]
        writer.writelines(text)
        return startfromalpha0

    def write_orders_c_header_file(self, writer, amp_split_size, amp_split_size_born):
        """writes the header file including the amp_split_size declaration for amcblast
	"""
        text = "#define __amp_split_size %d\n" % amp_split_size
        text+= "#define __amp_split_size_born %d" % amp_split_size_born

        writer.writelines(text)


    def write_rescale_a0gmu_file(self, writer, startfroma0, matrix_element, split_types):
        """writes the function that computes the rescaling factor needed in
        the case of external photons.
        If split types does not contain [QED] or if there are not tagged photons,
        dummy informations are filled
        """

        # get the model parameters
        params = sum([v for v in self.model.get('parameters').values()], [])
        parnames = [p.name.lower() for p in params]

        bornproc = matrix_element.born_me['processes'][0]
        # this is to ensure compatibility with standard processes
        if not any([l['is_tagged'] and l['id'] == 22 for l in bornproc['legs']])\
                or 'QED' not in split_types:
            to_check = []
            expr = '1d0'
            conv_pol = '0d0'
            conv_fin = '0d0'
        
        elif startfroma0:
            to_check = ['mdl_aewgmu', 'mdl_aew']
            base = 'mdl_aewgmu/mdl_aew'
            exp = 'qed_pow/2d0-ntag'
            expr = '(%s)**(%s)' % (base, exp)
            conv_fin = '(qed_pow - ntagph * 2d0) * MDL_ECOUP_DGMUA0_UV_EW_FIN_ * born_wgt'
            conv_pol = '(qed_pow - ntagph * 2d0) * MDL_ECOUP_DGMUA0_UV_EW_1EPS_ * born_wgt'
        else:
            to_check = ['mdl_aew', 'mdl_aew0']
            base = 'mdl_aew0/mdl_aew'
            exp = 'ntag'
            expr = '(%s)**(%s)' % (base, exp)
            conv_fin = '- ntagph * 2d0 * MDL_ECOUP_DGMUA0_UV_EW_FIN_ * born_wgt'
            conv_pol = '- ntagph * 2d0 * MDL_ECOUP_DGMUA0_UV_EW_1EPS_ * born_wgt'

        replace_dict = {'rescale_fact': expr,
                        'virtual_a0Gmu_conv_finite': conv_fin,
                        'virtual_a0Gmu_conv_pole': conv_pol}

        if not all(p in parnames for p in to_check):
            raise fks_common.FKSProcessError(
                    'Some parameters needed when there are tagged '+\
                    'photons cannot be found in the model.\n' +\
                    'Please load the correct model and restriction ' +\
                    '(e.g loop_qcd_qed_sm_Gmu-a0 or loop_qcd_qed_sm_a0-Gmu)')

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/rescale_alpha_tagged.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)



    def write_orders_file(self, writer, matrix_element):
        """writes the include file with the informations about coupling orders.
        In particular this file should contain the constraints requested by the user
        for all the orders which are split"""

        born_orders = {}
        for ordd, val in matrix_element.born_me['processes'][0]['born_sq_orders'].items():
            born_orders[ordd] = val 

        nlo_orders = {}
        for ordd, val in matrix_element.born_me['processes'][0]['squared_orders'].items():
            nlo_orders[ordd] = val
        
        split_orders = \
                matrix_element.born_me['processes'][0]['split_orders']

        pert_orders = \
                matrix_element.born_me['processes'][0]['perturbation_couplings']

        max_born_orders = {}
        max_nlo_orders = {}

        model = matrix_element.born_me['processes'][0]['model']

        # first get the max_born_orders
        if list(born_orders.keys()) == ['WEIGHTED']:
            # if user has not specified born_orders, check the 'weighted' for each
            # of the split_orders contributions
            wgt_ord_max = born_orders['WEIGHTED']
            squared_orders, amp_orders = matrix_element.born_me.get_split_orders_mapping()
            for sq_order in squared_orders:
                # put the numbers in sq_order in a dictionary, with as keys
                # the corresponding order name
                ord_dict = {}
                assert len(sq_order) == len(split_orders) 
                for o, v in zip(split_orders, list(sq_order)):
                    ord_dict[o] = v

                wgt = sum([v * model.get('order_hierarchy')[o] for \
                        o, v in ord_dict.items()])
                if wgt > wgt_ord_max:
                    continue

                for o, v in ord_dict.items():
                    try:
                        max_born_orders[o] = max(max_born_orders[o], v)
                    except KeyError:
                        max_born_orders[o] = v

        else:
            for o in [oo for oo in split_orders if oo != 'WEIGHTED']:
                try:
                    max_born_orders[o] = born_orders[o]
                except KeyError:
                    # if the order is not in born_orders set it to 1000
                    max_born_orders[o] = 1000
                try:
                    max_nlo_orders[o] = nlo_orders[o]
                except KeyError:
                    # if the order is not in born_orders set it to 1000
                    max_nlo_orders[o] = 1000

        # keep track also of the position of QED, QCD in the order array
        # might be useful in the fortran code
        qcd_pos = -1
        qed_pos = -1
        if 'QCD' in split_orders:
            qcd_pos = split_orders.index('QCD') + 1
        if 'QED' in split_orders:
            qed_pos = split_orders.index('QED') + 1

        # determine the size of the array that keeps track
        # of the different split orders, and the position 
        # of the different split order combinations in this array
        # to be written in orders_to_amp_split_pos.inc and
        #                  amp_split_pos_to_orders.inc
                          
        # the number of squared orders of the born ME
        amp_split_orders = []
        squared_orders, amp_orders = matrix_element.born_me.get_split_orders_mapping()
        amp_split_size_born =  len(squared_orders)
        amp_split_orders += squared_orders
        
        #then check the real emissions
        for realme in matrix_element.real_processes:
            squared_orders, amp_orders = realme.matrix_element.get_split_orders_mapping()
            for order in squared_orders:
                if not order in amp_split_orders:
                    amp_split_orders.append(order)

        # check also the virtual 
        #  may be needed for processes without real emissions, e.g. z > v v 
        #  Note that for a loop_matrix_element squared_orders has a different format
        #  (see the description of the get_split_orders_mapping function in loop_helas_objects)
        try:
            squared_orders, amp_orders = matrix_element.virt_matrix_element.get_split_orders_mapping()
            squared_orders = [so[0] for so in squared_orders]
            for order in squared_orders:
                if not order in amp_split_orders:
                    amp_split_orders.append(order)
        except AttributeError:
            pass

        # special treatment needed for the case when the sudakov
        # matrix elements are generated. This can be either the case
        # of LOonly=SDK (no virtual/reals, only sudakov)
        # or of [QCD SDK] virtual and reals, but of QCD origin.
        # In both case a coupling combination corresponding to 
        # born_orders + 2*QED must be added
        if  matrix_element.ewsudakov:
            # compute the born orders
            born_orders = []
            split_orders = matrix_element.born_me['processes'][0]['split_orders'] 
            for ordd in split_orders:
                born_orders.append(matrix_element.born_me['processes'][0]['born_sq_orders'][ordd])
            # increase the QED order
            born_orders[split_orders.index('QED')] += 2
            if tuple(born_orders) not in amp_split_orders:
                amp_split_orders.append(tuple(born_orders))

        amp_split_size=len(amp_split_orders)

        text = '! The orders to be integrated for the Born and at NLO\n'
        text += 'integer nsplitorders\n'
        text += 'parameter (nsplitorders=%d)\n' % len(split_orders)

        text += 'character*%d ordernames(nsplitorders)\n' % max([len(o) for o in split_orders])
        step = 5
        if len(split_orders) < step:
            text += 'data ordernames / %s /\n' % ', '.join(['"%3s"' % o for o in split_orders])
        else:
            # this file is linked from f77 and f90 so need to be smart about line splitting
            text += "INTEGER ORDERNAMEINDEX\n"
            for i in range(1,len(split_orders),step):
                start = i
                stop = i+step -1
                data = ', '.join(['"%3s"' % o for o in split_orders[start-1: stop]])
                if stop > len(split_orders):
                    stop = len(split_orders)
                text += 'data (ordernames(ORDERNAMEINDEX), ORDERNAMEINDEX=%s,%s)  / %s /\n' % (start, stop, data)

        text += 'integer born_orders(nsplitorders), nlo_orders(nsplitorders)\n'
        text += '! the order of the coupling orders is %s\n' % ', '.join(split_orders)
        text += 'data born_orders / %s /\n' % ', '.join([str(max_born_orders[o]) for o in split_orders])
        text += 'data nlo_orders / %s /\n' % ', '.join([str(max_nlo_orders[o]) for o in split_orders])
        text += '! The position of the QCD /QED orders in the array\n'
        text += 'integer qcd_pos, qed_pos\n'
        text += '! if = -1, then it is not in the split_orders\n'
        text += 'parameter (qcd_pos = %d)\n' % qcd_pos
        text += 'parameter (qed_pos = %d)\n' % qed_pos
        text += '! this is to keep track of the various \n'
        text += '! coupling combinations entering each ME\n'
        text += 'integer amp_split_size, amp_split_size_born\n'
        text += 'parameter (amp_split_size = %d)\n' % amp_split_size
        text += '! the first entries in the next line in amp_split are for the born \n'
        text += 'parameter (amp_split_size_born = %d)\n' % amp_split_size_born
        text += 'double precision amp_split(amp_split_size)\n'
        text += 'double complex amp_split_cnt(amp_split_size,2,nsplitorders)\n'
        text += 'common /to_amp_split/amp_split, amp_split_cnt\n'
        writer.line_length=132
        writer.writelines(text)

        return amp_split_orders, amp_split_size, amp_split_size_born


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

        if mass_particles:
            iflines_mass += 'else\n'
        else:
            iflines_mass = 'if (.true.) then\n'

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

        if width_particles:
            iflines_width += 'else\n'
        else:
            iflines_width = 'if (.true.) then\n'

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
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        max_iconfig=0
        max_leg_number=0

        ########################################################
        # this is for standard processes with [(real=)XXX]
        ########################################################
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

        ########################################################
        # this is for [LOonly=XXX]
        ########################################################
        if not matrix_element.get_fks_info_list():
            born_me = matrix_element.born_me
            # as usual, in this case we assume just one FKS configuration 
            # exists with diagrams corresponding to born ones X the ij -> i,j
            # splitting. Here j is chosen to be the last colored particle in
            # the particle list
            bornproc = born_me.get('processes')[0]
            colors = [l.get('color') for l in bornproc.get('legs')] 

            fks_i = len(colors)
            # use the last colored particle if it exists, or 
            # just the last
            fks_j=1
            for cpos, col in enumerate(colors):
                if col != 1:
                    fks_j = cpos+1
                    fks_j_id = [l.get('id') for l in bornproc.get('legs')][cpos]

            # for the moment, if j is initial-state, we do nothing
            if fks_j > ninitial:
                iFKS=1
                iconfig = 0
                s_and_t_channels = []
                mapconfigs = []
                base_diagrams = born_me.get('base_amplitude').get('diagrams')
                model = born_me.get('base_amplitude').get('process').get('model')
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
                    helas_diag = born_me.get('diagrams')[idiag]
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

                    #the first thing to write is the splitting ij -> i,j
                    lines.append("F   %4d   %4d   %4d   %4d" % \
                                     (iFKS,-1,iconfig,2))
                                     #(iFKS,last_leg.get('number'), iconfig, len(daughters)))
                    lines.append("D   %4d" % nexternal)
                    lines.append("D   %4d" % fks_j)
                    lines.append("S   %4d   %4d   %4d   %10d" % \
                                         (iFKS,-1, iconfig,fks_j_id)) 
                    # now we continue with all the other vertices of the diagrams;
                    # we need to shift the 'last_leg' by 1 and replace leg fks_j with -1

                    # Write out propagators for s-channel and t-channel vertices
                    allchannels = schannels
                    if len(tchannels) > 1:
                        # Write out tchannels only if there are any non-trivial ones
                        allchannels = schannels + tchannels

                    for vert in allchannels:
                        daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                        last_leg = vert.get('legs')[-1]
                        lines.append("F   %4d   %4d   %4d   %4d" % \
                                         (iFKS,last_leg.get('number')-1, iconfig, len(daughters)))

                        # legs with negative number in daughters have to be shifted by -1
                        for i_dau in range(len(daughters)):
                            if daughters[i_dau] < 0:
                                daughters[i_dau] += -1
                        # finally relable fks with -1 if it appears in daughters
                        if fks_j in daughters:
                            daughters[daughters.index(fks_j)] = -1
                        for d in daughters:
                            lines.append("D   %4d" % d)
                        if vert in schannels:
                            lines.append("S   %4d   %4d   %4d   %10d" % \
                                             (iFKS,last_leg.get('number')-1, iconfig,
                                              last_leg.get('id')))
                        elif vert in tchannels[:-1]:
                            lines.append("T   %4d   %4d   %4d   %10d" % \
                                             (iFKS,last_leg.get('number')-1, iconfig,
                                              abs(last_leg.get('id'))))

                        # update what the array sizes (mapconfig,iforest,etc) will be
                        max_leg_number = min(max_leg_number,last_leg.get('number')-1)
                    max_iconfig = max(max_iconfig,iconfig)
        
                # Write out number of configs
                lines.append("# Number of configs for nFKSprocess %d" % iFKS)
                lines.append("C   %4d   %4d   %4d" % (iFKS,0,iconfig))

                # write the props.inc information
                lines2.append("# ")
                particle_dict = born_me.get('processes')[0].get('model').\
                    get('particle_dict')
        
                for iconf, configs in enumerate(s_and_t_channels):
                    lines2.append("M   %4d   %4d   %4d   %10d " % \
                                    (iFKS,-1, iconf + 1, fks_j_id))
                    pow_part = 1 + int(particle_dict[fks_j_id].is_boson())
                    lines2.append("P   %4d   %4d   %4d   %4d " % \
                                    (iFKS,-1, iconf + 1, pow_part))
                    for vertex in configs[0] + configs[1][:-1]:
                        leg = vertex.get('legs')[-1]
                        if leg.get('id') not in particle_dict:
                            # Fake propagator used in multiparticle vertices
                            pow_part = 0
                        else:
                            particle = particle_dict[leg.get('id')]
        
                            pow_part = 1 + int(particle.is_boson())
        
                        lines2.append("M   %4d   %4d   %4d   %10d " % \
                                         (iFKS,leg.get('number')-1, iconf + 1, leg.get('id')))
                        lines2.append("P   %4d   %4d   %4d   %4d " % \
                                         (iFKS,leg.get('number')-1, iconf + 1, pow_part))
    
        # Write the file
        open(filename,'w').write('\n'.join(lines+lines2))

        return max_iconfig, max_leg_number


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


    def write_maxparticles_file(self, writer, maxparticles):
        """Write the maxparticles.inc file for MadEvent"""
        lines = "integer max_particles, max_branch\n"
        lines += "parameter (max_particles=%d) \n" % maxparticles
        lines += "parameter (max_branch=max_particles-1)"
        writer.writelines(lines)


    def write_maxconfigs_file(self, writer, maxconfigs):
        """Write the maxconfigs.inc file for MadEvent"""
        lines = "integer lmaxconfigs\n"
        lines += "parameter (lmaxconfigs=%d)" % maxconfigs
        writer.writelines(lines)


    def write_genps(self, writer, maxproc,ngraphs,ncolor,maxflow, fortran_model):
        """writes the genps.inc file
        """
        lines = []
        lines.append("include 'maxparticles.inc'")
        lines.append("include 'maxconfigs.inc'")
        lines.append("integer maxproc,ngraphs,ncolor,maxflow")
        lines.append("parameter (maxproc=%d,ngraphs=%d,ncolor=%d,maxflow=%d)" % \
                     (maxproc,ngraphs,ncolor,maxflow))
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
            (newlines, nprocs, nflows) = self.get_leshouche_lines(
                    matrix_element.real_processes[conf['n_me'] - 1].matrix_element, i + 1)
            lines.extend(newlines)
            maxproc = max(maxproc, nprocs)
            maxflow = max(maxflow, nflows)

        # this is for LOonly
        if not matrix_element.get_fks_info_list():
            (newlines, nprocs, nflows) = self.get_leshouche_lines_dummy(matrix_element.born_me, 1)
            lines.extend(newlines)

        # Write the file
        open(filename,'w').write('\n'.join(lines))

        return nfksconfs, maxproc, maxflow, nexternal


    def write_real_wrappers(self, writer_me, writer_lum, matrix_element, sqsolist, fortran_model):
        """writes the wrappers which allows to chose among the different real matrix elements
        and among the different parton luminosities and 
        among the various helper functions for the split-orders"""

        # the real me wrapper
        text = \
            """subroutine smatrix_real(p, wgt)
            implicit none
            include 'nexternal.inc'
            double precision p(0:3, nexternal)
            double precision wgt
            integer nfksprocess
            common/c_nfksprocess/nfksprocess
            """ 
        # the pdf wrapper
        text1 = \
            """\n\ndouble precision function dlum()
            implicit none
            integer nfksprocess
            common/c_nfksprocess/nfksprocess
            """

        if matrix_element.real_processes:
            for n, info in enumerate(matrix_element.get_fks_info_list()):
                text += \
                    """if (nfksprocess.eq.%(n)d) then
                    call smatrix%(n_me)d(p, wgt)
                    else""" % {'n': n + 1, 'n_me' : info['n_me']}
                text1 += \
                    """if (nfksprocess.eq.%(n)d) then
                    call dlum_%(n_me)d(dlum)
                    else""" % {'n': n + 1, 'n_me' : info['n_me']}

            text += \
                """
                write(*,*) 'ERROR: invalid n in real_matrix :', nfksprocess
                stop\n endif
                return \n end
                """
            text1 += \
                """
                write(*,*) 'ERROR: invalid n in dlum :', nfksprocess\n stop\n endif
                return \nend
                """
        else:
            text += \
                """
                wgt=0d0
                return
                end
                """
            text1 += \
                """
                call dlum_0(dlum)
                return
                end
                """

        # Write the file
        writer_me.writelines(text)
        writer_lum.writelines(text1)
        return 0


    def draw_feynman_diagrams(self, matrix_element):
        """Create the ps files containing the feynman diagrams for the born process,
        as well as for all the real emission processes"""

        filename = 'born.ps'
        plot = draw.MultiEpsDiagramDrawer(
                matrix_element.born_me.get('base_amplitude').get('diagrams'),
                filename,
                model=matrix_element.born_me.get('processes')[0].get('model'),
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
        
        sqsorders_list = []
        for n, fksreal in enumerate(matrix_element.real_processes):
            filename = 'matrix_%d.f' % (n + 1)
            ncalls, ncolors, nsplitorders, nsqsplitorders = \
                                    self.write_split_me_fks(\
                                        writers.FortranWriter(filename),
                                        fksreal.matrix_element, 
                                        fortran_model, 'real', "%d" % (n+1))
            sqsorders_list.append(nsqsplitorders)
        return sqsorders_list

        
    
    def write_extra_cnt_wrapper(self, writer, cnt_me_list, fortran_model):
        """write a wrapper for the extra born counterterms that may be 
        present e.g. if the process has gluon at the born
        """

        replace_dict = {'ncnt': max(len(cnt_me_list),1)}

        # this is the trivial case with no cnt.
        # fill everything with 0s (or 1 for color)
        if not cnt_me_list:
            replace_dict['cnt_charge_lines'] = \
                    "data (cnt_charge(1,i), i=1,nexternalB) / nexternalB * 0d0 /" 
            replace_dict['cnt_color_lines'] = \
                    "data (cnt_color(1,i), i=1,nexternalB) / nexternalB * 1 /" 
            replace_dict['cnt_pdg_lines'] = \
                    "data (cnt_pdg(1,i), i=1,nexternalB) / nexternalB * 0 /" 

            replace_dict['iflines'] = ''

        else:
            iflines = ''
            cnt_charge_lines = ''
            cnt_color_lines = ''
            cnt_pdg_lines = ''

            for i, cnt in enumerate(cnt_me_list):
                icnt = i+1
                if not iflines:
                    iflines = \
                       'if (icnt.eq.%d) then\n call sborn_cnt%d(p,cnts)\n' % (icnt, icnt)
                else:
                    iflines += \
                       'else if (icnt.eq.%d) then\n call sborn_cnt%d(p,cnts)\n' % (icnt, icnt)

                cnt_charge_lines += 'data (cnt_charge(%d,i), i=1,nexternalB) / %s /\n' % \
                        (icnt, ', '.join(['%19.15fd0' % l['charge'] for l in cnt['processes'][0]['legs']]))
                cnt_color_lines += 'data (cnt_color(%d,i), i=1,nexternalB) / %s /\n' % \
                        (icnt, ', '.join(['%d' % l['color'] for l in cnt['processes'][0]['legs']]))
                cnt_pdg_lines += 'data (cnt_pdg(%d,i), i=1,nexternalB) / %s /\n' % \
                        (icnt, ', '.join(['%d' % l['id'] for l in cnt['processes'][0]['legs']]))

            iflines += 'endif\n'

            replace_dict['iflines'] = iflines
            replace_dict['cnt_color_lines'] = cnt_color_lines
            replace_dict['cnt_charge_lines'] = cnt_charge_lines
            replace_dict['cnt_pdg_lines'] = cnt_pdg_lines

        file = open(pjoin(_file_path, \
            'iolibs/template_files/extra_cnt_wrapper_fks.inc')).read()

        file = file % replace_dict

        # Write the file
        writer.writelines(file)



    #===========================================================================
    # write_split_me_fks
    #===========================================================================
    def write_split_me_fks(self, writer, matrix_element, fortran_model,
                                    proc_type, proc_prefix='',start_dict={}):
        """Export a matrix element using the split_order format
        proc_type is either born, bhel, real or cnt,
        start_dict contains additional infos to be put in replace_dict"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            
        if 'sa_symmetry 'not  in self.opt:
            self.opt['sa_symmetry']=False


        # Add information relevant for FxFx matching:
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

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        replace_dict = {'global_variable':'', 'amp2_lines':''}
        if proc_prefix:
            replace_dict['proc_prefix'] = proc_prefix

        # update replace_dict according to start_dict
        for k,v in start_dict.items():
            replace_dict[k] = v

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 20
        else:
            replace_dict['wavefunctionsize'] = 8

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

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

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        if self.opt['export_format']=='standalone_msP':
        # For MadSpin need to return the AMP2
            amp2_lines = self.get_amp2_lines(matrix_element, [] )
            replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
            replace_dict['global_variable'] = "       Double Precision amp2(NGRAPHS)\n       common/to_amps/  amp2\n"

        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines, nb_temp_jamp = self.get_JAMP_lines(matrix_element)
        else:
            split_orders_name = matrix_element['processes'][0]['split_orders']
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            # add a comment line
            replace_dict['sqsplitorders']= \
    'C the values listed below are for %s\n' % ', '.join(split_orders_name)
            replace_dict['sqsplitorders']+='\n'.join(sqamp_so)           
            jamp_lines, nb_temp_jamp = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders)
            
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)    
        replace_dict['nb_temp_jamp'] = nb_temp_jamp
        
        if proc_type=='born':
            file = open(pjoin(_file_path, \
            'iolibs/template_files/bornmatrix_splitorders_fks.inc')).read()
        elif proc_type=='bhel':
            file = open(pjoin(_file_path, \
            'iolibs/template_files/born_hel_splitorders_fks.inc')).read()
        elif proc_type=='real':
            file = open(pjoin(_file_path, \
            'iolibs/template_files/realmatrix_splitorders_fks.inc')).read()
        elif proc_type=='cnt':
            # MZ this is probably not the best way to go
            file = open(pjoin(_file_path, \
            'iolibs/template_files/born_cnt_splitorders_fks.inc')).read()

        file = file % replace_dict

        # Write the file
        writer.writelines(file)

        return len(list([call for call in helas_calls if call.find('#') != 0])), ncolor, \
                replace_dict['nAmpSplitOrders'], replace_dict['nSqAmpSplitOrders']


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
                                                matrix_element.born_me, 0, 
                                                fortran_model)


    def generate_born_fks_files(self, matrix_element, fortran_model, me_number, path):
        """generates the files needed for the born amplitude in the P* directory, which will
        be needed by the P* directories"""
        pathdir = os.getcwd()

        born_me = matrix_element.born_me

        # the .inc files
        filename = 'born_conf.inc'
        nconfigs, mapconfigs, s_and_t_channels = \
                    self.write_born_conf_file(
                    writers.FortranWriter(filename),
                    born_me, fortran_model)

        filename = 'born_props.inc'
        self.write_born_props_file(
                    writers.FortranWriter(filename),
                    born_me, s_and_t_channels, fortran_model)

        filename = 'born_leshouche.inc'
        nflows = self.write_born_leshouche_file(writers.FortranWriter(filename),
                             born_me, fortran_model)

        filename = 'born_nhel.inc'
        self.write_born_nhel_file(writers.FortranWriter(filename),
                           born_me, nflows, fortran_model)

        filename = 'born_coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                                  mapconfigs, born_me, fortran_model)
        
        # the born ME's and color/charge links
        sqsorders_list = []
        filename = 'born.f'

        born_dict = {}
        born_dict['nconfs'] = max(len(matrix_element.get_fks_info_list()),1)

        den_factor_lines = self.get_den_factor_lines(matrix_element)
        born_dict['den_factor_lines'] = '\n'.join(den_factor_lines)

        ij_lines = self.get_ij_lines(matrix_element)
        born_dict['ij_lines'] = '\n'.join(ij_lines)

        #this is to skip computing amp_split_cnt if the process has no corrections
        if not matrix_element.real_processes:
            born_dict['skip_amp_cnt'] = 'goto 999 ! LOonly, no need to compute amp_split_cnt'
        else:
            born_dict['skip_amp_cnt'] = ''

        calls_born, ncolor_born, norders, nsqorders = \
            self.write_split_me_fks(writers.FortranWriter(filename),
                                    born_me, fortran_model, 'born', '',
                                    start_dict = born_dict)

        filename = 'born_maxamps.inc'
        maxamps = len(matrix_element.get('diagrams'))
        maxflows = ncolor_born
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           max([len(matrix_element.get('processes')) for me in \
                                matrix_element.born_me]),1)


        # the second call is for the born_hel file. use the same writer
        # function
        filename = 'born_hel.f'
        calls_born, ncolor_born, norders, nsqorders = \
            self.write_split_me_fks(writers.FortranWriter(filename),
                                    born_me, fortran_model, 'bhel', '',
                                    start_dict = born_dict)

        sqsorders_list.append(nsqorders)
    
        self.color_link_files = [] 
        for j in range(len(matrix_element.color_links)):
            filename = 'b_sf_%3.3d.f' % (j + 1)
            self.color_link_files.append(filename)
            self.write_b_sf_fks(writers.FortranWriter(filename),
                         matrix_element, j,
                         fortran_model)

        #write the sborn_sf.f and the b_sf_files
        filename = 'sborn_sf.f'
        self.write_sborn_sf(writers.FortranWriter(filename),
                            matrix_element,
                            nsqorders,
                            fortran_model)

        # finally the matrix elements needed for the sudakov approximation
        # of ew corrections. 
        # First, the squared amplitudes involving the goldstones

        filename = 'has_ewsudakov.inc'
        self.write_has_ewsudakov(writers.FortranWriter(filename), matrix_element.ewsudakov)

        filename = 'ewsudakov_haslo.inc'
        has_lo = self.write_ewsud_has_lo(writers.FortranWriter(filename), matrix_element)

        for j, sud_me in enumerate([me for me in matrix_element.sudakov_matrix_elements if me['type'] == 'goldstone']):
            filename = "ewsudakov_goldstone_me_%d.f" % (j + 1)
            self.write_sudakov_goldstone_me(writers.FortranWriter(filename),
                         sud_me['matrix_element'], j, fortran_model)
            # the file where the numeric derivative for the parameter renormalisation
            #   is computed 
            filename = "numder_ewsudakov_goldstone_me_%d.f" % (j + 1)
            self.write_numder_me(writers.FortranWriter(filename),
                         j, fortran_model)

        if matrix_element.ewsudakov:
            # the file where the numeric derivative for the parameter renormalisation
            #   is computed for the born 
            filename = "numder_born.f" 
            self.write_numder_me(writers.FortranWriter(filename),
                             None, fortran_model)

        # Then, the interferences with the goldstones or with the born amplitudes
        for j, sud_me in enumerate([me for me in matrix_element.sudakov_matrix_elements if me['type'] != 'goldstone']):
            filename = "ewsudakov_me_%d.f" % (j + 1)
            if sud_me['base_amp']:
                # remember, base_amp ==0 means the born, from 1 onwards it refers
                # to the sudakov matrix elements
                base_me = matrix_element.sudakov_matrix_elements[sud_me['base_amp']-1]['matrix_element']
            else:
                base_me = matrix_element.born_me
            self.write_sudakov_me(writers.FortranWriter(filename),
                         base_me, sud_me, j, fortran_model)

        # finally, the wrapper for all matrix elements needed
        # for the Sudakov approximation
        filename = "ewsudakov_wrapper.f"
        self.write_sudakov_wrapper(writers.FortranWriter(filename),
                                   matrix_element,has_lo,fortran_model)



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
        for line in open(contract_file_path):
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

        # We also need to write the overall maximum quantities for this group
        # of processes in 'global_specs.inc'. In aMCatNLO, there is always
        # only one process, so this is trivial
        self.write_global_specs(matrix_element, output_path=pjoin(dirpath,'global_specs.inc'))
        open('unique_id.inc','w').write(
"""      integer UNIQUE_ID
      parameter(UNIQUE_ID=1)""")

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
        if not orders:
            orders = {o : v / 2 for (o, v) in process_list[0].get('squared_orders').items()}
        if 'QED' in list(orders.keys()) and 'QCD' in list(orders.keys()):
            QED=orders['QED']
            QCD=orders['QCD']
        elif 'QED' in list(orders.keys()):
            QED=orders['QED']
            QCD=0
        elif 'QCD' in list(orders.keys()):
            QED=0
            QCD=orders['QCD']
        else:
            QED, QCD = fks_common.get_qed_qcd_orders_from_weighted(\
                    len(process_list[0].get('legs')),
                    process_list[0].get('model').get('order_hierarchy'),
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

        matrix_element = fksborn.born_me
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            
        # Add information relevant for FxFx matching:
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
        misc.sprint(self.proc_characteristic['max_n_matched_jets'], max_qcd_order,max_n_light_final_partons)
        self.proc_characteristic['max_n_matched_jets'] = max(
                               self.proc_characteristic['max_n_matched_jets'],
                                   min(max_qcd_order,max_n_light_final_partons))    
            
            
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
        jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_tmp_jamp


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
        logger.warning('This function should not be called')
    
        return len([call for call in helas_calls if call.find('#') != 0]), ncolor


    def write_born_hel(self, writer, fksborn, fortran_model):
        """Export a matrix element to a born_hel.f file in MadFKS format"""

        matrix_element = fksborn.born_me
        
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
        jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_tmp_jamp

        # Extract den_factor_lines
        den_factor_lines = self.get_den_factor_lines(fksborn)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
        misc.sprint(replace_dict['den_factor_lines'])
        replace_dict['den_factor_lines'] = ''

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
    def write_sborn_sf(self, writer, me, nsqorders, fortran_model):
        """Creates the sborn_sf.f file, containing the calls to the different 
        color linked borns"""
        
        replace_dict = {}
        color_links = me.color_links
        nlinks = len(color_links)

        replace_dict['nsqorders'] = nsqorders
        replace_dict['iflines_col'] = ''
         
        for i, c_link in enumerate(color_links):
            ilink = i+1
            iff = {True : 'if', False : 'elseif'}[i==0]
            m, n = c_link['link']
            if m != n:
                replace_dict['iflines_col'] += \
                "c link partons %(m)d and %(n)d \n\
                    %(iff)s ((m.eq.%(m)d .and. n.eq.%(n)d).or.(m.eq.%(n)d .and. n.eq.%(m)d)) then \n\
                    call sb_sf_%(ilink)3.3d(p_born,wgt_col)\n" \
                    % {'m':m, 'n': n, 'iff': iff, 'ilink': ilink}
            else:
                replace_dict['iflines_col'] += \
                "c link partons %(m)d and %(n)d \n\
                    %(iff)s (m.eq.%(m)d .and. n.eq.%(n)d) then \n\
                    call sb_sf_%(ilink)3.3d(p_born,wgt_col)\n" \
                    % {'m':m, 'n': n, 'iff': iff, 'ilink': ilink}

        
        if replace_dict['iflines_col']:
            replace_dict['iflines_col'] += 'endif\n'
        else:
            # this is when no color links are there
            replace_dict['iflines_col'] += 'write(*,*) \'Error in sborn_sf, no color links\'\nstop\n'

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/sborn_sf_fks.inc')).read()
        file = file % replace_dict
        writer.writelines(file)

    
    def get_sudakov_imag_power(self, base_me, sudakov_me):
        """return the exponent of I to account for the Z -> Chi replacement.
        Since the result is base * conj(sudakov), the exponent is the difference
        between the number of Chi's in base_me and in sudakov_me
        """
        base_ids = [leg['id'] for leg in base_me['processes'][0]['legs']]
        other_ids = [leg['id'] for leg in sudakov_me['processes'][0]['legs']]
        return base_ids.count(250) - other_ids.count(250)



    def get_chargeprod(self, charge_list, ninitial, n, m):
        """return the product of charges (as a string) of particles m and n.
        Special sign conventions may be needed for initial/final state particles
        """
        return charge_list[n - 1] * charge_list[m - 1]


    def write_sudakov_wrapper(self, writer, matrix_element, has_lo, fortran_model):
        """Write the wrapper for the sudakov matrix elements.
        has_lo is a dictionary with keys i=1,2, which tells if LO_i exists
        """

        born_me = matrix_element.born_me
        sudakov_list = matrix_element.sudakov_matrix_elements

        # need to know the number of splitorders at the born
        squared_orders, amp_orders = born_me.get_split_orders_mapping()
        amp_split_size_born =  len(squared_orders)

        # classify the different kind of matrix-elements
        goldstone_mes = [sud for sud in sudakov_list if sud['type'] == 'goldstone']
        non_goldstone_mes = [sud for sud in sudakov_list if sud['type'] != 'goldstone']
        non_goldstone_mes_lsc = [sud for sud in non_goldstone_mes if sud['type'] == 'cew'] 
        non_goldstone_mes_ssc_n1 = [sud for sud in non_goldstone_mes if sud['type'] == 'iz1'] 
        non_goldstone_mes_ssc_n2 = [sud for sud in non_goldstone_mes if sud['type'] == 'iz2'] 
        non_goldstone_mes_ssc_c = [sud for sud in non_goldstone_mes if sud['type'] == 'ipm2'] 

        replace_dict = {}

        # number of matrix elmenets
        replace_dict['ngoldstone_me'] = max(len(goldstone_mes),1)
        # identical particle factors
        if goldstone_mes:
            replace_dict['sdk_ident_goldstone'] = ",".join(
                    [str(me['matrix_element']['identical_particle_factor']) for me in goldstone_mes])
        else:
            replace_dict['sdk_ident_goldstone'] = "0"

        den_factor_lines = self.get_den_factor_lines(matrix_element)
        replace_dict['den_factor_lines'] = '\n'.join(den_factor_lines)
        replace_dict['bornspincol'] = born_me.get_denominator_factor() / born_me['identical_particle_factor']

        helicity_lines = self.get_helicity_lines(born_me)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract ncomb
        ncomb = born_me.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        ifsign_dict = {True: 1, False: -1}
        replace_dict['iflist'] = "iflist = (/%s/)" % ','.join([str(ifsign_dict[leg['state']]) for leg in born_me['processes'][0]['legs']])

        calls_to_me = ""

        # the calls to the goldstone matrix elements (for longitudinal polarisations and the born
        for i, me in enumerate(goldstone_mes+[{'matrix_element': born_me}]):
            # skip everything if there are no sudakovs
            if not matrix_element.ewsudakov:
                continue


            if i==len(goldstone_mes):
                # the last one (or only one if no MEs with goldstones exists), will use the born
                if goldstone_mes:
                    calls_to_me += "else\n"
                calls_to_me += "call sborn_onehel(p,nhel(1,ihel),ihel,ans_summed)\n"
                calls_to_me += "comp_idfac = 1d0\n"
                par_ren = "par_ren_sborn_onehel"
                i = -1 # so that i+1 is 0
            else:
                # these will use the ME's with goldstones
                if i==0:
                    # the first one
                    calls_to_me += "if"
                else:
                    calls_to_me += "else if"

                conditions = ["nhel(%d,ihel).eq.0" % (leg['number']) for leg in me['legs']]
                calls_to_me += " (%s) then\n" % ".and.".join(conditions)
                calls_to_me += "call EWSDK_GOLD_ME_%d(p,nhel(1,ihel),ans_summed)\n" % (i + 1)
                calls_to_me += "comp_idfac = compensate_identical_factor(%d)\n" % (i + 1)
                par_ren = "par_ren_EWSDK_GOLD_ME_%d" % (i + 1)

            # here the calls to all contributions where the particles of base_amp are not changed
            # these are corrections on top of the LO1
            if has_lo[1]:
                calls_to_me += "pdglist = (/%s/)\n" % ','.join([str(leg['id']) for leg in me['matrix_element']['processes'][0]['legs']])
                calls_to_me += "C the LSC term (diagonal)\n" 
                calls_to_me += "AMP_SPLIT_EWSUD_LSC(:) = AMP_SPLIT_EWSUD_LSC(:)+AMP_SPLIT_EWSUD(:)*get_lsc_diag(pdglist,nhel(1,ihel),iflist,invariants)*comp_idfac\n"
                calls_to_me += "C the SSC term (neutral/diagonal)\n" 
                calls_to_me += "AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)+AMP_SPLIT_EWSUD(:)*get_ssc_n_diag(pdglist,nhel(1,ihel),iflist,invariants)*comp_idfac\n"
                calls_to_me += "C the C term (diagonal)\n" 
                calls_to_me += "AMP_SPLIT_EWSUD_XXC(:) = AMP_SPLIT_EWSUD_XXC(:)+AMP_SPLIT_EWSUD(:)*get_xxc_diag(pdglist,nhel(1,ihel),iflist,invariants)*comp_idfac\n"

            # terms of QCD origin on top of LO2, if it exists
            if has_lo[2]:
                calls_to_me += "C the terms stemming from QCD corrections on top of the LO2\n" 
#                calls_to_me += "AMP_SPLIT_EWSUD_QCD(:) = AMP_SPLIT_EWSUD_QCD(:)+AMP_SPLIT_EWSUD_LO2(:)*get_qcd_lo2(pdglist,nhel(1,ihel),iflist,invariants)*comp_idfac\n"
                calls_to_me += "DO IAMP = 1, AMP_SPLIT_SIZE\n"
                calls_to_me += "  AMP_SPLIT_EWSUD_QCD(IAMP) = AMP_SPLIT_EWSUD_QCD(IAMP)+AMP_SPLIT_EWSUD_LO2(IAMP)*GET_QCD_LO2(PDGLIST,NHEL(1,IHEL),IFLIST,INVARIANTS,IAMP)*COMP_IDFAC\n"
                calls_to_me += "ENDDO\n"

            # the parameter renormalisation needs to be written in any case, and it will check internally
            # what parameters to take into account base on ewsudakov_haslo.inc
            calls_to_me += "C the parameter renormalisation\n"
            calls_to_me += "call %s(P,nhel(1,ihel),ihel,invariants)\n" % par_ren
            calls_to_me += "AMP_SPLIT_EWSUD_PAR(:) = AMP_SPLIT_EWSUD_PAR(:)+AMP_SPLIT_EWSUD(:)*comp_idfac\n"

            # now the call to the LSC and C non-diagonal, if LO1 exists
            if not has_lo[1]: 
                continue

            mes_same_charge_lsc = [me for me in non_goldstone_mes_lsc if me['base_amp'] == i+1]
            if mes_same_charge_lsc:
                calls_to_me += "C LSC and C non diag\n"
            for mesc in mes_same_charge_lsc:
                idx = non_goldstone_mes.index(mesc)
                calls_to_me += "call EWSDK_ME_%d(p,nhel(1,ihel),ans_summed)\n" % (idx + 1)
                calls_to_me += "pdglist_oth = (/%s/)\n" % ','.join([str(leg['id']) for leg in mesc['matrix_element']['processes'][0]['legs']])
                calls_to_me += "AMP_SPLIT_EWSUD_LSC(:) = AMP_SPLIT_EWSUD_LSC(:)+AMP_SPLIT_EWSUD(:)*get_lsc_nondiag(pdglist,nhel(1,ihel),iflist,invariants,%d,%d,%d)*comp_idfac\n" % \
                                        (mesc['legs'][0]['number'],mesc['pdgs'][0][0], mesc['pdgs'][1][0]) # old and new pdg of the leg that changes
                calls_to_me += "AMP_SPLIT_EWSUD_XXC(:) = AMP_SPLIT_EWSUD_XXC(:)+AMP_SPLIT_EWSUD(:)*get_xxc_nondiag(pdglist,nhel(1,ihel),iflist,invariants,%d,%d,%d)*comp_idfac\n" % \
                                        (mesc['legs'][0]['number'],mesc['pdgs'][0][0], mesc['pdgs'][1][0]) # old and new pdg of the leg that changes

            # now the calls to the SSC non-diagonal, with one particle different wrt base amp
            mes_same_charge_ssc1 = [me for me in non_goldstone_mes_ssc_n1 if me['base_amp'] == i+1]
            if mes_same_charge_ssc1:
                calls_to_me += "C SSC non diag #1\n"
            for mesc in mes_same_charge_ssc1:
                idx = non_goldstone_mes.index(mesc)
                calls_to_me += "call EWSDK_ME_%d(p,nhel(1,ihel),ans_summed)\n" % (idx + 1)
                calls_to_me += "pdglist_oth = (/%s/)\n" % ','.join([str(leg['id']) for leg in mesc['matrix_element']['processes'][0]['legs']])
                calls_to_me += "AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)+AMP_SPLIT_EWSUD(:)*get_ssc_n_nondiag_1(pdglist,nhel(1,ihel),iflist,invariants,%d,%d,%d)*comp_idfac\n" % \
                                        (mesc['legs'][0]['number'],mesc['pdgs'][0][0], mesc['pdgs'][1][0]) # number, old and new pdg of the leg that changes

            # now the calls to the SSC non-diagonal, with two particles different wrt base amp
            mes_same_charge_ssc2 = [me for me in non_goldstone_mes_ssc_n2 if me['base_amp'] == i+1]
            if mes_same_charge_ssc2:
                calls_to_me += "C SSC non diag #2\n"
            for mesc in mes_same_charge_ssc2:
                idx = non_goldstone_mes.index(mesc)
                calls_to_me += "call EWSDK_ME_%d(p,nhel(1,ihel),ans_summed)\n" % (idx + 1)
                calls_to_me += "pdglist_oth = (/%s/)\n" % ','.join([str(leg['id']) for leg in mesc['matrix_element']['processes'][0]['legs']])
                calls_to_me += "AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)+AMP_SPLIT_EWSUD(:)*get_ssc_n_nondiag_2(pdglist,nhel(1,ihel),iflist,invariants,%d,%d,%d,%d,%d,%d)*comp_idfac\n" % \
                                        (mesc['legs'][0]['number'],mesc['pdgs'][0][0], mesc['pdgs'][1][0], \
                                         mesc['legs'][1]['number'],mesc['pdgs'][0][1], mesc['pdgs'][1][1]) # number, old and new pdg of the legs that change

            # SSC calls, charged
            mes_ssc_charged = [me for me in non_goldstone_mes_ssc_c if me['base_amp'] == i+1]
            if mes_ssc_charged:
                calls_to_me += "C the SSC terms (charged)\n"
            for messc in mes_ssc_charged:
                idx = non_goldstone_mes.index(messc)
                calls_to_me += "call EWSDK_ME_%d(p,nhel(1,ihel),ans_summed)\n" % (idx + 1)
                calls_to_me += "pdglist_oth = (/%s/)\n" % ','.join([str(leg['id']) for leg in messc['matrix_element']['processes'][0]['legs']])
                calls_to_me += "AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)+AMP_SPLIT_EWSUD(:)*get_ssc_c(%d,%d,pdglist,%d,%d,nhel(1,ihel),iflist,invariants)*comp_idfac\n" % \
                                (messc['legs'][0]['number'], messc['legs'][1]['number'], messc['pdgs'][1][0], messc['pdgs'][1][1])

        if goldstone_mes:
            calls_to_me += "endif\n"
            
        replace_dict['calls_to_me'] = calls_to_me

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/ewsudakov_wrapper.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return 



    #===============================================================================
    # write_sudakov_goldstone_me
    #===============================================================================
    def write_numder_me(self, writer, ime, fortran_model):
        """Create the file where the derivative of the ime-th sudakov matrix element 
        (or of the Born, if ime=None) is computed
        """

        replace_dict = {}
        
        if ime != None:
            replace_dict['mename'] = 'EWSDK_GOLD_ME_%d' % (ime + 1)
            replace_dict['hell'] = ''
        else:
            replace_dict['mename'] = 'SBORN_ONEHEL'
            replace_dict['hell'] = 'hell,'

        if not 'Gmu' in  self.model.get('name'):
            logger.warning('Warning, the parameter renormalisation should be done in the alpha(MZ) scheme')
            file = open(os.path.join(_file_path, \
                              'iolibs/template_files/ewsudakov_numder_me_alphamz.inc')).read()
        else:
            logger.warning('Warning, the parameter renormalisation should be done in the Gmu scheme')
            file = open(os.path.join(_file_path, \
                              'iolibs/template_files/ewsudakov_numder_me_gmu.inc')).read()

        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return 



    def write_has_ewsudakov(self, writer, has_sudakov):
        """Write the include file which tells whether the process
        has been generated with or without Sudakov matrix elements
        """

        bool_dict = {True: '.true.', False: '.false.'}
        text = "      logical has_ewsudakov\n      parameter (has_ewsudakov=%s)\n" % \
                bool_dict[has_sudakov]
        
        # Write the file
        writer.writelines(text)
    
        return 


    def write_ewsud_has_lo(self, writer, matrix_element):
        """write an include file with the information whether the matrix element
        has contributions from LO1 and has a LO2, and the corresponding 
        position (iamp)
        """

        if matrix_element.ewsudakov:
            # get the coupling combination of the born
            squared_orders_born, amp_orders = matrix_element.born_me.get_split_orders_mapping()
            split_orders = \
                    matrix_element.born_me['processes'][0]['split_orders']

            # compute the born orders
            born_orders = []
            split_orders = matrix_element.born_me['processes'][0]['split_orders'] 
            for ordd in split_orders:
                born_orders.append(matrix_element.born_me['processes'][0]['born_sq_orders'][ordd])

            # check that there is at most one coupling combination
            # that satisfies the born_orders constraints 
            # (this is a limitation of the current implementation of the EW sudakov
            nborn = 0
            for orders in squared_orders_born:
                if all([orders[i] <= born_orders[i] for i in range(len(born_orders))]):
                    nborn += 1

            if nborn > 1:
                raise MadGraph5Error("ERROR: Sudakov approximation does not support cases where" + \
                        " the Born has more than one coupling combination, found %d)" % nborn)

            # now we can see if the process has a LO1
            has_lo1 = bool(nborn)
            if has_lo1:
                lo1_pos = squared_orders_born.index(tuple(born_orders)) + 1
            else:
                lo1_pos = -100

            # now determine the LO2 orders
            lo2_orders = born_orders
            lo2_orders[split_orders.index('QCD')] += -2
            lo2_orders[split_orders.index('QED')] += 2

            has_lo2 = tuple(lo2_orders) in squared_orders_born

            if has_lo2:
                lo2_pos = squared_orders_born.index(tuple(lo2_orders)) + 1
            else:
                lo2_pos = -100
        else:
            has_lo1 = False
            has_lo2 = False
            lo1_pos = -100
            lo2_pos = -100

        bool_dict = {True: '.true.', False: '.false.'}

        text = "      logical has_lo1, has_lo2\n      parameter (has_lo1=%s)\n      parameter (has_lo2=%s)\n" % \
                        (bool_dict[has_lo1], bool_dict[has_lo2])
        text+= "      integer lo1_pos, lo2_pos\n      parameter (lo1_pos=%d)\n      parameter (lo2_pos=%d)\n" % \
                        (lo1_pos, lo2_pos)
        
        # Write the file
        writer.writelines(text)

        return {1: has_lo1, 2: has_lo2}




    #===============================================================================
    # write_sudakov_goldstone_me
    #===============================================================================
    def write_sudakov_goldstone_me(self, writer, sudakov_me, ime, fortran_model):
        """Create the sudakov_goldstone_me_*.f file with external goldstone bosones
        for the sudakov approximation of EW corrections
        """

        matrix_element = copy.copy(sudakov_me)

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        replace_dict = {}
        
        replace_dict['ime'] = ime + 1
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines 
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(sudakov_me)
        replace_dict['process_lines'] = self.get_process_info_lines(sudakov_me)

        # Extract den_factor_lines
        den_factor = matrix_element.get_denominator_factor()
        replace_dict['den_factor'] = den_factor
    
        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 20
        else:
            replace_dict['wavefunctionsize'] = 8
    
        # Extract nwavefuncs (this is for the sudakov me)
        nwavefuncs = sudakov_me.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs
    
        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        # Extract helas calls of the base  matrix element
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        replace_dict['helas_calls'] = "\n".join(helas_calls).replace('AMP','AMP')

        # Extract JAMP lines
        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines = self.get_JAMP_lines(matrix_element)
        else:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            replace_dict['sqsplitorders']='\n'.join(sqamp_so)           
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders)

        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_tmp_jamp

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/ewsudakov_goldstone_splitorders_fks.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return 



    #===============================================================================
    # write_sudakov_me
    #===============================================================================
    def write_sudakov_me(self, writer, base_me, sudakov, ime, fortran_model):
        """Create the sudakov_me_*.f file for the sudakov approximation of EW
        corrections
        """

        sudakov_me = sudakov['matrix_element']
        ibase_me = sudakov['base_amp']
        pdgs = copy.copy(sudakov['pdgs'])
        legs = copy.copy(sudakov['legs'])

        matrix_element = copy.copy(base_me)
        model = matrix_element.get('processes')[0].get('model')

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        replace_dict = {}
        
        replace_dict['ime'] = ime + 1
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines 
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(sudakov_me)
        replace_dict['process_lines'] = "C  Sudakov approximation for the interference " + \
         "between\n" + self.get_process_info_lines(base_me) + "\nC" + \
         " and\n" + self.get_process_info_lines(sudakov_me)

        # Extract den_factor_lines
        den_factor = base_me.get_denominator_factor()
        replace_dict['den_factor'] = den_factor
    
        # Extract ngraphs
        ngraphs1 = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs1'] = ngraphs1
        ngraphs2 = sudakov_me.get_number_of_amplitudes()
        replace_dict['ngraphs2'] = ngraphs2

        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 20
        else:
            replace_dict['wavefunctionsize'] = 8
    
        # Extract nwavefuncs (take the max of base_me and sudakov me)
        nwavefuncs1 = matrix_element.get_number_of_wavefunctions()
        nwavefuncs2 = sudakov_me.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = max([nwavefuncs1, nwavefuncs2])
    
        # Extract ncolor
        ncolor1 = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor1'] = ncolor1
        ncolor2 = max(1, len(sudakov_me.get('color_basis')))
        replace_dict['ncolor2'] = ncolor2

        # compute the color matrix between basis of the Born and of the Sudakov
        color_matrix= color_amp.ColorMatrix(matrix_element.get('color_basis'), sudakov_me.get('color_basis'))
    
        # Extract color data lines
        color_data_lines = self.get_color_data_lines_from_color_matrix(color_matrix)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines)

        # the power of the imaginary unit to compensate for the neutral Goldstones
        replace_dict['imag_power'] = self.get_sudakov_imag_power(base_me, sudakov_me)

        # Extract helas calls of the base  matrix element
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        replace_dict['helas_calls1'] = "\n".join(helas_calls).replace('AMP','AMP1')

        # Extract helas calls of the sudakov matrix element
        helas_calls = fortran_model.get_matrix_element_calls(\
                    sudakov_me)
        replace_dict['helas_calls2'] = "\n".join(helas_calls).replace('AMP','AMP2')
    
        # Extract JAMP lines
        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines = self.get_JAMP_lines(matrix_element)
        else:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            replace_dict['sqsplitorders']='\n'.join(sqamp_so)           
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders,
                                                        JAMP_format="JAMP1(%s,{0})")

        replace_dict['jamp1_lines'] = '\n'.join(jamp_lines).replace('AMP(', 'AMP1(') 
        replace_dict['nb_temp_jamp1'] = nb_tmp_jamp

        # now the jamp for the sudakov me
        # NOTE: this ASSUMES that the splitorders of the sudakov and of the born me
        # are the same
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines = self.get_JAMP_lines(sudakov_me)
        else:
            squared_orders, amp_orders = sudakov_me.get_split_orders_mapping()
            # safety check
            if replace_dict['nAmpSplitOrders'] != len(amp_orders) or \
               replace_dict['nSqAmpSplitOrders'] != len(squared_orders) or \
               replace_dict['nSplitOrders'] != len(split_orders):
                raise MadGraph5Error("ERROR in write_sudakov_me (%d,%d,%d) != (%d,%d,%d)" % (
                             replace_dict['nAmpSplitOrders'],replace_dict['nSqAmpSplitOrders'],replace_dict['nSplitOrders'],
                                len(amp_orders),len(squared_orders),len(split_orders)))

            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                       sudakov_me,amp_orders,split_order_names=split_orders,
                                                    JAMP_format="JAMP2(%s,{0})")

        replace_dict['jamp2_lines'] = '\n'.join(jamp_lines).replace('AMP(', 'AMP2(')   
        replace_dict['nb_temp_jamp2'] = nb_tmp_jamp

        # the calls for the momentum reshuffling
        replace_dict['reshuffle_calls'] = 'pass_reshuffle = .true.\n'

        pdgs_in, pdgs_out = pdgs
        # make sure all the lists have lenght = 2. In case, pad with zero's
        for resh_list in [legs, pdgs_in, pdgs_out]:
            while len(resh_list) < 2:
                if resh_list == legs:
                    resh_list.append({'number':0})
                else:
                    resh_list.append(0)

        replace_dict['reshuffle_calls'] += "call reshuffle_momenta(p,p_resh,(/%s/),(/%s/),(/%s/),pass_reshuffle)\n" \
                                        % (','.join(['%d' % leg['number'] for leg in legs]), 
                                           ','.join(['%d' % p for p in pdgs_in]),
                                           ','.join(['%d' % p for p in pdgs_out]))
        replace_dict['reshuffle_calls'] += "p(:,:)=p_resh(:,:)\n"
    
        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/ewsudakov_splitorders_fks.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return


    def get_chargeprod(self, charge_list, ninitial, n, m):
        """return the product of charges (as a string) of particles m and n.
        Special sign conventions may be needed for initial/final state particles
        """
        return charge_list[n - 1] * charge_list[m - 1]

    
    #===============================================================================
    # write_b_sf_fks
    #===============================================================================
    #test written
    def write_b_sf_fks(self, writer, fksborn, ilink, fortran_model):
        """Create the b_sf_xxx.f file for the ilink-th soft linked born 
        """

        matrix_element = copy.copy(fksborn.born_me)

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
    
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        link = fksborn.color_links[ilink]
    
        replace_dict = {}
        
        replace_dict['ilink'] = ilink + 1
    
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
        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element, JAMP_format="JAMP1(%s)")
        else:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            replace_dict['sqsplitorders']='\n'.join(sqamp_so)           
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                                      matrix_element,amp_orders,split_order_names=split_orders,
                                                                      JAMP_format="JAMP1(%s,{0})")

        replace_dict['jamp1_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_tmp_jamp

        matrix_element.set('color_basis', link['link_basis'] )
        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element, JAMP_format="JAMP2(%s)")
        else:
            jamp_lines,nb_tmp_jamp  = self.get_JAMP_lines_split_order(\
                                            matrix_element,amp_orders,split_order_names=split_orders,
                                            JAMP_format="JAMP2(%s,{0})")
        replace_dict['jamp2_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = max(nb_tmp_jamp, replace_dict['nb_temp_jamp'])
        
        # Extract the number of FKS process
        replace_dict['nconfs'] = len(fksborn.get_fks_info_list())

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/b_sf_xxx_splitorders_fks.inc')).read()
        file = file % replace_dict
        
        # Write the file
        writer.writelines(file)
    
        return 0 , ncolor1
    

    #===============================================================================
    # write_born_nhel_file_list
    #===============================================================================
    def write_born_nhel_file(self, writer, me, nflows, fortran_model):
        """Write the born_nhel.inc file for MG4. Write the maximum as they are
        typically used for setting array limits."""
    
        ncomb = me.get_helicity_combinations()
        file = "integer    max_bhel, max_bcol \n"
        file += "parameter (max_bhel=%d)\nparameter(max_bcol=%d)" % \
               (ncomb, nflows)
    
        # Write the file
        writer.writelines(file)
    
        return True
    
    #===============================================================================
    # write_nfksconfigs_file
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
        split_orders = fksborn.born_me['processes'][0]['split_orders']
        replace_dict['nconfs'] = max(len(fks_info_list), 1)
        replace_dict['nsplitorders'] = len(split_orders)
        replace_dict['splitorders_name'] = ', '.join(split_orders)

        bool_dict = {True: '.true.', False: '.false.'}
        split_types_return = set(sum([info['fks_info']['splitting_type'] for info in fks_info_list], []))

        # this is for processes with 'real' or 'all' as NLO mode 
        if len(fks_info_list) > 0:
            replace_dict['fks_i_values'] = ', '.join(['%d' % info['fks_info']['i'] \
                                                 for info in fks_info_list]) 
            replace_dict['fks_j_values'] = ', '.join(['%d' % info['fks_info']['j'] \
                                                 for info in fks_info_list]) 
            replace_dict['extra_cnt_values'] = ', '.join(['%d' % (info['fks_info']['extra_cnt_index'] + 1) \
                                                 for info in fks_info_list]) 
            # extra array to be filled, with the type of the splitting of the born and of the extra cnt
            isplitorder_born = []
            isplitorder_cnt = []
            for info in fks_info_list:
                # fill 0 if no extra_cnt is needed
                if info['fks_info']['extra_cnt_index'] == -1:
                    isplitorder_born.append(0)
                    isplitorder_cnt.append(0)
                else:
                    # the 0th component of split_type correspond to the born, the 1st
                    # to the extra_cnt
                    isplitorder_born.append(split_orders.index(
                                            info['fks_info']['splitting_type'][0]) + 1)
                    isplitorder_cnt.append(split_orders.index(
                                           info['fks_info']['splitting_type'][1]) + 1)

            replace_dict['isplitorder_born_values'] = \
                            ', '.join(['%d' % n for n in isplitorder_born])
            replace_dict['isplitorder_cnt_values'] = \
                            ', '.join(['%d' % n for n in isplitorder_cnt])

            replace_dict['need_color_links'] = ', '.join(\
                    [bool_dict[info['fks_info']['need_color_links']] for \
                    info in fks_info_list ])
            replace_dict['need_charge_links'] = ', '.join(\
                    [bool_dict[info['fks_info']['need_charge_links']] for \
                    info in fks_info_list ])

            col_lines = []
            pdg_lines = []
            charge_lines = []
            tag_lines = []
            fks_j_from_i_lines = []
            split_type_lines = []
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
                tag_lines.append( \
                    'DATA (PARTICLE_TAG_D(%d, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                    % (i + 1, ', '.join(bool_dict[tag] for tag in fksborn.real_processes[info['n_me']-1].particle_tags) ))
                fks_j_from_i_lines.extend(self.get_fks_j_from_i_lines(fksborn.real_processes[info['n_me']-1],\
                                                i + 1))
                split_type_lines.append( \
                    'DATA (SPLIT_TYPE_D (%d, IPOS), IPOS=1, %d) / %s /' %
                      (i + 1, len(split_orders), 
                       ', '.join([bool_dict[ordd in info['fks_info']['splitting_type']] for ordd in split_orders])))
        else:
        # this is for 'LOonly', generate a fake FKS configuration with
        # - i_fks = nexternal, pdg type = -21 and color =8
        # - j_fks = the last colored particle
            bornproc = fksborn.born_me.get('processes')[0]
            pdgs = [l.get('id') for l in bornproc.get('legs')] + [-21]
            colors = [l.get('color') for l in bornproc.get('legs')] + [8]
            charges = [l.get('charge') for l in bornproc.get('legs')] + [0.]
            tags = [l.get('is_tagged') for l in bornproc.get('legs')] + [False]

            fks_i = len(colors)
            # fist look for a colored legs (set j to 1 otherwise)
            fks_j=0
            for cpos, col in enumerate(colors[:-1]):
                if col != 1:
                    fks_j = cpos+1
            # if no colored leg exist, look for a charged leg
            if fks_j == 0:
                for cpos, chg in enumerate(charges[:-1]):
                    if chg != 0.:
                        fks_j = cpos+1
            # no coloured or charged particle found. Pick the final particle in the (Born) process
            if fks_j==0: fks_j=len(colors)-1    

            # this is special for 2->1 processes: j must be picked initial
            # keep in mind that colors include the fake extra particle
            if len(colors) == 4:
                fks_j = 2

            replace_dict['fks_i_values'] = str(fks_i)
            replace_dict['fks_j_values'] = str(fks_j)
            replace_dict['extra_cnt_values'] = '0'
            replace_dict['isplitorder_born_values'] = '0'
            replace_dict['isplitorder_cnt_values'] = '0'
            # set both color/charge links to true
            replace_dict['need_color_links'] = '.true.'
            replace_dict['need_charge_links'] = '.true.'

            col_lines = ['DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join([str(col) for col in colors])]
            pdg_lines = ['DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join([str(pdg) for pdg in pdgs])]
            charge_lines = ['DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            % ', '.join('%19.15fd0' % charg for charg in charges)]
            tag_lines = ['DATA (PARTICLE_TAG_D(1, IPOS), IPOS=1, NEXTERNAL) / %s /' \
                            %  ', '.join(bool_dict[tag] for tag in tags)]
            fks_j_from_i_lines = ['DATA (FKS_J_FROM_I_D(1, %d, JPOS), JPOS = 0, 1)  / 1, %d /' \
                            % (fks_i, fks_j)]
            split_type_lines = [ \
                    'DATA (SPLIT_TYPE_D (%d, IPOS), IPOS=1, %d) / %s /' %
                      (1, len(split_orders), 
                       ', '.join([bool_dict[False]] * len(split_orders)))]
            

        replace_dict['col_lines'] = '\n'.join(col_lines)
        replace_dict['pdg_lines'] = '\n'.join(pdg_lines)
        replace_dict['charge_lines'] = '\n'.join(charge_lines)
        replace_dict['tag_lines'] = '\n'.join(tag_lines)
        replace_dict['fks_j_from_i_lines'] = '\n'.join(fks_j_from_i_lines)
        replace_dict['split_type_lines'] = '\n'.join(split_type_lines)

        content = open(os.path.join(_file_path, \
                    'iolibs/template_files/fks_info.inc')).read() % replace_dict

        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        writer.writelines(content)
    
        return split_types_return

 
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
            raise writers.FortranWriter.FortranWriterError("""Need ninitial = 1 or 2 to write auto_dsig file""")
    
        replace_dict = {}

        replace_dict['N_me'] = n
    
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
    
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines
    
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                self.get_pdf_lines_mir(matrix_element, ninitial, False, False)
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['ee_comp_vars'] = eepdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/parton_lum_n_fks.inc')).read()
        file = file % replace_dict
    
        # Write the file
        writer.writelines(file)



    #===============================================================================
    # write_coloramps_file
    #===============================================================================
    def write_coloramps_file(self, writer, mapconfigs, me, fortran_model):
        """Write the coloramps.inc file for MadEvent"""

        lines = []
        lines.append( "logical icolamp(%d,%d,1)" % \
                        (max([len(list(me.get('color_basis').keys())), 1]),
                         len(mapconfigs)))

        lines += self.get_icolamp_lines(mapconfigs, me, 1)
    
        # Write the file
        writer.writelines(lines)
    
        return True


    #===============================================================================
    # write_leshouche_file_list
    #===============================================================================
    def write_born_leshouche_file(self, writer, me, fortran_model):
        """Write the leshouche.inc file for MG4"""
    
        # Extract number of external particles
        (nexternal, ninitial) = me.get_nexternal_ninitial()
    
        lines = []

        for iproc, proc in enumerate(me.get('processes')):
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
                if not me.get('color_basis'):
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
                        me.get('color_basis').color_flow_decomposition(repr_dict, ninitial)
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
    # write_born_conf_file
    #===============================================================================
    def write_born_conf_file(self, writer, me, fortran_model):
        """Write the configs.inc file for the list of born matrix-elements"""
    
        # Extract number of external particles
        (nexternal, ninitial) = me.get_nexternal_ninitial()
        model = me.get('processes')[0].get('model')
        lines = ['', 'C Here are the congifurations']
        lines_P = ['', 'C Here are the propagators']
        lines_BW = ['', 'C Here are the BWs']
    
        iconfig = 0
    
        iconfig_list = []
        mapconfigs_list = [] 
        s_and_t_channels_list = []
        nschannels = []
    
        particle_dict = me.get('processes')[0].get('model').\
                        get('particle_dict')

        booldict = {True: '.false.', False: '.false'} 

        max_leg_number = 0

        ######first get the configurations
        s_and_t_channels = []
        mapconfigs = []
        lines.extend(['C     %s' % proc.nice_string() for proc in me.get('processes')])
        base_diagrams = me.get('base_amplitude').get('diagrams')
        minvert = min([max([len(vert.get('legs')) for vert in \
                            diag.get('vertices')]) for diag in base_diagrams])

        for idiag, diag in enumerate(base_diagrams):
            if any([len(vert.get('legs')) > minvert for vert in
                    diag.get('vertices')]):
                # Only 3-vertices allowed in configs.inc
                continue
            iconfig = iconfig + 1
            helas_diag = me.get('diagrams')[idiag]
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
                lines.append("data (iforest(ifr,%3d,%4d),ifr=1,%d)/%s/" % \
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

                max_leg_number = min(max_leg_number,last_leg.get('number'))

        ##### Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%4d/" % (iconfig))

        ######finally the BWs
        for iconf, config in enumerate(s_and_t_channels):
            schannels = config[0]
            nschannels.append(len(schannels))
            for vertex in schannels:
                # For the resulting leg, pick out whether it comes from
                # decay or not, as given by the from_group flag
                leg = vertex.get('legs')[-1]
                lines_BW.append("data gForceBW(%d,%d)/%s/" % \
                             (leg.get('number'), iconf + 1,
                              booldict[leg.get('from_group')]))

        #lines for the declarations
        firstlines = []
        firstlines.append('integer ifr')
        firstlines.append('integer lmaxconfigsb_used\nparameter (lmaxconfigsb_used=%d)' % iconfig)
        firstlines.append('integer max_branchb_used\nparameter (max_branchb_used=%d)' % -max_leg_number)
        firstlines.append('integer mapconfig(0 : lmaxconfigsb_used)')
        firstlines.append('integer iforest(2, -max_branchb_used:-1, lmaxconfigsb_used)')
        firstlines.append('integer sprop(-max_branchb_used:-1, lmaxconfigsb_used)')
        firstlines.append('integer tprid(-max_branchb_used:-1, lmaxconfigsb_used)')
        firstlines.append('logical gforceBW(-max_branchb_used : -1, lmaxconfigsb_used)')
    
        # Write the file
        writer.writelines(firstlines + lines + lines_BW)
    
        return iconfig, mapconfigs, s_and_t_channels


    #===============================================================================
    # write_born_props_file
    #===============================================================================
    def write_born_props_file(self, writer, me, s_and_t_channels, fortran_model):
        """Write the configs.inc file for the list of born matrix-elements"""
    
        # Extract number of external particles
        lines_P = ['', 'C Here are the propagators']
    
        particle_dict = me.get('processes')[0].get('model').\
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
    
                lines_P.append("pmass(%3d,%4d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines_P.append("pwidth(%3d,%4d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines_P.append("pow(%3d,%4d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))

        # Write the file
        writer.writelines(lines_P)

    
    
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


    def get_leshouche_lines_dummy(self, matrix_element, ime):
        #test written
        """As get_leshouche_lines, but for 'fake' real emission processes (LOonly
        In this case, write born color structure times ij -> i,j splitting)
        """

        bornproc = matrix_element.get('processes')[0]
        colors = [l.get('color') for l in bornproc.get('legs')] 

        fks_i = len(colors)
        # use the last colored particle if it exists, or 
        # just the last
        fks_j=1
        for cpos, col in enumerate(colors):
            if col != 1:
                fks_j = cpos+1
    
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        nexternal+=1 # remember, in this case matrix_element is born
    
        lines = []
        for iproc, proc in enumerate(matrix_element.get('processes')):
            # add the fake extra leg
            legs = proc.get_legs_with_decays() + \
                   [fks_common.FKSLeg({'id': -21,
                                       'number': nexternal,
                                       'state': True,
                                       'fks': 'i',
                                       'color': 8,
                                       'charge': 0.,
                                       'massless': True,
                                       'spin': 3,
                                       'is_part': True,
                                       'self_antipart': True})]
                                        
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
                    # in this case the last particle (-21) has two color indices
                    # and it has to be emitted by j_fks
                    # First build a color representation dictionnary
                    repr_dict = {}
                    for l in legs[:-1]:
                        repr_dict[l.get('number')] = \
                            proc.get('model').get_particle(l.get('id')).get_color()\
                            * (-1)**(1+l.get('state'))
                    # Get the list of color flows
                    color_flow_list = \
                        matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                                   ninitial)
                    # And output them properly
                    for cf_i, color_flow_dict in enumerate(color_flow_list):
                        # we have to add the extra leg (-21), linked to the j_fks leg
                        # first, find the maximum color label
                        maxicol = max(sum(list(color_flow_dict.values()), []))
                        #then, replace the color labels
                        if color_flow_dict[fks_j][0] == 0:
                            anti = True
                            icol_j = color_flow_dict[fks_j][1]
                        else:
                            anti = False
                            icol_j = color_flow_dict[fks_j][0]

                        if anti:
                            color_flow_dict[nexternal] = (maxicol + 1, color_flow_dict[fks_j][1])
                            color_flow_dict[fks_j][1] = maxicol + 1
                        else:
                            color_flow_dict[nexternal] = (color_flow_dict[fks_j][0], maxicol + 1) 
                            color_flow_dict[fks_j][0] = maxicol + 1

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
    def get_den_factor_lines(self, fks_born, born_me=None):
        """returns the lines with the information on the denominator keeping care
        of the identical particle factors in the various real emissions
        If born_me is procided, it is used instead of fksborn.born_me"""
        
        compensate = True
        if not born_me:
            born_me = fks_born.born_me
            compensate = False
    
        lines = []
        info_list = fks_born.get_fks_info_list()
        if info_list:
            # if the reals have been generated, fill with the corresponding average factor
            lines.append('INTEGER IDEN_VALUES(%d)' % len(info_list))
            if not compensate:
                lines.append('DATA IDEN_VALUES /' + \
                             ', '.join(['%d' % ( 
                             born_me.get_denominator_factor()) \
                             for info in info_list]) + '/')
            else:
                lines.append('DATA IDEN_VALUES /' + \
                             ', '.join(['%d' % ( 
                             born_me.get_denominator_factor() / \
                                     born_me['identical_particle_factor'] * \
                                     fks_born.born_me['identical_particle_factor']) \
                             for info in info_list]) + '/')
        else:
            # otherwise use the born
            lines.append('INTEGER IDEN_VALUES(1)')
            lines.append('DATA IDEN_VALUES / %d /' \
                    % fks_born.born_me.get_denominator_factor())

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
            # if the reals have been generated, fill with the corresponding value of ij if
            # ij is massless, or with 0 if ij is massive (no collinear singularity)
            ij_list = [info['fks_info']['ij']if \
                    fks_born.born_me['processes'][0]['legs'][info['fks_info']['ij']-1]['massless'] \
                    else 0 for info in info_list]
            lines.append('INTEGER IJ_VALUES(%d)' % len(info_list))
            lines.append('DATA IJ_VALUES /' + ', '.join(['%d' % ij for ij in ij_list]) + '/')
        else:
            #otherwise just put zero
            lines.append('INTEGER IJ_VALUES(1)')
            lines.append('DATA IJ_VALUES / 0 /')

        return lines


    def get_pdf_lines_mir(self, matrix_element, ninitial, subproc_group = False,\
                          mirror = False): #test written
        """Generate the PDF lines for the auto_dsig.f file"""

        processes = matrix_element.get('processes')
        model = processes[0].get('model')

        pdf_definition_lines = ""
        ee_pdf_definition_lines = ""
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
            pdgtopdf = {21: 0, 22: 7, -11: -8, 11: 8, -13: -9, 13: 9, -15: -10, 15: 10}
            # Fill in missing entries of pdgtopdf
            for pdg in sum(initial_states,[]):
                if not pdg in pdgtopdf and not pdg in list(pdgtopdf.values()):
                    pdgtopdf[pdg] = pdg
                elif pdg not in pdgtopdf and pdg in list(pdgtopdf.values()):
                    # If any particle has pdg code 7, we need to use something else
                    pdgtopdf[pdg] = 6000000 + pdg

            # Get PDF variable declarations for all initial states
            ee_pdf_definition_lines += "DOUBLE PRECISION dummy_components(n_ee)\n" 
            for i in [0,1]:
                pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in \
                                                 initial_states[i]]) + \
                                                 "\n"
                ee_pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d_components(n_ee)" % (pdf_codes[pdg],i+1) \
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
                           "IF (ABS(LPP(IB(%d))).GE.1) THEN\n" \
                                 % (ibeam)
                else:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(%d)) .GE. 1) THEN\n" \
                                 % (ibeam)

#                for iproc, initial_state in enumerate(init_states):
                for initial_state in init_states:
                    if initial_state in list(pdf_codes.keys()):
                        if subproc_group:
                            if abs(pdgtopdf[initial_state]) <= 10:  
                                pdf_lines = pdf_lines + \
                                     ("%s%d=PDG2PDF(LPP(IB(%d)),%d, IB(%d)," + \
                                         "XBK(IB(%d)),DSQRT(Q2FACT(%d)))\n" + \
                                     "IF ((ABS(LPP(%d)).EQ.4.or.ABS(LPP(%d)).EQ.3).and.pdlabel.ne.'none') %s%d_components(1:n_ee) = ee_components(1:n_ee)\n") % \
                                         (
                                           pdf_codes[initial_state], i + 1, ibeam, pdgtopdf[initial_state], ibeam,
                                          ibeam, ibeam,
                                          ibeam, ibeam, pdf_codes[initial_state], ibeam)
                            else:
                                # setting other partons flavours outside quark, gluon, photon to be 0d0
                                pdf_lines = pdf_lines + \
                                    ("c settings other partons flavours outside quark, gluon, photon to 0d0\n" + \
                                     "%s%d=0d0\n") % \
                                         (pdf_codes[initial_state],i + 1)                                
                        else:
                            if abs(pdgtopdf[initial_state]) <= 10:  
                                pdf_lines = pdf_lines + \
                                     ("%s%d=PDG2PDF(LPP(%d),%d,%d," + \
                                         "XBK(%d),DSQRT(Q2FACT(%d)))\n" + \
                                     "IF ((ABS(LPP(%d)).EQ.4.or.ABS(LPP(%d)).EQ.3).and.pdlabel.ne.'none') %s%d_components(1:n_ee) = ee_components(1:n_ee)\n") % \
                                         (
                                           pdf_codes[initial_state], i + 1, ibeam, pdgtopdf[initial_state], ibeam,
                                          ibeam, ibeam,
                                          ibeam, ibeam, pdf_codes[initial_state], ibeam)
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
                pdf_lines += "if (ABS(LPP(1)).EQ.ABS(LPP(2)).and. (ABS(LPP(1)).EQ.3.or.ABS(LPP(1)).EQ.4).and.pdlabel.ne.'none')" + \
                             "PD(IPROC)=ee_comp_prod(%s_components,%s_components)\n" % \
                             tuple(comp_list)

        # Remove last line break from pdf_lines
        return pdf_definition_lines[:-1], pdf_data_lines[:-1], pdf_lines[:-1], ee_pdf_definition_lines


    #test written
    def get_color_data_lines_from_color_matrix(self, color_matrix, n=6):
        """Return the color matrix definition lines for the given color_matrix. Split
        rows in chunks of size n."""
    
        if not color_matrix:
            return ["DATA Denom(1)/1/", "DATA (CF(i,1),i=1,1) /1/"]
        else:
            ret_list = []
            my_cs = color.ColorString()
            denominator = min(color_matrix.get_line_denominators())
            ret_list.append("DATA Denom/%i/" % denominator)
            
            for index in range(len(color_matrix._col_basis1)):
                num_list = color_matrix.get_line_numerators(index, denominator)  
                for k in range(0, len(num_list), n):
                    ret_list.append("DATA (CF(i,%3r),i=%3r,%3r) /%s/" % \
                                    (index + 1, k + 1, min(k + n, len(num_list)),
                                     ','.join(["%u" % int(i) for i in num_list[k:k + n]])))
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
    
    #===========================================================================
    # write_colors_file
    #===========================================================================
    def write_colors_file(self, writer, matrix_element):
        """Write the get_color.f file for MadEvent, which returns color
        for all particles used in the matrix element."""

        try:
            matrix_elements=matrix_element.real_processes[0].matrix_element
        except IndexError:
            matrix_elements=[matrix_element.born_me]

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
                              for d in matrix_element.born_me.get('diagrams')],[])))

        # loop on the real emissions
        leg_ids = set(sum([sum([sum([[l.get('id') for l in \
                                p.get_legs_with_decays()] for p in \
                                me.get('processes')], []) for me in \
                                [real_proc.matrix_element]], []) for real_proc in \
                                matrix_element.real_processes],[]))
        # and also on the born
        leg_ids = leg_ids.union(set(sum([[l.get('id') for l in \
                                p.get_legs_with_decays()] for p in \
                                matrix_element.born_me.get('processes')], [])))
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
        stop 1
        return
        endif
        end
        """
        
        lines+= """
        function get_spin(ipdg)
        implicit none
        integer get_spin, ipdg

        if(ipdg.eq.%d)then
        get_spin=%d
        return
        """ % (particle_ids[0], model.get_particle(particle_ids[0]).get('spin'))

        for part_id in particle_ids[1:]:
            lines += """else if(ipdg.eq.%d)then
            get_spin=%d
            return
            """ % (part_id, model.get_particle(part_id).get('spin'))
        # Dummy particle for multiparticle vertices with pdg given by
        # first code not in the model
        lines += """else if(ipdg.eq.%d)then
c           This is dummy particle used in multiparticle vertices
            get_spin=-2
            return
            """ % model.get_first_non_pdg()
        lines += """else
        write(*,*)'Error: No spin given for pdg ',ipdg
        stop 1
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

    jamp_optim = True 

    def finalize(self, *args, **opts):
        ProcessExporterFortranFKS.finalize(self, *args, **opts)
        #export_v4.ProcessExporterFortranSA.finalize(self, *args, **opts)

#===============================================================================
# copy the Template in a new directory.
#===============================================================================
    def copy_fkstemplate(self, model):
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
                raise MadGraph5Error("No valid MG_ME path given for MG4 run directory creation.")
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(dir_path))
            misc.copytree(os.path.join(mgme_dir, 'Template', 'NLO'), dir_path, True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template', 'Common'),
                               dir_path)
            # Copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards', card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to copy " + card + ".dat to default")

        elif not os.path.isfile(os.path.join(dir_path, 'TemplateVersion.txt')):
            if not mgme_dir:
                raise MadGraph5Error("No valid MG_ME path given for MG4 run directory creation.")
        try:
            shutil.copy(os.path.join(mgme_dir, 'MGMEVersion.txt'), dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(os.path.join(dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])
        
        #Ensure that the Template is clean
        if clean:
            logger.info('remove old information in %s' % os.path.basename(dir_path))
            if 'MADGRAPH_BASE' in os.environ:
                subprocess.call([os.path.join('bin', 'internal', 'clean_template'), 
                    '--web'], cwd=dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'internal', 'clean_template')], \
                                                                       cwd=dir_path)
                except Exception as why:
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
                if tir in ['pjfry','ninja','golem', 'samurai','collier']:
                    # We should link dynamically when possible, so we use the original
                    # location of these libraries.
                    link_tir_libs.append('-L%s/ -l%s'%(libpath,tir))
                    tir_libs.append('%s/lib%s.$(libext)'%(libpath,tir))
                    # For Ninja, we must also link against OneLoop.
                    if tir in ['ninja']:
                        if not any(os.path.isfile(pjoin(libpath,'libavh_olo.%s'%ext)) 
                                              for ext in ['a','dylib','so']):
                            raise MadGraph5Error(
"The OneLOop library 'libavh_olo.(a|dylib|so)' could no be found in path '%s'. Please place a symlink to it there."%libpath)
                        link_tir_libs.append('-L%s/ -l%s'%(libpath,'avh_olo'))
                        tir_libs.append('%s/lib%s.$(libext)'%(libpath,'avh_olo'))
                    # We must add the corresponding includes for these TIR
                    if tir in ['golem','samurai','ninja','collier']:
                        trg_path = pjoin(os.path.dirname(libpath),'include')
                        if os.path.isdir(trg_path):
                            to_include = misc.find_includes_path(trg_path,
                                                        self.include_names[tir])
                        else:
                            to_include = None
                        # Special possible location for collier
                        if to_include is None and tir=='collier':
                            to_include = misc.find_includes_path(
                               pjoin(libpath,'modules'),self.include_names[tir])
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

        # Copy the Pythia8 Sudakov tables (needed for MC@NLO-DELTA matching)
        shutil.copy(os.path.join(self.mgme_dir,'vendor','SudGen','sudakov.f'), \
                    os.path.join(self.dir_path,'SubProcesses','sudakov.f'),follow_symlinks=True)
        
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
                                   'print_banner_commands':self.MadLoop_banner},
               context={'collier_available':self.tir_available_dict['collier']})
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


        if model["running_elements"]:
            shutil.copytree(pjoin(MG5DIR, 'Template',"Running"), 
                            pjoin(self.dir_path,'Source','RUNNING'))
        
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

        # We also need to write the overall maximum quantities for this group
        # of processes in 'global_specs.inc'. In aMCatNLO, there is always
        # only one process, so this is trivial
        self.write_global_specs(matrix_element, output_path=pjoin(dirpath,'global_specs.inc'))
        
        open('unique_id.inc','w').write(
"""      integer UNIQUE_ID
      parameter(UNIQUE_ID=1)""")

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
    


            
class ProcessExporterEWSudakovSA(ProcessOptimizedExporterFortranFKS):
    """exports the EW sudakov matrix element in a standalone format
    """
    dirstopdg = []

    def finalize(self, *args, **opts):
        """do the usual finalize, then call the function that writes
        the python module with all the calls
        """
        super(ProcessExporterEWSudakovSA, self).finalize(*args, **opts)
        self.write_python_wrapper(os.path.join(self.dir_path, 'bin', 'internal', 'ewsud_pydispatcher.py'))

    def write_python_wrapper(self, fname):
        """write a wrapper to be able to call the Sudakov for a specific subfolder given its PDG"""
        
        template = open(os.path.join(_file_path, \
                          'iolibs/template_files/ewsudakov_pydispatcher.inc')).read()

        replace_dict = {}
        replace_dict['path'] = os.path.join(self.dir_path, 'SubProcesses')
        replace_dict['pdir_list'] = ", ".join(["'%s'" % dd[0] for dd in self.dirstopdg])  
        replace_dict['pdg2sud'] = ",\n".join([str(self.get_pdg_tuple(dd[1], dd[2], sortfinal=True)) + \
                ": importlib.import_module('%s.ewsudpy')" % dd[0] for dd in self.dirstopdg])   

        replace_dict['pdgsorted'] = ",\n".join(["%s: %s" % (
                        str(self.get_pdg_tuple(dd[1], dd[2], sortfinal=True)),
                        str(self.get_pdg_tuple(dd[1], dd[2], sortfinal=False))) \
                                                for dd in self.dirstopdg])

        outfile = open(fname ,'w')
        outfile.write(template % replace_dict)
        outfile.close()

    def get_pdg_tuple(self, pdgs, nincoming, sortfinal):
        """write a tuple of 2 tuple, with the incoming particles unsorted
        and the outgoing ones sorted if sortfinal = True
        """
        incoming = pdgs[:nincoming]
        outgoing = pdgs[nincoming:]
        if sortfinal:
            return (tuple(incoming), tuple(sorted(outgoing)))
        else:
            return (tuple(incoming), tuple(outgoing))


    #===============================================================================
    # generate_directories_fks
    #===============================================================================
    def generate_directories_fks(self, matrix_element, fortran_model, me_number,
                                    me_ntot, path=os.getcwd(),OLP='MadLoop'):
        """Generate the Pxxxxx_i directories for a subprocess in MadFKS,
        only generating the relevant files for the EW Sudakov"""
        proc = matrix_element.born_me['processes'][0]

        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')
        
        cwd = os.getcwd()
        try:
            os.chdir(path)
        except OSError as error:
            error_msg = "The directory %s should exist in order to be able " % path + \
                        "to \"export\" in it. If you see this error message by " + \
                        "typing the command \"export\" please consider to use " + \
                        "instead the command \"output\". "
            raise MadGraph5Error(error_msg) 
        
        calls = 0
        
        self.fksdirs = []
        #first make and cd the direcrory corresponding to the born process:
        borndir = "P%s" % \
        (matrix_element.born_me.get('processes')[0].shell_string())
        os.mkdir(borndir)
        os.chdir(borndir)
        logger.info('Writing files in %s (%d / %d)' % (borndir, me_number + 1, me_ntot))

## write the files corresponding to the born process in the P* directory
        self.generate_born_fks_files(matrix_element,
                fortran_model, me_number, path)


#write the infortions for the different real emission processes
        sqsorders_list = \
            self.write_real_matrix_elements(matrix_element, fortran_model)

        filename = 'extra_cnt_wrapper.f'
        self.write_extra_cnt_wrapper(writers.FortranWriter(filename),
                                     matrix_element.extra_cnt_me_list, 
                                     fortran_model)

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                              me_number)

        filename = 'fks_info.inc'
        # write_fks_info_list returns a set of the splitting types
        self.proc_characteristic['splitting_types'] = list(\
                set(self.proc_characteristic['splitting_types']).union(\
                    self.write_fks_info_file(writers.FortranWriter(filename), 
                                 matrix_element, 
                                 fortran_model)))

        filename = 'leshouche_info.dat'
        nfksconfs,maxproc,maxflow,nexternal=\
                self.write_leshouche_info_file(filename,matrix_element)

        # if no corrections are generated ([LOonly] mode), get 
        # these variables from the born
        if nfksconfs == maxproc == maxflow == 0:
            nfksconfs = 1
            (dummylines, maxproc, maxflow) = self.get_leshouche_lines(
                    matrix_element.born_me, 1)

        filename = 'genps.inc'
        ngraphs = matrix_element.born_me.get_number_of_amplitudes()
        ncolor = max(1,len(matrix_element.born_me.get('color_basis')))
        self.write_genps(writers.FortranWriter(filename),maxproc,ngraphs,\
                         ncolor,maxflow,fortran_model)

#        filename = 'maxconfigs.inc'
#        self.write_maxconfigs_file(writers.FortranWriter(filename),
#                max(nconfigs,matrix_element.born_me.get_number_of_amplitudes()))

        filename = 'nexternal.inc'
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'orders.inc'
        amp_split_orders, amp_split_size, amp_split_size_born = \
			   self.write_orders_file(
                            writers.FortranWriter(filename),
                            matrix_element)

        filename = 'amp_split_orders.inc'
        self.write_amp_split_orders_file(
                            writers.FortranWriter(filename),
                            amp_split_orders)
        self.proc_characteristic['ninitial'] = ninitial
        self.proc_characteristic['nexternal'] = max(self.proc_characteristic['nexternal'], nexternal)
        
        filename = 'maxparticles.inc'
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     nexternal)
        
        filename = 'pmass.inc'
        try:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.real_processes[0].matrix_element)
        except IndexError:
            self.write_pmass_file(writers.FortranWriter(filename),
                             matrix_element.born_me)

        #draw the diagrams
        self.draw_feynman_diagrams(matrix_element)

        linkfiles = ['sa_ewsudakov.f',
                     'sub_f2py_ewsudakov.f',
                     'sa_ewsudakov_dummyfcts.f',
                     'ewsudakov_functions.f',
                     'momentum_reshuffling.f',
                     'splitorders_stuff.f',
                     'add_write_info.f',
                     'coupl.inc',
                     'weight_lines.f',
                     'run.inc',
                     'run_card.inc',
                     'q_es.inc',
                     'setscales.f',
                     'randinit',
                     'timing_variables.inc',
                     'orderstag_base.inc',
                     'orderstags_glob.dat']

        for file in linkfiles:
            ln('../' + file , '.')
        os.system("ln -s ../../Cards/param_card.dat .")

        #copy the makefile 
        os.system("ln -s ../makefile_fks_dir ./makefile")

        # touch a dummy analyse_opts

        os.system('touch %s/analyse_opts' % os.path.join(self.dir_path,'SubProcesses'))

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
        
        # update the dirs to pdg information
        self.dirstopdg.extend([(borndir, [l.get('id') for l in pp['legs']], [l.get('state') for l in pp['legs']].count(False)) for pp in matrix_element.born_me['processes']])

        return calls, amp_split_orders
