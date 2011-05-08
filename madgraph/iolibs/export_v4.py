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
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks


import aloha.create_aloha as create_aloha
import models.write_param_card as param_writer
from madgraph import MadGraph5Error, MG5DIR
from madgraph.iolibs.files import cp, ln, mv
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_v4')

#===============================================================================
# ProcessExporterFortran
#===============================================================================
class ProcessExporterFortran(object):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    def __init__(self, mgme_dir = "", dir_path = "", clean = False):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = mgme_dir
        self.dir_path = dir_path
        self.clean = clean
        self.model = None

    #===========================================================================
    # copy the Template in a new directory.
    #===========================================================================
    def copy_v4template(self):
        """create the directory run_name as a copy of the MadEvent
        Template, and clean the directory
        """

        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(self.dir_path):
            if not self.mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(self.dir_path))
            shutil.copytree(os.path.join(self.mgme_dir, 'Template'), self.dir_path, True)
        elif not os.path.isfile(os.path.join(self.dir_path, 'TemplateVersion.txt')):
            if not self.mgme_dir:
                raise MadGraph5Error, \
                      "No valid MG_ME path given for MG4 run directory creation."
        try:
            shutil.copy(os.path.join(self.mgme_dir, 'MGMEVersion.txt'), self.dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(os.path.join(self.dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])

        #Ensure that the Template is clean
        if self.clean:
            logger.info('remove old information in %s' % \
                                                  os.path.basename(self.dir_path))
            if os.environ.has_key('MADGRAPH_BASE'):
                subprocess.call([os.path.join('bin', 'clean_template'),
                                 '--web'], cwd=self.dir_path)
            else:
                try:
                    subprocess.call([os.path.join('bin', 'clean_template')], \
                                                                       cwd=self.dir_path)
                except Exception, why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(self.dir_path),why))

            #Write version info
            MG_version = misc.get_pkg_info()
            open(os.path.join(self.dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])


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
        process_text += process_template.substitute({'process': process_str, \
                                                            'coupling': coupling})

        text = proc_card_template.substitute({'process': process_text,
                                            'model': modelname,
                                            'multiparticle':''})
        ff = open(file_pos, 'w')
        ff.write(text)
        ff.close()

    #===========================================================================
    # Create jpeg diagrams, html pages,proc_card_mg5.dat and madevent.tar.gz
    #===========================================================================
    def finalize_v4_directory(self, matrix_elements, history = "", makejpg = False, online = False):
        """Function to finalize v4 directory, for inheritance.
        """
        pass
    
    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self):
        """Function to write a matrix.f file, for inheritance.
        """
        pass
    
    #===========================================================================
    # export the model
    #===========================================================================
    def export_model_files(self, model_path):
        """Configure the files/link of the process according to the model"""

        # Import the model
        for file in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, file)):
                shutil.copy2(os.path.join(model_path, file), \
                                     os.path.join(self.dir_path, 'Source', 'MODEL'))    
        self.make_model_symbolic_link()

    def make_model_symbolic_link(self):
        """Make the copy/symbolic links"""
        model_path = self.dir_path + '/Source/MODEL/'
        if os.path.exists(os.path.join(model_path, 'ident_card.dat')):
            mv(model_path + '/ident_card.dat', self.dir_path + '/Cards')
        if os.path.exists(os.path.join(model_path, 'particles.dat')):
            ln(model_path + '/particles.dat', self.dir_path + '/SubProcesses')
            ln(model_path + '/interactions.dat', self.dir_path + '/SubProcesses')
        cp(model_path + '/param_card.dat', self.dir_path + '/Cards')
        mv(model_path + '/param_card.dat', self.dir_path + '/Cards/param_card_default.dat')
        ln(model_path + '/coupl.inc', self.dir_path + '/Source')
        ln(model_path + '/coupl.inc', self.dir_path + '/SubProcesses')
        ln(self.dir_path + '/Source/run.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/genps.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxconfigs.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxparticles.inc', self.dir_path + '/SubProcesses', log=False)

    #===========================================================================
    # export the helas routine
    #===========================================================================
    def export_helas(self, helas_path):
        """Configure the files/link of the process according to the model"""

        # Import helas routine
        for filename in os.listdir(helas_path):
            filepos = os.path.join(helas_path, filename)
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
    #        helas_dir = os.path.join(mgme_dir, 'HELAS')
    #        for filename in os.listdir(helas_dir): 
    #            if os.path.isfile(os.path.join(helas_dir, filename)):
    #                shutil.copy2(os.path.join(helas_dir, filename),
    #                            os.path.join(dir_path, 'Source', 'DHELAS'))
    #        shutil.move(os.path.join(dir_path, 'Source', 'DHELAS', 'Makefile.template'),
    #                    os.path.join(dir_path, 'Source', 'DHELAS', 'Makefile'))
    #  

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    def generate_subprocess_directory_v4(self, matrix_element,
                                         fortran_model,
                                         me_number):
        """Routine to generate a subprocess directory (for inheritance)"""

        pass

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
        writer.writelines(file)

        return True

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
    # Routines to output UFO models in MG4 format
    #===========================================================================

    def convert_model_to_mg4(self, model, wanted_lorentz = [],
                             wanted_couplings = []):
        """ Create a full valid MG4 model from a MG5 model (coming from UFO)"""

        # create the MODEL
        write_dir=os.path.join(self.dir_path, 'Source', 'MODEL')
        model_builder = UFO_model_to_mg4(model, write_dir)
        model_builder.build(wanted_couplings)

        # Create and write ALOHA Routine
        aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
        if wanted_lorentz:
            aloha_model.compute_subset(wanted_lorentz)
        else:
            aloha_model.compute_all(save=False)
        write_dir=os.path.join(self.dir_path, 'Source', 'DHELAS')
        aloha_model.write(write_dir, 'Fortran')

        #copy Helas Template
        cp(MG5DIR + '/aloha/template_files/Makefile_F', write_dir+'/makefile')
        for filename in os.listdir(os.path.join(MG5DIR,'aloha','template_files')):
            if not filename.lower().endswith('.f'):
                continue
            cp((MG5DIR + '/aloha/template_files/' + filename), write_dir)
        create_aloha.write_aloha_file_inc(write_dir, '.f', '.o')

        # Make final link in the Process
        self.make_model_symbolic_link()

    #===========================================================================
    # Helper functions
    #===========================================================================
    def get_mg5_info_lines(self):
        """Return info lines for MG5, suitable to place at beginning of
        Fortran files"""

        info = misc.get_pkg_info()
        info_lines = ""
        if info and info.has_key('version') and  info.has_key('date'):
            info_lines = "#  Generated by MadGraph 5 v. %s, %s\n" % \
                         (info['version'], info['date'])
            info_lines = info_lines + \
                         "#  By the MadGraph Development Team\n" + \
                         "#  Please visit us at https://launchpad.net/madgraph5"
        else:
            info_lines = "#  Generated by MadGraph 5\n" + \
                         "#  By the MadGraph Development Team\n" + \
                         "#  Please visit us at https://launchpad.net/madgraph5"        

        return info_lines

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "C " + process.nice_string().replace('\n', '\nC * ') \
                         for process in matrix_element.get('processes')])


    def get_helicity_lines(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line_list = []
        i = 0
        for helicities in matrix_element.get_helicity_matrix():
            i = i + 1
            int_list = [i, len(helicities)]
            int_list.extend(helicities)
            helicity_line_list.append(\
                ("DATA (NHEL(I,%4r),I=1,%d) /" + \
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

        # Crate dictionary between diagram number and JAMP number
        diag_jamp = {}
        for ijamp, col_basis_elem in \
                enumerate(sorted(matrix_element.get('color_basis').keys())):
            for diag_tuple in matrix_element.get('color_basis')[col_basis_elem]:
                diag_num = diag_tuple[0] + 1
                # Add this JAMP number to this diag_num
                diag_jamp[diag_num] = diag_jamp.setdefault(diag_num, []) + \
                                    [ijamp+1]

        colamps = ijamp + 1

        for iconfig, num_diag in enumerate(mapconfigs):        
            if num_diag == 0:
                continue

            # List of True or False 
            bool_list = [(i + 1 in diag_jamp[num_diag]) for i in \
                              range(colamps)]
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
        minvert = min([max(diag.get_vertex_leg_numbers()) for diag in \
                       matrix_element.get('diagrams')])

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

                line += "+".join(["AMP(%(num)d)*dconjg(AMP(%(num)d))" % \
                                  {"num": a.get('number')} for a in \
                                  sum([diagrams[idiag].get('amplitudes') for \
                                       idiag in config_to_diag_dict[config]], [])])
                ret_lines.append(line)
        else:
            for idiag, diag in enumerate(matrix_element.get('diagrams')):
                # Ignore any diagrams with 4-particle vertices.
                if max(diag.get_vertex_leg_numbers()) > minvert:
                    continue
                # Now write out the expression for AMP2, meaning the sum of
                # squared amplitudes belonging to the same diagram
                line = "AMP2(%(num)d)=AMP2(%(num)d)+" % {"num": (idiag + 1)}
                line += "+".join(["AMP(%(num)d)*dconjg(AMP(%(num)d))" % \
                                  {"num": a.get('number')} for a in \
                                  diag.get('amplitudes')])
                ret_lines.append(line)

        return ret_lines

    def get_JAMP_lines(self, matrix_element):
        """Return the JAMP = sum(fermionfactor * AMP(i)) lines"""

        res_list = []

        for i, coeff_list in \
                enumerate(matrix_element.get_color_amplitudes()):

            res = "JAMP(%i)=" % (i + 1)

            # Optimization: if all contributions to that color basis element have
            # the same coefficient (up to a sign), put it in front
            list_fracs = [abs(coefficient[0][1]) for coefficient in coeff_list]
            common_factor = False
            diff_fracs = list(set(list_fracs))
            if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                common_factor = True
                global_factor = diff_fracs[0]
                res = res + '%s(' % self.coeff(1, global_factor, False, 0)

            for (coefficient, amp_number) in coeff_list:
                if common_factor:
                    res = res + "%sAMP(%d)" % (self.coeff(coefficient[0],
                                               coefficient[1] / abs(coefficient[1]),
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number)
                else:
                    res = res + "%sAMP(%d)" % (self.coeff(coefficient[0],
                                               coefficient[1],
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number)

            if common_factor:
                res = res + ')'

            res_list.append(res)

        return res_list

    def get_pdf_lines(self, matrix_element, ninitial, subproc_group = False):
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
                                         "XBK(IB(%d)),DSQRT(Q2FACT(IB(%d))))\n") % \
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

        res_str = '%+i' % total_coeff.numerator

        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + './%i.' % total_coeff.denominator

        if is_imaginary:
            res_str = res_str + '*imag1'

        return res_str + '*'

#===============================================================================
# ProcessExporterFortranSA
#===============================================================================
class ProcessExporterFortranSA(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 StandAlone format."""

    def copy_v4template(self):
        """Additional actions needed for setup of Template
        """
        
        super(ProcessExporterFortranSA, self).copy_v4template()

        try:
            subprocess.call([os.path.join('bin', 'standalone')],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = os.open(os.devnull, os.O_RDWR),
                            cwd=self.dir_path)
        except OSError:
            # Probably standalone already called
            pass

    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS and MODEL directories, to set up
        everything for running standalone
        """

        source_dir = os.path.join(self.dir_path, "Source")
        logger.info("Running make for Helas")
        subprocess.call(['make', '../lib/libdhelas3.a'],
                        stdout = open(os.devnull, 'w'), cwd=source_dir)
        logger.info("Running make for Model")
        subprocess.call(['make', '../lib/libmodel.a'],
                        stdout = open(os.devnull, 'w'), cwd=source_dir)


    #===========================================================================
    # Create proc_card_mg5.dat for Standalone directory
    #===========================================================================
    def finalize_v4_directory(self, matrix_elements, history, makejpg = False,
                              online = False):
        """Finalize Standalone MG4 directory by generation proc_card_mg5.dat"""

        if not misc.which('g77'):
            logger.info('Change makefiles to use gfortran')
            subprocess.call(['python','./bin/Passto_gfortran.py'], cwd=self.dir_path, \
                            stdout = open(os.devnull, 'w')) 

        self.make()

        # Write command history as proc_card_mg5
        if os.path.isdir(os.path.join(self.dir_path, 'Cards')):
            output_file = os.path.join(self.dir_path, 'Cards', 'proc_card_mg5.dat')
            output_file = open(output_file, 'w')
            text = ('\n'.join(history) + '\n') % misc.get_time_info()
            output_file.write(text)
            output_file.close()

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    def generate_subprocess_directory_v4(self, matrix_element,
                                         fortran_model):
        """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
        including the necessary matrix.f and nexternal.inc files"""

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

        # Create the matrix.f file and the nexternal.inc file
        filename = 'matrix.f'
        calls = self.write_matrix_element_v4(
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

        # Generate diagrams
        filename = "matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                             get('diagrams'),
                                          filename,
                                          model=matrix_element.get('processes')[0].\
                                             get('model'),
                                          amplitude='')
        logger.info("Generating Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        linkfiles = ['check_sa.f', 'coupl.inc', 'makefile']


        for file in linkfiles:
            ln('../%s' % file)

        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model):
        """Export a matrix element to a matrix.f file in MG4 standalone format"""

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

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/matrix_standalone_v4.inc')).read()
        file = file % replace_dict

        # Write the file
        writer.writelines(file)

        return len(filter(lambda call: call.find('#') != 0, helas_calls))

#===============================================================================
# ProcessExporterFortranME
#===============================================================================
class ProcessExporterFortranME(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadEvent format."""

    matrix_file = "matrix_madevent_v4.inc"

    def copy_v4template(self):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranME, self).copy_v4template()

    #===========================================================================
    # generate_subprocess_directory_v4 
    #===========================================================================
    def generate_subprocess_directory_v4(self, matrix_element,
                                         fortran_model,
                                         me_number):
        """Generate the Pxxxxx directory for a subprocess in MG4 madevent,
        including the necessary matrix.f and various helper files"""

        cwd = os.getcwd()
        path = os.path.join(self.dir_path, 'SubProcesses')

        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')

        try:
             os.chdir(path)
        except OSError, error:
            error_msg = "The directory %s should exist in order to be able " % path + \
                        "to \"export\" in it. If you see this error message by " + \
                        "typing the command \"export\" please consider to use " + \
                        "instead the command \"output\". "
            raise MadGraph5Error, error_msg 


        pathdir = os.getcwd()

        # Create the directory PN_xx_xxxxx in the specified path
        subprocdir = "P%s" % matrix_element.get('processes')[0].shell_string()
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

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Create the matrix.f file, auto_dsig.f file and all inc files
        filename = 'matrix.f'
        calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename),
                                                matrix_element,
                                                fortran_model)

        filename = 'auto_dsig.f'
        self.write_auto_dsig_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = 'configs.inc'
        mapconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            matrix_element)

        filename = 'coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                             mapconfigs,
                             matrix_element)

        filename = 'decayBW.inc'
        self.write_decayBW_file(writers.FortranWriter(filename),
                           s_and_t_channels)

        filename = 'dname.mg'
        self.write_dname_file(writers.FortranWriter(filename),
                         matrix_element.get('processes')[0].shell_string())

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                         me_number)

        filename = 'leshouche.inc'
        self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = 'maxamps.inc'
        self.write_maxamps_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           ncolor,
                           len(matrix_element.get('processes')),
                           1)

        filename = 'mg.sym'
        self.write_mg_sym_file(writers.FortranWriter(filename),
                          matrix_element)

        filename = 'ncombs.inc'
        self.write_ncombs_file(writers.FortranWriter(filename),
                          nexternal)

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(mapconfigs))


        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'props.inc'
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        # Find config symmetries and permutations
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(matrix_element)

        filename = 'symswap.inc'
        self.write_symswap_file(writers.FortranWriter(filename),
                                ident_perms)

        filename = 'symfact.dat'
        self.write_symfact_file(writers.FortranWriter(filename),
                           symmetry)

        # Generate diagrams
        filename = "matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                             get('diagrams'),
                                          filename,
                                          model=matrix_element.get('processes')[0].\
                                             get('model'),
                                          amplitude='')
        logger.info("Generating Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()


        linkfiles = ['addmothers.f',
                     'cluster.f',
                     'cluster.inc',
                     'coupl.inc',
                     'cuts.f',
                     'cuts.inc',
                     'driver.f',
                     'genps.f',
                     'genps.inc',
                     'initcluster.f',
                     'makefile',
                     'message.inc',
                     'myamp.f',
                     'reweight.f',
                     'run.inc',
                     'maxconfigs.inc',
                     'maxparticles.inc',
                     'setcuts.f',
                     'setscales.f',
                     'sudakov.inc',
                     'symmetry.f',
                     'unwgt.f']

        for file in linkfiles:
            ln('../' + file , '.')

        #import nexternal/leshouch in Source
        ln('nexternal.inc', '../../Source', log=False)
        ln('leshouche.inc', '../../Source', log=False)
        ln('maxamps.inc', '../../Source', log=False)

        # Return to SubProcesses dir
        os.chdir(pathdir)

        # Add subprocess to subproc.mg
        filename = 'subproc.mg'
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)

        # Return to original dir
        os.chdir(cwd)

        # Generate info page
        gen_infohtml.make_info_html(self.dir_path)


        if not calls:
            calls = 0
        return calls

    def finalize_v4_directory(self, matrix_elements, history, makejpg = False,
                              online = False):
        """Finalize ME v4 directory by creating jpeg diagrams, html
        pages,proc_card_mg5.dat and madevent.tar.gz."""

        # Write maxconfigs.inc based on max of ME's/subprocess groups
        filename = os.path.join(self.dir_path,'Source','maxconfigs.inc')
        self.write_maxconfigs_file(writers.FortranWriter(filename),
                                   matrix_elements)
        
        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = os.path.join(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements)
        
        # Touch "done" file
        os.system('touch %s/done' % os.path.join(self.dir_path,'SubProcesses'))

        if not misc.which('g77'):
            logger.info('Change makefiles to use gfortran')
            subprocess.call(['python','./bin/Passto_gfortran.py'], cwd=self.dir_path, \
                            stdout = open(os.devnull, 'w')) 

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
                subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'gen_jpeg-pl')],
                                stdout = devnull)
                os.chdir(os.path.pardir)

        logger.info("Generate web pages")
        # Create the WebPage using perl script

        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'gen_cardhtml-pl')], \
                                                                stdout = devnull)

        os.chdir(os.path.pardir)

        gen_infohtml.make_info_html(self.dir_path)
        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'gen_crossxhtml-pl')],
                        stdout = devnull)
        [mv(name, './HTML/') for name in os.listdir('.') if \
                            (name.endswith('.html') or name.endswith('.jpg')) and \
                            name != 'index.html']               

        # Write command history as proc_card_mg5
        if os.path.isdir('Cards'):
            output_file = os.path.join('Cards', 'proc_card_mg5.dat')
            output_file = open(output_file, 'w')
            text = ('\n'.join(history) + '\n') % misc.get_time_info()
            output_file.write(text)
            output_file.close()

        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'gen_cardhtml-pl')],
                        stdout = devnull)

        # Run "make" to generate madevent.tar.gz file
        if os.path.exists(os.path.join('SubProcesses', 'subproc.mg')):
            if os.path.exists('madevent.tar.gz'):
                os.remove('madevent.tar.gz')
            subprocess.call(['make'], stdout = devnull)


        if online:
            # Touch "Online" file
            os.system('touch %s/Online' % self.dir_path)

        subprocess.call([os.path.join(old_pos, self.dir_path, 'bin', 'gen_cardhtml-pl')],
                        stdout = devnull)

        #return to the initial dir
        os.chdir(old_pos)               

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                proc_id = "", config_map = []):
        """Export a matrix element to a matrix.f file in MG4 madevent format"""

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

        # Extract helas calls
        helas_calls = fortran_model.get_matrix_element_calls(\
                    matrix_element)
        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element, config_map)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)

        # Extract JAMP lines
        jamp_lines = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)).read()
        file = file % replace_dict

        # Write the file
        writer.writelines(file)

        return len(filter(lambda call: call.find('#') != 0, helas_calls)), ncolor

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
            dsig_line = "pd(IPROC)*dsiguu"
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(IPROC)*conv*dsiguu"

        replace_dict['dsig_line'] = dsig_line

        # Extract pdf lines
        pdf_lines = self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
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

        file = open(os.path.join(_file_path, \
                          'iolibs/template_files/auto_dsig_v4.inc')).read()
        file = file % replace_dict

        # Write the file
        writer.writelines(file)

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
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        configs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [c[1] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial)

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial):
        """Write the actual configs.inc file.
        configs is the diagrams corresponding to configs,
        mapconfigs gives the diagram number for each config."""

        lines = []

        s_and_t_channels = []

        minvert = min([max(diag.get_vertex_leg_numbers()) for diag in configs])

        nconfigs = 0

        for iconfig, helas_diag in enumerate(configs):
            if any([vert > minvert for vert in
                    helas_diag.get_vertex_leg_numbers()]):
                # Only 3-vertices allowed in configs.inc
                continue
            nconfigs += 1

            # Need to reorganize the topology so that we start with all
            # final state external particles and work our way inwards

            schannels, tchannels = helas_diag.get('amplitudes')[0].\
                                              get_s_and_t_channels(ninitial)

            s_and_t_channels.append([schannels, tchannels])

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("data mapconfig(%d)/%d/" % (nconfigs,
                                                     mapconfigs[iconfig]))

            for vert in allchannels:
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), nconfigs, len(daughters),
                              ",".join([str(d) for d in daughters])))
                if vert in schannels:
                    lines.append("data sprop(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  last_leg.get('id')))
                    lines.append("data tprid(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))
                elif vert in tchannels[:-1]:
                    lines.append("data tprid(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  abs(last_leg.get('id'))))
                    lines.append("data sprop(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))

        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%d/" % nconfigs)

        # Write the file
        writer.writelines(lines)

        return s_and_t_channels

    #===========================================================================
    # write_decayBW_file
    #===========================================================================
    def write_decayBW_file(self, writer, s_and_t_channels):
        """Write the decayBW.inc file for MadEvent"""

        lines = []

        booldict = {False: ".false.", True: ".true."}

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

                lines.append("pmass(%d,%d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines.append("pwidth(%d,%d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines.append("pow(%d,%d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))

        # Write the file
        writer.writelines(lines)

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
                    legs = copy.copy(proc.get('legs'))
                    legs.insert(0, legs.pop(1))
                    proc.set("legs", legs)
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


        # Write out lines for symswap.inc file (used to permute the
        # external leg momenta
        lines = [ "%3r %3r" %(i+1, s) for i,s in enumerate(symmetry) if s != 0] 

        # Write the file
        writer.writelines(lines)

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

    #===========================================================================
    # copy the Template in a new directory.
    #===========================================================================
    def copy_v4template(self):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranME, self).copy_v4template()

        # Update values in run_config.inc
        run_config = \
                open(os.path.join(self.dir_path, 'Source', 'run_config.inc')).read()
        run_config = run_config.replace("ChanPerJob=5",
                                        "ChanPerJob=2")
        open(os.path.join(self.dir_path, 'Source', 'run_config.inc'), 'w').\
                                    write(run_config)
        # Update values in generate_events
        generate_events = \
                open(os.path.join(self.dir_path, 'bin', 'generate_events')).read()
        generate_events = generate_events.replace(\
                                        "$dirbin/refine $a $mode $n 5 $t",
                                        "$dirbin/refine $a $mode $n 1 $t")
        open(os.path.join(self.dir_path, 'bin', 'generate_events'), 'w').\
                                    write(generate_events)

    #===========================================================================
    # generate_subprocess_directory_v4
    #===========================================================================
    def generate_subprocess_directory_v4(self, subproc_group,
                                         fortran_model,
                                         group_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files"""

        if not isinstance(subproc_group, group_subprocs.SubProcessGroup):
            raise base_objects.PhysicsObject.PhysicsObjectError,\
                  "subproc_group object not SubProcessGroup"

        if not self.model:
            self.model = subproc_group.get('matrix_elements')[0].\
                         get('processes')[0].get('model')

        cwd = os.getcwd()
        path = os.path.join(self.dir_path, 'SubProcesses')

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

        for ime, matrix_element in \
                enumerate(matrix_elements):
            filename = 'matrix%d.f' % (ime+1)
            calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename), 
                                                matrix_element,
                                                fortran_model,
                                                str(ime+1),
                                                subproc_group.get('diagram_maps')[\
                                                                              ime])

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
                                              amplitude='')
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

        filename = 'config_subproc_map.inc'
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           subproc_diagrams_for_config)

        filename = 'configs.inc'
        nconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            subproc_group,
            subproc_diagrams_for_config)

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

        filename = 'symfact.dat'
        self.write_symfact_file(writers.FortranWriter(filename),
                           symmetry)

        filename = 'symperms.inc'
        self.write_symperms_file(writers.FortranWriter(filename),
                           perms)

        # Generate jpgs -> pass in make_html
        #os.system(os.path.join('..', '..', 'bin', 'gen_jpeg-pl'))

        linkfiles = ['addmothers.f',
                     'cluster.f',
                     'cluster.inc',
                     'coupl.inc',
                     'cuts.f',
                     'cuts.inc',
                     'driver.f',
                     'genps.f',
                     'genps.inc',
                     'initcluster.f',
                     'makefile',
                     'message.inc',
                     'myamp.f',
                     'reweight.f',
                     'run.inc',
                     'maxconfigs.inc',
                     'maxparticles.inc',
                     'setcuts.f',
                     'setscales.f',
                     'sudakov.inc',
                     'symmetry.f',
                     'unwgt.f']

        for file in linkfiles:
            ln('../' + file , '.')

        #import nexternal/leshouch in Source
        ln('nexternal.inc', '../../Source', log=False)
        ln('leshouche.inc', '../../Source', log=False)
        ln('maxamps.inc', '../../Source', log=False)

        # Return to SubProcesses dir
        os.chdir(pathdir)

        # Add subprocess to subproc.mg
        filename = 'subproc.mg'
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)
        
        # Generate info page
        gen_infohtml.make_info_html(os.path.pardir)
        
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

        file = open(os.path.join(_file_path, \
                       'iolibs/template_files/super_auto_dsig_group_v4.inc')).read()
        file = file % replace_dict

        # Write the file
        writer.writelines(file)

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
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, diagrams_for_config, maxflows,
                                   matrix_elements):
        """Write the coloramps.inc file for MadEvent in Subprocess group mode"""

        # Create a map from subprocess (matrix element) to a list of the diagrams corresponding to each config

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
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, subproc_group, diagrams_for_config):
        """Write the configs.inc file with topology information for a
        subprocess group. Use the first subprocess with a diagram for each
        configuration."""

        matrix_elements = subproc_group.get('matrix_elements')

        diagrams = []
        config_numbers = []
        for iconfig, config in enumerate(diagrams_for_config):
            # Check if any diagrams correspond to this config
            if set(config) == set([0]):
                continue
            subproc, diag = [(i,d - 1) for (i,d) in enumerate(config) \
                             if d > 0][0]
            diagrams.append(matrix_elements[subproc].get('diagrams')[diag])
            config_numbers.append(iconfig + 1)

        # Extract number of external particles
        (nexternal, ninitial) = subproc_group.get_nexternal_ninitial()

        return len(diagrams), \
               self.write_configs_file_from_diagrams(writer, diagrams,
                                                config_numbers,
                                                nexternal, ninitial)

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

#===============================================================================
# UFO_model_to_mg4
#===============================================================================
python_to_fortran = lambda x: parsers.UFOExpressionParserFortran().parse(x)

class UFO_model_to_mg4(object):
    """ A converter of the UFO-MG5 Model to the MG4 format """
    
    def __init__(self, model, output_path):
        """ initialization of the objects """
       
        self.model = model
        self.model_name = model['name']
        self.dir_path = output_path
        
        self.coups_dep = []    # (name, expression, type)
        self.coups_indep = []  # (name, expression, type)
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)
        self.params_ext = []   # external parameter
        self.p_to_f = parsers.UFOExpressionParserFortran()
    
    def pass_parameter_to_case_insensitive(self):
        """modify the parameter if some of them are identical up to the case"""
        
        lower_dict={}
        duplicate = set()
        keys = self.model['parameters'].keys()
        for key in keys:
            for param in self.model['parameters'][key]:
                lower_name = param.name.lower()
                try:
                    lower_dict[lower_name].append(param)
                except KeyError:
                    lower_dict[lower_name] = [param]
                else:
                    duplicate.add(lower_name)
        
        if not duplicate:
            return
        
        re_expr = r'''\b(%s)\b'''
        to_change = []
        change={}
        for value in duplicate:
            for i, var in enumerate(lower_dict[value][1:]):
                to_change.append(var.name)
                change[var.name] = '%s__%s' %( var.name.lower(), i+2)
                var.name = '%s__%s' %( var.name.lower(), i+2)
        
        replace = lambda match_pattern: change[match_pattern.groups()[0]]
        
        rep_pattern = re.compile(re_expr % '|'.join(to_change))
        for key in keys:
            if key == ('external',):
                continue
            for param in self.model['parameters'][key]: 
                param.expr = rep_pattern.sub(replace, param.expr)
            
        
    def refactorize(self, wanted_couplings = []):    
        """modify the couplings to fit with MG4 convention """
            
        # Keep only separation in alphaS        
        keys = self.model['parameters'].keys()
        keys.sort(key=len)
        for key in keys:
            if key == ('external',):
                self.params_ext += self.model['parameters'][key]
            elif 'aS' in key:
                self.params_dep += self.model['parameters'][key]
            else:
                self.params_indep += self.model['parameters'][key]
        # same for couplings
        keys = self.model['couplings'].keys()
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if 'aS' in key:
                self.coups_dep += [c for c in coup_list if
                                   (not wanted_couplings or c.name in \
                                    wanted_couplings)]
            else:
                self.coups_indep += [c for c in coup_list if
                                     (not wanted_couplings or c.name in \
                                      wanted_couplings)]
                
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        index = self.params_dep.index('G')
        self.params_indep.insert(0, self.params_dep.pop(index))
        index = self.params_dep.index('sqrt__aS')
        self.params_indep.insert(0, self.params_dep.pop(index))
        
    def build(self, wanted_couplings = [], full=True):
        """modify the couplings to fit with MG4 convention and creates all the 
        different files"""
        
        self.pass_parameter_to_case_insensitive()
        self.refactorize(wanted_couplings)

        # write the files
        if full:
            self.write_all()

    def open(self, name, comment='c', format='default'):
        """ Open the file name in the correct directory and with a valid
        header."""
        
        file_path = os.path.join(self.dir_path, name)
        
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
        self.create_intparam_def()
        
        
        # definition of the coupling.
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        
        # the makefile
        self.create_makeinc()
        self.create_param_write()
        
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
        file_to_link = ['formats.inc', 'lha_read.f', 'makefile','printout.f', \
                        'rw_para.f', 'testprog.f', 'rw_para.f']
    
        for filename in file_to_link:
            cp( MG5DIR + '/models/template_files/fortran/' + filename, self.dir_path)

    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        
        # Write header
        header = """double precision G
                common/strong/ G
                 
                double complex gal(2)
                common/weak/ gal

                """        
        fsock.writelines(header)
        
        # Write the Mass definition/ common block
        masses = set()
        widths = set()
        for particle in self.model.get('particles'):
            #find masses
            one_mass = particle.get('mass')
            if one_mass.lower() != 'zero':
                masses.add(one_mass)
                
            # find width
            one_width = particle.get('width')
            if one_width.lower() != 'zero':
                widths.add(one_width)
            
        
        fsock.writelines('double precision '+','.join(masses)+'\n')
        fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
        if widths:
            fsock.writelines('double precision '+','.join(widths)+'\n')
            fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
        
        # Write the Couplings
        coupling_list = [coupl.name for coupl in self.coups_dep + self.coups_indep]       
        fsock.writelines('double complex '+', '.join(coupling_list)+'\n')
        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        
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

        #find mass/ width since they are already define
        already_def = set()
        for particle in self.model.get('particles'):
            already_def.add(particle.get('mass').lower())
            already_def.add(particle.get('width').lower())

        is_valid = lambda name: name!='G' and name.lower() not in already_def
        
        real_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'real'
                            and is_valid(param.name)]

        real_parameters += [param.name for param in self.params_ext 
                            if param.type == 'real'and 
                               is_valid(param.name)]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        
        complex_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'complex']


        fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
        fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')                
    
    

    def create_intparam_def(self):
        """ create intparam_definition.inc """

        fsock = self.open('intparam_definition.inc', format='fortran')
        
        fsock.write_comments(\
                "Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        
        for param in self.params_indep:
            if param.name == 'ZERO':
                continue
            fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
        
        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        for param in self.params_dep:
            fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))

        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")
        fsock.writelines(""" gal(1) = 1d0
                             gal(2) = 1d0
                         """)

    
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = 25
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        for i in range(nb_coup_indep):
            data = self.coups_indep[nb_def_by_file * i: 
                             min(len(self.coups_indep), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data)
            
        for i in range(nb_coup_dep):
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data)        
        
        
    def create_couplings_main(self, nb_def_by_file=25):
        """ create couplings.f """

        fsock = self.open('couplings.f', format='fortran')
        
        fsock.writelines("""subroutine coup(readlha)

                            implicit none
                            logical readlha
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            
                            include \'input.inc\'
                            include \'coupl.inc\'
                            include \'intparam_definition.inc\'\n\n
                         """)
        
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        fsock.writelines('if (readlha) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        fsock.writelines('''\nendif\n''')
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')


    def create_couplings_part(self, nb_file, data):
        """ create couplings[nb_file].f containing information coming from data
        """
        
        fsock = self.open('couplings%s.f' % nb_file, format='fortran')
        fsock.writelines("""subroutine coup%s()
        
          implicit none
          double precision PI
          parameter  (PI=3.141592653589793d0)
      
      
          include 'input.inc'
          include 'coupl.inc'
                        """ % nb_file)
        
        for coupling in data:            
            fsock.writelines('%s = %s' % (coupling.name,
                                          self.p_to_f.parse(coupling.expr)))
        fsock.writelines('end')


    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o '
        
        nb_coup_indep = 1 + len(self.coups_dep) // 25 
        nb_coup_dep = 1 + len(self.coups_indep) // 25
        text += ' '.join(['couplings%s.o' % (i+1) \
                                  for i in range(nb_coup_dep + nb_coup_indep) ])
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
                                                         if data.name != 'ZERO']
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data.name) for data in self.params_dep]
        
        fsock.writelines('\n'.join(lines))                
        
 
    
    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock.lower()] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in self.params_ext]
        fsock.writelines('\n'.join(external_param))

        
    def create_param_read(self):    
        """create param_read"""
    
        def format_line(parameter):
            """return the line for the ident_card corresponding to this 
            parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)""" \
                % {'name': parameter.name,
                   'value': self.p_to_f.parse(str(parameter.value.real))}
        
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
                
        
        fsock.writelines('\n'.join(res_strings))

    def create_param_card(self):
        """ create the param_card.dat """

        out_path = os.path.join(self.dir_path, 'param_card.dat')
        param_writer.ParamCardWriter(self.model, out_path)

