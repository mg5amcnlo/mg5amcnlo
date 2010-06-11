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

import fractions
import glob
import logging
import os
import re
import shutil
import subprocess

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.misc as misc
import madgraph.iolibs.template_files as Template

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('export_v4')

#===============================================================================
# copy the Template in a new directory.
#===============================================================================
def copy_v4template(mgme_dir, dir_path, model_dir, clean):
    """create the directory run_name as a copy of the Template
       and import the model, Helas, and clean the directory 
    """
    
    #First copy the full template tree if dir_path doesn't exit
    if not os.path.isdir(dir_path):
        logger.info('initialize a new directory: %s' % \
                    os.path.basename(dir_path))
        shutil.copytree(os.path.join(mgme_dir, 'Template'), dir_path, True)

    #Ensure that the Template is clean
    if clean:
        logger.info('remove old information in %s' % os.path.basename(dir_path))
        old_pos = os.getcwd()
        os.chdir(dir_path)
        if os.environ.has_key('MADGRAPH_BASE'):
            subprocess.call([os.path.join('bin', 'clean_template'), '--web'])
        else:
            try:
                subprocess.call([os.path.join('bin', 'clean_template')])
            except Exception, why:
                logger.error('Failed to clean correctly Template. ' + \
                                     'The following error is returned:\n %s' % why)
        os.chdir(old_pos)
        
        #Write version info
        MG_version = misc.get_pkg_info()
        open(os.path.join(dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                          MG_version['version'])
        
#===============================================================================
# copy the Template in a new directory and set up Standalone MG
#===============================================================================
def copy_v4standalone(mgme_dir, dir_path, model_dir, clean):
    """create the directory run_name as a copy of the Template,
       run standalone, import the model and Helas, and clean the directory 
    """
    
    #First copy the full template tree if dir_path doesn't exit
    if not os.path.isdir(dir_path):
        logger.info('initialize a new directory: %s' % \
                    os.path.basename(dir_path))
        shutil.copytree(os.path.join(mgme_dir, 'Template'), dir_path, True)

    # Run standalone
    logger.info('Setup directory %s for standalone' % \
                os.path.basename(dir_path))
    old_pos = os.getcwd()
    os.chdir(dir_path)
    try:
        subprocess.call([os.path.join('bin', 'standalone')],
                        stdout = os.open(os.devnull, os.O_RDWR))
    except OSError:
        # Probably standalone already called
        pass
    os.chdir(old_pos)

    #Ensure that the Template is clean
    if clean:
        logger.info('remove old information in %s' % \
                    os.path.basename(dir_path))
        old_pos = os.getcwd()
        os.chdir(dir_path)
        for pdir in glob.glob(os.path.join("SubProcesses","P*")):
            shutil.rmtree(pdir)
        #Write version info
        MG_version = misc.get_pkg_info()
        open(os.path.join('SubProcesses', 'MGVersion.txt'), 'w').write(
                                                          MG_version['version'])

        # Copy the HELAS directory
        
        for file in glob.glob(os.path.join("..", "HELAS", "*")):
            if os.path.isfile(file):
                shutil.copy2(file,
                            os.path.join('Source', 'DHELAS'))
        shutil.move(os.path.join('Source', 'DHELAS', 'Makefile.template'),
                    os.path.join('Source', 'DHELAS', 'Makefile'))
        os.chdir(old_pos)
        
#===============================================================================
# Make the Helas and Model directories for Standalone directory
#===============================================================================
def make_v4standalone(dir_path):
    """Run make in the DHELAS and MODEL directories, to set up
    everything for running standalone
    """
    
    # Run standalone
    old_pos = os.getcwd()
    os.chdir(os.path.join(dir_path, "Source"))
    logger.info("Running make for Helas")
    subprocess.call(['make', '../lib/libdhelas3.a'],
                    stdout = open(os.devnull, 'w'))
    logger.info("Running make for Model")
    subprocess.call(['make', '../lib/libmodel.a'],
                    stdout = open(os.devnull, 'w'))
    os.chdir(old_pos)
        
#===============================================================================
# write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
#===============================================================================
def write_procdef_mg5(file_pos, modelname, process_str):
    """ write an equivalent of the MG4 proc_card in order that all the Madevent
    Perl script of MadEvent4 are still working properly for pure MG5 run."""
    
    proc_card_template = Template.mg4_proc_card.mg4_template
    process_template = Template.mg4_proc_card.process_template
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
# Create Web Page via external routines
#===============================================================================
def finalize_madevent_v4_directory(dir_path, makejpg, history):
    """call the perl script creating the web interface for MadEvent"""

    old_pos = os.getcwd()
    os.chdir(os.path.join(dir_path, 'SubProcesses'))
    P_dir_list = [proc for proc in os.listdir('.') if os.path.isdir(proc) and \
                                                                proc[0] == 'P']
    
    devnull = os.open(os.devnull, os.O_RDWR)
    # Convert the poscript in jpg files (if authorize)
    if makejpg:
        logger.info("Generate jpeg diagrams")
        for Pdir in P_dir_list:
            os.chdir(Pdir)
            subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_jpeg-pl')],
                            stdout = devnull)
            os.chdir(os.path.pardir)

    logger.info("Generate web pages")
    # Create the WebPage using perl script

    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_cardhtml-pl')], \
                                                            stdout = devnull)
    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_infohtml-pl')], \
                                                            stdout = devnull)
    os.chdir(os.path.pardir)
    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_crossxhtml-pl')],
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

    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_cardhtml-pl')],
                    stdout = devnull)
    
    # Run "make" to generate madevent.tar.gz file
    if os.path.exists(os.path.join('SubProcesses', 'subproc.mg')):
        if os.path.exists('madevent.tar.gz'):
            os.remove('madevent.tar.gz')
        subprocess.call(['make'], stdout = devnull)
    
    
    subprocess.call([os.path.join(old_pos, dir_path, 'bin', 'gen_cardhtml-pl')],
                    stdout = devnull)
    
    #return to the initial dir
    os.chdir(old_pos)               
           
#===============================================================================
# write_matrix_element_v4_standalone
#===============================================================================
def write_matrix_element_v4_standalone(fsock, matrix_element, fortran_model):
    """Export a matrix element to a matrix.f file in MG4 standalone format"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    writer = FortranWriter()
    # Set lowercase/uppercase Fortran code
    FortranWriter.downcase = False

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract process info lines
    process_lines = get_process_info_lines(matrix_element)
    replace_dict['process_lines'] = process_lines

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    replace_dict['nexternal'] = nexternal

    # Extract ncomb
    ncomb = matrix_element.get_helicity_combinations()
    replace_dict['ncomb'] = ncomb

    # Extract helicity lines
    helicity_lines = get_helicity_lines(matrix_element)
    replace_dict['helicity_lines'] = helicity_lines

    # Extract overall denominator
    # Averaging initial state color, spin, and identical FS particles
    den_factor_line = get_den_factor_line(matrix_element)
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
    color_data_lines = get_color_data_lines(matrix_element)
    replace_dict['color_data_lines'] = "\n".join(color_data_lines)

    # Extract helas calls
    helas_calls = fortran_model.get_matrix_element_calls(\
                matrix_element)
    replace_dict['helas_calls'] = "\n".join(helas_calls)

    # Extract JAMP lines
    jamp_lines = get_JAMP_lines(matrix_element)
    replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

    file = open(os.path.join(_file_path, \
                      'iolibs/template_files/matrix_standalone_v4.inc')).read()
    file = file % replace_dict

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return len(filter(lambda call: call.find('#') != 0, helas_calls))

#===============================================================================
# write_matrix_element_v4_madevent
#===============================================================================
def write_matrix_element_v4_madevent(fsock, matrix_element, fortran_model):
    """Export a matrix element to a matrix.f file in MG4 madevent format"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    writer = FortranWriter()
    # Set lowercase/uppercase Fortran code
    FortranWriter.downcase = False

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract process info lines
    process_lines = get_process_info_lines(matrix_element)
    replace_dict['process_lines'] = process_lines

    # Extract ncomb
    ncomb = matrix_element.get_helicity_combinations()
    replace_dict['ncomb'] = ncomb

    # Extract helicity lines
    helicity_lines = get_helicity_lines(matrix_element)
    replace_dict['helicity_lines'] = helicity_lines

    # Extract IC line
    ic_line = get_ic_line(matrix_element)
    replace_dict['ic_line'] = ic_line

    # Extract overall denominator
    # Averaging initial state color, spin, and identical FS particles
    den_factor_line = get_den_factor_line(matrix_element)
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
    color_data_lines = get_color_data_lines(matrix_element)
    replace_dict['color_data_lines'] = "\n".join(color_data_lines)

    # Extract helas calls
    helas_calls = fortran_model.get_matrix_element_calls(\
                matrix_element)
    replace_dict['helas_calls'] = "\n".join(helas_calls)

    # Extract JAMP lines
    jamp_lines = get_JAMP_lines(matrix_element)
    replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

    file = open(os.path.join(_file_path, \
                      'iolibs/template_files/matrix_madevent_v4.inc')).read()
    file = file % replace_dict
    
    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return len(filter(lambda call: call.find('#') != 0, helas_calls))

#===============================================================================
# write_auto_dsig_file
#===============================================================================
def write_auto_dsig_file(fsock, matrix_element, fortran_model):
    """Write the auto_dsig.f file for the differential cross section
    calculation, includes pdf call information"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    nexternal, ninitial = matrix_element.get_nexternal_ninitial()

    if ninitial < 1 or ninitial > 2:
        raise FortranWriter.FortranWriterError, \
              """Need ninitial = 1 or 2 to write auto_dsig file"""

    writer = FortranWriter()

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract process info lines
    process_lines = get_process_info_lines(matrix_element)
    replace_dict['process_lines'] = process_lines

    pdf_lines = get_pdf_lines(matrix_element, ninitial)
    replace_dict['pdf_lines'] = pdf_lines

    if ninitial == 1:
        # No conversion, since result of decay should be given in GeV
        dsig_line = "pd(IPROC)*dsiguu"
    else:
        # Convert result (in GeV) to pb
        dsig_line = "pd(IPROC)*conv*dsiguu"

    replace_dict['dsig_line'] = dsig_line

    file = open(os.path.join(_file_path, \
                      'iolibs/template_files/auto_dsig_v4.inc')).read()
    file = file % replace_dict

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

#===============================================================================
# write_coloramps_file
#===============================================================================
def write_coloramps_file(fsock, matrix_element, fortran_model):
    """Write the coloramps.inc file for MadEvent"""

    writer = FortranWriter()

    lines = get_icolamp_lines(matrix_element)

    # Write the file
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_configs_file
#===============================================================================
def write_configs_file(fsock, matrix_element, fortran_model):
    """Write the configs.inc file for MadEvent"""

    writer = FortranWriter()

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

    lines = []

    iconfig = 0

    s_and_t_channels = []

    for idiag, diag in enumerate(matrix_element.get('base_amplitude').\
                                                get('diagrams')):
        if any([len(vert.get('legs')) > 3 for vert in diag.get('vertices')]):
            # Only 3-vertices allowed in configs.inc
            continue
        iconfig = iconfig + 1
        helas_diag = matrix_element.get('diagrams')[idiag]
        amp_number = helas_diag.get('amplitudes')[0].get('number')
        lines.append("# Diagram %d, Amplitude %d" % \
                     (helas_diag.get('number'), amp_number))
        # Correspondance between the config and the amplitudes
        lines.append("data mapconfig(%d)/%d/" % (iconfig, amp_number))

        # Need to reorganize the topology so that we start with all
        # final state external particles and work our way inwards

        schannels, tchannels = helas_diag.get('amplitudes')[0].\
                                     get_s_and_t_channels(ninitial)

        s_and_t_channels.append([schannels, tchannels])

        # Write out propagators for s-channel and t-channel vertices
        allchannels = schannels
        if len(tchannels) > 1:
            # Write out tchannels only if there are any non-trivial ones
            allchannels = schannels + tchannels

        for vert in allchannels:
            daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
            last_leg = vert.get('legs')[-1]
            lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                         (last_leg.get('number'), iconfig, len(daughters),
                          ",".join([str(d) for d in daughters])))
            if vert in schannels:
                lines.append("data sprop(%d,%d)/%d/" % \
                             (last_leg.get('number'), iconfig,
                              last_leg.get('id')))
            elif vert in tchannels[:-1]:
                lines.append("data tprid(%d,%d)/%d/" % \
                             (last_leg.get('number'), iconfig,
                              last_leg.get('id')))

    # Write out number of configs
    lines.append("# Number of configs")
    lines.append("data mapconfig(0)/%d/" % iconfig)

    # Write the file
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return iconfig, s_and_t_channels

#===============================================================================
# write_decayBW_file
#===============================================================================
def write_decayBW_file(fsock, matrix_element, fortran_model,
                       s_and_t_channels):
    """Write the decayBW.inc file for MadEvent"""

    writer = FortranWriter()

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
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_dname_file
#===============================================================================
def write_dname_file(fsock, matrix_element, fortran_model):
    """Write the dname.mg file for MG4"""

    line = "DIRNAME=P%s" % \
           matrix_element.get('processes')[0].shell_string_v4()

    # Write the file
    fsock.write(line + "\n")

    return True

#===============================================================================
# write_iproc_file
#===============================================================================
def write_iproc_file(fsock, matrix_element, fortran_model):
    """Write the iproc.inc file for MG4"""

    writer = FortranWriter()

    line = "%d" % \
           matrix_element.get('processes')[0].get('id')

    # Write the file
    writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_leshouche_file
#===============================================================================
def write_leshouche_file(fsock, matrix_element, fortran_model):
    """Write the leshouche.inc file for MG4"""

    writer = FortranWriter()

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

            else:
                # First build a color representation dictionnary
                repr_dict = {}
                for l in legs:
                    repr_dict[l.get('number')] = \
                        proc.get('model').get_particle(l.get('id')).get_color()
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
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_maxamps_file
#===============================================================================
def write_maxamps_file(fsock, matrix_element, fortran_model):
    """Write the maxamps.inc file for MG4."""

    writer = FortranWriter()

    file = "       integer    maxamps\n"
    file = file + "parameter (maxamps=%d)" % \
           len(matrix_element.get_all_amplitudes())

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_mg_sym_file
#===============================================================================
def write_mg_sym_file(fsock, matrix_element, fortran_model):
    """Write the mg.sym file for MadEvent."""

    writer = FortranWriter()

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
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_ncombs_file
#===============================================================================
def write_ncombs_file(fsock, matrix_element, fortran_model):
    """Write the ncombs.inc file for MadEvent."""

    writer = FortranWriter()

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

    # ncomb (used for clustering) is 2^(nexternal + 1)
    file = "       integer    n_max_cl\n"
    file = file + "parameter (n_max_cl=%d)" % (2 ** (nexternal + 1))

    # Write the file

    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_nexternal_file
#===============================================================================
def write_nexternal_file(fsock, matrix_element, fortran_model):
    """Write the nexternal.inc file for MG4"""

    writer = FortranWriter()

    replace_dict = {}

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    replace_dict['nexternal'] = nexternal
    replace_dict['ninitial'] = ninitial

    file = """ \
      integer    nexternal
      parameter (nexternal=%(nexternal)d)
      integer    nincoming
      parameter (nincoming=%(ninitial)d)""" % replace_dict

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_ngraphs_file
#===============================================================================
def write_ngraphs_file(fsock, matrix_element, fortran_model, nconfigs):
    """Write the ngraphs.inc file for MG4. Needs input from
    write_configs_file."""

    writer = FortranWriter()

    file = "       integer    n_max_cg\n"
    file = file + "parameter (n_max_cg=%d)" % nconfigs

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_pmass_file
#===============================================================================
def write_pmass_file(fsock, matrix_element, fortran_model):
    """Write the pmass.inc file for MG4"""

    writer = FortranWriter()

    model = matrix_element.get('processes')[0].get('model')

    lines = []
    for wf in matrix_element.get_external_wavefunctions():
        mass = model.get('particle_dict')[wf.get('pdg_code')].get('mass')
        if mass.lower() != "zero":
            mass = "abs(%s)" % mass

        lines.append("pmass(%d)=%s" % \
                     (wf.get('number_external'), mass))

    # Write the file
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_props_file
#===============================================================================
def write_props_file(fsock, matrix_element, fortran_model, s_and_t_channels):
    """Write the props.inc file for MadEvent. Needs input from
    write_configs_file."""

    writer = FortranWriter()

    lines = []

    particle_dict = matrix_element.get('processes')[0].get('model').\
                    get('particle_dict')

    for iconf, configs in enumerate(s_and_t_channels):
        for vertex in configs[0] + configs[1][:-1]:
            leg = vertex.get('legs')[-1]
            particle = particle_dict[leg.get('id')]
            # Get mass
            if particle.get('mass').lower() == 'zero':
                mass = particle.get('mass')
            else:
                mass = "abs(%s)" % particle.get('mass')
            # Get width
            if particle.get('width') == 'zero':
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
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_subproc
#===============================================================================
def write_subproc(fsock, matrix_element, fortran_model):
    """Append this subprocess to the subproc.mg file for MG4"""

    line = "P%s" % \
           matrix_element.get('processes')[0].shell_string_v4()

    # Write line to file
    fsock.write(line + "\n")

    return True

#===============================================================================
# export the model
#===============================================================================
def export_model(model_path, process_path):
    """Configure the files/link of the process according to the model"""
    
    # Import the model
    for file in os.listdir(model_path):
        if os.path.isfile(os.path.join(model_path, file)):
            shutil.copy2(os.path.join(model_path, file), \
                                 os.path.join(process_path, 'Source', 'MODEL'))    


    #make the copy/symbolic link
    model_path = process_path + '/Source/MODEL/'
    ln(model_path + '/ident_card.dat', process_path + '/Cards', log=False)
    cp(model_path + '/param_card.dat', process_path + '/Cards')
    mv(model_path + '/param_card.dat', process_path + '/Cards/param_card_default.dat')
    ln(model_path + '/particles.dat', process_path + '/SubProcesses')
    ln(model_path + '/interactions.dat', process_path + '/SubProcesses')
    ln(model_path + '/coupl.inc', process_path + '/Source')
    ln(model_path + '/coupl.inc', process_path + '/SubProcesses')
    ln(process_path + '/Source/run.inc', process_path + '/SubProcesses', log=False)

#===============================================================================
# generate_subprocess_directory_v4_standalone
#===============================================================================
def generate_subprocess_directory_v4_standalone(matrix_element,
                                                fortran_model,
                                                path=os.getcwd()):
    """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
    including the necessary matrix.f and nexternal.inc files"""

    cwd = os.getcwd()

    # Create the directory PN_xx_xxxxx in the specified path
    dirpath = os.path.join(path, \
                   "P%s" % matrix_element.get('processes')[0].shell_string_v4())
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

    # Create the matrix.f file and the nexternal.inc file
    filename = 'matrix.f'
    calls = files.write_to_file(filename,
                                write_matrix_element_v4_standalone,
                                matrix_element,
                                fortran_model)

    filename = 'nexternal.inc'
    files.write_to_file(filename,
                        write_nexternal_file,
                        matrix_element,
                        fortran_model)

    filename = 'pmass.inc'
    files.write_to_file(filename,
                        write_pmass_file,
                        matrix_element,
                        fortran_model)

    filename = 'ngraphs.inc'
    files.write_to_file(filename,
                        write_ngraphs_file,
                        matrix_element,
                        fortran_model,
                        len(matrix_element.get_all_amplitudes()))

    linkfiles = ['check_sa.f', 'coupl.inc', 'makefile']

    
    for file in linkfiles:
        ln('../%s' % file)

    # Return to original PWD
    os.chdir(cwd)

    if not calls:
        calls = 0
    return calls
#===============================================================================
# generate_subprocess_directory_v4_madevent
#===============================================================================
def generate_subprocess_directory_v4_madevent(matrix_element,
                                              fortran_model,
                                              path=os.getcwd()):
    """Generate the Pxxxxx directory for a subprocess in MG4 madevent,
    including the necessary matrix.f and various helper files"""

    cwd = os.getcwd()

    os.chdir(path)

    pathdir = os.getcwd()

    # Create the directory PN_xx_xxxxx in the specified path
    subprocdir = "P%s" % matrix_element.get('processes')[0].shell_string_v4()
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

    # Create the matrix.f file, auto_dsig.f file and all inc files
    filename = 'matrix.f'
    calls = files.write_to_file(filename,
                                write_matrix_element_v4_madevent,
                                matrix_element,
                                fortran_model)

    filename = 'auto_dsig.f'
    files.write_to_file(filename,
                                write_auto_dsig_file,
                                matrix_element,
                                fortran_model)

    filename = 'coloramps.inc'
    files.write_to_file(filename,
                        write_coloramps_file,
                        matrix_element,
                        fortran_model)

    filename = 'configs.inc'
    nconfigs, s_and_t_channels = files.write_to_file(filename,
                        write_configs_file,
                        matrix_element,
                        fortran_model)

    filename = 'decayBW.inc'
    files.write_to_file(filename,
                        write_decayBW_file,
                        matrix_element,
                        fortran_model,
                        s_and_t_channels)

    filename = 'dname.mg'
    files.write_to_file(filename,
                        write_dname_file,
                        matrix_element,
                        fortran_model)

    filename = 'iproc.dat'
    files.write_to_file(filename,
                        write_iproc_file,
                        matrix_element,
                        fortran_model)

    filename = 'leshouche.inc'
    files.write_to_file(filename,
                        write_leshouche_file,
                        matrix_element,
                        fortran_model)

    filename = 'maxamps.inc'
    files.write_to_file(filename,
                        write_maxamps_file,
                        matrix_element,
                        fortran_model)

    filename = 'mg.sym'
    files.write_to_file(filename,
                        write_mg_sym_file,
                        matrix_element,
                        fortran_model)

    filename = 'ncombs.inc'
    files.write_to_file(filename,
                        write_ncombs_file,
                        matrix_element,
                        fortran_model)

    filename = 'nexternal.inc'
    files.write_to_file(filename,
                        write_nexternal_file,
                        matrix_element,
                        fortran_model)

    filename = 'ngraphs.inc'
    files.write_to_file(filename,
                        write_ngraphs_file,
                        matrix_element,
                        fortran_model,
                        nconfigs)

    filename = 'pmass.inc'
    files.write_to_file(filename,
                        write_pmass_file,
                        matrix_element,
                        fortran_model)

    filename = 'props.inc'
    files.write_to_file(filename,
                        write_props_file,
                        matrix_element,
                        fortran_model,
                        s_and_t_channels)

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

    # Return to SubProcesses dir
    os.chdir(pathdir)

    # Add subprocess to subproc.mg
    filename = 'subproc.mg'
    files.append_to_file(filename,
                        write_subproc,
                        matrix_element,
                        fortran_model)
    # Generate info page
    os.system(os.path.join('..', 'bin', 'gen_infohtml-pl'))

    # Return to original dir
    os.chdir(cwd)

    if not calls:
        calls = 0
    return calls

#===============================================================================
# Helper functions
#===============================================================================
def get_mg5_info_lines():
    """Return info lines for MG5, suitable to place at beginning of
    Fortran files"""

    info = misc.get_pkg_info()
    info_lines = ""
    if info and info.has_key('version') and  info.has_key('date'):
        info_lines = "C  Generated by MadGraph 5 v. %s, %s\n" % \
                     (info['version'], info['date'])
        info_lines = info_lines + \
                     "C  By the MadGraph Development Team\n" + \
                     "C  Please visit us at https://launchpad.net/madgraph5"
    else:
        info_lines = "C  Generated by MadGraph 5\n" + \
                     "C  By the MadGraph Development Team\n" + \
                     "C  Please visit us at https://launchpad.net/madgraph5"        

    return info_lines

def get_process_info_lines(matrix_element):
    """Return info lines describing the processes for this matrix element"""

    return"\n".join([ "C " + process.nice_string().replace('\n', '\nC * ') \
                     for process in matrix_element.get('processes')])


def get_helicity_lines(matrix_element):
    """Return the Helicity matrix definition lines for this matrix element"""

    helicity_line_list = []
    i = 0
    for helicities in matrix_element.get_helicity_matrix():
        i = i + 1
        int_list = [i, len(helicities)]
        int_list.extend(helicities)
        helicity_line_list.append(\
            ("DATA (NHEL(IHEL,%4r),IHEL=1,%d) /" + \
             ",".join(['%2r'] * len(helicities)) + "/") % tuple(int_list))

    return "\n".join(helicity_line_list)

def get_ic_line(matrix_element):
    """Return the IC definition line coming after helicities, required by
    switchmom in madevent"""

    nexternal = matrix_element.get_nexternal_ninitial()[0]
    int_list = range(1, nexternal + 1)

    return "DATA (IC(IHEL,1),IHEL=1,%i) /%s/" % (nexternal,
                                                 ",".join([str(i) for \
                                                           i in int_list]))

def get_color_data_lines(matrix_element, n=6):
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


def get_den_factor_line(matrix_element):
    """Return the denominator factor line for this matrix element"""

    return "DATA IDEN/%2r/" % \
           matrix_element.get_denominator_factor()

def get_icolamp_lines(matrix_element):
    """Return the ICOLAMP matrix, showing which AMPs are parts of
    which JAMPs."""

    ret_list = []

    booldict = {False: ".false.", True: ".true."}

    amplitudes = matrix_element.get_all_amplitudes()

    color_amplitudes = matrix_element.get_color_amplitudes()

    ret_list.append("logical icolamp(%d,%d)" % \
                    (len(amplitudes), len(color_amplitudes)))

    bool_list = []

    for coeff_list in color_amplitudes:

        # List of amplitude numbers used in this JAMP
        amp_list = [amp_number for (dummy, amp_number) in coeff_list]

        # List of True or False 
        bool_list.extend([(i + 1 in amp_list) for i in \
                          range(len(amplitudes))])
    # Add line
    ret_list.append("DATA icolamp/%s/" % \
                         ','.join(["%s" % booldict[i] for i in \
                                   bool_list]))

    return ret_list

def get_JAMP_lines(matrix_element):
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
            res = res + '%s(' % coeff(1, global_factor, False, 0)

        for (coefficient, amp_number) in coeff_list:
            if common_factor:
                res = res + "%sAMP(%d)" % (coeff(coefficient[0],
                                           coefficient[1] / abs(coefficient[1]),
                                           coefficient[2],
                                           coefficient[3]),
                                           amp_number)
            else:
                res = res + "%sAMP(%d)" % (coeff(coefficient[0],
                                           coefficient[1],
                                           coefficient[2],
                                           coefficient[3]),
                                           amp_number)

        if common_factor:
            res = res + ')'

        res_list.append(res)

    return res_list

def get_pdf_lines(matrix_element, ninitial):
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
            pdf_lines = pdf_lines + \
                   "IF (ABS(LPP(%d)) .GE. 1) THEN\nLP=SIGN(1,LPP(%d))\n" \
                         % (i + 1, i + 1)

            for initial_state in init_states:
                if initial_state in pdf_codes.keys():
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

#===============================================================================
# FortranWriter
#===============================================================================
class FortranWriter():
    """Routines for writing fortran lines. Keeps track of indentation
    and splitting of long lines"""

    class FortranWriterError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a FortranWriter."""
        pass

    # Parameters defining the output of the Fortran writer
    keyword_pairs = {'^if.+then\s*$': ('^endif', 2),
                     '^do\s+': ('^enddo\s*$', 2),
                     '^subroutine': ('^end\s*$', 0),
                     'function': ('^end\s*$', 0)}
    single_indents = {'^else\s*$':-2,
                      '^else\s*if.+then\s*$':-2}
    line_cont_char = '$'
    comment_char = 'c'
    downcase = False
    line_length = 71
    max_split = 10
    split_characters = "+-*/,) "
    comment_split_characters = " "

    # Private variables
    __indent = 0
    __keyword_list = []
    __comment_pattern = re.compile(r"^(\s*#|c$|(c\s+([^=]|$)))", re.IGNORECASE)

    def write_fortran_line(self, fsock, line):
        """Write a fortran line, with correct indent and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.FortranWriterError, \
                  "write_fortran_line must have a single line as argument"

        # Check if this line is a comment
        if self.__comment_pattern.search(line):
            # This is a comment
            myline = " " * (5 + self.__indent) + line.lstrip()[1:].lstrip()
            if FortranWriter.downcase:
                self.comment_char = self.comment_char.lower()
            else:
                self.comment_char = self.comment_char.upper()
            myline = self.comment_char + myline
            part = ""
            post_comment = ""
            # Break line in appropriate places
            # defined (in priority order) by the characters in
            # comment_split_characters
            res = self.split_line(myline,
                                  self.comment_split_characters,
                                  self.comment_char + " " * (5 + self.__indent))
        else:
            # This is a regular Fortran line

            # Strip leading spaces from line
            myline = line.lstrip()

            # Convert to upper or lower case
            # Here we need to make exception for anything within quotes.
            (myline, part, post_comment) = myline.partition("!")
            # Set space between line and post-comment
            if part:
                part = "  " + part
            # Replace all double quotes by single quotes
            myline = myline.replace('\"', '\'')
            # Downcase or upcase Fortran code, except for quotes
            splitline = myline.split('\'')
            myline = ""
            i = 0
            while i < len(splitline):
                if i % 2 == 1:
                    # This is a quote - check for escaped \'s
                    while splitline[i][len(splitline[i]) - 1] == '\\':
                        splitline[i] = splitline[i] + '\'' + splitline.pop(i + 1)
                else:
                    # Otherwise downcase/upcase
                    if FortranWriter.downcase:
                        splitline[i] = splitline[i].lower()
                    else:
                        splitline[i] = splitline[i].upper()
                i = i + 1

            myline = "\'".join(splitline).rstrip()

            # Check if line starts with dual keyword and adjust indent 
            if self.__keyword_list and re.search(self.keyword_pairs[\
                self.__keyword_list[len(self.__keyword_list) - 1]][0],
                                               myline.lower()):
                key = self.__keyword_list.pop()
                self.__indent = self.__indent - self.keyword_pairs[key][1]

            # Check for else and else if
            single_indent = 0
            for key in self.single_indents.keys():
                if re.search(key, myline.lower()):
                    self.__indent = self.__indent + self.single_indents[key]
                    single_indent = -self.single_indents[key]
                    break

            # Break line in appropriate places
            # defined (in priority order) by the characters in split_characters
            res = self.split_line(" " * (6 + self.__indent) + myline,
                                  self.split_characters,
                                  " " * 5 + self.line_cont_char + \
                                  " " * (self.__indent + 1))

            # Check if line starts with keyword and adjust indent for next line
            for key in self.keyword_pairs.keys():
                if re.search(key, myline.lower()):
                    self.__keyword_list.append(key)
                    self.__indent = self.__indent + self.keyword_pairs[key][1]
                    break

            # Correct back for else and else if
            if single_indent != None:
                self.__indent = self.__indent + single_indent
                single_indent = None

        # Write line(s) to file
        fsock.write("\n".join(res) + part + post_comment + "\n")

        return True

    def split_line(self, line, split_characters, line_start):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters, and start each new line with line_start."""

        res_lines = [line]

        while len(res_lines[-1]) > self.line_length:
            split_at = self.line_length
            for character in split_characters:
                index = res_lines[-1][(self.line_length - self.max_split): \
                                      self.line_length].rfind(character)
                if index >= 0:
                    split_at = self.line_length - self.max_split + index
                    break

            res_lines.append(line_start + \
                             res_lines[-1][split_at:])
            res_lines[-2] = res_lines[-2][:split_at]

        return res_lines

#===============================================================================
# HelasFortranModel
#===============================================================================
class HelasFortranModel(helas_objects.HelasModel):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""

    # Dictionaries used for automatic generation of Helas calls
    # Dictionaries from spin states to letters in Helas call
    mother_dict = {1: 'S', 2: 'O', -2: 'I', 3: 'V', 5: 'T'}
    self_dict = {1: 'H', 2: 'F', -2: 'F', 3: 'J', 5: 'U'}
    # Dictionaries used for sorting the letters in the Helas call
    sort_wf = {'O': 0, 'I': 1, 'S': 2, 'T': 3, 'V': 4}
    sort_amp = {'S': 1, 'V': 2, 'T': 0, 'O': 3, 'I': 4}

    def default_setup(self):
        """Set up special Helas calls (wavefunctions and amplitudes)
        that can not be done automatically by generate_helas_call"""

        super(HelasFortranModel, self).default_setup()

        # Add special fortran Helas calls, which are not automatically
        # generated

        # Gluon 4-vertex division tensor calls ggT for the FR sm and mssm

        key = ((3, 3, 5), 'A')

        call = lambda wf: \
               "CALL UVVAXX(W(1,%d),W(1,%d),%s,zero,zero,zero,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 5, 3), 'A')

        call = lambda wf: \
               "CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 5), 'A')

        call = lambda amp: \
               "CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),

                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

        # SM gluon 4-vertex components

        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[2].get('number'),
                wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[2].get('number'),
                amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('mothers')[0].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[0].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function."""

        val = super(HelasFortranModel, self).get_wavefunction_call(wavefunction)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(wavefunction.get('mothers')) > 3:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran wavefunctions not
                  implemented for > 3 mothers"""

        self.generate_helas_call(wavefunction)
        return super(HelasFortranModel, self).get_wavefunction_call(\
            wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude corresponding
        to the key. If the function doesn't exist, generate_helas_call
        is called to automatically create the function."""

        val = super(HelasFortranModel, self).get_amplitude_call(amplitude)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(amplitude.get('mothers')) > 4:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran amplitudes not
                  implemented for > 4 mothers"""

        self.generate_helas_call(amplitude)
        return super(HelasFortranModel, self).get_amplitude_call(amplitude)

    def generate_helas_call(self, argument):
        """Routine for automatic generation of Fortran Helas calls
        according to just the spin structure of the interaction.

        First the call string is generated, using a dictionary to go
        from the spin state of the calling wavefunction and its
        mothers, or the mothers of the amplitude, to letters.

        Then the call function is generated, as a lambda which fills
        the call string with the information of the calling
        wavefunction or amplitude. The call has different structure,
        depending on the spin of the wavefunction and the number of
        mothers (multiplicity of the vertex). The mother
        wavefunctions, when entering the call, must be sorted in the
        correct way - this is done by the sorted_mothers routine.

        Finally the call function is stored in the relevant
        dictionary, in order to be able to reuse the function the next
        time a wavefunction with the same Lorentz structure is needed.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"

        call = "CALL "

        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasFortranModel.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call))
            call = call + "(P(0,%d),"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "%s,NHEL(%d),"
            call = call + "%+d*IC(%d),W(1,%d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('number'))
        else:
            # String is FOVXXX, FIVXXX, JIOXXX etc.
            if isinstance(argument, helas_objects.HelasWavefunction):
                call = call + \
                       HelasFortranModel.self_dict[\
                argument.get_spin_state_number()]

            mother_letters = HelasFortranModel.sorted_letters(argument)

            # If Lorentz structure is given, by default add this
            # to call name
            addition = argument.get('lorentz')

            # Take care of special case: WWWW or WWVV calls
            if len(argument.get('lorentz')) > 3 and \
                   argument.get('lorentz')[:2] == "WW":
                if argument.get('lorentz')[:4] == "WWWW":
                    mother_letters = "WWWW"[:len(mother_letters)]
                if argument.get('lorentz')[:4] == "WWVV":
                    mother_letters = "W3W3"[:len(mother_letters)]
                addition = argument.get('lorentz')[4:]

            call = call + mother_letters
            call = call + addition

            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + 'C'

            if len(call) > 11:
                raise self.PhysicsObjectError, \
                      "Call to Helas routine %s should be maximum 6 chars" \
                      % call[5:]

            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call)) + '('
            # Wavefunctions
            call = call + "W(1,%d)," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"


            if isinstance(argument, helas_objects.HelasWavefunction):
                # Extra dummy coupling for 4-vector vertices
                if argument.get('lorentz') == 'WWVV':
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == 'WWWW':
                    # SM WWWW vertex
                    call = call + "0D0,"
                elif argument.get('spin') == 3 and \
                       [wf.get('spin') for wf in argument.get('mothers')] == \
                       [3, 3, 3]:
                    # All other 4-vector vertices (FR) - note that gggg
                    # has already been defined
                    call = call + "DUM0,"
                # Mass and width
                call = call + "%s,%s,"
                # New wavefunction
                call = call + "W(1,%d))"
            else:
                # Extra dummy coupling for 4-particle vertices
                # Need to replace later with the correct type
                if argument.get('lorentz') == 'WWVV':
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == 'WWWW':
                    # SM WWWW vertex
                    call = call + "0D0,"
                elif [wf.get('spin') for wf in argument.get('mothers')] == \
                       [3, 3, 3, 3]:
                    # Other 4-vector vertices (FR) - note that gggg
                    # has already been defined
                    call = call + "DUM0,"
                # Amplitude
                call = call + "AMP(%d))"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                if len(argument.get('mothers')) == 2:
                    call_function = lambda wf: call % \
                                    (wf.get('mothers')[0].\
                                     get('number'),
                                     wf.get('mothers')[1].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (wf.get('mothers')[0].\
                                     get('number'),
                                     wf.get('mothers')[1].\
                                     get('number'),
                                     wf.get('mothers')[2].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (amp.get('mothers')[0].\
                                     get('number'),
                                     amp.get('mothers')[1].\
                                     get('number'),
                                     amp.get('mothers')[2].\
                                     get('number'),

                                     amp.get('coupling'),
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (amp.get('mothers')[0].\
                                     get('number'),
                                     amp.get('mothers')[1].\
                                     get('number'),
                                     amp.get('mothers')[2].\
                                     get('number'),
                                     amp.get('mothers')[3].\
                                     get('number'),
                                     amp.get('coupling'),
                                     amp.get('number'))

        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

    # Static helper functions

    @staticmethod
    def sorted_letters(arg):
        """Gives a list of letters sorted according to
        the order of letters in the Fortran Helas calls"""

        if isinstance(arg, helas_objects.HelasWavefunction):
            return "".join(sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_wf[l2] - \
                          HelasFortranModel.sort_wf[l1]))

        if isinstance(arg, helas_objects.HelasAmplitude):
            return "".join(sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_amp[l2] - \
                          HelasFortranModel.sort_amp[l1]))

#===============================================================================
# Global helper methods
#===============================================================================

def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
    """Returns a nicely formatted string for the coefficients in JAMP lines"""

    total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

    if total_coeff == 1:
        if is_imaginary:
            return '+complex(0,1)*'
        else:
            return '+'
    elif total_coeff == -1:
        if is_imaginary:
            return '-complex(0,1)*'
        else:
            return '-'

    res_str = '%+i' % total_coeff.numerator

    if total_coeff.denominator != 1:
        # Check if total_coeff is an integer
        res_str = res_str + './%i.' % total_coeff.denominator

    if is_imaginary:
        res_str = res_str + '*complex(0,1)'

    return res_str + '*'


################################################################################
## helper function for universal file treatment
################################################################################
def format_path(path):
    """Format the path in local format taking in entry a unix format"""
    if path[0] != '/':
        return os.path.join(*path.split('/'))
    else:
        return os.path.sep + os.path.join(*path.split('/'))
def cp(path1, path2, log=True):
    """ simple cp taking linux or mix entry"""
    path1 = format_path(path1)
    path2 = format_path(path2)
    try:
        shutil.copy2(path1, path2)
    except IOError, why:
        if log:
            logger.warning(why)
        
    
def mv(path1, path2):
    """simple mv taking linux or mix format entry"""
    path1 = format_path(path1)
    path2 = format_path(path2)
    try:
        shutil.move(path1, path2)
    except:
        # An error can occur if the files exist at final destination
        if os.path.isfile(path2):
            os.remove(path2)
            shutil.move(path1, path2)
            return
        elif os.path.isdir(path2) and os.path.exists(
                                   os.path.join(path2, os.path.basename(path1))):      
            path2 = os.path.join(path2, os.path.basename(path1))
            os.remove(path2)
            shutil.move(path1, path2)
        else:
            raise
        
def ln(file_pos, starting_dir='.', name='', log=True):
    """a simple way to have a symbolic link whithout to have to change directory
    starting_point is the directory where to write the link
    file_pos is the file to link
    WARNING: not the linux convention
    """
    file_pos = format_path(file_pos)
    starting_dir = format_path(starting_dir)
    if not name:
        name = os.path.split(file_pos)[1]
        
    try:
        os.symlink(os.path.relpath(file_pos, starting_dir), \
                        os.path.join(starting_dir, name))
    except:
        if log:
            logger.warning('Could not link %s at position: %s' % (file_pos, \
                                                os.path.realpath(starting_dir)))



    
    
    
    
    
    
    
    


