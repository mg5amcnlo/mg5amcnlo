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

"""Unit test library for the export Python format routines"""

import copy
import fractions
import itertools
import logging
import math
import os
import re

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.rambo as rambo

from madgraph import MG5DIR

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.process_checks')

#===============================================================================
# Global helper function get_momenta
#===============================================================================
def get_momenta(process, model, energy = 1000.):
    """Get a point in phase space for the external states in the given
    process, with the CM energy given. The incoming particles are
    assumed to be oriented along the z axis, with particle 1 along the
    positive z axis."""

    if not (isinstance(process, base_objects.Process) and \
            isinstance(model, model_reader.ModelReader) and \
            isinstance(energy, float)):
        raise rambo.RAMBOError, "Not correct type for arguments to get_momenta"

    sorted_legs = sorted(process.get('legs'), lambda l1, l2:\
                         l1.get('number') - l2.get('number'))

    nincoming = len([leg for leg in sorted_legs if leg.get('state') == False])
    nfinal = len(sorted_legs) - nincoming

    # Find masses of particles
    mass_strings = [model.get_particle(l.get('id')).get('mass') \
                     for l in sorted_legs]
    mass = [abs(model.get('parameter_dict')[m]) for m in mass_strings]

    # Make sure energy is large enough for incoming and outgoing particles
    energy = max(energy, sum(mass[:nincoming]) + 200.,
                 sum(mass[nincoming:]) + 200.)

    e2 = energy**2
    m1 = mass[0]

    p = []

    masses = rambo.FortranList(nfinal)
    for i in range(nfinal):
        masses[i+1] = mass[nincoming + i]

    if nincoming == 1:
             
        # Momenta for the incoming particle
        p.append([m1, 0., 0., 0.])
        
        p_rambo, w_rambo = rambo.RAMBO(nfinal, m1, masses)

        # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
        for i in range(1, nfinal+1):
            momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                    p_rambo[(2,i)], p_rambo[(3,i)]]
            p.append(momi)

        return p, w_rambo

    if nincoming != 2:
        raise RAMBOError('Need 1 or 2 incoming particles')

    if nfinal == 1:
        energy = masses[0]
        
    m2 = mass[1]

    mom = math.sqrt((e2**2 - 2*e2*m1**2 + m1**4 - 2*e2*m2**2 - \
              2*m1**2*m2**2 + m2**4) / (4*e2))
    e1 = math.sqrt(mom**2+m1**2)
    e2 = math.sqrt(mom**2+m2**2)
    # Set momenta for incoming particles
    p.append([e1, 0., 0., mom])
    p.append([e2, 0., 0., -mom])

    if nfinal == 1:
        p.append([energy, 0., 0., 0.])
        return p, 1.

    p_rambo, w_rambo = rambo.RAMBO(nfinal, energy, masses)

    # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
    for i in range(1, nfinal+1):
        momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                p_rambo[(2,i)], p_rambo[(3,i)]]
        p.append(momi)

    return p, w_rambo

#===============================================================================
# check_processes
#===============================================================================
def check_processes(processes, param_card = None, quick = []):
    """Check processes by generating them with all possible orderings
    of particles (which means different diagram building and Helas
    calls), and comparing the resulting matrix element values."""

    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes
        model = multiprocess.get('model')
        isids = [[model.get_particle(id).get_anti_pdg_code() for id in \
                  leg.get('ids')] for leg in processes.get('legs') \
                  if not leg.get('state')]
        fsids = [leg.get('ids') for leg in processes.get('legs') \
                 if leg.get('state')]
        sorted_ids = []
        processes = base_objects.ProcessList()
        for is_prod in apply(itertools.product, isids):
            for fs_prod in apply(itertools.product, fsids):
                prod = sorted(list(is_prod) + list(fs_prod))
                if prod in sorted_ids:
                    continue
                is_prod = [model.get_particle(id).get_anti_pdg_code() \
                           for id in is_prod]
                sorted_ids.append(prod)
                processes.append(base_objects.Process({\
                    'legs': base_objects.LegList(\
                            [base_objects.Leg({'id': id, 'state':False}) for \
                             id in is_prod] + \
                            [base_objects.Leg({'id': id, 'state':True}) for \
                             id in fs_prod]),
                    'model':multiprocess.get('model'),
                    'id': multiprocess.get('id'),
                    'orders': multiprocess.get('orders'),
                    'required_s_channels': \
                                  multiprocess.get('required_s_channels'),
                    'forbidden_s_channels': \
                                  multiprocess.get('forbidden_s_channels'),
                    'forbidden_particles': \
                                  multiprocess.get('forbidden_particles'),
                    'is_decay_chain': \
                                  multiprocess.get('is_decay_chain'),
                    'overall_orders': \
                                  multiprocess.get('overall_orders')}))
    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise MadGraph5Error("processes is of non-supported format")

    if not processes:
        raise MadGraph5Error("No processes given")

    model = processes[0].get('model')

    # Read a param_card and calculate couplings
    full_model = model_reader.ModelReader(model)

    full_model.set_parameters_and_couplings(param_card)

    # Write the matrix element(s) in Python
    helas_writer = helas_call_writer.PythonUFOHelasCallWriter(model)

    # Keep track of tested processes, matrix elements, color and already
    # initiated Lorentz routines, to reuse as much as possible
    tested_processes = []
    used_lorentz = []
    matrix_elements = []
    list_colorize = []
    list_color_basis = []
    list_color_matrices = []        

    comparison_results = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Check if process is already checked
        ids = [l.get('id') for l in process.get('legs') if l.get('state')]
        ids.extend([model.get_particle(l.get('id')).get_anti_pdg_code() \
                    for l in process.get('legs') if not l.get('state')])
        ids = sorted(ids) + [process.get('id')]

        if ids in tested_processes:
            # We have already tested a version of this process, continue
            continue

        # Add this process to tested_processes
        tested_processes.append(ids)
        # Add also antiprocess, since these are identical
        anti_ids = sorted([model.get_particle(id).get_anti_pdg_code() \
                           for id in ids[:-1]]) + [process.get('id')]
        tested_processes.append(anti_ids)

        # Generate phase space point to use
        p, w_rambo = get_momenta(process, full_model)
        # Initiate the value array
        values = []
        for i, leg in enumerate(process.get('legs')):
            leg.set('number', i+1)

        logger.info("Checking %s" % \
                    process.nice_string().replace('Process', 'process'))

        process_matrix_elements = []

        # For quick checks, only test twp permutations with leg "1" in
        # each position
        if quick:
            leg_positions = [[] for leg in process.get('legs')]
            quick = range(1,len(process.get('legs')) + 1)
        # Now, generate all possible permutations of the legs
        for legs in itertools.permutations(process.get('legs')):

            if quick:
                found_leg = True
                for num in quick:
                    # Only test one permutation for each position of the
                    # specified legs
                    leg_position = legs.index([l for l in legs if \
                                               l.get('number') == num][0])

                    if not leg_position in leg_positions[num-1]:
                        found_leg = False
                        leg_positions[num-1].append(leg_position)
                        
                if found_leg:
                    continue
                
            legs = base_objects.LegList(legs)

            if legs != process.get('legs'):
                logger.info("Testing permutation: %s" % \
                            ([l.get('number') for l in legs]))

            # Generate a process with these legs
            newproc = copy.copy(process)
            newproc.set('legs', legs)
            # Generate the amplitude for this process
            amplitude = diagram_generation.Amplitude(newproc)
            if not amplitude.get('diagrams'):
                # This process has no diagrams; go to next process
                logging.info("No diagrams for %s" % \
                             newproc.nice_string().replace('Process', 'process'))
                break

            # Generate the HelasMatrixElement for the process
            matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                              gen_color=False)

            if matrix_element in process_matrix_elements:
                # Exactly the same matrix element has been tested
                # for other permutation of same process
                index = process_matrix_elements.index(matrix_element)
                logger.info("Skipping permutation, " + \
                            "identical matrix element as previous")
                continue
            elif matrix_element in matrix_elements:
                # Exactly the same matrix element has been tested
                logger.info("Skipping %s, " % process.nice_string() + \
                            "identical matrix element already tested" \
                            )
                values = []
                break

            process_matrix_elements.append(matrix_element)

            # Create an empty color basis, and the list of raw
            # colorize objects (before simplification) associated
            # with amplitude
            col_basis = color_amp.ColorBasis()
            new_amp = matrix_element.get_base_amplitude()
            matrix_element.set('base_amplitude', new_amp)
            colorize_obj = col_basis.create_color_dict_list(new_amp)

            try:
                # If the color configuration of the ME has
                # already been considered before, recycle
                # the information
                col_index = list_colorize.index(colorize_obj)

            except ValueError:
                # If not, create color basis and color
                # matrix accordingly
                list_colorize.append(colorize_obj)
                col_basis.build()
                list_color_basis.append(col_basis)
                col_matrix = color_amp.ColorMatrix(col_basis)
                list_color_matrices.append(col_matrix)
                col_index = -1

            # Set the color for the matrix element
            matrix_element.set('color_basis', list_color_basis[col_index])
            matrix_element.set('color_matrix',
                               list_color_matrices[col_index])

            # Create the needed aloha routines
            me_used_lorentz = set(matrix_element.get_used_lorentz())
            me_used_lorentz = [lorentz for lorentz in me_used_lorentz \
                               if lorentz not in used_lorentz]

            aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
            aloha_model.compute_subset(me_used_lorentz)

            # Write out the routines in Python
            aloha_routines = []
            for routine in aloha_model.values():
                aloha_routines.append(routine.write(output_dir = None,
                                                    language = 'Python').\
                      replace('import wavefunctions',
                              'import aloha.template_files.wavefunctions' +\
                              ' as wavefunctions'))

            # Define the routines to be available globally
            for routine in aloha_routines:
                exec(routine, globals())

            # Add the defined Aloha routines to used_lorentz
            used_lorentz.extend(me_used_lorentz)

            # Export the matrix element to Python calls
            exporter = export_python.ProcessExporterPython(matrix_element,
                                                           helas_writer)
            matrix_methods = exporter.get_python_matrix_methods()

            # Define the routines (locally is enough)
            exec(matrix_methods[newproc.shell_string()])

            # Evaluate the matrix element for the momenta p
            values.append(eval("Matrix().smatrix(p, full_model)"))

            # Check if we failed badly (1% is already bad) - in that
            # case done for this process
            if abs(max(values)) + abs(min(values)) > 0 and \
                   2 * abs(max(values) - min(values)) / \
                   (abs(max(values)) + abs(min(values))) > 0.01:
                break


        # Check if process was interrupted
        if not values:
            continue

        # Add new matrix element to list och checked matrix elements
        matrix_elements.extend(process_matrix_elements)

        # Done with this process. Collect values, and store
        # process and momenta
        diff = 0
        if abs(max(values)) + abs(min(values)) > 0:
            diff = 2* abs(max(values) - min(values)) / \
                   (abs(max(values)) + abs(min(values)))

        passed = diff < 1.e-8

        comparison_results.append({"process": process,
                                   "momenta": p,
                                   "values": values,
                                   "difference": diff,
                                   "passed": passed})

    return comparison_results

def output_comparisons(comparison_results):
    """Present the results of a comparison in a nice list format"""

    proc_col_size = 17

    for proc in comparison_results:
        if len(proc['process'].base_string()) + 1 > proc_col_size:
            proc_col_size = len(proc['process'].base_string()) + 1

    col_size = 17

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length("Process", proc_col_size) + \
              fixed_string_length("Min element", col_size) + \
              fixed_string_length("Max element", col_size) + \
              fixed_string_length("Relative diff.", col_size) + \
              "Result"

    for result in comparison_results:
        proc = result['process'].base_string()
        values = result['values']
        
        if len(values) <= 1:
            res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   "    * No permutations, process not checked *" 
            no_check_proc += 1
            no_check_proc_list.append(result['process'].nice_string())
            continue

        passed = result['passed']

        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % min(values), col_size) + \
                   fixed_string_length("%1.10e" % max(values), col_size) + \
                   fixed_string_length("%1.10e" % result['difference'],
                                       col_size)
        if passed:
            pass_proc += 1
            res_str += "Passed"
        else:
            fail_proc += 1
            failed_proc_list.append(result['process'].nice_string())
            res_str += "Failed"

    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc != 0:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)

    return res_str


def fixed_string_length(mystr, length):
    """Helper function to fix the length of a string by cutting it 
    or adding extra space."""

    if len(mystr) > length:
        return mystr[0:length]
    else:
        return mystr + " " * (length - len(mystr))

