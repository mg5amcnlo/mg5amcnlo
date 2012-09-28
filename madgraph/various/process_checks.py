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
"""Several different checks for processes (and hence models):
permutation tests, gauge invariance tests, lorentz invariance
tests. Also class for evaluation of Python matrix elements,
MatrixElementEvaluator."""

from __future__ import division

import array
import copy
import fractions
import itertools
import logging
import math
import os
import re

import aloha
import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.rambo as rambo
import madgraph.various.misc as misc


from madgraph import MG5DIR, InvalidCmd

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx, txxxxx, irxxxx, orxxxx

ADDED_GLOBAL = []

def clean_added_globals(to_clean):
    for value in list(to_clean):
        del globals()[value]
        to_clean.remove(value)

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.process_checks')

#===============================================================================
# Helper class MatrixElementEvaluator
#===============================================================================
class MatrixElementEvaluator(object):
    """Class taking care of matrix element evaluation, storing
    relevant quantities for speedup."""

    def __init__(self, model, param_card = None,
                 auth_skipping = False, reuse = True, cmass_scheme = False):
        """Initialize object with stored_quantities, helas_writer,
        model, etc.
        auth_skipping = True means that any identical matrix element will be
                        evaluated only once
        reuse = True means that the matrix element corresponding to a
                given process can be reused (turn off if you are using
                different models for the same process)"""
 
        # Writer for the Python matrix elements
        self.helas_writer = helas_call_writers.PythonUFOHelasCallWriter(model)
    
        # Read a param_card and calculate couplings
        self.full_model = model_reader.ModelReader(model)
        self.full_model.set_parameters_and_couplings(param_card)

        self.auth_skipping = auth_skipping
        self.reuse = reuse
        self.store_aloha = []
        self.stored_quantities = {}
        
    #===============================================================================
    # Helper function evaluate_matrix_element
    #===============================================================================
    def evaluate_matrix_element(self, matrix_element, p=None, full_model=None, 
                                gauge_check=False, auth_skipping=None, output='m2'):
        """Calculate the matrix element and evaluate it for a phase space point
           output is either m2, amp, jamp
        """
        if full_model:
            self.full_model = full_model

        process = matrix_element.get('processes')[0]
        model = process.get('model')

        if "matrix_elements" not in self.stored_quantities:
            self.stored_quantities['matrix_elements'] = []
            matrix_methods = {}

        if self.reuse and "Matrix_%s" % process.shell_string() in globals() and p:
            # Evaluate the matrix element for the momenta p
            matrix = eval("Matrix_%s()" % process.shell_string())
            me_value = matrix.smatrix(p, self.full_model)
            if output == "m2":
                return matrix.smatrix(p, self.full_model), matrix.amp2
            else:
                m2 = matrix.smatrix(p, self.full_model)
            return {'m2': m2, output:getattr(matrix, output)}

        if (auth_skipping or self.auth_skipping) and matrix_element in \
               self.stored_quantities['matrix_elements']:
            # Exactly the same matrix element has been tested
            logger.info("Skipping %s, " % process.nice_string() + \
                        "identical matrix element already tested" \
                        )
            return None

        self.stored_quantities['matrix_elements'].append(matrix_element)

        # Create an empty color basis, and the list of raw
        # colorize objects (before simplification) associated
        # with amplitude
        if "list_colorize" not in self.stored_quantities:
            self.stored_quantities["list_colorize"] = []
        if "list_color_basis" not in self.stored_quantities:
            self.stored_quantities["list_color_basis"] = []
        if "list_color_matrices" not in self.stored_quantities:
            self.stored_quantities["list_color_matrices"] = []        

        col_basis = color_amp.ColorBasis()
        new_amp = matrix_element.get_base_amplitude()
        matrix_element.set('base_amplitude', new_amp)
        colorize_obj = col_basis.create_color_dict_list(new_amp)

        try:
            # If the color configuration of the ME has
            # already been considered before, recycle
            # the information
            col_index = self.stored_quantities["list_colorize"].index(colorize_obj)
        except ValueError:
            # If not, create color basis and color
            # matrix accordingly
            self.stored_quantities['list_colorize'].append(colorize_obj)
            col_basis.build()
            self.stored_quantities['list_color_basis'].append(col_basis)
            col_matrix = color_amp.ColorMatrix(col_basis)
            self.stored_quantities['list_color_matrices'].append(col_matrix)
            col_index = -1

        # Set the color for the matrix element
        matrix_element.set('color_basis',
                           self.stored_quantities['list_color_basis'][col_index])
        matrix_element.set('color_matrix',
                           self.stored_quantities['list_color_matrices'][col_index])

        # Create the needed aloha routines
        if "used_lorentz" not in self.stored_quantities:
            self.stored_quantities["used_lorentz"] = []

        me_used_lorentz = set(matrix_element.get_used_lorentz())
        me_used_lorentz = [lorentz for lorentz in me_used_lorentz \
                               if lorentz not in self.store_aloha]

        aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
        aloha_model.add_Lorentz_object(model.get('lorentz'))
        aloha_model.compute_subset(me_used_lorentz)

        # Write out the routines in Python
        aloha_routines = []
        for routine in aloha_model.values():
            aloha_routines.append(routine.write(output_dir = None, 
                                                mode='mg5',
                                                language = 'Python'))
        for routine in aloha_model.external_routines:
            aloha_routines.append(
                     open(aloha_model.locate_external(routine, 'Python')).read())


        # Define the routines to be available globally
        previous_globals = list(globals().keys())
        for routine in aloha_routines:
            exec(routine, globals())
        for key in globals().keys():
            if key not in previous_globals:
                ADDED_GLOBAL.append(key)

        # Add the defined Aloha routines to used_lorentz
        self.store_aloha.extend(me_used_lorentz)

        # Export the matrix element to Python calls
        exporter = export_python.ProcessExporterPython(matrix_element,
                                                       self.helas_writer)
        try:
            matrix_methods = exporter.get_python_matrix_methods(\
                gauge_check=gauge_check)
        except helas_call_writers.HelasWriterError, error:
            logger.info(error)
            return None

        if self.reuse:
            # Define the routines (globally)
            exec(matrix_methods[process.shell_string()], globals())
            ADDED_GLOBAL.append('Matrix_%s'  % process.shell_string())
        else:
            # Define the routines (locally is enough)
            exec(matrix_methods[process.shell_string()])

        # Generate phase space point to use
        if not p:
            p, w_rambo = self.get_momenta(process)

        # Evaluate the matrix element for the momenta p
        exec("data = Matrix_%s()" % process.shell_string())
        if output == "m2": 
            return data.smatrix(p, self.full_model), data.amp2
        else:
            m2 = data.smatrix(p, self.full_model)
            return {'m2': m2, output:getattr(data, output)}
    
    #===============================================================================
    # Helper function get_momenta
    #===============================================================================
    def get_momenta(self, process, energy = 1000.):
        """Get a point in phase space for the external states in the given
        process, with the CM energy given. The incoming particles are
        assumed to be oriented along the z axis, with particle 1 along the
        positive z axis."""

        if not (isinstance(process, base_objects.Process) and \
                isinstance(energy, float)):
            raise rambo.RAMBOError, "Not correct type for arguments to get_momenta"

        sorted_legs = sorted(process.get('legs'), lambda l1, l2:\
                             l1.get('number') - l2.get('number'))

        nincoming = len([leg for leg in sorted_legs if leg.get('state') == False])
        nfinal = len(sorted_legs) - nincoming

        # Find masses of particles
        mass_strings = [self.full_model.get_particle(l.get('id')).get('mass') \
                         for l in sorted_legs]
        mass = [self.full_model.get('parameter_dict')[m] for m in mass_strings]
        mass = [m.real for m in mass]
        #mass = [math.sqrt(m.real) for m in mass]

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
            raise rambo.RAMBOError('Need 1 or 2 incoming particles')

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
# Global helper function run_multiprocs
#===============================================================================
def run_multiprocs_no_crossings(function, multiprocess, stored_quantities,
                                opt=None):
    """A wrapper function for running an iteration of a function over
    a multiprocess, without having to first create a process list
    (which makes a big difference for very large multiprocesses.
    stored_quantities is a dictionary for any quantities that we want
    to reuse between runs."""
                   
    model = multiprocess.get('model')
    isids = [leg.get('ids') for leg in multiprocess.get('legs') \
              if not leg.get('state')]
    fsids = [leg.get('ids') for leg in multiprocess.get('legs') \
             if leg.get('state')]
    # Create dictionary between isids and antiids, to speed up lookup
    id_anti_id_dict = {}
    for id in set(tuple(sum(isids+fsids, []))):
        id_anti_id_dict[id] = model.get_particle(id).get_anti_pdg_code()
        id_anti_id_dict[model.get_particle(id).get_anti_pdg_code()] = id        

    sorted_ids = []
    results = []
    for is_prod in apply(itertools.product, isids):
        for fs_prod in apply(itertools.product, fsids):

            # Check if we have already checked the process
            if check_already_checked(is_prod, fs_prod, sorted_ids,
                                     multiprocess, model, id_anti_id_dict):
                continue

            # Generate process based on the selected ids
            process = base_objects.Process({\
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
                              multiprocess.get('overall_orders')})
            if opt is not None:
                if isinstance(opt, dict):
                    try:
                        value = opt[process.base_string()]
                    except:
                        continue
                    result = function(process, stored_quantities, value)
                else:
                    result = function(process, stored_quantities, opt)
            else:
                result = function(process, stored_quantities)
                        
            if result:
                results.append(result)
                
    return results

#===============================================================================
# Helper function check_already_checked
#===============================================================================
def check_already_checked(is_ids, fs_ids, sorted_ids, process, model,
                          id_anti_id_dict = {}):
    """Check if process already checked, if so return True, otherwise add
    process and antiprocess to sorted_ids."""

    # Check if process is already checked
    if id_anti_id_dict:
        is_ids = [id_anti_id_dict[id] for id in \
                  is_ids]
    else:
        is_ids = [model.get_particle(id).get_anti_pdg_code() for id in \
                  is_ids]        

    ids = array.array('i', sorted(is_ids + list(fs_ids)) + \
                      [process.get('id')])

    if ids in sorted_ids:
        # We have already checked (a crossing of) this process
        return True

    # Add this process to tested_processes
    sorted_ids.append(ids)

    # Skip adding antiprocess below, since might be relevant too
    return False

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

        # Initialize matrix element evaluation
        evaluator = MatrixElementEvaluator(model,
                                           auth_skipping = True, reuse = False)
        results = run_multiprocs_no_crossings(check_process,
                                              multiprocess,
                                              evaluator,
                                              quick)

        if "used_lorentz" not in evaluator.stored_quantities:
            evaluator.stored_quantities["used_lorentz"] = []
        return results, evaluator.stored_quantities["used_lorentz"]

    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    if not processes:
        raise InvalidCmd("No processes given")

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = True, reuse = False)

    # Keep track of tested processes, matrix elements, color and already
    # initiated Lorentz routines, to reuse as much as possible
    sorted_ids = []
    comparison_results = []

    # Check process by process
    for process in processes:
        
        # Check if we already checked process        
        if check_already_checked([l.get('id') for l in process.get('legs') if \
                                  not l.get('state')],
                                 [l.get('id') for l in process.get('legs') if \
                                  l.get('state')],
                                 sorted_ids, process, model):
            continue
        # Get process result
        res = check_process(process, evaluator, quick)
        if res:
            comparison_results.append(res)

    if "used_lorentz" not in evaluator.stored_quantities:
        evaluator.stored_quantities["used_lorentz"] = []
    return comparison_results, evaluator.stored_quantities["used_lorentz"]


def check_process(process, evaluator, quick):
    """Check the helas calls for a process by generating the process
    using all different permutations of the process legs (or, if
    quick, use a subset of permutations), and check that the matrix
    element is invariant under this."""

    model = process.get('model')

    # Ensure that leg numbers are set
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

    values = []

    # Now, generate all possible permutations of the legs
    for legs in itertools.permutations(process.get('legs')):

        order = [l.get('number') for l in legs]

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

        if order != range(1,len(legs) + 1):
            logger.info("Testing permutation: %s" % \
                        order)

        # Generate a process with these legs
        newproc = copy.copy(process)
        newproc.set('legs', legs)

        # Generate the amplitude for this process
        try:
            amplitude = diagram_generation.Amplitude(newproc)
        except InvalidCmd:
            result=False
        else:
            result = amplitude.get('diagrams')

        if not result:
            # This process has no diagrams; go to next process
            logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
            break

        if order == range(1,len(legs) + 1):
            # Generate phase space point to use
            p, w_rambo = evaluator.get_momenta(process)

        # Generate the HelasMatrixElement for the process
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                          gen_color=False)

        if matrix_element in process_matrix_elements:
            # Exactly the same matrix element has been tested
            # for other permutation of same process
            continue

        process_matrix_elements.append(matrix_element)

        res = evaluator.evaluate_matrix_element(matrix_element, p = p)
        if res == None:
            break

        values.append(res[0])

        # Check if we failed badly (1% is already bad) - in that
        # case done for this process
        if abs(max(values)) + abs(min(values)) > 0 and \
               2 * abs(max(values) - min(values)) / \
               (abs(max(values)) + abs(min(values))) > 0.01:
            break
    
    # Check if process was interrupted
    if not values:
        return None

    # Done with this process. Collect values, and store
    # process and momenta
    diff = 0
    if abs(max(values)) + abs(min(values)) > 0:
        diff = 2* abs(max(values) - min(values)) / \
               (abs(max(values)) + abs(min(values)))

    passed = diff < 1.e-8

    return {"process": process,
            "momenta": p,
            "values": values,
            "difference": diff,
            "passed": passed}


def output_comparisons(comparison_results):
    """Present the results of a comparison in a nice list format
       mode short: return the number of fail process
    """
    
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
    

#===============================================================================
# check_gauge
#===============================================================================
def check_gauge(processes, param_card=None, cmass_scheme=False):
    """Check gauge invariance of the processes by using the BRS check.
    For one of the massless external bosons (e.g. gluon or photon), 
    replace the polarization vector (epsilon_mu) with its momentum (p_mu)
    """

    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes

        model = multiprocess.get('model')
        
        # Initialize matrix element evaluation
        evaluator = MatrixElementEvaluator(model, param_card, 
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping= True, reuse= False,)
        
        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        return run_multiprocs_no_crossings(check_gauge_process,
                                           multiprocess,
                                           evaluator)

    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = True, reuse = False)

    comparison_results = []
    comparison_explicit_flip = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Check if we already checked process
        #if check_already_checked([l.get('id') for l in process.get('legs') if \
        #                          not l.get('state')],
        ##                         [l.get('id') for l in process.get('legs') if \
        #                          l.get('state')],
        #                         sorted_ids, process, model):
        #    continue
        
        # Get process result
        result = check_gauge_process(process, evaluator)
        if result:
            comparison_results.append(result)
        
        
            
    return comparison_results


def check_gauge_process(process, evaluator):
    """Check gauge invariance for the process, unless it is already done."""

    model = process.get('model')

    # Check that there are massless vector bosons in the process
    found_gauge = False
    for i, leg in enumerate(process.get('legs')):
        part = model.get_particle(leg.get('id'))
        if part.get('spin') == 3 and part.get('mass').lower() == 'zero':
            found_gauge = True
            break

    if not found_gauge:
        # This process can't be checked
        return None

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking gauge %s" % \
                process.nice_string().replace('Process', 'process'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        amplitude = diagram_generation.Amplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None    
    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None

    # Generate the HelasMatrixElement for the process
    matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)

    brsvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = True,
                                                 output='jamp')


    matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)    
    mvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = False,
                                               output='jamp')
    
    if mvalue and mvalue['m2']:
        return {'process':process.base_string(),'value':mvalue,'brs':brsvalue}

def output_gauge(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format"""

    proc_col_size = 17
    for one_comp in comparison_results:
        proc = one_comp['process']
        mvalue = one_comp['value']
        brsvalue = one_comp['brs']
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    col_size = 17

    pass_proc = 0
    fail_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length("Process", proc_col_size) + \
              fixed_string_length("matrix", col_size) + \
              fixed_string_length("BRS", col_size) + \
              fixed_string_length("ratio", col_size) + \
              "Result"

    for  one_comp in comparison_results:
        proc = one_comp['process']
        mvalue = one_comp['value']
        brsvalue = one_comp['brs']
        ratio = (brsvalue['m2']/mvalue['m2'])
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                    fixed_string_length("%1.10e" % mvalue['m2'], col_size)+ \
                    fixed_string_length("%1.10e" % brsvalue['m2'], col_size)+ \
                    fixed_string_length("%1.10e" % ratio, col_size)
         
        if ratio > 1e-12:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"
        else:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"

        #check all the JAMP
        # loop over jamp
        for k in range(len(mvalue['jamp'][0])):
            m_sum = 0
            brs_sum = 0
            # loop over helicity
            for j in range(len(mvalue['jamp'])):
                #values for the different lorentz boost
                m_sum += abs(mvalue['jamp'][j][k])**2
                brs_sum += abs(brsvalue['jamp'][j][k])**2                                            
                    
            # Compare the different helicity  
            if not m_sum:
                continue
            ratio = brs_sum / m_sum

            tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
                   fixed_string_length("%1.10e" % m_sum, col_size) + \
                   fixed_string_length("%1.10e" % brs_sum, col_size) + \
                   fixed_string_length("%1.10e" % ratio, col_size)        
                   
            if ratio > 1e-15:
                if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                    fail_proc += 1
                    pass_proc -= 1
                    failed_proc_list.append(proc)
                res_str += tmp_str + "Failed"
            elif not proc_succeed:
                 res_str += tmp_str + "Passed"


    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

    if output=='text':
        return res_str
    else:
        return fail_proc
#===============================================================================
# check_lorentz
#===============================================================================
def check_lorentz(processes, param_card = None, cmass_scheme=False):
    """ Check if the square matrix element (sum over helicity) is lorentz 
        invariant by boosting the momenta with different value."""
    
    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes

        model = multiprocess.get('model')
        
        # Initialize matrix element evaluation
        evaluator = MatrixElementEvaluator(model, param_card, 
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping = False, reuse = True)

        if not cmass_scheme:
            # Set all widths to zero for lorentz check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[\
                                                     particle.get('width')] = 0.
        return run_multiprocs_no_crossings(check_lorentz_process,
                                           multiprocess,
                                           evaluator)
    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = False, reuse = True)

    comparison_results = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Check if we already checked process
        #if check_already_checked([l.get('id') for l in process.get('legs') if \
        #                          not l.get('state')],
        #                         [l.get('id') for l in process.get('legs') if \
        #                          l.get('state')],
        #                         sorted_ids, process, model):
        #    continue
        
        # Get process result
        result = check_lorentz_process(process, evaluator)
        if result:
            comparison_results.append(result)
            
    return comparison_results


def check_lorentz_process(process, evaluator):
    """Check gauge invariance for the process, unless it is already done."""

    amp_results = []
    model = process.get('model')

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking lorentz %s" % \
                process.nice_string().replace('Process', 'process'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        amplitude = diagram_generation.Amplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None

    # Generate phase space point to use
    p, w_rambo = evaluator.get_momenta(process)

    # Generate the HelasMatrixElement for the process
    matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = True)

    data = evaluator.evaluate_matrix_element(matrix_element, p=p, output='jamp',
                                             auth_skipping = True)

    if data and data['m2']:
        results = [data]
    else:
        return  {'process':process.base_string(), 'results':'pass'}
    
    for boost in range(1,4):
        boost_p = boost_momenta(p, boost)
        results.append(evaluator.evaluate_matrix_element(matrix_element,
                                                         p=boost_p,
                                                         output='jamp'))
        
        
    return {'process': process.base_string(), 'results': results}


#===============================================================================
# check_gauge
#===============================================================================
def check_unitary_feynman(processes_unit, processes_feynm, param_card=None, cmass_scheme=False):
    """Check gauge invariance of the processes by flipping
       the gauge of the model
    """

    if isinstance(processes_unit, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess_unit = processes_unit
        resutls = []
        model = multiprocess_unit.get('model')
        
        # Initialize matrix element evaluation
        aloha.unitary_gauge = True
        evaluator = MatrixElementEvaluator(model, param_card, 
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping= True, reuse= False,)
                
        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_u = run_multiprocs_no_crossings(get_value,
                                           multiprocess_unit,
                                           evaluator)
        
        clean_added_globals(ADDED_GLOBAL)
        
        momentum = {}
        for data in output_u:
            momentum[data['process']] = data['p']
        
        multiprocess_feynm = processes_feynm
        model = multiprocess_feynm.get('model')
        
        # Initialize matrix element evaluation
        aloha.unitary_gauge = False
        evaluator = MatrixElementEvaluator(model, param_card, 
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping= True, reuse= False,)
                
        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_f = run_multiprocs_no_crossings(get_value,
                                           multiprocess_feynm,
                                           evaluator, momentum)  
        
        output = []
        for data in output_f:
            local_dico = {}
            local_dico['process'] = data['process']
            local_dico['value_feynm'] = data['value']
            local_dico['value_unit'] = [d['value'] for d in output_u 
                                      if d['process'] == data['process']][0]
            output.append(local_dico)
        return output
        
#    elif isinstance(processes, base_objects.Process):
#        processes = base_objects.ProcessList([processes])
#    elif isinstance(processes, base_objects.ProcessList):
#        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert False
    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = True, reuse = False)

    comparison_results = []
    comparison_explicit_flip = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Get process result
        result = check_gauge_process(process, evaluator)
        if result:
            comparison_results.append(result)
        
        
            
    return comparison_results


def get_value(process, evaluator, p=None):
    """Return the value/momentum for a phase space point"""

    model = process.get('model')

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)


    logger.info("Checking gauge %s" % \
                process.nice_string().replace('Process', 'process'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        amplitude = diagram_generation.Amplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None    
    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    
    if not p:
        # Generate phase space point to use
        p, w_rambo = evaluator.get_momenta(process)
        
    # Generate the HelasMatrixElement for the process
    matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)    
    mvalue = evaluator.evaluate_matrix_element(matrix_element, p=p,
                                                                  output='jamp')
    
    if mvalue and mvalue['m2']:
        return {'process':process.base_string(),'value':mvalue,'p':p}



def boost_momenta(p, boost_direction=1, beta=0.5):
    """boost the set momenta in the 'boost direction' by the 'beta' 
       factor"""
    boost_p = []    
    gamma = 1/ math.sqrt(1 - beta**2)
    for imp in p:
        bosst_p = imp[boost_direction]
        E, px, py, pz = imp
        boost_imp = []
        # Energy:
        boost_imp.append(gamma * E - gamma * beta * bosst_p)
        # PX
        if boost_direction == 1:
            boost_imp.append(-gamma * beta * E + gamma * px)
        else: 
            boost_imp.append(px)
        # PY
        if boost_direction == 2:
            boost_imp.append(-gamma * beta * E + gamma * py)
        else: 
            boost_imp.append(py)    
        # PZ
        if boost_direction == 3:
            boost_imp.append(-gamma * beta * E + gamma * pz)
        else: 
            boost_imp.append(pz) 
        #Add the momenta to the list
        boost_p.append(boost_imp)                   
            
    return boost_p

def output_lorentz_inv(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format
        if output='fail' return the number of failed process -- for test-- 
    """

    proc_col_size = 17
    for proc, values in comparison_results:
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

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

    for one_comp in comparison_results:
        proc = one_comp['process']
        data = one_comp['results']
        
        if data == 'pass':
            no_check_proc += 1
            no_check_proc_list.append(proc)
            continue
        
        values = [data[i]['m2'] for i in range(len(data))]
        
        min_val = min(values)
        max_val = max(values)
        diff = (max_val - min_val) / max_val 
        
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % min_val, col_size) + \
                   fixed_string_length("%1.10e" % max_val, col_size) + \
                   fixed_string_length("%1.10e" % diff, col_size)
                   
        if diff < 1e-8:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"
        else:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"

        #check all the JAMP
        # loop over jamp
        for k in range(len(data[0]['jamp'][0])):
            sum = [0] * len(data)
            # loop over helicity
            for j in range(len(data[0]['jamp'])):
                #values for the different lorentz boost
                values = [abs(data[i]['jamp'][j][k])**2 for i in range(len(data))]
                sum = [sum[i] + values[i] for i in range(len(values))]

            # Compare the different lorentz boost  
            min_val = min(sum)
            max_val = max(sum)
            if not max_val:
                continue
            diff = (max_val - min_val) / max_val 
        
            tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
                       fixed_string_length("%1.10e" % min_val, col_size) + \
                       fixed_string_length("%1.10e" % max_val, col_size) + \
                       fixed_string_length("%1.10e" % diff, col_size)
                   
            if diff > 1e-10:
                if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                    fail_proc += 1
                    pass_proc -= 1
                    failed_proc_list.append(proc)
                res_str += tmp_str + "Failed"
            elif not proc_succeed:
                 res_str += tmp_str + "Passed" 
            
            
        
    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)
    
    if output == 'text':
        return res_str        
    else: 
        return fail_proc

def output_unitary_feynman(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format
        if output='fail' return the number of failed process -- for test-- 
    """
    
    proc_col_size = 17
    for data in comparison_results:
        proc = data['process']
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    col_size = 17

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length("Process", proc_col_size) + \
              fixed_string_length("Unitary", col_size) + \
              fixed_string_length("Feynman", col_size) + \
              fixed_string_length("Relative diff.", col_size) + \
              "Result"

    for one_comp in comparison_results:
        proc = one_comp['process']
        data = [one_comp['value_unit'], one_comp['value_feynm']]
        
        
        if data[0] == 'pass':
            no_check_proc += 1
            no_check_proc_list.append(proc)
            continue
        
        values = [data[i]['m2'] for i in range(len(data))]
        
        min_val = min(values)
        max_val = max(values)
        diff = (max_val - min_val) / max_val 
        
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % values[0], col_size) + \
                   fixed_string_length("%1.10e" % values[1], col_size) + \
                   fixed_string_length("%1.10e" % diff, col_size)
                   
        if diff < 1e-8:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"
        else:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"

        #check all the JAMP
        # loop over jamp
        for k in range(len(data[0]['jamp'][0])):
            sum = [0, 0]
            # loop over helicity
            for j in range(len(data[0]['jamp'])):
                #values for the different lorentz boost
                values = [abs(data[i]['jamp'][j][k])**2 for i in range(len(data))]
                sum = [sum[i] + values[i] for i in range(len(values))]

            # Compare the different lorentz boost  
            min_val = min(sum)
            max_val = max(sum)
            if not max_val:
                continue
            diff = (max_val - min_val) / max_val 
        
            tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
                       fixed_string_length("%1.10e" % sum[0], col_size) + \
                       fixed_string_length("%1.10e" % sum[1], col_size) + \
                       fixed_string_length("%1.10e" % diff, col_size)
                   
            if diff > 1e-10:
                if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                    fail_proc += 1
                    pass_proc -= 1
                    failed_proc_list.append(proc)
                res_str += tmp_str + "Failed"
            elif not proc_succeed:
                 res_str += tmp_str + "Passed" 
            
            
        
    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)
    
    
    print res_str
    if output == 'text':
        return res_str        
    else: 
        return fail_proc




