################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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

"""Module for calculation of symmetries between diagrams, by
evaluating amp2 values for permutations of modementa."""

from __future__ import division

import array
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
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.process_checks as process_checks

from madgraph import MG5DIR

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.diagram_symmetry')

#===============================================================================
# find_symmetry
#===============================================================================

def find_symmetry(matrix_element):
    """Find symmetries between amplitudes by comparing the squared
    amplitudes for all permutations of identical particles.
    
    Return list of positive number corresponding to number of
    symmetric diagrams and negative numbers corresponding to the
    equivalent diagram (for e+e->3a, get [6, -1, -1, -1, -1, -1]),
    list of the corresponding permutations needed, and list of all
    permutations of identical particles.
    For amp2s which are 0 (e.g. multiparticle vertices), return 0 in
    symmetry list."""

    if isinstance(matrix_element, group_subprocs.SubProcessGroup):
        return find_symmetry_subproc_group(matrix_element)

    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

    # Prepare the symmetry vector with non-used amp2s (due to
    # multiparticle vertices)
    symmetry = []
    for diag in matrix_element.get('diagrams'):
        if max(diag.get_vertex_leg_numbers()) > 3:
            # Ignore any diagrams with 4-particle vertices
            symmetry.append(0)
        else:
            symmetry.append(1)

    # Check for matrix elements with no identical particles
    if matrix_element.get("identical_particle_factor") == 1:
        return symmetry, \
               [range(nexternal)]*len(symmetry),\
               [range(nexternal)]

    logger.info("Finding symmetric diagrams for process %s" % \
                 matrix_element.get('processes')[0].nice_string().\
                 replace("Process: ", ""))

    process = matrix_element.get('processes')[0]
    base_model = process.get('model')
    full_model = model_reader.ModelReader(base_model)
    full_model.set_parameters_and_couplings()
    equivalent_process = base_objects.Process({\
                     'legs': base_objects.LegList([base_objects.Leg({
                               'id': wf.get('pdg_code'),
                               'state': wf.get('leg_state')}) \
                       for wf in matrix_element.get_external_wavefunctions()]),
                     'model': base_model})
    # Writer for the Python matrix elements
    helas_writer = helas_call_writer.PythonUFOHelasCallWriter(base_model)

    # Initialize matrix element evaluation
    evaluator = process_checks.MatrixElementEvaluator(full_model, helas_writer)

    # Get phase space point
    p, w_rambo = evaluator.get_momenta(equivalent_process)
    
    # Check matrix element value for all permutations
    amp2start = []
    final_states = [l.get('id') for l in \
                    equivalent_process.get('legs')[ninitial:]]
    nperm = 0
    perms = []
    ident_perms = []
    for perm in itertools.permutations(range(ninitial, nexternal)):
        if [equivalent_process.get('legs')[i].get('id') for i in perm] != \
           final_states:
            # Non-identical particles permutated
            continue
        ident_perms.append([0,1]+list(perm))
        nperm += 1
        new_p = p[:ninitial] + [p[i] for i in perm]
        
        res = evaluator.evaluate_matrix_element(matrix_element, new_p)
        if not res:
            break
        me_value, amp2 = res
        # Make a list with (8-pos value, magnitude) to easily compare
        amp2sum = sum(amp2)
        amp2mag = []
        for a in amp2:
            a = a*me_value/amp2sum
            if a > 0:
                amp2mag.append(int(math.floor(math.log10(abs(a)))))
            else:
                amp2mag.append(0)
        amp2 = [(int(a*10**(8-am)), am) for (a, am) in zip(amp2, amp2mag)]

        if not perms:
            # This is the first iteration - initialize lists
            # Initiate symmetry with all 1:s
            symmetry = [1 for i in range(len(amp2))]
            # Store initial amplitudes
            amp2start = amp2
            # Initialize list of permutations
            perms = [range(nexternal) for i in range(len(amp2))]
            continue
            
        for i, val in enumerate(amp2):
            if val == (0,0):
                # If amp2 is 0, just set symmetry to 0
                symmetry[i] = 0
                continue
            # Only compare with diagrams below this one
            if val in amp2start[:i]:
                ind = amp2start.index(val)
                # Replace if 1) this amp is unmatched (symmetry[i] > 0) or
                # 2) this amp is matched but matched to an amp larger
                # than ind
                if symmetry[ind] > 0 and \
                   (symmetry[i] > 0 or \
                    symmetry[i] < 0 and -symmetry[i] > ind + 1):
                    symmetry[i] = -(ind+1)
                    perms[i] = [0, 1] + list(perm)
                    symmetry[ind] += 1

    return (symmetry, perms, ident_perms)

def find_symmetry_subproc_group(subproc_group):
    """Find symmetries between the configs in the subprocess group.
    For each config, find all matrix elements with maximum identical
    particle factor. Then take minimal set of these matrix elements,
    and determine symmetries based on these."""

    assert isinstance(subproc_group, group_subprocs.SubProcessGroup),\
           "Argument to find_symmetry_subproc_group has to be SubProcessGroup"

    matrix_elements = subproc_group.get('matrix_elements')

    contributing_mes, me_config_dict = \
                      find_matrix_elements_for_configs(subproc_group)

    nexternal, ninitial = matrix_elements[0].get_nexternal_ninitial()

    all_symmetry = {}
    all_perms = {}

    for me_number in contributing_mes:
        diagram_config_map = dict([(i,n) for i,n in \
                       enumerate(subproc_group.get('diagram_maps')[me_number]) \
                                   if n > 0])
        symmetry, perms, ident_perms = find_symmetry(matrix_elements[me_number])

        # Go through symmetries and remove those for any diagrams
        # where this ME is not supposed to contribute
        for isym, sym_config in enumerate(symmetry):
            if sym_config == 0 or isym not in diagram_config_map:
                continue
            config = diagram_config_map[isym]
            if config not in me_config_dict[me_number] or \
               sym_config < 0 and diagram_config_map[-sym_config-1] not in \
               me_config_dict[me_number]:
                symmetry[isym] = 1
                perms[isym]=range(nexternal)
                if sym_config < 0 and diagram_config_map[-sym_config-1] in \
                       me_config_dict[me_number]:
                    symmetry[-sym_config-1] -= 1

        # Now update the maps all_symmetry and all_perms
        for isym, (perm, sym_config) in enumerate(zip(perms, symmetry)):
            if sym_config in [0,1] or isym not in diagram_config_map:
                continue
            config = diagram_config_map[isym]

            all_perms[config] = perm

            if sym_config > 0:
                all_symmetry[config] = sym_config
            else:
                all_symmetry[config] = -diagram_config_map[-sym_config-1]

    # Fill up all_symmetry and all_perms also for configs that have no symmetry
    for iconf in range(len(subproc_group.get('mapping_diagrams'))):
        all_symmetry.setdefault(iconf+1, 1)
        all_perms.setdefault(iconf+1, range(nexternal))
        # Since we don't want to multiply by symmetry factor here, set to 1
        if all_symmetry[iconf+1] > 1:
            all_symmetry[iconf+1] = 1

    symmetry = [all_symmetry[key] for key in sorted(all_symmetry.keys())]
    perms = [all_perms[key] for key in sorted(all_perms.keys())]

    return symmetry, perms, [perms[0]]
        

def find_matrix_elements_for_configs(subproc_group):
    """For each config, find all matrix elements with maximum identical
    particle factor. Then take minimal set of these matrix elements."""

    matrix_elements = subproc_group.get('matrix_elements')

    n_mes = len(matrix_elements)

    me_config_dict = {}

    # Find the MEs with maximum ident factor corresponding to each config.
    # Only include MEs with identical particles (otherwise no contribution)
    for iconf, diagram_list in \
                           enumerate(subproc_group.get('diagrams_for_configs')):
        # Check if any diagrams contribute to config
        if set(diagram_list) == set([0]):
            continue
        # Add list of MEs with maximum ident factor contributing to this config
        max_ident = max([matrix_elements[i].get('identical_particle_factor') \
                         for i in range(n_mes) if diagram_list[i] > 0])
        max_mes = [i for i in range(n_mes) if \
                   matrix_elements[i].get('identical_particle_factor') == \
                   max_ident and diagram_list[i] > 0 and  max_ident > 1]
        for me in max_mes:
            me_config_dict.setdefault(me, [iconf+1]).append(iconf + 1)

    # Make set of the configs
    for me in me_config_dict:
        me_config_dict[me] = sorted(set(me_config_dict[me]))

    # Sort MEs according to 1) ident factor, 2) number of configs they
    # contribute to
    def me_sort(me1, me2):
        return (matrix_elements[me2].get('identical_particle_factor') \
                - matrix_elements[me1].get('identical_particle_factor'))\
                or (len(me_config_dict[me2]) - len(me_config_dict[me1]))

    sorted_mes = sorted([me for me in me_config_dict], me_sort)

    # Reduce to minimal set of matrix elements
    latest_me = 0
    checked_configs = []
    while latest_me < len(sorted_mes):
        checked_configs.extend(me_config_dict[sorted_mes[latest_me]])
        for me in sorted_mes[latest_me+1:]:
            me_config_dict[me] = [conf for conf in me_config_dict[me] if \
                                  conf not in checked_configs]
            if me_config_dict[me] == []:
                del me_config_dict[me]
        # Re-sort MEs
        sorted_mes = sorted([me for me in me_config_dict], me_sort)
        latest_me += 1

    return sorted_mes, me_config_dict    
