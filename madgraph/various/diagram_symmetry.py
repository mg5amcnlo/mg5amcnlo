################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Module for calculation of symmetries between diagrams, by
evaluating amp2 values for permutations of momenta."""

from __future__ import division

import array
import copy
import fractions
import itertools
import logging
import math
import os
import re
import signal

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_helas_objects as loop_helas_objects

import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.process_checks as process_checks
import madgraph.various.misc as misc

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
    """Find symmetries between amplitudes by comparing diagram tags
    for all the diagrams in the process. Identical diagram tags
    correspond to different external particle permutations of the same
    diagram.
    
    Return list of positive number corresponding to number of
    symmetric diagrams and negative numbers corresponding to the
    equivalent diagram (for e+e->3a, get [6, -1, -1, -1, -1, -1]),
    list of the corresponding permutations needed, and list of all
    permutations of identical particles."""

    if isinstance(matrix_element, group_subprocs.SubProcessGroup):
        return find_symmetry_subproc_group(matrix_element)

    nexternal, ninitial = matrix_element.get_nexternal_ninitial()

    # diagram_numbers is a list of all relevant diagram numbers
    diagram_numbers = []
    # Prepare the symmetry vector with non-used amp2s (due to
    # multiparticle vertices)
    symmetry = []
    permutations = []
    ident_perms = []
    process = matrix_element.get('processes')[0]
    base_model = process.get('model')
    
    if isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement):
        # For loop induced processes we consider only the loops (no R2) and
        # the shrunk diagram instead of the lcut one.
        FDStructRepo = loop_base_objects.FDStructureList([])
        base_diagrams = base_objects.DiagramList(
                   [(d.get_contracted_loop_diagram(base_model,FDStructRepo) if  
                   isinstance(d,loop_base_objects.LoopDiagram) else d) for d in
               matrix_element.get('base_amplitude').get('loop_diagrams') \
                                                            if d.get('type')>0])
        diagrams = matrix_element.get_loop_diagrams()
    else:
        diagrams = matrix_element.get('diagrams')
        base_diagrams = matrix_element.get_base_amplitude().get('diagrams')

    vert_list = [max(diag.get_vertex_leg_numbers()) for diag in diagrams if \
                                        diag.get_vertex_leg_numbers()!=[]]
    min_vert = min(vert_list) if vert_list!=[] else 0
    
    for diag in matrix_element.get('diagrams'):
        diagram_numbers.append(diag.get('number'))
        permutations.append(range(nexternal))
        if diag.get_vertex_leg_numbers()!=[] and \
                                  max(diag.get_vertex_leg_numbers()) > min_vert:
            # Ignore any diagrams with 4-particle vertices
            symmetry.append(0)
        else:
            symmetry.append(1)

    # Check for matrix elements with no identical particles
    if matrix_element.get("identical_particle_factor") == 1:
        return symmetry, \
               permutations,\
               [range(nexternal)]

    logger.info("Finding symmetric diagrams for process %s" % \
                 matrix_element.get('processes')[0].nice_string().\
                 replace("Process: ", ""))

    # diagram_tags is a list of unique tags
    diagram_tags = []
    # diagram_classes is a list of lists of diagram numbers belonging
    # to the different classes
    diagram_classes = []
    perms = []
    for diag, base_diagram in zip(diagrams, base_diagrams):
        if any([vert > min_vert for vert in
                diag.get_vertex_leg_numbers()]):
            # Only 3-vertices allowed in configs.inc
            continue
        
        tag = diagram_generation.DiagramTag(base_diagram)
        try:
            ind = diagram_tags.index(tag)
        except ValueError:
            diagram_classes.append([diag.get('number')])
            perms.append([tag.get_external_numbers()])
            diagram_tags.append(tag)
        else:
            diagram_classes[ind].append(diag.get('number'))
            perms[ind].append(tag.get_external_numbers())

    for inum, diag_number in enumerate(diagram_numbers):
        if symmetry[inum] == 0:
            continue
        idx1 = [i for i, d in enumerate(diagram_classes) if \
                diag_number in d][0]
        idx2 = diagram_classes[idx1].index(diag_number)
        if idx2 == 0:
            symmetry[inum] = len(diagram_classes[idx1])
        else:
            symmetry[inum] = -diagram_classes[idx1][0]
        # Order permutations according to how to reach the first perm
        permutations[inum] = diagram_generation.DiagramTag.reorder_permutation(perms[idx1][idx2],
                                                            perms[idx1][0])
        # ident_perms ordered according to order of external momenta
        perm = diagram_generation.DiagramTag.reorder_permutation(perms[idx1][0],
                                                           perms[idx1][idx2])
        if not perm in ident_perms:
            ident_perms.append(perm)

    return (symmetry, permutations, ident_perms)

def find_symmetry_by_evaluation(matrix_element, evaluator, max_time = 600):
    """Find symmetries between amplitudes by comparing the squared
    amplitudes for all permutations of identical particles.
    
    Return list of positive number corresponding to number of
    symmetric diagrams and negative numbers corresponding to the
    equivalent diagram (for e+e->3a, get [6, -1, -1, -1, -1, -1]),
    list of the corresponding permutations needed, and list of all
    permutations of identical particles.
    max_time gives a cutoff time for finding symmetries (in s)."""

    #if isinstance(matrix_element, group_subprocs.SubProcessGroup):
    #    return find_symmetry_subproc_group(matrix_element, evaluator, max_time)

    assert isinstance(matrix_element, helas_objects.HelasMatrixElement)

    # Exception class and routine to handle timeout
    class TimeOutError(Exception):
        pass
    def handle_alarm(signum, frame):
        raise TimeOutError

    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
            matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
    min_vert = min(vert_list) if vert_list!=[] else 0
    # Prepare the symmetry vector with non-used amp2s (due to
    # multiparticle vertices)
    symmetry = []
    for diag in matrix_element.get('diagrams'):
        # It used to be hardcoded to three instead of min_vert. Need to check
        # if it is ok to use the general min_vert instead.
        if diag.get_vertex_leg_numbers()!=[] and \
                                  max(diag.get_vertex_leg_numbers()) > min_vert:
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
    equivalent_process = base_objects.Process({\
                     'legs': base_objects.LegList([base_objects.Leg({
                               'id': wf.get('pdg_code'),
                               'state': wf.get('leg_state')}) \
                       for wf in matrix_element.get_external_wavefunctions()]),
                     'model': base_model})

    # Get phase space point
    p, w_rambo = evaluator.get_momenta(equivalent_process)
    
    # Check matrix element value for all permutations
    amp2start = []
    final_states = [l.get('id') for l in \
                    equivalent_process.get('legs')[ninitial:]]
    nperm = 0
    perms = []
    ident_perms = []

    # Set timeout for max_time
    signal.signal(signal.SIGALRM, handle_alarm)
    signal.alarm(max_time)
    try:
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
                a = a*me_value/max(amp2sum, 1e-30)
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
    except TimeOutError:
        # Symmetry canceled due to time limit
        logger.warning("Cancel diagram symmetry - time exceeded")

    # Stop the alarm since we're done with this process
    signal.alarm(0)

    return (symmetry, perms, ident_perms)

#===============================================================================
# DiagramTag class to identify matrix elements
#===============================================================================

class IdentifySGConfigTag(diagram_generation.DiagramTag):
    """DiagramTag daughter class to identify diagrams giving the same
    config. Need to compare state, spin, mass, width, and color.
    Warning: If changing this tag, then also CanonicalConfigTag in 
             helas_objects.py must be changed!
    """

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the end link for a leg needed to identify symmetric
        configs: ((leg number for initial state, spin, mass,
        width, color), number)."""

        part = model.get_particle(leg.get('id'))

        state = 0
        if not leg.get('state'):
            # Distinguish identical initial state particles
            state = leg.get('number')

        if part.get('color') != 1:
            charge = 0
        else:
            charge = abs(part.get('charge'))
        
        return [((state, part.get('spin'), part.get('color'), charge,
                  part.get('mass'), part.get('width')),
                 leg.get('number'))]
        
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify symmetric configs:
        interaction color, mass, width."""

        inter = model.get_interaction(vertex.get('id'))
                   
        if last_vertex:
            return (0,)
        else:
            part = model.get_particle(vertex.get('legs')[-1].get('id'))
            return ((part.get('color'),
                     part.get('mass'), part.get('width')),)

def find_symmetry_subproc_group(subproc_group):
    """Find symmetric configs by directly comparing the configurations
    using IdentifySGConfigTag."""

    assert isinstance(subproc_group, group_subprocs.SubProcessGroup),\
           "Argument to find_symmetry_subproc_group has to be SubProcessGroup"

    # diagram_numbers is a list of all relevant diagram numbers
    diagram_numbers = []
    # Prepare the symmetry vector with non-used amp2s (due to
    # multiparticle vertices)
    symmetry = []
    permutations = []
    diagrams = subproc_group.get('mapping_diagrams')
    nexternal, ninitial = \
               subproc_group.get('matrix_elements')[0].get_nexternal_ninitial()
    model = subproc_group.get('matrix_elements')[0].get('processes')[0].\
            get('model')
    vert_list = [max(diag.get_vertex_leg_numbers()) for diag in diagrams if \
                                        diag.get_vertex_leg_numbers()!=[]]
    min_vert = min(vert_list) if vert_list!=[] else 0

    for idiag,diag in enumerate(diagrams):
        diagram_numbers.append(idiag+1)
        permutations.append(range(nexternal))
        if diag.get_vertex_leg_numbers()!=[] and \
                                  max(diag.get_vertex_leg_numbers()) > min_vert:
            # Ignore any diagrams with 4-particle vertices
            symmetry.append(0)
        else:
            symmetry.append(1)

    logger.info("Finding symmetric diagrams for subprocess group %s" % \
                subproc_group.get('name'))

    # diagram_tags is a list of unique tags
    diagram_tags = []
    # diagram_classes is a list of lists of diagram numbers belonging
    # to the different classes
    diagram_classes = []
    perms = []
    for idiag, diag in enumerate(diagrams):
        if diag.get_vertex_leg_numbers()!=[] and \
                                  max(diag.get_vertex_leg_numbers()) > min_vert:
            # Only include vertices up to min_vert
            continue
        tag = IdentifySGConfigTag(diag, model)
        try:
            ind = diagram_tags.index(tag)
        except ValueError:
            diagram_classes.append([idiag + 1])
            perms.append([tag.get_external_numbers()])
            diagram_tags.append(tag)
        else:
            diagram_classes[ind].append(idiag + 1)
            perms[ind].append(tag.get_external_numbers())
    for inum, diag_number in enumerate(diagram_numbers):
        if symmetry[inum] == 0:
            continue
        idx1 = [i for i, d in enumerate(diagram_classes) if \
                diag_number in d][0]
        idx2 = diagram_classes[idx1].index(diag_number)
        # Note that for subproc groups, we want symfact to be 1
        if idx2 > 0:
            symmetry[inum] = -diagram_classes[idx1][0]
        # Order permutations according to how to reach the first perm
        permutations[inum] = diagram_generation.DiagramTag.reorder_permutation(perms[idx1][idx2],
                                                            perms[idx1][0])
    return (symmetry, permutations, [permutations[0]])
    

def old_find_symmetry_subproc_group(subproc_group):
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
