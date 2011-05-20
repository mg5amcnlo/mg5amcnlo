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
import signal

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
    """Find symmetries between amplitudes by comparing diagram tags
    for all the digrams in the process. Identical diagram tags
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
    for diag in matrix_element.get('diagrams'):
        diagram_numbers.append(diag.get('number'))
        permutations.append(range(nexternal))
        if max(diag.get_vertex_leg_numbers()) > 3:
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

    process = matrix_element.get('processes')[0]
    base_model = process.get('model')
    equivalent_process = base_objects.Process({\
                     'legs': base_objects.LegList([base_objects.Leg({
                               'id': wf.get('pdg_code'),
                               'state': wf.get('leg_state')}) \
                       for wf in matrix_element.get_external_wavefunctions()]),
                     'model': base_model})

    nperm = 0
    perms = []

    diagrams = matrix_element.get('diagrams')
    base_diagrams = matrix_element.get_base_amplitude().get('diagrams')
    model = matrix_element.get('processes')[0].get('model')
    minvert = min([max(diag.get_vertex_leg_numbers()) for diag in diagrams])
    # diagram_tags is a list of unique tags
    diagram_tags = []
    # diagram_classes is a list of lists of diagram numbers belonging
    # to the different classes
    diagram_classes = []
    perms = []
    for diag, base_diagram in zip(diagrams, base_diagrams):
        if any([vert > minvert for vert in
                diag.get_vertex_leg_numbers()]):
            # Only 3-vertices allowed in configs.inc
            continue
        tag = DiagramTag(base_diagram)
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
        permutations[inum] = DiagramTag.reorder_permutation(perms[idx1][idx2],
                                                            perms[idx1][0])
        # ident_perms ordered according to order of external momenta
        perm = DiagramTag.reorder_permutation(perms[idx1][0],
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

class DiagramTag(object):
    """Class to tag diagrams based on objects with some __lt__ measure, e.g.
    PDG code/interaction id (for comparing diagrams from the same amplitude),
    or Lorentz/coupling/mass/width (for comparing AMPs from different MEs).
    Algorithm: Create chains starting from external particles:
    1 \        / 6
    2 /\______/\ 7
    3_ /  |   \_ 8
    4 /   5    \_ 9
                \ 10
    gives ((((9,10,id910),8,id9108),(6,7,id67),id910867)
           (((1,2,id12),(3,4,id34)),id1234),
           5,id91086712345)
    where idN is the id of the corresponding interaction. The ordering within
    chains is based on chain length (depth; here, 1234 has depth 2, 910867 has
    depth 3, 5 has depht 0), and if equal on the ordering of the chain elements.
    The determination of central vertex is based on minimizing the chain length
    for the longest subchain. 
    This gives a unique tag which can be used to identify diagrams
    (instead of symmetry), as well as identify identical matrix elements from
    different processes."""


    def __init__(self, diagram):
        """Initialize with a diagram. Create DiagramTagChainLinks according to
        the diagram, and figure out if we need to shift the central vertex."""

        # wf_dict keeps track of the intermediate particles
        leg_dict = {}
        # Create the chain which will be the diagram tag
        for vertex in diagram.get('vertices'):
            # Only add incoming legs
            legs = vertex.get('legs')[:-1]
            if vertex == diagram.get('vertices')[-1]:
                # If last vertex, all legs are incoming
                legs = vertex.get('legs')
            # Add links corresponding to the relevant legs
            link = DiagramTagChainLink([leg_dict.setdefault(leg.get('number'),
                                        DiagramTagChainLink(self.link_from_leg(leg))) \
                                        for leg in legs],
                                        self.vertex_id_from_vertex(vertex))
            # Add vertex to leg_dict if not last one
            if vertex != vertex.get('legs')[-1]:
                leg_dict[vertex.get('legs')[-1].get('number')] = link

        # The resulting link is the hypothetical result
        self.tag = link

        # Now make sure to find the central vertex in the diagram,
        # defined by the longest leg being as short as possible
        done = False
        while not done:
            # Identify the longest chain in the tag
            longest_chain = self.tag.links[0]
            # Create a new link corresponding to moving one step
            new_link = DiagramTagChainLink(self.tag.links[1:],
                                           self.tag.vertex_id)
            # Create a new final vertex in the direction of the longest link
            other_link = DiagramTagChainLink(list(longest_chain.links) + \
                                             [new_link],
                                             longest_chain.vertex_id)

            if other_link.links[0] < self.tag.links[0]:
                # Switch to new tag, continue search
                self.tag = other_link
            else:
                # We have found the central vertex
                done = True

    def get_external_numbers(self):
        """Get the order of external particles in this tag"""

        return self.tag.get_external_numbers()

    @staticmethod
    def reorder_permutation(perm, start_perm):
        """Reorder a permutation with respect to start_perm"""
        order = [i for (p,i) in \
                 sorted([(p,i) for (i,p) in enumerate(perm)])]
        return [start_perm[i]-1 for i in order]

    @staticmethod
    def link_from_leg(leg):
        """Returns the default end link for a leg: ((id, state), number).
        Note that the number is not taken into account if tag comparison,
        but is used only to extract leg permutations."""
        if leg.get('state'):
            # Identify identical final state particles
            return [((leg.get('id'), 0), leg.get('number'))]
        else:
            # Distinguish identical initial state particles
            return [((leg.get('id'), leg.get('number')), leg.get('number'))]

    @staticmethod
    def vertex_id_from_vertex(vertex):
        """Returns the default vertex id: just the interaction id"""
        return vertex.get('id')

    def __eq__(self, other):
        """Equal if same tag"""
        if type(self) != type(other):
            return False
        return self.tag == other.tag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.tag)

    __repr__ = __str__

class DiagramTagChainLink(object):
    """Chain link for a DiagramTag. A link is a tuple + vertex id + depth,
    with a comparison operator defined"""

    def __init__(self, objects, vertex_id = None):
        """Initialize, either with a tuple of DiagramTagChainLinks and
        a vertex_id (defined by DiagramTag.vertex_id_from_vertex), or
        with an external leg object (end link) defined by
        DiagramTag.link_from_leg"""

        if vertex_id == None:
            # This is an end link, corresponding to an external leg
            self.links = tuple(objects)
            self.vertex_id = 0
            self.depth = 0
            self.end_link = True
            return
        # This is an internal link, corresponding to an internal line
        self.links = tuple(sorted(list(tuple(objects)), reverse=True))
        self.vertex_id = vertex_id
        self.depth = sum([l.depth for l in self.links], 1)
        self.end_link = False

    def get_external_numbers(self):
        """Get the permutation of external numbers (assumed to be the
        second entry in the end link tuples)"""

        if self.end_link:
            return [self.links[0][1]]

        return sum([l.get_external_numbers() for l in self.links], [])

    def __lt__(self, other):
        """Compare self with other in the order:
        1. depth 2. len(links) 3. vertex id 4. measure of links"""

        if self == other:
            return False

        if self.depth != other.depth:
            return self.depth < other.depth

        if len(self.links) != len(other.links):
            return len(self.links) < len(other.links)

        if self.vertex_id != other.vertex_id:
            return self.vertex_id < other.vertex_id

        for i, link in enumerate(self.links):
            if i > len(other.links) - 1:
                return False
            if link != other.links[i]:
                return link < other.links[i]

    def __gt__(self, other):
        return self != other and not self.__lt__(other)

    def __eq__(self, other):
        """For end link,
        consider equal if self.links[0][0] == other.links[0][0],
        i.e., ignore the leg number (in links[0][1])."""

        if self.end_link and other.end_link and  \
               self.depth == other.depth and self.vertex_id == other.vertex_id:
            return self.links[0][0] == other.links[0][0]
        
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)


    def __str__(self):
        if self.end_link:
            return str(self.links)
        return "%s, %s; %d" % (str(self.links),
                               str(self.vertex_id),
                               self.depth)

    __repr__ = __str__
