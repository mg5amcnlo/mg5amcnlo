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
"""Methods and classes to group subprocesses according to initial
states, and produce the corresponding grouped subprocess directories."""

import array
import copy
import fractions
import glob
import itertools
import logging
import os
import re
import shutil
import subprocess

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers

import aloha.create_aloha as create_aloha

import models.sm.write_param_card as write_param_card
from madgraph import MadGraph5Error, MG5DIR
from madgraph.iolibs.files import cp, ln, mv
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_v4')

#===============================================================================
# SubProcessGroup
#===============================================================================

class SubProcessGroup(base_objects.PhysicsObject):
    """Class to group a number of amplitudes with same initial states
    into a subprocess group"""

    def default_setup(self):
        """Define object and give default values"""

        self['number'] = 0
        self['name'] = ""
        self['amplitudes'] = diagram_generation.AmplitudeList()
        self['matrix_elements'] = helas_objects.HelasMatrixElementList()
        self['mapping_diagrams'] = []
        self['diagram_maps'] = {}
        self['diagrams_for_configs'] = []
        self['amplitude_map'] = {}

    def filter(self, name, value):
        """Filter for valid property values."""

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid int object" % str(value)
        if name == 'name':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid str object" % str(value)
        if name == 'amplitudes':
            if not isinstance(value, diagram_generation.AmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid amplitudelist" % str(value)
        if name in ['mapping_diagrams', 'diagrams_for_configs']:
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
        if name == 'diagram_maps':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict" % str(value)
        if name == 'matrix_elements':
            if not isinstance(value, helas_objects.HelasMatrixElementList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMatrixElementList" % str(value)

        if name == 'amplitude_map':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict object" % str(value)

        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['number', 'name', 'amplitudes', 'mapping_diagrams',
                'diagram_maps', 'matrix_elements', 'amplitude_map']

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'matrix_elements' and not self[name]:
            self.generate_matrix_elements()
        
        if name in ['mapping_diagrams', 'diagram_maps'] and not self[name]:
            self.set_mapping_diagrams()
        
        if name in ['diagrams_for_configs'] and not self[name]:
            self.set_diagrams_for_configs()
        
        return super(SubProcessGroup, self).get(name)

    def set_mapping_diagrams(self):
        """Set mapping_diagrams and diagram_maps, to prepare for
        generation of the super-config.inc files."""

        if not self.get('amplitudes'):
            raise self.PhysicsObjectError, \
                  "Need amplitudes to set mapping_diagrams"

        # All amplitudes are already in the same class
        mapping_diagrams, diagram_maps = \
              self.find_mapping_diagrams()

        self.set('mapping_diagrams', mapping_diagrams)
        self.set('diagram_maps', diagram_maps)

    #===========================================================================
    # generate_matrix_elements
    #===========================================================================
    def generate_matrix_elements(self):
        """Create a HelasMultiProcess corresponding to the amplitudes
        in self"""

        if not self.get('amplitudes'):
            raise self.PhysicsObjectError, \
                  "Need amplitudes to generate matrix_elements"

        self.set_mapping_diagrams()

        amplitudes = copy.copy(self.get('amplitudes'))

        self.set('matrix_elements',
                 helas_objects.HelasMatrixElementList.\
                                   generate_matrix_elements(amplitudes))

        self.rearrange_diagram_maps()
        self.set('amplitudes', diagram_generation.AmplitudeList())

    def generate_name(self, process):
        """Generate a convenient name for the group, based on  and
        masses"""

        beam = [l.get('id') for l in process.get('legs') if not l.get('state')]
        fs = [l.get('id') for l in process.get('legs') if l.get('state')]
        name = ""
        for beam in beam:
            part = process.get('model').get_particle(beam)
            if part.get('mass').lower() == 'zero' and part.is_fermion() and \
                   part.get('color') != 1:
                name += "q"
            elif part.get('mass').lower() == 'zero' and part.is_fermion() and \
                   part.get('color') == 1 and part.get('pdg_code') % 2 == 1:
                name += "l"
            elif part.get('mass').lower() == 'zero' and part.is_fermion() and \
                   part.get('color') == 1 and part.get('pdg_code') % 2 == 0:
                name += "vl"
            else:
                name += part.get_name().replace('~', 'x').\
                            replace('+', 'p').replace('-', 'm')
        name += "_"
        for fs_part in fs:
            part = process.get('model').get_particle(fs_part)
            if part.get('mass').lower() == 'zero' and part.get('color') != 1 \
                   and part.get('spin') == 2:
                name += "q" # "j"
            elif part.get('mass').lower() == 'zero' and part.get('color') == 1 \
                   and part.get('spin') == 2:
                if part.get('charge') == 0:
                    name += "vl"
                else:
                    name += "l"
            else:
                name += part.get_name().replace('~', 'x').\
                            replace('+', 'p').replace('-', 'm')
        
        for dc in process.get('decay_chains'):
            name += "_" + self.generate_name(dc)

        return name

    def rearrange_diagram_maps(self):
        """Rearrange the diagram_maps according to the matrix elements in
        the HelasMultiProcess"""

        amp_procs = [array.array('i',[l.get('id') for l in \
                                      amp.get('process').get('legs')]) \
                     for amp in self.get('amplitudes')]

        new_diagram_maps = {}

        for ime, me in enumerate(self.get('matrix_elements')):
            me_proc = array.array('i',[l.get('id') for l in \
                                       me.get('processes')[0].get('legs')])
            new_diagram_maps[ime] = \
                       self.get('diagram_maps')[amp_procs.index(me_proc)]

        self.set('diagram_maps', new_diagram_maps)

    def get_nexternal_ninitial(self):
        """Get number of external and initial particles for this group"""

        assert self.get('matrix_elements'), \
               "Need matrix element to call get_nexternal_ninitial"

        return self.get('matrix_elements')[0].\
               get_nexternal_ninitial()

    def get_num_configs(self):
        """Get number of configs for this group"""

        return len(self.get('mapping_diagrams'))

    def find_mapping_diagrams(self):
        """Find all unique diagrams for all processes in this
        process class, and the mapping of their diagrams unto this
        unique diagram."""

        assert self.get('amplitudes'), \
               "Need amplitudes to run find_mapping_diagrams"

        amplitudes = self.get('amplitudes')
        model = amplitudes[0].get('process').get('model')
        # mapping_diagrams: The configurations for the non-reducable
        # diagram topologies
        mapping_diagrams = []
        # diagram_maps: A dict from amplitude number to list of
        # diagram maps, pointing to the mapping_diagrams (starting at
        # 1). Diagrams with multi-particle vertices will have 0.
        diagram_maps = {}
        masswidth_to_pdg = {}

        for iamp, amplitude in enumerate(amplitudes):
            diagrams = amplitude.get('diagrams')
            # Check the minimal number of legs we need to include in order
            # to make sure we'll have some valid configurations
            max_legs = min([max([len(v.get('legs')) for v in \
                                   d.get('vertices') if v.get('id') > 0]) \
                              for d in diagrams])
            diagram_maps[iamp] = []
            for diagram in diagrams:
                # Only use diagrams with all vertices == min_legs
                if any([len(v.get('legs')) > max_legs \
                        for v in diagram.get('vertices') if v.get('id') > 0]):
                    diagram_maps[iamp].append(0)
                    continue
                # Create the equivalent diagram, in the format
                # [[((ext_number1, mass_width_id1), ..., )],
                #  ...]                 (for each vertex)
                equiv_diag = [[(l.get('number'),
                                    (model.get_particle(l.get('id')).\
                                         get('mass'),
                                     model.get_particle(l.get('id')).\
                                         get('width'),
                                     model.get_particle(l.get('id')).\
                                         get('color'))) \
                               for l in v.get('legs')] \
                              for v in diagram.get('vertices')]
                try:
                    diagram_maps[iamp].append(mapping_diagrams.index(\
                                                                equiv_diag) + 1)
                except ValueError:
                    mapping_diagrams.append(equiv_diag)
                    diagram_maps[iamp].append(mapping_diagrams.index(\
                                                                equiv_diag) + 1)

        return mapping_diagrams, diagram_maps

    def get_subproc_diagrams_for_config(self, iconfig):
        """Find the diagrams (number + 1) for all subprocesses
        corresponding to config number iconfig. Return 0 for subprocesses
        without corresponding diagram. Note that the iconfig should
        start at 0."""

        assert self.get('diagram_maps'), \
               "Need diagram_maps to run get_subproc_diagrams_for_config"

        subproc_diagrams = []
        for iproc in \
                range(len(self.get('matrix_elements'))):
            try:
                subproc_diagrams.append(self.get('diagram_maps')[iproc].\
                                        index(iconfig + 1) + 1)
            except ValueError:
                subproc_diagrams.append(0)

        return subproc_diagrams

    def set_diagrams_for_configs(self):
        """Get a list of all diagrams_for_configs"""

        subproc_diagrams_for_config = []
        for iconf in range(len(self.get('mapping_diagrams'))):
            subproc_diagrams_for_config.append(\
                  self.get_subproc_diagrams_for_config(iconf))

        self['diagrams_for_configs'] = subproc_diagrams_for_config
    
    #===========================================================================
    # group_amplitudes
    #===========================================================================
    @staticmethod
    def group_amplitudes(amplitudes):
        """Return a SubProcessGroupList with the amplitudes divided
        into subprocess groups"""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "Argument to group_amplitudes must be AmplitudeList"

        process_classes = SubProcessGroup.find_process_classes(amplitudes)
        ret_list = SubProcessGroupList()
        process_class_numbers = sorted(list(set(process_classes.values())))
        for num in process_class_numbers:
            amp_nums = [key for (key, val) in process_classes.items() if \
                          val == num]
            group = SubProcessGroup()
            group.set('amplitudes',
                      diagram_generation.AmplitudeList([amplitudes[i] for i in \
                                                        amp_nums]))
            group.set('number', group.get('amplitudes')[0].get('process').\
                                                                     get('id'))
            group.set('name', group.generate_name(\
                                    group.get('amplitudes')[0].get('process')))
            ret_list.append(group)

        return ret_list

    @staticmethod
    def find_process_classes(amplitudes):
        """Find all different process classes, classified according to
        initial state and final state. For initial state, we
        differentiate fermions, antifermions, gluons, and masses. For
        final state, only masses."""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "Argument to find_process_classes must be AmplitudeList"
        assert amplitudes

        model = amplitudes[0].get('process').get('model')
        proc_classes = []
        amplitude_classes = {}

        for iamp, amplitude in enumerate(amplitudes):
            is_parts = [model.get_particle(l.get('id')) for l in \
                        amplitude.get('process').get('legs') if not \
                        l.get('state')]
            fs_parts = [model.get_particle(l.get('id')) for l in \
                        amplitude.get('process').get('legs') if l.get('state')]
            diagrams = amplitude.get('diagrams')

            # This is where the requirements for which particles to
            # combine are defined. Include p.get('is_part') in
            # is_parts selection to distinguish between q and qbar,
            # remove p.get('spin') from fs_parts selection to combine
            # q and g into "j"
            proc_class = [ [(p.is_fermion(), ) \
                            for p in is_parts], # p.get('is_part')
                           [(p.get('mass'), p.get('spin'),
                             p.get('color') != 1) for p in \
                            is_parts + fs_parts],
                           amplitude.get('process').get('id')]
            try:
                amplitude_classes[iamp] = proc_classes.index(proc_class)
            except ValueError:
                proc_classes.append(proc_class)
                amplitude_classes[iamp] = proc_classes.index(proc_class)

        return amplitude_classes

#===============================================================================
# SubProcessGroupList
#===============================================================================
class SubProcessGroupList(base_objects.PhysicsObjectList):
    """List of SubProcessGroup objects"""

    def is_valid_element(self, obj):
        """Test if object obj is a valid element."""

        return isinstance(obj, SubProcessGroup)

    def get_matrix_elements(self):
        """Extract the list of matrix elements"""
        return helas_objects.HelasMatrixElementList(\
            sum([group.get('matrix_elements') for group in self], []))

    def get_used_lorentz(self):
        """Return the list of ALOHA routines used in these matrix elements"""
        
        return helas_objects.HelasMultiProcess(
            {'matrix_elements': self.get_matrix_elements()}).get_used_lorentz()
    
    def get_used_couplings(self):
        """Return the list of ALOHA routines used in these matrix elements"""
        
        return helas_objects.HelasMultiProcess(
            {'matrix_elements': self.get_matrix_elements()}).get_used_couplings()
    
#===============================================================================
# DecayChainSubProcessGroup
#===============================================================================

class DecayChainSubProcessGroup(SubProcessGroup):
    """Class to keep track of subprocess groups from a decay chain"""

    def default_setup(self):
        """Define object and give default values"""

        self['core_groups'] = SubProcessGroupList()
        self['decay_groups'] = DecayChainSubProcessGroupList()
        # decay_chain_amplitude is the original DecayChainAmplitude
        self['decay_chain_amplitude'] = diagram_generation.DecayChainAmplitude()
        
    def filter(self, name, value):
        """Filter for valid property values."""

        if name == 'core_groups':
            if not isinstance(value, SubProcessGroupList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid core_groups" % str(value)
        if name == 'decay_groups':
            if not isinstance(value, DecayChainSubProcessGroupList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid decay_groups" % str(value)
        if name == 'decay_chain_amplitude':
            if not isinstance(value, diagram_generation.DecayChainAmplitude):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DecayChainAmplitude" % str(value)
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['core_groups', 'decay_groups', 'decay_chain_amplitude']

    def nice_string(self, indent = 0):
        """Returns a nicely formatted string of the content."""

        mystr = ""
        for igroup, group in enumerate(self.get('core_groups')):
            mystr += " " * indent + "Group %d:\n" % (igroup + 1)
            for amplitude in group.get('amplitudes'):
                mystr = mystr + amplitude.nice_string(indent + 2) + "\n"

        if self.get('decay_groups'):
            mystr += " " * indent + "Decay groups:\n"
            for dec in self.get('decay_groups'):
                mystr = mystr + dec.nice_string(indent + 2) + "\n"

        return  mystr[:-1]

    #===========================================================================
    # generate_helas_decay_chain_subproc_groups
    #===========================================================================
    def generate_helas_decay_chain_subproc_groups(self):
        """Combine core_groups and decay_groups to give
        HelasDecayChainProcesses and new diagram_maps.
        """

        # Combine decays
        matrix_elements = \
                helas_objects.HelasMatrixElementList.generate_matrix_elements(\
                                   diagram_generation.AmplitudeList(\
                                          [self.get('decay_chain_amplitude')]))

        # For each matrix element, check which group it should go into and
        # calculate diagram_maps
        me_assignments = {}
        for me in matrix_elements:
            group_assignment, diagram_map, mapping_diagrams = \
                              self.assign_group_to_decay_process(\
                                     me.get('processes')[0])
            try:
                me_assignments[group_assignment].append((me, diagram_map,
                                                         mapping_diagrams))
            except KeyError:
                me_assignments[group_assignment] = [(me, diagram_map,
                                                     mapping_diagrams)]

        # Create subprocess groups corresponding to the different
        # group_assignments

        subproc_groups = SubProcessGroupList()
        for key in sorted(me_assignments.keys()):
            group = SubProcessGroup()
            group.set('matrix_elements', helas_objects.HelasMatrixElementList(\
                [me[0] for me in me_assignments[key]]))
            group.set('diagram_maps', dict((imap, map[1]) for (imap, map) in \
                                           enumerate(me_assignments[key])))
            group.set('mapping_diagrams', range(me_assignments[key][0][2]))
            group.set('number', group.get('matrix_elements')[0].\
                                      get('processes')[0].get('id'))
            group.set('name', group.generate_name(\
                              group.get('matrix_elements')[0].\
                                    get('processes')[0]))
            subproc_groups.append(group)
        
        return subproc_groups

    def assign_group_to_decay_process(self, process):
        """Recursively identify which group process belongs to,
        and determine the mapping_diagrams for the process."""

        # Determine properties for the decay chains
        # The entries of group_assignments are:
        # [(decay_index, (decay_group_index, ...)),
        #  diagram_map (updated), len(mapping_diagrams)]

        group_assignments = []
        
        for decay in process.get('decay_chains'):
            # Find decay group that has this decay in it
            ids = [l.get('id') for l in decay.get('legs')]
            decay_group = [(i, group) for (i, group) in \
                           enumerate(self.get('decay_groups')) \
                           if any([ids in [[l.get('id') for l in \
                                            a.get('process').get('legs')] \
                                           for a in g.get('amplitudes')] \
                                   for g in group.get('core_groups')])]

            assert len(decay_group) == 1
            decay_group = decay_group[0]

            group_assignment, diagram_map, mapping_diagrams = \
                              decay_group[1].assign_group_to_decay_process(\
                                                                         decay)

            group_assignments.append(((decay_group[0], group_assignment),
                                      diagram_map,
                                      mapping_diagrams))

        # Now calculate the corresponding properties for process

        # Find core process group
        ids = [l.get('id') for l in process.get('legs')]
        core_group = [(i, group) for (i, group) in \
                      enumerate(self.get('core_groups')) \
                      if ids in [[l.get('id') for l in \
                                  a.get('process').get('legs')] \
                                 for a in group.get('amplitudes')]]
        assert len(core_group) == 1
        
        core_group = core_group[0]
        # This is the first return argument - the chain of group indices
        group_assignment = (core_group[0],
                            tuple([g[0] for g in group_assignments]))

        # Get (maximum) length of mapping diagrams
        org_mapping_diagrams = len(core_group[1].get('mapping_diagrams'))

        # Calculate the diagram map for this process
        proc_index = [[l.get('id') for l in a.get('process').get('legs')] \
                      for a in core_group[1].get('amplitudes')].index(ids)
        org_diagram_map = core_group[1].get('diagram_maps')[proc_index]

        if not group_assignments:
            # No decays - return the values for this process
            return group_assignment, org_diagram_map, org_mapping_diagrams

        # mapping_diagrams is simply the product of org_mapping_diagrams
        # with all decay mapping_diagrams
        decay_mapping_diagrams = reduce(lambda x, y: x * y,
                                        [g[2] for g in group_assignments])
        mapping_diagrams = org_mapping_diagrams * decay_mapping_diagrams

        # Now construct the diagram_map from the diagram maps of the
        # decays and org_diagram_map
        diagram_map = []
        for diag_prod in itertools.product(org_diagram_map,
                                         *[ga[1] for ga in group_assignments]):
            if any([d == 0 for d in diag_prod]):
                diagram_map.append(0)
                continue
            diag_num = 1+(diag_prod[0]-1)*decay_mapping_diagrams
            for idm, dm in enumerate(diag_prod[1:]):
                diag_num += (dm-1)*reduce(lambda x, y: x * y,
                                          [g[2] for g in \
                                           group_assignments[idm+1:]], 1)
            diagram_map.append(diag_num)

        return group_assignment, diagram_map, mapping_diagrams
    
    #===========================================================================
    # group_amplitudes
    #===========================================================================
    @staticmethod
    def group_amplitudes(decay_chain_amp):
        """Recursive function. Starting from a DecayChainAmplitude,
        return a DecayChainSubProcessGroup with the core amplitudes
        and decay chains divided into subprocess groups"""

        assert isinstance(decay_chain_amp, diagram_generation.DecayChainAmplitude), \
                  "Argument to group_amplitudes must be DecayChainAmplitude"


        # Determine core process groups
        core_groups = SubProcessGroup.group_amplitudes(\
            decay_chain_amp.get('amplitudes'))

        for group in core_groups:
            group.set_mapping_diagrams()

        dc_subproc_group = DecayChainSubProcessGroup(\
            {'core_groups': core_groups,
             'decay_chain_amplitude': decay_chain_amp})

        # Recursively determine decay chain groups
        for decay_chain in decay_chain_amp.get('decay_chains'):
            dc_subproc_group.get('decay_groups').append(\
                DecayChainSubProcessGroup.group_amplitudes(decay_chain))

        return dc_subproc_group




#===============================================================================
# DecayChainSubProcessGroupList
#===============================================================================
class DecayChainSubProcessGroupList(base_objects.PhysicsObjectList):
    """List of DecayChainSubProcessGroup objects"""

    def is_valid_element(self, obj):
        """Test if object obj is a valid element."""

        return isinstance(obj, DecayChainSubProcessGroup)
    
