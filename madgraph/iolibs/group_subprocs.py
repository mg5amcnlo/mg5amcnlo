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

import fractions
import glob
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

        self['matrix_elements'], self['amplitude_map'] = \
                       helas_objects.HelasMatrixElementList.\
                               generate_matrix_elements(self.get('amplitudes'))
        
        self.rearrange_diagram_maps()

    def generate_name(self):
        """Generate a convenient name for the group, based on IS and
        masses"""

        process = self.get('amplitudes')[0].get('process')
        beam = [l.get('id') for l in process.get('legs') if not l.get('state')]
        fs = [l.get('id') for l in process.get('legs') if l.get('state')]
        name = ""
        for beam in beam:
            part = process.get('model').get_particle(beam)
            if part.get('mass').lower() == 'zero' and part.is_fermion() and \
                   part.get('color') != 1:
                name += "q"
                #if not part.get('is_part'):
                #    name +="bar"
            else:
                name += part.get_name().replace('~', 'bar').\
                            replace('+', 'p').replace('-', 'm')
        name += "_"
        for fs_part in fs:
            part = process.get('model').get_particle(fs_part)
            if part.get('mass').lower() == 'zero' and part.get('color') != 1 \
                   and part.get('spin') == 2:
                name += "q" # "j"
            else:
                name += part.get_name().replace('~', 'bar').\
                            replace('+', 'p').replace('-', 'm')
        
        return name

    def rearrange_diagram_maps(self):
        """Rearrange the diagram_maps according to the matrix elements in
        the HelasMultiProcess"""

        amplitude_map = self.get('amplitude_map')
        new_diagram_maps = {}
        for key in amplitude_map:
            new_diagram_maps[amplitude_map[key]] = self.get('diagram_maps')[key]

        self.set("diagram_maps", new_diagram_maps)

    def get_nexternal_ninitial(self):
        """Get number of external and initial particles for this group"""

        assert self.get('matrix_elements'), \
               "Need matrix element to call get_nexternal_ninitial"

        return self.get('matrix_elements')[0].\
               get_nexternal_ninitial()

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
            #max_legs = min([max([len(v.get('legs')) for v in \
            #                       d.get('vertices') if v.get('id') > 0]) \
            #                  for d in diagrams])
            # For now, just use 3-vertices. Will need to fix in MadEvent
            max_legs = 3
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
                                         get('width'))) \
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
            group.set('name', group.generate_name())
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
    
#===============================================================================
# DecayChainSubProcessGroup
#===============================================================================

class DecayChainSubProcessGroup(SubProcessGroup):
    """Class to keep track of subprocess groups from a decay chain"""

    def default_setup(self):
        """Define object and give default values"""

        self['core_groups'] = SubProcessGroupList()
        self['decay_groups'] = DecayChainSubProcessGroupList()
        # decay_ids is a list of PDG codes for all decaying particles
        self['decay_ids'] = []
        # helas_decay_chains is a list of HelasDecayChainProcessLists,
        # corresponding to the core groups combined with decays
        self['helas_decay_chains'] = []
        
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
        if name == 'decay_ids':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
        if name == 'helas_decay_chains':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['core_groups', 'decay_groups', 'decay_ids',
                'helas_decay_chains']

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
    # generate_helas_decay_chain_processes
    #===========================================================================
    def generate_helas_decay_chain_processes(self):
        """Create a HelasDecayChainProcess corresponding to each core_group"""

        # Start with calling generate_helas_decay_chain_process for
        # all decay chain groups
        for decay_group in self.get('decay_groups'):
            decay_group.generate_helas_decay_chain_processes()

        # Now start working on the core groups

        for core_group in self.get('core_groups'):
            if not core_group.get('amplitudes'):
                raise self.PhysicsObjectError, \
                      "Need amplitudes to generate matrix_elements"

            # Determine mapping diagrams for this group
            core_group.set_mapping_diagrams()
            print "Before:"
            print core_group.get('mapping_diagrams')
            print core_group.get('diagram_maps')

            matrix_elements, amplitude_map = \
                      helas_objects.HelasMatrixElementList.\
                         generate_matrix_elements(core_group.get('amplitudes'),
                                                  False,
                                                  self.get('decay_ids'))
            core_group.set('matrix_elements', matrix_elements)
            core_group.set('amplitude_map', amplitude_map)

            core_group.rearrange_diagram_maps()

            print "After:"
            print core_group.get('mapping_diagrams')
            print core_group.get('diagram_maps')

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

        dc_subproc_group = DecayChainSubProcessGroup(\
            {'core_groups': core_groups,
             'decay_ids': decay_chain_amp.get_decay_ids()})
        
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
    
