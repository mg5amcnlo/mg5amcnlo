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
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects

import madgraph.various.misc as misc

import aloha.create_aloha as create_aloha

import models.write_param_card as write_param_card
from madgraph import MG5DIR
from madgraph.iolibs.files import cp, ln, mv
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.group_subprocs')

#===============================================================================
# DiagramTag class to identify diagrams giving the same config
#===============================================================================

class IdentifyConfigTag(diagram_generation.DiagramTag):
    """DiagramTag daughter class to identify diagrams giving the same
    config. Need to compare leg number, mass, width, and color."""

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the end link for a leg needed to identify configs: 
        ((leg numer, spin, mass, width, color), number)."""

        part = model.get_particle(leg.get('id'))

        return [((leg.get('number'), part.get('spin'),
                  part.get('mass'), part.get('width'), part.get('color')),
                 leg.get('number'))]
        
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify configs:
        interaction color, mass, width."""

        inter = model.get_interaction(vertex.get('id'))
                   
        if last_vertex:
            return ((0,),)
        else:
            part = model.get_particle(vertex.get('legs')[-1].get('id'))
            return ((part.get('color'),
                     part.get('mass'), part.get('width')),
                    0)

    @staticmethod
    def flip_vertex(new_vertex, old_vertex, links):
        """Move the wavefunction part of propagator id appropriately"""

        if len(new_vertex[0]) == 1 and len(old_vertex[0]) > 1:
            # We go from a last link to next-to-last link - 
            return (old_vertex[0], new_vertex[0][0])
        elif len(new_vertex[0]) > 1 and len(old_vertex[0]) == 1:
            # We go from next-to-last link to last link - remove propagator info
            return (old_vertex[0],)
        # We should not get here
        raise diagram_generation.DiagramTag.DiagramTagError, \
              "Error in IdentifyConfigTag, wrong setup of vertices in link."
        
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
        self['matrix_element_opts'] = {}

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

        if name == 'matrix_element_opts':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary object" % str(value)

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

        # Find the mapping diagrams
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

        amplitudes = copy.copy(self.get('amplitudes'))

        # The conditional statement tests whether we are dealing with a 
        # loop induced process. We must set compute_loop_nc to True here
        # since the knowledge of the power of Nc coming from potential 
        # loop color trace is necessary for the loop induced output with MadEvent
        if isinstance(amplitudes[0], loop_diagram_generation.LoopAmplitude):
            self.set('matrix_elements', 
              loop_helas_objects.LoopHelasProcess.generate_matrix_elements(
              amplitudes, compute_loop_nc=True,
              matrix_element_opts = self['matrix_element_opts']))
        else:
            self.set('matrix_elements',
                 helas_objects.HelasMultiProcess.\
                                   generate_matrix_elements(amplitudes))

        self.set('amplitudes', diagram_generation.AmplitudeList())

    def generate_name(self, process, criteria='madevent'):
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
            elif criteria == 'madweight':
                 name += part.get_name().replace('~', 'x').\
                            replace('+', 'p').replace('-', 'm')
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
            elif criteria == 'madweight':
                name += part.get_name().replace('~', 'x').\
                            replace('+', 'p').replace('-', 'm')
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
            name += "_" + self.generate_name(dc, criteria)

        return name

    def get_nexternal_ninitial(self):
        """Get number of external and initial particles for this group"""

        assert self.get('matrix_elements'), \
               "Need matrix element to call get_nexternal_ninitial"

        return self.get('matrix_elements')[0].\
               get_nexternal_ninitial()

    def get_num_configs(self):
        """Get number of configs for this group"""

        model = self.get('matrix_elements')[0].get('processes')[0].\
                get('model')
        
        next, nini = self.get_nexternal_ninitial()
        
        return sum([md.get_num_configs(model, nini) for md in 
                    self.get('mapping_diagrams')])

    def find_mapping_diagrams(self):
        """Find all unique diagrams for all processes in this
        process class, and the mapping of their diagrams unto this
        unique diagram."""

        assert self.get('matrix_elements'), \
               "Need matrix elements to run find_mapping_diagrams"

        matrix_elements = self.get('matrix_elements')
        model = matrix_elements[0].get('processes')[0].get('model')
        # mapping_diagrams: The configurations for the non-reducable
        # diagram topologies
        mapping_diagrams = []
        # equiv_diags: Tags identifying diagrams that correspond to
        # the same configuration
        equiv_diagrams = []
        # diagram_maps: A dict from amplitude number to list of
        # diagram maps, pointing to the mapping_diagrams (starting at
        # 1). Diagrams with multi-particle vertices will have 0.
        diagram_maps = {}

        for ime, me in enumerate(matrix_elements):
            # Define here a FDStructure repository which will be used for the 
            # tagging all the diagrams in get_contracted_loop_diagram. Remember
            # the the tagging of each loop updates the FDStructre repository
            # with the new structures identified.
                                    
            if isinstance(me, loop_helas_objects.LoopHelasMatrixElement):
                FDStructRepo = loop_base_objects.FDStructureList([])
                diagrams = [(d.get_contracted_loop_diagram(model,FDStructRepo) if  
                   isinstance(d,loop_base_objects.LoopDiagram) else d) for d in
               me.get('base_amplitude').get('loop_diagrams') if d.get('type')>0]
            else:
                diagrams = me.get('base_amplitude').get('diagrams')
            
            # Check the minimal number of legs we need to include in order
            # to make sure we'll have some valid configurations
            vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
                                  diagrams if diag.get_vertex_leg_numbers()!=[]]
            minvert = min(vert_list) if vert_list!=[] else 0
            
            diagram_maps[ime] = []
            
            for diagram in diagrams:
                # Only use diagrams with all vertices == min_legs, but do not
                # consider artificial vertices, such as those coming from a
                # contracted loop for example, which should be considered as new
                # topologies (the contracted vertex has id == -2.)
                if diagram.get_vertex_leg_numbers()!=[] and \
                                max(diagram.get_vertex_leg_numbers()) > minvert:
                    diagram_maps[ime].append(0)
                    continue
                # Create the equivalent diagram, in the format
                # [[((ext_number1, mass_width_id1), ..., )],
                #  ...]                 (for each vertex)
                equiv_diag = IdentifyConfigTag(diagram, model)
                try:
                    diagram_maps[ime].append(equiv_diagrams.index(\
                                                                equiv_diag) + 1)
                except ValueError:
                    equiv_diagrams.append(equiv_diag)
                    mapping_diagrams.append(diagram)
                    diagram_maps[ime].append(equiv_diagrams.index(\
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
    def group_amplitudes(amplitudes, criteria='madevent', matrix_elements_opts={}):
        """Return a SubProcessGroupList with the amplitudes divided
        into subprocess groups"""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "Argument to group_amplitudes must be AmplitudeList"

        if criteria in ['matrix', 'standalone','pythia8','standalone_cpp','matchbox_cpp',
                        'matchbox']:
            criteria = 'madevent'
        assert criteria in ['madevent', 'madweight']

        logger.info("Organizing processes into subprocess groups")

        process_classes = SubProcessGroup.find_process_classes(amplitudes,criteria)
        ret_list = SubProcessGroupList()
        process_class_numbers = sorted(list(set(process_classes.values())))
        for num in process_class_numbers:
            amp_nums = [key for (key, val) in process_classes.items() if \
                          val == num]
            group = SubProcessGroup({'matrix_element_opts':matrix_elements_opts})
            group.set('amplitudes',
                      diagram_generation.AmplitudeList([amplitudes[i] for i in \
                                                        amp_nums]))
            group.set('number', group.get('amplitudes')[0].get('process').\
                                                                     get('id'))
            group.set('name', group.generate_name(\
                                    group.get('amplitudes')[0].get('process'),
                                    criteria=criteria))
            ret_list.append(group)

        return ret_list

    @staticmethod
    def find_process_classes(amplitudes, criteria):
        """Find all different process classes, classified according to
        initial state and final state. For initial state, we
        differentiate fermions, antifermions, gluons, and masses. For
        final state, only masses."""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "Argument to find_process_classes must be AmplitudeList"
        assert amplitudes
        assert criteria in ['madevent','madweight']

        model = amplitudes[0].get('process').get('model')
        proc_classes = []
        amplitude_classes = {}

        for iamp, amplitude in enumerate(amplitudes):
            process = amplitude.get('process')
            is_parts = [model.get_particle(l.get('id')) for l in \
                        process.get('legs') if not \
                        l.get('state')]
            fs_parts = [model.get_particle(l.get('id')) for l in \
                        process.get('legs') if l.get('state')]

            # This is where the requirements for which particles to
            # combine are defined. Include p.get('is_part') in
            # is_parts selection to distinguish between q and qbar,
            # remove p.get('spin') from fs_parts selection to combine
            # q and g into "j"
            if (criteria=="madevent"):
              proc_class = [ [(p.is_fermion(), ) \
                            for p in is_parts], # p.get('is_part')
                           [(p.get('mass'), p.get('spin'),
                             abs(p.get('color')),l.get('onshell')) for (p, l) \
                             in zip(is_parts + fs_parts, process.get('legs'))],
                           amplitude.get('process').get('id'),
                           process.get('id')]
            if (criteria=="madweight"):
              proc_class = [ [(abs(p.get('pdg_code'))==5, abs(p.get('pdg_code'))==11, 
                           abs(p.get('pdg_code'))==13, abs(p.get('pdg_code'))==15) for p in \
                            fs_parts],
                           amplitude.get('process').get('id')]

            try:
                amplitude_classes[iamp] = proc_classes.index(proc_class)
            except ValueError:
                proc_classes.append(proc_class)
                amplitude_classes[iamp] = len(proc_classes)-1

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
            
    def split_lepton_grouping(self):
        """Return a list of grouping where they are no groupoing over the leptons."""
        
        output = SubProcessGroupList()
        for group in self:
            new_mes = {}
            for me in group['matrix_elements']:
                tags = {}
                for proc in me['processes']:
                    ids = proc.get_final_ids_after_decay()
                    ids = tuple([t if abs(t) in [11, 13,15] else 0 for t in ids])
                    if ids not in tags:
                        tags[ids] = base_objects.ProcessList()
                    tags[ids].append(proc)
                for tag in tags:
                    new_me = copy.copy(me)
                    new_me['processes'] = tags[tag]
                    if tag not in new_mes:
                        new_mes[tag] = helas_objects.HelasMatrixElementList()
                    new_mes[tag].append(new_me)
            for tag in tags:
                new_group = copy.copy(group)
                new_group['matrix_elements'] = new_mes[tag]
                new_group.set('name', new_group.generate_name(\
                                    new_group['matrix_elements'][0]['processes'][0],
                                    criteria='madweight'))
                output.append(new_group)
        return output
        
        
    
#===============================================================================
# DecayChainSubProcessGroup
#===============================================================================

class DecayChainSubProcessGroup(SubProcessGroup):
    """Class to keep track of subprocess groups from a list of decay chains"""

    def default_setup(self):
        """Define object and give default values"""

        self['core_groups'] = SubProcessGroupList()
        self['decay_groups'] = DecayChainSubProcessGroupList()
        # decay_chain_amplitudes is the original DecayChainAmplitudeList
        self['decay_chain_amplitudes'] = diagram_generation.DecayChainAmplitudeList()
        
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
        if name == 'decay_chain_amplitudes':
            if not isinstance(value, diagram_generation.DecayChainAmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DecayChainAmplitudeList" % str(value)
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['core_groups', 'decay_groups', 'decay_chain_amplitudes']

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
                helas_objects.HelasMultiProcess.generate_matrix_elements(\
                              diagram_generation.AmplitudeList(\
                                            self.get('decay_chain_amplitudes')))


        # For each matrix element, check which group it should go into and
        # calculate diagram_maps
        me_assignments = {}
        for me in matrix_elements:
            group_assignment = self.assign_group_to_decay_process(\
                                    me.get('processes')[0])
            assert group_assignment
            try:
                me_assignments[group_assignment].append(me)
            except KeyError:
                me_assignments[group_assignment] = [me]

        # Create subprocess groups corresponding to the different
        # group_assignments

        subproc_groups = SubProcessGroupList()
        for key in sorted(me_assignments.keys()):
            group = SubProcessGroup()
            group.set('matrix_elements', helas_objects.HelasMatrixElementList(\
                me_assignments[key]))
            group.set('number', group.get('matrix_elements')[0].\
                                      get('processes')[0].get('id'))
            group.set('name', group.generate_name(\
                              group.get('matrix_elements')[0].\
                                    get('processes')[0]))
            subproc_groups.append(group)
        
        return subproc_groups

    def assign_group_to_decay_process(self, process):
        """Recursively identify which group process belongs to."""

        # Determine properties for the decay chains
        # The entries of group_assignments are:
        # [(decay_index, (decay_group_index, ...)),
        #  diagram_map (updated), len(mapping_diagrams)]

        group_assignments = []
        
        for decay in process.get('decay_chains'):
            # Find decay group that has this decay in it
            ids = [l.get('id') for l in decay.get('legs')]
            decay_groups = [(i, group) for (i, group) in \
                           enumerate(self.get('decay_groups')) \
                           if any([ids in [[l.get('id') for l in \
                                            a.get('process').get('legs')] \
                                           for a in g.get('amplitudes')] \
                                   for g in group.get('core_groups')])]

            for decay_group in decay_groups:

                group_assignment = \
                    decay_group[1].assign_group_to_decay_process(decay)

                if group_assignment:
                    group_assignments.append((decay_group[0], group_assignment))

        if process.get('decay_chains') and not group_assignments:
            return None

        # Now calculate the corresponding properties for process

        # Find core process group
        ids = [(l.get('id'),l.get('onshell')) for l in process.get('legs')]
        core_groups = [(i, group) for (i, group) in \
                      enumerate(self.get('core_groups')) \
                      if ids in [[(l.get('id'),l.get('onshell')) for l in \
                                  a.get('process').get('legs')] \
                                 for a in group.get('amplitudes')] \
                       and process.get('id') == group.get('number')]

        if not core_groups:
            return None
        
        assert len(core_groups) == 1
        
        core_group = core_groups[0]
        # This is the first return argument - the chain of group indices
        group_assignment = (core_group[0],
                            tuple([g for g in group_assignments]))

        if not group_assignments:
            # No decays - return the values for this process
            return group_assignment

        return group_assignment
    
    #===========================================================================
    # group_amplitudes
    #===========================================================================
    @staticmethod
    def group_amplitudes(decay_chain_amps, criteria='madevent', matrix_elements_opts={}):
        """Recursive function. Starting from a DecayChainAmplitude,
        return a DecayChainSubProcessGroup with the core amplitudes
        and decay chains divided into subprocess groups"""

        assert isinstance(decay_chain_amps, diagram_generation.DecayChainAmplitudeList), \
                  "Argument to group_amplitudes must be DecayChainAmplitudeList"
        if criteria in ['matrix', 'standalone','pythia8','standalone_cpp']:
            criteria = 'madevent'
        assert criteria in ['madevent', 'madweight']
        
        # Collect all amplitudes
        amplitudes = diagram_generation.AmplitudeList()
        for amp in decay_chain_amps:
            amplitudes.extend(amp.get('amplitudes'))

        # Determine core process groups
        core_groups = SubProcessGroup.group_amplitudes(amplitudes, criteria)

        dc_subproc_group = DecayChainSubProcessGroup(\
            {'core_groups': core_groups,
             'decay_chain_amplitudes': decay_chain_amps})

        decays = diagram_generation.DecayChainAmplitudeList()

        # Recursively determine decay chain groups
        for decay_chain_amp in decay_chain_amps:
            decays.extend(decay_chain_amp.get('decay_chains'))
                          
        if decays:
            dc_subproc_group.get('decay_groups').append(\
                DecayChainSubProcessGroup.group_amplitudes(decays, criteria))

        return dc_subproc_group




#===============================================================================
# DecayChainSubProcessGroupList
#===============================================================================
class DecayChainSubProcessGroupList(base_objects.PhysicsObjectList):
    """List of DecayChainSubProcessGroup objects"""

    def is_valid_element(self, obj):
        """Test if object obj is a valid element."""

        return isinstance(obj, DecayChainSubProcessGroup)
    
