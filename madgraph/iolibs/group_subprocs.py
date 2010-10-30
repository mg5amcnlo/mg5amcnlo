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
        self['mapping_diagrams'] = []
        self['diagram_maps'] = {}
        self['multi_matrix'] = helas_objects.HelasMultiProcess()

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
        if name == 'mapping_diagrams':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
        if name == 'diagram_maps':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict" % str(value)
        if name == 'multi_matrix':
            if not isinstance(value, helas_objects.HelasMultiProcess):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMultiProcess" % str(value)
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['number', 'name', 'amplitudes', 'mapping_diagrams',
                'diagram_maps', 'multi_matrix']

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'multi_matrix' and not self[name].get('matrix_elements'):
            self.generate_multi_matrix()
        
        if name in ['mapping_diagrams', 'diagram_maps'] and not self[name]:
            self.set_mapping_diagrams()
        
        return super(SubProcessGroup, self).get(name)

    def set_mapping_diagrams(self):
        """Set mapping_diagrams and diagram_maps, to prepare for
        generation of the super-config.inc files."""

        if not self.get('amplitudes'):
            raise self.PhysicsObjectError, \
                  "Need amplitudes to set mapping_diagrams"

        # All amplitudes are already in the same class
        process_classes = dict([(i, 0) for i in \
                                range(len(self.get('amplitudes')))])
        mapping_diagrams, diagram_maps = \
              self.get('amplitudes').find_mapping_diagrams(process_classes, 0)

        self.set('mapping_diagrams', mapping_diagrams)
        self.set('diagram_maps', diagram_maps)

    def generate_multi_matrix(self):
        """Create a HelasMultiProcess corresponding to the amplitudes
        in self"""

        if not self.get('amplitudes'):
            raise self.PhysicsObjectError, \
                  "Need amplitudes to generate matrix_elements"

        self.set_mapping_diagrams()

        self['multi_matrix'] = helas_objects.HelasMultiProcess(\
            self.get('amplitudes'))

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
            if part.get('mass').lower() == 'zero' and part.is_fermion():
                name += "q"
                if not part.get('is_part'):
                    name +="bar"
            else:
                name += part.get_name().replace('~', 'bar').\
                            replace('+', 'p').replace('-', 'm')
        name += "_"
        for fs_part in fs:
            part = process.get('model').get_particle(fs_part)
            if part.get('mass').lower() == 'zero':
                name += "j"
            else:
                name += part.get_name().replace('~', 'bar').\
                            replace('+', 'p').replace('-', 'm')
        
        return name

    def rearrange_diagram_maps(self):
        """Rearrange the diagram_maps according to the matrix elements in
        the HelasMultiProcess"""

        amplitude_map = self.get('multi_matrix').get('amplitude_map')
        new_diagram_maps = {}
        for key in amplitude_map:
            new_diagram_maps[amplitude_map[key]] = self.get('diagram_maps')[key]

        self.set("diagram_maps", new_diagram_maps)

    @staticmethod
    def group_amplitudes(amplitudes):
        """Return a SubProcessGroupList with the amplitudes divided
        into subprocess groups"""

        if not isinstance(amplitudes, diagram_generation.AmplitudeList):
            raise SubProcessGroup.PhysicsObjectError, \
                  "Argument to group_amplitudes must be AmplitudeList"

        process_classes = amplitudes.find_process_classes()
        ret_list = SubProcessGroupList()
        process_class_numbers = sorted(list(set(process_classes.values())))
        for inum, num in enumerate(process_class_numbers):
            amp_nums = [key for (key, val) in process_classes.items() if \
                          val == num]
            group = SubProcessGroup()
            group.set('number', inum+1)
            group.set('amplitudes',
                      diagram_generation.AmplitudeList([amplitudes[i] for i in \
                                                        amp_nums]))
            group.set('name', group.generate_name())
            ret_list.append(group)

        return ret_list

#===============================================================================
# SubProcessGroupList
#===============================================================================
class SubProcessGroupList(base_objects.PhysicsObjectList):
    """List of SubProcessGroup objects"""

    def is_valid_element(self, obj):
        """Test if object obj is a valid element."""

        return isinstance(obj, SubProcessGroup)
    
