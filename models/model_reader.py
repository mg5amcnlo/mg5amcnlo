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
"""Module to allow reading a param_card and setting all parameters and
couplings for a model"""

from __future__ import division

import array
import cmath
import copy
import itertools
import logging
import math
import os
import re

import madgraph.core.base_objects as base_objects
from madgraph import MadGraph5Error, MG5DIR

ZERO = 0

#===============================================================================
# Logger for model_reader
#===============================================================================

logger = logging.getLogger('models.model_reader')

#===============================================================================
# ModelReader: Used to read a param_card and calculate parameters and
#              couplings of the model.
#===============================================================================
class ModelReader(base_objects.Model):
    """Object to read all parameters and couplings of a model
    """

    def default_setup(self):
        """The particles is changed to ParticleList"""
        self['coupling_dict'] = {}
        self['parameter_dict'] = {}
        super(ModelReader, self).default_setup()

    def set_parameters_and_couplings(self, param_card = None):
        """Read a param_card and calculate all parameters and
        couplings. Set values directly in the parameters and
        couplings, plus add new dictionary coupling_dict from
        parameter name to value."""

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]

        # Read in param_card
        if param_card:
            # Check that param_card exists
            if not os.path.isfile(param_card):
                raise MadGraph5Error, \
                      "No such file %s" % param_card

            # Create a dictionary from LHA block name and code to parameter name
            parameter_dict = {}
            for param in external_parameters:
                try:
                    dictionary = parameter_dict[param.lhablock.lower()]
                except KeyError:
                    dictionary = {}
                    parameter_dict[param.lhablock.lower()] = dictionary
                dictionary[tuple(param.lhacode)] = param

            # Now read parameters from the param_card
            param_lines = open(param_card, 'r').read().split('\n')

            # Define regular expressions
            re_block = re.compile("^\s*block\s+(?P<name>\w+)")
            re_decay = re.compile("^\s*decay\s+(?P<pid>\d+)\s+(?P<value>[\d\.e\+-]+)")
            re_single_index = re.compile("^\s*(?P<i1>\d+)\s+(?P<value>[\d\.e\+-]+)")
            re_double_index = re.compile(\
                           "^\s*(?P<i1>\d+)\s+(?P<i2>\d+)\s+(?P<value>[\d\.e\+-]+)")
            block = ""
            # Go through lines in param_card
            for line in param_lines:
                if not line.strip() or line[0] == '#':
                    continue
                line = line.lower()
                # Look for decays
                decay_match = re_decay.match(line)
                if decay_match:
                    block = ""
                    pid = int(decay_match.group('pid'))
                    value = decay_match.group('value')
                    try:
                        exec("locals()[\'%s\'] = %s" % \
                             (parameter_dict['decay'][(pid,)].name,
                              value))
                        parameter_dict['decay'][(pid,)].value = complex(value)
                    except KeyError:
                        pass
                        #logger.warning('No decay parameter found for %d' % pid)
                    continue
                # Look for blocks
                block_match = re_block.match(line)
                if block_match:
                    block = block_match.group('name')
                    continue
                # Look for double indices
                double_index_match = re_double_index.match(line)
                if block and double_index_match:
                    i1 = int(double_index_match.group('i1'))
                    i2 = int(double_index_match.group('i2'))
                    value = double_index_match.group('value')
                    try:
                        exec("locals()[\'%s\'] = %s" % (parameter_dict[block][(i1,i2)].name,
                                          value))
                        parameter_dict[block][(i1,i2)].value = float(value)
                    except KeyError:
                            logger.warning('No parameter found for block %s index %d %d' %\
                                       (block, i1, i2))
                    continue
                # Look for single indices
                single_index_match = re_single_index.match(line)
                if block and single_index_match:
                    i1 = int(single_index_match.group('i1'))
                    value = single_index_match.group('value')
                    try:
                        exec("locals()[\'%s\'] = %s" % (parameter_dict[block][(i1,)].name,
                                          value))
                        parameter_dict[block][(i1,)].value = complex(value)
                    except KeyError:
                        if block not in  ['qnumbers','mass']:
                            logger.warning('No parameter found for block %s index %d' %\
                                       (block, i1))
                    continue
        else:
            # No param_card, use default values
            for param in external_parameters:
                exec("locals()[\'%s\'] = %s" % (param.name, param.value))
                    
        # Define all functions used
        for func in self['functions']:
            exec("def %s(%s):\n   return %s" % (func.name,
                                                ",".join(func.arguments),
                                                func.expr))

        # Extract derived parameters
        derived_parameters = []
        keys = [key for key in self['parameters'].keys() if \
                key != ('external',)]
        keys.sort(key=len)
        for key in keys:
            derived_parameters += self['parameters'][key]


        # Now calculate derived parameters
        for param in derived_parameters:
            try:
                exec("locals()[\'%s\'] = %s" % (param.name, param.expr))
            except Exception as error:
                msg = 'Unable to evaluate %s: raise error: %s' % (param.expr, error)
                raise MadGraph5Error, msg
            param.value = complex(eval(param.name))
            if not eval(param.name) and eval(param.name) != 0:
                logger.warning("%s has no expression: %s" % (param.name,
                                                             param.expr))
        
        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)
        for particle in self.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero' and \
                   eval(particle.get('mass')).real < 0:
                exec("locals()[\'%(width)s\'] = -abs(%(width)s)" % \
                     {'width': particle.get('width')})

        # Extract couplings
        couplings = sum(self['couplings'].values(), [])
        # Now calculate all couplings
        for coup in couplings:
            exec("locals()[\'%s\'] = %s" % (coup.name, coup.expr))
            coup.value = complex(eval(coup.name))
            if not eval(coup.name) and eval(coup.name) != 0:
                logger.warning("%s has no expression: %s" % (coup.name,
                                                             coup.expr))

        # Set parameter and coupling dictionaries

        self.set('parameter_dict', dict([(param.name, param.value) \
                                        for param in external_parameters + \
                                         derived_parameters]))

        # Add "zero"
        self.get('parameter_dict')['ZERO'] = complex(0.)

        self.set('coupling_dict', dict([(coup.name, coup.value) \
                                        for coup in couplings]))
