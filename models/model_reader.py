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
import models.check_param_card as card_reader
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
                
            param_card = card_reader.ParamCard(param_card)
            
            key = [k for k in param_card.keys() if not k.startswith('qnumbers ')
                                            and not k.startswith('decay_table')]
            
            if set(key) != set(parameter_dict.keys()):
                raise MadGraph5Error, '''Invalid restriction card (not same block)
                %s != %s
                ''' % (set(key), set(parameter_dict.keys()))
            for block in key:
                for id in parameter_dict[block]:
                    try:
                        value = param_card[block].get(id).value
                    except:
                        raise MadGraph5Error, '%s %s not define' % (block, id)
                    else:
                        exec("locals()[\'%s\'] = %s" % (parameter_dict[block][id].name,
                                          value))
                        parameter_dict[block][id].value = float(value)
                    
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
