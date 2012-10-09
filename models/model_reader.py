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

    def set_parameters_and_couplings(self, param_card = None, scale=None):
        """Read a param_card and calculate all parameters and
        couplings. Set values directly in the parameters and
        couplings, plus add new dictionary coupling_dict from
        parameter name to value."""

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]
        
        # Read in param_card
        if param_card:
            # Create a dictionary from LHA block name and code to parameter name
            parameter_dict = {}
            for param in external_parameters:
                try:
                    dictionary = parameter_dict[param.lhablock.lower()]
                except KeyError:
                    dictionary = {}
                    parameter_dict[param.lhablock.lower()] = dictionary
                dictionary[tuple(param.lhacode)] = param
                
            if isinstance(param_card, basestring):
                # Check that param_card exists
                if not os.path.isfile(param_card):
                    raise MadGraph5Error, "No such file %s" % param_card
                param_card = card_reader.ParamCard(param_card)
            assert isinstance(param_card, card_reader.ParamCard)
                
           
            
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
                        if isinstance(value, str) and value.lower() == 'auto':
                            value = '0.0' 
                        if scale and parameter_dict[block][id].name == 'aS':
                            runner = Alphas_Runner(value, nloop=2)
                            value = runner(scale)
                        exec("locals()[\'%s\'] = %s" % (parameter_dict[block][id].name,
                                          value))
                        parameter_dict[block][id].value = float(value)
           
        else:
            # No param_card, use default values
            for param in external_parameters:
                if scale and parameter_dict[block][id].name == 'aS':
                    runner = Alphas_Runner(value, nloop=3)
                    value = runner(scale)
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
        
        return locals()

class Alphas_Runner(object):
    """Evaluation of strong coupling constant alpha_S"""
    #     Author: Olivier Mattelaer translated from a fortran routine 
    #     written by R. K. Ellis
    #
    #     q -- scale at which alpha_s is to be evaluated
    #
    #     asmz -- value of alpha_s at the mass of the Z-boson
    #     nloop -- the number of loops (1,2, or 3) at which beta 
    #
    #     function is evaluated to determine running.
    #     the values of the cmass and the bmass should be set
    #---------------------------------------------------------------------------    
    
    def __init__(self, asmz, nloop, zmass=91.188, cmass=1.4, bmass=4.7):
    
        self.asmz = asmz
        self.nloop = nloop
        self.zmass = zmass
        self.cmass = cmass
        self.bmass = bmass
    
        assert asmz > 0
        assert cmass > 0
        assert bmass > 0
        assert nloop > -1
    
        t = 2 * math.log(bmass/zmass)
        self.amb = self.newton1(t, asmz, 5)
        t = 2 * math.log(cmass/bmass)
        self.amc = self.newton1(t, self.amb, 4)    
    
    def __call__(self, scale):
        """Evaluation of strong coupling constant alpha_S at scale 'scale'."""
        assert scale > 0
        
        
        if scale < 0.188775276209:
            return 0
        elif scale < self.cmass:
            t = 2 * math.log(scale/self.cmass)
            return self.newton1(t, self.amc, 3)
        elif scale < self.bmass:
            t = 2 * math.log(scale/self.bmass)
            return self.newton1(t, self.amb, 4)
        else:
            t = 2 * math.log(scale/self.zmass)
            return self.newton1(t, self.asmz, 5)

    # B0=(11.-2.*NF/3.)/4./PI
    b0 = [0.716197243913527, 0.66314559621623, 0.61009394851893]
    # C1=(102.D0-38.D0/3.D0*NF)/4.D0/PI/(11.D0-2.D0/3.D0*NF)
    c1 = [0.565884242104515, 0.49019722472304, 0.40134724779695]
    # C2=(2857.D0/2.D0-5033*NF/18.D0+325*NF**2/54)/16.D0/PI**2/(11.D0-2.D0/3.D0*NF)
    c2 = [0.453013579178645, 0.30879037953664, 0.14942733137107]
    # DEL=SQRT(4*C2-C1**2)
    d = [1.22140465909230, 0.99743079911360, 0.66077962451190]
    
    def newton1(self, t, alphas, nf):
        """calculate a_out using nloop beta-function evolution 
        with nf flavours, given starting value as-in
        given alphas and logarithmic separation between 
        input scale and output scale t.
        Evolution is performed using Newton's method,
        with a precision given by tol."""        
        
        nloop = self.nloop
        tol = 5e-4
        arg = nf-3
        b0, c1, c2, d = self.b0[arg], self.c1[arg], self.c2[arg], self.d[arg] 

        if nloop == 2:
            f = lambda AS: 1.0/AS+c1*math.log((c1*AS)/(1+c1*AS))
        elif nloop == 3:
            f = lambda AS: 1.0/AS+0.5*c1*math.log((c2*AS**2)/(1+c1*AS+c2*AS**2)) \
                 -(c1**2-2*c2)/d*math.atan((2*c2*AS+c1)/d)
        
        a_out = alphas / (1 + alphas * b0 * t)
        if nloop == 1:
            return a_out

        a_out = alphas/(1+b0*alphas*t+c1*alphas*math.log(1+alphas*b0*t))
        if a_out < 0:
            a_out = 0.3
        
        while 1:
            AS = a_out
            F = b0 * t + f(alphas) -f(AS)
            if nloop == 2:
                FP=1/(AS**2*(1+c1*AS))
            elif nloop == 3:
                FP=1/(AS**2*(1+c1*AS + c2 * AS**2))
            a_out = AS - F/FP
            delta = abs(F/FP/AS)
            if delta < tol:
                break
        return a_out
            



