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
""" How to import a UFO model to the MG5 format """


import fractions
import logging
import os
import re
import sys

from madgraph import MadGraph5Error, MG5DIR
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.iolibs.files as files
import madgraph.iolibs.save_load_object as save_load_object
from madgraph.core.color_algebra import *

import aloha.create_aloha as create_aloha

import models as ufomodels

logger = logging.getLogger('models.import_ufo')



def import_model(model_name):
    """ a practical and efficient way to import one of those models """

    # Check for a valid directory
    if os.path.isdir(model_name):
        model_path = model_name
    elif os.path.isdir(os.path.join(MG5DIR, 'models', model_name)):
        model_path = os.path.join(MG5DIR, 'models', model_name)
    else:
        raise MadGraph5Error("Path %s is not a valid pathname" % model_name)
            
    # Check the validity of the model
    files_list_prov = ['couplings.py','lorentz.py','parameters.py',
                       'particles.py', 'vertices.py']
    files_list = []
    for filename in files_list_prov:
        filepath = os.path.join(model_path, filename)
        if not os.path.isfile(filepath):
            raise MadGraph5Error,  "%s directory is not a valid UFO model: \n %s is missing" % \
                                                         (model_path, filename)
        files_list.append(filepath)
        
    # use pickle files if defined and up-to-date
    if files.is_uptodate(os.path.join(model_path, 'model.pkl'), files_list):
        try:
            model = save_load_object.load_from_file( \
                                          os.path.join(model_path, 'model.pkl'))
        except Exception:
            logger.info('failed to load model from pickle file. Try importing UFO from File')
        else:
            return model

    # Load basic information
    ufo_model = ufomodels.load_model(model_name)
    ufo2mg5_converter = UFOMG5Converter(ufo_model)
    model = ufo2mg5_converter.load_model()
    model.set('name', os.path.split(model_name)[-1])
 
    # Load Abstract Helas routine from Aloha
    #abstract_model = create_aloha.AbstractALOHAModel(model_name)
    #abstract_model.compute_all(save=False)
    #model.set('lorentz', abstract_model)
    
    # Load the Parameter/Coupling in a convinient format.
    parameters, couplings = OrganizeModelExpression(ufo_model).main()
    model.set('parameters', parameters)
    model.set('couplings', couplings)
    model.set('functions', ufo_model.all_functions)
    
    # save in a pickle files to fasten future usage
    save_load_object.save_to_file(os.path.join(model_path, 'model.pkl'), model) 
 
    return model
    

class UFOMG5Converter(object):
    """Convert a UFO model to the MG5 format"""

    use_lower_part_names = False

    def __init__(self, model, auto=False):
        """ initialize empty list for particles/interactions """
        
        self.particles = base_objects.ParticleList()
        self.interactions = base_objects.InteractionList()
        self.model = base_objects.Model()
        self.model.set('particles', self.particles)
        self.model.set('interactions', self.interactions)
        
        self.ufomodel = model
        
        if auto:
            self.load_model()

    def load_model(self):
        """load the different of the model first particles then interactions"""

        logger.info('load particles')
        # Check if multiple particles have the same name but different case.
        # Otherwise, we can use lowercase particle names.
        if len(set([p.name for p in self.ufomodel.all_particles] + \
                   [p.antiname for p in self.ufomodel.all_particles])) == \
           len(set([p.name.lower() for p in self.ufomodel.all_particles] + \
                   [p.antiname.lower() for p in self.ufomodel.all_particles])):
            self.use_lower_part_names = True

        for particle_info in self.ufomodel.all_particles:            
            self.add_particle(particle_info)
            
        logger.info('load vertices')
        for interaction_info in self.ufomodel.all_vertices:
            self.add_interaction(interaction_info)
        
        return self.model
        
    
    def add_particle(self, particle_info):
        """ convert and add a particle in the particle list """
        
        # MG5 have only one entry for particle and anti particles.
        #UFO has two. use the color to avoid duplictions
        if particle_info.pdg_code < 0:
            return
        
        # MG5 doesn't use ghost (use unitary gauges)
        if particle_info.spin < 0:
            return 
        # MG5 doesn't use goldstone boson 
        if hasattr(particle_info, 'GoldstoneBoson'):
            if particle_info.GoldstoneBoson:
                return
               
        # Initialize a particles
        particle = base_objects.Particle()

        nb_property = 0   #basic check that the UFO information is complete
        # Loop over the element defining the UFO particles
        for key,value in particle_info.__dict__.items():
            # Check if we use it in the MG5 definition of a particles
            if key in base_objects.Particle.sorted_keys:
                nb_property +=1
                if key in ['name', 'antiname']:
                    if self.use_lower_part_names:
                        particle.set(key, value.lower())
                    else:
                        particle.set(key, value)
                elif key == 'charge':
                    particle.set(key, float(value))
                else:
                    particle.set(key, value)
            
        assert(12 == nb_property) #basic check that all the information is there         
        
        # Identify self conjugate particles
        if particle_info.name == particle_info.antiname:
            particle.set('self_antipart', True)
            
        # Add the particles to the list
        self.particles.append(particle)


    def add_interaction(self, interaction_info):
        """add an interaction in the MG5 model. interaction_info is the 
        UFO vertices information."""
        
        # Import particles content:
        particles = [self.model.get_particle(particle.pdg_code) \
                                    for particle in interaction_info.particles]

        if None in particles:
            # Interaction with a ghost/goldstone
            return 
            
        particles = base_objects.ParticleList(particles)
        
        # Import Lorentz content:
        lorentz = [helas.name for helas in interaction_info.lorentz]
        
        # Import color information:
        colors = [self.treat_color(color_obj, interaction_info) for color_obj in \
                                    interaction_info.color]
        
        order_to_int={}
        
        for key, couplings in interaction_info.couplings.items():
            if not isinstance(couplings, list):
                couplings = [couplings]
            for coupling in couplings:
                order = tuple(coupling.order.items())
                if order in order_to_int:
                    order_to_int[order].get('couplings')[key] = coupling.name
                else:
                    # Initialize a new interaction with a new id tag
                    interaction = base_objects.Interaction({'id':len(self.interactions)+1})                
                    interaction.set('particles', particles)              
                    interaction.set('lorentz', lorentz)
                    interaction.set('couplings', {key: coupling.name})
                    interaction.set('orders', coupling.order)            
                    interaction.set('color', colors)
                    order_to_int[order] = interaction
                    # add to the interactions
                    self.interactions.append(interaction)

    
    _pat_T = re.compile(r'T\((?P<first>\d*),(?P<second>\d*)\)')
    _pat_id = re.compile(r'Identity\((?P<first>\d*),(?P<second>\d*)\)')
    
    def treat_color(self, data_string, interaction_info):
        """ convert the string to ColorString"""
        
        #original = copy.copy(data_string)
        #data_string = p.sub('color.T(\g<first>,\g<second>)', data_string)
        
        
        output = []
        factor = 1
        for term in data_string.split('*'):
            pattern = self._pat_id.search(term)
            if pattern:
                particle = interaction_info.particles[int(pattern.group('first'))-1]
                if particle.color == -3 :
                    output.append(self._pat_id.sub('color.T(\g<second>,\g<first>)', term))
                elif particle.color == 3:
                    output.append(self._pat_id.sub('color.T(\g<first>,\g<second>)', term))
                elif particle.color == 8:
                    output.append(self._pat_id.sub('color.Tr(\g<first>,\g<second>)', term))
                    factor *= 2
                else:
                    raise MadGraph5Error, \
                          "Unknown use of Identity for particle with color %d" \
                          % particle.color
            else:
                output.append(term)
        data_string = '*'.join(output)

        # Change convention for summed indices
        p = re.compile(r'''\'\w(?P<number>\d+)\'''')
        data_string = p.sub('-\g<number>', data_string)
         
        # Compute how change indices to match MG5 convention
        info = [(i+1,part.color) for i,part in enumerate(interaction_info.particles) 
                 if part.color!=1]
        order = sorted(info, lambda p1, p2:p1[1] - p2[1])
        new_indices={}
        for i,(j, pcolor) in enumerate(order):
            new_indices[j]=i
                        
#            p = re.compile(r'''(?P<prefix>[^-@])(?P<nb>%s)(?P<postfix>\D)''' % j)
#            data_string = p.sub('\g<prefix>@%s\g<postfix>' % i, data_string)
#        data_string = data_string.replace('@','')                    

        output = data_string.split('*')
        output = color.ColorString([eval(data) \
                                              for data in output if data !='1'])
        output.coeff = fractions.Fraction(factor)
        for col_obj in output:
            col_obj.replace_indices(new_indices)
        
        return output
      
class OrganizeModelExpression:
    """Organize the couplings/parameters of a model"""
    
    track_dependant = ['aS','aEWM1'] # list of variable from which we track 
                                   #dependencies those variables should be define
                                   #as external parameters
    
    # regular expression to shorten the expressions
    complex_number = re.compile(r'''complex\((?P<real>[^,\(\)]+),(?P<imag>[^,\(\)]+)\)''')
    expo_expr = re.compile(r'''(?P<expr>[\w.]+)\s*\*\*\s*(?P<expo>\d+)''')
    cmath_expr = re.compile(r'''cmath.(?P<operation>\w+)\((?P<expr>\w+)\)''')
    #operation is usualy sqrt / sin / cos / tan
    conj_expr = re.compile(r'''complexconjugate\((?P<expr>\w+)\)''')
    
    #RE expression for is_event_dependent
    separator = re.compile(r'''[+,\-*/()]''')    
    
    def __init__(self, model):
    
        self.model = model  # UFOMODEL
        self.params = {}     # depend on -> ModelVariable
        self.couplings = {}  # depend on -> ModelVariable
        self.all_expr = {} # variable_name -> ModelVariable
    
    def main(self):
        """Launch the actual computation and return the associate 
        params/couplings."""
        
        self.analyze_parameters()
        self.analyze_couplings()
        return self.params, self.couplings


    def analyze_parameters(self):
        """ separate the parameters needed to be recomputed events by events and
        the others"""
        
        for param in self.model.all_parameters:
            if param.nature == 'external':
                parameter = base_objects.ParamCardVariable(param.name, param.value, \
                                               param.lhablock, param.lhacode)
                
            else:
                expr = self.shorten_expr(param.value)
                depend_on = self.find_dependencies(expr)
                parameter = base_objects.ModelVariable(param.name, expr, param.type, depend_on)
            
            self.add_parameter(parameter)

            
    def add_parameter(self, parameter):
        """ add consistently the parameter in params and all_expr.
        avoid duplication """
        
        assert isinstance(parameter, base_objects.ModelVariable)
        
        if parameter.name in self.all_expr.keys():
            return
        
        self.all_expr[parameter.name] = parameter
        try:
            self.params[parameter.depend].append(parameter)
        except:
            self.params[parameter.depend] = [parameter]
            
    def add_coupling(self, coupling):
        """ add consistently the coupling in couplings and all_expr.
        avoid duplication """
        
        assert isinstance(coupling, base_objects.ModelVariable)
        
        if coupling.name in self.all_expr.keys():
            return
        
        self.all_expr[coupling.value] = coupling
        try:
            self.coupling[coupling.depend].append(coupling)
        except:
            self.coupling[coupling.depend] = [coupling]            
                
                

    def analyze_couplings(self):
        """creates the shortcut for all special function/parameter
        separate the couplings dependent of track variables of the others"""
        
        for coupling in self.model.all_couplings:
            
            # shorten expression, find dependencies, create short object
            expr = self.shorten_expr(coupling.value)
            depend_on = self.find_dependencies(expr)
            parameter = base_objects.ModelVariable(coupling.name, expr, 'complex', depend_on)
            
            # Add consistently in the couplings/all_expr
            try:
                self.couplings[depend_on].append(parameter)
            except KeyError:
                self.couplings[depend_on] = [parameter]
            self.all_expr[coupling.value] = parameter
            

    def find_dependencies(self, expr):
        """check if an expression should be evaluated points by points or not
        """
        depend_on = set()

        # Treat predefined result
        #if name in self.track_dependant:  
        #    return tuple()
        
        # Split the different part of the expression in order to say if a 
        #subexpression is dependent of one of tracked variable
        expr = self.separator.sub(' ',expr)
        
        # look for each subexpression
        for subexpr in expr.split():
            if subexpr in self.track_dependant:
                depend_on.add(subexpr)
                
            elif subexpr in self.all_expr.keys() and self.all_expr[subexpr].depend:
                [depend_on.add(value) for value in self.all_expr[subexpr].depend 
                                if  self.all_expr[subexpr].depend != ('external',)]

        if depend_on:
            return tuple(depend_on)
        else:
            return tuple()


    def shorten_expr(self, expr):
        """ apply the rules of contraction and fullfill
        self.params with dependent part"""

        expr = self.complex_number.sub(self.shorten_complex, expr)
        expr = self.expo_expr.sub(self.shorten_expo, expr)
        expr = self.cmath_expr.sub(self.shorten_cmath, expr)
        expr = self.conj_expr.sub(self.shorten_conjugate, expr)
        return expr
    

    def shorten_complex(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        real = float(matchobj.group('real'))
        imag = float(matchobj.group('imag'))
        if real == 0 and imag ==1:
            new_param = base_objects.ModelVariable('complexi', 'complex(0,1)', 'complex')
            self.add_parameter(new_param)
            return 'complexi'
        else:
            return 'complex(%s, %s)' % (real, imag)
        
        
    def shorten_expo(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        expr = matchobj.group('expr')
        exponent = matchobj.group('expo')
        output = '%s__exp__%s' % (expr, exponent)
        old_expr = '%s**%s' % (expr,exponent)

        if expr.startswith('cmath'):
            return old_expr
        
        if expr.isdigit():
            output = '_' + output #prevent to start with a number
            new_param = base_objects.ModelVariable(output, old_expr,'real')
        else:
            depend_on = self.find_dependencies(expr)
            type = self.search_type(expr)
            new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)
        return output
        
    def shorten_cmath(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        expr = matchobj.group('expr')
        operation = matchobj.group('operation')
        output = '%s__%s' % (operation, expr)
        old_expr = ' cmath.%s(%s) ' %  (operation, expr)
        if expr.isdigit():
            new_param = base_objects.ModelVariable(output, old_expr , 'real')
        else:
            depend_on = self.find_dependencies(expr)
            type = self.search_type(expr)
            new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)
        
        return output        
        
    def shorten_conjugate(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        
        expr = matchobj.group('expr')
        output = 'conjg__%s' % (expr)
        old_expr = ' complexconjugate(%s) ' % expr
        depend_on = self.find_dependencies(expr)
        type = 'complex'
        new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)  
                    
        return output            
    

     
    def search_type(self, expr, dep=''):
        """return the type associate to the expression if define"""
        
        try:
            return self.all_expr[expr].type
        except:
            return 'complex'
            
            
        
        
    
