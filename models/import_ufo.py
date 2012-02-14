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
from compiler.ast import Continue
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
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object
from madgraph.core.color_algebra import *

import aloha.create_aloha as create_aloha
import aloha.aloha_fct as aloha_fct

import models as ufomodels
import models.model_reader as model_reader
logger = logging.getLogger('models.import_ufo')
logger_mod = logging.getLogger('madgraph.model')

root_path = os.path.dirname(os.path.realpath( __file__ ))
sys.path.append(root_path)

sys.path.append(os.path.join(root_path, os.path.pardir, 'Template', 'bin', 'internal'))
import check_param_card 



class UFOImportError(MadGraph5Error):
    """ a error class for wrong import of UFO model""" 

class InvalidModel(MadGraph5Error):
    """ a class for invalid Model """

def find_ufo_path(model_name):
    """ find the path to a model """

    # Check for a valid directory
    if model_name.startswith('./') and os.path.isdir(model_name):
        model_path = model_name
    elif os.path.isdir(os.path.join(MG5DIR, 'models', model_name)):
        model_path = os.path.join(MG5DIR, 'models', model_name)
    elif os.path.isdir(model_name):
        model_path = model_name
    else:
        raise UFOImportError("Path %s is not a valid pathname" % model_name)

    return model_path

def import_model(model_name):
    """ a practical and efficient way to import a model"""
    
    # check if this is a valid path or if this include restriction file       
    try:
        model_path = find_ufo_path(model_name)
    except UFOImportError:
        if '-' not in model_name:
            raise
        split = model_name.split('-')
        model_name = '-'.join([text for text in split[:-1]])
        model_path = find_ufo_path(model_name)
        restrict_name = split[-1]
         
        restrict_file = os.path.join(model_path, 'restrict_%s.dat'% restrict_name)
        
        #if restriction is full, then we by pass restriction (avoid default)
        if split[-1] == 'full':
            restrict_file = None
    else:
        # Check if by default we need some restrictions
        restrict_name = ""
        if os.path.exists(os.path.join(model_path,'restrict_default.dat')):
            restrict_file = os.path.join(model_path,'restrict_default.dat')
        else:
            restrict_file = None
    
    #import the FULL model
    model = import_full_model(model_path) 
    # restore the model name
    if restrict_name:
        model["name"] += '-' + restrict_name 
    
    #restrict it if needed       
    if restrict_file:
        try:
            logger.info('Restrict model %s with file %s .' % (model_name, os.path.relpath(restrict_file)))
        except OSError:
            # sometimes has trouble with relative path
            logger.info('Restrict model %s with file %s .' % (model_name, restrict_file))
            
        if logger_mod.getEffectiveLevel() > 10:
            logger.info('Run \"set stdout_level DEBUG\" before import for more information.')
            
        # Modify the mother class of the object in order to allow restriction
        model = RestrictModel(model)
        model.restrict_model(restrict_file)
        
    return model

_import_once = []
def import_full_model(model_path):
    """ a practical and efficient way to import one of those models 
        (no restriction file use)"""

    assert model_path == find_ufo_path(model_path)
            
    # Check the validity of the model
    files_list_prov = ['couplings.py','lorentz.py','parameters.py',
                       'particles.py', 'vertices.py']
    files_list = []
    for filename in files_list_prov:
        filepath = os.path.join(model_path, filename)
        if not os.path.isfile(filepath):
            raise UFOImportError,  "%s directory is not a valid UFO model: \n %s is missing" % \
                                                         (model_path, filename)
        files_list.append(filepath)
        
    # use pickle files if defined and up-to-date
    if files.is_uptodate(os.path.join(model_path, 'model.pkl'), files_list):
        try:
            model = save_load_object.load_from_file( \
                                          os.path.join(model_path, 'model.pkl'))
        except Exception, error:
            logger.info('failed to load model from pickle file. Try importing UFO from File')
        else:
            # check path is correct 
            if model.has_key('version_tag') and model.get('version_tag') == os.path.realpath(model_path) + str(misc.get_pkg_info()):
                _import_once.append(model_path)
                return model

    if model_path in _import_once:
        raise MadGraph5Error, 'This model is modified on disk. To reload it you need to quit/relaunch mg5' 

    # Load basic information
    ufo_model = ufomodels.load_model(model_path)
    ufo2mg5_converter = UFOMG5Converter(ufo_model)
    model = ufo2mg5_converter.load_model()
    
    if model_path[-1] == '/': model_path = model_path[:-1] #avoid empty name
    model.set('name', os.path.split(model_path)[-1])
    model.set('version_tag', os.path.realpath(model_path) + str(misc.get_pkg_info()))

    # Load the Parameter/Coupling in a convinient format.
    parameters, couplings = OrganizeModelExpression(ufo_model).main()
    model.set('parameters', parameters)
    model.set('couplings', couplings)
    model.set('functions', ufo_model.all_functions)
    
    # save in a pickle files to fasten future usage
    save_load_object.save_to_file(os.path.join(model_path, 'model.pkl'), model) 
 
    #if default and os.path.exists(os.path.join(model_path, 'restrict_default.dat')):
    #    restrict_file = os.path.join(model_path, 'restrict_default.dat') 
    #    model = import_ufo.RestrictModel(model)
    #    model.restrict_model(restrict_file)
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
        self.conservecharge = set(['charge'])
        
        self.ufomodel = model
        self.checked_lor = set()

        if auto:
            self.load_model()

    def load_model(self):
        """load the different of the model first particles then interactions"""

        # Check the validity of the model
        # 1) check that all lhablock are single word.
        for param in self.ufomodel.all_parameters:
            if param.nature == "external":
                if len(param.lhablock.split())>1:
                    raise UFOImportError, '''LHABlock should be single word which is not the case for
    \'%s\' parameter with lhablock \'%s\'''' % (param.name, param.lhablock)
            

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

        # Find which particles is in the 3/3bar color states (retrun {id: 3/-3})
        color_info = self.find_color_anti_color_rep()


        logger.info('load vertices')
        for interaction_info in self.ufomodel.all_vertices:
            self.add_interaction(interaction_info, color_info)
        
        self.model.set('conserved_charge', self.conservecharge)

        # If we deal with a Loop model here, the order hierarchy MUST be 
        # defined in the file coupling_orders.py and we import it from 
        # there.

        hierarchy={}
        try:
            all_orders = self.ufomodel.all_orders
            for order in all_orders:
                hierarchy[order.name]=order.hierarchy
        except AttributeError:
            pass
        else:
            self.model.set('order_hierarchy', hierarchy)            
        
        # Also set expansion_order, i.e., maximum coupling order per process

        expansion_order={}
        try:
            all_orders = self.ufomodel.all_orders
            for order in all_orders:
                expansion_order[order.name]=order.expansion_order
        except AttributeError:
            pass
        else:
            self.model.set('expansion_order', expansion_order)
        
        #clean memory
        del self.checked_lor
        
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
                elif key in ['mass','width']:
                    particle.set(key, str(value))
                else:
                    particle.set(key, value)
            elif key.lower() not in ('ghostnumber','selfconjugate','goldstoneboson'):
                # add charge -we will check later if those are conserve 
                self.conservecharge.add(key)
                particle.set(key,value, force=True)
            
        assert(12 == nb_property) #basic check that all the information is there         
        
        # Identify self conjugate particles
        if particle_info.name == particle_info.antiname:
            particle.set('self_antipart', True)
            
        # Add the particles to the list
        self.particles.append(particle)


    def find_color_anti_color_rep(self):
        """find which color are in the 3/3bar states"""
        # method look at the 3 3bar 8 configuration.
        # If the color is T(3,2,1) and the interaction F1 F2 V
        # Then set F1 to anticolor (and F2 to color)
        # if this is T(3,1,2) set the opposite
        output = {}
        
        for interaction_info in self.ufomodel.all_vertices:
            if len(interaction_info.particles) != 3:
                continue
            colors = [abs(p.color) for p in interaction_info.particles]
            if colors[:2] == [3,3]:
                if 'T(3,2,1)' in interaction_info.color:
                    color, anticolor, other = interaction_info.particles
                elif 'T(3,1,2)' in interaction_info.color:
                    anticolor, color, other = interaction_info.particles
                else:
                    continue
            elif colors[1:] == [3,3]:
                if 'T(1,2,3)' in interaction_info.color:
                    other, anticolor, color = interaction_info.particles
                elif 'T(1,3,2)' in interaction_info.color:
                    other, color, anticolor = interaction_info.particles
                else:
                    continue                  
               
            elif colors.count(3) == 2:
                if 'T(2,3,1)' in interaction_info.color:
                    color, other, anticolor = interaction_info.particles
                elif 'T(2,1,3)' in interaction_info.color:
                    anticolor, other, color = interaction_info.particles
                else:
                    continue                 
            else:
                continue    
            
            # Check/assign for the color particle
            if color.pdg_code in output: 
                if output[color.pdg_code] == -3:
                    raise InvalidModel, 'Particles %s is sometimes in the 3 and sometimes in the 3bar representations' \
                                    % color.name
            else:
                output[color.pdg_code] = 3
            
            # Check/assign for the anticolor particle
            if anticolor.pdg_code in output: 
                if output[anticolor.pdg_code] == 3:
                    raise InvalidModel, 'Particles %s is sometimes set as in the 3 and sometimes in the 3bar representations' \
                                    % anticolor.name
            else:
                output[anticolor.pdg_code] = -3
        
        return output

            
    def add_interaction(self, interaction_info, color_info):
        """add an interaction in the MG5 model. interaction_info is the 
        UFO vertices information."""
        
        # Import particles content:
        particles = [self.model.get_particle(particle.pdg_code) \
                                    for particle in interaction_info.particles]
      
        if None in particles:
            # Interaction with a ghost/goldstone
            return 
        
        particles = base_objects.ParticleList(particles)
        
        # Check the coherence of the Fermion Flow
        nb_fermion = sum([p.is_fermion() and 1 or 0 for p in particles])
        try:
            if nb_fermion:
                [aloha_fct.check_flow_validity(helas.structure, nb_fermion) \
                                          for helas in interaction_info.lorentz
                                          if helas.name not in self.checked_lor]
                self.checked_lor.update(set([helas.name for helas in interaction_info.lorentz]))
        except aloha_fct.WrongFermionFlow, error:
            text = 'Fermion Flow error for interactions %s: %s: %s\n %s' % \
             (', '.join([p.name for p in interaction_info.particles]), 
                                             helas.name, helas.structure, error)
            raise InvalidModel, text
            
        # Import Lorentz content:
        lorentz = [helas.name for helas in interaction_info.lorentz]
        
        # Import color information:
        colors = [self.treat_color(color_obj, interaction_info, color_info) for color_obj in \
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

        # check if this interaction conserve the charge defined
        for charge in list(self.conservecharge): #duplicate to allow modification
            total = 0
            for part in interaction_info.particles:
                try:
                    total += getattr(part, charge)
                except AttributeError:
                    pass
            if abs(total) > 1e-12:
                logger.info('The model has interaction violating the charge: %s' % charge)
                self.conservecharge.discard(charge)
    
    _pat_T = re.compile(r'T\((?P<first>\d*),(?P<second>\d*)\)')
    _pat_id = re.compile(r'Identity\((?P<first>\d*),(?P<second>\d*)\)')
    
    def treat_color(self, data_string, interaction_info, color_info):
        """ convert the string to ColorString"""
        
        #original = copy.copy(data_string)
        #data_string = p.sub('color.T(\g<first>,\g<second>)', data_string)
        
        
        output = []
        factor = 1
        for term in data_string.split('*'):
            pattern = self._pat_id.search(term)
            if pattern:
                particle = interaction_info.particles[int(pattern.group('first'))-1]
                particle2 = interaction_info.particles[int(pattern.group('second'))-1]
                if particle.color == particle2.color and particle.color in [-6, 6]:
                    error_msg = 'UFO model have inconsistency in the format:\n'
                    error_msg += 'interactions for  particles %s has color information %s\n'
                    error_msg += ' but both fermion are in the same representation %s'
                    raise UFOFormatError, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                if particle.color == particle2.color and particle.color in [-3, 3]:
                    if particle.pdg_code in color_info and particle2.pdg_code in color_info:
                      if color_info[particle.pdg_code] == color_info[particle2.pdg_code]:
                        error_msg = 'UFO model have inconsistency in the format:\n'
                        error_msg += 'interactions for  particles %s has color information %s\n'
                        error_msg += ' but both fermion are in the same representation %s'
                        raise UFOFormatError, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                    elif particle.pdg_code in color_info:
                        color_info[particle2.pdg_code] = -particle.pdg_code
                    elif particle2.pdg_code in color_info:
                        color_info[particle.pdg_code] = -particle2.pdg_code
                    else:
                        error_msg = 'UFO model have inconsistency in the format:\n'
                        error_msg += 'interactions for  particles %s has color information %s\n'
                        error_msg += ' but both fermion are in the same representation %s'
                        raise UFOFormatError, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                
                
                if particle.color == 6:
                    output.append(self._pat_id.sub('color.T6(\g<first>,\g<second>)', term))
                elif particle.color == -6 :
                    output.append(self._pat_id.sub('color.T6(\g<second>,\g<first>)', term))
                elif particle.color == 8:
                    output.append(self._pat_id.sub('color.Tr(\g<first>,\g<second>)', term))
                    factor *= 2
                elif particle.color in [-3,3]:
                    if particle.pdg_code not in color_info:
                        logger.debug('Not able to find the 3/3bar rep from the interactions for particle %s' % particle.name)
                        color_info[particle.pdg_code] = particle.color
                    if particle2.pdg_code not in color_info:
                        logger.debug('Not able to find the 3/3bar rep from the interactions for particle %s' % particle2.name)
                        color_info[particle2.pdg_code] = particle2.color                    
                
                
                    if color_info[particle.pdg_code] == 3 :
                        output.append(self._pat_id.sub('color.T(\g<second>,\g<first>)', term))
                    elif color_info[particle.pdg_code] == -3:
                        output.append(self._pat_id.sub('color.T(\g<first>,\g<second>)', term))
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
         
        # Shift indices by -1
        new_indices = {}
        new_indices = dict([(j,i) for (i,j) in \
                           enumerate(range(1,
                                    len(interaction_info.particles)+1))])

                        
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
    expo_expr = re.compile(r'''(?P<expr>[\w.]+)\s*\*\*\s*(?P<expo>[\d.+-]+)''')
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
        new_exponent = exponent.replace('.','_').replace('+','').replace('-','_m_')
        output = '%s__exp__%s' % (expr, new_exponent)
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
            
class RestrictModel(model_reader.ModelReader):
    """ A class for restricting a model for a given param_card.
    rules applied:
     - Vertex with zero couplings are throw away
     - external parameter with zero/one input are changed into internal parameter.
     - identical coupling/mass/width are replace in the model by a unique one
     """
     
    def default_setup(self):
        """define default value"""
        self.del_coup = []
        super(RestrictModel, self).default_setup()
        self.rule_card = check_param_card.ParamCardRule()
     
    def restrict_model(self, param_card):
        """apply the model restriction following param_card"""

        # Reset particle dict to ensure synchronized particles and interactions
        self.set('particles', self.get('particles'))

        # compute the value of all parameters
        self.set_parameters_and_couplings(param_card)
        # associte to each couplings the associated vertex: def self.coupling_pos
        self.locate_coupling()
        # deal with couplings
        zero_couplings, iden_couplings = self.detect_identical_couplings()

        # remove the out-dated interactions
        self.remove_interactions(zero_couplings)
                
        # replace in interactions identical couplings
        for iden_coups in iden_couplings:
            self.merge_iden_couplings(iden_coups)
        
        # remove zero couplings and other pointless couplings
        self.del_coup += zero_couplings
        self.remove_couplings(self.del_coup)
                
        # deal with parameters
        parameters = self.detect_special_parameters()
        self.fix_parameter_values(*parameters)
        
        # deal with identical parameters
        iden_parameters = self.detect_identical_parameters()
        for iden_param in iden_parameters:
            self.merge_iden_parameters(iden_param)
            
        # change value of default parameter if they have special value:
        # 9.999999e-1 -> 1.0
        # 0.000001e-99 -> 0 Those value are used to avoid restriction
        for name, value in self['parameter_dict'].items():
            if value == 9.999999e-1:
                self['parameter_dict'][name] = 1
            elif value == 0.000001e-99:
                self['parameter_dict'][name] = 0

    def locate_coupling(self):
        """ create a dict couplings_name -> vertex """
        
        self.coupling_pos = {}
        for vertex in self['interactions']:
            for key, coupling in vertex['couplings'].items():
                if coupling in self.coupling_pos:
                    if vertex not in self.coupling_pos[coupling]:
                        self.coupling_pos[coupling].append(vertex)
                else:
                    self.coupling_pos[coupling] = [vertex]
                    
        return self.coupling_pos
        
    def detect_identical_couplings(self, strict_zero=False):
        """return a list with the name of all vanishing couplings"""
        
        dict_value_coupling = {}
        iden_key = set()
        zero_coupling = []
        iden_coupling = []
        
        for name, value in self['coupling_dict'].items():
            if value == 0:
                zero_coupling.append(name)
                continue
            elif not strict_zero and abs(value) < 1e-10:
                return self.detect_identical_couplings(strict_zero=True)
            elif not strict_zero and abs(value) < 1e-15:
                zero_coupling.append(name)
            
            if value in dict_value_coupling:
                iden_key.add(value)
                dict_value_coupling[value].append(name)
            else:
                dict_value_coupling[value] = [name]
        
        for key in iden_key:
            iden_coupling.append(dict_value_coupling[key])

        return zero_coupling, iden_coupling
    
    
    def detect_special_parameters(self):
        """ return the list of (name of) parameter which are zero """
        
        null_parameters = []
        one_parameters = []
        for name, value in self['parameter_dict'].items():
            if value == 0 and name != 'ZERO':
                null_parameters.append(name)
            elif value == 1:
                one_parameters.append(name)
                
        return null_parameters, one_parameters
    
    def detect_identical_parameters(self):
        """ return the list of tuple of name of parameter with the same 
        input value """

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]
        
        # define usefull variable to detect identical input
        block_value_to_var={} #(lhablok, value): list_of_var
        mult_param = set([])       # key of the previous dict with more than one
                              #parameter.
                              
        #detect identical parameter and remove the duplicate parameter
        for param in external_parameters[:]:
            value = self['parameter_dict'][param.name]
            if value == 0:
                continue
            key = (param.lhablock, value)
            mkey =  (param.lhablock, -value)
            if key in block_value_to_var:
                block_value_to_var[key].append((param,1))
                mult_param.add(key)
            elif mkey in block_value_to_var:
                block_value_to_var[mkey].append((param,-1))
                mult_param.add(mkey)
            else: 
                block_value_to_var[key] = [(param,1)]        
        
        output=[]  
        for key in mult_param:
            output.append(block_value_to_var[key])
            
        return output


    def merge_iden_couplings(self, couplings):
        """merge the identical couplings in the interactions"""

        
        logger_mod.debug(' Fuse the Following coupling (they have the same value): %s '% \
                        ', '.join([obj for obj in couplings]))
        
        main = couplings[0]
        self.del_coup += couplings[1:] # add the other coupl to the suppress list
        
        for coupling in couplings[1:]:
            # check if param is linked to an interaction
            if coupling not in self.coupling_pos:
                continue
            # replace the coupling, by checking all coupling of the interaction
            vertices = self.coupling_pos[coupling]
            for vertex in vertices:
                for key, value in vertex['couplings'].items():
                    if value == coupling:
                        vertex['couplings'][key] = main

         
    def merge_iden_parameters(self, parameters):
        """ merge the identical parameters given in argument """
            
        logger_mod.debug('Parameters set to identical values: %s '% \
                        ', '.join(['%s*%s' % (f, obj.name) for (obj,f) in parameters]))
        
        # Extract external parameters
        external_parameters = self['parameters'][('external',)]
        for i, (obj, factor) in enumerate(parameters):
            # Keeped intact the first one and store information
            if i == 0:
                obj.info = 'set of param :' + \
                                     ', '.join([str(f)+'*'+param.name for (param, f) in parameters])
                expr = obj.name
                continue
            # Add a Rule linked to the param_card
            if factor ==1:
                self.rule_card.add_identical(obj.lhablock.lower(), obj.lhacode, 
                                                         parameters[0][0].lhacode )
            else:
                self.rule_card.add_opposite(obj.lhablock.lower(), obj.lhacode, 
                                                         parameters[0][0].lhacode )
            # delete the old parameters                
            external_parameters.remove(obj)    
            # replace by the new one pointing of the first obj of the class
            new_param = base_objects.ModelVariable(obj.name, '%s*%s' %(factor, expr), 'real')
            self['parameters'][()].insert(0, new_param)
        
        # For Mass-Width, we need also to replace the mass-width in the particles
        #This allows some optimization for multi-process.
        if parameters[0][0].lhablock in ['MASS','DECAY']:
            new_name = parameters[0][0].name
            if parameters[0][0].lhablock == 'MASS':
                arg = 'mass'
            else:
                arg = 'width'
            change_name = [p.name for (p,f) in parameters[1:]]
            [p.set(arg, new_name) for p in self['particle_dict'].values() 
                                                       if p[arg] in change_name]
            
    def remove_interactions(self, zero_couplings):
        """ remove the interactions associated to couplings"""
        
        mod = []
        for coup in zero_couplings:
            # some coupling might be not related to any interactions
            if coup not in self.coupling_pos:
                coup, self.coupling_pos.keys()
                continue
            for vertex in self.coupling_pos[coup]:
                modify = False
                for key, coupling in vertex['couplings'].items():
                    if coupling in zero_couplings:
                        modify=True
                        del vertex['couplings'][key]
                if modify:
                    mod.append(vertex)

        # print usefull log and clean the empty interaction
        for vertex in mod:
            part_name = [part['name'] for part in vertex['particles']]
            orders = ['%s=%s' % (order,value) for order,value in vertex['orders'].items()]
                                        
            if not vertex['couplings']:
                logger_mod.debug('remove interactions: %s at order: %s' % \
                                        (' '.join(part_name),', '.join(orders)))
                self['interactions'].remove(vertex)
            else:
                logger_mod.debug('modify interactions: %s at order: %s' % \
                                (' '.join(part_name),', '.join(orders)))       
        
        return
                
    def remove_couplings(self, couplings):                
        #clean the coupling list:
        for name, data in self['couplings'].items():
            for coupling in data[:]:
                if coupling.name in couplings:
                    data.remove(coupling)
                            
        
    def fix_parameter_values(self, zero_parameters, one_parameters):
        """ Remove all instance of the parameters in the model and replace it by 
        zero when needed."""

        # Add a rule for zero/one parameter
        external_parameters = self['parameters'][('external',)]
        for param in external_parameters[:]:
            value = self['parameter_dict'][param.name]
            block = param.lhablock.lower()
            if value == 0:
                self.rule_card.add_zero(block, param.lhacode)
            elif value == 1:
                self.rule_card.add_one(block, param.lhacode)




        special_parameters = zero_parameters + one_parameters
        
        # treat specific cases for masses and width
        for particle in self['particles']:
            if particle['mass'] in zero_parameters:
                particle['mass'] = 'ZERO'
            if particle['width'] in zero_parameters:
                particle['width'] = 'ZERO'
        for pdg, particle in self['particle_dict'].items():
            if particle['mass'] in zero_parameters:
                particle['mass'] = 'ZERO'
            if particle['width'] in zero_parameters:
                particle['width'] = 'ZERO'            

        # check if the parameters is still usefull:
        re_str = '|'.join(special_parameters)
        re_pat = re.compile(r'''\b(%s)\b''' % re_str)
        used = set()
        # check in coupling
        for name, coupling_list in self['couplings'].items():
            for coupling in coupling_list:
                for use in  re_pat.findall(coupling.expr):
                    used.add(use)  
        
        # simplify the regular expression
        re_str = '|'.join([param for param in special_parameters 
                                                          if param not in used])
        re_pat = re.compile(r'''\b(%s)\b''' % re_str)
        
        param_info = {}
        # check in parameters
        for dep, param_list in self['parameters'].items():
            for tag, parameter in enumerate(param_list):
                # update information concerning zero/one parameters
                if parameter.name in special_parameters:
                    param_info[parameter.name]= {'dep': dep, 'tag': tag, 
                                                               'obj': parameter}
                    continue
                                    
                # Bypass all external parameter
                if isinstance(parameter, base_objects.ParamCardVariable):
                    continue

                # check the presence of zero/one parameter
                for use in  re_pat.findall(parameter.expr):
                    used.add(use)

        # modify the object for those which are still used
        for param in used:
            data = self['parameters'][param_info[param]['dep']]
            data.remove(param_info[param]['obj'])
            tag = param_info[param]['tag']
            data = self['parameters'][()]
            if param in zero_parameters:
                data.insert(0, base_objects.ModelVariable(param, '0.0', 'real'))
            else:
                data.insert(0, base_objects.ModelVariable(param, '1.0', 'real'))
                
        # remove completely useless parameters
        for param in special_parameters:
            #by pass parameter still in use
            if param in used:
                logger_mod.debug('fix parameter value: %s' % param)
                continue 
            logger_mod.debug('remove parameters: %s' % param)
            data = self['parameters'][param_info[param]['dep']]
            data.remove(param_info[param]['obj'])
        
 
        

                
                
        
        
        
         
      
    
      
        
        
    
