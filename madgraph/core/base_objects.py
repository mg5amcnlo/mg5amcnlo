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

"""Definitions of all basic objects used in the core code: particle, 
interaction, model, leg, vertex, process, ..."""

import copy
import itertools
import logging
import numbers
import os
import re
import StringIO
import madgraph.core.color_algebra as color
from madgraph import MadGraph5Error, MG5DIR

logger = logging.getLogger('madgraph.base_objects')

#===============================================================================
# PhysicsObject
#===============================================================================
class PhysicsObject(dict):
    """A parent class for all physics objects."""

    class PhysicsObjectError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a physics object."""
        pass

    def __init__(self, init_dict={}):
        """Creates a new particle object. If a dictionary is given, tries to 
        use it to give values to properties."""

        dict.__init__(self)
        self.default_setup()

        assert isinstance(init_dict, dict), \
                            "Argument %s is not a dictionary" % repr(init_dict)


        for item in init_dict.keys():
            self.set(item, init_dict[item])
        

    def __getitem__(self, name):
        """ force the check that the property exist before returning the 
            value associated to value. This ensure that the correct error 
            is always raise
        """

        try:
            return dict.__getitem__(self, name)
        except KeyError:
            self.is_valid_prop(name) #raise the correct error


    def default_setup(self):
        """Function called to create and setup default values for all object
        properties"""
        pass

    def is_valid_prop(self, name):
        """Check if a given property name is valid"""

        assert isinstance(name, str), \
                                 "Property name %s is not a string" % repr(name)

        if name not in self.keys():
            raise self.PhysicsObjectError, \
                        "%s is not a valid property for this object" % name

        return True

    def get(self, name):
        """Get the value of the property name."""

        return self[name]

    def set(self, name, value, force=False):
        """Set the value of the property name. First check if value
        is a valid value for the considered property. Return True if the
        value has been correctly set, False otherwise."""

        if not __debug__ or force:
            self[name] = value
            return True

        if self.is_valid_prop(name):
            try:
                self.filter(name, value)
                self[name] = value
                return True
            except self.PhysicsObjectError, why:
                logger.warning("Property " + name + " cannot be changed:" + \
                                str(why))
                return False

    def filter(self, name, value):
        """Checks if the proposed value is valid for a given property
        name. Returns True if OK. Raises an error otherwise."""

        return True

    def get_sorted_keys(self):
        """Returns the object keys sorted in a certain way. By default,
        alphabetical."""

        return self.keys().sort()

    def __str__(self):
        """String representation of the object. Outputs valid Python 
        with improved format."""

        mystr = '{\n'
        for prop in self.get_sorted_keys():
            if isinstance(self[prop], str):
                mystr = mystr + '    \'' + prop + '\': \'' + \
                        self[prop] + '\',\n'
            elif isinstance(self[prop], float):
                mystr = mystr + '    \'' + prop + '\': %.2f,\n' % self[prop]
            else:
                mystr = mystr + '    \'' + prop + '\': ' + \
                        repr(self[prop]) + ',\n'
        mystr = mystr.rstrip(',\n')
        mystr = mystr + '\n}'

        return mystr

    __repr__ = __str__


#===============================================================================
# PhysicsObjectList
#===============================================================================
class PhysicsObjectList(list):
    """A class to store lists of physics object."""

    class PhysicsObjectListError(Exception):
        """Exception raised if an error occurs in the definition
        or execution of a physics object list."""
        pass

    def __init__(self, init_list=None):
        """Creates a new particle list object. If a list of physics 
        object is given, add them."""

        list.__init__(self)

        if init_list is not None:
            for object in init_list:
                self.append(object)
                
    def append(self, object):
        """Appends an element, but test if valid before."""
        
        assert self.is_valid_element(object), \
            "Object %s is not a valid object for the current list" % repr(object)

        list.append(self, object)

    def is_valid_element(self, obj):
        """Test if object obj is a valid element for the list."""
        return True

    def __str__(self):
        """String representation of the physics object list object. 
        Outputs valid Python with improved format."""

        mystr = '['

        for obj in self:
            mystr = mystr + str(obj) + ',\n'

        mystr = mystr.rstrip(',\n')

        return mystr + ']'

#===============================================================================
# Particle
#===============================================================================
class Particle(PhysicsObject):
    """The particle object containing the whole set of information required to
    univocally characterize a given type of physical particle: name, spin, 
    color, mass, width, charge,... The is_part flag tells if the considered
    particle object is a particle or an antiparticle. The self_antipart flag
    tells if the particle is its own antiparticle."""

    sorted_keys = ['name', 'antiname', 'spin', 'color',
                   'charge', 'mass', 'width', 'pdg_code',
                   'texname', 'antitexname', 'line', 'propagating',
                   'is_part', 'self_antipart']

    def default_setup(self):
        """Default values for all properties"""

        self['name'] = 'none'
        self['antiname'] = 'none'
        self['spin'] = 1
        self['color'] = 1
        self['charge'] = 1.
        self['mass'] = 'zero'
        self['width'] = 'zero'
        self['pdg_code'] = 0
        self['texname'] = 'none'
        self['antitexname'] = 'none'
        self['line'] = 'dashed'
        self['propagating'] = True
        self['is_part'] = True
        self['self_antipart'] = False

    def filter(self, name, value):
        """Filter for valid particle property values."""

        if name in ['name', 'antiname']:
            # Must start with a letter, followed by letters,  digits,
            # - and + only
            p = re.compile('\A[a-zA-Z]+[\w]*[\-\+]*~?\Z')
            if not p.match(value):
                raise self.PhysicsObjectError, \
                        "%s is not a valid particle name" % value

        if name is 'spin':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Spin %s is not an integer" % repr(value)
            if value < 1 or value > 5:
                raise self.PhysicsObjectError, \
                   "Spin %i is smaller than one" % value

        if name is 'color':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Color %s is not an integer" % repr(value)
            if value not in [1, 3, 6, 8]:
                raise self.PhysicsObjectError, \
                   "Color %i is not valid" % value

        if name in ['mass', 'width']:
            # Must start with a letter, followed by letters, digits or _
            p = re.compile('\A[a-zA-Z]+[\w\_]*\Z')
            if not p.match(value):
                raise self.PhysicsObjectError, \
                        "%s is not a valid name for mass/width variable" % \
                        value

        if name is 'pdg_code':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "PDG code %s is not an integer" % repr(value)

        if name is 'line':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Line type %s is not a string" % repr(value)
            if value not in ['dashed', 'straight', 'wavy', 'curly', 'double','swavy','scurly']:
                raise self.PhysicsObjectError, \
                   "Line type %s is unknown" % value

        if name is 'charge':
            if not isinstance(value, float):
                raise self.PhysicsObjectError, \
                    "Charge %s is not a float" % repr(value)

        if name is 'propagating':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "Propagating tag %s is not a boolean" % repr(value)

        if name in ['is_part', 'self_antipart']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a boolean" % (name, repr(value))

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return self.sorted_keys

    # Helper functions

    def get_pdg_code(self):
        """Return the PDG code with a correct minus sign if the particle is its
        own antiparticle"""

        if not self['is_part'] and not self['self_antipart']:
            return - self['pdg_code']
        else:
            return self['pdg_code']

    def get_anti_pdg_code(self):
        """Return the PDG code of the antiparticle with a correct minus sign 
        if the particle is its own antiparticle"""

        if not self['self_antipart']:
            return - self.get_pdg_code()
        else:
            return self['pdg_code']

    def get_color(self):
        """Return the color code with a correct minus sign"""

        if not self['is_part'] and abs(self['color']) in [3, 6]:
            return - self['color']
        else:
            return self['color']

    def get_anti_color(self):
        """Return the color code of the antiparticle with a correct minus sign
        """

        if self['is_part'] and self['color'] not in [1, 8]:
            return - self['color']
        else:
            return self['color']

    def get_name(self):
        """Return the name if particle, antiname if antiparticle"""

        if not self['is_part'] and not self['self_antipart']:
            return self['antiname']
        else:
            return self['name']

    def get_helicity_states(self):
        """Return a list of the helicity states for the onshell particle"""

        spin = self.get('spin')
        if spin == 1:
            # Scalar
            return [ 0 ]
        elif spin == 2:
            # Spinor
            return [ -1, 1 ]
        elif spin == 3 and self.get('mass').lower() == 'zero':
            # Massless vector
            return [ -1, 1 ]
        elif spin == 3:
            # Massive vector
            return [ -1, 0, 1 ]
        elif spin == 5 and self.get('mass').lower() == 'zero':
            # Massless tensor
            return [-2, -1, 1, 2]
        elif spin == 5:
            # Massive tensor
            return [-2, -1, 0, 1, 2]
        
        raise self.PhysicsObjectError, \
              "No helicity state assignment for spin %d particles" % spin

    def is_fermion(self):
        """Returns True if this is a fermion, False if boson"""

        return self['spin'] % 2 == 0

    def is_boson(self):
        """Returns True if this is a boson, False if fermion"""

        return self['spin'] % 2 == 1

#===============================================================================
# ParticleList
#===============================================================================
class ParticleList(PhysicsObjectList):
    """A class to store lists of particles."""

    def is_valid_element(self, obj):
        """Test if object obj is a valid Particle for the list."""
        return isinstance(obj, Particle)
                    

    def find_name(self, name):
        """Try to find a particle with the given name. Check both name
        and antiname. If a match is found, return the a copy of the 
        corresponding particle (first one in the list), with the 
        is_part flag set accordingly. None otherwise."""

        assert isinstance(name, str), "%s is not a valid string" % str(name) 

        for part in self:
            mypart = copy.copy(part)
            if part.get('name') == name:
                mypart.set('is_part', True)
                return mypart
            elif part.get('antiname') == name:
                mypart.set('is_part', False)
                return mypart

        return None

    def generate_ref_dict(self):
        """Generate a dictionary of part/antipart pairs (as keys) and
        0 (as value)"""

        ref_dict_to0 = {}

        for part in self:
            ref_dict_to0[(part.get_pdg_code(), part.get_anti_pdg_code())] = [0]
            ref_dict_to0[(part.get_anti_pdg_code(), part.get_pdg_code())] = [0]

        return ref_dict_to0

    def generate_dict(self):
        """Generate a dictionary from particle id to particle.
        Include antiparticles.
        """

        particle_dict = {}

        for particle in self:
            particle_dict[particle.get('pdg_code')] = particle
            if not particle.get('self_antipart'):
                antipart = copy.copy(particle)
                antipart.set('is_part', False)
                particle_dict[antipart.get_pdg_code()] = antipart

        return particle_dict


#===============================================================================
# Interaction
#===============================================================================
class Interaction(PhysicsObject):
    """The interaction object containing the whole set of information 
    required to univocally characterize a given type of physical interaction: 
    
    particles: a list of particle ids
    color: a list of string describing all the color structures involved
    lorentz: a list of variable names describing all the Lorentz structure
             involved
    couplings: dictionary listing coupling variable names. The key is a
               2-tuple of integers referring to color and Lorentz structures
    orders: dictionary listing order names (as keys) with their value
    """

    sorted_keys = ['id', 'particles', 'color', 'lorentz', 'couplings', 'orders']

    def default_setup(self):
        """Default values for all properties"""

        self['id'] = 0
        self['particles'] = []
        self['color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        self['orders'] = {}

    def filter(self, name, value):
        """Filter for valid interaction property values."""

        if name == 'id':
            #Should be an integer
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)

        if name == 'particles':
            #Should be a list of valid particle names
            if not isinstance(value, ParticleList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of particles" % str(value)

        if name == 'orders':
            #Should be a dict with valid order names ask keys and int as values
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict for coupling orders" % \
                                                                    str(value)
            for order in value.keys():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(order)
                if not isinstance(value[order], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value[order])

        if name in ['color']:
            #Should be a list of list strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of Color Strings" % str(value)
            for mycolstring in value:
                if not isinstance(mycolstring, color.ColorString):
                    raise self.PhysicsObjectError, \
                            "%s is not a valid list of Color Strings" % str(value)

        if name in ['lorentz']:
            #Should be a list of list strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of strings" % str(value)
            for mystr in value:
                if not isinstance(mystr, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        if name == 'couplings':
            #Should be a dictionary of strings with (i,j) keys
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary for couplings" % \
                                                                str(value)

            for key in value.keys():
                if not isinstance(key, tuple):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple" % str(key)
                if len(key) != 2:
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple with 2 elements" % str(key)
                if not isinstance(key[0], int) or not isinstance(key[1], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple of integer" % str(key)
                if not isinstance(value[key], str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % value[key]

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return self.sorted_keys 
                

    def generate_dict_entries(self, ref_dict_to0, ref_dict_to1):
        """Add entries corresponding to the current interactions to 
        the reference dictionaries (for n>0 and n-1>1)"""

        # Create n>0 entries. Format is (p1,p2,p3,...):interaction_id.
        # We are interested in the unordered list, so use sorted()

        pdg_tuple = tuple(sorted([p.get_pdg_code() for p in self['particles']]))
        if pdg_tuple not in ref_dict_to0.keys():
            ref_dict_to0[pdg_tuple] = [self['id']]
        else:
            ref_dict_to0[pdg_tuple].append(self['id'])

        # Create n-1>1 entries. Note that, in the n-1 > 1 dictionary,
        # the n-1 entries should have opposite sign as compared to
        # interaction, since the interaction has outgoing particles,
        # while in the dictionary we treat the n-1 particles as
        # incoming

        for part in self['particles']:

            # We are interested in the unordered list, so use sorted()
            pdg_tuple = tuple(sorted([p.get_pdg_code() for (i, p) in \
                                      enumerate(self['particles']) if \
                                      i != self['particles'].index(part)]))
            pdg_part = part.get_anti_pdg_code()
            if pdg_tuple in ref_dict_to1.keys():
                if (pdg_part, self['id']) not in  ref_dict_to1[pdg_tuple]:
                    ref_dict_to1[pdg_tuple].append((pdg_part, self['id']))
            else:
                ref_dict_to1[pdg_tuple] = [(pdg_part, self['id'])]

    def __str__(self):
        """String representation of an interaction. Outputs valid Python 
        with improved format. Overrides the PhysicsObject __str__ to only
        display PDG code of involved particles."""

        mystr = '{\n'

        for prop in self.get_sorted_keys():
            if isinstance(self[prop], str):
                mystr = mystr + '    \'' + prop + '\': \'' + \
                        self[prop] + '\',\n'
            elif isinstance(self[prop], float):
                mystr = mystr + '    \'' + prop + '\': %.2f,\n' % self[prop]
            elif isinstance(self[prop], ParticleList):
                mystr = mystr + '    \'' + prop + '\': [%s],\n' % \
                   ','.join([str(part.get_pdg_code()) for part in self[prop]])
            else:
                mystr = mystr + '    \'' + prop + '\': ' + \
                        repr(self[prop]) + ',\n'
        mystr = mystr.rstrip(',\n')
        mystr = mystr + '\n}'

        return mystr

#===============================================================================
# InteractionList
#===============================================================================
class InteractionList(PhysicsObjectList):
    """A class to store lists of interactionss."""

    def is_valid_element(self, obj):
        """Test if object obj is a valid Interaction for the list."""

        return isinstance(obj, Interaction)

    def generate_ref_dict(self):
        """Generate the reference dictionaries from interaction list.
        Return a list where the first element is the n>0 dictionary and
        the second one is n-1>1."""

        ref_dict_to0 = {}
        ref_dict_to1 = {}

        for inter in self:
            inter.generate_dict_entries(ref_dict_to0, ref_dict_to1)

        return [ref_dict_to0, ref_dict_to1]

    def generate_dict(self):
        """Generate a dictionary from interaction id to interaction.
        """

        interaction_dict = {}

        for inter in self:
            interaction_dict[inter.get('id')] = inter

        return interaction_dict

    def synchronize_interactions_with_particles(self, particle_dict):
        """Make sure that the particles in the interactions are those
        in the particle_dict, and that there are no interactions
        refering to particles that don't exist. To be called when the
        particle_dict is updated in a model.
        """

        iint = 0
        while iint < len(self):
            inter = self[iint]
            particles = inter.get('particles')
            try:
                for ipart, part in enumerate(particles):
                    particles[ipart] = particle_dict[part.get_pdg_code()]
                iint += 1
            except KeyError:
                # This interaction has particles that no longer exist
                self.pop(iint)

#===============================================================================
# Model
#===============================================================================
class Model(PhysicsObject):
    """A class to store all the model information."""
    
    def default_setup(self):

        self['name'] = ""
        self['particles'] = ParticleList()
        self['interactions'] = InteractionList()
        self['parameters'] = None
        self['functions'] = None
        self['couplings'] = None
        self['lorentz'] = None
        self['particle_dict'] = {}
        self['interaction_dict'] = {}
        self['ref_dict_to0'] = {}
        self['ref_dict_to1'] = {}
        self['got_majoranas'] = None
        self['conserved_charge'] = set()
        self['coupling_orders'] = None

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'name':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a string" % \
                                                            type(value)
        elif name == 'particles':
            if not isinstance(value, ParticleList):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a ParticleList object" % \
                                                            type(value)
        elif name == 'interactions':
            if not isinstance(value, InteractionList):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a InteractionList object" % \
                                                            type(value)
        elif name == 'particle_dict':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                        type(value)
        elif name == 'interaction_dict':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % type(value)

        elif name == 'ref_dict_to0':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % type(value)
                    
        elif name == 'ref_dict_to1':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % type(value)

        elif name == 'got_majoranas':
            if not (isinstance(value, bool) or value == None):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a boolean" % type(value)

        elif name == 'conserved_charge':
            if not (isinstance(value, set)):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a set" % type(value)

        return True

    def get(self, name):
        """Get the value of the property name."""

        if (name == 'ref_dict_to0' or name == 'ref_dict_to1') and \
                                                                not self[name]:
            if self['interactions']:
                [self['ref_dict_to0'], self['ref_dict_to1']] = \
                            self['interactions'].generate_ref_dict()
                self['ref_dict_to0'].update(
                                self['particles'].generate_ref_dict())

        if (name == 'particle_dict') and not self[name]:
            if self['particles']:
                self['particle_dict'] = self['particles'].generate_dict()
            if self['interactions']:
                self['interactions'].synchronize_interactions_with_particles(\
                                                          self['particle_dict'])
                

        if (name == 'interaction_dict') and not self[name]:
            if self['interactions']:
                self['interaction_dict'] = self['interactions'].generate_dict()

        if (name == 'got_majoranas') and self[name] == None:
            if self['particles']:
                self['got_majoranas'] = self.check_majoranas()

        if (name == 'coupling_orders') and self[name] == None:
            if self['interactions']:
                self['coupling_orders'] = self.get_coupling_orders()

        return Model.__bases__[0].get(self, name) # call the mother routine

    def set(self, name, value):
        """Special set for particles and interactions - need to
        regenerate dictionaries."""

        if name == 'particles':
            # Ensure no doublets in particle list
            make_unique(value)
            # Reset dictionaries
            self['particle_dict'] = {}
            self['ref_dict_to0'] = {}
            self['got_majoranas'] = None

        if name == 'interactions':
            # Ensure no doublets in interaction list
            make_unique(value)
            # Reset dictionaries
            self['interaction_dict'] = {}
            self['ref_dict_to1'] = {}
            self['ref_dict_to0'] = {}
            self['got_majoranas'] = None
            self['coupling_orders'] = None

        Model.__bases__[0].set(self, name, value) # call the mother routine

        if name == 'particles':
            # Recreate particle_dict
            self.get('particle_dict')
            
    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['name', 'particles', 'parameters', 'interactions', 'couplings',
                'lorentz']

    def get_particle(self, id):
        """Return the particle corresponding to the id"""
        
        if id in self.get("particle_dict").keys():
            return self["particle_dict"][id]
        else:
            return None

    def get_interaction(self, id):
        """Return the interaction corresponding to the id"""


        if id in self.get("interaction_dict").keys():
            return self["interaction_dict"][id]
        else:
            return None

    def get_coupling_orders(self):
        """Determine the coupling orders of the model"""

        return set(sum([i.get('orders').keys() for i in \
                        self.get('interactions')], []))

    def check_majoranas(self):
        """Return True if there is fermion flow violation, False otherwise"""

        if any([part.is_fermion() and part.get('self_antipart') \
                for part in self.get('particles')]):
            return True

        # No Majorana particles, but may still be fermion flow
        # violating interactions
        for inter in self.get('interactions'):
            fermions = [p for p in inter.get('particles') if p.is_fermion()]
            for i in range(0, len(fermions), 2):
                if fermions[i].get('is_part') == \
                   fermions[i+1].get('is_part'):
                    # This is a fermion flow violating interaction
                    return True
        # No fermion flow violations
        return False

    def reset_dictionaries(self):
        """Reset all dictionaries and got_majoranas. This is necessary
        whenever the particle or interaction content has changed. If
        particles or interactions are set using the set routine, this
        is done automatically."""

        self['particle_dict'] = {}
        self['ref_dict_to0'] = {}
        self['got_majoranas'] = None
        self['interaction_dict'] = {}
        self['ref_dict_to1'] = {}
        self['ref_dict_to0'] = {}
        
    def pass_particles_name_in_mg_default(self):
        """Change the name of the particles such that all SM and MSSM particles
        follows the MG convention"""

        # Check that default name/antiname is not already use 
        def check_name_free(self, name):
            """ check if name is not use for a particle in the model if it is 
            raise an MadGraph5error"""
            part = self['particles'].find_name(name)
            if part: 
                error_text = \
                '%s particles with pdg code %s is in conflict with MG ' + \
                'convention name for particle %s.\n Use -modelname in order ' + \
                'to use the particles name defined in the model and not the ' + \
                'MadGraph convention'
                
                raise MadGraph5Error, error_text % \
                                     (part.get_name(), part.get_pdg_code(), pdg)                

        default = self.load_default_name()

        for pdg in default.keys():
            part = self.get_particle(pdg)
            if not part:
                continue
            antipart = self.get_particle(-pdg)
            name = part.get_name()
            if name != default[pdg]:
                check_name_free(self, default[pdg])
                if part.get('is_part'):
                    part.set('name', default[pdg])
                    if antipart:
                        antipart.set('name', default[pdg])
                    else:
                        part.set('antiname', default[pdg])                        
                else:
                    part.set('antiname', default[pdg])
                    if antipart:
                        antipart.set('antiname', default[pdg])
                
    def write_param_card(self):
        """Write out the param_card, and return as string."""
        
        import models.write_param_card as writter
        out = StringIO.StringIO() # it's suppose to be written in a file
        param = writter.ParamCardWriter(self)
        param.define_output_file(out)
        param.write_card()
        return out.getvalue()
        
    @ staticmethod
    def load_default_name():
        """ load the default for name convention """
        
        logger.info('Change particles name to pass to MG5 convention')    
        default = {}
        for line in open(os.path.join(MG5DIR, 'input', \
                                                 'particles_name_default.txt')):
            line = line.lstrip()
            if line.startswith('#'):
                continue
            
            args = line.split()
            if len(args) != 2:
                logger.warning('Invalid syntax in interface/default_name:\n %s' % line)
                continue
            default[int(args[0])] = args[1].lower()
        
        return default

################################################################################
# Class for Parameter / Coupling
################################################################################
class ModelVariable(object):
    """A Class for storing the information about coupling/ parameter"""
    
    def __init__(self, name, expression, type, depend=()):
        """Initialize a new parameter/coupling"""
        
        self.name = name
        self.expr = expression # python expression
        self.type = type # real/complex
        self.depend = depend # depend on some other parameter -tuple-
        self.value = None
    
    def __eq__(self, other):
        """Object with same name are identical, If the object is a string we check
        if the attribute name is equal to this string"""
        
        try:
            return other.name == self.name
        except:
            return other == self.name

class ParamCardVariable(ModelVariable):
    """ A class for storing the information linked to all the parameter 
    which should be define in the param_card.dat"""
    
    depend = ('external',)
    type = 'real'
    
    def __init__(self, name, value, lhablock, lhacode):
        """Initialize a new ParamCardVariable
        name: name of the variable
        value: default numerical value
        lhablock: name of the block in the param_card.dat
        lhacode: code associate to the variable
        """
        self.name = name
        self.value = value 
        self.lhablock = lhablock
        self.lhacode = lhacode


#===============================================================================
# Classes used in diagram generation and process definition:
#    Leg, Vertex, Diagram, Process
#===============================================================================

#===============================================================================
# Leg
#===============================================================================
class Leg(PhysicsObject):
    """Leg object: id (Particle), number, I/F state, flag from_group
    """

    def default_setup(self):
        """Default values for all properties"""

        self['id'] = 0
        self['number'] = 0
        # True = final, False = initial (boolean to save memory)
        self['state'] = True
        self['from_group'] = True

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name in ['id', 'number']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for leg id" % str(value)

        if name == 'state':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid leg state (True|False)" % \
                                                                    str(value)

        if name == 'from_group':
            if not isinstance(value, bool) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid boolean for leg flag from_group" % \
                                                                    str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id', 'number', 'state', 'from_group']

    def is_fermion(self, model):
        """Returns True if the particle corresponding to the leg is a
        fermion"""

        assert isinstance(model, Model), "%s is not a model" % str(model)

        return model.get('particle_dict')[self['id']].is_fermion()

    def is_incoming_fermion(self, model):
        """Returns True if leg is an incoming fermion, i.e., initial
        particle or final antiparticle"""

        assert isinstance(model, Model), "%s is not a model" % str(model)

        part = model.get('particle_dict')[self['id']]
        return part.is_fermion() and \
               (self.get('state') == False and part.get('is_part') or \
                self.get('state') == True and not part.get('is_part'))

    def is_outgoing_fermion(self, model):
        """Returns True if leg is an outgoing fermion, i.e., initial
        antiparticle or final particle"""

        assert isinstance(model, Model), "%s is not a model" % str(model)        
        
        part = model.get('particle_dict')[self['id']]
        return part.is_fermion() and \
               (self.get('state') == True and part.get('is_part') or \
                self.get('state') == False and not part.get('is_part'))

    # Make sure sort() sorts lists of legs according to 'number'
    def __lt__(self, other):
        return self['number'] < other['number']

#===============================================================================
# LegList
#===============================================================================
class LegList(PhysicsObjectList):
    """List of Leg objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid Leg for the list."""

        return isinstance(obj, Leg)

    # Helper methods for diagram generation

    def from_group_elements(self):
        """Return all elements which have 'from_group' True"""

        return filter(lambda leg: leg.get('from_group'), self)

    def minimum_one_from_group(self):
        """Return True if at least one element has 'from_group' True"""

        return len(self.from_group_elements()) > 0

    def minimum_two_from_group(self):
        """Return True if at least two elements have 'from_group' True"""

        return len(self.from_group_elements()) > 1

    def can_combine_to_1(self, ref_dict_to1):
        """If has at least one 'from_group' True and in ref_dict_to1,
           return the return list from ref_dict_to1, otherwise return False"""
        if self.minimum_one_from_group():
            return ref_dict_to1.has_key(tuple(sorted([leg.get('id') for leg in self])))
        else:
            return False

    def can_combine_to_0(self, ref_dict_to0, is_decay_chain=False):
        """If has at least two 'from_group' True and in ref_dict_to0,
        
        return the vertex (with id from ref_dict_to0), otherwise return None

        If is_decay_chain = True, we only allow clustering of the
        initial leg, since we want this to be the last wavefunction to
        be evaluated.
        """
        if is_decay_chain:
            # Special treatment - here we only allow combination to 0
            # if the initial leg (marked by from_group = None) is
            # unclustered, since we want this to stay until the very
            # end.
            return any(leg.get('from_group') == None for leg in self) and \
                   ref_dict_to0.has_key(tuple(sorted([leg.get('id') \
                                                      for leg in self])))

        if self.minimum_two_from_group():
            return ref_dict_to0.has_key(tuple(sorted([leg.get('id') for leg in self])))
        else:
            return False

    def get_outgoing_id_list(self, model):
        """Returns the list of ids corresponding to the leglist with
        all particles outgoing"""

        res = []

        assert isinstance(model, Model), "Error! model not model"


        for leg in self:
            if leg.get('state') == False:
                res.append(model.get('particle_dict')[leg.get('id')].get_anti_pdg_code())
            else:
                res.append(leg.get('id'))

        return res


#===============================================================================
# MultiLeg
#===============================================================================
class MultiLeg(PhysicsObject):
    """MultiLeg object: ids (Particle or particles), I/F state
    """

    def default_setup(self):
        """Default values for all properties"""

        self['ids'] = []
        self['state'] = True

    def filter(self, name, value):
        """Filter for valid multileg property values."""

        if name == 'ids':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for i in value:
                if not isinstance(i, int):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid list of integers" % str(value)

        if name == 'state':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid leg state (initial|final)" % \
                                                                    str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['ids', 'state']

#===============================================================================
# LegList
#===============================================================================
class MultiLegList(PhysicsObjectList):
    """List of MultiLeg objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid MultiLeg for the list."""

        return isinstance(obj, MultiLeg)

#===============================================================================
# Vertex
#===============================================================================
class Vertex(PhysicsObject):
    """Vertex: list of legs (ordered), id (Interaction)
    """
    
    sorted_keys = ['id', 'legs']
    
    def default_setup(self):
        """Default values for all properties"""

        self['id'] = 0
        self['legs'] = LegList()

    def filter(self, name, value):
        """Filter for valid vertex property values."""

        if name == 'id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for vertex id" % str(value)

        if name == 'legs':
            if not isinstance(value, LegList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid LegList object" % str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return self.sorted_keys  #['id', 'legs']

    def get_s_channel_id(self, model, ninitial):
        """Returns the id for the last leg as an outgoing
        s-channel. Returns 0 if leg is t-channel, or if identity
        vertex. Used to check for required and forbidden s-channel
        particles."""

        leg = self.get('legs')[-1]

        if ninitial == 1:
            # For one initial particle, all legs are s-channel
            # Only need to flip particle id if state is False
            if leg.get('state') == True:
                return leg.get('id')
            else:
                return model.get('particle_dict')[leg.get('id')].\
                       get_anti_pdg_code()

        # Number of initial particles is at least 2
        if self.get('id') == 0 or \
           leg.get('state') == False:
            # identity vertex or t-channel particle
            return 0

        # Check if the particle number is <= ninitial
        # In that case it comes from initial and we should switch direction
        if leg.get('number') > ninitial:
            return leg.get('id')
        else:
            return model.get('particle_dict')[leg.get('id')].\
                       get_anti_pdg_code()

        ## Check if the other legs are initial or final.
        ## If the latter, return leg id, if the former, return -leg id
        #if self.get('legs')[0].get('state') == True:
        #    return leg.get('id')
        #else:
        #    return model.get('particle_dict')[leg.get('id')].\
        #               get_anti_pdg_code()

#===============================================================================
# VertexList
#===============================================================================
class VertexList(PhysicsObjectList):
    """List of Vertex objects
    """

    orders = {}

    def is_valid_element(self, obj):
        """Test if object obj is a valid Vertex for the list."""

        return isinstance(obj, Vertex)

    def __init__(self, init_list=None, orders=None):
        """Creates a new list object, with an optional dictionary of
        coupling orders."""

        list.__init__(self)

        if init_list is not None:
            for object in init_list:
                self.append(object)

        if isinstance(orders, dict):
            self.orders = orders


#===============================================================================
# Diagram
#===============================================================================
class Diagram(PhysicsObject):
    """Diagram: list of vertices (ordered)
    """

    def default_setup(self):
        """Default values for all properties"""

        self['vertices'] = VertexList()
        self['orders'] = {}

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'vertices':
            if not isinstance(value, VertexList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid VertexList object" % str(value)

        if name == 'orders':
            Interaction.filter(Interaction(), 'orders', value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['vertices', 'orders']

    def nice_string(self):
        """Returns a nicely formatted string of the diagram content."""

        if self['vertices']:
            mystr = '('
            for vert in self['vertices']:
                mystr = mystr + '('
                for leg in vert['legs'][:-1]:
                    mystr = mystr + str(leg['number']) + '(%s)' % str(leg['id']) + ','

                if self['vertices'].index(vert) < len(self['vertices']) - 1:
                    # Do not want ">" in the last vertex
                    mystr = mystr[:-1] + '>'
                mystr = mystr + str(vert['legs'][-1]['number']) + '(%s)' % str(vert['legs'][-1]['id']) + ','
                mystr = mystr + 'id:' + str(vert['id']) + '),'
            mystr = mystr[:-1] + ')'
            mystr += " (%s)" % ",".join(["%s=%d" % (key, self['orders'][key]) \
                                        for key in self['orders'].keys()])
            return mystr
        else:
            return '()'

    def calculate_orders(self, model):
        """Calculate the actual coupling orders of this diagram"""

        coupling_orders = dict([(c, 0) for c in model.get('coupling_orders')])
        for vertex in self['vertices']:
            if vertex.get('id') == 0: continue
            couplings = model.get('interaction_dict')[vertex.get('id')].\
                        get('orders')
            for coupling in couplings:
                coupling_orders[coupling] += couplings[coupling]

        self.set('orders', coupling_orders)

#===============================================================================
# DiagramList
#===============================================================================
class DiagramList(PhysicsObjectList):
    """List of Diagram objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid Diagram for the list."""

        return isinstance(obj, Diagram)

    def nice_string(self, indent=0):
        """Returns a nicely formatted string"""
        mystr = " " * indent + str(len(self)) + ' diagrams:\n'
        for i, diag in enumerate(self):
            mystr = mystr + " " * indent + str(i+1) + "  " + \
                    diag.nice_string() + '\n'
        return mystr[:-1]


#===============================================================================
# Process
#===============================================================================
class Process(PhysicsObject):
    """Process: list of legs (ordered)
                dictionary of orders
                model
                process id
    """

    def default_setup(self):
        """Default values for all properties"""

        self['legs'] = LegList()
        self['orders'] = {}
        self['model'] = Model()
        # Optional number to identify the process
        self['id'] = 0
        self['uid'] = 0 # should be a uniq id number
        # Required s-channels are given as a list of id lists. Only
        # diagrams with all s-channels in any of the lists are
        # allowed. This enables generating e.g. Z/gamma as s-channel
        # propagators.
        self['required_s_channels'] = []
        self['forbidden_s_channels'] = []
        self['forbidden_particles'] = []
        self['is_decay_chain'] = False
        self['overall_orders'] = {}
        # Decay chain processes associated with this process
        self['decay_chains'] = ProcessList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'legs':
            if not isinstance(value, LegList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid LegList object" % str(value)

        if name in ['orders', 'overall_orders']:
            Interaction.filter(Interaction(), 'orders', value)

        if name == 'model':
            if not isinstance(value, Model):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Model object" % str(value)
        if name in ['id', 'uid']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Process %s %s is not an integer" % (name, repr(value))

        if name == 'required_s_channels':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for l in value:
                if not isinstance(l, list):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid list of lists" % str(value)
                for i in l:
                    if not isinstance(i, int):
                        raise self.PhysicsObjectError, \
                              "%s is not a valid list of integers" % str(l)
                    if i == 0:
                        raise self.PhysicsObjectError, \
                          "Not valid PDG code %d for s-channel particle" % i

        if name == 'forbidden_s_channels':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for i in value:
                if not isinstance(i, int):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid list of integers" % str(value)
                if i == 0:
                    raise self.PhysicsObjectError, \
                      "Not valid PDG code %d for s-channel particle" % str(value)

        if name == 'forbidden_particles':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for i in value:
                if not isinstance(i, int):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid list of integers" % str(value)
                if i <= 0:
                    raise self.PhysicsObjectError, \
                      "Forbidden particles should have a positive PDG code" % str(value)

        if name == 'is_decay_chain':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value)

        if name == 'decay_chains':
            if not isinstance(value, ProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessList" % str(value)

        return True

    def set(self, name, value):
        """Special set for forbidden particles - set to abs value."""

        if name == 'forbidden_particles':
            try:
                value = [abs(i) for i in value]
            except:
                pass

        if name == 'required_s_channels':
            # Required s-channels need to be a list of lists of ids
            if value and isinstance(value, list) and \
               not isinstance(value[0], list):
                value = [value]

        return super(Process, self).set(name, value) # call the mother routine

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['legs', 'orders', 'overall_orders', 'model', 'id',
                'required_s_channels', 'forbidden_s_channels',
                'forbidden_particles', 'is_decay_chain', 'decay_chains']

    def nice_string(self, indent=0):
        """Returns a nicely formated string about current process
        content"""

        mystr = " " * indent + "Process: "
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '> '
                # Add required s-channels
                if self['required_s_channels'] and \
                       self['required_s_channels'][0]:
                    mystr += "|".join([" ".join([self['model'].\
                                       get('particle_dict')[req_id].get_name() \
                                                for req_id in id_list]) \
                                    for id_list in self['required_s_channels']])
                    mystr = mystr + ' > '

            mystr = mystr + mypart.get_name() + ' '
            #mystr = mystr + '(%i) ' % leg['number']
            prevleg = leg

        # Add forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '/ '
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        if self['orders']:
            mystr = mystr + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('id') or self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '
        
        if not self.get('decay_chains'):
            return mystr

        for decay in self['decay_chains']:
            mystr = mystr + '\n' + \
                    decay.nice_string(indent + 2).replace('Process', 'Decay')

        return mystr

    def input_string(self):
        """Returns a process string corresponding to the input string
        in the command line interface."""

        mystr = ""
        prevleg = None

        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '> '
                # Add required s-channels
                if self['required_s_channels'] and \
                       self['required_s_channels'][0]:
                    mystr += "|".join([" ".join([self['model'].\
                                       get('particle_dict')[req_id].get_name() \
                                                for req_id in id_list]) \
                                    for id_list in self['required_s_channels']])
                    mystr = mystr + '> '

            mystr = mystr + mypart.get_name() + ' '
            #mystr = mystr + '(%i) ' % leg['number']
            prevleg = leg

        # Add forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '/ '
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        if self['orders']:
            mystr = mystr + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '
        
        if not self.get('decay_chains'):
            return mystr

        for decay in self['decay_chains']:
            paren1 = ''
            paren2 = ''
            if decay.get('decay_chains'):
                paren1 = '('
                paren2 = ')'
            mystr += ', ' + paren1 + decay.input_string() + paren2

        return mystr

    def base_string(self):
        """Returns a string containing only the basic process (w/o decays)."""

        mystr = ""
        prevleg = None
        for leg in self.get_legs_with_decays():
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '> '
            mystr = mystr + mypart.get_name() + ' '
            prevleg = leg

        # Remove last space
        return mystr[:-1]

    def shell_string(self, schannel=True, forbid=True, main=True):
        """Returns process as string with '~' -> 'x', '>' -> '_',
        '+' -> 'p' and '-' -> 'm', including process number,
        intermediate s-channels and forbidden particles"""

        mystr = ""
        if not self.get('is_decay_chain'):
            mystr += "%d_" % self['id']
        
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '_'
                # Add required s-channels
                if self['required_s_channels'] and \
                       self['required_s_channels'][0] and schannel:
                    mystr += "_or_".join(["".join([self['model'].\
                                       get('particle_dict')[req_id].get_name() \
                                                for req_id in id_list]) \
                                    for id_list in self['required_s_channels']])
                    mystr = mystr + '_'
            if mypart['is_part']:
                mystr = mystr + mypart['name']
            else:
                mystr = mystr + mypart['antiname']
            prevleg = leg

        # Check for forbidden particles
        if self['forbidden_particles'] and forbid:
            mystr = mystr + '_no_'
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name()

        # Replace '~' with 'x'
        mystr = mystr.replace('~', 'x')
        # Replace '+' with 'p'
        mystr = mystr.replace('+', 'p')
        # Replace '-' with 'm'
        mystr = mystr.replace('-', 'm')
        # Just to be safe, remove all spaces
        mystr = mystr.replace(' ', '')

        for decay in self.get('decay_chains'):
            mystr = mystr + "_" + decay.shell_string(schannel,forbid, main=False)

        # Too long name are problematic so restrict them to a maximal of 70 char
        if len(mystr) > 64 and main:
            if schannel and forbid:
                return self.shell_string(True, False, False)+ '_%s' % self['uid']
            elif schannel:
                return self.shell_string(False, False, False)+'_%s' % self['uid']
            else:
                return mystr[:64]+'_%s' % self['uid']
            
            
            

        return mystr

    def shell_string_v4(self):
        """Returns process as v4-compliant string with '~' -> 'x' and
        '>' -> '_'"""

        mystr = "%d_" % self['id']
        prevleg = None
        for leg in self.get_legs_with_decays():
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '_'
            if mypart['is_part']:
                mystr = mystr + mypart['name']
            else:
                mystr = mystr + mypart['antiname']
            prevleg = leg

        # Replace '~' with 'x'
        mystr = mystr.replace('~', 'x')
        # Just to be safe, remove all spaces
        mystr = mystr.replace(' ', '')

        return mystr

    # Helper functions

    def get_ninitial(self):
        """Gives number of initial state particles"""

        return len(filter(lambda leg: leg.get('state') == False,
                           self.get('legs')))

    def get_initial_ids(self):
        """Gives the pdg codes for initial state particles"""

        return [leg.get('id') for leg in \
                filter(lambda leg: leg.get('state') == False,
                       self.get('legs'))]

    def get_initial_pdg(self, number):
        """Return the pdg codes for initial state particles for beam number"""

        return filter(lambda leg: leg.get('state') == False and\
                       leg.get('number') == number,
                       self.get('legs'))[0].get('id')

    def get_final_legs(self):
        """Gives the pdg codes for initial state particles"""

        return filter(lambda leg: leg.get('state') == True,
                       self.get('legs'))
    
    def get_legs_with_decays(self):
        """Return process with all decay chains substituted in."""

        legs = copy.deepcopy(self.get('legs'))
        if self.get('is_decay_chain'):
            legs.pop(0)
        ileg = 0
        for decay in self.get('decay_chains'):
            while legs[ileg].get('state') == False or \
                      legs[ileg].get('id') != decay.get('legs')[0].get('id'):
                ileg = ileg + 1
            decay_legs = decay.get_legs_with_decays()
            legs = legs[:ileg] + decay_legs + legs[ileg+1:]
            ileg = ileg + len(decay_legs)

        for ileg, leg in enumerate(legs):
            leg.set('number', ileg + 1)
            
        return LegList(legs)

    def compare_for_sort(self, other):
        """Sorting routine which allows to sort processes for
        comparison. Compare only process id and legs."""

        if self['id'] != other['id']:
            return self['id'] - other['id']

        initlegs = sorted([l.get('id') for l in \
                           filter(lambda leg: not leg.get('state'),
                                  self['legs'])])
        otherinitlegs = sorted([l.get('id') for l in \
                           filter(lambda leg: not leg.get('state'),
                                  self['legs'])])

        if len(initlegs) != len(otherinitlegs):
            return len(initlegs) - len(otherinitlegs)
                
        for leg, otherleg in zip(initlegs, otherinitlegs):
            if leg != otherleg:
                return leg - otherleg
        
        legs = sorted([l.get('id') for l in \
                       filter(lambda leg: leg.get('state'),
                              self.get_legs_with_decays())])
        otherlegs = sorted([l.get('id') for l in \
                       filter(lambda leg: leg.get('state'),
                              other.get_legs_with_decays())])

        if len(legs) != len(otherlegs):
            return len(legs) - len(otherlegs)
                
        for leg, otherleg in zip(legs, otherlegs):
            if leg != otherleg:
                return leg - otherleg
        
        return 0
        
    def __eq__(self, other):
        """Overloading the equality operator, so that only comparison
        of process id and legs is being done, using compare_for_sort."""

        if not isinstance(other, Process):
            return False

        return self.compare_for_sort(other) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

#===============================================================================
# ProcessList
#===============================================================================
class ProcessList(PhysicsObjectList):
    """List of Process objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid Process for the list."""

        return isinstance(obj, Process)

#===============================================================================
# ProcessDefinition
#===============================================================================
class ProcessDefinition(Process):
    """ProcessDefinition: list of multilegs (ordered)
                          dictionary of orders
                          model
                          process id
    """

    def default_setup(self):
        """Default values for all properties"""

        super(ProcessDefinition, self).default_setup()

        self['legs'] = MultiLegList()
        # Decay chain processes associated with this process
        self['decay_chains'] = ProcessDefinitionList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'legs':
            if not isinstance(value, MultiLegList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid MultiLegList object" % str(value)
        elif name == 'decay_chains':
            if not isinstance(value, ProcessDefinitionList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessDefinitionList" % str(value)

        else:
            return super(ProcessDefinition, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return super(ProcessDefinition, self).get_sorted_keys()

    def nice_string(self, indent=0):
        """Returns a nicely formated string about current process
        content"""

        mystr = " " * indent + "Process: "
        prevleg = None
        for leg in self['legs']:
            myparts = \
                   "/".join([self['model'].get('particle_dict')[id].get_name() \
                             for id in leg.get('ids')])
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by ">"
                mystr = mystr + '> '
                # Add required s-channels
                if self['required_s_channels'] and \
                       self['required_s_channels'][0]:
                    mystr += "|".join([" ".join([self['model'].\
                                       get('particle_dict')[req_id].get_name() \
                                                for req_id in id_list]) \
                                    for id_list in self['required_s_channels']])
                    mystr = mystr + '> '

            mystr = mystr + myparts + ' '
            #mystr = mystr + '(%i) ' % leg['number']
            prevleg = leg

        # Add forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '/ '
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        if self['orders']:
            mystr = mystr + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('id') or self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '
        
        if not self.get('decay_chains'):
            return mystr

        for decay in self['decay_chains']:
            mystr = mystr + '\n' + \
                    decay.nice_string(indent + 2).replace('Process', 'Decay')

        return mystr

    def __eq__(self, other):
        """Overloading the equality operator, so that only comparison
        of process id and legs is being done, using compare_for_sort."""

        return super(Process, self).__eq__(other)

#===============================================================================
# ProcessDefinitionList
#===============================================================================
class ProcessDefinitionList(PhysicsObjectList):
    """List of ProcessDefinition objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid ProcessDefinition for the list."""

        return isinstance(obj, ProcessDefinition)

#===============================================================================
# Global helper functions
#===============================================================================

def make_unique(doubletlist):
    """Make sure there are no doublets in the list doubletlist.
    Note that this is a slow implementation, so don't use if speed 
    is needed"""

    assert isinstance(doubletlist, list), \
           "Argument to make_unique must be list"
    

    uniquelist = []
    for elem in doubletlist:
        if elem not in uniquelist:
            uniquelist.append(elem)

    doubletlist[:] = uniquelist[:]
