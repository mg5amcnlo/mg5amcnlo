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
import re

import madgraph.core.color_algebra as color

logger = logging.getLogger('base_objects')

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

        if not isinstance(init_dict, dict):
            raise self.PhysicsObjectError, \
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
        except:
            self.is_valid_prop(name) #raise the correct error


    def default_setup(self):
        """Function called to create and setup default values for all object
        properties"""
        pass

    def is_valid_prop(self, name):
        """Check if a given property name is valid"""

        if not isinstance(name, str):
            raise self.PhysicsObjectError, \
                "Property name %s is not a string" % repr(name)

        if name not in self.keys():
            raise self.PhysicsObjectError, \
                        "%s is not a valid property for this object" % name

        return True

    def __getitem__(self, name):
        """ Get the value of the property name."""

        if self.is_valid_prop(name):
            return dict.__getitem__(self, name)

    def get(self, name):
        """Get the value of the property name."""

        #if self.is_valid_prop(name): #done automaticaly in __getitem__
        return self[name]

    def set(self, name, value):
        """Set the value of the property name. First check if value
        is a valid value for the considered property. Return True if the
        value has been correctly set, False otherwise."""

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

    def def_model(self, model):
        """ 
        make a link between the  present object and the associate model 
        """

        if isinstance(model, Model):
            self._def_model(model)
        else:
            raise self.PhysicsObjectError(' try to assign a non model obect')

    def _def_model(self, model):
        """
        make a link between the  present object and the associate model 
        no class verification
        """
        self.model = model

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
        if not self.is_valid_element(object):
            raise self.PhysicsObjectListError, \
                "Object %s is not a valid object for the current list" % \
                                                             repr(object)
        else:
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
            if value not in ['dashed', 'straight', 'wavy', 'curly', 'double']:
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

        return ['name', 'antiname', 'spin', 'color',
                'charge', 'mass', 'width', 'pdg_code',
                'texname', 'antitexname', 'line', 'propagating',
                'is_part', 'self_antipart']

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

        if not self['is_part'] and self['color'] in [3, 6]:
            return -self['color']
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
        if spin == 2:
            # Spinor
            return [ -1, 1 ]
        if spin == 3 and self.get('mass').lower() == 'zero':
            # Massless vector
            return [ -1, 1 ]
        if spin == 3:
            # Massive vector
            return [ -1, 0, 1 ]

        raise self.PhysicsObjectError, \
              "No helicity state assignment for spin %d particles" % spin

    def is_fermion(self):
        """Returns True if this is a fermion, False if boson"""

        return self['spin'] % 2 == 0

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

        if not isinstance(name, str):
            raise self.PhysicsObjectError, \
                "%s is not a valid string" % str(name)

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
            ref_dict_to0[(part.get_pdg_code(), part.get_anti_pdg_code())] = 0
            ref_dict_to0[(part.get_anti_pdg_code(), part.get_pdg_code())] = 0

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

    def default_setup(self):
        """Default values for all properties"""

        self['id'] = 0
        self['particles'] = []
        self['color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        self['orders'] = {}

    def __init__(self, init_dict={}):
        """Creates a new Interaction object. Since there are special
        checks for the \'couplings\' variable, it needs to be set
        last."""

        super(Interaction, self).__init__(init_dict)

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
                        "%s is not a valid string" % str(mystr)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id', 'particles', 'color', 'lorentz',
                'couplings', 'orders']

    def generate_dict_entries(self, ref_dict_to0, ref_dict_to1):
        """Add entries corresponding to the current interactions to 
        the reference dictionaries (for n>0 and n-1>1)"""

        # Create n>0 entries. Format is (p1,p2,p3,...):interaction_id.
        # We are interested in the unordered list, so use sorted()

        pdg_tuple = tuple(sorted([p.get_pdg_code() for p in self['particles']]))
        if pdg_tuple not in ref_dict_to0.keys():
            ref_dict_to0[pdg_tuple] = self['id']

        # Create n-1>1 entries. Note that, in the n-1 > 1 dictionary,
        # the n-1 entries should have opposite sign as compared to
        # interaction, since the interaction has outgoing particles,
        # while in the dictionary we treat the n-1 particles as
        # incoming

        for part in self['particles']:

            # Create a list w/o part
            short_part_list = copy.copy(self['particles'])
            short_part_list.remove(part)

            # We are interested in the unordered list, so use sorted()
            pdg_tuple = tuple(sorted([p.get_pdg_code() for p in short_part_list]))
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

#===============================================================================
# Model
#===============================================================================
class Model(PhysicsObject):
    """A class to store all the model information."""

    def default_setup(self):

        self['name'] = ""
        self['particles'] = ParticleList()
        self['parameters'] = None
        self['interactions'] = InteractionList()
        self['couplings'] = None
        self['lorentz'] = None
        self['particle_dict'] = {}
        self['interaction_dict'] = {}
        self['ref_dict_to0'] = {}
        self['ref_dict_to1'] = {}
        self['got_majoranas'] = None

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'name':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a string" % \
                                                            type(value)
        if name == 'particles':
            if not isinstance(value, ParticleList):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a ParticleList object" % \
                                                            type(value)
        if name == 'interactions':
            if not isinstance(value, InteractionList):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a InteractionList object" % \
                                                            type(value)
        if name == 'particle_dict':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                        type(value)
        if name == 'interaction_dict':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                        type(value)

        if name == 'ref_dict_to0':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                        type(value)
        if name == 'ref_dict_to1':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                        type(value)

        if name == 'got_majoranas':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a boolean" % \
                                                        type(value)

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

        if (name == 'interaction_dict') and not self[name]:
            if self['interactions']:
                self['interaction_dict'] = self['interactions'].generate_dict()

        if (name == 'got_majoranas') and self[name] == None:
            if self['particles']:
                self['got_majoranas'] = self.check_majoranas()

        return Model.__bases__[0].get(self, name) # call the mother routine

    def set(self, name, value):
        """Special set for particles and interactions - need to
        regenerate dictionaries."""

        if name == 'particles':
            self['particle_dict'] = {}
            self['ref_dict_to0'] = {}
            self['got_majoranas'] = None

        if name == 'interactions':
            self['interaction_dict'] = {}
            self['ref_dict_to1'] = {}
            self['ref_dict_to0'] = {}

        Model.__bases__[0].set(self, name, value) # call the mother routine

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

    def check_majoranas(self):
        """Return True if there are Majorana fermions, False otherwise"""

        return any([part.is_fermion() and part.get('self_antipart') \
                    for part in self.get('particles')])

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
        self['state'] = 'final'
        self['from_group'] = True

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name in ['id', 'number']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for leg id" % str(value)

        if name == 'state':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for leg state" % \
                                                                    str(value)
            if value not in ['initial', 'final']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid leg state (initial|final)" % \
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

        if not isinstance(model, Model):
            raise self.PhysicsObjectError, \
                  "%s is not a model" % str(model)

        return model.get('particle_dict')[self['id']].is_fermion()

    def is_incoming_fermion(self, model):
        """Returns True if leg is an incoming fermion, i.e., initial
        particle or final antiparticle"""

        if not isinstance(model, Model):
            raise self.PhysicsObjectError, \
                  "%s is not a model" % str(model)

        part = model.get('particle_dict')[self['id']]
        return part.is_fermion() and \
               (self.get('state') == 'initial' and part.get('is_part') or \
                self.get('state') == 'final' and not part.get('is_part'))

    def is_outgoing_fermion(self, model):
        """Returns True if leg is an outgoing fermion, i.e., initial
        antiparticle or final particle"""

        if not isinstance(model, Model):
            raise self.PhysicsObjectError, \
                  "%s is not a model" % str(model)

        part = model.get('particle_dict')[self['id']]
        return part.is_fermion() and \
               (self.get('state') == 'final' and part.get('is_part') or \
                self.get('state') == 'initial' and not part.get('is_part'))

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

    def can_combine_to_0(self, ref_dict_to0, is_decay_chain = False):
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

        if not isinstance(model, Model):
            print "Error! model not model"
            return res

        for leg in self:
            if leg.get('state') == 'initial':
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
        self['state'] = 'final'

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
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for leg state" % \
                                                                    str(value)
            if value not in ['initial', 'final']:
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

        return ['id', 'legs']

    def get_s_channel_id(self, model, ninitial):
        """Returns the id for the last leg as an outgoing
        s-channel. Returns 0 if leg is t-channel, or if identity
        vertex. Used to check for required and forbidden s-channel
        particles."""

        leg = self.get('legs')[-1]

        if ninitial == 1:
            # For one initial particle, all legs are s-channel
            # Only need to flip particle id if state is 'initial'
            if leg.get('state') == 'final':
                return leg.get('id')
            else:
                return model.get('particle_dict')[leg.get('id')].\
                       get_anti_pdg_code()

        # Number of initial particles is at least 2
        if self.get('id') == 0 or \
           leg.get('state') == 'initial':
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
        #if self.get('legs')[0].get('state') == 'final':
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

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'vertices':
            if not isinstance(value, VertexList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid VertexList object" % str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['vertices']

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
            return mystr
        else:
            return '()'

#===============================================================================
# DiagramList
#===============================================================================
class DiagramList(PhysicsObjectList):
    """List of Diagram objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid Diagram for the list."""

        return isinstance(obj, Diagram)

    def nice_string(self):
        """Returns a nicely formatted string"""
        mystr = str(len(self)) + ' diagrams:\n'
        for diag in self:
            mystr = mystr + "  " + diag.nice_string() + '\n'
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
        self['required_s_channels'] = []
        self['forbidden_s_channels'] = []
        self['forbidden_particles'] = []
        self['is_decay_chain'] = False
        # Decay chain processes associated with this process
        self['decay_chains'] = ProcessList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'legs':
            if not isinstance(value, LegList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid LegList object" % str(value)
        if name == 'orders':
            Interaction.filter(Interaction(), 'orders', value)

        if name == 'model':
            if not isinstance(value, Model):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Model object" % str(value)
        if name == 'id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Process id %s is not an integer" % repr(value)

        if name in ['required_s_channels',
                    'forbidden_s_channels']:
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

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['legs', 'orders', 'model', 'id',
                'required_s_channels', 'forbidden_s_channels',
                'forbidden_particles', 'is_decay_chain', 'decay_chains']

    def nice_string(self, indent = 0):
        """Returns a nicely formated string about current process
        content"""

        mystr = " " * indent + "Process: "
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == 'initial' \
                   and leg['state'] == 'final':
                # Separate initial and final legs by ">"
                mystr = mystr + '> '
                # Add required s-channels
                if self['required_s_channels']:
                    for req_id in self['required_s_channels']:
                        reqpart = self['model'].get('particle_dict')[req_id]
                        mystr = mystr + reqpart.get_name() + ' '
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
            mystr = mystr[:-1] + "\n" + " " * indent
            mystr = mystr + 'Orders: '
            mystr = mystr + ", ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '

        # Remove last space
        mystr = mystr[:-1]

        for decay in self['decay_chains']:
            mystr = mystr + '\n' + \
                    decay.nice_string(indent + 2).replace('Process', 'Decay')

        return mystr

    def shell_string(self):
        """Returns process as string with '~' -> 'x' and '>' -> '_',
        including process number, intermediate s-channels and forbidden
        particles"""

        mystr = "%d_" % self['id']
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == 'initial' \
                   and leg['state'] == 'final':
                # Separate initial and final legs by ">"
                mystr = mystr + '_'
                # Add required s-channels
                if self['required_s_channels']:
                    for req_id in self['required_s_channels']:
                        reqpart = self['model'].get('particle_dict')[req_id]
                        mystr = mystr + reqpart.get_name()
                    mystr = mystr + '_'
            if mypart['is_part']:
                mystr = mystr + mypart['name']
            else:
                mystr = mystr + mypart['antiname']
            prevleg = leg

        # Check for forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '-'
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name()

        # Replace '~' with 'x'
        mystr = mystr.replace('~', 'x')
        # Just to be safe, remove all spaces
        mystr = mystr.replace(' ', '')

        for decay in self.get('decay_chains'):
            mystr = mystr + decay.shell_string().replace("%d_" % decay.get('id'),
                                                        "_", 1)
        return mystr

    def shell_string_v4(self):
        """Returns process as v4-compliant string with '~' -> 'x' and
        '>' -> '_'"""

        mystr = "%d_" % self['id']
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == 'initial' \
                   and leg['state'] == 'final':
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

        for decay in self.get('decay_chains'):
            mystr = mystr + decay.shell_string_v4().replace("%d_" % decay.get('id'),
                                                        "_", 1)

        return mystr

    # Helper functions

    def get_ninitial(self):
        """Gives number of initial state particles"""

        return len(filter(lambda leg: leg.get('state') == 'initial',
                           self.get('legs')))

    def get_initial_ids(self):
        """Gives the pdg codes for initial state particles"""

        return [leg.get('id') for leg in \
                filter(lambda leg: leg.get('state') == 'initial',
                       self.get('legs'))]

    def get_final_legs(self):
        """Gives the pdg codes for initial state particles"""

        return filter(lambda leg: leg.get('state') == 'final',
                       self.get('legs'))

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

#===============================================================================
# ProcessDefinitionList
#===============================================================================
class ProcessDefinitionList(PhysicsObjectList):
    """List of ProcessDefinition objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid ProcessDefinition for the list."""

        return isinstance(obj, ProcessDefinition)
