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
interaction, model, ..."""

import logging
import re

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

    def get(self, name):
        """Get the value of the property name."""

        if self.is_valid_prop(name):
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
                logging.warning("Property " + name + " cannot be changed:" + \
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
        if not self._is_valid_element(object):
            raise self.PhysicsObjectListError, \
                "Object %s is not a valid object for the current list" % \
                                                             repr(object)
        else:
            list.append(self, object)

    def _is_valid_element(self, obj):
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
    color, mass, width, charge,..."""

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
            if value < 0:
                 raise self.PhysicsObjectError, \
                    "PDG code %i is smaller than one" % value

        if name is 'line':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Line type %s is not a string" % repr(value)
            if value not in ['dashed', 'straight', 'wavy', 'curly']:
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

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['name', 'antiname', 'spin', 'color',
                'charge', 'mass', 'width', 'pdg_code',
                'texname', 'antitexname', 'line', 'propagating']

#===============================================================================
# ParticleList
#===============================================================================
class ParticleList(PhysicsObjectList):
    """A class to store lists of particles."""

    def _is_valid_element(self, obj):
        """Test if object obj is a valid Particle for the list."""
        return isinstance(obj, Particle)

    def find_name(self, name):
        """Try to find a particle with the given name. Check both name
        and antiname. If a match is found, return the corresponding
        particle (first one in the list), None otherwise."""

        if not Particle.filter(Particle(), 'name', name):
            raise self.PhysicsObjectError, \
                "%s is not a valid particle name" % str(name)

        for part in self:
            if part.get('name') == name or \
                part.get('antiname') == name:
                return part

        return None


#===============================================================================
# Interaction
#===============================================================================
class Interaction(PhysicsObject):
    """The interaction object containing the whole set of information 
    required to univocally characterize a given type of physical interaction: 
    
    particles: a list of particle names
    color: a list of string describing all the color structures involved
    lorentz: a list of variable names describing all the Lorentz structure
             involved
    couplings: dictionary listing coupling variable names. The key is a
               2-tuple of integers referring to color and Lorentz structures
    orders: dictionary listing order names (as keys) with their value
    """

    def default_setup(self):
        """Default values for all properties"""

        self['particles'] = []
        self['color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        self['orders'] = {}

    def filter(self, name, value):
        """Filter for valid interaction property values."""

        if name == 'particles':
            #Should be a list of valid particle names
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of particle names" % str(value)
            for part in value:
                if not Particle.filter(Particle(), 'name', part):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid particle name" % str(part)

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

        if name in ['color', 'lorentz']:
            #Should be a list of strings
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

            if len(value) != len(self['color']) * len(self['lorentz']):
                raise self.PhysicsObjectError, \
                        "Dictionary " + str(value) + \
                        " for couplings has not the right number of entry"

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
                if key[0] < 0 or key[1] < 0 or \
                   key[0] >= len(self['color']) or key[1] >= \
                                                    len(self['lorentz']):
                    raise self.PhysicsObjectError, \
                        "%s is not a tuple with valid range" % str(key)
                if not isinstance(value[key], str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['particles', 'color', 'lorentz', 'couplings', 'orders']


#===============================================================================
# InteractionList
#===============================================================================
class InteractionList(PhysicsObjectList):
    """A class to store lists of interactionss."""

    def _is_valid_element(self, obj):
       """Test if object obj is a valid Interaction for the list."""

       return isinstance(obj, Interaction)

#===============================================================================
# Model
#===============================================================================
class Model(PhysicsObject):
    """A class to store all the model information."""

    def default_setup(self):

        self['particles'] = ParticleList()
        self['parameters'] = None
        self['interactions'] = InteractionList()
        self['couplings'] = None
        self['lorentz'] = None

    def filter(self, name, value):
        """Filter for model property values"""

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

        return True
