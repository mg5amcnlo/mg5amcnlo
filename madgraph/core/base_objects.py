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
import copy

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
    particle object is a particle or an antiparticle."""

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

        if name is 'is_part':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "is_part tag %s is not a boolean" % repr(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['name', 'antiname', 'spin', 'color',
                'charge', 'mass', 'width', 'pdg_code',
                'texname', 'antitexname', 'line', 'propagating', 'is_part']

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
        and antiname. If a match is found, return the a copy of the corresponding
        particle (first one in the list), with the is_part flag set
        accordingly. None otherwise."""

        if not Particle.filter(Particle(), 'name', name):
            raise self.PhysicsObjectError, \
                "%s is not a valid particle name" % str(name)

        for part in self:
            mypart = copy.copy(part)
            if part.get('name') == name:
                mypart.set('is_part', True)
                return mypart
            elif part.get('antiname') == name:
                mypart.set('is_part', False)
                return mypart

        return None


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

        self['particles'] = []
        self['color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        self['orders'] = {}

    def filter(self, name, value):
        """Filter for valid interaction property values."""

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

    def __permutate(self, seq):
        """permutate a sequence and return a list of all permutations"""
        if not seq:
            return [seq] # is an empty sequence
        else:
            temp = []
            for k in range(len(seq)):
                part = seq[:k] + seq[k + 1:]
                for m in self.__permutate(part):
                    temp.append(seq[k:k + 1] + m)
            return temp

    def generate_dict_entries(self, ref_dict):
        """Add entries corresponding to the current interactions to 
        the reference dictionary (for n>0 and n-1>1)"""

        # Create a list of particle ids (pdg code)
        part_list = []
        for part in self['particles']:
            if part['is_part']:
                part_list.append(part['pdg_code'])
            else:
                part_list.append(-part['pdg_code'])

        # Create n>0 entries
        for permut in self.__permutate(part_list):
            permut_tuple = tuple(permut)
            if permut_tuple in ref_dict.keys():
                if None not in ref_dict[permut_tuple]:
                    ref_dict[permut_tuple].append(None)
            else:
                ref_dict[permut_tuple] = [None]

        # Create n-1>1 entries Comment by Johan: Note that, in the n-1
        # > 1 dictionnary, the 1 entry should have opposite sign as
        # compared to the n > 0 dictionnary, since this should replace
        # the n-1 particles. I prefer to keep track of the sign
        # (part/antipart) here rather than in the diagram generation.
        for part in part_list:
            short_part_list = copy.copy(part_list)
            short_part_list.remove(part)
            for permut in self.__permutate(short_part_list):
                permut_tuple = tuple(permut)
                if permut_tuple in ref_dict.keys():
                    if part not in  ref_dict[permut_tuple]:
                        ref_dict[permut_tuple].append(part)
                else:
                    ref_dict[permut_tuple] = [part]


#===============================================================================
# InteractionList
#===============================================================================
class InteractionList(PhysicsObjectList):
    """A class to store lists of interactionss."""

    def is_valid_element(self, obj):
        """Test if object obj is a valid Interaction for the list."""

        return isinstance(obj, Interaction)

    def generate_ref_dict(self):
        """Generate the reference dictionary from interaction list."""

        ref_dict = {}

        for inter in self:
            inter.generate_dict_entries(ref_dict)

        return ref_dict



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
        self['state'] = 'initial'
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
           if not isinstance(value, bool):
               raise self.PhysicsObjectError, \
                       "%s is not a valid boolean for leg flagr from_group" % \
                                                                   str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id', 'number', 'state', 'from_group']

#===============================================================================
# LegList
#===============================================================================
class LegList(PhysicsObjectList):
    """List of Leg objects
    """

    def is_valid_element(self, obj):
       """Test if object obj is a valid Leg for the list."""

       return isinstance(obj, Leg)

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


#===============================================================================
# VertexList
#===============================================================================
class VertexList(PhysicsObjectList):
    """List of Vertex objects
    """

    def is_valid_element(self, obj):
       """Test if object obj is a valid Vertex for the list."""

       return isinstance(obj, Vertex)


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

#===============================================================================
# DiagramList
#===============================================================================
class DiagramList(PhysicsObjectList):
    """List of Diagram objects
    """

    def is_valid_element(self, obj):
       """Test if object obj is a valid Diagram for the list."""

       return isinstance(obj, Diagram)


#===============================================================================
# Amplitude
#===============================================================================
class Amplitude(PhysicsObject):
    """Amplitude: list of diagrams (ordered)
    """

    def default_setup(self):
        """Default values for all properties"""

        self['diagrams'] = DiagramList()

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'diagrams':
            if not isinstance(value, DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList object" % str(value)

        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['diagrams']


