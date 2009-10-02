##############################################################################
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
##############################################################################

"""Definitions of all basic objects used in the core code: particle, 
vertex, ..."""

import re

##############################################################################
##  Particle
##############################################################################

class Particle(dict):
    """The particle object containing the whole set of information required to
    univocally characterize a given type of physical particle: name, spin, 
    color, mass, width, charge,..."""

    prop_list = ['name',
                 'antiname',
                 'spin',
                 'color',
                 'charge',
                 'mass',
                 'width',
                 'pdg_code',
                 'texname',
                 'antitexname',
                 'line',
                 'propagating']

    class ParticleError(Exception):
        """Exception raised if an error occurs in the definition
        of a particle"""
        pass

    def __init__(self, init_dict=None):
        """Creates a new particle object. If no argument is passed, assigns 
        dummy values to all properties. If a dictionary is given, tries to 
        use it to give values to properties."""

        dict.__init__(self)

        self.set('name', 'none')
        self.set('antiname', 'none')
        self.set('spin', 1)
        self.set('color', 1)
        self.set('charge', 1.)
        self.set('mass', 'zero')
        self.set('width', 'zero')
        self.set('pdg_code', 0)
        self.set('texname', 'none')
        self.set('antitexname', 'none')
        self.set('line', 'dashed')
        self.set('propagating', True)

        if init_dict is not None:

            if not isinstance(init_dict, dict):
                raise self.ParticleError, \
                    "Argument %s is not a dictionary" % repr(init_dict)

            for item in init_dict.keys():
                if item in self.prop_list:
                    self.set(item, init_dict[item])
                else:
                    raise self.ParticleError, \
                        "Key %s is not a valid particle property" % item

    def get(self, name):
        """Get the value of the property name."""

        if not isinstance(name, str):
            raise self.ParticleError, \
                "Property name %s is not a string" % repr(name)

        if name not in self.prop_list:
            raise self.ParticleError, \
                        "%s is not a valid particle property" % name

        return self[name]

    def set(self, name, value):
        """Set the value of the property name. First check if value
        is a valid value for the considered property. Return True if the
        value has been correctly set."""

        if not isinstance(name, str):
            raise self.ParticleError, \
                "Property name %s is not a string" % repr(name)

        if name not in self.prop_list:
            raise self.ParticleError, \
                        "%s is not a valid particle property" % name

        if name in ['name', 'antiname']:
            # Must start with a letter, followed by letters,  digits,
            # - and + only
            p = re.compile('\A[a-zA-Z]+[\w~\-\+]*\Z')
            if not p.match(value):
                raise self.ParticleError, \
                        "%s is not a valid particle name" % value

        if name is 'spin':
            if not isinstance(value, int):
                raise self.ParticleError, \
                    "Spin %s is not an integer" % repr(value)
            if value < 1 or value > 5:
                 raise self.ParticleError, \
                    "Spin %i is smaller than one" % value

        if name is 'color':
            if not isinstance(value, int):
                raise self.ParticleError, \
                    "Color %s is not an integer" % repr(value)
            if abs(value) not in [1, 3, 6, 8]:
                 raise self.ParticleError, \
                    "Color %i is not valid" % value

        if name in ['mass', 'width']:
            # Must start with a letter, followed by letters, digits or _
            p = re.compile('\A[a-zA-Z]+[\w\_]*\Z')
            if not p.match(value):
                raise self.ParticleError, \
                        "%s is not a valid name for mass/width variable" % \
                        value

        if name is 'pdg_code':
            if not isinstance(value, int):
                raise self.ParticleError, \
                    "PDG code %s is not an integer" % repr(value)
            if value < 0:
                 raise self.ParticleError, \
                    "PDG code %i is smaller than one" % value

        if name is 'line':
            if not isinstance(value, str):
                raise self.ParticleError, \
                    "Line type %s is not a string" % repr(value)
            if value not in ['dashed', 'straight', 'wavy', 'curly']:
                 raise self.ParticleError, \
                    "Line type %s is unknown" % value

        if name is 'charge':
            if not isinstance(value, float):
                raise self.ParticleError, \
                    "Charge %s is not a float" % repr(value)

        if name is 'propagating':
            if not isinstance(value, bool):
                raise self.ParticleError, \
                    "Propagating tag %s is not a boolean" % repr(value)

        self[name] = value

        return True

    def __str__(self):
        """String representation of the Particle object. Outputs valid Python 
        with improved format."""

        mystr = '{\n'

        for prop in self.prop_list:
            if isinstance(self[prop], str):
                mystr = mystr + '    \'' + prop + '\': \'' + self[prop] + '\',\n'
            elif isinstance(self[prop], float):
                mystr = mystr + '    \'' + prop + '\': %.2f,\n' % self[prop]
            else:
                mystr = mystr + '    \'' + prop + '\': ' + repr(self[prop]) + ',\n'
        mystr = mystr.rstrip(',\n')
        mystr = mystr + '\n}'

        return mystr



##############################################################################
##  ParticleList
##############################################################################

class ParticleList(list):
    """A class to store lists of particles."""

    class ParticleListError(Exception):
        """Exception raised if an error occurs in the definition
        of a particle list."""
        pass

    def __init__(self, init_list=None):
        """Creates a new particle list object. If a list of particle is given,
        add them."""

        list.__init__(self)

        if init_list is not None:
            for part in init_list:
                if not isinstance(part, Particle):
                    raise self.ParticleListError, \
                        "Object %s is not a particle" % repr(part)
                else:
                    self.append(part)

    def __str__(self):
        """String representation of the particle list object. 
        Outputs valid Python with improved format."""

        mystr = '['

        for part in self:
            mystr = mystr + str(part) + ',\n'

        mystr = mystr.rstrip(',\n')

        return mystr + ']'

