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

##############################################################################
##  Particle
##############################################################################

class Particle(dict):
    """The particle object containing the whole set of information required to
    univokely characterize a given type of physical particle: name, spin, 
    color, mass, width, charge,..."""

    _prop_list = ['name',
                 'antiname',
                 'spin',
                 'color',
                 'charge',
                 'mass',
                 'width',
                 'pdg_code',
                 'texname',
                 'antitexname',
                 'line']

    def ParticleError(Exception):
        """Exception raised if an error occurs in the definition
        of a particle"""
        pass

    def __init__(self, init_dict=None):
        """Creates a new particle object. If no argument is passed, assigns 
        None values to all properties. If a dictionary is given, tries to 
        use it to give values to properties."""

        for prop in self._prop_list:
            self[prop] = None

        if init_dict is not None:

            if not isinstance(init_dict, dict):
                raise ValueError,
                    "Argument %s is not a dictionary" % repr(init_dict)

            for item in init_dict.keys():
                if item in prop_list:
                    self[item] = init_dict[item]
                else:
                    raise self.ParticleError,
                        "Key %s is not a valid particle property" % item

    def get(self, name):
        """Get the value of the property name."""

        if not isinstance(name, str):
            raise ValueError,
                "Property name %s is not a string" % repr(name)

        if name not in prop_list:
            raise self.ParticleError,
                        "Key %s is not a valid particle property" % item

        return self[name]


