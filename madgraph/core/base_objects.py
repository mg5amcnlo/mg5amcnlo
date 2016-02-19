################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Definitions of all basic objects used in the core code: particle, 
interaction, model, leg, vertex, process, ..."""

import copy
import itertools
import logging
import math
import numbers
import os
import re
import StringIO
import madgraph.core.color_algebra as color
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
import madgraph.various.misc as misc 


logger = logging.getLogger('madgraph.base_objects')
pjoin = os.path.join

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
                        """%s is not a valid property for this object: %s\n
    Valid property are %s""" % (name,self.__class__.__name__, self.keys())
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
                  'line', 'propagator',
                   'is_part', 'self_antipart', 'type', 'counterterm']

    def default_setup(self):
        """Default values for all properties"""

        self['name'] = 'none'
        self['antiname'] = 'none'
        self['spin'] = 1
        self['color'] = 1
        self['charge'] = 1.
        self['mass'] = 'ZERO'
        self['width'] = 'ZERO'
        self['pdg_code'] = 0
        #self['texname'] = 'none'
        #self['antitexname'] = 'none'
        self['line'] = 'dashed'
        #self['propagating'] = True -> removed in favor or 'line' = None
        self['propagator'] = ''
        self['is_part'] = True
        self['self_antipart'] = False
        # True if ghost, False otherwise
        #self['ghost'] = False
        self['type'] = '' # empty means normal can also be ghost or goldstone 
        # Counterterm defined as a dictionary with format:
        # ('ORDER_OF_COUNTERTERM',((Particle_list_PDG))):{laurent_order:CTCouplingName}
        self['counterterm'] = {}

    def get(self, name):
        
        if name == 'ghost':
            return self['type'] == 'ghost'
        elif name == 'goldstone':
            return self['type'] == 'goldstone'
        elif name == 'propagating':
            return self['line'] is not None
        else:
            return super(Particle, self).get(name)

    def set(self, name, value, force=False):
        
        if name in ['texname', 'antitexname']:
            return True
        elif name == 'propagating':
            if not value:
                return self.set('line', None, force=force)
            elif not self.get('line'):
                return self.set('line', 'dashed',force=force)
            return True
        elif name  in ['ghost', 'goldstone']:
            if self.get('type') == name:
                if value:
                    return True
                else:
                    return self.set('type', '', force=force)
            else:
                if value:
                    return self.set('type', name, force=force)
                else:
                    return True
        return super(Particle, self).set(name, value,force=force)
        

    def filter(self, name, value):
        """Filter for valid particle property values."""

        if name in ['name', 'antiname']:
            # Forbid special character but +-~_
            p=re.compile('''^[\w\-\+~_]+$''')
            if not p.match(value):
                raise self.PhysicsObjectError, \
                        "%s is not a valid particle name" % value

        if name is 'ghost':
            if not isinstance(value,bool):
                raise self.PhysicsObjectError, \
                 "%s is not a valid bool for the 'ghost' attribute" % str(value)
    
        if name is 'counterterm':
            if not isinstance(value,dict):
                raise self.PhysicsObjectError, \
                    "counterterm %s is not a valid dictionary" % repr(value)
            for key, val in value.items():
                if not isinstance(key,tuple):
                    raise self.PhysicsObjectError, \
                        "key %s is not a valid tuple for counterterm key" % repr(key)
                if not isinstance(key[0],str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % repr(key[0])
                if not isinstance(key[1],tuple):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid list" % repr(key[1])
                for elem in key[1]:
                    if not isinstance(elem,tuple):
                        raise self.PhysicsObjectError, \
                            "%s is not a valid list" % repr(elem)
                    for partPDG in elem:
                        if not isinstance(partPDG,int):
                            raise self.PhysicsObjectError, \
                                "%s is not a valid integer for PDG" % repr(partPDG)
                        if partPDG<=0:
                            raise self.PhysicsObjectError, \
                                "%s is not a valid positive PDG" % repr(partPDG)
                if not isinstance(val,dict):
                    raise self.PhysicsObjectError, \
                        "value %s is not a valid dictionary for counterterm value" % repr(val)
                for vkey, vvalue in val.items():
                    if vkey not in [0,-1,-2]:
                        raise self.PhysicsObjectError, \
                            "Key %s is not a valid laurent serie order" % repr(vkey)
                    if not isinstance(vvalue,str):
                        raise self.PhysicsObjectError, \
                            "Coupling %s is not a valid string" % repr(vvalue)
        if name is 'spin':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Spin %s is not an integer" % repr(value)
            if (value < 1 or value > 5) and value != 99:
                raise self.PhysicsObjectError, \
                   "Spin %i not valid" % value

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
            if value not in ['dashed', 'straight', 'wavy', 'curly', 'double','swavy','scurly','dotted']:
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

    def is_perturbating(self,order,model):
        """Returns wether this particle contributes in perturbation of the order passed
           in argument given the model specified. It is very fast for usual models"""
           
        for int in model['interactions'].get_type('base'):
            # We discard the interactions with more than one type of orders
            # contributing because it then doesn't necessarly mean that this
            # particle (self) is charged under the group corresponding to the
            # coupling order 'order'. The typical example is in SUSY which 
            # features a ' photon-gluon-squark-antisquark ' interaction which
            # has coupling orders QED=1, QCD=1 and would induce the photon
            # to be considered as a valid particle to circulate in a loop of
            # type "QCD".
            if len(int.get('orders'))>1:
                continue
            if order in int.get('orders').keys() and self.get('pdg_code') in \
              [part.get('pdg_code') for part in int.get('particles')]:
                return True
            
        return False
           
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
        
    def get_charge(self):
        """Return the charge code with a correct minus sign"""

        if not self['is_part']:
            return - self['charge']
        else:
            return self['charge']

    def get_anti_charge(self):
        """Return the charge code of the antiparticle with a correct minus sign
        """

        if self['is_part']:
            return - self['charge']
        else:
            return self['charge']
        
    def get_name(self):
        """Return the name if particle, antiname if antiparticle"""

        if not self['is_part'] and not self['self_antipart']:
            return self['antiname']
        else:
            return self['name']

    def get_helicity_states(self, allow_reverse=True):
        """Return a list of the helicity states for the onshell particle"""

        spin = self.get('spin')
        if spin ==1:
            # Scalar
            res = [ 0 ]
        elif spin == 2:
            # Spinor
            res = [ -1, 1 ]                
        elif spin == 3 and self.get('mass').lower() == 'zero':
            # Massless vector
            res = [ -1, 1 ]
        elif spin == 3:
            # Massive vector
            res = [ -1, 0, 1 ]
        elif spin == 4 and self.get('mass').lower() == 'zero':
            # Massless tensor
            res = [-3, 3]
        elif spin == 4:
            # Massive tensor
            res = [-3, -1, 1, 3]
        elif spin == 5 and self.get('mass').lower() == 'zero':
            # Massless tensor
            res = [-2, -1, 1, 2]
        elif spin in [5, 99]:
            # Massive tensor
            res = [-2, -1, 0, 1, 2]
        else:
            raise self.PhysicsObjectError, \
              "No helicity state assignment for spin %d particles" % spin
                  
        if allow_reverse and not self.get('is_part'):
            res.reverse()


        return res

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
                    
    def get_copy(self, name):
        """Try to find a particle with the given name. Check both name
        and antiname. If a match is found, return the a copy of the 
        corresponding particle (first one in the list), with the 
        is_part flag set accordingly. None otherwise."""
        
        assert isinstance(name, str)
        
        part = self.find_name(name)
        if not part:
            # Then try to look for a particle with that PDG
            try:
                pdg = int(name)
            except ValueError:
                return None

            for p in self:
                if p.get_pdg_code()==pdg:
                    part = copy.copy(p)
                    part.set('is_part', True)
                    return part
                elif p.get_anti_pdg_code()==pdg:
                    part = copy.copy(p)
                    part.set('is_part', False)
                    return part

            return None
        part = copy.copy(part)
          
        if part.get('name') == name:
            part.set('is_part', True)
            return part
        elif part.get('antiname') == name:
            part.set('is_part', False)
            return part
        return None

    def find_name(self, name):
        """Try to find a particle with the given name. Check both name
        and antiname. If a match is found, return the a copy of the 
        corresponding particle (first one in the list), with the 
        is_part flag set accordingly. None otherwise."""

        assert isinstance(name, str), "%s is not a valid string" % str(name) 

        for part in self:
            if part.get('name') == name:
                return part
            elif part.get('antiname') == name:
                return part

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
                antipart = copy.deepcopy(particle)
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

    sorted_keys = ['id', 'particles', 'color', 'lorentz', 'couplings',
                   'orders','loop_particles','type','perturbation_type']

    def default_setup(self):
        """Default values for all properties"""

        self['id'] = 0
        self['particles'] = []
        self['color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        self['orders'] = {}
        # The type of interactions can be 'base', 'UV' or 'R2'.
        # For 'UV' or 'R2', one can always specify the loop it corresponds
        # to by a tag in the second element of the list. If the tag is an
        # empty list, then the R2/UV interaction will be recognized only
        # based on the nature of the identity of the particles branching
        # off the loop and the loop orders. 
        # Otherwise, the tag can be specified and it will be used when 
        # identifying the R2/UV interaction corresponding to a given loop
        # generated.
        # The format is [(lp1ID,int1ID),(lp1ID,int1ID),(lp1ID,int1ID),etc...]
        # Example of a tag for the following loop
        #
        #             ___34_____   The ';' line is a gluon with ID 21
        #          45/   ;         The '|' line is a d-quark with ID 1
        #     ------<    ;         The numbers are the interactions ID
        #            \___;______   The tag for this loop would be:
        #                12          ((21,34),(1,45),(1,12))
        #                         
        # This tag is equivalent to all its cyclic permutations. This is why
        # it must be specified in the canonical order which is defined with 
        # by putting in front of the tag the lowest 2-tuple it contains.
        # (the order relation is defined by comparing the particle ID first
        # and the interaction ID after in case the particle ID are the same).
        # In case there are two identical lowest 2-tuple in the tag, the
        # tag chosen is such that it has the lowest second 2-tuple. The procedure
        # is repeated again with the subsequent 2-tuple until there is only
        # one cyclic permutation remaining and the ambiguity is resolved.
        # This insures to have one unique unambiguous canonical tag chosen.
        # In the example above, it would be:
        #       ((1,12),(21,34),(1,45))
        # PS: Notice that in the UFO model, the tag-information is limited to 
        # the minimally relevant one which are the loop particles specified in
        # in the attribute below. In this case, 'loop_particles' is the list of 
        # all the loops giving this same counterterm contribution. 
        # Each loop being represented by a set of the PDG of the particles 
        # (not repeated) constituting it. In the example above, it would simply
        # be (1,21). In the UFO, if the loop particles are not specified then
        # MG5 will account for this counterterm only once per concerned vertex.
        # Taking the example of the three gluon vertex counterterm, one can
        # possibly have in the ufo:
        #                VertexB = blabla, loop_particles = (b)
        #                VertexT = blabla, loop_particles = (t)
        # or 
        #                VertexALL = blabla, loop_particles = ()
        # In the first case UFO specifies the specific counterterm to the three-
        # gluon loop with the bottom running in (VertexB) and with the top running
        # in (VertexT). So MG5 will associate these counterterm vertices once to
        # each of the two loop.
        # In the case where UFO defined VertexALL, then whenever MG5 encounters
        # a triangle three-gluon loop (say the bottom one), it will associate to
        # it the vertex VertexALL but will not do so again when encountering the 
        # same loop with the top quark running in. This, because it assumes that
        # the UFO vertexALL comprises all contributions already.
        
        self['loop_particles']=[[]]
        self['type'] = 'base'
        self['perturbation_type'] = None

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

        if name == 'perturbation_type':
            if value!=None and not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)            

        if name == 'type':
            #Should be a string
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)
        if name == 'loop_particles':
            if isinstance(value,list):
                for l in value:
                    if isinstance(l,list):
                        for part in l:
                            if not isinstance(part,int):
                                raise self.PhysicsObjectError, \
                                    "%s is not a valid integer" % str(part)
                            if part<0:
                                raise self.PhysicsObjectError, \
                                    "%s is not a valid positive integer" % str(part)

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
                
    def is_perturbating(self, orders_considered):
        """ Returns if this interaction comes from the perturbation of one of
        the order listed in the argument """
        
        if self['perturbation_type']==None:
            return True
        else:
            return (self['perturbation_type'] in orders_considered)
                
    def is_R2(self):
        """ Returns if the interaction is of R2 type."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'type' in self.keys():
            return (len(self['type'])>=2 and self['type'][:2]=='R2')
        else:
            return False

    def is_UV(self):
        """ Returns if the interaction is of UV type."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'type' in self.keys():
            return (len(self['type'])>=2 and self['type'][:2]=='UV')
        else:
            return False
        
    def is_UVmass(self):
        """ Returns if the interaction is of UVmass type."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'type' in self.keys():
            return (len(self['type'])>=6 and self['type'][:6]=='UVmass')
        else:
            return False
        
    def is_UVloop(self):
        """ Returns if the interaction is of UVmass type."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'type' in self.keys():
            return (len(self['type'])>=6 and self['type'][:6]=='UVloop')
        else:
            return False
        
    def is_UVtree(self):
        """ Returns if the interaction is of UVmass type."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'type' in self.keys():
            return (len(self['type'])>=6 and self['type'][:6]=='UVtree')
        else:
            return False
        
    def is_UVCT(self):
        """ Returns if the interaction is of the UVCT type which means that 
        it has been selected as a possible UV counterterm interaction for this
        process. Such interactions are marked by having the 'UVCT_SPECIAL' order
        key in their orders."""

        # Precaution only useful because some tests have a predefined model
        # bypassing the default_setup and for which type was not defined.
        if 'UVCT_SPECIAL' in self['orders'].keys():
            return True
        else:
            return False
        
    def get_epsilon_order(self):
        """ Returns 0 if this interaction contributes to the finite part of the
        amplitude and 1 (2) is it contributes to its single (double) pole """
        
        if 'type' in self.keys():
            if '1eps' in self['type']:
                return 1
            elif '2eps' in self['type']:
                return 2
            else:
                return 0
        else:
            return 0

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

    def get_WEIGHTED_order(self, model):
        """Get the WEIGHTED order for this interaction, for equivalent
        3-particle vertex. Note that it can be fractional."""

        return float(sum([model.get('order_hierarchy')[key]*self.get('orders')[key]\
                          for key in self.get('orders')]))/ \
               max((len(self.get('particles'))-2), 1)

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

    def generate_ref_dict(self,useR2UV=False, useUVCT=False):
        """Generate the reference dictionaries from interaction list.
        Return a list where the first element is the n>0 dictionary and
        the second one is n-1>1."""

        ref_dict_to0 = {}
        ref_dict_to1 = {}
        buffer = {}

        for inter in self:
            if useR2UV or (not inter.is_UV() and not inter.is_R2() and \
                           not inter.is_UVCT()):
                inter.generate_dict_entries(ref_dict_to0, ref_dict_to1)
            if useUVCT and inter.is_UVCT():
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

    def get_type(self, type):
        """ return all interactions in the list of type 'type' """
        return InteractionList([int for int in self if int.get('type')==type])

    def get_R2(self):
        """ return all interactions in the list of type R2 """
        return InteractionList([int for int in self if int.is_R2()])

    def get_UV(self):
        """ return all interactions in the list of type UV """
        return InteractionList([int for int in self if int.is_UV()])

    def get_UVmass(self):
        """ return all interactions in the list of type UVmass """
        return InteractionList([int for int in self if int.is_UVmass()])

    def get_UVtree(self):
        """ return all interactions in the list of type UVtree """
        return InteractionList([int for int in self if int.is_UVtree()])
    
    def get_UVloop(self):
        """ return all interactions in the list of type UVloop """
        return InteractionList([int for int in self if int.is_UVloop()])

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
        self['order_hierarchy'] = {}
        self['conserved_charge'] = set()
        self['coupling_orders'] = None
        self['expansion_order'] = None
        self['version_tag'] = None # position of the directory (for security)
        self['gauge'] = [0, 1]
        self['case_sensitive'] = True
        # attribute which might be define if needed
        #self['name2pdg'] = {'name': pdg}
        

    def filter(self, name, value):
        """Filter for model property values"""

        if name in ['name']:
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a string" %type(value)

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

        elif name == 'version_tag':
            if not (isinstance(value, str)):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a string" % type(value)

        elif name == 'order_hierarchy':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a dictionary" % \
                                                            type(value)
            for key in value.keys():
                if not isinstance(value[key],int):
                    raise self.PhysicsObjectError, \
                        "Object of type %s is not an integer" % \
                                                            type(value[key])
        elif name == 'gauge':
            if not (isinstance(value, list)):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a list" % type(value)

        elif name == 'case_sensitive':
            if not value in [True ,False]:
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a boolean" % type(value)
            

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
        if name == 'modelpath':
            modeldir = self.get('version_tag').rsplit('##',1)[0]
            if os.path.exists(modeldir):
                return modeldir
            else:
                raise Exception, "path %s not valid anymore." % modeldir
            #modeldir = os.path.join(os.path.dirname(modeldir),
            #                        os.path.basename(modeldir).rsplit("-",1)[0])
            #if os.path.exists(modeldir):
            #    return modeldir 
            #raise Exception, 'Invalid Path information: %s' % self.get('version_tag')          
        elif name == 'modelpath+restriction':
            modeldir = self.get('version_tag').rsplit('##',1)[0]
            modelname = self['name']            
            if not  os.path.exists(modeldir):
                raise Exception, "path %s not valid anymore" % modeldir
            modeldir = os.path.dirname(modeldir)
            modeldir = pjoin(modeldir, modelname)
            return modeldir
        elif name == 'restrict_name':
            modeldir = self.get('version_tag').rsplit('##',1)[0]
            modelname = self['name']            
            basename = os.path.basename(modeldir)
            restriction = modelname[len(basename)+1:]
            return restriction

        if (name == 'interaction_dict') and not self[name]:
            if self['interactions']:
                self['interaction_dict'] = self['interactions'].generate_dict()

        if (name == 'got_majoranas') and self[name] == None:
            if self['particles']:
                self['got_majoranas'] = self.check_majoranas()

        if (name == 'coupling_orders') and self[name] == None:
            if self['interactions']:
                self['coupling_orders'] = self.get_coupling_orders()

        if (name == 'order_hierarchy') and not self[name]:
            if self['interactions']:
                self['order_hierarchy'] = self.get_order_hierarchy()    

        if (name == 'expansion_order') and self[name] == None:
            if self['interactions']:
                self['expansion_order'] = \
                   dict([(order, -1) for order in self.get('coupling_orders')])
                   
        if (name == 'name2pdg') and 'name2pdg' not in self:
            self['name2pdg'] = {}
            for p in self.get('particles'):
                self['name2pdg'][p.get('antiname')] = -1*p.get('pdg_code')
                self['name2pdg'][p.get('name')] =  p.get('pdg_code')
                
        return Model.__bases__[0].get(self, name) # call the mother routine

    def set(self, name, value, force = False):
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
            self['order_hierarchy'] = {}
            self['expansion_order'] = None

        result = Model.__bases__[0].set(self, name, value, force) # call the mother routine

        if name == 'particles':
            # Recreate particle_dict
            self.get('particle_dict')

        return result

    def actualize_dictionaries(self):
        """This function actualizes the dictionaries"""

        [self['ref_dict_to0'], self['ref_dict_to1']] = \
                self['interactions'].generate_ref_dict()
        self['ref_dict_to0'].update(
                                self['particles'].generate_ref_dict())

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['name', 'particles', 'parameters', 'interactions',
                'couplings','lorentz', 'gauge']

    def get_particle(self, id):
        """Return the particle corresponding to the id / name"""
        
        try:
            return self["particle_dict"][id]
        except Exception:
            if isinstance(id, int):
                try:
                    return self.get("particle_dict")[id]
                except Exception,error:
                    return None
            else:
                if not hasattr(self, 'name2part'):
                    self.create_name2part()
                try: 
                    return self.name2part[id]
                except:
                    return None

    def create_name2part(self):
        """create a dictionary name 2 part"""
        
        self.name2part = {}
        for part in self.get("particle_dict").values():
            self.name2part[part.get('name')] = part
        
            

    def get_lorentz(self, name):
        """return the lorentz object from the associate name"""
        if hasattr(self, 'lorentz_name2obj'):
            return self.lorentz_name2obj[name]  
        else:
            self.create_lorentz_dict()
            return self.lorentz_name2obj[name]

    def create_lorentz_dict(self):
        """create the dictionary linked to the lorentz structure"""
        self.lorentz_name2obj = {}
        self.lorentz_expr2name = {}
        if not self.get('lorentz'):
            return
        for lor in self.get('lorentz'):
            self.lorentz_name2obj[lor.name] = lor
            self.lorentz_expr2name[lor.structure] = lor.name

    def get_interaction(self, id):
        """Return the interaction corresponding to the id"""

        try:
            return self.get("interaction_dict")[id]
        except Exception:
            return None

    def get_parameter(self, name):
        """Return the parameter associated to the name NAME"""
        
        # If information is saved
        if hasattr(self, 'parameters_dict') and self.parameters_dict:
            try:
                return self.parameters_dict[name]
            except Exception:
                # try to reload it before crashing 
                pass
            
        # Else first build the dictionary
        self.parameters_dict = {}
        for data in self['parameters'].values():
            [self.parameters_dict.__setitem__(p.name,p) for p in data]
        
        return self.parameters_dict[name]

    def get_coupling_orders(self):
        """Determine the coupling orders of the model"""
        return set(sum([i.get('orders').keys() for i in \
                        self.get('interactions')], []))

    def get_order_hierarchy(self):
        """Set a default order hierarchy for the model if not set by the UFO."""
        # Set coupling hierachy
        hierarchy = dict([(order, 1) for order in self.get('coupling_orders')])
        # Special case for only QCD and QED couplings, unless already set
        if self.get('coupling_orders') == set(['QCD', 'QED']):
            hierarchy['QED'] = 2
        return hierarchy


    def get_nflav(self):
        """returns the number of light quark flavours in the model."""
        return len([p for p in self.get('particles') \
                if p['spin'] == 2 and p['is_part'] and \
                p ['color'] != 1 and p['mass'].lower() == 'zero'])

    
    def get_particles_hierarchy(self):
        """Returns the order hierarchies of the model and the
        particles which have interactions in at least this hierarchy
        (used in find_optimal_process_orders in MultiProcess diagram
        generation):

        Check the coupling hierarchy of the model. Assign all
        particles to the different coupling hierarchies so that a
        particle is considered to be in the highest hierarchy (i.e.,
        with lowest value) where it has an interaction.
        """
        
        # Find coupling orders in model
        coupling_orders = self.get('coupling_orders')
        # Loop through the different coupling hierarchy values, so we
        # start with the most dominant and proceed to the least dominant
        hierarchy = sorted(list(set([self.get('order_hierarchy')[k] for \
                                     k in coupling_orders])))

        # orders is a rising list of the lists of orders with a given hierarchy
        orders = []
        for value in hierarchy:
            orders.append([ k for (k, v) in \
                            self.get('order_hierarchy').items() if \
                            v == value ])

        # Extract the interaction that correspond to the different
        # coupling hierarchies, and the corresponding particles
        interactions = []
        particles = []
        for iorder, order in enumerate(orders):
            sum_orders = sum(orders[:iorder+1], [])
            sum_interactions = sum(interactions[:iorder], [])
            sum_particles = sum([list(p) for p in particles[:iorder]], [])
            # Append all interactions that have only orders with at least
            # this hierarchy
            interactions.append([i for i in self.get('interactions') if \
                                 not i in sum_interactions and \
                                 not any([k not in sum_orders for k in \
                                          i.get('orders').keys()])])
            # Append the corresponding particles, excluding the
            # particles that have already been added
            particles.append(set(sum([[p.get_pdg_code() for p in \
                                      inter.get('particles') if \
                                       p.get_pdg_code() not in sum_particles] \
                                      for inter in interactions[-1]], [])))

        return particles, hierarchy

    def get_max_WEIGHTED(self):
        """Return the maximum WEIGHTED order for any interaction in the model,
        for equivalent 3-particle vertices. Note that it can be fractional."""

        return max([inter.get_WEIGHTED_order(self) for inter in \
                        self.get('interactions')])
            

    def check_majoranas(self):
        """Return True if there is fermion flow violation, False otherwise"""

        if any([part.is_fermion() and part.get('self_antipart') \
                for part in self.get('particles')]):
            return True

        # No Majorana particles, but may still be fermion flow
        # violating interactions
        for inter in self.get('interactions'):
            # Do not look at UV Wfct renormalization counterterms
            if len(inter.get('particles'))==1:
                continue
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
                'MadGraph5_aMC@NLO convention'
                
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
        
        #additional check for the Higgs in the mssm
        if self.get('name') == 'mssm' or self.get('name').startswith('mssm-'):
            part = self.get_particle(25)
            part.set('name', 'h1')
            part.set('antiname', 'h1')


            
    def change_parameter_name_with_prefix(self, prefix='mdl_'):
        """ Change all model parameter by a given prefix.
        Modify the parameter if some of them are identical up to the case"""

        lower_dict={}
        duplicate = set()
        keys = self.get('parameters').keys()
        for key in keys:
            for param in self['parameters'][key]:
                lower_name = param.name.lower()
                if not lower_name:
                    continue
                try:
                    lower_dict[lower_name].append(param)
                except KeyError:
                    lower_dict[lower_name] = [param]
                else:
                    duplicate.add(lower_name)
                    logger.debug('%s is defined both as lower case and upper case.' 
                                                                   % lower_name)
        
        if prefix == '' and  not duplicate:
            return
                
        re_expr = r'''\b(%s)\b'''
        to_change = []
        change={}
        # recast all parameter in prefix_XX
        for key in keys:
            for param in self['parameters'][key]:
                value = param.name.lower()
                if value in ['as','mu_r', 'zero','aewm1','g']:
                    continue
                elif value.startswith(prefix):
                    continue
                elif value in duplicate:
                    continue # handle later
                elif value:
                    change[param.name] = '%s%s' % (prefix,param.name)
                    to_change.append(param.name)
                    param.name = change[param.name]
            
        for value in duplicate:
            for i, var in enumerate(lower_dict[value]):
                to_change.append(var.name)
                new_name = '%s%s%s' % (prefix, var.name.lower(), 
                                                  ('__%d'%(i+1) if i>0 else ''))
                change[var.name] = new_name
                var.name = new_name
                to_change.append(var.name)
        assert 'zero' not in to_change
        replace = lambda match_pattern: change[match_pattern.groups()[0]]
        
        if not to_change:
            return
        
        if 'parameter_dict' in self:
            new_dict = dict( (change[name] if (name in change) else name, value) for
                             name, value in self['parameter_dict'].items())
            self['parameter_dict'] = new_dict
    
        if hasattr(self,'map_CTcoup_CTparam'):
            # If the map for the dependence of couplings to CTParameters has
            # been defined, we must apply the renaming there as well. 
            self.map_CTcoup_CTparam = dict( (coup_name, 
            [change[name] if (name in change) else name for name in params]) 
                  for coup_name, params in self.map_CTcoup_CTparam.items() )

        i=0
        while i*1000 <= len(to_change): 
            one_change = to_change[i*1000: min((i+1)*1000,len(to_change))]
            i+=1
            rep_pattern = re.compile('\\b%s\\b'% (re_expr % ('\\b|\\b'.join(one_change))))
            
            # change parameters
            for key in keys:
                if key == ('external',):
                    continue
                for param in self['parameters'][key]:
                    param.expr = rep_pattern.sub(replace, param.expr)
            # change couplings
            for key in self['couplings'].keys():
                for coup in self['couplings'][key]:
                    coup.expr = rep_pattern.sub(replace, coup.expr)
                    
            # change mass/width
            for part in self['particles']:
                if str(part.get('mass')) in one_change:
                    part.set('mass', rep_pattern.sub(replace, str(part.get('mass'))))
                if str(part.get('width')) in one_change:
                    part.set('width', rep_pattern.sub(replace, str(part.get('width'))))  
                if  hasattr(part, 'partial_widths'):
                    for key, value in part.partial_widths.items():    
                        part.partial_widths[key] = rep_pattern.sub(replace, value)
                
        #ensure that the particle_dict is up-to-date
        self['particle_dict'] =''
        self.get('particle_dict') 

        

    def get_first_non_pdg(self):
        """Return the first positive number that is not a valid PDG code"""
        return [c for c in range(1, len(self.get('particles')) + 1) if \
                c not in self.get('particle_dict').keys()][0]
                
    def write_param_card(self):
        """Write out the param_card, and return as string."""
        
        import models.write_param_card as writer
        out = StringIO.StringIO() # it's suppose to be written in a file
        param = writer.ParamCardWriter(self)
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

    def change_electroweak_mode(self, mode):
        """Change the electroweak mode. The only valid mode now is external.
        Where in top of the default MW and sw2 are external parameters."""

        assert mode in ["external",set(['mz','mw','alpha'])]
        
        try:
            W = self.get('particle_dict')[24]
        except KeyError:
            raise InvalidCmd('No W particle in the model impossible to '+
                                                        'change the EW scheme!')
        
        if mode=='external':
            MW = self.get_parameter(W.get('mass'))
            if not isinstance(MW, ParamCardVariable):
                newMW = ParamCardVariable(MW.name, MW.value, 'MASS', [24])
                if not newMW.value:
                    newMW.value = 80.385
                #remove the old definition
                self.get('parameters')[MW.depend].remove(MW)
                # add the new one
                self.add_param(newMW, ['external'])
                
            # Now check for sw2. if not define bypass this
            try:
                sw2 = self.get_parameter('sw2')
            except KeyError:
                try:
                    sw2 = self.get_parameter('mdl_sw2')
                except KeyError:
                    sw2=None
    
            if sw2:
                newsw2 = ParamCardVariable(sw2.name,sw2.value, 'SMINPUTS', [4])
                if not newsw2.value:
                    newsw2.value = 0.222246485786
                #remove the old definition
                self.get('parameters')[sw2.depend].remove(sw2)
                # add the new one
                self.add_param(newsw2, ['external'])
            # Force a refresh of the parameter dictionary
            self.parameters_dict = None
            return true

        elif mode==set(['mz','mw','alpha']):
            # For now, all we support is to go from mz, Gf, alpha to mz, mw, alpha
            W = self.get('particle_dict')[24]
            mass = self.get_parameter(W.get('mass'))
            mass_expr = 'cmath.sqrt(%(prefix)sMZ__exp__2/2. + cmath.sqrt('+\
              '%(prefix)sMZ__exp__4/4. - (%(prefix)saEW*cmath.pi*%(prefix)s'+\
              'MZ__exp__2)/(%(prefix)sGf*%(prefix)ssqrt__2)))'
            if 'external' in mass.depend:
                # Nothing to be done
                return True
            match = False
            if mass.expr == mass_expr%{'prefix':''}:
                prefix = ''
                match = True
            elif mass.expr == mass_expr%{'prefix':'mdl_'}:
                prefix = 'mdl_'
                match = True
            if match:
                MW = ParamCardVariable(mass.name, mass.value, 'MASS', [24])
                if not MW.value:
                    MW.value = 80.385                
                self.get('parameters')[('external',)].append(MW)
                self.get('parameters')[mass.depend].remove(mass)
                # Make Gf an internal parameter
                new_param = ModelVariable('Gf',
                '-%(prefix)saEW*%(prefix)sMZ**2*cmath.pi/(cmath.sqrt(2)*%(MW)s**2*(%(MW)s**2 - %(prefix)sMZ**2))' %\
                {'MW': mass.name,'prefix':prefix}, 'complex', mass.depend)
                Gf = self.get_parameter('%sGf'%prefix)
                self.get('parameters')[('external',)].remove(Gf)
                self.add_param(new_param, ['%saEW'%prefix])
                # Force a refresh of the parameter dictionary
                self.parameters_dict = None
                return True
            else:
                return False

    def change_mass_to_complex_scheme(self, toCMS=True):
        """modify the expression changing the mass to complex mass scheme"""
        
        # 1) Change the 'CMSParam' of loop_qcd_qed model to 1.0 so as to remove
        #    the 'real' prefix fromall UVCT wf renormalization expressions.
        #    If toCMS is False, then it makes sure CMSParam is 0.0 and returns
        #    immediatly.
        # 2) Find All input parameter mass and width associated
        #   Add a internal parameter and replace mass with that param
        # 3) Find All mass fixed by the model and width associated
        #   -> Both need to be fixed with a real() /Imag()
        # 4) Find All width set by the model
        #   -> Need to be set with a real()
        # 5) Fix the Yukawa mass to the value of the complex mass/ real mass
        # 6) Loop through all expression and modify those accordingly
        #    Including all parameter expression as complex
        
        try:
            CMSParam = self.get_parameter('CMSParam')
        except KeyError:
            try:
                CMSParam = self.get_parameter('mdl_CMSParam')
            except KeyError:
                CMSParam = None
                
        # Handle the case where we want to make sure the CMS is turned off
        if not toCMS:
            if CMSParam:
                CMSParam.expr = '0.0'
            return            
        
        # Now handle the case where we want to turn to CMS.    
        if CMSParam:
            CMSParam.expr = '1.0'
        
        to_change = {}
        mass_widths = [] # parameter which should stay real
        for particle in self.get('particles'):
            m = particle.get('width')
            if m in mass_widths:
                continue
            mass_widths.append(particle.get('width'))
            mass_widths.append(particle.get('mass'))
            width = self.get_parameter(particle.get('width'))
            if (isinstance(width.value, (complex,float)) and abs(width.value)==0.0) or \
                                                    width.name.lower() =='zero':
                #everything is fine since the width is zero
                continue
            if not isinstance(width, ParamCardVariable):
                width.expr = 're(%s)' % width.expr
            mass = self.get_parameter(particle.get('mass'))
            if (isinstance(width.value, (complex,float)) and abs(width.value)!=0.0) or \
                                                    mass.name.lower() != 'zero':
                # special SM treatment to change the gauge scheme automatically.
                if particle.get('pdg_code') == 24 and isinstance(mass, 
                                                                 ModelVariable):
                    status = self.change_electroweak_mode(
                                                   set(['mz','mw','alpha']))
                    # Use the newly defined parameter for the W mass
                    mass = self.get_parameter(particle.get('mass'))
                    if not status:
                        logger.warning('The W mass is not an external '+
                        'parameter in this model and the automatic change of'+
                        ' electroweak scheme changed. This is not advised for '+
                                            'applying the complex mass scheme.')

                # Add A new parameter CMASS
                #first compute the dependencies (as,...)
                depend = list(set(mass.depend + width.depend))
                if len(depend)>1 and 'external' in depend:
                    depend.remove('external')
                depend = tuple(depend)
                if depend == ('external',):
                    depend = ()
                
                # Create the new parameter
                if isinstance(mass, ParamCardVariable):
                    New_param = ModelVariable('CMASS_'+mass.name,
                        'cmath.sqrt(%(mass)s**2 - complex(0,1) * %(mass)s * %(width)s)' \
                              % {'mass': mass.name, 'width': width.name}, 
                        'complex', depend)              
                else:
                    New_param = ModelVariable('CMASS_'+mass.name,
                        mass.expr, 'complex', depend)
                    # Modify the treatment of the width in this case
                    if not isinstance(width, ParamCardVariable):
                        width.expr = '- im(%s**2) / cmath.sqrt(re(%s**2))' % (mass.expr, mass.expr)
                    else:
                        # Remove external parameter from the param_card
                        New_width = ModelVariable(width.name,
                        '-1 * im(CMASS_%s**2) / %s' % (mass.name, mass.name), 'real', mass.depend)
                        self.get('parameters')[('external',)].remove(width)
                        self.add_param(New_param, (mass,))
                        self.add_param(New_width, (New_param,))
                        mass.expr = 'cmath.sqrt(re(%s**2))' % mass.expr                
                        to_change[mass.name] = New_param.name
                        continue                        
                        
                    mass.expr = 're(%s)' % mass.expr                
                self.add_param(New_param, (mass, width))
                to_change[mass.name] = New_param.name
        
        # Remove the Yukawa and fix those accordingly to the mass/complex mass
        yukawas = [p for p in self.get('parameters')[('external',)] 
                                              if p.lhablock.lower() == 'yukawa']
        for yukawa in yukawas:
            # clean the pevious parameter
            self.get('parameters')[('external',)].remove(yukawa)
            
            particle = self.get_particle(yukawa.lhacode[0])
            mass = self.get_parameter(particle.get('mass'))
            
            # add the new parameter in the correct category
            if mass.depend == ('external',):
                depend = ()
            else:
                depend = mass.depend
                
            New_param = ModelVariable(yukawa.name, mass.name, 'real', depend)
            
            # Add it in the model at the correct place (for the dependences)
            if mass.name in to_change:
                expr = 'CMASS_%s' % mass.name
            else:
                expr = mass.name
            param_depend = self.get_parameter(expr)
            self.add_param(New_param, [param_depend])
            
        if not to_change:
            return
            
            
        # So at this stage we still need to modify all parameters depending of
        # particle's mass. In addition all parameter (but mass/width/external 
        # parameter) should be pass in complex mode.
        pat = '|'.join(to_change.keys())
        pat = r'(%s)\b' % pat
        pat = re.compile(pat)
        def replace(match):
            return to_change[match.group()]
        
        # Modify the parameters
        for dep, list_param in self['parameters'].items():
            for param in list_param:
                if param.name.startswith('CMASS_') or param.name in mass_widths or\
                              isinstance(param, ParamCardVariable):
                    continue
                param.type = 'complex'
#                print param.expr,  to_change
                
                param.expr = pat.sub(replace, param.expr)
        
        # Modify the couplings        
        for dep, list_coup in self['couplings'].items():
            for coup in list_coup:                
                coup.expr = pat.sub(replace, coup.expr)
                
    def add_param(self, new_param, depend_param):
        """add the parameter in the list of parameter in a correct position"""
            
        pos = 0
        for i,param in enumerate(self.get('parameters')[new_param.depend]):
            if param.name in depend_param:
                pos = i + 1
        self.get('parameters')[new_param.depend].insert(pos, new_param)


    #def __repr__(self):
    #    """ """
    #    raise Exception
    #    return "Model(%s)" % self.get_name()
    #__str__ = __repr__
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
        except Exception:
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
        # state: True = final, False = initial (boolean to save memory)
        self['state'] = True
        #self['loop_line'] = False
        self['loop_line'] = False
        # from_group: Used in diagram generation
        self['from_group'] = True
        # onshell: decaying leg (True), forbidden s-channel (False), none (None)
        self['onshell'] = None

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

        if name == 'loop_line':
            if not isinstance(value, bool) and value != None:
                raise self.PhysicsObjectError, \
                    "%s is not a valid boolean for leg flag loop_line" % \
                                                                    str(value)

        if name == 'onshell':
            if not isinstance(value, bool) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid boolean for leg flag onshell" % \
                                                                    str(value)
        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id', 'number', 'state', 'from_group', 'loop_line', 'onshell']

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

    # Helper function. We don't overload the == operator because it might be useful
    # to define it differently than that later.

    def same(self, leg):
        """ Returns true if the leg in argument has the same ID and the same numer """

        # In case we want to check this leg with an integer in the tagging procedure, 
        # then it only has to match the leg number.
        if isinstance(leg,int):
            if self['number']==leg:
                return True
            else:
                return False

        # If using a Leg object instead, we also want to compare the other relevant
        # properties.
        elif isinstance(leg, Leg):
            if self['id']==leg.get('id') and \
               self['number']==leg.get('number') and \
               self['loop_line']==leg.get('loop_line') :
                return True
            else:
                return False

        else :
            return False

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
    
    def sort(self,*args, **opts):
        """Match with FKSLegList"""
        Opts=copy.copy(opts)
        if 'pert' in Opts.keys():
            del Opts['pert']
        return super(LegList,self).sort(*args, **Opts)


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
    
    # This sets what are the ID's of the vertices that must be ignored for the
    # purpose of the multi-channeling. 0 and -1 are ID's of various technical
    # vertices which have no relevance from the perspective of the diagram 
    # topology, while -2 is the ID of a vertex that results from a shrunk loop
    # (for loop-induced integration with MadEvent) and one may or may not want
    # to consider these higher point loops for the purpose of the multi-channeling.
    # So, adding -2 to the list below makes sur that all loops are considered
    # for multichanneling.
    ID_to_veto_for_multichanneling = [0,-1,-2]
    
    # For loop-induced integration, considering channels from up to box loops 
    # typically leads to better efficiencies. Beyond that, it is detrimental 
    # because the phase-space generation is not suited to map contact interactions
    # This parameter controls up to how many legs should loop-induced diagrams
    # be considered for multichanneling.
    # Notice that, in the grouped subprocess case mode, if -2 is not added to 
    # the list ID_to_veto_for_multichanneling then all loop are considered by 
    # default and the constraint below is not applied.
    max_n_loop_for_multichanneling = 4
    
    def default_setup(self):
        """Default values for all properties"""

        # The 'id' of the vertex corresponds to the interaction ID it is made of.
        # Notice that this 'id' can take the special values :
        #  -1 : A two-point vertex which either 'sews' the two L-cut particles
        #       together or simply merges two wavefunctions to create an amplitude
        #       (in the case of tree-level diagrams).
        #  -2 : The id given to the ContractedVertices (i.e. a shrunk loop) so 
        #       that it can be easily identified when constructing the DiagramChainLinks.
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

    def nice_string(self):
        """return a nice string"""
        
        mystr = []
        for leg in self['legs']:
            mystr.append( str(leg['number']) + '(%s)' % str(leg['id']))
        mystr = '(%s,id=%s ,obj_id:%s)' % (', '.join(mystr), self['id'], id(self))
        
        return(mystr)


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

        if leg.get('loop_line'):
            # Loop lines never count as s-channel
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
# ContractedVertex
#===============================================================================
class ContractedVertex(Vertex):
    """ContractedVertex: When contracting a loop to a given vertex, the created
    vertex object is then a ContractedVertex object which has additional 
    information with respect to a regular vertex object. For example, it contains
    the PDG of the particles attached to it. (necessary because the contracted
    vertex doesn't have an interaction ID which would allow to retrieve such
    information).
    """

    def default_setup(self):
        """Default values for all properties"""

        self['PDGs'] = []
        self['loop_tag'] = tuple()
        self['loop_orders'] = {}
        super(ContractedVertex, self).default_setup()

    def filter(self, name, value):
        """Filter for valid vertex property values."""

        if name == 'PDGs':
            if isinstance(value, list):
                for elem in value:
                    if not isinstance(elem,int):
                        raise self.PhysicsObjectError, \
                            "%s is not a valid integer for leg PDG" % str(elem)
            else:
                raise self.PhysicsObjectError, \
                  "%s is not a valid list for contracted vertex PDGs"%str(value)                
        if name == 'loop_tag':
            if isinstance(value, tuple):
                for elem in value:
                    if not (isinstance(elem,int) or isinstance(elem,tuple)):
                        raise self.PhysicsObjectError, \
                          "%s is not a valid int or tuple for loop tag element"%str(elem)
            else:
                raise self.PhysicsObjectError, \
                  "%s is not a valid tuple for a contracted vertex loop_tag."%str(value)
        if name == 'loop_orders':
            Interaction.filter(Interaction(), 'orders', value)
        else:
            return super(ContractedVertex, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return super(ContractedVertex, self).get_sorted_keys()+['PDGs']

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

        pass_sanity = True
        if self['vertices']:
            mystr = '('
            for vert in self['vertices']:
                used_leg = [] 
                mystr = mystr + '('
                for leg in vert['legs'][:-1]:
                    mystr = mystr + str(leg['number']) + '(%s)' % str(leg['id']) + ','
                    used_leg.append(leg['number'])
                if __debug__ and len(used_leg) != len(set(used_leg)):
                    pass_sanity = False
                    responsible = id(vert)
                    
                if self['vertices'].index(vert) < len(self['vertices']) - 1:
                    # Do not want ">" in the last vertex
                    mystr = mystr[:-1] + '>'
                mystr = mystr + str(vert['legs'][-1]['number']) + '(%s)' % str(vert['legs'][-1]['id']) + ','
                mystr = mystr + 'id:' + str(vert['id']) + '),'
                                
            mystr = mystr[:-1] + ')'
            mystr += " (%s)" % (",".join(["%s=%d" % (key, self['orders'][key]) \
                                     for key in sorted(self['orders'].keys())]))
            
            if not pass_sanity:
                raise Exception, "invalid diagram: %s. vert_id: %s" % (mystr, responsible) 
                
            return mystr
        else:
            return '()'
    
    def calculate_orders(self, model):
        """Calculate the actual coupling orders of this diagram. Note
        that the special order WEIGTHED corresponds to the sum of
        hierarchys for the couplings."""

        coupling_orders = dict([(c, 0) for c in model.get('coupling_orders')])
        weight = 0
        for vertex in self['vertices']:
            if vertex.get('id') in [0,-1]: continue
            if vertex.get('id') == -2:
                couplings = vertex.get('loop_orders')
            else:
                couplings = model.get('interaction_dict')[vertex.get('id')].\
                                                                   get('orders')
            for coupling in couplings:
                coupling_orders[coupling] += couplings[coupling]
            weight += sum([model.get('order_hierarchy')[c]*n for \
                              (c,n) in couplings.items()])
        coupling_orders['WEIGHTED'] = weight
        self.set('orders', coupling_orders)

    def pass_squared_order_constraints(self, diag_multiplier, squared_orders,
                                                               sq_orders_types):
        """ Returns wether the contributiong consisting in the current diagram 
        multiplied by diag_multiplier passes the *positive* squared_orders 
        specified ( a dictionary ) of types sq_order_types (a dictionary whose 
        values are the relational operator used to define the constraint of the
        order in key)."""
        
        for order, value in squared_orders.items():
            if value<0:
                continue
            combined_order = self.get_order(order) + \
                                                diag_multiplier.get_order(order)
            if ( sq_orders_types[order]=='==' and combined_order != value ) or \
               ( sq_orders_types[order] in ['=', '<='] and combined_order > value) or \
               ( sq_orders_types[order]=='>' and combined_order <= value) :
                return False
        return True

    def get_order(self, order):
        """Return the order of this diagram. It returns 0 if it is not present."""

        try:
            return self['orders'][order]
        except Exception:
            return 0

    def get_contracted_loop_diagram(self, struct_rep=None):
        """ Returns a Diagram which correspond to the loop diagram with the 
        loop shrunk to a point. Of course for a instance of base_objects.Diagram
        one must simply return self."""
        
        return self
        
    def get_external_legs(self):
        """ Return the list of external legs of this diagram """
        
        external_legs = LegList([])
        for leg in sum([vert.get('legs') for vert in self.get('vertices')],[]):
            if not leg.get('number') in [l.get('number') for l in external_legs]:
               external_legs.append(leg) 
               
        return external_legs
        
    def renumber_legs(self, perm_map, leg_list):
        """Renumber legs in all vertices according to perm_map"""

        vertices = VertexList()
        min_dict = copy.copy(perm_map)
        # Dictionary from leg number to state
        state_dict = dict([(l.get('number'), l.get('state')) for l in leg_list])
        # First renumber all legs in the n-1->1 vertices
        for vertex in self.get('vertices')[:-1]:
            vertex = copy.copy(vertex)
            leg_list = LegList([copy.copy(l) for l in vertex.get('legs')])
            for leg in leg_list[:-1]:
                leg.set('number', min_dict[leg.get('number')])
                leg.set('state', state_dict[leg.get('number')])
            min_number = min([leg.get('number') for leg in leg_list[:-1]])
            leg = leg_list[-1]
            min_dict[leg.get('number')] = min_number
            # resulting leg is initial state if there is exactly one
            # initial state leg among the incoming legs
            state_dict[min_number] = len([l for l in leg_list[:-1] if \
                                          not l.get('state')]) != 1
            leg.set('number', min_number)
            leg.set('state', state_dict[min_number])
            vertex.set('legs', leg_list)
            vertices.append(vertex)
        # Now renumber the legs in final vertex
        vertex = copy.copy(self.get('vertices')[-1])
        leg_list = LegList([copy.copy(l) for l in vertex.get('legs')])
        for leg in leg_list:
            leg.set('number', min_dict[leg.get('number')])
            leg.set('state', state_dict[leg.get('number')])
        vertex.set('legs', leg_list)
        vertices.append(vertex)
        # Finally create new diagram
        new_diag = copy.copy(self)
        new_diag.set('vertices', vertices)
        state_dict = {True:'T',False:'F'}
        return new_diag

    def get_vertex_leg_numbers(self, 
                        veto_inter_id=Vertex.ID_to_veto_for_multichanneling,
                        max_n_loop=0):
        """Return a list of the number of legs in the vertices for
        this diagram. 
        This function is only used for establishing the multi-channeling, so that
        we exclude from it all the fake vertices and the vertices resulting from
        shrunk loops (id=-2)"""


        if max_n_loop == 0:
            max_n_loop = Vertex.max_n_loop_for_multichanneling
        
        res = [len(v.get('legs')) for v in self.get('vertices') if (v.get('id') \
                                  not in veto_inter_id) or (v.get('id')==-2 and 
                                                 len(v.get('legs'))>max_n_loop)]

        return res
    
    def get_num_configs(self, model, ninitial):
        """Return the maximum number of configs from this diagram,
        given by 2^(number of non-zero width s-channel propagators)"""

        s_channels = [v.get_s_channel_id(model,ninitial) for v in \
                              self.get('vertices')[:-1]]
        num_props = len([i for i in s_channels if i != 0 and \
                         model.get_particle(i).get('width').lower() != 'zero'])
        
        if num_props < 1:
            return 1
        else:
            return 2**num_props
        
    def get_flow_charge_diff(self, model):
        """return the difference of total diff of charge occuring on the 
        lofw of the initial parton. return [None,None] if the two initial parton
        are connected and the (partial) value if None if the initial parton is 
        not a fermiom"""
        
        import madgraph.core.drawing as drawing
        drawdiag = drawing.FeynmanDiagram(self, model)
        drawdiag.load_diagram()
        out = []
        
        for v in drawdiag.initial_vertex:
            init_part = v.lines[0]
            if not init_part.is_fermion():
                out.append(None)
                continue
            
            init_charge = model.get_particle(init_part.id).get('charge')
            
            l_last = init_part
            v_last = v
            vcurrent = l_last.end
            if vcurrent == v:
                vcurrent = l_last.begin
            security =0
            while not vcurrent.is_external():
                if security > 1000:
                    raise Exception, 'wrong diagram'
                next_l = [l for l in vcurrent.lines if l is not l_last and l.is_fermion()][0]
                next_v = next_l.end
                if next_v == vcurrent:
                    next_v = next_l.begin
                l_last, vcurrent = next_l, next_v
            if vcurrent in drawdiag.initial_vertex:
                return [None, None]
            
            out.append(model.get_particle(l_last.id).get('charge') - init_charge)    
        return out
                

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

    # Helper function

    def get_max_order(self,order):
        """ Return the order of the diagram in the list with the maximum coupling
        order for the coupling specified """
        max_order=-1

        for diag in self:
            if order in diag['orders'].keys():
                if max_order==-1 or diag['orders'][order] > max_order:
                    max_order = diag['orders'][order]

        return max_order

    def apply_negative_sq_order(self, ref_diag_list, order, value, order_type):
        """ This function returns a fitlered version of the diagram list self
        which satisfy the negative squared_order constraint 'order' with negative
        value 'value' and of type 'order_type', assuming that the diagram_list
        it must be squared against is 'reg_diag_list'. It also returns the
        new postive target squared order which correspond to this negative order
        constraint. Example: u u~ > d d~ QED^2<=-2 means that one wants to
        pick terms only up to the the next-to-leading order contributiong in QED,
        which is QED=2 in this case, so that target_order=4 is returned."""
        
        # First we must compute all contributions to that order
        target_order = min(ref_diag_list.get_order_values(order))+\
                                  min(self.get_order_values(order))+2*(-value-1)
        
        new_list = self.apply_positive_sq_orders(ref_diag_list, 
                                       {order:target_order}, {order:order_type})
        
        return new_list, target_order
        
    def apply_positive_sq_orders(self, ref_diag_list, sq_orders, sq_order_types):
        """ This function returns a filtered version of self which contain
        only the diagram which satisfy the positive squared order constraints
        sq_orders of type sq_order_types and assuming that the diagrams are
        multiplied with those of the reference diagram list ref_diag_list."""
                
        new_diag_list = DiagramList()
        for tested_diag in self:
            for ref_diag in ref_diag_list:
                if tested_diag.pass_squared_order_constraints(ref_diag,
                                                      sq_orders,sq_order_types):
                    new_diag_list.append(tested_diag)
                    break
        return new_diag_list

    def filter_constrained_orders(self, order, value, operator):
        """ This function modifies the current object and remove the diagram
        which do not obey the condition """ 
        
        new = []
        for tested_diag in self:
            if operator == '==':
                if tested_diag['orders'][order] == value:
                    new.append(tested_diag)
            elif operator == '>':
                if tested_diag['orders'][order] > value:
                    new.append(tested_diag)                
        self[:] = new
        return self


    def get_min_order(self,order):
        """ Return the order of the diagram in the list with the mimimum coupling
        order for the coupling specified """
        min_order=-1
        for diag in self:
            if order in diag['orders'].keys():
                if min_order==-1 or diag['orders'][order] < min_order:
                    min_order = diag['orders'][order]
            else:
                return 0

        return min_order

    def get_order_values(self, order):
        """ Return the list of possible values appearing in the diagrams of this
        list for the order given in argument """

        values=set([])
        for diag in self:
            if order in diag['orders'].keys():
                values.add(diag['orders'][order])
            else:
                values.add(0)  

        return list(values)

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
        # These define the orders restrict the born and loop amplitudes.
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
        self['forbidden_onsh_s_channels'] = []
        self['forbidden_s_channels'] = []
        self['forbidden_particles'] = []
        self['is_decay_chain'] = False
        self['overall_orders'] = {}
        # Decay chain processes associated with this process
        self['decay_chains'] = ProcessList()
        # Legs with decay chains substituted in
        self['legs_with_decays'] = LegList()
        # Loop particles if the process is to be computed at NLO
        self['perturbation_couplings']=[]        
        # These orders restrict the order of the squared amplitude.
        # This dictionary possibly contains a key "WEIGHTED" which
        # gives the upper bound for the total weighted order of the
        # squared amplitude.
        self['squared_orders'] = {}
        # The squared order (sqorders) constraints above can either be upper 
        # bound (<=) or exact match (==) depending on how they were specified
        # in the user input. This choice is stored in the dictionary below.
        # Notice that the upper bound is the default
        self['sqorders_types'] = {}
        # other type of constraint at amplitude level
        self['constrained_orders'] = {} # {QED: (4,'>')}
        self['has_born'] = True
        # The NLO_mode is always None for a tree-level process and can be
        # 'all', 'real', 'virt' for a loop process.
        self['NLO_mode'] = 'tree'
        # The user might want to have the individual matrix element evaluations
        # for specific values of the coupling orders. The list below specifies
        # what are the coupling names which need be individually treated.
        # For example, for the process p p > j j [] QED=2 (QED=2 is 
        # then a squared order constraint), then QED will appear in the 
        # 'split_orders' list so that the subroutine in matrix.f return the
        # evaluation of the matrix element individually for the pure QCD 
        # contribution 'QCD=4 QED=0', the pure interference 'QCD=2 QED=2' and
        # the pure QED contribution of order 'QCD=0 QED=4'.
        self['split_orders'] = []

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name in ['legs', 'legs_with_decays'] :
            if not isinstance(value, LegList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid LegList object" % str(value)

        if name in ['orders', 'overall_orders','squared_orders']:
            Interaction.filter(Interaction(), 'orders', value)

        if name == 'constrained_orders':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary" % str(value)            

        if name == 'sqorders_types':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary" % str(value)
            for order in value.keys()+value.values():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid string" % str(value)

        if name == 'split_orders':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for order in value:
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid string" % str(value)

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

        if name in ['forbidden_onsh_s_channels', 'forbidden_s_channels']:
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

        if name == 'perturbation_couplings':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for order in value:
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                          "%s is not a valid string" % str(value)

        if name == 'is_decay_chain':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value)

        if name == 'has_born':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value)

        if name == 'decay_chains':
            if not isinstance(value, ProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessList" % str(value)

        if name == 'NLO_mode':
            import madgraph.interface.madgraph_interface as mg
            if value not in mg.MadGraphCmd._valid_nlo_modes:
                raise self.PhysicsObjectError, \
                        "%s is not a valid NLO_mode" % str(value)
        return True

    def has_multiparticle_label(self):
        """ A process, not being a ProcessDefinition never carries multiple
        particles labels"""
        
        return False

    def set(self, name, value):
        """Special set for forbidden particles - set to abs value."""

        if name == 'forbidden_particles':
            try:
                value = [abs(i) for i in value]
            except Exception:
                pass

        if name == 'required_s_channels':
            # Required s-channels need to be a list of lists of ids
            if value and isinstance(value, list) and \
               not isinstance(value[0], list):
                value = [value]

        return super(Process, self).set(name, value) # call the mother routine

    def get_squared_order_type(self, order):
        """ Return what kind of squared order constraint was specified for the
        order 'order'."""

        if order in self['sqorders_types'].keys():
            return self['sqorders_types'][order]
        else:
            # Default behavior '=' is interpreted as upper bound '<='
            return '='

    def get(self, name):
        """Special get for legs_with_decays"""
        
        if name == 'legs_with_decays':
            self.get_legs_with_decays()

        if name == 'sqorders_types':
            # We must make sure that there is a type for each sqorder defined
            for order in self['squared_orders'].keys():
                if order not in self['sqorders_types']:
                    # Then assign its type to the default '='
                    self['sqorders_types'][order]='='

        return super(Process, self).get(name) # call the mother routine

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['legs', 'orders', 'overall_orders', 'squared_orders',
                'constrained_orders',
                'model', 'id', 'required_s_channels', 
                'forbidden_onsh_s_channels', 'forbidden_s_channels',
                'forbidden_particles', 'is_decay_chain', 'decay_chains',
                'legs_with_decays', 'perturbation_couplings', 'has_born', 
                'NLO_mode','split_orders']

    def nice_string(self, indent=0, print_weighted = True, prefix=True):
        """Returns a nicely formated string about current process
        content. Since the WEIGHTED order is automatically set and added to 
        the user-defined list of orders, it can be ommitted for some info
        displays."""

        if prefix:
            mystr = " " * indent + "Process: "
        else:
            mystr = ""
        prevleg = None
        for leg in self['legs']:
            mypart = self['model'].get('particle_dict')[leg['id']]
            if prevleg and prevleg['state'] == False \
                   and leg['state'] == True:
                # Separate initial and final legs by >
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

        # Add orders
        if self['orders']:
            to_add = []
            for key in self['orders']:
                if not print_weighted and key == 'WEIGHTED':
                    continue
                value = int(self['orders'][key])
                if key in self['squared_orders']:
                    if self.get_squared_order_type(key) in ['<=', '==', '='] and \
                        self['squared_orders'][key] == value:
                        continue 
                    if self.get_squared_order_type(key) in ['>'] and value == 99:
                        continue
                if key in self['constrained_orders']:
                    if value == self['constrained_orders'][key][0] and\
                       self['constrained_orders'][key][1] in ['=', '<=', '==']:
                        continue
                if value == 0:
                    to_add.append('%s=0' % key)
                else:
                    to_add.append('%s<=%s' % (key,value))
                 
            if to_add:
                mystr = mystr + " ".join(to_add) + ' '

        if self['constrained_orders']:
            mystr = mystr + " ".join('%s%s%d' % (key, type, value) for 
                                     (key,(value,type)) 
                                         in self['constrained_orders'].items())  + ' '

        # Add perturbation_couplings
        if self['perturbation_couplings']:
            mystr = mystr + '[ '
            if self['NLO_mode']!='tree':
                if self['NLO_mode']=='virt' and not self['has_born']:
                    mystr = mystr + 'sqrvirt = '
                else:
                    mystr = mystr + self['NLO_mode'] + ' = '
            for order in self['perturbation_couplings']:
                mystr = mystr + order + ' '
            mystr = mystr + '] '

        # Add squared orders
        if self['squared_orders']:
            to_add = []
            for key in self['squared_orders'].keys():
                if not print_weighted and key == 'WEIGHTED':
                    continue
                if key in self['constrained_orders']:
                    if self['constrained_orders'][key][0] == self['squared_orders'][key]/2 and \
                       self['constrained_orders'][key][1] == self.get_squared_order_type(key):
                        continue
                to_add.append(key + '^2%s%d'%\
                (self.get_squared_order_type(key),self['squared_orders'][key]))
            
            if to_add:
                mystr = mystr + " ".join(to_add) + ' '
            

        # Add forbidden s-channels
        if self['forbidden_onsh_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_onsh_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add double forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$$ '
            for forb_id in self['forbidden_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '/ '
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('id') or self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in sorted(self['orders'])]) + ' '
        
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

        if self['orders']:
            mystr = mystr + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in self['orders']]) + ' '

        # Add perturbation orders
        if self['perturbation_couplings']:
            mystr = mystr + '[ '
            if self['NLO_mode']:
                mystr = mystr + self['NLO_mode']
                if not self['has_born']:
                    mystr = mystr + '^2'
                mystr = mystr + '= '
                
            for order in self['perturbation_couplings']:
                mystr = mystr + order + ' '
            mystr = mystr + '] '

        # Add squared orders
        if self['perturbation_couplings'] and self['squared_orders']:
            mystr = mystr + " ".join([key + '=' + repr(self['squared_orders'][key]) \
                       for key in self['squared_orders']]) + ' '

        # Add forbidden s-channels
        if self['forbidden_onsh_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_onsh_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add double forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$$ '
            for forb_id in self['forbidden_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add forbidden particles
        if self['forbidden_particles']:
            mystr = mystr + '/ '
            for forb_id in self['forbidden_particles']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in sorted(self['orders'])]) + ' '
        
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

    def shell_string(self, schannel=True, forbid=True, main=True, pdg_order=False,
                                                                print_id = True):
        """Returns process as string with '~' -> 'x', '>' -> '_',
        '+' -> 'p' and '-' -> 'm', including process number,
        intermediate s-channels and forbidden particles,
        pdg_order allow to order to leg order by pid."""

        mystr = ""
        if not self.get('is_decay_chain') and print_id:
            mystr += "%d_" % self['id']
        
        prevleg = None
        if pdg_order:
            legs = [l for l in self['legs'][1:]]
            def order_leg(l1,l2):
                id1 = l1.get('id')
                id2 = l2.get('id')
                return id2-id1
            legs.sort(cmp=order_leg)
            legs.insert(0, self['legs'][0])
        else:
            legs = self['legs']
        
        
        for leg in legs:
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
            mystr = mystr + "_" + decay.shell_string(schannel,forbid, main=False,
                                                     pdg_order=pdg_order)

        # Too long name are problematic so restrict them to a maximal of 70 char
        if len(mystr) > 64 and main:
            if schannel and forbid:
                out = self.shell_string(True, False, True, pdg_order)
            elif schannel:
                out = self.shell_string(False, False, True, pdg_order)
            else:
                out = mystr[:64]
            if not out.endswith('_%s' % self['uid']):    
                out += '_%s' % self['uid']
            return out

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

    def are_negative_orders_present(self):
        """ Check iteratively that no coupling order constraint include negative
        values."""

        if any(val<0 for val in self.get('orders').values()+\
                                           self.get('squared_orders').values()):
            return True
        
        for procdef in self['decay_chains']:
            if procdef.are_negative_orders_present():
                return True

        return False

    def are_decays_perturbed(self):
        """ Check iteratively that the decayed processes are not perturbed """
        
        for procdef in self['decay_chains']:
            if procdef['perturbation_couplings'] or procdef.are_decays_perturbed():
                return True
        return False
    
    def decays_have_squared_orders(self):
        """ Check iteratively that the decayed processes are not perturbed """
        
        for procdef in self['decay_chains']:
            if procdef['squared_orders']!={} or procdef.decays_have_squared_orders():
                return True
        return False
    
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

    def get_initial_final_ids(self):
        """return a tuple of two tuple containing the id of the initial/final
           state particles. Each list is ordered"""
           
        initial = []
        final = [l.get('id') for l in self.get('legs')\
              if l.get('state') or initial.append(l.get('id'))]
        initial.sort()
        final.sort()
        return (tuple(initial), tuple(final))
    
    def get_final_ids_after_decay(self):
        """Give the pdg code of the process including decay"""
        
        finals = self.get_final_ids()
        for proc in self.get('decay_chains'):
            init = proc.get_initial_ids()[0]
            #while 1:
            try:
                pos = finals.index(init)
            except:
                break
            finals[pos] = proc.get_final_ids_after_decay()
        output = []
        for d in finals:
            if isinstance(d, list):
                output += d
            else:
                output.append(d)
        
        return output
    

    def get_final_legs(self):
        """Gives the final state legs"""

        return filter(lambda leg: leg.get('state') == True,
                       self.get('legs'))
    
    def get_final_ids(self):
        """Gives the pdg codes for final state particles"""

        return [l.get('id') for l in self.get_final_legs()]
    
                
    def get_legs_with_decays(self):
        """Return process with all decay chains substituted in."""

        if self['legs_with_decays']:
            return self['legs_with_decays']

        legs = copy.deepcopy(self.get('legs'))
        org_decay_chains = copy.copy(self.get('decay_chains'))
        sorted_decay_chains = []
        # Sort decay chains according to leg order
        for leg in legs:
            if not leg.get('state'): continue
            org_ids = [l.get('legs')[0].get('id') for l in \
                           org_decay_chains]
            if leg.get('id') in org_ids:
                sorted_decay_chains.append(org_decay_chains.pop(\
                                        org_ids.index(leg.get('id'))))
        assert not org_decay_chains
        ileg = 0
        for decay in sorted_decay_chains:
            while legs[ileg].get('state') == False or \
                      legs[ileg].get('id') != decay.get('legs')[0].get('id'):
                ileg = ileg + 1
            decay_legs = decay.get_legs_with_decays()
            legs = legs[:ileg] + decay_legs[1:] + legs[ileg+1:]
            ileg = ileg + len(decay_legs) - 1

        # Replace legs with copies
        legs = [copy.copy(l) for l in legs]

        for ileg, leg in enumerate(legs):
            leg.set('number', ileg + 1)
            
        self['legs_with_decays'] = LegList(legs)

        return self['legs_with_decays']

    def list_for_sort(self):
        """Output a list that can be compared to other processes as:
        [id, sorted(initial leg ids), sorted(final leg ids),
        sorted(decay list_for_sorts)]"""

        sorted_list =  [self.get('id'),
                        sorted(self.get_initial_ids()),
                        sorted(self.get_final_ids())]
        
        if self.get('decay_chains'):
            sorted_list.extend(sorted([d.list_for_sort() for d in \
                                       self.get('decay_chains')]))

        return sorted_list

    def compare_for_sort(self, other):
        """Sorting routine which allows to sort processes for
        comparison. Compare only process id and legs."""

        if self.list_for_sort() > other.list_for_sort():
            return 1
        if self.list_for_sort() < other.list_for_sort():
            return -1
        return 0
        
    def identical_particle_factor(self):
        """Calculate the denominator factor for identical final state particles
        """

        final_legs = filter(lambda leg: leg.get('state') == True, \
                              self.get_legs_with_decays())

        identical_indices = {}
        for leg in final_legs:
            if leg.get('id') in identical_indices:
                identical_indices[leg.get('id')] = \
                                    identical_indices[leg.get('id')] + 1
            else:
                identical_indices[leg.get('id')] = 1
        return reduce(lambda x, y: x * y, [ math.factorial(val) for val in \
                        identical_indices.values() ], 1)

    def check_expansion_orders(self):
        """Ensure that maximum expansion orders from the model are
        properly taken into account in the process"""

        # Ensure that expansion orders are taken into account
        expansion_orders = self.get('model').get('expansion_order')
        orders = self.get('orders')
        sq_orders = self.get('squared_orders')
        
        tmp = [(k,v) for (k,v) in expansion_orders.items() if 0 < v < 99]
        for (k,v) in tmp:  
            if k in orders:
                if v < orders[k]:
                    if k in sq_orders.keys() and \
                                             (sq_orders[k]>v or sq_orders[k]<0):
                        logger.warning(
'''The process with the squared coupling order (%s^2%s%s) specified can potentially 
recieve contributions with powers of the coupling %s larger than the maximal 
value allowed by the model builder (%s). Hence, MG5_aMC sets the amplitude order
for that coupling to be this maximal one. '''%(k,self.get('sqorders_types')[k],
                                             self.get('squared_orders')[k],k,v))
                    else:
                        logger.warning(
'''The coupling order (%s=%s) specified is larger than the one allowed 
             by the model builder. The maximal value allowed is %s. 
             We set the %s order to this value''' % (k,orders[k],v,k))
                    orders[k] = v
            else:
                orders[k] = v

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

    def nice_string(self, indent = 0):
        """Returns a nicely formatted string of the matrix element processes."""

        mystr = "\n".join([p.nice_string(indent) for p in self])

        return mystr

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
        if 'legs_with_decays' in self: del self['legs_with_decays']

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

    def has_multiparticle_label(self):
        """ Check that this process definition will yield a single process, as
        each multileg only has one leg"""
        
        for process in self['decay_chains']:
            if process.has_multiparticle_label():
                return True
            
        for mleg in self['legs']:
            if len(mleg['ids'])>1:
                return True
        
        return False

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        keys = super(ProcessDefinition, self).get_sorted_keys()
        keys.remove('legs_with_decays')                                  

        return keys

    def get_minimum_WEIGHTED(self):
        """Retrieve the minimum starting guess for WEIGHTED order, to
        use in find_optimal_process_orders in MultiProcess diagram
        generation (as well as particles and hierarchy). The algorithm:

        1) Pick out the legs in the multiprocess according to the
        highest hierarchy represented (so don't mix particles from
        different hierarchy classes in the same multiparticles!)

        2) Find the starting maximum WEIGHTED order as the sum of the
        highest n-2 weighted orders

        3) Pick out required s-channel particle hierarchies, and use
        the highest of the maximum WEIGHTED order from the legs and
        the minimum WEIGHTED order extracted from 2*s-channel
        hierarchys plus the n-2-2*(number of s-channels) lowest
        leg weighted orders.
        """

        model = self.get('model')
        
        # Extract hierarchy and particles corresponding to the
        # different hierarchy levels from the model
        particles, hierarchy = model.get_particles_hierarchy()

        # Find legs corresponding to the different orders
        # making sure we look at lowest hierarchy first for each leg
        max_order_now = []
        new_legs =  copy.copy(self.get('legs'))
        for parts, value in zip(particles, hierarchy):
            ileg = 0
            while ileg < len(new_legs):
                if any([id in parts for id in new_legs[ileg].get('ids')]):
                    max_order_now.append(value)
                    new_legs.pop(ileg)
                else:
                    ileg += 1

        # Now remove the two lowest orders to get maximum (since the
        # number of interactions is n-2)
        max_order_now = sorted(max_order_now)[2:]

        # Find s-channel propagators corresponding to the different orders
        max_order_prop = []
        for idlist in self.get('required_s_channels'):
            max_order_prop.append([0,0])
            for id in idlist:
                for parts, value in zip(particles, hierarchy):
                    if id in parts:
                        max_order_prop[-1][0] += 2*value
                        max_order_prop[-1][1] += 1
                        break

        if max_order_prop:
            if len(max_order_prop) >1:
                max_order_prop = min(*max_order_prop, key=lambda x:x[0])
            else:
                max_order_prop = max_order_prop[0]

            # Use either the max_order from the external legs or
            # the maximum order from the s-channel propagators, plus
            # the appropriate lowest orders from max_order_now
            max_order_now = max(sum(max_order_now),
                                max_order_prop[0] + \
                                sum(max_order_now[:-2 * max_order_prop[1]]))
        else:
            max_order_now = sum(max_order_now)            

        return max_order_now, particles, hierarchy

    def __iter__(self):
        """basic way to loop over all the process definition. 
           not used by MG which used some smarter version (use by ML)"""
        
        isids = [leg['ids'] for leg in self['legs'] \
                 if leg['state'] == False]
        fsids = [leg['ids'] for leg in self['legs'] \
                 if leg['state'] == True]

        red_isidlist = []
        # Generate all combinations for the initial state
        for prod in itertools.product(*isids):
            islegs = [Leg({'id':id, 'state': False}) for id in prod]  
            if tuple(sorted(prod)) in red_isidlist:
                    continue        
            red_isidlist.append(tuple(sorted(prod)))      
            red_fsidlist = []
            for prod in itertools.product(*fsids):
                # Remove double counting between final states
                if tuple(sorted(prod)) in red_fsidlist:
                    continue        
                red_fsidlist.append(tuple(sorted(prod)))
                leg_list = [copy.copy(leg) for leg in islegs]
                leg_list.extend([Leg({'id':id, 'state': True}) for id in prod])
                legs = LegList(leg_list)
                process = self.get_process_with_legs(legs)
                yield process

    def nice_string(self, indent=0, print_weighted=False, prefix=True):
        """Returns a nicely formated string about current process
        content"""

        if prefix:
            mystr = " " * indent + "Process: "
        else:
            mystr=""
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
        if self['forbidden_onsh_s_channels']:
            mystr = mystr + '$ '
            for forb_id in self['forbidden_onsh_s_channels']:
                forbpart = self['model'].get('particle_dict')[forb_id]
                mystr = mystr + forbpart.get_name() + ' '

        # Add double forbidden s-channels
        if self['forbidden_s_channels']:
            mystr = mystr + '$$ '
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
                       for key in sorted(self['orders'])]) + ' '

        if self['constrained_orders']:
            mystr = mystr + " ".join('%s%s%d' % (key, operator, value) for 
                                     (key,(value, operator)) 
                                   in self['constrained_orders'].items()) + ' '

        # Add perturbation_couplings
        if self['perturbation_couplings']:
            mystr = mystr + '[ '
            if self['NLO_mode']!='tree':
                if self['NLO_mode']=='virt' and not self['has_born']:
                    mystr = mystr + 'sqrvirt = '
                else:
                    mystr = mystr + self['NLO_mode'] + ' = '
            for order in self['perturbation_couplings']:
                mystr = mystr + order + ' '
            mystr = mystr + '] '

        if self['squared_orders']:
            mystr = mystr + " ".join([key + '^2%s%d'%\
                (self.get_squared_order_type(key),self['squared_orders'][key]) \
              for key in self['squared_orders'].keys() \
                                    if print_weighted or key!='WEIGHTED']) + ' '

        # Remove last space
        mystr = mystr[:-1]

        if self.get('id') or self.get('overall_orders'):
            mystr += " @%d" % self.get('id')
            if self.get('overall_orders'):
                mystr += " " + " ".join([key + '=' + repr(self['orders'][key]) \
                       for key in sorted(self['orders'])]) + ' '
        
        if not self.get('decay_chains'):
            return mystr

        for decay in self['decay_chains']:
            mystr = mystr + '\n' + \
                    decay.nice_string(indent + 2).replace('Process', 'Decay')

        return mystr

    def get_process_with_legs(self, LegList):
        """ Return a Process object which has the same properties of this 
            ProcessDefinition but with the specified LegList as legs attribute. 
            """
            
        return Process({\
            'legs': LegList,
            'model':self.get('model'),
            'id': self.get('id'),
            'orders': self.get('orders'),
            'sqorders_types': self.get('sqorders_types'),
            'squared_orders': self.get('squared_orders'),
            'constrained_orders': self.get('constrained_orders'),
            'has_born': self.get('has_born'),
            'required_s_channels': self.get('required_s_channels'),
            'forbidden_onsh_s_channels': self.get('forbidden_onsh_s_channels'),            
            'forbidden_s_channels': self.get('forbidden_s_channels'),
            'forbidden_particles': self.get('forbidden_particles'),
            'perturbation_couplings': self.get('perturbation_couplings'),
            'is_decay_chain': self.get('is_decay_chain'),
            'overall_orders': self.get('overall_orders'),
            'split_orders': self.get('split_orders'),
            'NLO_mode': self.get('NLO_mode')
            })
            
    def get_process(self, initial_state_ids, final_state_ids):
        """ Return a Process object which has the same properties of this 
            ProcessDefinition but with the specified given leg ids. """
        
        # First make sure that the desired particle ids belong to those defined
        # in this process definition.
        my_isids = [leg.get('ids') for leg in self.get('legs') \
              if not leg.get('state')]
        my_fsids = [leg.get('ids') for leg in self.get('legs') \
             if leg.get('state')]            
        for i, is_id in enumerate(initial_state_ids):
            assert is_id in my_isids[i]
        for i, fs_id in enumerate(final_state_ids):
            assert fs_id in my_fsids[i]
        
        return self.get_process_with_legs(LegList(\
               [Leg({'id': id, 'state':False}) for id in initial_state_ids] + \
               [Leg({'id': id, 'state':True}) for id in final_state_ids]))

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
