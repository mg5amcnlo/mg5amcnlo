##
##
## Feynrules Header
##
##
##
##
##
import numbers

class FRBaseClass(object):
    """The class from which all FeynRules classes are derived."""

    require_args = []

    def __init__(self, *args, **options):
        assert(len(self.require_args) == len (args))
    
        for i, name in enumerate(self.require_args):
            setattr(self, name, args[i])
    
        for (option, value) in options.items():
            setattr(self, option, value)

    def get(self, name):
        return getattr(self, name)
    
    def set(self, name, value):
        setattr(self, name, value)
        
    def get_all(self):
        """Return a dictionary containing all the information of the object"""
        return self.__dict__



all_particles = []

class Particle(FRBaseClass):
    """A standard Particle"""

    require_args=['pdg_code', 'name', 'antiname', 'spin', 'color', 'mass', 'width', 'texname',
                 'antitexname', 'line', 'charge']

    def __init__(self, pdg_code, name, antiname, spin, color, mass, width, texname,
                 antitexname, line, charge , propagating=True, GoldstoneBoson=False, **options):
        
        args= (pdg_code, name, antiname, spin, color, mass, width, texname,
                 antitexname, line, charge)

        FRBaseClass.__init__(self, *args,  **options)

        global all_particles
        all_particles.append(self)
        
        self.propagating = propagating
        self.goldstone = GoldstoneBoson

        self.selfconjugate = (name == antiname)

    def __str__(self):
        return self.name
    __repr__ = __str__

    def anti(self):
        if self.selfconjugate:
           raise Exception('%s has no anti particle.' % self.name) 
        outdic = {}
        for k,v in self.__dict__.iteritems():
            if k not in self.require_args:                
		if isinstance(v, bool):
		    outdic[k] = v
                else:
                    outdic[k] = -v

        return Particle(-self.pdg_code, self.antiname, self.name, self.spin, -self.color, self.mass, self.width, self.antitexname,
                  self.texname, self.line, -self.charge, **outdic)



all_parameters = []

class Parameter(FRBaseClass):

    require_args=['name', 'nature', 'type', 'value', 'texname']

    def __init__(self, name, nature, type, value, texname, lhablock=None, lhacode=None):

	args = (name,nature,type,value,texname)

        FRBaseClass.__init__(self, *args)

        args=(name,nature,type,value,texname)

        global all_parameters
        all_parameters.append(self)

        if (lhablock is None or lhacode is None)  and nature == 'external':
            raise Exception('Need LHA information for external parameter "%s".' % name)
        self.lhablock = lhablock
        self.lhacode = lhacode

    def __str__(self):
        return self.name
    __repr__ = __str__



all_vertices = []

class Vertex(FRBaseClass):

    require_args=['particles', 'color', 'lorentz', 'couplings']

    def __init__(self, particles, color, lorentz, couplings, **opt):
 
	args = (particles, color, lorentz, couplings)
	
        FRBaseClass.__init__(self, *args, **opt)

        args=(particles,color,lorentz,couplings)

        global all_vertices
        all_vertices.append(self)

all_couplings = []

class Coupling(FRBaseClass):

    require_args=['name', 'value', 'order']

    def __init__(self, name, value, order, **opt):
	
	args =(name, value, order)	
        FRBaseClass.__init__(self, *args, **opt)
	
        global all_couplings
        all_couplings.append(self)

    


all_lorentz = []

class Lorentz(FRBaseClass):

    require_args=['name','spins','structure']
    
    def __init__(self, name, spins, structure, **opt):
        args = (name, spins, structure)
        FRBaseClass.__init__(self, *args, **opt)

        global all_lorentz
        all_lorentz.append(self)

    def __str__(self):
        return self.name
    __repr__ = __str__
