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
"""Unit test Library for importing and restricting model"""
from __future__ import division

import copy
import os
import sys
import time
import tempfile 
import shutil 

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import models.import_ufo as import_ufo
import models.usermod as usermod
import models.model_reader as model_reader
import madgraph.iolibs.export_v4 as export_v4

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
pjoin = os.path.join

import cmath

#
# UFO CLASS SINCE THEY WILL BE USEFULL!
#

class UFOBaseClass(object):
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

    def __str__(self):
        return self.name

    def nice_string(self):
        """ return string with the full information """
        return '\n'.join(['%s \t: %s' %(name, value) for name, value in self.__dict__.items()])

    def __repr__(self):
        replacements = [
            ('+','__plus__'),
            ('-','__minus__'),
            ('@','__at__'),
            ('!','__exclam__'),
            ('?','__quest__'),
            ('*','__star__'),
            ('~','__tilde__')
            ]
        text = self.name
        for orig,sub in replacements:
            text = text.replace(orig,sub)
        return text



all_particles = []

    

class Particle(UFOBaseClass):
    """A standard Particle"""

    require_args=['pdg_code', 'name', 'antiname', 'spin', 'color', 'mass', 'width', 'texname', 'antitexname', 'charge']

    require_args_all = ['pdg_code', 'name', 'antiname', 'spin', 'color', 'mass', 'width', 'texname', 'antitexname', 'charge', 'line', 'propagating', 'goldstoneboson']

    def __init__(self, pdg_code, name, antiname, spin, color, mass, width, texname,
                 antitexname, charge , line=None, propagating=True, goldstoneboson=False, **options):

        args= (pdg_code, name, antiname, spin, color, mass, width, texname,
                 antitexname, float(charge))

        UFOBaseClass.__init__(self, *args,  **options)

        global all_particles
        all_particles.append(self)

        self.propagating = propagating
        self.goldstoneboson= goldstoneboson

        self.selfconjugate = (name == antiname)
        if 1: #not line:
            self.line = self.find_line_type()
        else:
            self.line = line




    def find_line_type(self):
        """ find how we draw a line if not defined
        valid output: dashed/straight/wavy/curly/double/swavy/scurly
        """
        
        spin = self.spin
        color = self.color
        
        #use default
        if spin == 1:
            return 'dashed'
        elif spin == 2:
            if not self.selfconjugate:
                return 'straight'
            elif color == 1:
                return 'swavy'
            else:
                return 'scurly'
        elif spin == 3:
            if color == 1:
                return 'wavy'
            
            else:
                return 'curly'
        elif spin == 5:
            return 'double'
        elif spin == -1:
            return 'dotted'
        else:
            return 'dashed' # not supported yet

    def anti(self):
        if self.selfconjugate:
            raise Exception('%s has no anti particle.' % self.name) 
        outdic = {}
        for k,v in self.__dict__.iteritems():
            if k not in self.require_args_all:                
                outdic[k] = -v
        if self.color in [1,8]:
            newcolor = self.color
        else:
            newcolor = -self.color
                
        return Particle(-self.pdg_code, self.antiname, self.name, self.spin, newcolor, self.mass, self.width,
                        self.antitexname, self.texname, -self.charge, self.line, self.propagating, self.goldstoneboson, **outdic)



all_parameters = []

class Parameter(UFOBaseClass):

    require_args=['name', 'nature', 'type', 'value', 'texname']

    def __init__(self, name, nature, type, value, texname, lhablock=None, lhacode=None):

        args = (name,nature,type,value,texname)

        UFOBaseClass.__init__(self, *args)

        args=(name,nature,type,value,texname)

        global all_parameters
        all_parameters.append(self)

        if (lhablock is None or lhacode is None)  and nature == 'external':
            raise Exception('Need LHA information for external parameter "%s".' % name)
        self.lhablock = lhablock
        self.lhacode = lhacode

all_vertices = []

class Vertex(UFOBaseClass):

    require_args=['name', 'particles', 'color', 'lorentz', 'couplings']

    def __init__(self, name, particles, color, lorentz, couplings, **opt):
 
        args = (name, particles, color, lorentz, couplings)

        UFOBaseClass.__init__(self, *args, **opt)

        args=(particles,color,lorentz,couplings)

        global all_vertices
        all_vertices.append(self)

all_couplings = []

class Coupling(UFOBaseClass):

    require_args=['name', 'value', 'order']

    def __init__(self, name, value, order, **opt):

        args =(name, value, order)    
        UFOBaseClass.__init__(self, *args, **opt)
        global all_couplings
        all_couplings.append(self)
  


all_lorentz = []

class Lorentz(UFOBaseClass):

    require_args=['name','spins','structure']
    
    def __init__(self, name, spins, structure='external', **opt):
        args = (name, spins, structure)
        UFOBaseClass.__init__(self, *args, **opt)

        global all_lorentz
        all_lorentz.append(self)


all_functions = []

class Function(object):

    def __init__(self, name, arguments, expression):

        global all_functions
        all_functions.append(self)

        self.name = name
        self.arguments = arguments
        self.expr = expression
    
    def __call__(self, *opt):

        for i, arg in enumerate(self.arguments):
            exec('%s = %s' % (arg, opt[i] ))

        return eval(self.expr)

all_orders = []

class CouplingOrder(object):

    def __init__(self, name, expansion_order, hierarchy, perturbative_expansion = 0):
        
        global all_orders
        all_orders.append(self)

        self.name = name
        self.expansion_order = expansion_order
        self.hierarchy = hierarchy

all_decays = []

class Decay(UFOBaseClass):
    require_args = ['particle','partial_widths']

    def __init__(self, particle, partial_widths, **opt):
        args = (particle, partial_widths)
        UFOBaseClass.__init__(self, *args, **opt)

        global all_decays
        all_decays.append(self)
    
        # Add the information directly to the particle
        particle.partial_widths = partial_widths

all_form_factors = []

class FormFactor(UFOBaseClass):
    require_args = ['name','type','value']

    def __init__(self, name, type, value, **opt):
        args = (name, type, value)
        UFOBaseClass.__init__(self, *args, **opt)

        global all_form_factors
        all_form_factors.append(self)



#===============================================================================
# Test The UFO usermod package 
#===============================================================================
class TestModUFO(unittest.TestCase):
    """Test class for the USERMOD object"""


    def setUp(self):
        
        self.path = tempfile.mkdtemp(prefix='unitest_usermod') 

        #Read the full SM
        self.sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = usermod.UFOModel(self.sm_path)
        
    def tearDown(self):
        
        shutil.rmtree(self.path)


    def test_write_model(self):
        """ Check that we can write all the require UFO files """
        
        output = pjoin(self.path, 'usrmod')
        self.base_model.write(output)
        sm_path = import_ufo.find_ufo_path('sm')
        self.assertEqual(12, 
                len([1 for name in os.listdir(sm_path) if name.endswith('.py')]), 
               'New file in  UFO format, usrmod need to be modified')

        self.assertEqual(11, 
                len([1 for name in os.listdir(output) if name.endswith('.py')]))

        sys.path.insert(0, os.path.dirname(output))
        import usrmod


    def compare(self, text1, text2, optional=[], default={}):
        """ """
        
        texts= [text1, text2]
        data = []
        
        for text in texts:
            curr_data = []
            data.append(curr_data)
            curr_object = {}
            for line in text.split('\n'):
                line = line.strip()
                if line.endswith(',') or line.endswith(')'):
                    line = line[:-1]
                    
                if (line.count('=') == 2 and line.count('(') == 1):
                    if curr_object:
                        curr_data.append(curr_object)
                    curr_object = dict(default)
                    k,value = line.split('(')[1].split('=')
                    curr_object[k.strip()] = value.strip()
                elif line.count('=') == 1:
                    k,value = line.split('=')
                    curr_object[k.strip()] =  value.strip()
            else:
                if curr_object:
                    curr_data.append(curr_object)
        
        for element in data[0]:
            #print element, type(element)
            for i in range(1, len(data)):
                #for element2 in data[i]:
                #    print element2,
                #    if element == element2:
                #        print 'identical'
                #        break
                #    else:
                #        print 'different'
                #else:
                #    self.assertFalse(True)
                self.assertTrue(element in data[i])


    def test_write_orders(self):
        """Check that the content of the file is valid"""

        output = self.path
        self.base_model.write_orders(output)
        filename = os.path.join(output, 'coupling_orders.py')
        text = open(os.path.join(filename)).read()
        
        target = """
# This file was automatically created by The UFO_usermod        

from object_library import all_orders, CouplingOrder
QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    hierarchy = 1,
                    perturbative_expansion = 0)


QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    hierarchy = 2,
                    perturbative_expansion = 0)

"""

        self.compare(target, text, default={'perturbative_expansion':'0'})
        




    def test_write_particles(self):
        """Check that the content of the file is valid"""

        output = self.path
        self.base_model.write_particles(output)
        filename = os.path.join(output, 'particles.py')
        text = open(os.path.join(filename)).read()
        target = open(pjoin(self.sm_path, 'particles.py')).read()


        
        #format the ouptut
        target = target.replace('0.0,','0,')
        target = target.replace('1/3,','0.333333333333,')
        target = target.replace('2/3,','0.666666666667,')
        target = target.split('\n')
        target = [l.strip() for l in target 
                  if l.strip() and not l.strip().startswith('#') and 
                  not l.split('=')[0].strip() in ['line', 'propagating', 'goldstoneboson', 'GoldstoneBoson','selfconjugate']]
        duplicate = []
        target = [l for l in target if not '.anti()' in l or duplicate.append(l.split('=')[0].strip())] 
        
        text = text.replace('.0,',',')
        text = text.split('\n')        
        text = [l.strip() for l in text
                  if l.strip() and not l.strip().startswith('#') and 
                  not l.split('=')[0].strip() in ['line', 'propagating', 'goldstoneboson', 'GoldstoneBoson','selfconjugate']]
        
        keep = True      
        new_text = []  
        for line in text:
            if 'Particle' in line:
                if line.split('=')[0].strip() in duplicate:
                    keep = False
                else:
                    keep = True
            if not keep:
                continue
            else:
                new_text.append(line)
        text=new_text
        
        
        self.assertEqual(target, text)

    def test_write_vertices(self):
        """Check that the content of the file is valid"""

        output = self.path
        self.base_model.vertices = self.base_model.vertices[:2]
        self.base_model.write_vertices(output)
        filename = os.path.join(output, 'vertices.py')
        text = open(os.path.join(filename)).read()
        target = """V_1 = Vertex(name = 'V_1',
             particles = [P.G0, P.G0, P.G0, P.G0],
             color = ['1'],
             lorentz = [L.SSSS1],
             couplings = {(0,0): C.GC_33})


V_2 = Vertex(name = 'V_2',
             particles = [P.G0, P.G0, P.G__minus__, P.G__plus__],
             color = ['1'],
             lorentz = [L.SSSS1],
             couplings = {(0,0): C.GC_31})
        """



#===============================================================================
# Test The UFO usermod package 
#===============================================================================
class Test_ADDON_UFO(unittest.TestCase):
    """Test class for the USERMOD object"""


    def setUp(self):
        
        self.path = tempfile.mkdtemp(prefix='unitest_usermod') 

        #Read the full SM
        self.sm_path = import_ufo.find_ufo_path('sm')
        self.base_model = usermod.UFOModel(self.sm_path)
        
    def tearDown(self):
        
        shutil.rmtree(self.path)
        
    def test_add_particle(self):
        """Check that we can an external parameter consistently"""
        
        #ZERO is define in all model => we should just do nothing
        ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

        MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])        

        WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'WIDTH',
               lhacode = [ 25 ])
        
        H = Particle(pdg_code = 25,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = MH,
             width = WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0)
        
        number_particles = len(self.base_model.particles)
        
        #Add a particle which is exactly the Higgs like in the Standard Model
        self.base_model.add_particle(H)
        self.assertEqual( number_particles, len(self.base_model.particles))
        
        #Same name but different pid ->add but with rename
        H = Particle(pdg_code = 26,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = MH,
             width = WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0) 
        self.base_model.add_particle(H)
        self.assertEqual( number_particles+1, len(self.base_model.particles))       
        number_particles+=1
        self.assertEqual(H.name, 'H__1')
        
        #Different name and different pid keep it
        H = Particle(pdg_code = 26,
             name = 'H2',
             antiname = 'H2',
             spin = 1,
             color = 1,
             mass = MH,
             width = WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0) 
        self.base_model.add_particle(H)
        self.assertEqual( number_particles+1, len(self.base_model.particles))       
        number_particles+=1
        self.assertEqual(H.name, 'H2')
        #Different name But different pid.
        H = Particle(pdg_code = 25,
             name = 'H3',
             antiname = 'H3',
             spin = 1,
             color = 1,
             mass = MH,
             width = WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0) 
        self.base_model.add_particle(H)
        self.assertEqual( number_particles, len(self.base_model.particles))       
        #number_particles+=1
        self.assertEqual(H.name, 'H3')
        
        ###################################################
        ##  ALL THOSE TEST WERE NOT CHEKING MASS / WIDTH ##
        ###################################################                       
        # plugin to zero -> keep the one of the model
        H = Particle(pdg_code = 25,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = ZERO,
             width = ZERO,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0)         
        self.base_model.add_particle(H)
        self.assertEqual( number_particles, len(self.base_model.particles))       
        self.assertEqual(H.name, 'H')        
        self.assertEqual(H.mass.name, 'ZERO')
        true_higgs = self.base_model.particle_dict[25]
        self.assertEqual(true_higgs.name, 'H')        
        self.assertEqual(true_higgs.mass.name, 'MH')        
        
        # base_model to zero -> keep the one of the plugin
        M5 = Parameter(name = 'M5',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 5 ]) 
        W5 = Parameter(name = 'W5',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'DECAY',
               lhacode = [ 5 ]) 
        B = Particle(pdg_code = 5,
             name = 'B',
             antiname = 'B~',
             spin = 1,
             color = 1,
             mass = M5,
             width = W5,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0)   

        self.base_model.add_parameter(M5)
        self.base_model.add_parameter(W5)
        self.base_model.add_particle(B)
        self.assertEqual( number_particles, len(self.base_model.particles))       
        # For the mass both are define, so this is should be a merge
        self.assertEqual(B.name, 'B')        
        self.assertEqual(B.mass.name, 'M5')
        true_b = self.base_model.particle_dict[5]
        self.assertEqual(true_b.name, 'b')        
        self.assertEqual(true_b.mass.name, 'MB') # keep MB since M5 is merge on MB            
        self.assertEqual(self.base_model.old_new['M5'], 'MB')
        # For the width the model one is zero => overwrite
        self.assertEqual(B.name, 'B')        
        self.assertEqual(B.width.name, 'W5')
        self.assertEqual(true_b.width.name, 'W5')



        
        
    def test_add_external_parameters(self):
        """Check that we can an external parameter consistently"""        
                
        nb_param = len(self.base_model.parameters)
        #ZERO is define in all model => we should just do nothing
        ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')
        # add it and check that nothing happen!
        self.base_model.add_parameter(ZERO)
        self.assertEqual(nb_param,  len(self.base_model.parameters))
        
        
        # MH is already define 
        MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])
        
        # add it and check that nothing happen!
        self.base_model.add_parameter(MH)
        self.assertEqual(nb_param,  len(self.base_model.parameters))           
        
        # MH is already definebut has a different name ib both model
        MH = Parameter(name = 'MH2',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])
        
        # add it and check that nothing happen!
        self.base_model.add_parameter(MH)
        self.assertEqual(nb_param,  len(self.base_model.parameters)) 
        # But the information should be present in the old->new dict
        self.assertEqual(self.base_model.old_new['MH2'], 'MH')

        # Add an internal parameter depending of MH2
        GH = Parameter(name = 'GH',
               nature = 'internal',
               type = 'real',
               texname = '\\text{MH}',
               value = '25*MH2**2*AMH2*MH25')
        
        self.base_model.add_parameter(GH)
        self.assertEqual(nb_param+1,  len(self.base_model.parameters)) 
        #check that the expression of GH is correctly modified
        self.assertEqual(GH.value, '25*MH**2*AMH2*MH25')
        self.assertEqual(GH.name, 'GH')
        nb_param = nb_param+1
        
        # Add an internal parameter depending of MH2
        # But with a name conflict
        Gf = Parameter(name = 'Gf',
               nature = 'internal',
               type = 'real',
               texname = '\\text{MH}',
               value = '25*MH2**2*AMH2*MH25')
        
        self.base_model.add_parameter(Gf)
        self.assertEqual(nb_param+1,  len(self.base_model.parameters)) 
        #check that the expression of GH is correctly modified
        self.assertEqual(Gf.value, '25*MH**2*AMH2*MH25')
        self.assertEqual(Gf.name, 'Gf__1')
        self.assertEqual(self.base_model.old_new['Gf'], 'Gf__1')       
        nb_param = nb_param+1
         
        # Add an internal parameter depending of MH2 and of Gf
        # But with a name conflict
        Gf2 = Parameter(name = 'Gf2',
               nature = 'internal',
               type = 'real',
               texname = '\\text{MH}',
               value = '25*MH2**2*AMH2*MH25*math.cmath(Gf)')
        
        self.base_model.add_parameter(Gf2)
        self.assertEqual(nb_param+1,  len(self.base_model.parameters)) 
        #check that the expression of GH is correctly modified
        self.assertEqual(Gf2.value, '25*MH**2*AMH2*MH25*math.cmath(Gf__1)')
        self.assertEqual(Gf2.name, 'Gf2') 
        nb_param = nb_param+1        
        
        # MH250 is a completely new external parameter
        MH250 = Parameter(name = 'MH250',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 250 ])
        self.base_model.add_parameter(MH250)
        self.assertEqual(nb_param+1,  len(self.base_model.parameters))         
        nb_param += 1

        # MH251 is a completely new external parameter with same name as MH250
        MH251 = Parameter(name = 'MH250',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 251 ])

        self.base_model.add_parameter(MH251)
        self.assertEqual(nb_param+1,  len(self.base_model.parameters)) 
        self.assertEqual(self.base_model.old_new['MH250'], 'MH250__1')
        self.assertEqual(MH251.name, 'MH250__1')         
        nb_param += 1

                          
                          
    def test_couplings(self):
        
        nb_coup = len(self.base_model.couplings)
        
        
        GC_107 = Coupling(name = 'GC_107',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})
        
        self.base_model.add_coupling(GC_107)
        self.assertEqual(nb_coup,  len(self.base_model.couplings))
        self.assertTrue(hasattr(GC_107, 'replace'))
 
        GC_107 = Coupling(name = 'GC_110',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})
        
        self.base_model.add_coupling(GC_107)
        self.assertEqual(nb_coup,  len(self.base_model.couplings))
        self.assertTrue(hasattr(GC_107, 'replace')) 
 
        
        GC_107 = Coupling(name = 'GC_107',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x99))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})
        
        self.base_model.add_coupling(GC_107)
        self.assertEqual(nb_coup+1,  len(self.base_model.couplings))
        self.assertFalse(hasattr(GC_107, 'replace'))        
        
    
    def test_interaction(self):
        
        GC_1 = Coupling(name = 'GC_1',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x100))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})        
        self.base_model.add_coupling(GC_1)
        M5 = Parameter(name = 'M5',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 5 ]) 
        W5 = Parameter(name = 'W5',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'DECAY',
               lhacode = [ 5 ]) 
        self.base_model.add_parameter(M5)
        self.base_model.add_parameter(W5)
        
        L = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')
        self.base_model.add_lorentz(L)

        B = Particle(pdg_code = 5,
             name = 'B',
             antiname = 'B~',
             spin = 1,
             color = 1,
             mass = M5,
             width = W5,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0,
             Y = 0) 
        self.base_model.add_particle(B)
        
        V_2 = Vertex(name = 'V_2',
             particles = [ B, B, B, B ],
             color = [ '1' ],
             lorentz = [ L ],
             couplings = {(0,0): GC_1})
        
        self.base_model.add_interaction(V_2)
        
        added = self.base_model.vertices[-1]
        self.assertEqual(added.name, 'V_2__1')
        self.assertNotEqual(id(added.particles[0]), id(B))
        
        
        
        
        
                        