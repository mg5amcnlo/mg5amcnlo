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
"""Unit test Library for the objects in decay module."""
from __future__ import division

import copy
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.save_model as save_model
import madgraph.iolibs.drawing_eps as drawing_eps
from madgraph import MG5DIR
import decay.decay_objects as decay_objects
import tests.input_files.import_vertexlist as import_vertexlist

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# DecayParticleTest
#===============================================================================
class Test_DecayParticle(unittest.TestCase):
    """Test class for the DecayParticle object"""

    mydict = {}
    mypart = None
    my_testmodel_base = import_ufo.import_model('sm')
    my_2bodyvertexlist = base_objects.VertexList()
    my_3bodyvertexlist = base_objects.VertexList()
    my_2bodyvertexlist_wrongini = base_objects.VertexList()
    my_3bodyvertexlist_wrongini = base_objects.VertexList()

    def setUp(self):

        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)
        #print len(self.my_testmodel_base.get('interactions')), len(self.my_testmodel.get('interactions'))

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        no_want_pid = [1, 2, 3, 4, 13, 14, 15, 16, 21, 23]
        for pid in no_want_pid:
            particles.remove(self.my_testmodel.get_particle(pid))

        for inter in inter_list:
            if any([p.get('pdg_code') in no_want_pid for p in \
                        inter.get('particles')]):
                interactions.remove(inter)

        # Set a new name
        self.my_testmodel.set('name', 'my_smallsm')
        self.my_testmodel.set('particles', particles)
        self.my_testmodel.set('interactions', interactions)

        #Setup the vertexlist for my_testmodel and save this model
        import_vertexlist.make_vertexlist(self.my_testmodel)
        #save_model.save_model(os.path.join(MG5DIR, 'tests/input_files', self.my_testmodel['name']), self.my_testmodel)

        # Setup vertexlist for test
        full_vertexlist = import_vertexlist.full_vertexlist

        self.my_2bodyvertexlist = base_objects.VertexList([
                full_vertexlist[(32, 24)], full_vertexlist[(44, 24)]])
        fake_vertex = copy.deepcopy(full_vertexlist[(44, 24)])
        fake_vertex['legs'].append(base_objects.Leg({'id':22}))
        fake_vertex2 = copy.deepcopy(full_vertexlist[(32, 24)])
        fake_vertex2['legs'].append(base_objects.Leg({'id': 11}))
        self.my_3bodyvertexlist = base_objects.VertexList([
                fake_vertex, fake_vertex2])

        self.my_2bodyvertexlist_wrongini = base_objects.VertexList([
                full_vertexlist[(35, -24)], full_vertexlist[(32, -6)]])
        fake_vertex3 = copy.deepcopy(full_vertexlist[(35, -24 )])
        fake_vertex3['legs'].append(base_objects.Leg({'id':12}))
        self.my_3bodyvertexlist_wrongini = base_objects.VertexList([
                fake_vertex3])

        fake_vertex4 = copy.deepcopy(full_vertexlist[(44, 24 )])
        fake_vertex4['legs'].append(base_objects.Leg({'id':24}))
        self.my_3bodyvertexlist_radiactive = base_objects.VertexList([
                fake_vertex4])
        
        self.mydict = {'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W+',
                      'antitexname':'W-',
                      'line':'wavy',
                      'charge': 1.00,
                      'pdg_code': 24,
                      'propagating':True,
                      'is_part': True,
                      'self_antipart': False,
                       # decay_vertexlist must have two lists, one for on-shell,
                       # one for off-shell
                      'decay_vertexlist': {\
                           (2, False): self.my_2bodyvertexlist,
                           (2, True) : self.my_2bodyvertexlist,
                           (3, False): self.my_3bodyvertexlist,
                           (3, True) : self.my_3bodyvertexlist}
                       }

        self.mypart = decay_objects.DecayParticle(self.mydict)
        


    def test_setgetinit_correct(self):
        """Test __init__, get, and set functions of DecayParticle
           mypart should give the dict as my dict
        """
        
        mypart2 = decay_objects.DecayParticle()

        #To avoid the error raised when setting the vertexlist
        #because of the wrong particle id.
        mypart2.set('pdg_code', self.mydict['pdg_code'])
        for key in self.mydict:
            #Test for the __init__ assign values as mydict
            self.assertEqual(self.mydict[key], self.mypart[key])

            #Test the set function
            mypart2.set(key, self.mydict[key])
            #print key, mypart2[key]
            self.assertEqual(mypart2[key], self.mydict[key])

        for key in self.mypart:
            #Test the get function return the value as in mypart
            self.assertEqual(self.mypart[key], self.mypart.get(key))


    def test_setgetinit_exceptions(self):
        """Test the exceptions raised by __init__, get, and set."""
        
        myNondict = 1.
        myWrongdict = self.mydict
        myWrongdict['Wrongkey'] = 'wrongvalue'

        #Test __init__
        self.assertRaises(AssertionError, decay_objects.DecayParticle,myNondict)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          decay_objects.DecayParticle, myWrongdict)
                          
        #Test get
        self.assertRaises(AssertionError, self.mypart.get, myNondict)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get, 'WrongParameter')
                          
        #Test set
        self.assertRaises(AssertionError, self.mypart.set, myNondict, 1)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set, 'WrongParameter', 1)

    def test_values_for_prop(self):
        """Test filters for DecayParticle properties."""

        test_values = [
                       {'prop':'name',
                        'right_list':['h', 'e+', 'e-', 'u~',
                                      'k++', 'k--', 'T', 'u+~'],
                        'wrong_list':['', 'x ', 'e?', '{}', '9x', 'd~3', 'd+g',
                                      'u~+', 'u~~']},
                       {'prop':'spin',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':[-1, 0, 'a', 6]},
                       {'prop':'color',
                        'right_list':[1, 3, 6, 8],
                        'wrong_list':[2, 0, 'a', 23, -1, -3, -6]},
                       {'prop':'mass',
                        'right_list':['me', 'zero', 'mm2'],
                        'wrong_list':['m+', '', ' ', 'm~']},
                       {'prop':'pdg_code',
                        #The last pdg_code must be 6 to be consistent with
                        #vertexlist
                        'right_list':[1, 12, 80000000, -1, 24],
                        'wrong_list':[1.2, 'a']},
                       {'prop':'line',
                        'right_list':['straight', 'wavy', 'curly', 'dashed'],
                        'wrong_list':[-1, 'wrong']},
                       {'prop':'charge',
                        'right_list':[1., -1., -2. / 3., 0.],
                        'wrong_list':[1, 'a']},
                       {'prop':'propagating',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'is_part',
                        #Restore the is_part to be consistent with vertexlist
                        'right_list':[True, False, True],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'self_antipart',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                      {'prop':'decay_vertexlist',
                        'right_list':[{(2, False):self.my_2bodyvertexlist,
                                       (2, True) :self.my_2bodyvertexlist,
                                       (3, False):self.my_3bodyvertexlist,
                                       (3, True) :self.my_3bodyvertexlist}],
                        'wrong_list':[1, 
                                      {'a': self.my_2bodyvertexlist},
                                      {(24, 2, False): self.my_2bodyvertexlist},
                                      {(5, True):self.my_2bodyvertexlist,
                                       (5, False):self.my_3bodyvertexlist},
                                      {(2, 'Not bool'):self.my_2bodyvertexlist},
                                      {(2, False): 'hey'},
                                      {(2, False): self.my_2bodyvertexlist, 
                                       (2, True) : self.my_3bodyvertexlist},
                                      {(2, False):self.my_2bodyvertexlist_wrongini, 
                                       (2, True): self.my_2bodyvertexlist,
                                       (3, False):self.my_3bodyvertexlist,
                                       (3, True): self.my_3bodyvertexlist},
                                      {(2, False):self.my_2bodyvertexlist, 
                                       (2, True): self.my_2bodyvertexlist,
                                       (3, False):self.my_3bodyvertexlist_wrongini,
                                       (3, True): self.my_3bodyvertexlist},
                                      {(2, False):self.my_2bodyvertexlist, 
                                       (2, True): self.my_2bodyvertexlist,
                                       (3, False):self.my_3bodyvertexlist,
                                       (3, True): self.my_3bodyvertexlist_radiactive}
                                      
                                     ]},
                       ]

        temp_part = self.mypart

        for test in test_values:
            for x in test['right_list']:
                self.assertTrue(temp_part.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_part.set(test['prop'], x))

    def test_getsetvertexlist_correct(self):
        """Test the get and set for vertexlist is correct"""
        temp_part = self.mypart
        #Reset the off-shell '2_body_decay_vertexlist'
        templist = self.my_2bodyvertexlist
        templist.extend(templist)
        temp_part.set_vertexlist(2, False, templist)
        #Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(2, False), \
                             templist)

        #Reset the on-shell '2_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_vertexlist(2, True, templist)
        #Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(2, True), \
                             templist)

        #Reset the off-shell '3_body_decay_vertexlist'
        templist = self.my_3bodyvertexlist
        templist.extend(templist)
        temp_part.set_vertexlist(3, False, templist)
        #Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(3, False), \
                             templist)

        #Reset the on-shell '3_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_vertexlist(3, True, templist)
        #Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(3, True), \
                             templist)

    def test_getsetvertexlist_exceptions(self):
        """Test for the exceptions raised by the get_ or set_vertexlist"""

        #Test of get_ and set_vertexlist
        #Test the exceptions raised from partnum and onshell
        for wrongpartnum in ['string', 1.5, 5]:
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.get_vertexlist, wrongpartnum, True)
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.set_vertexlist, wrongpartnum, True,
                              self.my_2bodyvertexlist, self.my_testmodel)

        for wrongbool in [15, 'NotBool']:           
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.get_vertexlist, 2, wrongbool)
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.set_vertexlist, 3, wrongbool,
                              self.my_3bodyvertexlist, self.my_testmodel)

        

        #Test the exceptions raised from value in set_vertexlist
        #Test for non vertexlist objects
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_vertexlist,
                          2, True, ['not', 'Vertexlist'], self.my_testmodel)

        #Test for vertexlist not consistent with partnum
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_vertexlist,
                          2, True, self.my_3bodyvertexlist, self.my_testmodel)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_vertexlist,
                          3, True, self.my_2bodyvertexlist, self.my_testmodel)

        #Test for vertexlist not consistent with initial particle
        #for both number and id
        #Use the vertexlist from test_getsetvertexlist_exceptions

        Wrong_vertexlist = [self.my_2bodyvertexlist_wrongini,
                            self.my_3bodyvertexlist_wrongini,
                            self.my_3bodyvertexlist_radiactive]

        for item in Wrong_vertexlist:
            for partnum in [2,3]:
                self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError
                             , self.mypart.set_vertexlist, partnum, False, item)
                


    def test_find_vertexlist(self):
        """ Test for the find_vertexlist function and 
            the get_max_vertexorder"""
        #undefine object: my_testmodel, mypart, extra_part
        
        #Test validity of arguments
        #Test if the calling particle is in the model
        extra_part = copy.copy(self.mypart)
        extra_part.set('pdg_code', 2)
        extra_part.set('name', 'u')
        # Test the return of get_max_vertexorder if  vertexlist_found = False
        self.assertEqual(None, extra_part.get_max_vertexorder())

        #print self.my_testmodel
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          extra_part.find_vertexlist, self.my_testmodel)

        #Test if option is boolean
        wronglist=[ 'a', 5, {'key': 9}, [1,5]]
        for wrongarg in wronglist:
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.find_vertexlist, self.my_testmodel,\
                              wrongarg)

        #Import the vertexlist from import_vertexlist
        full_vertexlist = import_vertexlist.full_vertexlist

        #Test the correctness of vertexlist by t quark
        tquark = decay_objects.DecayParticle(self.my_testmodel.get_particle(6))
        tquark.find_vertexlist(self.my_testmodel)
        #Name convention: 'pdg_code'+'particle #'+'on-shell'
        my_vertexlist620 = base_objects.VertexList()
        my_vertexlist621 = base_objects.VertexList([full_vertexlist[(35, 6)]])
        my_vertexlist630 = base_objects.VertexList()
        my_vertexlist631 = base_objects.VertexList()
        rightlist6 = [my_vertexlist620, my_vertexlist621, my_vertexlist630, my_vertexlist631]

        #Test the find_vertexlist for W+
        wboson_p = decay_objects.DecayParticle(self.my_testmodel.get_particle(24))
        wboson_p.find_vertexlist(self.my_testmodel)
        #List must follow the order of interaction id so as to be consistent
        #with the find_vertexlist function
        my_vertexlist2420 = base_objects.VertexList([full_vertexlist[(32, 24)]])
        my_vertexlist2421 = base_objects.VertexList([full_vertexlist[(44, 24)]])
        my_vertexlist2430 = base_objects.VertexList()
        my_vertexlist2431 = base_objects.VertexList()
        #List of the total decay vertex list for W+
        rightlist24 =[my_vertexlist2420, my_vertexlist2421, my_vertexlist2430, my_vertexlist2431]

        #Test the find_vertexlist for A (photon)
        photon = decay_objects.DecayParticle(self.my_testmodel.get_particle(22))
        photon.find_vertexlist(self.my_testmodel)
        #vertex is in the order of interaction id
        my_vertexlist2220 = base_objects.VertexList()
        my_vertexlist2221 = base_objects.VertexList()
        my_vertexlist2230 = base_objects.VertexList()
        my_vertexlist2231 = base_objects.VertexList()
        #List of the total decay vertex list for photon
        rightlist22 =[my_vertexlist2220, my_vertexlist2221, my_vertexlist2230, my_vertexlist2231]

        i=0
        for partnum in [2,3]:
            for onshell in [False, True]:
                self.assertEqual(tquark.get_vertexlist(partnum, onshell),
                                 rightlist6[i])
                self.assertEqual(wboson_p.get_vertexlist(partnum, onshell),
                                 rightlist24[i])
                self.assertEqual(photon.get_vertexlist(partnum, onshell),
                                 rightlist22[i])
                i +=1

        # Test for correct get_max_vertexorder()
        self.assertEqual(0, photon.get_max_vertexorder())
        self.assertEqual(2, tquark.get_max_vertexorder())
        self.mypart.vertexlist_found = True
        self.assertEqual(3, self.mypart.get_max_vertexorder())

    def test_setget_channel(self):
        """ Test of the get_channel set_channel functions (and the underlying
            check_vertexlist.)"""
        # Prepare the channel
        full_vertexlist = import_vertexlist.full_vertexlist

        vert_0 = base_objects.Vertex({'id': 0, 'legs': base_objects.LegList([\
                 base_objects.Leg({'id':25, 'number':1, 'state': False}),
                 base_objects.Leg({'id':25, 'number':2})])})
        vert_1 = copy.deepcopy(full_vertexlist[(40, 25)])
        vert_1['legs'][0]['number'] = 2
        vert_1['legs'][1]['number'] = 3
        vert_1['legs'][2]['number'] = 2
        vert_2 = copy.deepcopy(full_vertexlist[(32, -6)])
        vert_2['legs'][0]['number'] = 2
        vert_2['legs'][1]['number'] = 4
        vert_2['legs'][2]['number'] = 2
        vert_3 = copy.deepcopy(full_vertexlist[(35, 6)])
        vert_3['legs'][0]['number'] = 3
        vert_3['legs'][1]['number'] = 5
        vert_3['legs'][2]['number'] = 3

        h_tt_bbww = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList([
                                           vert_3, vert_2, 
                                           vert_1, vert_0])})
        channellist = decay_objects.ChannelList([h_tt_bbww])

        # Test set and get
        higgs = self.my_testmodel.get_particle(25)
        higgs.set('decay_channels', {(4, True): channellist})
        self.assertEqual(higgs.get('decay_channels'), {(4, True): channellist})

        # Test set_channel and get_channel
        higgs = self.my_testmodel.get_particle(25)
        higgs.set_channels(4, True, [h_tt_bbww])
        self.assertEqual(higgs.get_channels(4, True), channellist)

        # Test for exceptions
        # Wrong final particle number
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          higgs.set_channels, 'non_int', True, [h_tt_bbww])
        # Test from the filter function
        self.assertFalse(higgs.set('decay_channels', 
                                   {('non_int', True): channellist}))
        # Wrong onshell
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          higgs.get_channels, 3, 5)
        # Wrong channel
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          higgs.set_channels, 3, True, ['non', 'channellist'])
        # Wrong initial particle
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          self.my_testmodel.get_particle(24).set_channels, 3,
                          True, [h_tt_bbww])
        # Wrong onshell condition (h is lighter than ww pair)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          higgs.set_channels, 3, True, [h_tt_bbww],
                          self.my_testmodel)
        non_sm = copy.copy(higgs)
        non_sm.set('pdg_code', 26)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError, 
                          higgs.set_channels, 3, False, [h_tt_bbww],
                          self.my_testmodel)
                
#===============================================================================
# TestDecayParticleList
#===============================================================================
class TestDecayParticleList(unittest.TestCase):
    """Test the DecayParticleList"""
    def setUp(self):
        self.mg5_part = base_objects.Particle({'pdg_code':6, 'is_part':True})
        self.mg5_partlist = base_objects.ParticleList([self.mg5_part]*5)

    def test_convert(self):
        #Test the conversion in __init__
        decay_partlist = decay_objects.DecayParticleList(self.mg5_partlist)
        for i in range(0, 5):
            self.assertTrue(isinstance(decay_partlist[i], 
                                       decay_objects.DecayParticle))

        #Test the conversion in append
        decay_partlist.append(self.mg5_part)
        self.assertTrue(isinstance(decay_partlist[-1], 
                                   decay_objects.DecayParticle))
        self.assertTrue(isinstance(decay_partlist,
                                   decay_objects.DecayParticleList))
        
        #Test the conversion in generate_dict
        for num, part in decay_partlist.generate_dict().items():
            self.assertTrue(isinstance(part, decay_objects.DecayParticle))
#===============================================================================
# TestDecayModel
#===============================================================================
class TestDecayModel(unittest.TestCase):
    """Test class for the DecayModel object"""

    base_model = import_ufo.import_model('sm')
    my_testmodel_base = import_ufo.import_model('sm')
    def setUp(self):
        """Set up decay model"""
        #Full SM DecayModel
        self.decay_model = decay_objects.DecayModel(self.base_model)

        #My_small sm DecayModel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        no_want_pid = [1, 2, 3, 4, 13, 14, 15, 16, 21, 23, 25]
        for pid in no_want_pid:
            particles.remove(self.my_testmodel.get_particle(pid))

        for inter in inter_list:
            if any([p.get('pdg_code') in no_want_pid for p in \
                        inter.get('particles')]):
                interactions.remove(inter)

        # Set a new name
        self.my_testmodel.set('name', 'my_smallsm')
        self.my_testmodel.set('particles', particles)
        self.my_testmodel.set('interactions', interactions)

        #import_vertexlist.make_vertexlist(self.my_testmodel)

        #import madgraph.iolibs.export_v4 as export_v4
        #writer = export_v4.UFO_model_to_mg4(self.base_model,'temp')
        #writer.build()

    def test_read_param_card(self):
        """Test reading a param card"""
        param_path = os.path.join(_file_path, '../input_files/param_card_sm.dat')
        self.decay_model.read_param_card(os.path.join(param_path))

        for param in sum([self.base_model.get('parameters')[key] for key \
                              in self.base_model.get('parameters')], []):
            value = eval("decay_objects.%s" % param.name)
            self.assertTrue(isinstance(value, int) or \
                            isinstance(value, float) or \
                            isinstance(value, complex)) 
    
    def test_particles_type(self):
        """Test if the DecayModel can convert the assign particle into
           decay particle"""

        #Test the particle is DecayParticle during generator stage
        #Test the default_setup first
        temp_model = decay_objects.DecayModel()
        self.assertTrue(isinstance(temp_model.get('particles'),
                              decay_objects.DecayParticleList))

        #Test the embeded set in __init__
        self.assertTrue(isinstance(self.decay_model.get('particles'), 
                                   decay_objects.DecayParticleList))

        #Test the conversion into DecayParticle explicitly
        #by the set function
        mg5_particlelist = self.base_model['particles']

        result = self.decay_model.set('particles', mg5_particlelist)

        #Using ParticleList to set should be fine, the result is converted
        #into DecayParticleList.
        self.assertTrue(result)
        self.assertTrue(isinstance(self.decay_model['particles'],
                              decay_objects.DecayParticleList))

        #particle_dict should contain DecayParticle
        self.assertTrue(isinstance(self.decay_model.get('particle_dict')[6],
                                   decay_objects.DecayParticle))

        #Test if the set function returns correctly when assign a bad value
        self.assertFalse(self.decay_model.set('particles', 'NotParticleList'))

        #Test if the particls in interaction is converted to DecayParticle
        self.assertTrue(isinstance(self.decay_model['interactions'][-1]['particles'], decay_objects.DecayParticleList))
                        

    def test_find_vertexlist(self):
        """Test of the find_vertexlist"""

        # Test the exception of get_max_vertexorder
        self.assertEqual(None, self.my_testmodel.get_max_vertexorder())
        self.my_testmodel.find_vertexlist()
        self.my_testmodel.get('particle_dict')[5]['charge'] = 8
        full_vertexlist = import_vertexlist.full_vertexlist_newindex

        for part in self.my_testmodel.get('particles'):
            for partnum in [2, 3]:
                for onshell in [True, False]:
                    #print part.get_pdg_code(), partnum, onshell
                    self.assertEqual(part.get_vertexlist(partnum, onshell),
                                     full_vertexlist[(part.get_pdg_code(),
                                                      partnum, onshell)])

        self.assertEqual(2, self.my_testmodel.get_max_vertexorder())
        # Test the get from particle
        self.assertEqual(2, 
                        self.my_testmodel.get_particle(6).get_max_vertexorder())

    def test_find_mssm_decay_groups_modified_mssm(self):
        """Test finding the decay groups of the MSSM"""

        mssm = import_ufo.import_model('mssm')
        particles = mssm.get('particles')
        no_want_particle_codes = [1000022, 1000023, 1000024, -1000024, 
                                  1000025, 1000035, 1000037, -1000037]
        no_want_particles = [p for p in particles if p.get('pdg_code') in \
                                 no_want_particle_codes]

        for particle in no_want_particles:
            particles.remove(particle)

        interactions = mssm.get('interactions')
        inter_list = copy.copy(interactions)
        for interaction in inter_list:
            if any([p.get('pdg_code') in no_want_particle_codes for p in \
                        interaction.get('particles')]):
                interactions.remove(interaction)
        
        mssm.set('particles', particles)
        mssm.set('interactions', interactions)
        decay_mssm = decay_objects.DecayModel(mssm)

        decay_mssm.find_decay_groups()
        goal_groups = set([(25, 35, 36, 37),
                           (1000001, 1000002, 1000003, 1000004, 1000005, 
                            1000006, 1000021, 2000001, 2000002, 2000003, 
                            2000004, 2000005, 2000006), 
                           (1000011, 1000012), 
                           (1000013, 1000014), 
                           (1000015, 1000016, 2000015), 
                           (2000011,), 
                           (2000013,)])

        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                                group])) \
                                  for group in decay_mssm.decay_groups]),
                         goal_groups)

    def test_find_mssm_decay_groups(self):
        """Test finding the decay groups of the MSSM"""

        mssm = import_ufo.import_model('mssm')
        decay_mssm = decay_objects.DecayModel(mssm)

        decay_mssm.find_decay_groups()
        goal_groups = [[25, 35, 36, 37],
                       [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000021, 1000022, 1000023, 1000024, 1000025, 1000035, 1000037, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000011, 2000013, 2000015]]

        for i, group in enumerate(decay_mssm.decay_groups):
            self.assertEqual(sorted([p.get('pdg_code') for p in group]),
                             goal_groups[i])

    def test_find_mssm_decay_groups_general(self):
        """Test finding the decay groups of the MSSM"""

        mssm = import_ufo.import_model('mssm')
        #decay_mssm = decay_objects.DecayModel(mssm)
        # Read data to find massless SM-like particle
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_mssm.dat')
        decay_mssm.read_param_card(param_path)
        print dir(decay_objects)

        decay_mssm.find_decay_groups_general()
        #print decay_mssm.decay_groups
        goal_groups = [[25, 35, 36, 37],
                       [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000021, 1000022, 1000023, 1000024, 1000025, 1000035, 1000037, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000011, 2000013, 2000015]]

        for i, group in enumerate(decay_mssm.decay_groups):
            print sorted([p.get('pdg_code') for p in group]), 'r'
            #self.assertEqual(sorted([p.get('pdg_code') for p in group]),
             #                goal_groups[i])

        """for inter in decay_mssm.reduced_interactions:
            print inter['id'], [p.get_pdg_code() for p in inter['particles']]"""

        """decay_mssm.find_stable_particles()
        for partlist in decay_mssm.stable_particles:
            print [p.get_pdg_code() for p in partlist]"""

        # Test if all useless interactions are deleted.
        for inter in decay_mssm.reduced_interactions:
            self.assertTrue(len(inter['particles']))


    def test_find_mssm_decay_groups_modified_mssm_general(self):
        """Test finding the decay groups of the MSSM using general way.
           Test to get decay_groups and stable_particles from get."""
        # Setup the mssm with parameters read in.
        mssm = import_ufo.import_model('mssm')
        particles = mssm.get('particles')
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_mssm.dat')
        self.my_testmodel.read_param_card(param_path)

        no_want_particle_codes = [1000022, 1000023, 1000024, -1000024, 
                                  1000025, 1000035, 1000037, -1000037]
        no_want_particles = [p for p in particles if p.get('pdg_code') in \
                                 no_want_particle_codes]

        for particle in no_want_particles:
            particles.remove(particle)

        interactions = mssm.get('interactions')
        inter_list = copy.copy(interactions)
        new_interaction = base_objects.Interaction({\
                'id': len(mssm.get('interactions'))+1,
                'particles': base_objects.ParticleList(
                             [mssm.get_particle(1000001),
                              mssm.get_particle(1000011),
                              mssm.get_particle(1000003),
                              mssm.get_particle(1000013),
                              # This new SM particle should be removed
                              # In the reduction level
                              mssm.get_particle(2000013),
                              mssm.get_particle(1000015)])})
        new_interaction_add_sm = base_objects.Interaction({\
                'id': len(mssm.get('interactions'))+2,
                'particles': base_objects.ParticleList(
                             [mssm.get_particle(25),
                              mssm.get_particle(2000013)])})

        for interaction in inter_list:
            if any([p.get('pdg_code') in no_want_particle_codes for p in \
                        interaction.get('particles')]):
                interactions.remove(interaction)
        
        mssm.set('particles', particles)
        mssm.set('interactions', interactions)

        mssm.get('interactions').append(new_interaction)
        mssm.get('interactions').append(new_interaction_add_sm)
        decay_mssm = decay_objects.DecayModel(mssm)

        mssm_decay_groups = decay_mssm.get('decay_groups')
        goal_groups = set([(25, 35, 36, 37, 2000013),
                           (1000001, 1000002, 1000003, 1000004, 1000005, 
                            1000006, 1000021, 2000001, 2000002, 2000003, 
                            2000004, 2000005, 2000006), 
                           (1000011, 1000012), 
                           (1000013, 1000014), 
                           (1000015, 1000016, 2000015), 
                           (2000011,), (5, 6)
                           ])

        #self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                                #group])) \
                                  #for group in mssm_decay_groups]),
                         #goal_groups)
        # Test if all useless interactions are deleted.
        for inter in decay_mssm.reduced_interactions:
            self.assertTrue(len(inter['particles']))

        """param_path = os.path.join(_file_path,'../input_files/param_card_mssm.dat')
        self.my_testmodel.read_param_card(param_path)

        for p in decay_mssm.get('particles'):
            print '(%s, %d)' % (p.get('pdg_code'), eval(p.get('mass')))"""
        #stable_particles = [


#===============================================================================
# Test_Channel
#===============================================================================
class Test_Channel(unittest.TestCase):
    """ Test for the channel object"""

    my_testmodel_base = import_ufo.import_model('sm')
    my_channel = decay_objects.Channel()
    h_tt_bbmmvv = decay_objects.Channel()

    def setUp(self):
        """ Set up necessary objects for the test"""
        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        # Pids that will be removed
        no_want_pid = [1, 2, 3, 4, 13, 14, 15, 16, 21, 23]
        for pid in no_want_pid:
            particles.remove(self.my_testmodel.get_particle(pid))

        for inter in inter_list:
            if any([p.get('pdg_code') in no_want_pid for p in \
                        inter.get('particles')]):
                interactions.remove(inter)

        # Set a new name
        self.my_testmodel.set('name', 'my_smallsm')
        self.my_testmodel.set('particles', particles)
        self.my_testmodel.set('interactions', interactions)

        #Setup the vertexlist for my_testmodel and save this model (optional)
        import_vertexlist.make_vertexlist(self.my_testmodel)
        #save_model.save_model(os.path.join(MG5DIR, 'tests/input_files', 
        #self.my_testmodel['name']), self.my_testmodel)
    
        full_vertexlist = import_vertexlist.full_vertexlist
        vert_0 = base_objects.Vertex({'id': 0, 'legs': base_objects.LegList([\
                    base_objects.Leg({'id':25, 'number':2}), \
                    base_objects.Leg({'id':25, 'number':1, 'state': False})])})
        vert_1 = copy.deepcopy(full_vertexlist[(40, 25)])
        vert_1['legs'][0]['number'] = 2
        vert_1['legs'][1]['number'] = 3
        vert_1['legs'][2]['number'] = 2
        vert_2 = copy.deepcopy(full_vertexlist[(35, 6)])
        vert_2['id'] = -vert_2['id']
        vert_2['legs'][0]['number'] = 2
        vert_2['legs'][0]['id'] = -vert_2['legs'][0]['id']
        vert_2['legs'][1]['number'] = 4
        vert_2['legs'][1]['id'] = -vert_2['legs'][1]['id']
        vert_2['legs'][2]['number'] = 2
        vert_2['legs'][2]['id'] = -vert_2['legs'][2]['id']
        vert_3 = copy.deepcopy(full_vertexlist[(35, 6)])
        vert_3['legs'][0]['number'] = 3
        vert_3['legs'][1]['number'] = 5
        vert_3['legs'][2]['number'] = 3
        vert_4 = copy.deepcopy(full_vertexlist[(44, 24)])
        vert_4['id'] = -vert_4['id']
        vert_4['legs'][0]['number'] = 4
        vert_4['legs'][0]['id'] = -vert_4['legs'][0]['id']
        vert_4['legs'][1]['number'] = 6
        vert_4['legs'][1]['id'] = -vert_4['legs'][1]['id']
        vert_4['legs'][2]['number'] = 4
        vert_4['legs'][2]['id'] = -vert_4['legs'][2]['id']
        vert_5 = copy.deepcopy(full_vertexlist[(44, 24)])
        vert_5['legs'][0]['number'] = 5
        vert_5['legs'][1]['number'] = 7
        vert_5['legs'][2]['number'] = 5

        #temp_vertices = base_objects.VertexList
        self.h_tt_bbmmvv = decay_objects.Channel({'vertices': \
                                             base_objects.VertexList([
                                             vert_5, vert_4, vert_3, vert_2, \
                                             vert_1, vert_0])})

        #print len(self.h_tt_bbmmvv['vertices'])
        #pic = drawing_eps.EpsDiagramDrawer(self.h_tt_bbmmvv, 'h_tt_bbmmvv', self.my_testmodel)
        #pic.draw()

    def test_get_initialfinal(self):
        """ test the get_initial_id and get_final_legs"""
        # Test the get_initial_id
        self.assertEqual(self.h_tt_bbmmvv.get_initial_id(), 25)
        
        # Test the get_final_legs
        vertexlist = self.h_tt_bbmmvv.get('vertices')
        goal_final_legs = base_objects.LegList([vertexlist[0]['legs'][0],
                                                vertexlist[0]['legs'][1],
                                                vertexlist[1]['legs'][0],
                                                vertexlist[1]['legs'][1],
                                                vertexlist[2]['legs'][0],
                                                vertexlist[3]['legs'][0]])
        self.assertEqual(self.h_tt_bbmmvv.get_final_legs(), goal_final_legs)

    def test_get_onshell(self):
        """ test the get_onshell function"""
        vertexlist = self.h_tt_bbmmvv.get('vertices')
        h_tt_bbww = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList(\
                                           vertexlist[2:])})
        # Test for on shell decay ( h > b b~ mu+ mu- vm vm~)
        self.assertTrue(self.h_tt_bbmmvv.get_onshell(self.my_testmodel))

        # Test for off-shell decay (h > b b~ w+ w-)
        # Raise the mass of higgs
        decay_objects.MH = 220
        self.assertTrue(h_tt_bbww.get_onshell(self.my_testmodel))

    def test_helper_find_channels(self):
        """ Test of the find_channels function of DecayParticle.
            Also the test for some helper function for find_channels."""

        higgs = self.my_testmodel.get_particle(25)
        
        vertexlist = self.h_tt_bbmmvv.get('vertices')
        h_tt_bbww = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList(\
                                           vertexlist[2:])})
        self.my_testmodel.find_vertexlist()
        # Artificially add 4 body decay vertex to w boson
        self.my_testmodel.get_particle(24)['decay_vertexlist'][(4, True)] =\
            vertexlist[2]        
        # The two middle are for wboson (see h_tt_bbww.get_final_legs()).
        goal_configlist = set([(2, 4, 1, 2), (2, 4, 2, 1), (2, 3, 1, 3),
                               (2, 3, 2, 2), (2, 2, 1, 4), (2, 2, 2, 3),
                               (2, 1, 2, 4), (1, 4, 2, 2), (1, 4, 1, 3),
                               (1, 3, 1, 4), (1, 3, 2, 3), (1, 2, 2, 4),])
        self.assertEqual(set([tuple(i) \
                              for i in higgs.generate_configlist(h_tt_bbww,
                                                                 9, 
                                                           self.my_testmodel)]),
                         goal_configlist)
        # Reset the decay_vertexlist of w boson.
        self.my_testmodel.get_particle(24)['decay_vertexlist'].pop((4, True))


        # Test the connect_channel_vertex
        h_tt_bwbmuvm = decay_objects.Channel({'vertices': \
                                              base_objects.VertexList(\
                                              vertexlist[1:])})
        #int_leg = h_tt_bbww.get_final_legs()[-1]
        #print int_leg, h_tt_bbww.get_final_legs()
        w_muvm = self.my_testmodel.get_particle(24).get_vertexlist(2, True)[0]
        #print w_muvm
        new_channel = higgs.connect_channel_vertex(h_tt_bbww, 3, w_muvm,
                                                   self.my_testmodel)
        #print 'c1:', new_channel.nice_string(), '\nc2:', h_tt_bwbmuvm.nice_string(), '\n'
        #print self.h_tt_bbmmvv.nice_string()
        h_tt_bwbmuvm.get_onshell(self.my_testmodel)
        self.assertEqual(new_channel, h_tt_bwbmuvm)
        

        # Test of check_idlegs
        temp_vert = copy.deepcopy(vertexlist[2])
        temp_vert['legs'].insert(2, temp_vert['legs'][1])
        temp_vert['legs'][2]['number'] = 4
        #print temp_vert
        self.assertFalse(decay_objects.Channel.check_idlegs(vertexlist[2]))
        self.assertEqual(decay_objects.Channel.check_idlegs(temp_vert),
                         [(35, 24, [1, 2])])

        # Test of get_idpart
        temp_vert2 = copy.deepcopy(temp_vert)
        temp_vert2['legs'].insert(3, temp_vert['legs'][0])
        temp_vert2['legs'].insert(4, temp_vert['legs'][0])
        #print temp_vert2
        idpart_c = decay_objects.Channel({'vertices': \
                              base_objects.VertexList([temp_vert])})
        idpart_c = higgs.connect_channel_vertex(idpart_c, 1, temp_vert2, 
                                                self.my_testmodel)
        self.assertEqual(idpart_c.get_idpartlist(),
                         {1: [(35, 24,[1, 2])], 
                          0: [(35, 24,[1, 2]), (35, 5, [0,3,4])]})
        self.assertTrue(idpart_c.get('has_idpart'))

    def test_findchannels(self):
        """ Test of the find_channels functions."""

        higgs = self.my_testmodel.get_particle(25)
        # Test exceptions of find_channels
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          'non_int', self.my_testmodel)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          4, higgs)
        non_sm = copy.copy(higgs)
        non_sm.set('pdg_code', 26)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          non_sm, self.my_testmodel)
        

        # Test of find_channels
        # Without running find_vertexlist before, but the program should run it
        # automatically.
        higgs.find_channels(6, self.my_testmodel)
        #print higgs.get('decay_channels').keys()
        result = higgs.get_channels(5, True)
        print result.nice_string()
        #self.assertEqual(result, self.h_tt_bbmmvv)
        
                                           


if __name__ == '__main__':
    unittest.unittest.main()
