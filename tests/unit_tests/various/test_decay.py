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
import math
import cmath

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import models.import_ufo as import_ufo
import models.model_reader as model_reader
import madgraph.iolibs.save_model as save_model
import madgraph.iolibs.drawing_eps as drawing_eps
import madgraph.iolibs.import_v4 as import_v4
from madgraph import MG5DIR
import mg5decay.decay_objects as decay_objects
import tests.input_files.import_vertexlist as import_vertexlist
import madgraph.core.diagram_generation as diagram_generation
import madgraph.various.diagram_symmetry as diagram_symmetry


_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# DecayParticleTest
#===============================================================================
class Test_DecayParticle(unittest.TestCase):
    """Test class for the DecayParticle object"""

    mydict = {}
    mypart = None
    my_2bodyvertexlist = base_objects.VertexList()
    my_3bodyvertexlist = base_objects.VertexList()
    my_2bodyvertexlist_wrongini = base_objects.VertexList()
    my_3bodyvertexlist_wrongini = base_objects.VertexList()

    def setUp(self):

        #Import a model from my_testmodel
        self.sm_path = import_ufo.find_ufo_path('sm')
        self.my_testmodel_base = import_ufo.import_model(self.sm_path)
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base, 
                                                     True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)
        #print len(self.my_testmodel_base.get('interactions')), 
        #len(self.my_testmodel.get('interactions'))


        # Simplify the model

        particles = self.my_testmodel.get('particles')
        #print 'Here\n', self.my_testmodel['particles']
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
        #save_model.save_model(os.path.join(MG5DIR, 'tests/input_files', 
        #self.my_testmodel['name']), self.my_testmodel)


        # Setup vertexlist for test

        full_vertexlist = import_vertexlist.full_vertexlist

        # Take w+ > t b~ and w+ > e+ ve (correct ones)
        #      w- > t~ b and t~ > w- b~ (wrong initial particle)
        self.my_2bodyvertexlist = base_objects.VertexList()
        self.my_2bodyvertexlist_wrongini = base_objects.VertexList()
        self.my_3bodyvertexlist = base_objects.VertexList()
        self.my_3bodyvertexlist_wrongini = base_objects.VertexList()
        self.my_3bodyvertexlist_radiactive = base_objects.VertexList()

        for index, vertex in full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            if legs_set == set([-5, 6, 24]) :
                self.my_2bodyvertexlist.append(vertex)
            elif legs_set == set([-11, 12, 24]):
                self.my_2bodyvertexlist.append(vertex)
            elif legs_set == set([-6, 5, -24]):
                self.my_2bodyvertexlist_wrongini.append(vertex)
            elif legs_set == set([-5, -24, -6]):
                self.my_2bodyvertexlist_wrongini.append(vertex)

        # Artificial 3-body vertices
        fake_vertex = copy.deepcopy(self.my_2bodyvertexlist[0])
        fake_vertex['legs'].append(base_objects.Leg({'id':22}))
        fake_vertex2 = copy.deepcopy(self.my_2bodyvertexlist[1])
        fake_vertex2['legs'].append(base_objects.Leg({'id': 11}))
        self.my_3bodyvertexlist.extend([fake_vertex, fake_vertex2])

        # Artificial 3-body vertices with wrong initial particle
        fake_vertex3 = copy.deepcopy(self.my_2bodyvertexlist_wrongini[0])
        fake_vertex3['legs'].append(base_objects.Leg({'id':12}))
        self.my_3bodyvertexlist_wrongini.append(fake_vertex3)

        # Artificial 3-body vertices with radiation
        fake_vertex4 = copy.deepcopy(self.my_2bodyvertexlist[0])
        fake_vertex4['legs'].append(base_objects.Leg({'id':24}))
        self.my_3bodyvertexlist_radiactive.append(fake_vertex4)

        
        # Testing particle
        self.mydict = {'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      #'texname':'W+',
                      #'antitexname':'W-',
                      'line':'wavy',
                      'charge': 1.00,
                      'pdg_code': 24,
                      #'propagating':True,
                      'is_part': True,
                      'self_antipart': False,
                       # decay_vertexlist must have two lists, one for on-shell,
                       # one for off-shell
                      'decay_vertexlist': {\
                           (2, False): self.my_2bodyvertexlist,
                           (2, True) : self.my_2bodyvertexlist,
                           (3, False): self.my_3bodyvertexlist,
                           (3, True) : self.my_3bodyvertexlist},
                       'is_stable': False,
                       'vertexlist_found': False,
                       'max_vertexorder': 0,
                       'apx_decaywidth': 0.,
                       'apx_decaywidth_err': 0.
                       }

        self.mypart = decay_objects.DecayParticle(self.mydict)



    def test_setgetinit_correct(self):
        """Test __init__, get, and set functions of DecayParticle
           mypart should give the dict as my dict
        """
        
        mypart2 = decay_objects.DecayParticle()

        # To avoid the error raised when setting the vertexlist
        # because of the wrong particle id.
        mypart2.set('pdg_code', self.mydict['pdg_code'])
        for key in self.mydict:
            # Test for the __init__ assign values as mydict
            self.assertEqual(self.mydict[key], self.mypart[key])

            # Test the set function
            mypart2.set(key, self.mydict[key])
            #print key, mypart2[key]
            self.assertEqual(mypart2[key], self.mydict[key])

        for key in self.mypart:
            # Test the get function return the value as in mypart
            # Note: for apx_decaywidth_err, .get will call 
            # estimate_decaywidth_error to recalculate
            # so the result will be one for zero width
            self.assertEqual(self.mypart.get(key), self.mypart[key])


    def test_setgetinit_exceptions(self):
        """Test the exceptions raised by __init__, get, and set."""
        
        myNondict = 1.
        myWrongdict = self.mydict
        myWrongdict['Wrongkey'] = 'wrongvalue'

        # Test __init__
        self.assertRaises(AssertionError, decay_objects.DecayParticle,myNondict)
        # Do not check if the input dict has wrong dict!
        #self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
        #                  decay_objects.DecayParticle, myWrongdict)
                          
        # Test get
        self.assertRaises(AssertionError, self.mypart.get, myNondict)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get, 'WrongParameter')
                          
        # Test set
        self.assertRaises(AssertionError, self.mypart.set, myNondict, 1)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set, 'WrongParameter', 1)

    def test_values_for_prop(self):
        """Test filters for DecayParticle properties."""

        test_values = [
                       {'prop':'name',
                        'right_list':['h', 'e+', 'e-', 'u~',
                                      'k++', 'k--', 'T', 'u+~'],
                        'wrong_list':['', 'x ', 'e?', '{}']},
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
                       #{'prop':'propagating',
                       # 'right_list':[True, False],
                       # 'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'is_part',
                        #Restore the is_part to be consistent with vertexlist
                        'right_list':[True, False, True],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'self_antipart',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'is_stable',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'vertexlist_found',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'max_vertexorder',
                        'right_list':[3, 4, 0],
                        'wrong_list':['a', 'true', None]},
                       {'prop':'apx_decaywidth',
                        'right_list':[3., 4.5, 0.2],
                        'wrong_list':['a', [12,2], None]},
                       {'prop':'apx_decaywidth_err',
                        'right_list':[3., 4.5, 0.2],
                        'wrong_list':['a', [12,2], None]},
                       {'prop':'2body_massdiff',
                        'right_list':[3., 4.5, 0.2],
                        'wrong_list':['a', [12,2], None]},
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

        # Reset the off-shell '2_body_decay_vertexlist'
        templist = self.my_2bodyvertexlist
        templist.extend(templist)
        temp_part.set_vertexlist(2, False, templist)

        # Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(2, False),
                         templist)

        # Test the result for keys don't exist
        self.assertEqual(temp_part.get_vertexlist(4, True),
                         [])

        # Reset the on-shell '2_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_vertexlist(2, True, templist)

        # Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(2, True),
                         templist)


        # Reset the off-shell '3_body_decay_vertexlist'
        templist = self.my_3bodyvertexlist
        templist.extend(templist)
        temp_part.set_vertexlist(3, False, templist)

        # Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(3, False),
                         templist)


        # Reset the on-shell '3_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_vertexlist(3, True, templist)

        # Test for equality from get_vertexlist
        self.assertEqual(temp_part.get_vertexlist(3, True),
                         templist)

    def test_getsetvertexlist_exceptions(self):
        """Test for the exceptions raised by the get_ or set_vertexlist"""

        # Test of get_ and set_vertexlist
        # Test the exceptions raised from partnum and onshell
        for wrongpartnum in ['string', 1.5]:
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

        

        # Test the exceptions raised from value in set_vertexlist
        # Test for non vertexlist objects
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

        # Test for vertexlist not consistent with initial particle
        # for both number and id
        # Use the vertexlist from test_getsetvertexlist_exceptions

        Wrong_vertexlist = [self.my_2bodyvertexlist_wrongini,
                            self.my_3bodyvertexlist_wrongini,
                            self.my_3bodyvertexlist_radiactive]

        for item in Wrong_vertexlist:
            for partnum in [2,3]:
                self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError
                             , self.mypart.set_vertexlist, partnum, False, item)
                

    def test_reset_decaywidth(self):
        """ Test for the reset and update of apx_decaywidth, branching ratio
            for amplitudes, and apx_decaywidth_err."""

        self.my_testmodel.find_all_channels(3)
        tquark = self.my_testmodel.get_particle(6)
        amp = tquark.get_amplitudes(2)[0]
        old_width = amp.get('apx_decaywidth')

        # Add a new channel to tquark
        tquark.get_amplitudes(2).append(amp)

        # Total width doubles. Update the width, br should remain the same.
        amp.get('apx_br')
        tquark.update_decay_attributes(True, True, False)
        self.assertAlmostEqual(tquark.get('apx_decaywidth'),old_width*2)
        self.assertAlmostEqual(amp.get('apx_br'), 1)

        # Update the br now. It should be 0.5.
        tquark.update_decay_attributes(False, True, True)
        self.assertAlmostEqual(tquark.get('apx_decaywidth'),old_width*2)
        self.assertAlmostEqual(amp.get('apx_br'), 0.5)

        # Test the estimate_width_error
        w = self.my_testmodel.get_particle(24)
        width_err = w.get('apx_decaywidth_err')
        offshell_clist = w.get_channels(3, False)
        w.get_channels(3, False).extend(offshell_clist)

        # Test if the width_err doubles after the update
        w.update_decay_attributes(False, True, False)
        self.assertAlmostEqual(w.get('apx_decaywidth_err'),width_err*2)

    def test_find_vertexlist(self):
        """ Test for the find_vertexlist function and 
            the get_max_vertexorder"""
        
        # Test validity of arguments
        # Test if the calling particle is in the model
        extra_part = copy.copy(self.mypart)
        extra_part.set('pdg_code', 2)
        extra_part.set('name', 'u')


        # Test the return of get_max_vertexorder if  vertexlist_found = False
        self.assertEqual(None, extra_part.get_max_vertexorder())

        #print self.my_testmodel
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          extra_part.find_vertexlist, self.my_testmodel)


        # Test if option is boolean
        wronglist=[ 'a', 5, {'key': 9}, [1,5]]
        for wrongarg in wronglist:
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              self.mypart.find_vertexlist, self.my_testmodel,\
                              wrongarg)


        # Test find_vertexlist in t quark, w+ boson, photon

        tquark = decay_objects.DecayParticle(self.my_testmodel.get_particle(6),
                                             True)
        tquark.find_vertexlist(self.my_testmodel)

        wboson_p = decay_objects.DecayParticle(\
            self.my_testmodel.get_particle(24), True)
        wboson_p.find_vertexlist(self.my_testmodel)

        # photon is known as stable after stable particles are found internally
        # in find_vertexlist
        photon = decay_objects.DecayParticle(self.my_testmodel.get_particle(22),
                                             True)
        photon.find_vertexlist(self.my_testmodel)

        # Get vertex list
        blank_vlist = base_objects.VertexList()
        # t > b w+ (t~ b w+)
        top_vlist_2_on = base_objects.VertexList()        
        # w+ > t b~
        w_vlist_2_on = base_objects.VertexList()
        # w+ > e+ ve        
        w_vlist_2_off = base_objects.VertexList()

        for index, vertex in import_vertexlist.full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            if legs_set == set([5, 24, 6]) :
                top_vlist_2_on.append(vertex)
            elif legs_set == set([-11, 12, 24]):
                w_vlist_2_on.append(vertex)
            elif legs_set == set([-5, 6, 24]):
                w_vlist_2_off.append(vertex)
        
        rightlist_top = [blank_vlist, top_vlist_2_on, blank_vlist, blank_vlist]
        rightlist_w =[w_vlist_2_off, w_vlist_2_on, blank_vlist, blank_vlist] 
        rightlist_a =[blank_vlist, blank_vlist, blank_vlist, blank_vlist]

        i=0
        for partnum in [2,3]:
            for onshell in [False, True]:
                self.assertEqual(tquark.get_vertexlist(partnum, onshell),
                                 rightlist_top[i])
                self.assertEqual(wboson_p.get_vertexlist(partnum, onshell),
                                 rightlist_w[i])
                self.assertEqual(photon.get_vertexlist(partnum, onshell),
                                 rightlist_a[i])
                i +=1


        # Test get_max_vertexorder()
        self.assertEqual(0, photon.get_max_vertexorder())
        self.assertEqual(2, tquark.get_max_vertexorder())
        self.mypart['vertexlist_found'] = True
        self.assertEqual(3, self.mypart.get_max_vertexorder())


    def test_setget_channel(self):
        """ Test of the get_channel set_channel functions (and the underlying
            check_channels.)"""

        # Prepare the channel
        full_vertexlist = import_vertexlist.full_vertexlist

        vert_0 = base_objects.Vertex({'id': 0, 'legs': base_objects.LegList([\
                 base_objects.Leg({'id':25, 'number':1, 'state': False}),
                 base_objects.Leg({'id':25, 'number':2})])})

        for index, vertex in import_vertexlist.full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            # h > t t~
            if legs_set == set([-6, 6, 25]) :
                vert_1 = copy.deepcopy(vertex)
            # t > b w+ = (t~ b w+)
            elif legs_set == set([-5, -24, -6]):
                vert_2 = copy.deepcopy(vertex)
            # t~ > b~ w- = (t b~ w-)
            elif legs_set == set([5, 24, 6]):
                vert_3 = copy.deepcopy(vertex)

        vert_1['legs'][0]['number'] = 2
        vert_1['legs'][1]['number'] = 3
        vert_1['legs'][2]['number'] = 2

        vert_2['legs'][0]['number'] = 2
        vert_2['legs'][1]['number'] = 4
        vert_2['legs'][2]['number'] = 2

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


    def test_get_max_level(self):
        """ Test the get_max_level function. """

        higgs = self.my_testmodel.get_particle(25)
        higgs.find_channels(2, self.my_testmodel)
        self.assertEqual(higgs.get_max_level(), 2)
        higgs.find_channels_nextlevel(self.my_testmodel)
        self.assertEqual(higgs.get_max_level(), 3)
        higgs.find_channels_nextlevel(self.my_testmodel)
        higgs.set_amplitudes(4, decay_objects.DecayAmplitudeList())
        self.assertEqual(higgs.get_max_level(), 4)

#===============================================================================
# TestDecayParticleList
#===============================================================================
class Test_DecayParticleList(unittest.TestCase):
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
class Test_DecayModel(unittest.TestCase):
    """Test class for the DecayModel object"""

    def setUp(self):
        """Set up decay model"""
        
        if not hasattr(Test_DecayModel, 'base_model'):
            Test_DecayModel.base_model = import_ufo.import_model('mssm')
            Test_DecayModel.my_testmodel_base = import_ufo.import_model('sm')
        
        #Full SM DecayModel
        self.decay_model = decay_objects.DecayModel(self.base_model, True)

        #My_small sm DecayModel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base, True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        no_want_pid = [1, 2, 3, 4, 13, 14, 15, 16, 21, 23, 25]
        for pid in no_want_pid:
            particles.remove(self.my_testmodel.get_particle(pid))

        for particle in particles:
            particle.set('charge', 0)

        for inter in inter_list:
            if any([p.get('pdg_code') in no_want_pid for p in \
                        inter.get('particles')]):
                interactions.remove(inter)

        # Set a new name
        self.my_testmodel.set('name', 'my_smallsm')
        self.my_testmodel.set('particles', particles)
        self.my_testmodel.set('interactions', interactions)

        import_vertexlist.make_vertexlist(self.my_testmodel)

        #import madgraph.iolibs.export_v4 as export_v4
        #writer = export_v4.UFO_model_to_mg4(self.base_model,'temp')
        #writer.build()

    def test_read_param_card(self):
        """Test reading a param card"""
        param_path = os.path.join(_file_path, '../input_files/param_card_mssm.dat')
        self.decay_model.read_param_card(os.path.join(param_path))

        for param in sum([self.base_model.get('parameters')[key] for key \
                              in self.base_model.get('parameters')], []):
            if param.name:
                value = eval("decay_objects.%s" % param.name)
                self.assertTrue(isinstance(value, int) or \
                                    isinstance(value, float) or \
                                    isinstance(value, complex))

    def test_setget(self):
        """ Test the set and get for special properties"""

        self.my_testmodel.set('vertexlist_found', True)
        self.assertEqual(self.my_testmodel.get('vertexlist_found'), True)

        self.my_testmodel.set('vertexlist_found', False)
        self.assertRaises(self.my_testmodel.PhysicsObjectError,
                          self.my_testmodel.filter, 'max_vertexorder', 'a')
        self.assertRaises(self.my_testmodel.PhysicsObjectError,
                          self.my_testmodel.filter, 'stable_particles', 
                          [self.my_testmodel.get('particles'), ['a']])
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.filter, 'vertexlist_found', 4)
                          

    def test_particles_type(self):
        """Test if the DecayModel can convert the assign particle into
           decay particle"""

        # Test the particle is DecayParticle during generator stage
        # Test the default_setup first
        temp_model = decay_objects.DecayModel()
        self.assertTrue(isinstance(temp_model.get('particles'),
                              decay_objects.DecayParticleList))


        # Test the embeded set in __init__
        self.assertTrue(isinstance(self.decay_model.get('particles'), 
                                   decay_objects.DecayParticleList))


        # Test the conversion into DecayParticle explicitly
        # by the set function
        mg5_particlelist = self.base_model['particles']

        result = self.decay_model.set('particles', mg5_particlelist, True)


        # Using ParticleList to set should be fine, the result is converted
        # into DecayParticleList.
        self.assertTrue(result)
        self.assertTrue(isinstance(self.decay_model['particles'],
                              decay_objects.DecayParticleList))


        # particle_dict should contain DecayParticle
        self.assertTrue(isinstance(self.decay_model.get('particle_dict')[6],
                                   decay_objects.DecayParticle))


        # Test if the set function returns correctly when assign a bad value
        try:
            self.assertFalse(self.decay_model.set('particles', 
                                                  'NotParticleList'))
        except:
            self.assertRaises(AssertionError, 
                              self.decay_model.set, 
                              'particles', 'NotParticleList')


        # Test if the particls in interaction is converted to DecayParticle
        self.assertTrue(isinstance(self.decay_model['interactions'][-1]['particles'], decay_objects.DecayParticleList))
                        

    def test_find_vertexlist(self):
        """Test of the find_vertexlist"""

        # Set the mass of bottom to be nonzero for the test of radiation
        self.my_testmodel.get_particle(5)['mass'] = '4.7'
        # Test the exception of get_max_vertexorder
        self.assertEqual(None, self.my_testmodel.get_max_vertexorder())


        # Test find vertex exception when vertices are repeated
        # (vertices already be stored in DecayParicles in the 1st search)
        self.my_testmodel.find_vertexlist()
        self.my_testmodel['vertexlist_found'] = False
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.find_vertexlist)

        # Ignore it by specifying it in argument.
        self.my_testmodel['vertexlist_found'] = False
        self.my_testmodel.find_vertexlist(True)

        # Reset everything
        self.my_testmodel['vertexlist_found'] = False
        for p in self.my_testmodel['particles']:
            p['decay_vertexlist'] = {}



        # Test find vertexlist
        self.my_testmodel.find_vertexlist()
        empty = []

        # General test by comparing to import_vertexlist
        full_vertexlist = import_vertexlist.full_vertexlist_newindex
        for part in self.my_testmodel.get('particles'):
            for partnum in [2, 3]:
                for onshell in [True, False]:
                    #print part.get_pdg_code(), partnum, onshell
                    # Use get(key, default) to prevent keys do not exist
                    result = part.get_vertexlist(partnum, onshell)
                    goal = full_vertexlist.get((part.get_pdg_code(),
                                                partnum, onshell), empty)

                    self.assertEqual(result, goal)

                    # Test the order of legs
                    for vert in part.get_vertexlist(partnum, onshell):
                        lids = [l['id'] for l in vert['legs'][:-1]]
                        goal_lids = sorted(lids, reverse= True)
                        self.assertEqual(goal_lids, lids)
                    
        # Specific test to top, w+, photon (very similar to another 
        # test_find_vertexlist in DecayParticle)
        tquark = decay_objects.DecayParticle(self.my_testmodel.get_particle(6),
                                             True)
        wboson_p = decay_objects.DecayParticle(\
            self.my_testmodel.get_particle(24), True)
        photon = decay_objects.DecayParticle(self.my_testmodel.get_particle(22),
                                             True)

        # Get vertex list
        # t > b w+ (t~ b w+)
        top_vlist_2_on = base_objects.VertexList()        
        # w+ > t b~
        w_vlist_2_on = base_objects.VertexList()
        # w+ > e+ ve        
        w_vlist_2_off = base_objects.VertexList()

        for index, vertex in import_vertexlist.full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            if legs_set == set([5, 24, 6]) :
                top_vlist_2_on.append(vertex)
            elif legs_set == set([-11, 12, 24]):
                w_vlist_2_on.append(vertex)
            elif legs_set == set([-5, 6, 24]):
                w_vlist_2_off.append(vertex)
        
        rightlist_top = [empty, top_vlist_2_on, empty, empty]
        rightlist_w =[w_vlist_2_off, w_vlist_2_on, empty, empty] 
        rightlist_a =[empty, empty, empty, empty]

        i=0
        for partnum in [2,3]:
            for onshell in [False, True]:
                self.assertEqual(tquark.get_vertexlist(partnum, onshell),
                                 rightlist_top[i])
                self.assertEqual(wboson_p.get_vertexlist(partnum, onshell),
                                 rightlist_w[i])
                self.assertEqual(photon.get_vertexlist(partnum, onshell),
                                 rightlist_a[i])
                i +=1

        return




    def test_find_mssm_decay_groups(self):
        """Test finding the decay groups of the MSSM"""

        mssm = import_ufo.import_model('mssm')
        decay_mssm = decay_objects.DecayModel(mssm, True)
        decay_mssm.find_decay_groups()
        goal_groups = [[25, 35, 36, 37],
                       [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000021, 1000022, 1000023, 1000024, 1000025, 1000035, 1000037, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000011, 2000013, 2000015]]

        # find_decay_groups_general should be run automatically
        for i, group in enumerate(decay_mssm['decay_groups']):
            self.assertEqual(sorted([p.get('pdg_code') for p in group]),
                             goal_groups[i])

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
        decay_mssm = decay_objects.DecayModel(mssm, True)

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
                                  for group in decay_mssm['decay_groups']]),
                         goal_groups)


    def test_find_mssm_decay_groups_general(self):
        """Test finding the decay groups of the MSSM"""

        mssm = import_ufo.import_model('mssm')
        decay_mssm = decay_objects.DecayModel(mssm, True)
        # Read data to find massless SM-like particle
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_mssm.dat')
        decay_mssm.read_param_card(param_path)

        # In first group, 15 and from 23 are calculated,
        # others are massless default.
        goal_groups = [[1,2,3,4,11,12,13,14, 15, 16,21,22, 23, 24, 25, 35, 36, 37], 
                       [1000001, 1000002, 1000003, 1000004, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000021, 1000022, 1000023, 1000024, 1000025, 1000035, 1000037, 2000001, 2000002, 2000003, 2000004, 2000011, 2000013, 2000015],
                       [1000005, 1000006, 2000005, 2000006], 
                       [5, 6]]
        goal_stable_particle_ids = set([(1,2,3,4,11,12,13,14,16,21,22),
                                        (5,),
                                        (1000022,)])
        for i, group in enumerate(decay_mssm.get('decay_groups')):
            pdg = sorted([p.get('pdg_code') for p in group])
            self.assertTrue( pdg in goal_groups)
        self.assertEqual(len(goal_groups), i+1)
            
        # Test if all useless interactions are deleted.
        for inter in decay_mssm['reduced_interactions']:
            self.assertTrue(len(inter['particles']))

        # Reset decay_groups, test the auto run from find_stable_particles
        decay_mssm['decay_groups'] = []
        
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in plist])) for plist in decay_mssm.get('stable_particles')]), goal_stable_particle_ids)

            

    def test_find_mssm_decay_groups_modified_mssm_general(self):
        """Test finding the decay groups of the MSSM using general way.
           Test to get decay_groups and stable_particles from get."""

        # Setup the mssm with parameters read in.
        mssm = import_ufo.import_model('mssm')
        decay_mssm = decay_objects.DecayModel(mssm, True)
        particles = decay_mssm.get('particles')
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_mssm.dat')
        decay_mssm.read_param_card(param_path)
        
        # Set no want particles
        no_want_particle_codes = [1000022, 1000023, 1000024, -1000024, 
                                  1000025, 1000035, 1000037, -1000037]
        no_want_particles = [p for p in particles if p.get('pdg_code') in \
                                 no_want_particle_codes]

        for particle in no_want_particles:
            particles.remove(particle)

        interactions = decay_mssm.get('interactions')
        inter_list = copy.copy(interactions)

        for interaction in inter_list:
            if any([p.get('pdg_code') in no_want_particle_codes for p in \
                        interaction.get('particles')]):
                interactions.remove(interaction)
        
        decay_mssm.set('particles', particles)
        decay_mssm.set('interactions', interactions)

        # Set sd4, sd5 quark mass the same as b quark, so that 
        # degeneracy happens and can be tested 
        # (both particle and anti-particle must be changed)
        # This reset of particle mass must before the reset of particles
        # so that the particles of all interactions can change simutaneuosly.
        decay_mssm.get_particle(2000003)['mass'] = \
            decay_mssm.get_particle(5).get('mass')
        decay_mssm.get_particle(2000001)['mass'] = \
            decay_mssm.get_particle(5).get('mass')
        decay_mssm.get_particle(1000012)['mass'] = \
            decay_mssm.get_particle(1000015).get('mass')

        decay_mssm.get_particle(-2000003)['mass'] = \
            decay_mssm.get_particle(5).get('mass')
        decay_mssm.get_particle(-2000001)['mass'] = \
            decay_mssm.get_particle(5).get('mass')


        # New interactions that mix different groups
        new_interaction = base_objects.Interaction({\
                'id': len(decay_mssm.get('interactions'))+1,
                'particles': base_objects.ParticleList(
                             [decay_mssm.get_particle(1000001),
                              decay_mssm.get_particle(1000011),
                              decay_mssm.get_particle(1000003),
                              decay_mssm.get_particle(1000013),
                              # This new SM particle should be removed
                              # In the reduction level
                              decay_mssm.get_particle(2000013),
                              decay_mssm.get_particle(1000015)])})
        new_interaction_add_sm = base_objects.Interaction({\
                'id': len(decay_mssm.get('interactions'))+2,
                'particles': base_objects.ParticleList(
                             [decay_mssm.get_particle(25),
                              decay_mssm.get_particle(2000013)])})

        decay_mssm.get('interactions').append(new_interaction)
        decay_mssm.get('interactions').append(new_interaction_add_sm)

        goal_groups = set([(1,2,3,4,11,12,13,14, 15, 16,21,22,
                            23, 24, 25, 35, 36, 37, 2000013), # 15 and from 23
                           # are calculated, others are massless default
                           (1000005, 1000006, 2000005, 2000006),
                           (1000015, 1000016, 2000015),                        
                           (1000001, 1000002, 1000003, 1000004, 
                            1000021, 2000001, 2000002, 2000003, 2000004),
                           (5, 6),
                           (1000011, 1000012), 
                           (1000013, 1000014), 
                           # 2000013 originally should be here, but the
                           # the new_interaction_add_sm change it to SM group
                           (2000011,)
                           ])

        # the stable_candidates that should appear in 1st stage of
        # find stable_particles
        goal_stable_candidates = [[1,2,3,4,11,12,13,14,16,21,22],
                                  [1000006], [1000015], [2000001, 2000003],
                                  [5], [1000012], [1000014],[2000011]]
        # Group 1,3,4 mixed; group 2, 5, 6 mixed
        goal_stable_particle_ids = set([(1,2,3,4,11,12,13,14,16,21,22),
                                        (1000015,),
                                        # 5 mass = squark
                                        # will be set later
                                        (2000001, 2000003),
                                        (5,),
                                        (1000012,),
                                        (1000014,),
                                        # all sleptons are combine
                                        (2000011,)])

        # Get the decay_groups (this should run find_decay_groups_general)
        # automatically.
        mssm_decay_groups = decay_mssm.get('decay_groups')

        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                            group])) \
                              for group in mssm_decay_groups]),
                         goal_groups)
 

        # Test if all useless interactions are deleted.
        for inter in decay_mssm['reduced_interactions']:
            self.assertTrue(len(inter['particles']))


        # Test stable particles
        # Reset the decay_groups, test the auto-run of find_decay_groups_general
        decay_mssm['decay_groups'] = []
        decay_mssm.find_stable_particles()

        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in plist])) for plist in decay_mssm['stable_particles']]), goal_stable_particle_ids)
        

        # Test the assignment of is_stable to particles
        goal_stable_pid = [1,2,3,4,5,11,12,13,14,16,21,22,1000012,1000014,
                           1000015, 2000001, 2000003, 2000011]
        self.assertEqual(sorted([p.get_pdg_code() \
                                     for p in decay_mssm.get('particles') \
                                     if p.get('is_stable')]), goal_stable_pid)
        self.assertTrue(decay_mssm.get_particle(-goal_stable_pid[0]).get('is_stable'))


        # Test the advance search of stable particles
        for p in decay_mssm['particles']:
            p['is_stable'] = False
        decay_mssm['stable_particles'] = []
        decay_mssm.find_stable_particles_advance()
        self.assertEqual(sorted([p.get_pdg_code() \
                                     for p in decay_mssm.get('particles') \
                                     if p.get('is_stable')]), goal_stable_pid)

        goal_stable_particles_ad = set([(1,),(2,),(3,),(4,),(5,),
                                        (11,),(12,),(13,),(14,),(16,),
                                        (21,),(22,),
                                        (1000012,),(1000014,),(1000015,),
                                        (2000001,),( 2000003,),( 2000011,)])
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in plist])) for plist in decay_mssm['stable_particles']]), goal_stable_particles_ad)


        
    def test_find_full_sm_decay_groups(self):
        """ Test the algorithm in find stable particle in full sm.
            First, test the full sm with massive neutrinos.
            Second, set the neutrinos as massless."""

        # Import the full sm with param_card
        sm_path = import_ufo.find_ufo_path('sm')
        full_sm_base = import_ufo.import_full_model(sm_path)
        full_sm = decay_objects.DecayModel(full_sm_base, True)
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_full_sm.dat')
        full_sm.read_param_card(param_path)
        
        # Stage 1: Find stable particles with 
        #          nonzero neutrino masses, quark masses
        # Turned on light quark and neutrino masses
        neutrinos = [12, 14, 16]
        lightquarks1 = [1, 3]
        for npid in neutrinos:
            full_sm.get_particle(npid)['mass'] = '1E-6'
            full_sm.get_particle(-npid)['mass'] = '1E-6'
        for qpid in lightquarks1:
            full_sm.get_particle(qpid)['mass'] = '5'
        full_sm.get_particle(2)['mass'] = '1'

        # the stable particles are the ones with lightest mass in
        # their group
        goal_groups_1 = set([(21,22, 23,25), # 23 and 25 are calculated
                             # others are massless default
                             (1,), (2,), (3,), (4,), (5,), (6,),
                             (11,), (12,), (13,), (14,), (15,), (16,),
                             (24,)])
        goal_stable_particles_1 = set([(21,22),
                                       (2,),
                                       (11,), (12,), (14,), (16,)])

        # Test decay groups
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                            group])) \
                                  for group in full_sm.get('decay_groups')]),
                         goal_groups_1)
        # Test stable particles
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                            group])) \
                                 for group in full_sm.get('stable_particles')]),
                         goal_stable_particles_1)


        # Stage 2: turn off the neutrino mass
        full_sm.get_particle(12)['mass'] = 'ZERO'
        full_sm.get_particle(14)['mass'] = 'ZERO'
        full_sm.get_particle(16)['mass'] = 'ZERO'
        full_sm.get_particle(-12)['mass'] = 'ZERO'
        full_sm.get_particle(-14)['mass'] = 'ZERO'
        full_sm.get_particle(-16)['mass'] = 'ZERO'

        full_sm['decay_groups'] = []
        full_sm['stable_particles'] = []
        for p in full_sm['particles']:
            p['is_stable'] = False

        goal_groups_2 = set([(12,14,16,21,22, 23,25), # 23,25 are
                             # calculated, others are massless
                             (1,), (2,), (3,), (4,), (5,), (6,),
                             (11,13,15,24)])
        goal_stable_particles_2 = set([(12,14,16,21,22),
                                       (2,),
                                       (11,)])
        goal_stable_pid_2 = [2, 11, 12,14,16,21,22]

        # Test decay groups
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                            group])) \
                                  for group in full_sm.get('decay_groups')]),
                         goal_groups_2)

        # Test stable particles
        self.assertEqual(set([tuple(sorted([p.get('pdg_code') for p in \
                                            group])) \
                                 for group in full_sm.get('stable_particles')]),
                         goal_stable_particles_2)

        # Test the assignment of is_stable
        self.assertEqual(sorted([p.get_pdg_code() \
                                     for p in full_sm.get('particles') \
                                     if p.get('is_stable')]), goal_stable_pid_2)



    def test_find_full_sm_decay_groups_advance(self):
        """ Test the algorithm in find_stable_particles_advance in full sm.
            First, test the full sm with massive neutrinos.
            Second, set the neutrinos as massless."""

        # Import the full sm with param_card
        sm_path = import_ufo.find_ufo_path('sm')
        full_sm_base = import_ufo.import_full_model(sm_path)
        full_sm = decay_objects.DecayModel(full_sm_base, True)
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_full_sm.dat')
        full_sm.read_param_card(param_path)
        
        # Stage 1: Find stable particles with 
        #          nonzero neutrino masses, quark masses
        # Turned on light quark and neutrino masses
        neutrinos = [12, 14, 16]
        lightquarks1 = [1, 3]
        for npid in neutrinos:
            full_sm.get_particle(npid)['mass'] = '1E-6'
            full_sm.get_particle(-npid)['mass'] = '1E-6'
        for qpid in lightquarks1:
            full_sm.get_particle(qpid)['mass'] = '5'
        full_sm.get_particle(2)['mass'] = '1'

        # Find stable particles with given spectrum
        goal_stable_pid_1 = [2, 11, 12,14,16,21,22]
        full_sm.find_stable_particles_advance()
        
        # Test the assignment of is_stable
        self.assertEqual(sorted([p.get_pdg_code() \
                                     for p in full_sm.get('particles') \
                                     if p.get('is_stable')]), goal_stable_pid_1)
        
        
        # Stage 2: turn off the neutrino mass
        full_sm.get_particle(12)['mass'] = 'ZERO'
        full_sm.get_particle(14)['mass'] = 'ZERO'
        full_sm.get_particle(16)['mass'] = 'ZERO'
        full_sm.get_particle(-12)['mass'] = 'ZERO'
        full_sm.get_particle(-14)['mass'] = 'ZERO'
        full_sm.get_particle(-16)['mass'] = 'ZERO'

        full_sm['decay_groups'] = []
        full_sm['stable_particles'] = []
        for p in full_sm['particles']:
            p['is_stable'] = False
        full_sm.find_stable_particles_advance()

        goal_stable_pid_2 = [2, 11, 12,14,16,21,22]
        
        # Test the assignment of is_stable
        self.assertEqual(sorted([p.get_pdg_code() \
                                     for p in full_sm.get('particles') \
                                     if p.get('is_stable')]), goal_stable_pid_2)
            


    def test_running_couplings(self):
        """ Test the running coupling constants in DecayModel.
            param_card uses:
            Block SMINPUTS 
            1 1.279340e+02 # aEWM1 
            2 1.166370e-05 # Gf 
            3 1.180000e-01 # aS 
            4 9.118760e+01 # MMZ 
            """


        # Read mssm
        model_base = import_ufo.import_model('mssm')
        model = decay_objects.DecayModel(model_base, True)
        param_path = os.path.join(_file_path,'../input_files/param_card_mssm.dat')
        model.read_param_card(param_path)
        #print decay_objects.mdl_MZ

        # Test for exception
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.running_externals, "not i")

        # Test for running_externals
        # No reference value for q higher than top quark mass

        # Set b quark mass to be consistent with SM model
        decay_objects.mdl_MB = 4.7
        model['parameter_dict']['mdl_MB'] = 4.7
        
        # q=400., Nf = 5 quarks
        model.running_externals(400., 1)
        self.assertAlmostEqual(decay_objects.aS, 0.0972887598, 5)
        model.running_externals(400.)
        self.assertAlmostEqual(decay_objects.aS, 0.096563954696, 5)

        # q=170., Nf = 5 quarks
        model.running_externals(170., 1)
        self.assertAlmostEqual(decay_objects.aS, 0.1082883202, 6)
        model.running_externals(170.)
        self.assertAlmostEqual(decay_objects.aS, 0.10788637604, 6)

        # q=10., Nf = 5 quarks
        model.running_externals(10., 1)
        self.assertAlmostEqual(decay_objects.aS, 0.1730836377, 6)
        model.running_externals(10.)
        self.assertAlmostEqual(decay_objects.aS, 0.17787426379, 6)

        # q=4., Nf = 4 quarks
        model.running_externals(4., 1)
        self.assertAlmostEqual(decay_objects.aS, 0.215406025, 6)
        model.running_externals(4.)
        self.assertAlmostEqual(decay_objects.aS, 0.22772914557, 6)

        # q=1., Nf = 3 quarks
        model.running_externals(1., 1)
        self.assertAlmostEqual(decay_objects.aS, 0.36145974008, 5)
        model.running_externals(1.)
        self.assertAlmostEqual(decay_objects.aS, 0.45187971053, 5)

        # Test for running_internals
        temp_aS = 100
        decay_objects.aS = temp_aS
        #Ru11_old = decay_objects.Ru11
        Ru11_old = decay_objects.mdl_RUU1x1
        # coupling of no running dependence
        try:
            coup0 = model['couplings'][('aEWM1',)][0]
        except KeyError:
            coup0 = model['couplings'][()][0]        
        # coupling depend on aS
        coup_aS = model['couplings'][('aS',)][0]
        # coupling depend on both aS and aEWM1
        try:
            coup_both = model['couplings'][('aS', 'aEWM1')][0]
        except KeyError:
            coup_both = model['couplings'][('aEWM1', 'aS')][0]

        coup0_old = eval('decay_objects.'+coup0.name)
        #print coup_aS.name
        #print coup_both.name
        model.running_internals()


        # Test for parameters

        # Ru11 should not change
        self.assertAlmostEqual(decay_objects.mdl_RUU1x1, Ru11_old)

        # G should change
        self.assertAlmostEqual(decay_objects.G, \
                                   2*cmath.sqrt(temp_aS)*cmath.sqrt(cmath.pi))


        # Test for couplings

        # GC_365 should not change
        self.assertAlmostEqual(eval('decay_objects.'+coup0.name), coup0_old)

        # Both of GC_114 ('aS',) and GC_15 ('aEWSM1', 'aS') should change
        self.assertAlmostEqual(eval('decay_objects.'+coup_aS.name), \
                                   -decay_objects.G)

        # copying the expr of 
        self.assertAlmostEqual(eval('decay_objects.'+coup_both.name), \
                                   (-2*decay_objects.mdl_ee*complex(0,1)*decay_objects.G*decay_objects.mdl_I12x33)/3. - (2*decay_objects.mdl_ee*complex(0,1)*decay_objects.G*decay_objects.mdl_I13x33)/3.)

#===============================================================================
# Test_Channel
#===============================================================================
class Test_Channel(unittest.TestCase):
    """ Test for the channel object"""

    my_channel = decay_objects.Channel()
    h_tt_bbmmvv = decay_objects.Channel()

    def setUp(self):
        """ Set up necessary objects for the test"""

        if not hasattr(self, 'my_testmodel_base'):
            self.my_testmodel_base = import_ufo.import_model('sm')            

        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base, True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)

        # Pids that will be removed
        no_want_pid = [1, 2, 3, 4, 15, 16, 21]
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
        
        # Set up vertexlist for my_testmodel
        self.my_testmodel.find_vertexlist()

        #Setup the vertexlist for my_testmodel and save this model (optional)
        import_vertexlist.make_vertexlist(self.my_testmodel)
        # Report results
        #import_vertexlist

        # Save files to check vertices number
        # check input_files/model_name/interaction.py for vertex number
        #save_model.save_model(os.path.join(MG5DIR, 'tests/input_files', 
        #                      self.my_testmodel['name']), self.my_testmodel)
    
        full_vertexlist = import_vertexlist.full_vertexlist

        # Get vertices
        for index, vertex in full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            if legs_set == set([-6, 6, 25]) :
                # h > t t~
                vert_1 = copy.deepcopy(vertex)
            elif legs_set == set([5, 24, 6]):
                # t~ > b~ w- (decay of antiparticle)
                # t > b w+ 
                vert_2 = copy.deepcopy(vertex)
                vert_3 = copy.deepcopy(vertex)
                vert_6 = copy.deepcopy(vertex)
            elif legs_set == set([-13, 14, 24]):
                # w- > mu- vm~ (decay of antiparticle)
                # w+ > mu+ vm
                vert_4 = copy.deepcopy(vertex)
                vert_5 = copy.deepcopy(vertex)


        # h > t t~ > b b~ w+ w-
        vert_1['legs'][0]['number'] = 2
        vert_1['legs'][1]['number'] = 3
        vert_1['legs'][2]['number'] = 1
        vert_1['legs'][2]['state'] = False

        # t~ > b~ w- (decay of antiparticle)
        vert_2['legs'][0]['number'] = 2
        vert_2['legs'][1]['number'] = 4
        vert_2['legs'][2]['number'] = 2
        vert_2['legs'][0]['id'] = -5
        vert_2['legs'][1]['id'] = -24
        vert_2['legs'][2]['id'] = -6
        vert_2['id'] = self.my_testmodel['conj_int_dict'][33]

        # t > b w+
        vert_3['legs'][0]['number'] = 3
        vert_3['legs'][1]['number'] = 5
        vert_3['legs'][2]['number'] = 3

        # w- > mu vm~ (decay of antiparticle)
        vert_4['legs'][0]['number'] = 4
        vert_4['legs'][1]['number'] = 6
        vert_4['legs'][2]['number'] = 4
        vert_4['legs'][0]['id'] = 13
        vert_4['legs'][1]['id'] = -14
        vert_4['legs'][2]['id'] = -24
        vert_4['id'] = self.my_testmodel['conj_int_dict'][vert_4['id']]

        # w+ > mu vm~
        vert_5['legs'][0]['number'] = 5
        vert_5['legs'][1]['number'] = 7
        vert_5['legs'][2]['number'] = 5

        #temp_vertices = base_objects.VertexList
        self.h_tt_bbmmvv = decay_objects.Channel({'vertices': \
                                             base_objects.VertexList([
                                             vert_5, vert_4, vert_3, vert_2, \
                                             vert_1])})

        #print self.h_tt_bbmmvv.nice_string()
        #pic = drawing_eps.EpsDiagramDrawer(self.h_tt_bbmmvv, 'h_tt_bbmmvv', self.my_testmodel)
        #pic.draw()

        # t > b w+
        vert_6['legs'][0]['number'] = 2
        vert_6['legs'][1]['number'] = 3
        vert_6['legs'][2]['number'] = 1
        # change initial id to anti particle id
        vert_6['legs'][2]['id'] = -6

        self.t_bw = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList([vert_6])})
        #print self.t_bw.nice_string()

    def test_get_initialfinal(self):
        """ test the get_initial_id and get_final_legs"""

        # Test the get_initial_id
        # Raise error when neither ini_pid is found nor model is given
        self.h_tt_bbmmvv['ini_pid'] = 0
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.h_tt_bbmmvv.get_initial_id)

        # After ini_pid is found, no model is needed.
        self.assertEqual(self.h_tt_bbmmvv.get_anti_initial_id(), 25)
        self.assertEqual(self.h_tt_bbmmvv.get_initial_id(self.my_testmodel), 25)
        self.assertEqual(self.h_tt_bbmmvv.get_initial_id(), 25)

        # Test for mother particle which is not self conjugate.
        self.t_bw['ini_pid'] = 0
        self.assertEqual(self.t_bw.get_anti_initial_id(), -6)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.t_bw.get_initial_id)
        self.assertEqual(self.t_bw.get_initial_id(self.my_testmodel), 6)
        self.assertEqual(self.t_bw.get_initial_id(), 6)


        # Test the get_final_legs
        vertexlist = self.h_tt_bbmmvv.get('vertices')
        goal_final_legs = base_objects.LegList([vertexlist[0]['legs'][0],
                                                vertexlist[0]['legs'][1],
                                                vertexlist[1]['legs'][0],
                                                vertexlist[1]['legs'][1],
                                                vertexlist[2]['legs'][0],
                                                vertexlist[3]['legs'][0]])
        self.assertEqual(self.h_tt_bbmmvv.get_final_legs(), goal_final_legs)

        # Test get_final_legs with force argument
        # Change final_legs manually
        self.h_tt_bbmmvv['final_legs'].append(vertexlist[3]['legs'][0])
        goal_final_legs.append(vertexlist[3]['legs'][0])
        self.assertEqual(self.h_tt_bbmmvv.get_final_legs(), goal_final_legs)
        self.assertEqual(self.h_tt_bbmmvv.get_final_legs(True), 
                         goal_final_legs[:-1])

    def test_get_onshell(self):
        """ test the get_onshell function"""
        vertexlist = self.h_tt_bbmmvv.get('vertices')
        h_tt_bbww = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList(\
                                           vertexlist[2:])})
        #print h_tt_bbww.nice_string()
        # Test for on shell decay ( h > b b~ mu+ mu- vm vm~)
        self.assertTrue(self.h_tt_bbmmvv.get_onshell(self.my_testmodel))

        # Test for off-shell decay (h > b b~ w+ w-)
        # Raise the mass of higgs
        setattr(decay_objects, self.my_testmodel.get_particle(25)['mass'], 220)
        #decay_objects.mdl_MH = 220
        self.assertTrue(h_tt_bbww.get_onshell(self.my_testmodel))

    def test_initial_setups(self):
        """ test the intial_setups function"""

        self.h_tt_bbmmvv.initial_setups(self.my_testmodel, True)
        # Test for existence of various properties
        self.assertTrue(self.h_tt_bbmmvv['onshell'])
        self.assertTrue(self.h_tt_bbmmvv['ini_pid'])
        self.assertTrue(self.h_tt_bbmmvv['final_legs'])
        self.assertTrue(self.h_tt_bbmmvv['orders'])

        t = self.my_testmodel.get_particle(6)
        t.find_channels(2, self.my_testmodel)
        c = t.get_channels(2, True)[0]

        # Test for existence of various properties
        self.assertTrue(c['onshell'])
        self.assertTrue(c['ini_pid'])
        self.assertTrue(c['final_legs'])
        self.assertTrue(c['orders'])

    def test_helper_find_channels(self):
        """ Test of the find_channels function of DecayParticle.
            Also the test for some helper function for find_channels."""

        higgs = self.my_testmodel.get_particle(25)
        t = self.my_testmodel.get_particle(6)

        vertexlist = self.h_tt_bbmmvv.get('vertices')
        h_tt_bbww = decay_objects.Channel({'vertices': \
                                           base_objects.VertexList(\
                                           vertexlist[2:])})
        h_tt_bbww.calculate_orders(self.my_testmodel)
        self.my_testmodel.find_vertexlist()

        """ Test the connect_channel_vertex """
        # 1. connect the w- in h_tt_bbww with w_muvm to get h_tt_bwbmuvm
        #    the interaction id should change from w+ decay into w- decay
        # 2. connect the w+ in h_tt_bwbmuvm with w_muvm to get h_tt_bbmmvv
        h_tt_bwbmuvm = decay_objects.Channel({'vertices': \
                                              base_objects.VertexList(\
                                              vertexlist[1:])})
        # run initial setups
        h_tt_bwbmuvm.initial_setups(self.my_testmodel, True)
        self.h_tt_bbmmvv.initial_setups(self.my_testmodel, True)


        # decay of w+ > mu+ vm
        for v in self.my_testmodel.get_particle(24).get_vertexlist(2, True):
            if set([l['id'] for l in v['legs']]) == set([24, -13, 14]):
                w_muvm = v

        # Connect h > w+ w- b b~  with w+ > mu+ vm 
        new_channel = higgs.connect_channel_vertex(h_tt_bbww, 3, w_muvm,
                                                   self.my_testmodel)

        #print 'source:', h_tt_bbww.nice_string()
        #print 'result:', new_channel.nice_string(),\
        #    '\ngoal  :', h_tt_bwbmuvm.nice_string(), '\n'
        #print self.h_tt_bbmmvv.nice_string()
        self.assertEqual(new_channel, h_tt_bwbmuvm)
        
        # Test the change of legs in mother channel doesn't change resulting
        # channel

        # Change final_legs in new_channel
        for index, l in enumerate(new_channel['final_legs']):
            if l['id'] == 24:
                l_index = index
                l_num = l['number']
                l['number'] = 7
        #print new_channel.get_final_legs()

        self.assertFalse(new_channel.get_final_legs()\
                             == h_tt_bwbmuvm.get_final_legs())
        # reset final_legs to be normal one
        new_channel['final_legs'][l_index]['number'] = l_num

        # Test of further connection
        #print [l['id'] for l in h_tt_bwbmuvm.get_final_legs()]
        new_channel = higgs.connect_channel_vertex(h_tt_bwbmuvm, 3, w_muvm,
                                                   self.my_testmodel)
        #print new_channel.nice_string(), self.h_tt_bbmmvv.nice_string()
        self.assertEqual(new_channel, self.h_tt_bbmmvv)
        


        """ Test of check_idlegs """
        # based on t > w+ b, add one more w+
        # w+, b, t
        temp_vert = copy.deepcopy(vertexlist[2])
        temp_vert['legs'].insert(2, temp_vert['legs'][0])
        temp_vert['legs'][2]['number'] = 3
        #print temp_vert
        self.assertEqual(decay_objects.Channel.check_idlegs(vertexlist[2]), {})
        self.assertEqual(decay_objects.Channel.check_idlegs(temp_vert),
                         {24: [0, 2]})

        # Test of get_idpartlist
        # add two b quarks
        # result: w+, b, w+, b, b, t
        temp_vert2 = copy.deepcopy(temp_vert)
        temp_vert2['legs'].insert(3, temp_vert['legs'][1])
        temp_vert2['legs'].insert(4, temp_vert['legs'][1])
        #print temp_vert2

        # Create a non-sensible channel to test the get_idpartlist
        idpart_channel = decay_objects.Channel({'vertices': \
                                       base_objects.VertexList([temp_vert])})
        idpart_channel = higgs.connect_channel_vertex(idpart_channel, 
                                                      1, temp_vert2, 
                                                      self.my_testmodel)
        #print idpart_c.nice_string()
        self.assertEqual(idpart_channel.get_idpartlist(),
                         {(1, temp_vert['id'], 24): [0, 2], 
                          (0, temp_vert['id'], 24): [0, 2],
                          (0, temp_vert['id'], 5):  [1,3,4]})
        self.assertTrue(idpart_channel.get('has_idpart'))



        """Test of check_channels_equiv """
        # Create several fake vertices for test
        # h > t t~ t t~ t~
        vert_1_id = copy.deepcopy(vertexlist[4])

        vert_1_id.set('id', 800)
        vert_1_id.get('legs').insert(2, copy.copy(vert_1_id.get('legs')[0]))
        vert_1_id.get('legs').insert(3, copy.copy(vert_1_id.get('legs')[1]))
        vert_1_id.get('legs').insert(4, copy.copy(vert_1_id.get('legs')[1]))
        vert_1_id.get('legs')[2]['number'] = 4
        vert_1_id.get('legs')[3]['number'] = 5
        vert_1_id.get('legs')[4]['number'] = 6
        #print vert_1_id

        # t > w+ b~ w+
        vert_2_id = copy.deepcopy(vertexlist[2])
        vert_2_id.set('id', 900)
        vert_2_id.get('legs').insert(2, copy.copy(vert_2_id.get('legs')[0]))
        vert_2_id.get('legs')[2]['number'] = 0
        #print vert_2_id

        # w+ > mu vm is given by vertex[0]

        # w+ > mu ve
        w_muve = copy.deepcopy(vertexlist[0])
        w_muve.set('id', 1100)
        w_muve.get('legs')[0].set('id', -13)
        w_muve.get('legs')[1].set('id', 12)

        # w+ > e- vm
        w_evmu = copy.deepcopy(vertexlist[0])
        w_evmu.set('id', 1200)
        w_evmu.get('legs')[0].set('id', -11)
        w_evmu.get('legs')[1].set('id', 14)


        # Update informations in model
        for new_inter_id in [800, 900, 1000, 1100, 1200]:
            self.my_testmodel.get('interactions').append(\
                base_objects.Interaction({'id':new_inter_id }))
            self.my_testmodel.get('interactions').append(\
                base_objects.Interaction({'id':new_inter_id+1}))
            self.my_testmodel['conj_int_dict'][new_inter_id] = new_inter_id+1

        self.my_testmodel.reset_dictionaries()

        # Information about channel a, b, c, for my own convenience.
        # Nice string for channel_a:
        # ((8(13),12(-14)>8(-24),id:1001), (7(11),11(-12)>7(-24),id:43),
        #  (5(-5),10(-24)>5(-6),id:54),(4(5),9(24)>4(6),id:33),
        #  (2(-5),7(-24),8(-24)>2(-6),id:901),
        #  (2(-6),3(6),4(6),5(-6),6(6)>1(25),id:800)) ()
        
        # Nice string of channel_b:
        #((10(11),12(-12)>10(-24),id:43),(9(13),11(-14)>9(-24),id:1001),
        # (6(-5),9(-24),10(-24)>6(-6),id:901),(3(5),8(24)>3(6),id:33),
        # (2(-5),7(-24)>2(-6),id:54),
        # (2(-6),3(6),4(6),5(-6),6(-6)>2(25),id:800)) ()

        # Nice string of channel_c:
        #((10(13),12(-14)>10(-24),id:1001),(9(13),11(-14)>9(-24),id:1001),
        # (6(-5),9(-24),10(-24)>6(-6),id:901),(3(5),8(24)>3(6),id:33),
        # (2(-5),7(-24)>2(-6),id:54),
        # (2(-6),3(6),4(6),5(-6),6(-6)>2(25),id:800)) ()
        """ 
        Nice string of channel_a,b,c (new leg ordering):
        h--t (2) 
          \t~(3) > b~(3) w-(7)          w-(8)
                         \ mu(7) vm(11) \ mu(8) vm(12)
          \t (4) > w+ (4) b(9)
          \t~(5) > b~ (5) w-(10)
          \t~(6) 
        Channel_a:
        ((8(13),12(-14)>8(-24),id:44),(7(13),11(-14)>7(-24),id:44),
        (5(-5),10(-24)>5(-6),id:54),(4(24),9(5)>4(6),id:33),
        (3(-5),7(-24),8(-24)>3(-6),id:901),
        (2(6),3(-6),4(6),5(-6),6(-6),1(25),id:800)) (QED=4) 
        (est. further width = 0.000e+00)
        --------------
        Compare to channel a:  t(2) <> t(4); t~(6) <> t~(3)
        h--t (2) > w+ (4) b(9)
          \t~(3) > b~ (5) w-(10)
          \t (4) 
          \t~(5) 
          \t~(6) > b~(6) w-(9)          w-(10)
                         \ mu(9) vm(11) \ mu(10) vm(12)
        Channel_b:
        ((10(13),12(-14)>10(-24),id:44),(9(13),11(-14)>9(-24),id:44),
        (6(-5),9(-24),10(-24)>6(-6),id:901),(3(-5),8(-24)>3(-6),id:54),
        (2(24),7(5)>2(6),id:33),
        (2(6),3(-6),4(6),5(-6),6(-6),1(25),id:800)) 
        (QED=4) (est. further width = 0.000e+00)
        --------------
        h--t (2) 
          \t~(3) > b~(3) w-(7)          w-(8)
                         \ mu(7) vm(11) 
          \t (4) > w+ (4) b(9)
          \t~(5) > b~ (5) w-(10)
                          \ mu(10) vm(12)
          \t~(6) 
        Channel_c:
        ((10(13),12(-14)>10(-24),id:44),(7(13),11(-14)>7(-24),id:44),
        (5(-5),10(-24)>5(-6),id:54),(4(24),9(5)>4(6),id:33),
        (3(-5),7(-24),8(-24)>3(-6),id:901),
        (2(6),3(-6),4(6),5(-6),6(-6),1(25),id:800)) 
        (QED=4) (est. further width = 0.000e+00)
        """


        # Initiate channel_a
        # h > t t~ t t~ t~
        channel_a = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_1_id])})

        # Add t~ > b~ w- w- to first t~
        channel_a = higgs.connect_channel_vertex(channel_a, 1,
                                                 vert_2_id,
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add t > b w+ to 2nd t
        channel_a = higgs.connect_channel_vertex(channel_a, 4,
                                                 vertexlist[2],
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add t~ > b~ w- to 2nd t~
        channel_a = higgs.connect_channel_vertex(channel_a, 6,
                                                 vertexlist[2],
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add w- > mu- vm~ to first w- in t~ > b~ w- w- decay chain
        channel_a = higgs.connect_channel_vertex(channel_a, 5,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add w- > mu vm~ to 2nd w- in t~ > b~ w- w- decay chain
        channel_a = higgs.connect_channel_vertex(channel_a, 7,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print 'Channel_a:\n', channel_a.nice_string()
        


        # Initiate channel_b
        # h > t t~ t t~ t~
        channel_b = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_1_id])})

        # Add t > b w+ to 1st t
        channel_b = higgs.connect_channel_vertex(channel_b, 0,
                                                 vertexlist[2],
                                                 self.my_testmodel)
        #print '\n', channel_b.nice_string()

        # Add t~ > b~ w- to 1st t~
        channel_b = higgs.connect_channel_vertex(channel_b, 2,
                                                 vertexlist[2],
                                                 self.my_testmodel)
        #print channel_b.nice_string()

        # Add t~ > b~ w- w- to final t~
        channel_b = higgs.connect_channel_vertex(channel_b, 6,
                                                 vert_2_id,
                                                 self.my_testmodel)
        #print channel_b.nice_string()

        # Add w- > mu vm~ to 1st w- in t~ decay chain
        channel_b = higgs.connect_channel_vertex(channel_b, 1,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print channel_b.nice_string()

        # Add w- > mu vm~ to 2nd w- in t~ decay chain
        channel_b = higgs.connect_channel_vertex(channel_b, 3,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print 'Channel_b:\n', channel_b.nice_string()


        # Initiate channel_c
        # h > t t~ t t~ t~
        channel_c = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_1_id])})

        # Add t~ > b~ w- w- to first t~
        channel_c = higgs.connect_channel_vertex(channel_c, 1,
                                                 vert_2_id,
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add t > b w+ to 2nd t
        channel_c = higgs.connect_channel_vertex(channel_c, 4,
                                                 vertexlist[2],
                                                 self.my_testmodel)
        #print channel_a.nice_string()

        # Add t~ > b~ w- to 2nd t~
        channel_c = higgs.connect_channel_vertex(channel_c, 6,
                                                 vertexlist[2],
                                                 self.my_testmodel)

        # Add w- > mu vm~ to 1st w- in t~ decay chain
        channel_c = higgs.connect_channel_vertex(channel_c, 5,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print channel_c.nice_string()

        # Add w- > mu vm~ to 2nd w- in t~ decay chain
        channel_c = higgs.connect_channel_vertex(channel_c, 3,
                                                 w_muvm,
                                                 self.my_testmodel)
        #print 'Channel_c:\n', channel_c.nice_string()


        """ Test for using DiagramTag """
        Tag_a = diagram_generation.DiagramTag(channel_a)
        Tag_b = diagram_generation.DiagramTag(channel_b)
        Tag_c = diagram_generation.DiagramTag(channel_c)
        #print 'Channel_a:\n', channel_a.nice_string(), '\n', Tag_a
        #print 'Channel_b:\n', channel_b.nice_string(), '\n', Tag_b
        #print 'Channel_c:\n', channel_c.nice_string(), '\n', Tag_c

        # Test the get function
        self.assertFalse(channel_a['tag'])
        self.assertEqual(channel_a.get('tag'), Tag_a)
        self.assertTrue(Tag_a == Tag_b)
        self.assertFalse(Tag_a == Tag_c)

        # Test the check_channels_equiv function
        self.assertTrue(decay_objects.Channel.check_channels_equiv(channel_a,
                                                                   channel_b))
        self.assertFalse(decay_objects.Channel.check_channels_equiv(channel_a,
                                                                    channel_c))
        self.assertFalse(decay_objects.Channel.check_channels_equiv(channel_a,
                                                                    self.h_tt_bbmmvv))


        """ Test of check_channels_equiv for fermionic mother """
        # Change initial pid
        vert_2_id['legs'][-1]['id'] = -6
        # Correct the leg numbers
        vert_2_id['legs'][0]['number'] = 2
        vert_2_id['legs'][1]['number'] = 3
        vert_2_id['legs'][2]['number'] = 4
        vert_2_id['legs'][-1]['number'] = 1

        # t > w+ b~ w+, w+ > mu+ vm, w+ > e+ ve
        channel_d = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_2_id])})
        channel_d = higgs.connect_channel_vertex(channel_d, 0,
                                                 w_muvm,
                                                 self.my_testmodel)        
        channel_d = higgs.connect_channel_vertex(channel_d, 3,
                                                 vertexlist[0],
                                                 self.my_testmodel)        
        # t > w+ b~ w+, w+ > e+ ve, w+ > mu+ vm 
        channel_e = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_2_id])})
        channel_e = higgs.connect_channel_vertex(channel_e, 0,
                                                 vertexlist[0],
                                                 self.my_testmodel)        
        channel_e = higgs.connect_channel_vertex(channel_e, 3,
                                                 w_muvm,
                                                 self.my_testmodel)
        # t > w+ b~ w+, w+ > e+ vm, w+ > mu+ ve
        channel_f = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_2_id])})
        channel_f = higgs.connect_channel_vertex(channel_f, 0,
                                                 w_evmu,
                                                 self.my_testmodel)        
        channel_f = higgs.connect_channel_vertex(channel_f, 3,
                                                 w_muve,
                                                 self.my_testmodel)
        
        """print "channel_d:", channel_d.nice_string(), '\n',\
            "channel_e:", channel_e.nice_string(), '\n',\
            "channel_f:", channel_f.nice_string()"""
        self.assertTrue(decay_objects.Channel.check_channels_equiv(channel_d,
                                                                   channel_e))
        self.assertFalse(decay_objects.Channel.check_channels_equiv(channel_d,
                                                                   channel_f))
        self.assertFalse(decay_objects.Channel.check_channels_equiv(channel_e,
                                                                   channel_f))
        Tag_d = diagram_generation.DiagramTag(channel_d)
        Tag_e = diagram_generation.DiagramTag(channel_e)
        Tag_f = diagram_generation.DiagramTag(channel_f)

        self.assertTrue(Tag_d == Tag_e)
        self.assertFalse(Tag_d == Tag_f)
        self.assertFalse(Tag_e == Tag_f)

        
    def test_findchannels(self):
        """ Test of the find_channels functions."""

        higgs = self.my_testmodel.get_particle(25)
        t = self.my_testmodel.get_particle(6)
        t.find_channels(2, self.my_testmodel)
        #print t.get_channels(2, True).nice_string()

        # Test exceptions of find_channels
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          'non_int', self.my_testmodel)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          4, higgs)
        non_sm = copy.copy(higgs)
        non_sm.set('pdg_code', 800)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.find_channels,
                          non_sm, self.my_testmodel)

        # Test for initial id: anti id and state
        for c in t.get_channels(2, True):
            self.assertEqual(c['vertices'][-1]['legs'][-1]['id'], -6)
            self.assertEqual(c['vertices'][-1]['legs'][-1]['state'], False)
            self.assertEqual(c.get_initial_id(), 6)


        # Create two equivalent channels,
        # check whether if only one of them is found.

        # Get vertices
        for index, vertex in import_vertexlist.full_vertexlist.items():
            legs_set = set([l['id'] for l in vertex['legs']])
            if legs_set == set([23, 23, 25]) :
                # h > z z
                vert_1 = copy.deepcopy(vertex)
            elif legs_set == set([11, -11, 23]):
                # z > e- e+ 
                vert_2 = copy.deepcopy(vertex)
            elif legs_set == set([24, -24, 25]):
                # h > w+ w-
                vert_3 = copy.deepcopy(vertex)
            elif legs_set == set([-11, 12, 24]):
                # w+ > e+ ve
                vert_4 = copy.deepcopy(vertex)

        # 1.) h > z (z > e e~)
        vert_0 = self.h_tt_bbmmvv.get('vertices')[-1] 
        vert_1['legs'][0]['number'] = 2
        vert_1['legs'][1]['number'] = 3
        vert_1['legs'][2]['number'] = 1
        vert_1['legs'][2]['state'] = False
        #print vert_1, vert_2

        channel_a = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_1])})
        channel_b = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_1])})        
        channel_a = higgs.connect_channel_vertex(channel_a, 0, vert_2,
                                                self.my_testmodel)
        channel_b = higgs.connect_channel_vertex(channel_b, 1, vert_2,
                                                self.my_testmodel)
        channel_a.get_apx_decaywidth(self.my_testmodel)
        channel_b.get_apx_decaywidth(self.my_testmodel)
        channel_a.get('tag')
        channel_b.get('tag')
        channel_a['has_idpart']=True
        channel_b['has_idpart']=True
        #print channel_a.nice_string(), '\n', channel_b.nice_string()


        # 2.) h > w+ w- > e+ ve~ e- ve
        vert_3['legs'][0]['number'] = 2
        vert_3['legs'][1]['number'] = 3
        vert_3['legs'][2]['number'] = 1
        vert_3['legs'][2]['state'] = False

        channel_c = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_3])})
        channel_d = decay_objects.Channel({'vertices': base_objects.VertexList(\
                    [vert_3])})        
        channel_c = higgs.connect_channel_vertex(channel_c, 1, vert_4,
                                                self.my_testmodel)
        channel_c = higgs.connect_channel_vertex(channel_c, 2, vert_4,
                                                self.my_testmodel)
        channel_d = higgs.connect_channel_vertex(channel_d, 0, vert_4,
                                                self.my_testmodel)
        channel_d = higgs.connect_channel_vertex(channel_d, 2, vert_4,
                                                self.my_testmodel)
        #print channel_c.nice_string(), '\n', channel_d.nice_string()
        channel_c.calculate_orders(self.my_testmodel)
        channel_d.calculate_orders(self.my_testmodel)
        channel_c.get('tag')
        channel_d.get('tag')
        

        # Test of find_channels
        # The program should run find_stable_particles automatically.
        self.my_testmodel.find_channels(self.my_testmodel.get_particle(5), 3)
        self.assertFalse(self.my_testmodel.get_particle(5).get_channels(3, True))
        self.assertTrue(self.my_testmodel['stable_particles'])


        higgs.find_channels(3, self.my_testmodel)
        higgs.find_channels_nextlevel(self.my_testmodel)
        result1 = higgs.get_channels(3, True)
        #print result1.nice_string()


        # Test if the equivalent channels appear only once.
        # For both has_idpart and not has_idpart channels
        self.assertEqual((result1.count(channel_b)+ result1.count(channel_a)),1)

        # Test if the 2bodymassdiff calculated during
        # find_channels_nextlevel is right.
        self.assertAlmostEqual(higgs.get('2body_massdiff'), decay_objects.mdl_MH-2*decay_objects.mdl_MB)


        # Set MH < MW to get the desire channels.
        decay_objects.mdl_MH = 50
        channel_c.get_apx_decaywidth(self.my_testmodel)
        channel_d.get_apx_decaywidth(self.my_testmodel)
        higgs['decay_channels'] = {}
        higgs.find_channels(3, self.my_testmodel)
        higgs.find_channels_nextlevel(self.my_testmodel)
        result2 = higgs.get_channels(4, True)
        #print result2.nice_string()
        self.assertEqual((result2.count(channel_c)+ result2.count(channel_d)),1)


        """ Test on MSSM, to get a feeling on the execution time. """        
        mssm = import_ufo.import_model('mssm')
        param_path = os.path.join(_file_path,'../../models/mssm/restrict_default.dat')
        decay_mssm = decay_objects.DecayModel(mssm, force=True)
        decay_mssm.read_param_card(param_path)
        
        susy_higgs = decay_mssm.get_particle(25)
        susy_higgs.find_channels(3, decay_mssm)
        #susy_higgs.find_channels_nextlevel(decay_mssm)
        #decay_mssm.find_all_channels(3)

        # Test the calculation of branching ratios
        # The total br should be unity
        total_br = sum([amp['apx_br'] for amp in susy_higgs.get_amplitudes(2)])
        total_br += sum([amp['apx_br'] for amp in susy_higgs.get_amplitudes(3)])
        self.assertAlmostEqual(total_br, 1.)

        # Test if the err is from the off-shell 3-body channels
        err = sum([c['apx_decaywidth_nextlevel'] \
                       for c in susy_higgs.get_channels(3, False)])
        self.assertAlmostEqual(err, susy_higgs['apx_decaywidth_err'])

                                           
    def test_apx_decaywidth(self):
        """ Test for the approximation of decay rate, including
            get_apx_fnrule, get_apx_matrixelement_sq (onshell and offshell),
            get_apx_psarea"""

        # Import SM

        full_sm_base = import_ufo.import_model('sm')
        full_sm = decay_objects.DecayModel(full_sm_base, True)
        # save interaction/particle content into input_files/sm
        #save_model.save_model(os.path.join(MG5DIR, 'tests/input_files', 
        #                                   full_sm['name']), full_sm)

        higgs = self.my_testmodel.get_particle(25)

        # Set the higgs mass < Z-boson mass so that identicle particles appear
        # in final state
        MH_new = 91
        #decay_objects.mdl_MH = MH_new
        setattr(decay_objects, higgs['mass'], MH_new)
        higgs.find_channels(4, self.my_testmodel)

        MT = getattr(decay_objects, self.my_testmodel.get_particle(6)['mass'])
        # Choose h > w- e+ ve

        for c in higgs.get_channels(3, True):
            final_ids = set([l.get('id') for l in c.get_final_legs()])
            if final_ids == set([-11, 12, -24]):
                h_ww_weve = c
        #print h_ww_weve.nice_string()

        # Choose h > w- e+ ve, h > w- t b~

        for c in higgs.get_channels(3, False):
            final_ids = set([l.get('id') for l in c.get_final_legs()])
            if final_ids == set([-5, 5, 23]):
                h_zz_zbb = c
            # Distiguish h > ww > w t b~ from h > tt~ > t w b~
            if final_ids == set([-5, 6, -24]) \
                    and abs(c['vertices'][-1]['legs'][0]['id']) == 24:
                h_ww_wtb = c
        #print h_ww_wtb.nice_string(), h_zz_zbb.nice_string()


        # Choose h > 2e+ 2e-, h > vt~ vt b b~
        """print higgs.get_channels(3, True).nice_string()
        print higgs.get_channels(4, True).nice_string()"""

        for c in higgs.get_channels(4, True):
            final_ids = set([l.get('id') for l in c.get_final_legs()])
            #print final_ids
            if final_ids == set([11, -11, 11, -11]):
                h_zz_2epairs = c
            if final_ids == set([14, -14, 5, -5]):
                h_zz_bbvtvt  = c
        #print h_zz_bbvtvt.nice_string(), h_zz_2epairs.nice_string()
        
        
        # Test of the symmetric factor

        h_zz_2epairs.get_apx_psarea(self.my_testmodel)
        h_zz_bbvtvt.get_apx_psarea(self.my_testmodel)
        self.assertEqual(4, h_zz_2epairs['s_factor'])
        self.assertEqual(1, h_zz_bbvtvt['s_factor'])


        # Test of the get_apx_fnrule

        MW = h_ww_weve.get('final_mass_list')[-1]
        #print h_ww_weve.get('final_mass_list')
        #print h_ww_weve.get_apx_matrixelement_sq(self.my_testmodel)
        #print 'Vertor boson, onshell:', \
        #    h_ww_weve.get_apx_fnrule(24, 0.5,
        #                            False, self.my_testmodel)
        q_offshell = 10
        q_offshell_2 = 88
        q_onshell = 200

        # Spin 1
        self.assertAlmostEqual(h_ww_weve.get_apx_fnrule(24, q_onshell, 
                                                  True, self.my_testmodel),
                         (1+1/(MW ** 2)*q_onshell **2))
        self.assertAlmostEqual(h_ww_weve.get_apx_fnrule(24, q_offshell, 
                                                  False, self.my_testmodel),
                          ((1-2*((q_offshell/MW) ** 2)+(q_offshell/MW) ** 4)/ \
                               ((q_offshell**2-MW **2)**2)))
        # Fermion
        self.assertEqual(h_ww_weve.get_apx_fnrule(11, q_onshell, 
                                                  True, self.my_testmodel),
                         q_onshell*2)

        self.assertEqual(h_ww_weve.get_apx_fnrule(6, q_onshell, 
                                                  True, self.my_testmodel),
                         q_onshell*2)
        self.assertAlmostEqual(h_ww_weve.get_apx_fnrule(6, q_offshell, 
                                                  False, self.my_testmodel),
                         q_offshell**2/(q_offshell ** 2 - MT **2)\
                             ** 2)
        # Scalar
        self.assertEqual(h_ww_weve.get_apx_fnrule(25, q_onshell, 
                                                  True, self.my_testmodel),
                         1)

        self.assertAlmostEqual(h_ww_weve.get_apx_fnrule(25, q_offshell_2, 
                                                  False, self.my_testmodel),
                               1/(q_offshell_2 ** 2 - MH_new ** 2)**2, 5)



        # Test of matrix element square calculation

        E_mean = (MH_new-MW)/3

        """
        print h_ww_weve.get_apx_fnrule(-24, 2*E_mean, False, full_sm)
        print self.my_testmodel.get_interaction(7), 
        self.my_testmodel.get_interaction(66)
        print h_ww_weve.get_apx_fnrule(24, E_mean+MW, True, self.my_testmodel)
        """

        g_hww, g_wev = None, None
        for candidate in full_sm_base['interactions']:
            if [p['pdg_code'] for p in candidate['particles']] == [24, 24, 25]:
                g_hww = candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [12,11,24]:
                g_wev = candidate['couplings'][(0,0)]

        

        #print self.my_testmodel.get_interaction(7), 
        #self.my_testmodel.get_interaction(63)
        self.assertAlmostEqual(
            h_ww_weve.get_apx_matrixelement_sq(self.my_testmodel),
            ((E_mean**2*4*(1-2*(2*E_mean/MW)**2+(2*E_mean/MW)**4)\
                  /(((2*E_mean)**2-MW **2)**2))*\
                 (1+(1/MW*(E_mean+MW))**2)*\
                 abs(getattr(decay_objects,g_wev)) **2*\
                 abs(getattr(decay_objects,g_hww)) **2)
            )

        #-------------------------------------
        # Test color multiplicity

        # Choose hadronic decay and leptonic decay of tau
        tau = full_sm.get_particle(15)
        MTAU = abs(eval('decay_objects.' + tau.get('mass')))
        tau.find_channels(3, full_sm)
        for c in tau.get_channels(3, True):
            final_ids = set([l.get('id') for l in c.get_final_legs()])
            #print final_ids
            if final_ids == set([-2, 1, 16]):
                tau_qdecay = c
            if final_ids == set([-14, 13, 16]):
                tau_ldecay = c
        """print tau_qdecay.nice_string(), tau_ldecay.nice_string()"""


        # width of hadronic decay ~ 3* leptonic
        self.assertAlmostEqual(tau_qdecay.get_apx_decaywidth(full_sm),
                               3*tau_ldecay.get_apx_decaywidth(full_sm))

        # Using the coupling constant of w > e ve
        self.assertAlmostEqual(tau_qdecay.get_apx_matrixelement_sq(full_sm),
                               ((MTAU/3) **3 *8*3*MTAU*
                                (1-2*(2*MTAU/(3*MW))**2 +(2*MTAU/(3*MW))**4)/ \
                                    ((2*MTAU/3) ** 2 - MW **2) **2 *\
                                    abs(getattr(decay_objects,g_wev)) **4))


        #-------------------------------------
        # Test for off-shell estimation of matrix element

        # matrix element: replacing some of the energy into E_est_mean
        E_est_mean = MH_new/4
        self.assertAlmostEqual(h_ww_wtb.get_apx_matrixelement_sq(full_sm),
                               h_ww_wtb.get_apx_fnrule(5, E_est_mean,
                                                       True, full_sm)* \
                                   h_ww_wtb.get_apx_fnrule(-6, E_est_mean,
                                                      True, full_sm)* \
                                   h_ww_wtb.get_apx_fnrule(24, E_est_mean,
                                                           True, full_sm)* \
                                   h_ww_wtb.get_apx_fnrule(24, MH_new,
                                                           False, full_sm, \
                                                               est = True)*\
                                   h_ww_wtb.get_apx_fnrule(25, MH_new,
                                                           True, full_sm)*\
                                   abs(getattr(decay_objects,g_hww)) **2 *\
                                   abs(getattr(decay_objects,g_wev)) **2)

        
        #-------------------------------------
        # Test of phase space area calculation

        # On-shell phase space area
        #print 'Tau decay ps_area', tau_qdecay.get_apx_psarea(full_sm)
        self.assertAlmostEqual(tau_qdecay.calculate_apx_psarea(1.777, [0,0]),
                                1/(8*math.pi))
        self.assertAlmostEqual(tau_qdecay.calculate_apx_psarea(1.777, [0,0,0]),
                               0.000477383, 5)
        self.assertAlmostEqual(h_ww_weve.get_apx_psarea(full_sm), 
                               0.0042786859,5)

        # Estimation for off-shell case
        self.assertAlmostEqual(h_ww_wtb.get_apx_psarea(full_sm),
                                1/512/math.pi **3 * 0.8 * MH_new **2)


        #-------------------------------------
        # Test of decay width = matrix element * phase space

        # tau hadronic decay
        self.assertAlmostEqual(tau_qdecay.get_apx_decaywidth(full_sm),
                               (0.5/1.777)*tau_qdecay.get_apx_psarea(full_sm)*\
                                   tau_qdecay.get_apx_matrixelement_sq(full_sm)
                               )


        # Test of the estimated higher level width for off-shell channel
        # Test of onshell width
        self.assertAlmostEqual(h_ww_wtb.get_apx_decaywidth(full_sm),
                               (0.5/MH_new)*\
                                   h_ww_wtb.get_apx_psarea(full_sm)*\
                                   h_ww_wtb.get_apx_matrixelement_sq(full_sm)
                               )
        # Channels with no next-level decay
        self.assertEqual(h_ww_wtb.get_apx_decaywidth_nextlevel(full_sm), 0.)

        #---------------------------------------
        # Test of the estimated higher level width for off-shell channel:
        # New Kinematics Regime: 2Mz > Mh > Mz + Mw 
        MH_new = 175
        setattr(decay_objects, higgs['mass'], MH_new)
        #decay_objects.mdl_MH = MH_new

        # find all channels, width of particles are updated automatically,
        # couplings are also ran according to mother particles.
        full_sm.find_all_channels(2)
        MZ = getattr(decay_objects, full_sm.get_particle(23)['mass'])
        MW = getattr(decay_objects, full_sm.get_particle(24)['mass'])
        WZ = full_sm.get_particle(23).get('apx_decaywidth')
        WW = full_sm.get_particle(24).get('apx_decaywidth')

        # The error of z should be non-zero
        WZ_err = full_sm.get_particle(23).get('apx_decaywidth_err')
        #Z_offshell_channels = full_sm.get_particle(23).get_channels(2, False)
        #print Z_offshell_channels.nice_string()
        #print WZ_err
        self.assertTrue(WZ_err != 0)

        # Choose h > z w+ w-
        for c in higgs.get_channels(3, False):
            final_ids = set([l.get('id') for l in c.get_final_legs()])
            if final_ids == set([-24, 24, 23]):
                h_zz_zww = c
                #print h_zz_zww.nice_string()

        # Brett-Wigner correction of propagator
        self.assertAlmostEqual(h_zz_zww.get_apx_fnrule(24, q_offshell, 
                                                 False, full_sm),
                               ((1-2*((q_offshell/MW) ** 2)+(q_offshell/MW) ** 4)/ \
                             (((q_offshell**2-MW **2)**2+MW**2*WW**2)))
                               )

        # Channels with next-level decay: h > z w+ w-
        err = (WZ*abs(MZ)/MH_new*(1/4/math.pi)*MH_new **3 *0.8 \
                *  h_zz_zww.get_apx_fnrule(23, MH_new, False, full_sm, True) \
                /(h_zz_zww.get_apx_fnrule(23, MH_new/4, True, full_sm)*
                  h_zz_zww.get_apx_fnrule(23, MZ, True, full_sm))\
              +2*WW*abs(MW)/MH_new*(1/4/math.pi)*MH_new **3 *0.8 \
                *  h_zz_zww.get_apx_fnrule(24, MH_new, False, full_sm, True) \
                /(h_zz_zww.get_apx_fnrule(24, MH_new/4, True, full_sm)*
                  h_zz_zww.get_apx_fnrule(24, MW, True, full_sm)))
           

        self.assertAlmostEqual(\
            h_zz_zww.get_apx_decaywidth(full_sm)*err,
            h_zz_zww.get_apx_decaywidth_nextlevel(full_sm), 5)




        """ MSSM test
        model_base = import_ufo.import_model('mssm')
        param_path = os.path.join(_file_path,
        '../input_files/param_card_mssm.dat')
        model = decay_objects.DecayModel(model_base, force=True)
        model.read_param_card(param_path)
        model.find_all_channels(2)

        channel_2 = copy.deepcopy(h_ww_weve)
        channel_2['vertices'][0]['legs'][0]['id'] = -1000024
        channel_2['vertices'][0]['legs'][1]['id'] = 1000024
        channel_2['vertices'][0]['legs'][2]['id'] = 23
        channel_2['vertices'][0]['id'] = 128
        channel_2['vertices'][1]['legs'][0]['id'] = 1000035
        channel_2['vertices'][1]['legs'][1]['id'] = 23
        channel_2['vertices'][1]['legs'][2]['id'] = 1000025
        channel_2['vertices'][1]['id'] = 516
        channel_2['onshell'] = 0
        #print channel_2.get_onshell(model)
        channel_2.get_final_legs()
        #print channel_2.nice_string()
        #print channel_2.get_apx_matrixelement_sq(model)
        #print channel_2.get_apx_psarea(model)
        #print channel_2.get_apx_decaywidth(model)
        #print channel_2.get_apx_decaywidth_nextlevel(model)
        #print channel_2.nice_string()
        """

    def test_colormultiplicity(self):
        """ Test the color_multiplicity_def of the DecayModel object and
            the get_color_multiplicity function of Channel object. """

        # Test for exception 
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.color_multiplicity_def,
                          'a')

        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.color_multiplicity_def,
                          [1, 'a'])

        # Test the color_multiplicity_def
        self.assertEqual(self.my_testmodel.color_multiplicity_def([6,3]),
                         [(3, 2), (8, 3./4)])
        
        # Test the get_color_multiplicity
        # Two-body decay
        self.assertEqual(self.h_tt_bbmmvv.get_color_multiplicity(\
                8, [3,3], self.my_testmodel, True),
                         0.5)


    def no_test_apx_decaywidth_full_read_MG4_paramcard(self):
        """ The test to show the estimation of decay width.
            and also read the param_card of MG4. 
            Also test the find_all_channels including:
            1. unity of total branching ratio
            2. the apx_decaywidth_nextlevel comes from the right level."""

        # Options for the test
        model_name = 'mssm'
        test_param_card = False
        test_param_card_suffix = 'test1'
        smart_find = False
        prec = 1E-2
        channel_number = 3


        # Read model_name
        model_base = import_ufo.import_model(model_name)
        model = decay_objects.DecayModel(model_base, True)

        # Read MG5 param_card
        if test_param_card:
            MG5_param_path = os.path.join(_file_path,
                                        '../input_files/param_card_'\
                                            +model_name \
                                            +'_'\
                                            +test_param_card_suffix
                                            +'.dat')
        else:
            MG5_param_path = os.path.join(_file_path,
                                        '../input_files/param_card_'\
                                            +model_name \
                                            +'.dat')
        model.read_param_card(MG5_param_path)

        # Find channels before read MG4 param_card (option for use smart find)
        if smart_find:
            model.find_all_channels_smart(prec)
        else:
            model.find_all_channels(channel_number)

        # Read MG4 param_card
        if model_name == "mssm":
            if test_param_card:
                MG4_param_path = os.path.join(_file_path,
                                              '../input_files/param_card_test1.dat')
            else:
                MG4_param_path = os.path.join(_file_path,
                                              '../input_files/param_card_0.dat')

            model.read_MG4_param_card_decay(MG4_param_path)

        # Write the decay summary, decay table, and helas collection
        if test_param_card:
            model.write_summary_decay_table(model_name\
                                                + '_decay_summary_'\
                                                + test_param_card_suffix \
                                                + '.dat')
            model.write_decay_table(MG5_param_path, 'cmp', 
                                    model_name\
                                        + '_decaytable_'\
                                        + test_param_card_suffix \
                                        + '.dat')
            model.write_helas_collection(model_name\
                                             + '_helascollection_'\
                                             + test_param_card_suffix \
                                             + '.dat')

        else:
            # file name 1: default name
            model.write_summary_decay_table()
            model.write_decay_table(MG5_param_path, 'cmp')
            model.write_helas_collection()

        """ Actual Test """
        # Test the sum of branching ratios is unity
        part = model.get_particle(25)
        total_br = sum([sum([amp['apx_br'] for amp in part.get_amplitudes(i)]) \
                            for i in range(2, part.get_max_level()+1)])
        self.assertAlmostEqual(total_br, 1.)

        # Test if the err is from the off-shell 3-body channels
        if part.get_max_level() > 2:
            err = sum([c['apx_decaywidth_nextlevel'] \
                           for c in part.get_channels(part.get_max_level(), 
                                                      False)])
            self.assertAlmostEqual(err/part.get('apx_decaywidth'), part['apx_decaywidth_err'])



        """ End of test """

        """# Test if the channels are find wisely
        for part in model['particles']:
            self.assertTrue(prec > part.get('apx_decaywidth_err'))

        # Check if the max_level is expected.
        self.assertEqual(model.get_particle(1000016).get_max_level(), 2)
        self.assertEqual(model.get_particle(25).get_max_level(), 3)"""

        # Miscellaneous
        # Test if the calculated ratio is float or None
        """for part in model.get('particles'):
            print part.get_pdg_code(), part.get('2body_massdiff')
            #n_max = len(part.decay_amplitudes.keys())
            for n in range(2,n_max+2):
                for amp in part.get_amplitudes(n):
                    self.assertTrue(isinstance(amp['exa_decaywidth'], bool) or \
                                        isinstance(amp['exa_decaywidth'], float))
        """
        #particle = model.get_particle(1000035)
        #particleb = model.get_particle(2000001)
        #channels = particleb.get_channels(3, False)
        #channels.sort(decay_objects.channelcmp_width)
        #print [c.nice_string() for c in channels[:100]]
        #a = particle.get_amplitudes(3)
        #print a
        #print a.nice_string()#, a['diagrams']
        #print model.get_interaction(166)
        #print decay_objects.GC_415, decay_objects.GC_681

        """
        #print particle.get_amplitude([-11, 2000011])['diagrams'].nice_string()
        print model.get_interaction(390)
        print decay_objects.GC_780, decay_objects.GC_752
        print model.get_interaction(388)
        print decay_objects.GC_757, decay_objects.GC_785
        
        print particle.get_amplitude([2000001, -1]).decaytable_string()
        print particle.get_amplitude([2000001, -1])['diagrams'][0].get_apx_psarea(model)
        print particle.get_amplitude([2000001, -1])['apx_decaywidth']
        print particle.get_amplitude([2000001, -1])['exa_decaywidth']

        #print particleb.get_amplitude([-1000024, 16, 22]).nice_string()

        #print particle.get_amplitude([-11, 2000011])['diagrams'][0].get_apx_matrixelement_sq(model)        
        #print particle.get_amplitude([-11, 2000011])['exa_decaywidth']

        #for part in model.get('particles'):
        #    print part['pdg_code'], part['decay_width']

        #particle.calculate_branch_ratio()
        #print decay_objects.mdl_MT, decay_objects.mdl_MW
        #print decay_objects.GC_857, decay_objects.GC_733, decay_objects.GC_437, decay_objects.GC_665
        #print particle.estimate_width_error()
        #print len(particle.get_channels(3, False))
        #print particle.get_channels(3, False)[0].nice_string(),\
        #    particle.get('apx_decaywidth')
        """

#===============================================================================
# Test_IdentifyHelasTag
#===============================================================================
class Test_IdentifyHelasTag(unittest.TestCase):
    """ Test for IdentifyHelasTag, and all the related functions. """

    my_testmodel_base = import_ufo.import_model('sm')
    my_channel = decay_objects.Channel()
    h_tt_bbmmvv = decay_objects.Channel()

    def setUp(self):
        """ Set up necessary objects for the test"""

        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base, True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)

        # Pids that will be removed
        no_want_pid = [1, 2, 15, 16, 21]
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
        
        # Set up vertexlist for my_testmodel
        self.my_testmodel.find_vertexlist()
    

    def test_helas_comparison(self):
        """Test the ability to identify Helas calls."""

        # Turn higgs decay into 4-body
        decay_objects.mdl_MH = 80

        h = self.my_testmodel.get_particle(25)
        t = self.my_testmodel.get_particle(6)

        h.find_channels(4, self.my_testmodel)
        
        #print h.get_amplitudes(4).nice_string()

        for c in h.get_channels(4, True):
            pids = set([l['id'] for l in c.get_final_legs()])
            # Z mediated hadronic decays
            if pids == set([3, -3, 4, -4]) and \
                    c['vertices'][0]['legs'][-1]['id'] == 23:
                h_zz_sscc = c
                h_zz_sscc_Tag = decay_objects.IdentifyHelasTag(h_zz_sscc, 
                                                               self.my_testmodel)
            if pids == set([3, -3, 3, -3]):
                h_zz_ssss = c
                h_zz_ssss_Tag = decay_objects.IdentifyHelasTag(h_zz_ssss, 
                                                               self.my_testmodel)

            if pids == set([5, -5, 5, -5]):
                h_zz_bbbb = c
                h_zz_bbbb_Tag = decay_objects.IdentifyHelasTag(h_zz_bbbb, 
                                                               self.my_testmodel)
            # Z mediated leptonic decays
            if pids == set([11, -11, 12, -12]) and \
                    c['vertices'][0]['legs'][-1]['id'] == 23:
                h_zz_eeveve = c
                h_zz_eeveve_Tag = decay_objects.IdentifyHelasTag(h_zz_eeveve, 
                                                                 self.my_testmodel)
            if pids == set([11, -11, 13, -13]):
                h_zz_eemumu = c
                h_zz_eemumu_Tag = decay_objects.IdentifyHelasTag(h_zz_eemumu, 
                                                                 self.my_testmodel)
            if pids == set([13, -13, 13, -13]):
                h_zz_mumumumu = c
                h_zz_mumumumu_Tag = decay_objects.IdentifyHelasTag(h_zz_mumumumu, 
                                                                 self.my_testmodel)

            # W mediated hadronic decays
            if pids == set([3, -3, 4, -4]) and \
                    abs(c['vertices'][0]['legs'][-1]['id']) == 24:
                h_ww_sscc = c
                h_ww_sscc_Tag = decay_objects.IdentifyHelasTag(h_ww_sscc, 
                                                               self.my_testmodel)
            # W mediated leptonic decays
            if pids == set([11, -11, 12, -12]) and \
                    abs(c['vertices'][0]['legs'][-1]['id']) == 24:
                h_ww_eeveve = c
                h_ww_eeveve_Tag = decay_objects.IdentifyHelasTag(h_ww_eeveve, 
                                                               self.my_testmodel)


        #print h_zz_eemumu_Tag, h_zz_eeveve_Tag
        self.assertTrue(h_zz_ssss_Tag == h_zz_bbbb_Tag)
        self.assertTrue(h_zz_eemumu_Tag == h_zz_mumumumu_Tag)
        # Lorentz structure of z > e e~ != z > ve ve~
        self.assertFalse(h_zz_eemumu_Tag == h_zz_eeveve_Tag)
        # Lorentz structure of z > up-type up-type != z > down-type down-type
        self.assertFalse(h_zz_ssss_Tag == h_zz_sscc_Tag)
        self.assertFalse(h_zz_sscc_Tag == h_ww_sscc_Tag)
        self.assertFalse(h_zz_sscc_Tag == h_zz_eemumu_Tag)

        # Test diagram_from_tag
        new_diagram = h_zz_ssss_Tag.diagram_from_tag(self.my_testmodel)
        #print new_diagram.nice_string()
        new_tag = diagram_generation.DiagramTag(new_diagram)
        old_tag = diagram_generation.DiagramTag(h_zz_ssss)
        #print new_tag, old_tag
        self.assertTrue(new_tag == old_tag)

    def test_helas_helpers(self):
        """ Test related helpers in DecayModel, Channel, DecayParticles. """

        t = self.my_testmodel.get_particle(6)
        t.find_channels(2, self.my_testmodel)
        t_bw = t.get_channels(2, True)[0]
        
        # Test get in Channel        
        self.assertFalse(t_bw['helastag'] and t_bw['std_diagram'])
        t_bw.get('helastag', self.my_testmodel)
        self.assertTrue(t_bw['helastag'])
        t_bw['helastag'] = None


        # Test get_helas_properties in Channel
        self.assertFalse(t_bw['helastag'] and t_bw['std_diagram'])
        tag, std_diagram = t_bw.get_helas_properties(self.my_testmodel)
        self.assertTrue(t_bw['helastag'] and t_bw['std_diagram'])
        self.assertTrue(isinstance(tag, decay_objects.IdentifyHelasTag))
        self.assertTrue(isinstance(std_diagram, base_objects.Diagram))
        t_bw['helastag'] = None
        t_bw['std_diagram'] = None


        # Test get/add helascalls in DecayModel
        self.assertEqual(self.my_testmodel.get_helascalls(4), [])
        #     Error when input diagram has no helastag.
        self.assertRaises(AssertionError, 
                          self.my_testmodel.add_helascalls, 2, t_bw)
        tag, std_diagram = t_bw.get_helas_properties(self.my_testmodel)
        self.my_testmodel.add_helascalls(2, t_bw)
        self.assertTrue(len(self.my_testmodel.get_helascalls(2))==1)
        self.assertTrue(self.my_testmodel.get_helascalls(2)[0]['helastag'])

        #     The helas calls should be a deepcopy of the argument.
        std_diagram['vertices'] = []
        self.assertTrue(self.my_testmodel.get_helascalls(2)[0]['vertices'])


    


    def test_collect_helascalls(self):
        """ Test the collect_helascalls. """


        
        

        h = self.my_testmodel.get_particle(25)
        t = self.my_testmodel.get_particle(6)

        # Turn top decay into 3-body
        setattr(decay_objects, t['mass'], 60)

        t.find_channels(3, self.my_testmodel)
        #print t.get_channels(3, True).nice_string()

        # Test exceptions
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.collect_helascalls, t, 'non-int')
        self.assertRaises(decay_objects.DecayModel.PhysicsObjectError,
                          self.my_testmodel.collect_helascalls, 'non-part', 3)


        self.my_testmodel.collect_helascalls(t, 3)
        # helas calls contain leptonic and hadronic top decay
        self.assertEqual(len(self.my_testmodel.get_helascalls(3)), 2)

        # Test the properties set during collect_helascalls
        t_beve = t.get_amplitude([-11, 12, 5])['diagrams'][0]
        t_bmuvm = t.get_amplitude([-13, 14, 5])['diagrams'][0]
        std_diagram = self.my_testmodel.get_helascalls(3)[t_beve['helas_number']]
        self.assertEqual(t_beve['helas_number'],
                         t_bmuvm['helas_number'])
        self.assertTrue(std_diagram['helastag'] == t_bmuvm['helastag'])
        

#===============================================================================
# Test_DecayAmplitude
#===============================================================================
class Test_DecayAmplitude(unittest.TestCase):
    """ Test for the DecayAmplitude and DecayAmplitudeList object."""

    def setUp(self):
        """ Set up necessary objects for the test"""
        self.my_testmodel_base = import_ufo.import_model('sm')
        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base, True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        my_channel = decay_objects.Channel()
        h_tt_bbmmvv = decay_objects.Channel()

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        # Pids that will be removed
        no_want_pid = [1, 2, 3, 4, 15, 16, 21]
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
    

    def test_init_setget(self):
        """ Test the set and get function of DecayAmplitude. """

        # Setup higgs and lower its mass
        higgs = self.my_testmodel.get_particle(25)
        t = self.my_testmodel.get_particle(6)
        decay_objects.mdl_MH = 50

        # Set channels
        self.my_testmodel.find_all_channels(4)
        #print higgs.get_channels(4, True).nice_string()
        #print t.get_channels(2,True)[0]

        # Get channels: h > ww > e+ e- ve ve~
        #               h > zz > e+ e- e+ e-
        for c in higgs.get_channels(4, True):
            final_ids = set([l['id'] for l in c.get_final_legs()])
            if final_ids == set([11, -11, 12, -12]):
                # w mediated, not z
                if abs(c['vertices'][0]['legs'][-1]['id']) == 24:
                    h_mmvv_1 = c
            if final_ids == set([11, -11, 11, -11]):
                # z mediated
                if abs(c['vertices'][0]['legs'][-1]['id']) == 23:
                    h_mmvv_2 = c


        # Test the initialization
        amplt_h_mmvv = decay_objects.DecayAmplitude(h_mmvv_1,
                                                    self.my_testmodel)
        amplt_t_bw = decay_objects.DecayAmplitude(t.get_channels(2,True)[0],
                                                  self.my_testmodel)
        #print amplt_t_bw.nice_string()

        # goal id list for legs in process
        # The legs are in the order of numbers!
        # Subtest: test decay_objects.legcmp_bynumber
        l1 = base_objects.Leg({'number':1, 'id':10})
        l2 = base_objects.Leg({'number':2, 'id': 8})
        sorted_legs = sorted([l2, l1], decay_objects.legcmp_bynumber)
        self.assertEqual([l1, l2], sorted_legs)

        # Note: initial id in process should be POSITIVE
        goal_id_list_h = set([25, 11, -11, -12, 12])
        goal_id_list_t = set([6, 5, 24])
        self.assertEqual(set([l.get('id') \
                              for l in amplt_h_mmvv.get('process').get('legs')]),
                         goal_id_list_h)
        self.assertEqual(set([l.get('id') \
                              for l in amplt_t_bw.get('process').get('legs')]),
                         goal_id_list_t)

        # Check the legs in process (id, number, and state)
        final_numbers = [2,3,4,5]
        for l in amplt_h_mmvv.get('process').get('legs'):
            if l.get('id') != 25:
                self.assertTrue(l.get('state'))
                final_numbers.remove(l.get('number'))
            else:
                self.assertFalse(l.get('state'))
                self.assertEqual(1, l.get('number'))
        self.assertFalse(final_numbers)


        # Test the set and get in Amplitude
        goal_width = h_mmvv_1.get('apx_decaywidth')
        self.assertEqual(amplt_h_mmvv.get('apx_decaywidth'), goal_width)
        goal_width += h_mmvv_2.get('apx_decaywidth')
        amplt_h_mmvv.get('diagrams').append(h_mmvv_2)


        # Test if the reset works.
        amplt_h_mmvv.reset_width_br()
        self.assertEqual(amplt_h_mmvv['apx_decaywidth'], 0.)
        # Test the get for decaywidth and branch ratio.
        self.assertEqual(amplt_h_mmvv.get('apx_decaywidth'), goal_width)
        self.assertEqual(amplt_h_mmvv.get('apx_br'),
                         goal_width/higgs.get('apx_decaywidth'))
        WH = higgs['apx_decaywidth']
        higgs['apx_decaywidth'] = 0.
        amplt_h_mmvv.reset_width_br()
        # WARNING should show for getting br from zero-width particle.
        amplt_h_mmvv.get('apx_br')


        # Test for non-self-conjugate mother
        self.assertEqual(amplt_t_bw.get('apx_br'), 1.)
        

        # Test for exceptions in set get of Amplitude
        wrong_prop_list = {'process': [1, 'a', base_objects.Diagram()],
                           'diagrams': [1, 'a', base_objects.Diagram()]}
        for key, proplist in wrong_prop_list.items():
            for prop in proplist:
                self.assertRaises(decay_objects.DecayAmplitude.PhysicsObjectError,
                                  amplt_h_mmvv.filter,
                                  key, prop)


        # Test for set and get in DecayParticle
        my_amplist = decay_objects.DecayAmplitudeList([amplt_h_mmvv])
        higgs.set('decay_amplitudes', {4: my_amplist})
        self.assertEqual(higgs.get('decay_amplitudes'), {4: my_amplist})


        # Test for exceptions
        valuelist = ['nondict', {'a': my_amplist}, {4: base_objects.Process()}]
        for value in valuelist:
            self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                              higgs.filter,
                              'decay_amplitudes', value)            


        # Test for set_amplitudes and get_amplitudes
        higgs.set_amplitudes(4, decay_objects.DecayAmplitudeList())
        self.assertEqual(higgs.get_amplitudes(4), 
                         decay_objects.DecayAmplitudeList())


        # Test the set from normal list of Amplitude
        higgs.set_amplitudes(4, [amplt_h_mmvv])
        self.assertEqual(higgs.get_amplitudes(4), my_amplist)
        self.assertEqual(higgs.get_amplitudes(6), [])


        # Test for exceptions
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.get_amplitudes, 'a')
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.set_amplitudes,
                          'a', decay_objects.DecayAmplitudeList())
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.set_amplitudes,
                          4, decay_objects.DecayAmplitude())


        # Test for get_amplitude
        #higgs.get_amplitudes(4)
        self.assertEqual(higgs.get_amplitude([12, -11, -12, 11]),
                         amplt_h_mmvv)
        self.assertEqual(higgs.get_amplitude([-12, -12, 11, 12]),
                         None)


        # Test for exception
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.get_amplitude,
                          'Non-list')
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.get_amplitude,
                          ['a', 1.23])


    def test_group_channels2amplitudes(self):
        """ Test the group_channels_2_amplitudes function."""

        # Setup higgs and lower its mass
        higgs = self.my_testmodel.get_particle(25)
        decay_objects.mdl_MH = 50

        # Set channels and amplitude

        self.my_testmodel.find_all_channels(4)
        #print higgs.get_channels(4, True).nice_string()

        # Get channels: h > ww > e+ e- ve ve~
        #               h > zz > e+ e- ve ve~
        #               h > zz > e+ e- e+ e-
        for c in higgs.get_channels(4, True):
            final_ids = set([l['id'] for l in c.get_final_legs()])
            if final_ids == set([13, -13, 14, -14]):
                # w mediated, not z
                if abs(c['vertices'][0]['legs'][-1]['id']) == 24:
                    h_mmvv_1 = c
                # z mediated
                elif abs(c['vertices'][0]['legs'][-1]['id']) == 23:
                    h_mmvv_2 = c
            if final_ids == set([11, -11, 11, -11]):
                h_epairs = c


        #print h_mmvv_1.nice_string(), h_mmvv_2.nice_string()
        amplt_h_mmvv = decay_objects.DecayAmplitude(h_mmvv_1, self.my_testmodel)
        amplt_h_mmvv.add_std_diagram(h_mmvv_2)
        amplt_h_epairs = decay_objects.DecayAmplitude(h_epairs, 
                                                      self.my_testmodel)


        # Test for exceptions

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.group_channels_2_amplitudes,
                          'a', self.my_testmodel)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          higgs.group_channels_2_amplitudes,
                          3, 'a')


        # Test if the group works

        higgs.find_channels(4, self.my_testmodel)
        higgs.group_channels_2_amplitudes(4, self.my_testmodel)
        #print higgs.get_amplitudes(4).nice_string()

        # Group function will calculate the decaywidth but not the apx_br
        # fix it manually to compare with automatic result
        amplt_h_mmvv.get('apx_decaywidth')
        amplt_h_epairs.get('apx_decaywidth')

        self.assertTrue(amplt_h_mmvv in higgs.get_amplitudes(4))
        self.assertTrue(amplt_h_epairs in higgs.get_amplitudes(4))

        # Test that all the final_legs has the same number
        for amp in higgs.get_amplitudes(4):
            final_list = set([(l['number'], l['id']) for l in amp['process']['legs'][1:]])
            for dia in amp['diagrams']:
                self.assertEqual(set([(l['number'], l['id']) for l in dia.get_final_legs()]), final_list)


    def test_decaytable_string(self):
        """ Test the decaytable_string """

        # Setup higgs and lower its mass
        higgs = self.my_testmodel.get_particle(25)
        decay_objects.mdl_MH = 50

        # Set channels and amplitude
        self.my_testmodel.find_all_channels(4)
        amp_list = higgs.get_amplitudes(4)
        # Test for exception
        self.assertRaises(decay_objects.DecayAmplitude.PhysicsObjectError,
                          amp_list.decaytable_string,'wrongformat')
        # Test for type
        self.assertTrue(isinstance(amp_list.decaytable_string('full'), str))
        self.assertTrue(isinstance(amp_list.decaytable_string(), str))

        # Test for decaytable_string from DecayParticle
        self.assertTrue(isinstance(higgs.decaytable_string(), str))

        self.my_testmodel.write_decay_table(os.path.join(_file_path,'../input_files/param_card_sm.dat'), 
                                            'full', 'mysmallmodel')

        #print self.my_testmodel['parameters'], '\n',\
        #    self.my_testmodel['functions']

    def test_get_amplitude_givenfinal(self):
        """ Test the get_amplitude function of DecayAmplitudeList. """

        # Setup higgs and lower its mass
        higgs = self.my_testmodel.get_particle(25)
        decay_objects.mdl_MH = 50

        # Set channels and amplitude
        higgs.find_channels(4, self.my_testmodel)
        amp_list = higgs.get_amplitudes(4)

        input_pids = [l.get('id') for l in amp_list[0]['process']['legs']\
                          if l.get('state')]
        self.assertEqual(amp_list.get_amplitude(input_pids),
                         amp_list[0])
        self.assertFalse(amp_list.get_amplitude([-11, 11, 6, -6]))


    def test_add_std_diagram(self):
        """ Test the add_std_diagram of DecayAmplitude."""

        # Setup higgs and lower its mass
        higgs = self.my_testmodel.get_particle(25)
        decay_objects.mdl_MH = 50

        # Set channels and amplitude
        self.my_testmodel.find_all_channels(4)
        h_eeveve =  higgs.get_amplitude([-11, 11, -12, 12])
        
        # Construct new amplitude
        # diagram 0: h > ww > e+ e- ve ve~
        std_amp = decay_objects.DecayAmplitude(h_eeveve['diagrams'][0], 
                                          self.my_testmodel)

        # Modify the number of legs in second diagram
        # diagram 1: h > zz > e+ e- ve ve~
        new_diagram = copy.deepcopy(h_eeveve['diagrams'][1])
        
        # Make sure which z goes to which vertex
        z_1 = new_diagram['vertices'][-1]['legs'][0]
        z_2 = new_diagram['vertices'][-1]['legs'][0]

        if z_1 == new_diagram['vertices'][0]['legs'][-1]:
            first_to_first = True
        else:
            first_to_first = False

        # Change the numbers
        for vert in new_diagram['vertices']:
            for leg in vert['legs']:
                # change number 2 -> 5, 5-> 4, 4->2
                old_number = leg['number']
                if old_number == 2:
                    leg['number'] = 5
                if old_number == 5:
                    leg['number'] = 4
                if old_number == 4:
                    leg['number'] = 2

        new_diagram.initial_setups(self.my_testmodel, True)

        #print 'dia1:', h_eeveve['diagrams'][0].nice_string(),\
        #    '\ndia2(old):', h_eeveve['diagrams'][1].nice_string()
        #print 'dia2 new:', new_diagram.nice_string()


        # Add diagram (with a different number assignment) to std_amp
        std_amp.add_std_diagram(new_diagram)
        #print std_amp['diagrams'][0].nice_string()
        #print std_amp['diagrams'][1].nice_string()

        # Test final legs
        list_a = sorted(std_amp['diagrams'][0].get_final_legs(), 
                        key=lambda l: l['id'])
        number_a = [(l['number'], l['id']) for l in list_a]
        list_b = sorted(std_amp['diagrams'][1].get_final_legs(), 
                        key=lambda l: l['id'])
        number_b = [(l['number'], l['id']) for l in list_b]
        self.assertEqual(number_a, number_b)



        # Test all legs,
        # numbers must be consistent.
        #print std_amp.nice_string()
        for dia in std_amp['diagrams']:
            for i, vert in enumerate(dia['vertices'][:-1]):
                # Check final (mother) leg of each vertices is the minimum
                # among its children
                previous_number = [l['number'] for l in vert['legs'][:-1]]
                self.assertEqual(vert['legs'][-1]['number'],
                                min(previous_number))
                
                # Check intermediate legs
                l=vert['legs'][-1]
                for vert2 in dia['vertices'][i+1:]:
                    for l2 in vert2['legs'][:-1]:
                        if l['number'] == l2['number']:
                            self.assertEqual(l['id'], l2['id'])

                # Check the particles are being placed before antiparticles
                ids = [l['id'] for l in vert['legs'][:-1]]
                goal_ids = sorted(ids, reverse=True)
                self.assertEqual(goal_ids, ids)
                    

        # Test if the two z bosons still connect to the correct vertex
        if first_to_first:
            # 1st vertex from 1st z
            self.assertEqual(\
                std_amp['diagrams'][1]['vertices'][0]['legs'][-1]['number'],
                std_amp['diagrams'][1]['vertices'][-1]['legs'][0]['number'])
            # 2nd vertex from 2nd z
            self.assertEqual(\
                std_amp['diagrams'][1]['vertices'][1]['legs'][-1]['number'],
                std_amp['diagrams'][1]['vertices'][-1]['legs'][1]['number'])

        # and vice versa
        else:
            self.assertEqual(\
                std_amp['diagrams'][1]['vertices'][0]['legs'][-1]['number'],
                std_amp['diagrams'][1]['vertices'][-1]['legs'][1]['number'])
            self.assertEqual(\
                std_amp['diagrams'][1]['vertices'][1]['legs'][-1]['number'],
                std_amp['diagrams'][1]['vertices'][-1]['legs'][0]['number'])
            

#===============================================================================
# Test_AbstractModel
#===============================================================================
class Test_AbstractModel(unittest.TestCase):
    """ Test for the AbstractModel object."""

    def setUp(self):
        """ Set up necessary objects for the test"""
        self.my_testmodel_base = import_ufo.import_model('sm')
        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base,
                                                     force=True)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        my_channel = decay_objects.Channel()
        h_tt_bbmmvv = decay_objects.Channel()

        # Simplify the model
        particles = self.my_testmodel.get('particles')
        interactions = self.my_testmodel.get('interactions')
        inter_list = copy.copy(interactions)
        # Pids that will be removed
        no_want_pid = [1, 2, 3, 4, 21]
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

    def test_get_particles_type(self):
        """Test the set_new_particle. """
        
        ab_model = self.my_testmodel['ab_model']

        # Set electron as Majorana fermion
        majorana_e = self.my_testmodel.get_particle(11)
        majorana_e['self_antipart'] = True
        # get_particles_type can be run only after generate the abstract_model
        self.my_testmodel.generate_abstract_model()

        # Test of particle input
        self.assertEqual((2, 3, False),
                         ab_model.get_particle_type(\
                self.my_testmodel.get_particle(6)))
        self.assertEqual(9902300,
                         ab_model.get_particle_type(\
                self.my_testmodel.get_particle(6), get_code=True))
        # Test of pdg_code input
        self.assertEqual((2, 3, False),
                         ab_model.get_particle_type(6))
        self.assertEqual(9902300,
                         ab_model.get_particle_type(6, get_code=True))
        # Test of get_particle_type_code
        self.assertEqual(-9902300,
                         ab_model.get_particle_type_code(self.my_testmodel.get_particle(-6)))
        self.assertEqual(9912100,
                         ab_model.get_particle_type_code(majorana_e))


        input_list = [6,5,24,-6,11,22]
        reorder_list = [5,6,24,11,-6,22]

        goal_list_noignore = \
            [9902300, 9902301, 9903101, -9902300, 9912100, 9903100]
        goal_list_default = \
            [9902302, 9902301, 9903101, -9902300, 9912100, 9903100]
        goal_list_nonzerostart = \
            [9902302, 9902303, 9903101, -9902302, 9912101, 9903100]

        goal_serial_dict_noignore = {(2, 3, False): 2, 
                                     (3, 1, True): 2, 
                                     (2, 1, True): 1}
        goal_serial_dict_default = {(2, 3, False): 3, 
                                    (3, 1, True): 2, 
                                    (2, 1, True): 1}
        goal_serial_dict_nonzerostart = {(2, 3, False): 4, 
                                         (3, 1, True): 2, 
                                         (2, 1, True): 2}
        #print ab_model.get_particlelist_type.serial_number_dict
        self.assertEqual(ab_model.get_particlelist_type(input_list, False)[0],
                         goal_list_noignore)
        self.assertEqual(ab_model.get_particlelist_type(input_list, False)[1],
                         goal_serial_dict_noignore)
        self.assertEqual(ab_model.get_particlelist_type(input_list)[0],
                         goal_list_default)
        self.assertEqual(ab_model.get_particlelist_type(input_list)[1],
                         goal_serial_dict_default)

        self.assertEqual(ab_model.get_particlelist_type(input_list, False,
                                                        sn_dict={\
                    (2, 3, False):2, (2, 1, True):1})[0],
                         goal_list_nonzerostart)
        self.assertEqual(ab_model.get_particlelist_type(input_list, False,
                                                        sn_dict={\
                    (2, 3, False):2, (2, 1, True):1})[1],
                         goal_serial_dict_nonzerostart)

        # Test if the output is regardless of input order
        self.assertEqual(sorted(ab_model.get_particlelist_type(input_list)[0]),
                         sorted(ab_model.get_particlelist_type(reorder_list)[0]))


    def test_add_ab_particle(self):
        """Test the set_new_particle. """
        
        ab_model = self.my_testmodel['ab_model']
        # Set electron to be Majorana
        majorana_e = self.my_testmodel.get_particle(11)
        majorana_e['self_antipart'] = True
        # Add new particle
        newpart = decay_objects.DecayParticle({'pdg_code': 999,
                                               'mass':'ZERO',
                                               'spin':4, 'color':6, 
                                               'self_antipart': True})
        self.my_testmodel['particles'].append(newpart)
        self.my_testmodel.reset_dictionaries()
        self.my_testmodel.generate_abstract_model()

        # Test for exceptions
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.add_ab_particle,
                          'NoneParticle')

        # Test get_particle_type
        self.assertEqual(ab_model.get_particle_type(22),
                         (3,1, True))
        self.assertEqual(ab_model.get_particle_type(11),
                         (2,1, True))


        # Test for setting existing particle
        ab_model.add_ab_particle(-6)
        # Check if new particle is in paticles, abstract_particles_dict,
        # and particle_dict of Model
        self.assertEqual(len(ab_model['abstract_particles_dict'][(2,3,False)]),
                         3)
        newpart = ab_model['abstract_particles_dict'][(2,3, False)][-1]
        self.assertTrue(newpart in ab_model.get('particles'))
        # Test if the get_particle can find the correct anti-particle
        self.assertEqual(ab_model.get_particle(-newpart.get_pdg_code())['is_part'],
                        False)

        # Check properties
        self.assertEqual(newpart.get('name'), 'F3_02')
        self.assertEqual(newpart.get('antiname'), 'F3_02~')
        self.assertEqual(newpart.get('mass'), 'MNF3_02')
        self.assertEqual(newpart.get('pdg_code'), 9902302)

        # Test for setting new particle
        ab_model.add_ab_particle(999)

        # Check if new particle is in paticles and abstract_particles_dict
        self.assertTrue((4,6, True) in \
                            ab_model['abstract_particles_dict'].keys())
        self.assertTrue(ab_model.get_particle(9914601))

        # Check properties
        newpart = ab_model['abstract_particles_dict'][(4,6, True)][-1]
        self.assertEqual(newpart.get('name'), 'P6_01')
        self.assertEqual(newpart.get('antiname'), 'none')
        self.assertEqual(newpart.get('mass'), 'MSP6_01')
        self.assertEqual(newpart.get('pdg_code'), 9914601)

    def test_setup_particles(self):
        """ Test the add_particles from the generate_abstract_model."""
        
        # The gen_abtmodel should automatically generate abstract particles

        sm_path = import_ufo.find_ufo_path('sm')
        full_sm_base = import_ufo.import_full_model(sm_path)
        self.my_testmodel_new = decay_objects.DecayModel(full_sm_base, True)
        param_path = os.path.join(_file_path,
                                  '../input_files/param_card_full_sm.dat')
        self.my_testmodel_new.read_param_card(param_path)
        

        ab_model = self.my_testmodel_new['ab_model']
        ab_model.setup_particles(self.my_testmodel_new['particles'])
        goal_abpart_keys = set([(1,1,True), (2,1,False), (2,3,False), 
                                (3,1,True), (3,8,True)])
        goal_abpart_prop = {(1,1,True):('S1_00', 'none', 
                                         'MSS1_00', 9901100),
                            (2,1,False):('F1_00', 'F1_00~', 
                                         'MNF1_00', 9902100),
                            (2,3,False):('F3_00', 'F3_00~', 
                                         'MNF3_00', 9902300),
                            (3,1,True):('V1_00', 'none',
                                         'MSV1_00', 9903100),
                            (3,8,True):('V8_00', 'none', 
                                         'MSV8_00', 9903800)}

        # Check keys in abstract_particles_dict
        self.assertEqual(set(ab_model.get('abstract_particles_dict').keys()),
                         goal_abpart_keys)
        for plist in ab_model.get('abstract_particles_dict').values():
            self.assertEqual(len(plist), 1)
            self.assertEqual(tuple([plist[0].get('name'),
                                    plist[0].get('antiname'),
                                    plist[0].get('mass'),
                                    plist[0].get('pdg_code')]),
                             goal_abpart_prop[\
                    ab_model.get_particle_type(plist[0])])

        # Check for exceptions
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.setup_particles,
                          'NoneParticleList')
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.setup_particle,
                          'NoneParticle')

    def test_setup_interactions(self):
        """ Test the sets_interactions and get_interaction_type. """

        normal_sm_base = import_ufo.import_model('sm-full')
        normal_sm = decay_objects.DecayModel(normal_sm_base,
                                             force=True)
        param_path = os.path.join(_file_path,'../../models/sm/restrict_ckm.dat')
        normal_sm.read_param_card(param_path)
        #print normal_sm.get_interaction(59)
        normal_sm.generate_abstract_model()
        ab_model = normal_sm['ab_model']

        # Test if the interaction_type_dict/ abstract_interactions_dict
        # has been constructed properly.
        for inter in normal_sm.get('interactions'):
            try:
                inter_type = ab_model['interaction_type_dict'][inter['id']]
            except KeyError:
                continue


            # Test the particlelist type
            # Note: the particlelist type in inter_type has transformed into
            # tuple
            parttype, sndict =ab_model.get_particlelist_type([p.get_pdg_code() for p in inter['particles']])
            self.assertEqual(set(inter_type[0]), set(parttype))

            # Test the lorentz type is the superset of real lorentz
            self.assertTrue(set(inter_type[1]).issuperset(inter['lorentz']))

            # Test the color is the superset of real lorentz
            self.assertTrue(set(inter_type[2]).issuperset([str(c) for c in \
                                                               inter['color']]))

            # Test if the abstract_interactions_dict has been established
            self.assertTrue(inter_type in \
                                ab_model['abstract_interactions_dict'].keys())

            # Test if the coupling_dict has been established
            for ab_key, real_coup in \
                    ab_model['interaction_coupling_dict'][inter['id']].items():

                ab_inter = ab_model['abstract_interactions_dict'][inter_type][0]
                lorentz = ab_inter['lorentz'][ab_key[1]]
                color = ab_inter['color'][ab_key[0]]
                try:
                    real_key = [0,0]
                    real_key[0] = inter['color'].index(color)
                    real_key[1] = inter['lorentz'].index(lorentz)
                    self.assertEqual(inter['couplings'][tuple(real_key)],
                                     real_coup)
                except (ValueError, KeyError):
                    self.assertEqual(real_coup, 'ZERO')

            # Test the interaction_dict
            ab_inter = ab_model['abstract_interactions_dict'][inter_type][0]
            self.assertTrue(ab_model.get_interaction(ab_inter['id']))

            # Test anti interaction
            _has_anti = False
            if inter['id'] in normal_sm['conj_int_dict'].keys():
                _has_anti = True
            if _has_anti:
                self.assertEqual(ab_model['interaction_type_dict'][normal_sm['conj_int_dict'][inter['id']]],
                                 ab_model['interaction_type_dict'][-inter['id']])
                self.assertEqual(ab_model['interaction_coupling_dict'][normal_sm['conj_int_dict'][inter['id']]], 
                                 ab_model['interaction_coupling_dict'][-inter['id']])

        # Test for non-repeated interaction id
        # Test if all interactions in ab_model is up-to-date
        id_list = [i['id'] for i in ab_model['interactions']]
        new_inter_list = [intlist[0] for key, intlist \
                              in ab_model['abstract_interactions_dict'].items()]

        for ab_inter in ab_model['interactions']:
            self.assertEqual(id_list.count(ab_inter['id']), 1)
            self.assertTrue(ab_inter in new_inter_list)


        # Test if the lorentz and color types have no intersection
        # if the particles type are the same            
        def lorentzcmp(x,y):
            return cmp(x[1],y[1])
        keylist = sorted(ab_model['abstract_interactions_dict'].keys(), lorentzcmp)
        #print normal_sm.get_particle(1000021)['self_antipart']
        #print "Interaction type (%d):" % len(keylist)
        #for k in keylist:
        #    print k

        def check_keys(keylist):
            try:
                key = keylist.pop()
            except:
                return True
            for other in keylist:
                if other[0] == key[0]:
                    self.assertTrue(set(other[1]).isdisjoint(key[1]))
                    self.assertTrue(set(other[2]).isdisjoint(key[2]))

            return check_keys(keylist)

        self.assertTrue(check_keys(keylist))


        # Test for exceptions
        # Set an interaction that will not be included in AbstractModel
        # (The [22, 6, -6] only works for testmodel = standard model)
        for inter in normal_sm['interactions']:
            if set([21, 6, -6]) == \
                    set([part.get_pdg_code() for part in inter['particles']]):
                I_rad = inter
            if [25, 25, 25] == \
                    [part.get_pdg_code() for part in inter['particles']]:
                I_hhh = inter

        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.get_interaction_type,
                          I_rad.get('id'))
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.get_interaction_type,
                          I_hhh.get('id'))


    def test_add_ab_interaction(self):
        """Test the add_ab_interaction. """
        
        ab_model = self.my_testmodel['ab_model']
        self.my_testmodel.generate_abstract_model()

        # Test for exceptions
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.add_ab_interaction,
                          'NoneParticle')

        # Test for setting existing particle
        for inter in self.my_testmodel['interactions']:
            if set([24, -6, 5]) == \
                    set([part.get_pdg_code() for part in inter['particles']]):
                I_wtb = inter

        #print I_wtb
        ab_model.add_ab_interaction(I_wtb.get('id'))
        inter_type = ab_model.get_interaction_type(I_wtb.get('id'))
        # The real inter_type is strictly sorted tuple
        inter_type = (inter_type[0], inter_type[1], inter_type[2])

        # Check if new interaction is in paticles and abstract_particles_dict
        self.assertEqual(len(ab_model['abstract_interactions_dict'][inter_type]),  
                         2)
        new_inter = ab_model['abstract_interactions_dict'][inter_type][-1]
        self.assertTrue(new_inter in ab_model.get('interactions'))

        # Check properties
        type_sn = int(new_inter.get('id') /1000)
        #print type_sn, inter_type[1], inter_type[2], new_inter['couplings']
        self.assertEqual(new_inter.get('id') % 10, 1)
        self.assertEqual(new_inter.get('color'), I_wtb['color'])
        self.assertEqual(new_inter.get('lorentz'), ['FFV2', 'FFV3', 'FFV5'])
        self.assertEqual(new_inter.get('couplings'), 
                         {(0,0):'G%03d0001' %type_sn,
                          (0,1):'G%03d0101' %type_sn,
                          (0,2):'G%03d0201' %type_sn})


    def test_get_interactions_type(self):
        """Test the set_new_particle. """
        
        ab_model = self.my_testmodel['ab_model']
        self.my_testmodel.generate_abstract_model()

        higgs = self.my_testmodel.get_particle(25)
        tau = self.my_testmodel.get_particle(15)

        higgs.find_channels(3, self.my_testmodel)
        tau.find_channels(3, self.my_testmodel)
        #for c in higgs.get_channels(3, True):
        #    print c.nice_string()
        #for c in tau.get_channels(3, True):
        #    print c.nice_string()

        h_zz_zbb = higgs.get_channels(3, True)[0]
        h_zz_zee = higgs.get_channels(3, True)[1]
        h_zz_zmm = higgs.get_channels(3, True)[4]
        #print h_zz_zbb.nice_string(), h_zz_zee.nice_string()
        tau_wvt_vteve = tau.get_channels(3, True)[0]
        tau_wvt_vtmvm = tau.get_channels(3, True)[1]
        #print tau_wvt_vteve.nice_string(), tau_wvt_vtmvm.nice_string()

        
        # Test get_interactionlist_type
        input_list_1 = [v.get('id') for v in h_zz_zbb['vertices']]
        input_list_3 = [v.get('id') for v in tau_wvt_vteve['vertices']]
        input_list_3.append(input_list_3[0])
        FFV_type = ab_model['interaction_type_dict'][abs(input_list_3[0])]
        FFV_type_sn = \
            int(ab_model['abstract_interactions_dict'][FFV_type][0]['id']/1000)

        goal_list3_default = \
            [FFV_type_sn*1000, FFV_type_sn*1000+1, FFV_type_sn*1000]
        goal_list3_ignoredup = \
            [FFV_type_sn*1000, FFV_type_sn*1000+1, FFV_type_sn*1000+2]
        goal_list_nonzerostart = \
            [FFV_type_sn*1000+3, FFV_type_sn*1000+4, FFV_type_sn*1000+3]

        goal_serial_dict3_default = {FFV_type: 2}
        goal_serial_dict3_ignoredup = {FFV_type: 3}
        goal_serial_dict_nonzerostart = {FFV_type: 5} 

        #print ab_model.get_particlelist_type.serial_number_dict
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3)[0],
                         goal_list3_default)
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3, 
                                                           True)[0],
                         goal_list3_ignoredup)
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3)[1],
                         goal_serial_dict3_default)
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3,
                                                           True)[1],
                         goal_serial_dict3_ignoredup)
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3,
                                                           sn_dict=goal_serial_dict3_ignoredup)[0],
                         goal_list_nonzerostart)
        self.assertEqual(ab_model.get_interactionlist_type(input_list_3,
                                                           sn_dict=goal_serial_dict3_ignoredup)[1],
                         goal_serial_dict_nonzerostart)



    def test_help_generate_ab_amplitude(self):
        """ Test helper functions for generate abstract amplitude,
        including compare_diagrams, add_ab_diagrams, 
        generate_variables_dicts, and set_final_legs_dict."""

        ab_model = self.my_testmodel['ab_model']
        self.my_testmodel.generate_abstract_model()

        #----------------------
        # Set amplitudes
        #----------------------
        # Setup higgs, lower the mass,  and find amplitudes
        #decay_objects.mdl_MH = 50

        higgs = self.my_testmodel.get_particle(25)
        tau = self.my_testmodel.get_particle(15)
        zboson = self.my_testmodel.get_particle(23)

        higgs.find_channels(3, self.my_testmodel)
        tau.find_channels(3, self.my_testmodel)
        #for c in higgs.get_channels(3, True):
        #    print c.nice_string()
        #for c in tau.get_channels(3, True):
        #    print c.nice_string()

        h_zz_zbb = higgs.get_channels(3, True)[0]
        h_zz_zee = higgs.get_channels(3, True)[4]
        h_ww_weve = higgs.get_channels(3, True)[2]
        h_zz_zmm = higgs.get_channels(3, True)[5]
        #print h_zz_zbb.nice_string(), h_zz_zee.nice_string(), h_ww_weve.nice_string()
        tau_wvt_vteve = tau.get_channels(3, True)[0]
        tau_wvt_vtmvm = tau.get_channels(3, True)[1]
        #print tau_wvt_vteve.nice_string(), tau_wvt_vtmvm.nice_string()


        #----------------------
        # Test set_final_legs_dict
        #----------------------
        ab2realdict = decay_objects.Ab2RealDict()
        ab2realdict['process'] = tau.get_amplitude([16, 11, -12])['process']
        ab_process = copy.deepcopy(tau.get_amplitude([16, 11, -12])['process'])
        ab_process['model'] = ab_model
        ab2realdict['ab_process'] = ab_process
        real_pids = [l['id'] for l in tau_wvt_vteve.get_final_legs()]
        ab2realdict.set_final_legs_dict()
        self.assertEqual({-9902101:-12, 9902102:11, 9902103:16},
                         ab2realdict['final_legs_dict'])
        ab2realdict.set_final_legs_dict(real_dia=tau_wvt_vteve)
        self.assertEqual({-9902101:-12, 9902102:11, 9902103:16},
                         ab2realdict['final_legs_dict'])
        ab2realdict.set_final_legs_dict(ab_object=ab_model)
        self.assertEqual({-9902101:-12, 9902102:11, 9902103:16},
                         ab2realdict['final_legs_dict'])
        self.assertRaises(decay_objects.Ab2RealDict.PhysicsObjectError,
                          ab2realdict.set_final_legs_dict,
                          'Non-model')
        

        #----------------------
        # Test add abstract diagram
        #----------------------
        ab_amp = decay_objects.DecayAmplitude()
        # test i/o
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.add_ab_diagram,
                          h_zz_zee, h_zz_zbb)
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.add_ab_diagram,
                          ab_amp, ab_amp)

        # Set the initial dict, including the final and initial legs
        ab_amp['part_sn_dict'] = {(1,1,True): 1, (2,3,False):2, (3,1,True):1}
        ab_amp['ab2real_dicts'].append(decay_objects.Ab2RealDict())

        ab_model.add_ab_diagram(ab_amp, h_zz_zbb)
        ab_dia = ab_amp['diagrams'][0]
        #print ab_dia.nice_string(), h_zz_zbb.nice_string()
        goal_pids = set([-9902300, 9902301, 9903101, 9903101, 9903100, 
                      9901100])
        result_pids = []
        [result_pids.extend([l.get('id') for l in v.get('legs')]) for v in ab_dia['vertices']]
        self.assertEqual(set(result_pids), goal_pids)
        real_iid_1 = zboson['decay_vertexlist'][(2, True)][0]['id']
        real_iid_2 = higgs['decay_vertexlist'][(2, False)][1]['id']
        goal_interids = [ab_model['abstract_interactions_dict'][\
                ab_model.get_interaction_type(real_iid_1)][0]['id'],
                         ab_model['abstract_interactions_dict'][\
                ab_model.get_interaction_type(real_iid_2)][0]['id']]

        self.assertEqual([v.get('id') for v in ab_dia['vertices']], 
                          goal_interids)
        self.assertEqual(ab_amp['diagrams'][-1]['abstract_type'],
                         [goal_interids, [9903100], [9902301, -9902300, 9903100]])
        # Add the second diagram
        # Abstract type should remain the same
        ab_model.add_ab_diagram(ab_amp, h_zz_zbb)
        self.assertEqual(ab_amp['diagrams'][-1]['abstract_type'], 
                         ab_amp['diagrams'][-2]['abstract_type'])


        #----------------------
        # Test compare_diagrams
        #----------------------
        # test i/o
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.compare_diagrams,
                          ab_amp, h_zz_zbb)
        self.assertRaises(decay_objects.AbstractModel.PhysicsObjectError,
                          ab_model.compare_diagrams,
                          ab_dia, h_zz_zbb, {})

        ab_amp['ab2real_dicts'][-1].set_final_legs_dict(ab_model, h_zz_zbb)
        #print ab_dia.nice_string(), h_zz_zbb.nice_string()
        self.assertTrue(ab_model.compare_diagrams(ab_dia, h_zz_zbb,
                                                  ab_amp['ab2real_dicts'][-1]))
        # Test if no Ab2RealDict
        self.assertTrue(ab_model.compare_diagrams(ab_dia, h_zz_zbb))
        self.assertEqual(ab_amp['ab2real_dicts'][-1]['final_legs_dict'],
                         {-9902300:-5, 9902301:5, 9903100:23})
        # Set a wrong interaction id
        h_zz_zbb_wrong = copy.deepcopy(h_zz_zbb)
        h_zz_zbb_wrong['vertices'][1]['id'] = real_iid_1
        self.assertFalse(ab_model.compare_diagrams(ab_dia, h_zz_zbb_wrong))

        # h_zz_zmm != h_zz_zbb
        # no final_legs_dict available
        self.assertFalse(ab_model.compare_diagrams(ab_dia, h_zz_zmm,
                                                   ab_amp['ab2real_dicts'][-1]))

        # Construct abstract diagram for h_zz_zmm
        ab_amp_2 = decay_objects.DecayAmplitude()
        ab_amp_2['part_sn_dict'] = {(1,1,True): 1, (2,1,False):2, (3,1,True):1}
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        ab_model.add_ab_diagram(ab_amp_2, h_zz_zmm)
        ab_dia_2 = ab_amp_2['diagrams'][0]

        # h_zz_zmm == h_zz_zee
        ab_amp_2['ab2real_dicts'][-1].set_final_legs_dict(ab_model, h_zz_zmm)
        self.assertTrue(ab_model.compare_diagrams(ab_dia_2, h_zz_zmm,
                                                  ab_amp_2['ab2real_dicts'][-1]))
        self.assertTrue(ab_model.compare_diagrams(ab_dia_2, h_zz_zmm))

        # Same final state, but not consistent as final_legs_dict
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        ab_amp_2['ab2real_dicts'][-1]['final_legs_dict'] = {-9902100:11, 9902101:-11, 9903100:23}
        self.assertFalse(ab_model.compare_diagrams(ab_dia_2, h_zz_zee,
                                                   ab_amp_2['ab2real_dicts'][-1]))
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        
        #self.assertTrue(ab_model.compare_diagrams(ab_dia_2, h_ww_weve,
        #                                          ab_amp_2['ab2real_dicts'][-1]))
        #self.assertFalse(ab_model.compare_diagrams(ab_dia_2, h_zz_zbb,
        #                                           ab_amp_2['ab2real_dicts'][-1]))



    def test_help_generate_ab_amplitude_2(self):
        """ Test helper functions for generate abstract amplitude,
        including compare_diagrams, add_ab_diagrams, and ."""

        ab_model = self.my_testmodel['ab_model']
        self.my_testmodel.generate_abstract_model()

        # Setup higgs, lower the mass,  and find amplitudes
        decay_objects.mdl_MH = 50

        higgs = self.my_testmodel.get_particle(25)
        tau = self.my_testmodel.get_particle(15)
        zboson = self.my_testmodel.get_particle(23)
        wboson = self.my_testmodel.get_particle(24)

        higgs.find_channels(4, self.my_testmodel)
        wboson.find_channels(2, self.my_testmodel)
        h_zz_bbbb = None
        # ta = tau
        h_zz_tatabb = None
        h_zz_tatatata = None
        h_zz_tataee = None
        h_zz_tatamm = None
        for i,c in enumerate(higgs.get_channels(4, True)):
            tag = [l['id'] for l in c.get_final_legs()]
            if tag == [5, -5, 5, -5] and not h_zz_bbbb:
                h_zz_bbbb = c
            elif tag == [15, -15, 5, -5] and not h_zz_tatabb:
                h_zz_tatabb = c
            elif tag == [15, -15, 15, -15] and not h_zz_tatatata:
                h_zz_tatatata = c
            elif tag == [15, -15, 13, -13] and not h_zz_tatamm:
                 h_zz_tatamm = c
            elif tag == [15, -15, 11, -11]and not h_zz_tataee:
                 h_zz_tataee = c         
                        
        for i,c in enumerate(wboson.get_channels(2, True)):
            tag = [l['id'] for l in c.get_final_legs()]
            if tag == [16, -15]:
                w_lvl = c
                break

        #print h_zz_bbbb.nice_string(), h_zz_tatabb.nice_string(), \
        #    h_zz_tatatata.nice_string(), h_zz_tataee.nice_string()

        # Amplitude
        h_zz_eevv = higgs.get_amplitude([-11, 11, 12, -12])
        #print h_zz_eevv.nice_string()

        #----------------------
        # Test add abstract diagram
        #----------------------
        ab_amp = decay_objects.DecayAmplitude()
        # Set the initial dict, including the final and initial legs
        ab_amp['part_sn_dict'] = {(1,1,True): 1, 
                                  (2,3,False):4}
        ab_amp['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        ab_model.add_ab_diagram(ab_amp, h_zz_bbbb)
        ab_dia = ab_amp['diagrams'][0]
        #print ab_dia.nice_string(), h_zz_bbbb.nice_string()
        goal_pids = set([-9902300, 9902302, 9903100, 
                      -9902301, 9902303, 9903101,
                      9903101, 9903100, 9901100])

        result_pids = []
        [result_pids.extend([l.get('id') for l in v.get('legs')]) for v in ab_dia['vertices']]
        self.assertEqual(set(result_pids), goal_pids)

        real_iid_1 = zboson['decay_vertexlist'][(2, True)][0]['id']
        real_iid_2 = higgs['decay_vertexlist'][(2, False)][1]['id']
        ab_iid_1 = ab_model['abstract_interactions_dict'][\
                ab_model.get_interaction_type(real_iid_1)][0]['id']
        ab_iid_2 = ab_model['abstract_interactions_dict'][\
                ab_model.get_interaction_type(real_iid_2)][0]['id']
        goal_interids = [ab_iid_1, ab_iid_1, ab_iid_2]
        self.assertEqual([v.get('id') for v in ab_dia['vertices']], 
                          goal_interids)

        self.assertEqual(ab_amp['diagrams'][-1]['abstract_type'],
                         [goal_interids, [9903100, 9903101], 
                          [9902302, -9902300, 9902303, -9902301]])

        #----------------------
        # Test compare diagrams, set_final_legs_dict, again
        #----------------------
        # Test set final_legs_dict
        ab_amp['ab2real_dicts'][-1].set_final_legs_dict(ab_model, h_zz_bbbb)
        self.assertTrue(ab_model.compare_diagrams(ab_dia, h_zz_bbbb,
                                                  ab_amp['ab2real_dicts'][-1]))
        #print ab_amp['ab2real_dicts'][-1]['final_legs_dict']
        self.assertEqual(ab_amp['ab2real_dicts'][-1]['final_legs_dict'],
                         {-9902300:-5,-9902301:-5, 9902302:5, 9902303:5})


        # Test if the diagrams in one amplitude will share the same
        # final particle correspondence
        ab_amp = decay_objects.DecayAmplitude()
        ab_amp['part_sn_dict'] = {(1,1,True): 1, 
                                  (2,1,False):4}
        ab_amp['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        ab_model.add_ab_diagram(ab_amp, h_zz_eevv['diagrams'][0])
        ab_dia_1 = ab_amp['diagrams'][0]
        ab_amp['ab2real_dicts'][-1].set_final_legs_dict(ab_model,
                                                        h_zz_eevv['diagrams'][0])
        ab_model.add_ab_diagram(ab_amp, h_zz_eevv['diagrams'][1])
        ab_dia_2 = ab_amp['diagrams'][-1]
        #print ab_amp.nice_string(), h_zz_eevv.nice_string()
        # ab_dia_1: -e, ve, e, -ve; ab_dia_2: -ve, ve, -e, e
        self.assertEqual(set([l['id'] for l in ab_dia_2.get_final_legs()]),
                         set([9902103, -9902100, 9902102, -9902101]))

        #----------------------
        # Test compare diagrams, set_final_legs_dict, again
        #----------------------
        # h_zz_zmm != h_zz_zbb
        # no final_legs_dict available
        self.assertFalse(ab_model.compare_diagrams(ab_dia, h_zz_tataee,
                                                   ab_amp['ab2real_dicts'][-1]))

        # Construct abstract diagram for h_zz_zmm
        ab_amp_2 = decay_objects.DecayAmplitude()
        ab_amp_2['part_sn_dict'] = {(1,1,True): 1, 
                                    (2,1,False):4}
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        ab_model.add_ab_diagram(ab_amp_2, h_zz_tataee)
        ab_dia_2 = ab_amp_2['diagrams'][0]
        # h_zz_tau tau e e == h_zz_tau tau tau tau
        ab_amp_2['ab2real_dicts'][-1].set_final_legs_dict(ab_model,
                                                          h_zz_tataee)
        self.assertTrue(ab_model.compare_diagrams(\
                ab_dia_2, 
                h_zz_tataee,
                ab_amp_2['ab2real_dicts'][-1]))
        # For other diagram, add new Ab2RealDict
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        # ZZ > tau tau tau tau != ZZ > tau tau e e
        # because the former use the same interaction but not the latter
        self.assertFalse(ab_model.compare_diagrams(\
                ab_dia_2, 
                h_zz_tatatata,
                ab_amp_2['ab2real_dicts'][-1]))

        # tau tau e e == tau tau mu mu
        ab_amp_2['ab2real_dicts'].append(decay_objects.Ab2RealDict())
        self.assertTrue(ab_model.compare_diagrams(\
                ab_dia_2, 
                h_zz_tatamm,
                ab_amp_2['ab2real_dicts'][-1]))
        # Set final legs, the ab_dia_2 should be the same type
        ab_amp_2['ab2real_dicts'][-1].set_final_legs_dict(ab_model, 
                                                          h_zz_tatamm)
        self.assertTrue(ab_model.compare_diagrams(\
                ab_dia_2, 
                h_zz_tatamm,
                ab_amp_2['ab2real_dicts'][-1]))

        #----------------------
        # Test generate_variables_dicts
        #----------------------       
        ab_amp['process']['legs'] = base_objects.LegList([\
                base_objects.Leg({'id':9901100, 'state':False}),
                base_objects.Leg({'id':-9902100, 'state':True}),
                base_objects.Leg({'id':9902101, 'state':True}),
                base_objects.Leg({'id':-9902102, 'state':True}),
                base_objects.Leg({'id':9902103, 'state':True})])
        ab_amp['process']['model'] = ab_model
        ab_amp['ab2real_dicts'][-1]['dia_sn_dict'] = {0:0, 1:1}
        ab_amp.generate_variables_dicts(h_zz_eevv)

        # Test for initial particle
        #print ab_amp.nice_string(), h_zz_eevv.nice_string()
        #print ab_amp['ab2real_dicts'][-1]['mass_dict']
        #print ab_amp['ab2real_dicts'][-1]['coup_dict']
        #print ab_model['interaction_coupling_dict'][54], self.my_testmodel.get_interaction(54), ab_model['abstract_interactions_dict'][ab_model['interaction_type_dict'][54]]

        self.assertEqual(ab_amp['ab2real_dicts'][-1]['mass_dict'],
                         {'MSS1_00':'mdl_MH',
                          # lepton mass
                          'MNF1_00':'ZERO', 'MNF1_01':'ZERO',
                          'MNF1_02':'ZERO', 'MNF1_03':'ZERO',
                          # w, z boson
                          'MSV1_00':'mdl_MW', 'MSV1_01':'mdl_MW',
                          'MSV1_02':'mdl_MZ', 'MSV1_03':'mdl_MZ'})
        
        coupling = {}
        for candidate in self.my_testmodel['interactions']:
            coupling[tuple([p['pdg_code'] for p in candidate['particles']])] = \
                                candidate['couplings'][(0,0)]
            if [p['pdg_code'] for p in candidate['particles']] == [11,11,23]:
                second_z = candidate['couplings'][(0,1)]
      
        self.assertEqual(ab_amp['ab2real_dicts'][-1]['coup_dict'],
                         {# type: s > vv
                          #         : h > w+ w-
                          'G0010000': coupling[(24,24,25)],
                          #         : h > z z
                          'G0010001': coupling[(23,23,25)],
                          # type: v > ff
                          #         : w+ > e+ ve
                          'G0060000': coupling[(12,11,24)],
                          #         : w- > e- ve~
                          'G0060001': coupling[(12,11,24)],
                          #         : z > ve ve~
                          'G0060002': coupling[(12,12,23)],
                          #         : z > e+ e- (lorentz=0)
                          'G0060003': coupling[(11,11,23)],
                          #         : z > e+ e- (lorentz=1)
                          'G0060103':second_z
                          })

        #----------------------
        # Test generate_ab_amplitude
        #----------------------       
        ab_model.generate_ab_amplitudes(higgs.get_amplitudes(4))
        #print higgs.get_amplitudes(4).nice_string()
        # amp of abstract higgs:
        # l l~ l' l'~, q q~ l l~, q q~ q q~
        amplist = ab_model.get_particle(9901100).get_amplitudes(4)
        self.assertEqual(len(amplist), 3)
        #print amplist.abstract_nice_string()

        wboson.find_channels(3, self.my_testmodel)
        ab_model.generate_ab_amplitudes(wboson.get_amplitudes(2))
        amplist2 = ab_model.get_particle(9903100).get_amplitudes(2)
        # w+ > l vl~ (3 families)
        self.assertEqual(len(amplist2), 1)
        #print amplist2.abstract_nice_string()

        tau.find_channels(3, self.my_testmodel)
        ab_model.generate_ab_amplitudes(tau.get_amplitudes(3))
        amplist3 = ab_model.get_particle(9902100).get_amplitudes(3)
        # tt > vt l vl~ (3 families)
        self.assertEqual(len(amplist3), 1)
        #print amplist3.abstract_nice_string()


        # Test if the diagrams in each amp have consistent final leg numbers.
        for testlist in [amplist, amplist2, amplist3]:
            for amp in testlist:
                number_to_id_dict = dict([(l['number'],l['id']) \
                                              for l in amp.get('process')['legs']])
                #print number_to_id_dict
                for channel in amp['diagrams']:
                    for l in channel.get_final_legs():
                        self.assertEqual(l['id'],
                                         number_to_id_dict[l['number']])
                        
                        # ini_leg has anti pid compared to process
                        ini_leg = channel['vertices'][-1]['legs'][-1]
                        anti_pid = ab_model.get_particle(ini_leg['id']).get_anti_pdg_code()
                        self.assertEqual(anti_pid,
                                         number_to_id_dict[ini_leg['number']])


        # Test the ab2real_dicts for mother which is not self-conjugate.
        # Note that w boson is viewed as self-conjugate in abstract model.
        #print amplist2[0].abstract_nice_string()
        self.assertEqual(amplist2[0]['ab2real_dicts'][-1]['mass_dict'],
                         {'MSV1_00':'mdl_MW',
                          # lepton mass
                          'MNF1_00':'mdl_MTA', 'MNF1_01':'ZERO'})
        self.assertEqual(amplist3[0]['ab2real_dicts'][-1]['mass_dict'],
                         {'MNF1_00':'mdl_MTA',
                          # wboson mass
                          'MSV1_00':'mdl_MW',
                          # lepton mass
                          'MNF1_01':'ZERO', 'MNF1_02':'ZERO', 'MNF1_03':'ZERO'})


    def test_generate_ab_amplitudes(self):
        """ Test generate the abstract amplitudes, matrixelement. """

        model_type = 'sm'
        if model_type == 'full_sm':
            sm_path = import_ufo.find_ufo_path('sm')
            full_sm_base = import_ufo.import_full_model(sm_path)
            full_sm = decay_objects.DecayModel(full_sm_base, True)
            param_path = os.path.join(_file_path,
                                      '../input_files/param_card_full_sm.dat')
            full_sm.read_param_card(param_path)
        else:
            
            normal_sm_base = import_ufo.import_model(\
                model_type)
            normal_sm = decay_objects.DecayModel(normal_sm_base,
                                                 force=True)
            param_path = os.path.join(_file_path,
                                      '../input_files/param_card_%s.dat'%model_type)

        normal_sm.read_param_card(param_path)
        #print normal_sm.get_interaction(59)        
        #normal_sm.generate_abstract_model()
        normal_sm.find_all_channels(3, generate_abstract=True)
        #normal_sm.find_vertexlist()
        #normal_sm.generate_abstract_amplitudes(3)
        ab_model = normal_sm['ab_model']
        if model_type == 'sm' or model_type == 'full_sm':
            #print ab_model.get_particle(9901100).get_amplitudes(3).abstract_nice_string()
            # h > q q'~ (w,z),  h > l l'~ (w,z)
            self.assertEqual(len(ab_model.get_particle(9901100).get_amplitudes(3)), 
                             2)

            # tau > q q'~ vt,  h > l l'~ vt
            self.assertEqual(len(ab_model.get_particle(9902100).get_amplitudes(3)), 
                             2)
            # tau > q q'~ vt,  h > l l'~ vt
            self.assertEqual(len(ab_model.get_particle(9902300).get_amplitudes(2)), 
                             1)
            # The CP is not invariant for w > q q'~
            if model_type == 'full_sm':
                self.assertTrue(ab_model.get_particle(9901100).get_amplitudes(3)[0]['ab2real_dicts'][0]['coup_dict'] != \
                                    ab_model.get_particle(9901100).get_amplitudes(3)[0]['ab2real_dicts'][1]['coup_dict'])


        ab_model.generate_ab_matrixelements_all()

if __name__ == '__main__':
    unittest.unittest.main()
