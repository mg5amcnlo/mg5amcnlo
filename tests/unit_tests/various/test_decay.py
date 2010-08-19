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

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.iolibs.import_ufo as import_ufo
import decay.decay_objects as decay_objects
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# DecayParticleTest
#===============================================================================
class Test_DecayParticle(unittest.TestCase):
    """Test class for the DecayParticle object"""

    mydict = {}
    mypart = None
    mymodel = {}
    my2leglist = base_objects.LegList([base_objects.Leg({'id':6,
                                                       'number': 4,
                                                       'state': False,
                                                       'from_group': False})]+ \
                                      [base_objects.Leg({'id':5,
                                                       'number': 4,
                                                       'state': True,
                                                       'from_group': False})] *2
                                     )
    my_2bodyvertexlist = base_objects.VertexList( 
                           [base_objects.Vertex({'id':1, 'legs':my2leglist})]*5)

    my3leglist = base_objects.LegList([base_objects.Leg({'id':6,
                                                       'number': 4,
                                                       'state': False,
                                                       'from_group': False})] +\
                                      [base_objects.Leg({'id':5,
                                                       'number': 4,
                                                       'state': True,
                                                       'from_group': False})]* 3
                                      )
    my_3bodyvertexlist = base_objects.VertexList(
                           [base_objects.Vertex({'id':1, 'legs':my3leglist})]*5)

    my_2bodyvertexlist_2ini = base_objects.VertexList()
    my_2bodyvertexlist_wrongini = base_objects.VertexList()
    my_3bodyvertexlist_2ini = base_objects.VertexList()
    my_3bodyvertexlist_wrongini = base_objects.VertexList()

    #Add one more initial particle to the leglist.
    my2leglist_2ini = base_objects.LegList(my2leglist + [my2leglist[0]])
    my3leglist_2ini = base_objects.LegList(my3leglist + [my3leglist[0]])

    #Vertex with more than one initial particles.
    my_2bodyvertexlist_2ini = base_objects.VertexList(my_2bodyvertexlist + \
        base_objects.VertexList([base_objects.Vertex({'id':1, 'legs':my2leglist_2ini})]))
    my_3bodyvertexlist_2ini = base_objects.VertexList(my_3bodyvertexlist + \
        base_objects.VertexList([base_objects.Vertex({'id':1, 'legs':my3leglist_2ini})]))


    #Initial particle of Vertex is not the same as the parent one.
    my_2bodyvertexlist_wrongini = copy.deepcopy(base_objects.VertexList(my_2bodyvertexlist))
    my_2bodyvertexlist_wrongini[0]['legs'][0]['id'] = 5
    my_3bodyvertexlist_wrongini = copy.deepcopy(base_objects.VertexList(my_3bodyvertexlist))
    my_3bodyvertexlist_wrongini[0]['legs'][0]['id'] = 5

    if my_2bodyvertexlist[0]['legs'][0]['id'] == 5:
        print 'Wrong here'

    def setUp(self):

        self.mydict = {'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'mt',
                      'width':'wt',
                      'texname':'t',
                      'antitexname':'\\overline{t}',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False,
                       # decay_vertexlist must have two lists, one for on-shell,
                       # one for off-shell
                      '2_body_decay_vertexlist':[self.my_2bodyvertexlist] *2,
                      '3_body_decay_vertexlist':[self.my_3bodyvertexlist] *2}

        self.mypart = decay_objects.DecayParticle(self.mydict)
        #self.mypart.set('2_body_decay_vertexlist',[self.my_2bodyvertexlist] *2)
        #self.mypart.set('3_body_decay_vertexlist',[self.my_3bodyvertexlist] *2)

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
        self.assertRaises(AssertionError,
                          decay_objects.DecayParticle,
                          myNondict)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          decay_objects.DecayParticle,
                          myWrongdict)
                          
        #Test get
        self.assertRaises(AssertionError,
                          self.mypart.get,
                          myNondict)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get,
                          'WrongParameter')
                          
        #Test set
        self.assertRaises(AssertionError,
                          self.mypart.set,
                          myNondict, 1)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set,
                          'WrongParameter', 1)

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
                        'right_list':[1, 12, 80000000, -1],
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
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'self_antipart',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                      {'prop':'2_body_decay_vertexlist',
                        'right_list':[[self.my_2bodyvertexlist] * 2],
                        'wrong_list':[1, [self.my_2bodyvertexlist] * 3,
                                      ['hey', self.my_2bodyvertexlist],
                                      ['hey'] *2,
                                      [self.my_2bodyvertexlist, 
                                       self.my_3bodyvertexlist],
                                      [self.my_2bodyvertexlist_2ini,
                                       self.my_2bodyvertexlist],
                                      [self.my_2bodyvertexlist,
                                       self.my_2bodyvertexlist_2ini],
                                      [self.my_2bodyvertexlist_wrongini,
                                       self.my_2bodyvertexlist],
                                      [self.my_2bodyvertexlist,
                                       self.my_2bodyvertexlist_wrongini]
                                     ]},
                       {'prop':'3_body_decay_vertexlist',
                        'right_list':[[self.my_3bodyvertexlist] *2 ],
                        'wrong_list':[1, [self.my_3bodyvertexlist] * 3,
                                      ['hey', self.my_3bodyvertexlist],
                                      ['hey'] *2,
                                      [self.my_2bodyvertexlist, 
                                       self.my_3bodyvertexlist],
                                      [self.my_3bodyvertexlist_2ini,
                                       self.my_3bodyvertexlist],
                                      [self.my_3bodyvertexlist,
                                       self.my_3bodyvertexlist_2ini],
                                      [self.my_3bodyvertexlist_wrongini,
                                       self.my_2bodyvertexlist],
                                      [self.my_2bodyvertexlist,
                                       self.my_3bodyvertexlist_wrongini]
                                     ]}
                       ]

        temp_part = self.mypart

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_part.set(test['prop'], x))
            for x in test['wrong_list']:
                #print test['prop'], x
                self.assertFalse(temp_part.set(test['prop'], x))

    def test_getsetvertexlist_correct(self):
        """Test the get and set for vertexlist is correct"""
        temp_part = self.mypart
        #Reset the off-shell '2_body_decay_vertexlist'
        templist = self.my_2bodyvertexlist
        templist.extend(templist)
        temp_part.set_decay_vertexlist(2, False, templist)
        #Test for equality from get_decay_vertexlist
        self.assertEqual(temp_part.get_decay_vertexlist(2, False), \
                             templist)

        #Reset the on-shell '2_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_decay_vertexlist(2, True, templist)
        #Test for equality from get_decay_vertexlist
        self.assertEqual(temp_part.get_decay_vertexlist(2, True), \
                             templist)

        #Reset the off-shell '3_body_decay_vertexlist'
        templist = self.my_3bodyvertexlist
        templist.extend(templist)
        temp_part.set_decay_vertexlist(3, False, templist)
        #Test for equality from get_decay_vertexlist
        self.assertEqual(temp_part.get_decay_vertexlist(3, False), \
                             templist)

        #Reset the on-shell '3_body_decay_vertexlist'
        templist.extend(templist)
        temp_part.set_decay_vertexlist(3, True, templist)
        #Test for equality from get_decay_vertexlist
        self.assertEqual(temp_part.get_decay_vertexlist(3, True), \
                             templist)

    def test_getsetvertexlist_exceptions(self):
        """Test for the exceptions raised by the get_ or set_decay_vertexlist"""

        #Test of get_decay_vertexlist
        #Test the exceptions raised from partnum
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get_decay_vertexlist,
                          'string', True)
        
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get_decay_vertexlist,
                          1.5, True)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get_decay_vertexlist,
                          5, True)
        #Test the exceptions raised from the onshell
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get_decay_vertexlist,
                          2, 15)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.get_decay_vertexlist,
                          2, 'Notbool')


        #Test of set_decay_vertexlist
        #Test the exceptions raised from partnum
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          'string', True, self.my_2bodyvertexlist, self.mymodel)
        
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          1.5, True, self.my_2bodyvertexlist, self.mymodel)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          5, True, self.my_2bodyvertexlist, self.mymodel)

        #Test the exceptions raised from the onshell
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          2, 15, self.my_2bodyvertexlist, self.mymodel)

        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          2, 'Notbool', self.my_2bodyvertexlist, self.mymodel)

        #Test the exceptions raised from value in set_decay_vertexlist
        #Test for non vertexlist objects
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          2, True, ['not', 'Vertexlist'], self.mymodel)

        #Test for vertexlist not consistent with partnum
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          2, True, self.my_3bodyvertexlist, self.mymodel)
        self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                          self.mypart.set_decay_vertexlist,
                          3, True, self.my_2bodyvertexlist, self.mymodel)

        #Test for vertexlist not consistent with initial particle
        #for both number and id
        #Use the vertexlist from test_getsetvertexlist_exceptions

        Wrong_vertexlist = [self.my_2bodyvertexlist_wrongini,
                            self.my_2bodyvertexlist_2ini,
                            self.my_3bodyvertexlist_wrongini,
                            self.my_3bodyvertexlist_2ini]


        for item in Wrong_vertexlist:
            for partnum in [2,3]:
                self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError,
                                  self.mypart.set_decay_vertexlist,
                                  partnum, False, item)
        
    def test_find_vertexlist(self):
        pass

#===============================================================================
# TestDecayModel
#===============================================================================
class TestDecayModel(unittest.TestCase):
    """Test class for the DecayModel object"""

    base_model = import_ufo.import_model('sm')

    def setUp(self):
        """Set up decay model"""
        self.decay_model = decay_objects.DecayModel(self.base_model)
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

if __name__ == '__main__':
    unittest.unittest.main()
