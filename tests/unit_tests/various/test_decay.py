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
import tests.input_files.import_vertexlist as import_vertexlist

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# DecayParticleTest
#===============================================================================
class Test_DecayParticle(unittest.TestCase):
    """Test class for the DecayParticle object"""

    mydict = {}
    mypart = None
    my_testmodel_base = import_ufo.import_model('my_smallsm')
    my_2bodyvertexlist = base_objects.VertexList()
    my_3bodyvertexlist = base_objects.VertexList()
    my_2bodyvertexlist_wrongini = base_objects.VertexList()
    my_3bodyvertexlist_wrongini = base_objects.VertexList()

    def setUp(self):

        #Import a model from my_testmodel
        self.my_testmodel = decay_objects.DecayModel(self.my_testmodel_base)
        param_path = os.path.join(_file_path,'../input_files/param_card_sm.dat')
        self.my_testmodel.read_param_card(param_path)

        #Setup the vertexlist for my_testmodel
        import_vertexlist.make_vertexlist(self.my_testmodel)
        
        #Setup vertexlist for test
        full_vertexlist = import_vertexlist.full_vertexlist

        self.my_2bodyvertexlist = base_objects.VertexList([
                full_vertexlist[(1, -24)], full_vertexlist[(7, -24)]])
        self.my_3bodyvertexlist = base_objects.VertexList([
                full_vertexlist[(2, -24)], full_vertexlist[(3, -24)]])

        self.my_2bodyvertexlist_wrongini = base_objects.VertexList([
                full_vertexlist[(1, -24)], full_vertexlist[(6, -6)]])
        fake_vertex = copy.deepcopy(full_vertexlist[(2, 22)])
        fake_vertex['legs'][0]['id'] = 24
        self.my_3bodyvertexlist_wrongini = base_objects.VertexList([
                fake_vertex, full_vertexlist[(2, 22)]])
        
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
                      '2_body_decay_vertexlist':[self.my_2bodyvertexlist] *2,
                      '3_body_decay_vertexlist':[self.my_3bodyvertexlist] *2}

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
                      {'prop':'2_body_decay_vertexlist',
                        'right_list':[[self.my_2bodyvertexlist] * 2],
                        'wrong_list':[1, [self.my_2bodyvertexlist] * 3,
                                      ['hey', self.my_2bodyvertexlist],
                                      ['hey'] *2,
                                      [self.my_2bodyvertexlist, 
                                       self.my_3bodyvertexlist],
                                      [self.my_2bodyvertexlist_wrongini,
                                       self.my_2bodyvertexlist],
                                      [self.my_2bodyvertexlist,
                                       self.my_2bodyvertexlist_wrongini]
                                     ]},
                       {'prop':'3_body_decay_vertexlist',
                        'right_list':[[self.my_3bodyvertexlist] * 2],
                         'wrong_list':[1, [self.my_3bodyvertexlist] * 3,
                                      ['hey', self.my_3bodyvertexlist],
                                      ['hey'] *2,
                                      [self.my_2bodyvertexlist, 
                                       self.my_3bodyvertexlist],
                                      [self.my_3bodyvertexlist_wrongini,
                                       self.my_3bodyvertexlist],
                                      [self.my_3bodyvertexlist,
                                       self.my_3bodyvertexlist_wrongini]
                                     ]}
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
                            self.my_3bodyvertexlist_wrongini]

        for item in Wrong_vertexlist:
            for partnum in [2,3]:
                self.assertRaises(decay_objects.DecayParticle.PhysicsObjectError
                             , self.mypart.set_vertexlist, partnum, False, item)
        
    def test_find_vertexlist(self):
        #undefine object: my_testmodel, mypart, extra_part

        #Test validity of arguments
        #Test if the calling particle is in the model
        extra_part = copy.copy(self.mypart)
        extra_part.set('pdg_code', 2)
        extra_part.set('name', 'u')
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
        my_vertexlist620 = base_objects.VertexList([full_vertexlist[(6, -6)]])
        my_vertexlist621 = base_objects.VertexList([full_vertexlist[(8, -6)]])
        my_vertexlist630 = base_objects.VertexList()
        my_vertexlist631 = base_objects.VertexList()
        rightlist6 = [my_vertexlist620, my_vertexlist621, my_vertexlist630, my_vertexlist631]

        #Test the find_vertexlist for W+
        wboson_p = decay_objects.DecayParticle(self.my_testmodel.get_particle(24))
        wboson_p.find_vertexlist(self.my_testmodel)
        #List must follow the order of interaction id so as to be consistent
        #with the find_vertexlist function
        my_vertexlist2420 = base_objects.VertexList([full_vertexlist[(1, -24)],
                                                     full_vertexlist[(7, -24)]])
        my_vertexlist2421 = base_objects.VertexList([full_vertexlist[(9, -24)]])
        my_vertexlist2430 = base_objects.VertexList([full_vertexlist[(2, -24)],
                                                     full_vertexlist[(3, -24)]])
        my_vertexlist2431 = base_objects.VertexList()
        #List of the total decay vertex list for W+
        rightlist24 =[my_vertexlist2420, my_vertexlist2421, my_vertexlist2430, my_vertexlist2431]

        #Test the find_vertexlist for A (photon)
        photon = decay_objects.DecayParticle(self.my_testmodel.get_particle(22))
        photon.find_vertexlist(self.my_testmodel)
        #vertex is in the order of interaction id
        my_vertexlist2220 = base_objects.VertexList([full_vertexlist[(1,  22)],
                                                     full_vertexlist[(4, 22)],
                                                     full_vertexlist[(5, 22)],                                                       full_vertexlist[(6, 22)]])
        my_vertexlist2221 = base_objects.VertexList()
        my_vertexlist2230 = base_objects.VertexList([full_vertexlist[(2, 22)]])
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
