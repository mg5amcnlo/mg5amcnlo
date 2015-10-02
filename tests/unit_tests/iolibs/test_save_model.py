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

"""Unit test library for the save model routines"""

import StringIO

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects

import madgraph.iolibs.files as files
import madgraph.iolibs.save_model as save_model

class IOSaveModel(unittest.TestCase):
    """Unit test class for the save model routines"""

    def test_error_particle_save(self):
        """Test error raising for particle save"""

        fsock = StringIO.StringIO('')

        my_part_list = "This is not a particle list"

        self.assertRaises(ValueError,
                         save_model.save_particles,
                         fsock, my_part_list)

    def test_particle_save(self):
        """Test the particle save routine"""

        mypartlist = \
        base_objects.ParticleList(\
                                [base_objects.Particle({'name':'ve',
                                                      'antiname':'ve~',
                                                      'spin':2,
                                                      'color':1,
                                                      'mass':'ZERO',
                                                      'width':'ZERO',
                                                      'texname':'ve',
                                                      'antitexname':'ve',
                                                      'line':'straight',
                                                      'charge': 0.,
                                                      'pdg_code':12,
                                                      'propagating':True,
                                                      'is_part':True,
                                                      'self_antipart':False}),
                                 base_objects.Particle({'name':'w+',
                                                      'antiname':'w-',
                                                      'spin':3,
                                                      'color':1,
                                                      'mass':'MW',
                                                      'width':'WW',
                                                      'texname':'W',
                                                      'antitexname':'W',
                                                      'line':'wavy',
                                                      'charge':0.,
                                                      'pdg_code':24,
                                                      'propagating':True,
                                                      'is_part':True,
                                                      'self_antipart':False})])

        fsock = StringIO.StringIO()
        save_model.save_particles(fsock, mypartlist)

        goal_str = "particles = [\n%s,%s]" % (str(mypartlist[0]),
                                            str(mypartlist[1]))

        self.assertEqual(fsock.getvalue(), goal_str)

    def test_error_interaction_save(self):
        """Test error raising for interaction save"""

        fsock = StringIO.StringIO('')

        my_inter_list = "This is not an interaction list"

        self.assertRaises(ValueError,
                         save_model.save_interactions,
                         fsock, my_inter_list)

    def test_interaction_save(self):
        """Test the interaction save routine"""

        mypart = base_objects.Particle({'name':'t',
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
                      'is_part':True})

        myinterlist = base_objects.InteractionList([\
                        base_objects.Interaction({'id': 1,
                       'particles': base_objects.ParticleList([mypart] * 4),
                       'color': [],
                       'lorentz':['L1', 'L2'],
                       'couplings':{(0, 0):'g00',
                                    (0, 1):'g01',
                                    (1, 0):'g10',
                                    (1, 1):'g11'},
                       'orders':{'QCD':1, 'QED':1}})])

        fsock = StringIO.StringIO()
        save_model.save_interactions(fsock, myinterlist)

        goal_str = "interactions = [\n%s]" % str(myinterlist[0])

        self.assertEqual(fsock.getvalue(), goal_str)

