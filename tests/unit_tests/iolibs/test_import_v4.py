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

"""Unit test library for the import v4 format routines"""

import unittest
import StringIO

import madgraph.iolibs.import_v4 as import_v4
import madgraph.core.base_objects as base_objects

class IOImportV4Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_particles(self):
        """Test the output of import particles.dat file"""

        particles_dat_str = """# Test string with particles.dat formating
                                ve ve~ F S ZERO ZERO S ve 12
                                w+ w- V W MW WW S W 24
                                T1 T1 T D ZERO ZERO O T1 8000002
                                # And now some bad format entries
                                # which should be ignored with a warning
                                k+ k- X S ZERO ZERO O K 60
                                1x k- X S ZERO ZERO O K 60
                                k+ k- S S ZERO ZERO V K 60"""

        fsock = StringIO.StringIO(particles_dat_str)

        goal_part_list = base_objects.ParticleList(\
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
                                                      'propagating':True}),
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
                                                      'propagating':True}),
                                 base_objects.Particle({'name':'T1',
                                                      'antiname':'T1',
                                                      'spin':5,
                                                      'color':8,
                                                      'mass':'ZERO',
                                                      'width':'ZERO',
                                                      'texname':'T1',
                                                      'antitexname':'T1',
                                                      'line':'dashed',
                                                      'charge': 0.,
                                                      'pdg_code':8000002,
                                                      'propagating':True})])

        self.assertEqual(import_v4.read_particles_v4(fsock), goal_part_list)
