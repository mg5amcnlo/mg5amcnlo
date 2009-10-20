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

"""Unit test library for the import v4 format routines"""

import StringIO
import unittest
import copy

import madgraph.iolibs.import_v4 as import_v4
import madgraph.core.base_objects as base_objects

#===============================================================================
# IOImportV4Test
#===============================================================================
class IOImportV4Test(unittest.TestCase):
    """Test class for the import v4 module"""

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
                                                      'self_antipart':False}),
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
                                                      'propagating':True,
                                                      'is_part':True,
                                                      'self_antipart':True})])

        self.assertEqual(import_v4.read_particles_v4(fsock), goal_part_list)

    def test_read_interactions(self):
        """Test the output of import interactions.dat file"""

        particles_dat_str = """ve ve~ F S ZERO ZERO S ve 12
                                vm vm~ F S ZERO ZERO S vm 14
                                vt vt~ F S ZERO ZERO S vt 16
                                e- e+ F S ZERO ZERO S e 11
                                m- m+ F S ZERO ZERO S m 13
                                tt- tt+ F S MTA ZERO S tt 15
                                u u~ F S ZERO ZERO T u 2
                                c c~ F S MC ZERO T c 4
                                t t~ F S MT WT T t 6
                                d d~ F S ZERO ZERO T d 1
                                s s~ F S ZERO ZERO T s 3
                                b b~ F S MB ZERO T b 5
                                a a V W ZERO ZERO S a 22
                                z z V W MZ WZ S Z 23
                                w+ w- V W MW WW S W 24
                                g g V C ZERO ZERO O G 21
                                h h S D MH WH S H 25
                                T1 T1 T D ZERO ZERO O T1 8000002"""

        interactions_dat_str = """# Interactions associated with Standard_Model
                                    w+   w-   a MGVX3   QED
                                    g   g   T1 MGVX2   QCD a
                                    w+   w-   w+   w- MGVX6   DUM0   QED QED n
                                    # And now some bad format entries
                                    # which should be ignored with a warning
                                    k+ k- a test QED
                                    g g test QCD"""

        fsock_part = StringIO.StringIO(particles_dat_str)
        fsock_inter = StringIO.StringIO(interactions_dat_str)

        myparts = import_v4.read_particles_v4(fsock_part)

        wplus = copy.copy(myparts[14])
        wmin = copy.copy(myparts[14])
        wmin.set('is_part', False)
        photon = copy.copy(myparts[12])
        gluon = copy.copy(myparts[15])
        t1 = copy.copy(myparts[17])

        goal_inter_list = base_objects.InteractionList([ \
                    base_objects.Interaction(
                                    {'id':1,
                                     'particles':base_objects.ParticleList([
                                                                wplus,
                                                                wmin,
                                                                photon]),
                                     'color':['guess'],
                                     'lorentz':['guess'],
                                     'couplings':{(0, 0):'MGVX3'},
                                     'orders':{'QED':1}}),
                     base_objects.Interaction(
                                    {'id':2,
                                     'particles':base_objects.ParticleList([
                                                                gluon,
                                                                gluon,
                                                                t1]),
                                     'color':['guess'],
                                     'lorentz':['guess'],
                                     'couplings':{(0, 0):'MGVX2'},
                                     'orders':{'QCD':1}}),
                     base_objects.Interaction(
                                    {'id':3,
                                     'particles':base_objects.ParticleList([
                                                                wplus,
                                                                wmin,
                                                                wplus,
                                                                wmin]),
                                     'color':['guess'],
                                     'lorentz':['guess'],
                                     'couplings':{(0, 0):'MGVX6'},
                                     'orders':{'QED':2}})])

        self.assertEqual(import_v4.read_interactions_v4(fsock_inter,
                                                        myparts),
                                                goal_inter_list)

