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

"""Unit test library for the import v4 format routines"""

import StringIO
import copy
import os

import tests.unit_tests as unittest

import madgraph.iolibs.import_v4 as import_v4
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.iolibs.files as files

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
#===============================================================================
# IOImportV4Test
#===============================================================================
class IOImportV4Test(unittest.TestCase):
    """Test class for the import v4 module"""

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
                                 base_objects.Particle({'name':'t1',
                                                      'antiname':'t1',
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
                                    e-   ve   w- MGVX24   QED
                                    e+   ve~   w+ MGVX25   QED
                                    u   u   g MGVX1   QCD
                                    u   u   a MGVX4   QED
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
        eminus = copy.copy(myparts[3])
        eplus = copy.copy(myparts[3])
        eplus.set('is_part', False)
        enu = copy.copy(myparts[0])
        enubar = copy.copy(myparts[0])
        enubar.set('is_part', False)
        photon = copy.copy(myparts[12])
        gluon = copy.copy(myparts[15])
        t1 = copy.copy(myparts[17])
        u = myparts[6]
        ubar = copy.copy(myparts[6])
        ubar.set('is_part', False)

        my_i_f = color.ColorString([color.f(0, 1, 2)])
        my_i_f.is_imaginary = True

        goal_inter_list = base_objects.InteractionList([ \
                    base_objects.Interaction(
                                    {'id':1,
                                     'particles':base_objects.ParticleList([
                                                                wplus,
                                                                wmin,
                                                                photon]),

                                     'color':[],
                                     'lorentz':[''],
                                     'couplings':{(0, 0):'MGVX3'},
                                     'orders':{'QED':1}}),
                     base_objects.Interaction(
                                    {'id':2,
                                     'particles':base_objects.ParticleList([
                                                                gluon,
                                                                gluon,
                                                                t1]),

                                     'color':[my_i_f],
                                     'lorentz':['A'],
                                     'couplings':{(0, 0):'MGVX2'},
                                     'orders':{'QCD':1}}),
                     base_objects.Interaction(
                                    {'id':3,
                                     'particles':base_objects.ParticleList([
                                                                wplus,
                                                                wmin,
                                                                wplus,
                                                                wmin]),
                                     'color':[],
                                     'lorentz':['WWVVN'],
                                     'couplings':{(0, 0):'MGVX6'},
                                     'orders':{'QED':2}}),

                     base_objects.Interaction(
                                    {'id':4,
                                     'particles':base_objects.ParticleList([
                                                                eplus,
                                                                enu,
                                                                wmin]),
                                     'color':[],
                                     'lorentz':[''],
                                     'couplings':{(0, 0):'MGVX24'},
                                     'orders':{'QED':1}}),

                     base_objects.Interaction(
                                    {'id':5,
                                     'particles':base_objects.ParticleList([
                                                                eminus,
                                                                enubar,
                                                                wplus]),
                                     'color':[],
                                     'lorentz':[''],
                                     'couplings':{(0, 0):'MGVX25'},
                                     'orders':{'QED':1}}),

                     base_objects.Interaction(
                                    {'id':6,
                                     'particles':base_objects.ParticleList([
                                                                ubar,
                                                                u,
                                                                gluon]),
                                     'color':[color.ColorString(\
                                              [color.T(2, 1, 0)])],
                                     'lorentz':[''],
                                     'couplings':{(0, 0):'MGVX1'},
                                     'orders':{'QCD':1}}),

                     base_objects.Interaction(
                                    {'id':7,
                                     'particles':base_objects.ParticleList([
                                                                ubar,
                                                                u,
                                                                photon]),
                                     'color':[color.ColorString(\
                                              [color.T(1, 0)])],
                                     'lorentz':[''],
                                     'couplings':{(0, 0):'MGVX4'},
                                     'orders':{'QED':1}})])
        
        result = import_v4.read_interactions_v4(fsock_inter, myparts)
        self.assertEqual(len(result), len(goal_inter_list))
        for i in range(len(result)):
            self.assertEqual(result[i], goal_inter_list[i])

    def test_full_import(self):
        """Test importing the full SM"""
        model_path = os.path.join(_file_path, os.pardir, os.pardir,
                                  'models', 'sm_v4')
        model = import_v4.import_model(model_path)[0]
        self.assertEqual(model.get('coupling_orders'),
                         set(['QCD', 'QED']))
        self.assertEqual(model.get('order_hierarchy'),
                         {'QCD': 1, 'QED': 2})
        self.assertEqual(model.get('expansion_order'),
                         {'QCD': -1, 'QED': -1})
        
        
class ProcCardV4ReaderTest(unittest.TestCase):
    """Test class for the proc_card v4 module"""

    def setUp(self):
        """ open the proc_card and initialize the object"""
        v4proccard_file = open(os.path.join(_file_path, os.path.pardir, 'input_files', \
                                       'v4_proc_card.dat'))
        
        self.proccard = import_v4.ProcCardv4Reader(v4proccard_file)
        
        # First define a valid model for Standard Model
        model = base_objects.Model()
        # Import Particles information
        input_path = os.path.join(_file_path, '../input_files/v4_sm_particles.dat')
        model.set('particles', files.read_from_file(input_path,
                                            import_v4.read_particles_v4))
        # Import Interaction information
        input_path = os.path.join(_file_path , '../input_files/v4_sm_interactions.dat')
        model.set('interactions', files.read_from_file(input_path, \
                                               import_v4.read_interactions_v4, \
                                               model.get('particles')))
        self.model = model
        
    def test_check_init(self):
        """ check the initialization of the reader"""
        #the initialization is done in the setUp subroutine
        
        proccard = self.proccard
        self.assertEqual(len(proccard.process), 8)
        self.assertEqual(proccard.model, 'sm')
        self.assertEqual(len(proccard.multipart), 6)
        # Check that multiparticles are already loaded
        self.assertEqual(proccard.particles_name, set(
                                           ['l-', 'j', 'vl', 'l+', 'p', 'vl~']))
        self.assertEqual(proccard.couplings_name, set())
        
    def test_line_creation(self):
        lines = self.proccard.extract_command_lines(self.model)
        solution =['# Define multiparticle labels',
                   'define p u u~ c c~ d d~ s s~ g',
                   'define j u u~ c c~ d d~ s s~ g', 
                   'define l+ e+ mu+', 
                   'define l- e- mu-', 
                   'define vl ve vm', 
                   'define vl~ ve~ vm~', 
                   '# Specify process(es) to run',
                   'generate p p > ve~ e- @1 QED=2 QCD=99', 
                   'add process p p > z, (z > w+ w-, w- > mu- vm) @2 QED=4 QCD=2',
                   'add process p p > t t~ $ a @3 QED=0 QCD=99',
                   'add process p p > t t~ $ g / a @2 QED=0 QCD=99', 
                   'add process p p > z $ a / g, (z > w+ w- $ a / g, w- > mu- vm $ a / g) @4 QED=4 QCD=99',
                   'add process p p > z z $ a / g, (z > w+ w- $ a / g, w- > mu- vm $ a / g), z > w+ w- $ a / g @4 QED=1 QCD=99',
                   'add process p p > Z Z, Z > W+ W- $a /g @4',
                   'add process p p > Z Z QCD=2 @4',
                   '# Output processes to MadEvent directory',
                   'output -f'] 
        self.assertEqual(len(lines),len(solution))
        for i,command in enumerate(lines):
            self.assertEqual(command,solution[i])

