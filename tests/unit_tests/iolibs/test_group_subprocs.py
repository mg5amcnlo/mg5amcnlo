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

"""Unit test library for the export v4 format routines"""

import StringIO
import copy
import fractions
import os 

import tests.unit_tests as unittest

import madgraph.iolibs.misc as misc
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_algebra as color
import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.unit_tests.iolibs.test_helas_call_writers as \
                                            test_helas_call_writers

#===============================================================================
# SubProcessGroupTest
#===============================================================================
class SubProcessGroupTest(unittest.TestCase):
    """Test class for the SubProcessGroup class"""

    def setUp(self):

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(mypartlist[2])
        antid.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(mypartlist[4])
        antie.set('is_part', False)

        # A Z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[-1]

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[4], \
                                             antie, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel = base_objects.Model()
        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)        

    def test_group_subprocs_and_generate_matrix_elements(self):
        """Test grouping subprocs and generating HelasMatrixElements"""

        max_fs = 2

        p = [21, 1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});
        
        diagram_maps = [[{0: [0, 1, 2, 3], 1: [1, 2, 3]},
                         {0: [1, 2, 3]},
                         {0: [1, 2, 3]},
                         {0: [1, 2, 3]},
                         {0: [1, 1, 2, 3, 3, 4], 1: [1, 1, 2], 2: [3, 3, 4]},
                         {0: [1, 2, 3], 1: [1, 1, 4, 2, 2, 5], 2: [1, 1, 4],
                          3: [2, 2, 5], 4: [3, 3, 6]},
                         {0: [1, 2, 3]},
                         {0: [1, 2, 3], 1: [1, 1, 4, 3, 3, 5], 2: [1, 1, 4],
                          3: [2, 2, 6], 4: [3, 3, 5]},
                         {0: [1, 1, 2, 3, 3, 4], 1: [1, 1, 2], 2: [3, 3, 4]}]]
        
        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (2 + nfs)])

            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({\
                                                     'legs':my_multi_leglist,
                                                     'model':self.mymodel,
                                                     'orders': {'QED': nfs}})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            nproc = 0

            # Calculate diagrams for all processes

            amplitudes = my_multiprocess.get('amplitudes')

            subprocess_groups = group_subprocs.SubProcessGroup.\
                                group_amplitudes(amplitudes)


            for igroup, group in enumerate(subprocess_groups):
                group.get('matrix_elements')
                self.assertEqual(group.get('diagram_maps'),
                                 diagram_maps[nfs-2][igroup])
                
