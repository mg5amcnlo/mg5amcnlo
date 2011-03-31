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
        g = mypartlist[-1]

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
        u = mypartlist[-1]
        antiu = copy.copy(u)
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
        d = mypartlist[-1]
        antid = copy.copy(d)
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
        a = mypartlist[-1]

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
        eminus = mypartlist[-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

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
                                            [g] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             a]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of Z to quarks and electrons
        
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel = base_objects.Model()
        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)        
        self.mymodel.set('name', 'sm')

    def test_group_subprocs_and_get_diagram_maps(self):
        """Test grouping subprocs and generating HelasMatrixElements"""

        max_fs = 2

        p = [21, 1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});
        
        diagram_maps = [[{0: [0, 1, 2, 3]},
                         {0: [1, 2, 3]},
                         {0: [1, 2, 3], 1: [1, 2, 3]},
                         {0: [1, 2, 3], 1: [1, 2, 3]},
                         {0: [1, 1, 2, 3, 3, 4], 1: [5, 5, 6, 1, 1, 2], 2: [5, 5, 6], 3: [1, 1, 2], 4: [1, 1, 2], 5: [5, 5, 6, 3, 3, 4], 6: [5, 5, 6], 7: [1, 1, 2, 3, 3, 4], 8: [1, 1, 2], 9: [1, 1, 2], 10: [3, 3, 4], 11: [3, 3, 4], 12: [3, 3, 4], 13: [3, 3, 4]},
                         {0: [1, 2, 3], 1: [1, 2, 3]}
                         ]]
        
        diags_for_config = [[[[2], [3], [4]],
                            [[1], [2], [3]],
                            [[1, 1], [2, 2], [3, 3]],
                            [[1, 1], [2, 2], [3, 3]],
                            [[1, 4, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0], [3, 6, 0, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0, 0], [4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 1, 1, 1, 1], [6, 0, 0, 0, 0, 6, 0, 6, 0, 0, 3, 3, 3, 3], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0]],
                            [[1, 1], [2, 2], [3, 3]]]]
        
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
                for iconfig, config in enumerate(group.get('mapping_diagrams')):
                    self.assertEqual(group.get_subproc_diagrams_for_config(\
                                                          iconfig),
                                     diags_for_config[nfs-2][igroup][iconfig])

    def test_find_process_classes_and_mapping_diagrams(self):
        """Test the find_process_classes and find_mapping_diagrams function."""

        max_fs = 2 # 3

        p = [21, 1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        proc_classes = [{0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 4, 9: 5, 10: 4, 11: 4, 12: 4, 13: 4, 14: 3, 15: 5, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 3, 22: 4, 23: 4, 24: 4, 25: 5, 26: 4, 27: 4, 28: 3, 29: 4, 30: 4, 31: 5, 32: 4, 33: 4, 34: 4},
                        {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 3, 9: 2, 10: 3, 11: 3, 12: 2, 13: 3, 14: 3, 15: 4, 16: 5, 17: 5, 18: 6, 19: 7, 20: 6, 21: 6, 22: 6, 23: 6, 24: 4, 25: 5, 26: 5, 27: 7, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 4, 34: 5, 35: 5, 36: 6, 37: 6, 38: 6, 39: 7, 40: 6, 41: 6, 42: 4, 43: 5, 44: 5, 45: 6, 46: 6, 47: 7, 48: 6, 49: 6, 50: 6}]

        all_diagram_maps = [[{0: [0, 1, 2, 3]},
                             {0: [1, 2, 3], 1: [1, 2, 3]},
                             {0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                             {0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                             {0: [1, 1, 2, 3, 3, 4], 1: [5, 5, 6, 1, 1, 2], 2: [5, 5, 6], 3: [1, 1, 2], 4: [1, 1, 2], 5: [5, 5, 6, 3, 3, 4], 6: [5, 5, 6], 7: [1, 1, 2, 3, 3, 4], 8: [1, 1, 2], 9: [1, 1, 2], 10: [3, 3, 4], 11: [3, 3, 4], 12: [1, 1, 2, 3, 3, 4], 13: [5, 5, 6], 14: [5, 5, 6, 1, 1, 2], 15: [3, 3, 4], 16: [3, 3, 4], 17: [5, 5, 6], 18: [5, 5, 6, 3, 3, 4], 19: [1, 1, 2, 3, 3, 4]},
                             {0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]}],
                            [{0: [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15, 0, 0, 0, 0, 0, 0]},
{0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]},
{0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]},
{0: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18], 1: [3, 3, 4, 7, 7, 8, 9, 9, 10, 13, 13, 14, 17], 2: [19, 19, 20, 1, 1, 2, 5, 5, 6, 21, 21, 22, 23, 23, 24, 11, 11, 12, 15, 15, 16, 25, 25, 26, 18, 27], 3: [3, 3, 4, 7, 7, 8, 9, 9, 10, 13, 13, 14, 17], 4: [19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27], 5: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18], 6: [19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27], 7: [19, 19, 20, 1, 1, 2, 5, 5, 6, 21, 21, 22, 23, 23, 24, 11, 11, 12, 15, 15, 16, 25, 25, 26, 18, 27]},
{0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15], 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15]},
{0: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 17, 18], 1: [3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 15, 15, 16], 2: [19, 19, 20, 1, 1, 2, 10, 10, 11, 12, 12, 13, 14, 21, 21, 22, 23, 23, 24, 25, 17, 17, 18, 26, 26, 27], 3: [3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 15, 15, 16], 4: [19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 26, 27], 5: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 17, 18], 6: [19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 26, 27], 7: [19, 19, 20, 1, 1, 2, 10, 10, 11, 12, 12, 13, 14, 21, 21, 22, 23, 23, 24, 25, 17, 17, 18, 26, 26, 27]},
{0: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18], 1: [19, 19, 20, 21, 21, 22, 23, 3, 3, 4, 24, 24, 25, 5, 5, 6, 7, 8, 8, 9, 26, 26, 27, 17, 17, 18], 2: [19, 19, 20, 21, 21, 22, 23, 24, 24, 25, 26, 26, 27], 3: [3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 17, 17, 18], 4: [3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 17, 17, 18], 5: [19, 19, 20, 21, 21, 22, 23, 1, 1, 2, 24, 24, 25, 10, 10, 11, 12, 13, 13, 14, 26, 26, 27, 15, 15, 16], 6: [19, 19, 20, 21, 21, 22, 23, 24, 24, 25, 26, 26, 27], 7: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18], 8: [3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 17, 17, 18], 9: [3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 17, 17, 18], 10: [1, 1, 2, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16], 11: [1, 1, 2, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16], 12: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18], 13: [19, 19, 20, 21, 21, 22, 23, 24, 24, 25, 26, 26, 27], 14: [19, 19, 20, 21, 21, 22, 23, 3, 3, 4, 24, 24, 25, 5, 5, 6, 7, 8, 8, 9, 26, 26, 27, 17, 17, 18], 15: [1, 1, 2, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16], 16: [1, 1, 2, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16], 17: [19, 19, 20, 21, 21, 22, 23, 24, 24, 25, 26, 26, 27], 18: [19, 19, 20, 21, 21, 22, 23, 1, 1, 2, 24, 24, 25, 10, 10, 11, 12, 13, 13, 14, 26, 26, 27, 15, 15, 16], 19: [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18]},
{0: [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1: [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2: [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 3: [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}]]
        
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
            process_classes = group_subprocs.SubProcessGroup.\
                              find_process_classes(amplitudes)

            #print process_classes

            self.assertEqual(process_classes,
                             proc_classes[nfs-2])

            subproc_groups = group_subprocs.SubProcessGroup.\
                             group_amplitudes(amplitudes)

            for inum, group in enumerate(subproc_groups):
                mapping_diagrams, diagram_maps = group.find_mapping_diagrams()
                #print "mapping_diagrams: "
                #print "\n".join(["%d: %s" % (i+1, str(a)) for i,a in \
                #                 enumerate(mapping_diagrams)])
                for iamp, amplitude in enumerate(group.get('amplitudes')):
                    #print amplitude.nice_string()
                    self.assertEqual(diagram_maps[iamp],
                                     all_diagram_maps[nfs-2][inum][iamp])

    def test_group_decay_chains(self):
        """Test group_amplitudes for decay chains."""

        max_fs = 2 # 3

        procs = [[1,-1,2,-2,23], [2,2,2,2,23], [2,-2,21,21,23], [1,-1,21,21,23]]
        decays = [[23,1,-1,21], [23,11,-11]]
        coreamplitudes = diagram_generation.AmplitudeList()
        decayamplitudes = diagram_generation.AmplitudeList()
        decayprocs = base_objects.ProcessList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':self.mymodel,
                                               'orders':{'QED':1}})
            my_amplitude = diagram_generation.Amplitude(my_process)
            coreamplitudes.append(my_amplitude)

        for proc in decays:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':self.mymodel,
                                               'is_decay_chain': True})
            my_amplitude = diagram_generation.Amplitude(my_process)
            decayamplitudes.append(my_amplitude)
            decayprocs.append(my_process)

        decays = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': decayamplitudes})])

        decay_chains = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes,
            'decay_chains': decays})

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
                          group_amplitudes(decay_chains)

        #print dc_subproc_group.nice_string()
        
        self.assertEqual(dc_subproc_group.nice_string(),
"""Group 1:
  Process: d d~ > u u~ z QED=1
  4 diagrams:
  1  ((1(-1),2(1)>1(21),id:5),(3(2),5(23)>3(2),id:8),(1(21),3(2),4(-2),id:3)) (QED=1,QCD=2)
  2  ((1(-1),2(1)>1(21),id:5),(4(-2),5(23)>4(-2),id:8),(1(21),3(2),4(-2),id:3)) (QED=1,QCD=2)
  3  ((1(-1),5(23)>1(-1),id:9),(3(2),4(-2)>3(21),id:3),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
  4  ((2(1),5(23)>2(1),id:9),(3(2),4(-2)>3(21),id:3),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
  Process: u u > u u z QED=1
  8 diagrams:
  1  ((1(-2),3(2)>1(21),id:3),(2(-2),5(23)>2(-2),id:8),(1(21),2(-2),4(2),id:3)) (QED=1,QCD=2)
  2  ((1(-2),3(2)>1(21),id:3),(4(2),5(23)>4(2),id:8),(1(21),2(-2),4(2),id:3)) (QED=1,QCD=2)
  3  ((1(-2),4(2)>1(21),id:3),(2(-2),5(23)>2(-2),id:8),(1(21),2(-2),3(2),id:3)) (QED=1,QCD=2)
  4  ((1(-2),4(2)>1(21),id:3),(3(2),5(23)>3(2),id:8),(1(21),2(-2),3(2),id:3)) (QED=1,QCD=2)
  5  ((1(-2),5(23)>1(-2),id:8),(2(-2),3(2)>2(21),id:3),(1(-2),2(21),4(2),id:3)) (QED=1,QCD=2)
  6  ((1(-2),5(23)>1(-2),id:8),(2(-2),4(2)>2(21),id:3),(1(-2),2(21),3(2),id:3)) (QED=1,QCD=2)
  7  ((2(-2),3(2)>2(21),id:3),(4(2),5(23)>4(2),id:8),(1(-2),2(21),4(2),id:3)) (QED=1,QCD=2)
  8  ((2(-2),4(2)>2(21),id:3),(3(2),5(23)>3(2),id:8),(1(-2),2(21),3(2),id:3)) (QED=1,QCD=2)
Group 2:
  Process: u u~ > g g z QED=1
  8 diagrams:
  1  ((1(-2),3(21)>1(-2),id:3),(2(2),4(21)>2(2),id:3),(1(-2),2(2),5(23),id:8)) (QED=1,QCD=2)
  2  ((1(-2),3(21)>1(-2),id:3),(2(2),5(23)>2(2),id:8),(1(-2),2(2),4(21),id:3)) (QED=1,QCD=2)
  3  ((1(-2),4(21)>1(-2),id:3),(2(2),3(21)>2(2),id:3),(1(-2),2(2),5(23),id:8)) (QED=1,QCD=2)
  4  ((1(-2),4(21)>1(-2),id:3),(2(2),5(23)>2(2),id:8),(1(-2),2(2),3(21),id:3)) (QED=1,QCD=2)
  5  ((1(-2),5(23)>1(-2),id:8),(2(2),3(21)>2(2),id:3),(1(-2),2(2),4(21),id:3)) (QED=1,QCD=2)
  6  ((1(-2),5(23)>1(-2),id:8),(2(2),4(21)>2(2),id:3),(1(-2),2(2),3(21),id:3)) (QED=1,QCD=2)
  7  ((1(-2),5(23)>1(-2),id:8),(3(21),4(21)>3(21),id:1),(1(-2),2(2),3(21),id:3)) (QED=1,QCD=2)
  8  ((2(2),5(23)>2(2),id:8),(3(21),4(21)>3(21),id:1),(1(-2),2(2),3(21),id:3)) (QED=1,QCD=2)
  Process: d d~ > g g z QED=1
  8 diagrams:
  1  ((1(-1),3(21)>1(-1),id:5),(2(1),4(21)>2(1),id:5),(1(-1),2(1),5(23),id:9)) (QED=1,QCD=2)
  2  ((1(-1),3(21)>1(-1),id:5),(2(1),5(23)>2(1),id:9),(1(-1),2(1),4(21),id:5)) (QED=1,QCD=2)
  3  ((1(-1),4(21)>1(-1),id:5),(2(1),3(21)>2(1),id:5),(1(-1),2(1),5(23),id:9)) (QED=1,QCD=2)
  4  ((1(-1),4(21)>1(-1),id:5),(2(1),5(23)>2(1),id:9),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
  5  ((1(-1),5(23)>1(-1),id:9),(2(1),3(21)>2(1),id:5),(1(-1),2(1),4(21),id:5)) (QED=1,QCD=2)
  6  ((1(-1),5(23)>1(-1),id:9),(2(1),4(21)>2(1),id:5),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
  7  ((1(-1),5(23)>1(-1),id:9),(3(21),4(21)>3(21),id:1),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
  8  ((2(1),5(23)>2(1),id:9),(3(21),4(21)>3(21),id:1),(1(-1),2(1),3(21),id:5)) (QED=1,QCD=2)
Decay groups:
  Group 1:
    Process: z > d d~ g
    2 diagrams:
    1  ((2(1),4(21)>2(1),id:5),(2(1),3(-1)>2(23),id:9),(1(23),2(23),id:0)) (QED=1,QCD=1)
    2  ((3(-1),4(21)>3(-1),id:5),(2(1),3(-1)>2(23),id:9),(1(23),2(23),id:0)) (QED=1,QCD=1)
  Group 2:
    Process: z > e- e+
    1 diagrams:
    1  ((2(11),3(-11)>2(23),id:10),(1(23),2(23),id:0)) (QED=1)""")

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 4)

        group_names = ['qq_qqz_z_qqg',
                       'qq_qqz_z_emep',
                       'qq_ggz_z_qqg',
                       'qq_ggz_z_emep']

        for igroup, group in enumerate(subproc_groups):
            self.assertEqual(group.get('name'),
                             group_names[igroup])

