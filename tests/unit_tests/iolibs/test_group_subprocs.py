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

"""Unit test library for the export v4 format routines"""

import StringIO
import copy
import fractions
import os 

import tests.unit_tests as unittest

import madgraph.various.misc as misc
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

        diagram_maps =  [[{0: [0, 1, 2, 3]},
                          {0: [1, 2, 3]},
                          {0: [1, 2, 3], 1: [1, 3, 2]},
                          {0: [1, 2, 3], 1: [1, 3, 2]},
                          {0: [1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 1, 2, 3], 2: [7, 8, 9], 3: [1, 2, 3], 4: [1, 2, 3], 5: [7, 8, 9, 4, 5, 6], 6: [7, 8, 9], 7: [1, 2, 3, 4, 5, 6], 8: [1, 2, 3], 9: [1, 2, 3]},
                          {0: [1, 2, 3], 1: [1, 3, 2]}]]
        diags_for_config =  [[[[2], [3], [4]],
                              [[1], [2], [3]],
                              [[1, 1], [2, 3], [3, 2]],
                              [[1, 1], [2, 3], [3, 2]],
                              [[1, 4, 0, 1, 1, 0, 0, 1, 1, 1], [2, 5, 0, 2, 2, 0, 0, 2, 2, 2], [3, 6, 0, 3, 3, 0, 0, 3, 3, 3], [4, 0, 0, 0, 0, 4, 0, 4, 0, 0], [5, 0, 0, 0, 0, 5, 0, 5, 0, 0], [6, 0, 0, 0, 0, 6, 0, 6, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0, 0], [0, 2, 2, 0, 0, 2, 2, 0, 0, 0], [0, 3, 3, 0, 0, 3, 3, 0, 0, 0]],
                              [[1, 1], [2, 3], [3, 2]]]]
        #new_diagram_maps = []
        #new_diags_for_config = []

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
                                group_amplitudes(amplitudes, "madevent")
            #dmaps = []
            #diags = []
            for igroup, group in enumerate(subprocess_groups):
                group.get('matrix_elements')
                #dmaps.append(group.get('diagram_maps'))
                self.assertEqual(group.get('diagram_maps'),
                                 diagram_maps[nfs-2][igroup])
                #diags.append([group.get_subproc_diagrams_for_config(ic) for\
                #              ic in range(len(group.get('mapping_diagrams')))])
                for iconfig, config in enumerate(group.get('mapping_diagrams')):
                    self.assertEqual(group.get_subproc_diagrams_for_config(\
                                                          iconfig),
                                     diags_for_config[nfs-2][igroup][iconfig])
                    pass
            #new_diagram_maps.append(dmaps)
            #new_diags_for_config.append(diags)
        #print 'diagram_maps = ',new_diagram_maps
        #print 'diags_for_config = ',new_diags_for_config

    def test_find_process_classes_and_mapping_diagrams(self):
        """Test the find_process_classes and find_mapping_diagrams function."""

        max_fs = 3

        p = [21, 1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        proc_classes = [{0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 4, 9: 5, 10: 4, 11: 4, 12: 4, 13: 4, 14: 3, 15: 5, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 3, 22: 4, 23: 4, 24: 4, 25: 5, 26: 4, 27: 4, 28: 3, 29: 4, 30: 4, 31: 5, 32: 4, 33: 4, 34: 4},
                        {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 3, 9: 2, 10: 3, 11: 3, 12: 2, 13: 3, 14: 3, 15: 4, 16: 5, 17: 5, 18: 6, 19: 7, 20: 6, 21: 6, 22: 6, 23: 6, 24: 4, 25: 5, 26: 5, 27: 7, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 4, 34: 5, 35: 5, 36: 6, 37: 6, 38: 6, 39: 7, 40: 6, 41: 6, 42: 4, 43: 5, 44: 5, 45: 6, 46: 6, 47: 7, 48: 6, 49: 6, 50: 6}]

        all_diagram_maps =  [[{0: [0, 1, 2, 3]},
                              {0: [1, 2, 3], 1: [1, 2, 3]},
                              {0: [1, 2, 3], 1: [1, 3, 2], 2: [1, 2, 3], 3: [1, 3, 2]},
                              {0: [1, 2, 3], 1: [1, 3, 2], 2: [1, 2, 3], 3: [1, 3, 2]},
                              {0: [1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 1, 2, 3], 2: [7, 8, 9], 3: [1, 2, 3], 4: [1, 2, 3], 5: [7, 8, 9, 4, 5, 6], 6: [7, 8, 9], 7: [1, 2, 3, 4, 5, 6], 8: [1, 2, 3], 9: [1, 2, 3], 10: [4, 5, 6], 11: [4, 5, 6], 12: [1, 2, 3, 4, 5, 6], 13: [7, 8, 9], 14: [7, 8, 9, 1, 2, 3], 15: [4, 5, 6], 16: [4, 5, 6], 17: [7, 8, 9], 18: [7, 8, 9, 4, 5, 6], 19: [1, 2, 3, 4, 5, 6]},
                              {0: [1, 2, 3], 1: [1, 3, 2], 2: [1, 2, 3], 3: [1, 3, 2]}],
                             [{0: [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15, 0, 0, 0, 0, 0, 0]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 3: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 1: [4, 5, 6, 10, 11, 12, 13, 14, 15, 19, 20, 21, 25], 2: [27, 28, 29, 16, 17, 18, 30, 31, 32, 1, 2, 3, 7, 8, 9, 33, 34, 35, 22, 23, 24, 36, 37, 38, 26, 39], 3: [10, 11, 12, 4, 5, 6, 13, 14, 15, 19, 20, 21, 25], 4: [27, 28, 29, 33, 34, 35, 36, 37, 38, 30, 31, 32, 39], 5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 6: [27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37, 38, 39], 7: [27, 28, 29, 16, 17, 18, 30, 31, 32, 1, 2, 3, 7, 8, 9, 33, 34, 35, 22, 23, 24, 36, 37, 38, 26, 39]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 3: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 1: [4, 5, 6, 10, 11, 12, 13, 14, 15, 19, 20, 21, 25], 2: [27, 28, 29, 16, 17, 18, 30, 31, 32, 1, 2, 3, 7, 8, 9, 33, 34, 35, 22, 23, 24, 36, 37, 38, 26, 39], 3: [10, 11, 12, 4, 5, 6, 13, 14, 15, 19, 20, 21, 25], 4: [27, 28, 29, 33, 34, 35, 36, 37, 38, 30, 31, 32, 39], 5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 6: [27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37, 38, 39], 7: [27, 28, 29, 16, 17, 18, 30, 31, 32, 1, 2, 3, 7, 8, 9, 33, 34, 35, 22, 23, 24, 36, 37, 38, 26, 39]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 1: [4, 5, 6, 27, 28, 29, 19, 20, 21, 30, 31, 32, 33, 34, 35, 10, 11, 12, 36, 37, 38, 13, 14, 15, 39, 25], 2: [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], 3: [4, 5, 6, 10, 11, 12, 13, 14, 15, 19, 20, 21, 25], 4: [4, 5, 6, 10, 11, 12, 19, 20, 21, 13, 14, 15, 25], 5: [22, 23, 24, 30, 31, 32, 1, 2, 3, 27, 28, 29, 33, 34, 35, 7, 8, 9, 36, 37, 38, 16, 17, 18, 39, 26], 6: [30, 31, 32, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39], 7: [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 25, 26], 8: [10, 11, 12, 4, 5, 6, 13, 14, 15, 19, 20, 21, 25], 9: [10, 11, 12, 4, 5, 6, 19, 20, 21, 13, 14, 15, 25], 10: [22, 23, 24, 7, 8, 9, 16, 17, 18, 1, 2, 3, 26], 11: [7, 8, 9, 22, 23, 24, 16, 17, 18, 1, 2, 3, 26], 12: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 13: [36, 37, 38, 33, 34, 35, 30, 31, 32, 27, 28, 29, 39], 14: [4, 5, 6, 27, 28, 29, 19, 20, 21, 30, 31, 32, 33, 34, 35, 10, 11, 12, 36, 37, 38, 13, 14, 15, 39, 25], 15: [22, 23, 24, 7, 8, 9, 1, 2, 3, 16, 17, 18, 26], 16: [7, 8, 9, 22, 23, 24, 1, 2, 3, 16, 17, 18, 26], 17: [36, 37, 38, 33, 34, 35, 27, 28, 29, 30, 31, 32, 39], 18: [22, 23, 24, 30, 31, 32, 1, 2, 3, 27, 28, 29, 33, 34, 35, 7, 8, 9, 36, 37, 38, 16, 17, 18, 39, 26], 19: [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 25, 26]},
                              {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 1: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], 3: [2, 1, 3, 5, 4, 6, 10, 11, 12, 7, 8, 9, 13, 15, 14, 0]}]]
        
        #new_diagram_maps = []

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
                              find_process_classes(amplitudes, "madevent")

            #print process_classes

            self.assertEqual(process_classes,
                             proc_classes[nfs-2])

            subproc_groups = group_subprocs.SubProcessGroup.\
                             group_amplitudes(amplitudes, "madevent")

            #dmaps = []
            for inum, group in enumerate(subproc_groups):
                mapping_diagrams, diagram_maps = group.find_mapping_diagrams()
                #print "mapping_diagrams: "
                #print "\n".join(["%d: %s" % (i+1, str(a)) for i,a in \
                #                 enumerate(mapping_diagrams)])
                #dmaps.append(diagram_maps)
                for iamp, amplitude in enumerate(group.get('amplitudes')):
                    #print amplitude.nice_string()
                    self.assertEqual(diagram_maps[iamp],
                                     all_diagram_maps[nfs-2][inum][iamp])
                    pass
            #new_diagram_maps.append(dmaps)

        #print "all_diagram_maps = ",new_diagram_maps
        
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
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([decay_chains]))

        #print dc_subproc_group.nice_string()
        
        self.assertEqual(dc_subproc_group.nice_string(),
"""Group 1:
  Process: d d~ > u u~ z QED<=1
  4 diagrams:
  1  ((1(-1),2(1)>1(21),id:5),(3(2),5(23)>3(2),id:8),(1(21),3(2),4(-2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  2  ((1(-1),2(1)>1(21),id:5),(4(-2),5(23)>4(-2),id:8),(1(21),3(2),4(-2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  3  ((1(-1),5(23)>1(-1),id:9),(3(2),4(-2)>3(21),id:3),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  4  ((2(1),5(23)>2(1),id:9),(3(2),4(-2)>3(21),id:3),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  Process: u u > u u z QED<=1
  8 diagrams:
  1  ((1(-2),3(2)>1(21),id:3),(2(-2),5(23)>2(-2),id:8),(1(21),2(-2),4(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  2  ((1(-2),3(2)>1(21),id:3),(4(2),5(23)>4(2),id:8),(1(21),2(-2),4(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  3  ((1(-2),4(2)>1(21),id:3),(2(-2),5(23)>2(-2),id:8),(1(21),2(-2),3(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  4  ((1(-2),4(2)>1(21),id:3),(3(2),5(23)>3(2),id:8),(1(21),2(-2),3(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  5  ((1(-2),5(23)>1(-2),id:8),(2(-2),3(2)>2(21),id:3),(1(-2),2(21),4(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  6  ((1(-2),5(23)>1(-2),id:8),(2(-2),4(2)>2(21),id:3),(1(-2),2(21),3(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  7  ((2(-2),3(2)>2(21),id:3),(4(2),5(23)>4(2),id:8),(1(-2),2(21),4(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  8  ((2(-2),4(2)>2(21),id:3),(3(2),5(23)>3(2),id:8),(1(-2),2(21),3(2),id:3)) (QCD=2,QED=1,WEIGHTED=4)
Group 2:
  Process: u u~ > g g z QED<=1
  8 diagrams:
  1  ((1(-2),3(21)>1(-2),id:3),(2(2),4(21)>2(2),id:3),(1(-2),2(2),5(23),id:8)) (QCD=2,QED=1,WEIGHTED=4)
  2  ((1(-2),3(21)>1(-2),id:3),(2(2),5(23)>2(2),id:8),(1(-2),2(2),4(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  3  ((1(-2),4(21)>1(-2),id:3),(2(2),3(21)>2(2),id:3),(1(-2),2(2),5(23),id:8)) (QCD=2,QED=1,WEIGHTED=4)
  4  ((1(-2),4(21)>1(-2),id:3),(2(2),5(23)>2(2),id:8),(1(-2),2(2),3(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  5  ((1(-2),5(23)>1(-2),id:8),(2(2),3(21)>2(2),id:3),(1(-2),2(2),4(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  6  ((1(-2),5(23)>1(-2),id:8),(2(2),4(21)>2(2),id:3),(1(-2),2(2),3(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  7  ((1(-2),5(23)>1(-2),id:8),(3(21),4(21)>3(21),id:1),(1(-2),2(2),3(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  8  ((2(2),5(23)>2(2),id:8),(3(21),4(21)>3(21),id:1),(1(-2),2(2),3(21),id:3)) (QCD=2,QED=1,WEIGHTED=4)
  Process: d d~ > g g z QED<=1
  8 diagrams:
  1  ((1(-1),3(21)>1(-1),id:5),(2(1),4(21)>2(1),id:5),(1(-1),2(1),5(23),id:9)) (QCD=2,QED=1,WEIGHTED=4)
  2  ((1(-1),3(21)>1(-1),id:5),(2(1),5(23)>2(1),id:9),(1(-1),2(1),4(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  3  ((1(-1),4(21)>1(-1),id:5),(2(1),3(21)>2(1),id:5),(1(-1),2(1),5(23),id:9)) (QCD=2,QED=1,WEIGHTED=4)
  4  ((1(-1),4(21)>1(-1),id:5),(2(1),5(23)>2(1),id:9),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  5  ((1(-1),5(23)>1(-1),id:9),(2(1),3(21)>2(1),id:5),(1(-1),2(1),4(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  6  ((1(-1),5(23)>1(-1),id:9),(2(1),4(21)>2(1),id:5),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  7  ((1(-1),5(23)>1(-1),id:9),(3(21),4(21)>3(21),id:1),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
  8  ((2(1),5(23)>2(1),id:9),(3(21),4(21)>3(21),id:1),(1(-1),2(1),3(21),id:5)) (QCD=2,QED=1,WEIGHTED=4)
Decay groups:
  Group 1:
    Process: z > d d~ g
    2 diagrams:
    1  ((2(1),4(21)>2(1),id:5),(2(1),3(-1)>2(23),id:9),(1(23),2(23),id:0)) (QCD=1,QED=1,WEIGHTED=3)
    2  ((3(-1),4(21)>3(-1),id:5),(2(1),3(-1)>2(23),id:9),(1(23),2(23),id:0)) (QCD=1,QED=1,WEIGHTED=3)
  Group 2:
    Process: z > e- e+
    1 diagrams:
    1  ((2(11),3(-11)>2(23),id:10),(1(23),2(23),id:0)) (QCD=0,QED=1,WEIGHTED=2)""")

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 4)

        group_names = ['qq_qqz_z_qqg',
                       'qq_qqz_z_ll',
                       'qq_ggz_z_qqg',
                       'qq_ggz_z_ll']

        for igroup, group in enumerate(subproc_groups):
            self.assertEqual(group.get('name'),
                             group_names[igroup])

    def test_special_group_decay_chain(self):
        """Test group_amplitudes for special decay chains."""

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':1,
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

        # A gluino
        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':1,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'go',
                      'antitexname':'go',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        go = mypartlist[-1]

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':1,
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

        # A d squark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'dl',
                      'antiname':'dl~',
                      'spin':0,
                      'color':1,
                      'mass':'MDL',
                      'width':'WDL',
                      'texname':'dl',
                      'antitexname':'\bar dl',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1000001,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        dl = mypartlist[-1]
        antidl = copy.copy(dl)
        antidl.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'MN1',
                      'width':'WN1',
                      'texname':'n1',
                      'antitexname':'n1',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[-1]

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Gluon couplings to gluinos
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             go, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Gluino and neutralino couplings to quarks and squarks
        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [go,
                                             d,
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [antid,
                                             go,
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             d, \
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             n1, \
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Multiparticle labels
        ds = [1,-1]
        dls = [1000001,-1000001]

        proc = [21,21,1000021,1000021]

        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': [id]}) for id in proc])
        my_leglist[0].set('state', False)
        my_leglist[1].set('state', False)

        process = base_objects.ProcessDefinition({'legs':my_leglist,
                                                  'model':mymodel})

        decayproc1 = [[1000021],ds,dls]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decayproc1])
        my_leglist[0].set('state', False)
        decayprocess1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        decayproc2 = [dls,ds,[1000022]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decayproc2])
        my_leglist[0].set('state', False)
        decayprocess2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        decayprocess1.get('decay_chains').append(decayprocess2)
        process.get('decay_chains').append(decayprocess1)
        process.get('decay_chains').append(decayprocess1)

        my_amplitude = diagram_generation.DecayChainAmplitude(process)

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([my_amplitude]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 1)

        self.assertEqual(len(subproc_groups[0].get('matrix_elements')),3)

        me_strings = ["""Process: g g > go go WEIGHTED<=2
  Decay: go > d dl~ WEIGHTED<=1
    Decay: dl~ > d~ n1 WEIGHTED<=2
  Decay: go > d dl~ WEIGHTED<=1
    Decay: dl~ > d~ n1 WEIGHTED<=2""",
                      """Process: g g > go go WEIGHTED<=2
  Decay: go > d dl~ WEIGHTED<=1
    Decay: dl~ > d~ n1 WEIGHTED<=2
  Decay: go > d~ dl WEIGHTED<=1
    Decay: dl > d n1 WEIGHTED<=2""",
                      """Process: g g > go go WEIGHTED<=2
  Decay: go > d~ dl WEIGHTED<=1
    Decay: dl > d n1 WEIGHTED<=2
  Decay: go > d~ dl WEIGHTED<=1
    Decay: dl > d n1 WEIGHTED<=2"""]
        

        for i,me in enumerate(subproc_groups[0].get('matrix_elements')):
            self.assertEqual(me.get('processes')[0].nice_string(),
                             me_strings[i])
            

    def test_even_more_special_group_decay_chain(self):
        """Test group_amplitudes for even more special decay chain"""

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A d quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':1,
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

        # A u quark and its antiparticle
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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # An s and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = mypartlist[-1]
        antis = copy.copy(s)
        antis.set('is_part', False)

        # A c quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'c',
                      'antitexname':'\bar c',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':4,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        c = mypartlist[len(mypartlist) - 1]
        antic = copy.copy(c)
        antic.set('is_part', False)

        # A b quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':1,
                      'mass':'MB',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[-1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        # A t quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'zero',
                      'texname':'t',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

        # A funny Zprime
        mypartlist.append(base_objects.Particle({
            'name': 'Zp',
            'antiname': 'Zp',
            'spin': 3,
            'color': 1,
            'charge': 0.00,
            'mass': 'MZp',
            'width': 'WZp',
            'pdg_code': 9900032,
            'texname': 'Zp',
            'antitexname': 'Zp',
            'line': 'wavy',
            'propagating': True,
            'is_part': True,
            'self_antipart': True}))

        Zp = mypartlist[-1]

        # A funny neutralino
        mypartlist.append(base_objects.Particle({
                    'name': '~n1',
                    'antiname': '~n1',
                    'spin': 2,
                    'color': 1,
                    'charge': 0.00,
                    'mass': 'MnH1',
                    'width': 'WnH1',
                    'pdg_code': 9910012,
                    'texname': '~n1',
                    'antitexname': '~n1',
                    'line': 'swavy',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': True}))
        n1 = mypartlist[-1]

        # A W
        mypartlist.append(base_objects.Particle({'name':'W+',
                      'antiname':'W-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W^+',
                      'antitexname':'W^-',
                     'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        Wplus = mypartlist[len(mypartlist) - 1]
        Wminus = copy.copy(Wplus)
        Wminus.set('is_part', False)

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'me',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # Interactions

        myinterlist.append(base_objects.Interaction({
                    'id': 1,
                    'particles': base_objects.ParticleList(\
                        [antid,u,Wminus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_35'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 2,
                    'particles': base_objects.ParticleList(\
                        [antiu,d,Wplus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_112'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 3,
                    'particles': base_objects.ParticleList(\
                        [antis,c,Wminus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_35'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 4,
                    'particles': base_objects.ParticleList(\
                        [antic,s,Wplus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_112'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 5,
                    'particles': base_objects.ParticleList(\
                        [antib,t,Wminus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_35'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 6,
                    'particles': base_objects.ParticleList(\
                        [antit,b,Wplus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_112'},
                    'orders': {'QED': 1}}))

        myinterlist.append(base_objects.Interaction({
                    'id': 7,
                    'particles': base_objects.ParticleList(\
                        [antid,d,Zp]),
                    'color': [],
                    'lorentz': ['FFV1'],
                    'couplings': {(0, 0): 'GC_7'},
                    'orders': {'QED': 1}
                    }))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 8,
                    'particles': base_objects.ParticleList(\
                        [n1,n1,Zp]),
                    'color': [],
                    'lorentz': ['FFV2'],
                    'couplings': {(0, 0): 'GC_22'},
                    'orders': {'QED': 1}
                    }))

        myinterlist.append(base_objects.Interaction({
                    'id': 9,
                    'particles': base_objects.ParticleList(\
                        [eplus,n1,Wminus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_46'},
                    'orders': {'QED': 1}
                    }))

        myinterlist.append(base_objects.Interaction({
                    'id': 10,
                    'particles': base_objects.ParticleList(\
                        [n1,eminus,Wplus]),
                    'color': [],
                    'lorentz': ['FFV3'],
                    'couplings': {(0, 0): 'GC_46'},
                    'orders': {'QED': 1}
                    }))

        myinterlist.append(base_objects.Interaction({
                    'id': 11,
                    'particles': base_objects.ParticleList(\
                        [antit,c,Zp]),
                    'color': [],
                    'lorentz': ['FFV4'],
                    'couplings': {(0, 0): 'GC_50'},
                    'orders': {'QED': 1}
                    }))

        myinterlist.append(base_objects.Interaction({
                    'id': 12,
                    'particles': base_objects.ParticleList(\
                        [antic,t,Zp]),
                    'color': [],
                    'lorentz': ['FFV4'],
                    'couplings': {(0, 0): 'GC_50'},
                    'orders': {'QED': 1}
                    }))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Multiparticle labels
        ds = [1,-1]
        qs = [1,3,2,4,-1,-3,-2,-4]

        my_leglist = base_objects.MultiLegList()
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))

        core_process = base_objects.ProcessDefinition({'legs':my_leglist,
                                                       'model':mymodel})

        decay1proc1 = [[n1.get('pdg_code')], 
                      [11], [24]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay1proc1])
        my_leglist[0].set('state', False)
        decay1process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})
        decay1process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        decay2proc1 = [[24], qs, qs]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc1])
        my_leglist[0].set('state', False)
        decay2process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})
        decay2proc2 = [[24], [6], [-5]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc2])
        my_leglist[0].set('state', False)
        decay2process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})

        decay1process1.get('decay_chains').append(decay2process1)
        decay1process2.get('decay_chains').append(decay2process2)
        core_process.get('decay_chains').append(decay1process1)
        core_process.get('decay_chains').append(decay1process2)

        my_amplitude = diagram_generation.DecayChainAmplitude(core_process)

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([my_amplitude]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 1)

        self.assertEqual(len(subproc_groups[0].get('matrix_elements')),2)

        me_strings = ["""Process: d d~ > ~n1 ~n1 WEIGHTED<=2
  Decay: ~n1 > e- W+ WEIGHTED<=1
    Decay: W+ > u d~ WEIGHTED<=1
  Decay: ~n1 > e- W+ WEIGHTED<=1
    Decay: W+ > t b~ WEIGHTED<=1""",
                      """Process: d~ d > ~n1 ~n1 WEIGHTED<=2
  Decay: ~n1 > e- W+ WEIGHTED<=1
    Decay: W+ > u d~ WEIGHTED<=1
  Decay: ~n1 > e- W+ WEIGHTED<=1
    Decay: W+ > t b~ WEIGHTED<=1"""]
        

        for i,me in enumerate(subproc_groups[0].get('matrix_elements')):
            self.assertEqual(me.get('processes')[0].nice_string(),
                             me_strings[i])
            

        # Now test also for different process ids

        ds = [1,-1]
        qs = [1,3,2,4,-1,-3,-2,-4]

        # First process
        my_leglist = base_objects.MultiLegList()
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))

        core_process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                       'model':mymodel,
                                                       'id': 1})

        decay1proc1 = [[n1.get('pdg_code')], 
                      [11], [24]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay1proc1])
        my_leglist[0].set('state', False)
        decay1process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})
        decay1process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        decay2proc1 = [[24], qs, qs]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc1])
        my_leglist[0].set('state', False)
        decay2process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})
        decay2proc2 = [[24], [6], [-5]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc2])
        my_leglist[0].set('state', False)
        decay2process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})

        decay1process1.get('decay_chains').append(decay2process1)
        decay1process2.get('decay_chains').append(decay2process2)
        core_process1.get('decay_chains').append(decay1process1)
        core_process1.get('decay_chains').append(decay1process2)

        my_amplitude1 = diagram_generation.DecayChainAmplitude(core_process1)

        # Second process

        my_leglist = base_objects.MultiLegList()
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': ds,
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))
        my_leglist.append(base_objects.MultiLeg({'ids': [n1.get('pdg_code')]}))

        core_process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                       'model':mymodel,
                                                       'id': 2})

        decay1proc1 = [[n1.get('pdg_code')], 
                      [11], [24]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay1proc1])
        my_leglist[0].set('state', False)
        decay1process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})
        decay1process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        decay2proc1 = [[24], [6], [-5]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc1])
        my_leglist[0].set('state', False)
        decay2process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})
        decay2proc2 = [[24], [6], [-5]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay2proc2])
        my_leglist[0].set('state', False)
        decay2process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})

        decay1process1.get('decay_chains').append(decay2process1)
        decay1process2.get('decay_chains').append(decay2process2)
        core_process2.get('decay_chains').append(decay1process1)
        core_process2.get('decay_chains').append(decay1process2)

        my_amplitude2 = diagram_generation.DecayChainAmplitude(core_process2)

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([my_amplitude1,
                                                             my_amplitude2]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 2)

        self.assertEqual(len(subproc_groups[0].get('matrix_elements')),2)
        self.assertEqual(len(subproc_groups[1].get('matrix_elements')),2)
        
    def test_single_decay_combinations(self):
        """Test combination of single decay of t and t~:
        b b~ > t t~, t > c d d~ and b b~ > t t~, t~ > c~ d d~"""

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

        # A top quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'zero',
                      'texname':'t',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

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
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
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
                      'id': 8,
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
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # FCNC coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             u, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GFCNC'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 13,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             t, \
                                             z]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GFCNC'},
                      'orders':{'QED':1}}))



        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # d d~ > t t~, t > u Z 
        my_leglist = base_objects.MultiLegList()
        my_leglist.append(base_objects.MultiLeg({'ids': [1],
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [-1],
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [6]}))
        my_leglist.append(base_objects.MultiLeg({'ids': [-6]}))

        core_process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                       'model':mymodel})

        decay1proc1 = [[6], [2], [23]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay1proc1])
        my_leglist[0].set('state', False)
        decay1process1 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        core_process1.get('decay_chains').append(decay1process1)
        my_amplitude1 = diagram_generation.DecayChainAmplitude(core_process1)

        # d d~ > t t~, t~ > u~ Z 
        my_leglist = base_objects.MultiLegList()
        my_leglist.append(base_objects.MultiLeg({'ids': [1],
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [-1],
                                                 'state': False}))
        my_leglist.append(base_objects.MultiLeg({'ids': [6]}))
        my_leglist.append(base_objects.MultiLeg({'ids': [-6]}))

        core_process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                       'model':mymodel})

        decay1proc2 = [[-6], [-2], [23]]
        my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id}) for id in decay1proc2])
        my_leglist[0].set('state', False)
        decay1process2 = base_objects.ProcessDefinition({'legs':my_leglist,
                                                        'model':mymodel})

        core_process2.get('decay_chains').append(decay1process2)

        my_amplitude2 = diagram_generation.DecayChainAmplitude(core_process2)

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([my_amplitude1,
                                                             my_amplitude2]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 2)

        self.assertEqual(len(subproc_groups[0].get('matrix_elements')),1)

        me_strings = ["""Process: d d~ > t t~ WEIGHTED<=2
  Decay: t > u z WEIGHTED<=2""",
                      """Process: d d~ > t t~ WEIGHTED<=2
  Decay: t~ > u~ z WEIGHTED<=2"""]
        

        for i, group in enumerate(subproc_groups):
            self.assertEqual(group.get('matrix_elements')[0].\
                                 get('processes')[0].nice_string(),
                             me_strings[i])
            

