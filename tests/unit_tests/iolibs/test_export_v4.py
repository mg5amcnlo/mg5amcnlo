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

from __future__ import absolute_import
import six
StringIO = six
import copy
import fractions
import os 
import sys
from six.moves import range

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import tests.IOTests as IOTests

import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.save_load_object as save_load_object        
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_algebra as color
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.misc as misc
import madgraph.various.process_checks as process_checks
import madgraph.core.color_amp as color_amp
import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.unit_tests.iolibs.test_helas_call_writers as \
                                            test_helas_call_writers

from madgraph import MG5DIR

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')



pjoin = os.path.join


#===============================================================================
# IOImportV4Test
#===============================================================================
class IOExportV4IOTest(IOTests.IOTestManager,
                     test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
    created_files = ['test'
                    ]

    def setUp(self):

        test_file_writers.CheckFileCreate.clean_files
        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)
        self.myfortranmodel.downcase = False

    tearDown = test_file_writers.CheckFileCreate.clean_files
 
    @IOTests.createIOTest() 
    def testIO_export_matrix_element_v4_standalone(self):
        """target: matrix.f
        """
        #Test the result of exporting a matrix element to file

        process_exporter = export_v4.ProcessExporterFortranSA()

        process_exporter.write_matrix_element_v4(\
            writers.FortranWriter(pjoin(self.IOpath, 'matrix.f')),
            self.mymatrixelement,
            self.myfortranmodel)
    
    @IOTests.createIOTest()
    def testIO_export_matrix_element_v4_madevent_group(self):
        """target: amp2lines.txt 
           target: configs.inc
           target: nqcd_list.inc
           target: config_subproc_map.inc
           target: coloramps.inc
           target: symfact.dat
           target: processes.dat
           target: mirrorprocs.inc
           target: matrix1.f
           target: auto_dsig.f
           target: super_auto_dsig.f
           
           """

        # Setup a model

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
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
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

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Set parameters
        external_parameters = [\
            base_objects.ParamCardVariable('zero', 0.,'DUM', 1),
            base_objects.ParamCardVariable('MZ', 91.,'MASS', 23),
            base_objects.ParamCardVariable('WZ', 2.,'DECAY', 23)]
        couplings = [\
            base_objects.ModelVariable('GQQ', '1.', 'complex'),
            base_objects.ModelVariable('GQED', '0.1', 'complex'),
            base_objects.ModelVariable('G', '1.', 'complex'),
            base_objects.ModelVariable('GUZ1', '0.1', 'complex'),
            base_objects.ModelVariable('GUZ2', '0.1', 'complex'),
            base_objects.ModelVariable('GDZ1', '0.05', 'complex'),
            base_objects.ModelVariable('GDZ2', '0.05', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
                    


        procs = [[2,-2,21,21], [2,-2,2,-2], [2,-2,1,-1]]
        amplitudes = diagram_generation.AmplitudeList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            amplitudes.append(my_amplitude)

        # Calculate diagrams for all processes
        amplitudes[1].set('has_mirror_process', True)
        subprocess_groups = group_subprocs.SubProcessGroup.\
                           group_amplitudes(amplitudes, "madevent")
        self.assertEqual(len(subprocess_groups), 2)
        self.assertEqual(subprocess_groups[0].get('name'), 'qq_gg')
        self.assertEqual(subprocess_groups[1].get('name'), 'qq_qq')

        subprocess_group = subprocess_groups[1]
        matrix_elements = subprocess_group.get('matrix_elements')

        maxflows = 0
        for me in matrix_elements:
            maxflows = max(maxflows,
                           len(me.get('color_basis')))
        
        self.assertEqual(maxflows, 2)

        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test amp2 lines
        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_elements[0],
                                          subprocess_group.get('diagram_maps')[0])

        
        open(pjoin(self.IOpath,'amp2lines.txt'),'w').write('\n'.join(amp2_lines))

        # Test configs.inc

        mapconfigs, (s_and_t_channels, nqcd_list) = \
                       exporter.write_configs_file(\
                                writers.FortranWriter(pjoin(self.IOpath,'configs.inc')),
                                subprocess_group,
                                subprocess_group.get('diagrams_for_configs'))

        # Test config_nqcd.inc
        exporter.write_config_nqcd_file(\
            writers.FortranWriter(pjoin(self.IOpath,'nqcd_list.inc')),
            nqcd_list)
    
        # Test config_subproc_map.inc

        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(pjoin(self.IOpath, "config_subproc_map.inc")),
            subprocess_group.get('diagrams_for_configs'))

        #open(pjoin(self.IOpath,"config_subproc_map.inc"),'w').write(goal_confsub)

        # Test coloramps.inc
        
        exporter.write_coloramps_file(\
            writers.FortranWriter(pjoin(self.IOpath,'coloramps.inc')),
            subprocess_group.get('diagrams_for_configs'),
            maxflows,
            matrix_elements)


        # Test find_matrix_elements_for_configs

        self.assertEqual(\
            diagram_symmetry.find_matrix_elements_for_configs(subprocess_group),
            ([], {}))

        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(subprocess_group)

        self.assertEqual(symmetry, [1,1,1,1,1])
        self.assertEqual(perms,
                         [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])
        self.assertEqual(ident_perms,
                         [[0,1,2,3]])

        # Test symfact.dat
        
        exporter.write_symfact_file(\
            writers.FortranWriter(pjoin(self.IOpath,'symfact.dat')),
            symmetry)


        # Test processes.dat

        files.write_to_file(pjoin(self.IOpath,'processes.dat'),
                            exporter.write_processes_file,
                            subprocess_group)


        # Test mirrorprocs.inc

        exporter.write_mirrorprocs(\
            writers.FortranWriter(pjoin(self.IOpath,'mirrorprocs.inc')),
            subprocess_group)

        # Test matrix1.f
        exporter.write_matrix_element_v4(\
            writers.FortranWriter(pjoin(self.IOpath,'matrix1.f')),
            matrix_elements[0],
            helas_call_writers.FortranUFOHelasCallWriter(mymodel),
            "1")


        # Test auto_dsig,f
        exporter.write_auto_dsig_file(\
            writers.FortranWriter(pjoin(self.IOpath, "auto_dsig.f")),
            matrix_elements[0],
            "1")



        # Test super auto_dsig.f
        exporter.write_super_auto_dsig_file(\
            writers.FortranWriter(pjoin(self.IOpath, "super_auto_dsig.f")),
            subprocess_group)
   


#===============================================================================
# IOImportV4Test
#===============================================================================
    @IOTests.createIOTest()
    def testIO_export_matrix_element_v4_madevent_nogroup(self):
        """target: configs.inc
           target: coloramps.inc
           target: symswap.inc
           target: symfact.inc
           target: matrix.f
           target: auto_dsig.f
           """

        # Setup a model

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
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
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

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Set parameters
        external_parameters = [\
            base_objects.ParamCardVariable('zero', 0.,'DUM', 1),
            base_objects.ParamCardVariable('MZ', 91.,'MASS', 23),
            base_objects.ParamCardVariable('WZ', 2.,'DECAY', 23)]
        couplings = [\
            base_objects.ModelVariable('GQQ', '1.', 'complex'),
            base_objects.ModelVariable('GQED', '0.1', 'complex'),
            base_objects.ModelVariable('G', '1.', 'complex'),
            base_objects.ModelVariable('GUZ1', '0.1', 'complex'),
            base_objects.ModelVariable('GUZ2', '0.1', 'complex'),
            base_objects.ModelVariable('GDZ1', '0.05', 'complex'),
            base_objects.ModelVariable('GDZ2', '0.05', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
                    


        procs = [[2,-2,21,21], [2,-2,2,-2], [2,-2,1,-1]]
        amplitudes = diagram_generation.AmplitudeList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            amplitudes.append(my_amplitude)

        # Calculate diagrams for all processes

        multiproc = helas_objects.HelasMultiProcess(amplitudes)
        matrix_elements = multiproc.get('matrix_elements')
        matrix_element = matrix_elements[0]

        maxflows = 0
        for me in matrix_elements:
            maxflows = max(maxflows,
                           len(me.get('color_basis')))
        
        self.assertEqual(maxflows, 2)

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines
        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))', 
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))', 
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))'])
        # Test configs.inc

        mapconfigs, s_and_t_channels = exporter.write_configs_file(\
            writers.FortranWriter(pjoin(self.IOpath,'configs.inc')),
            matrix_element)



        # Test coloramps.inc
        
        exporter.write_coloramps_file(\
            writers.FortranWriter(pjoin(self.IOpath,'coloramps.inc')),
            mapconfigs,
            matrix_element)


        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(matrix_element)

        self.assertEqual(symmetry, [1, 2, -2])
        self.assertEqual(perms,
                         [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 3, 2]])
        self.assertEqual(ident_perms,
                         [[0, 1, 2, 3], [0, 1, 3, 2]])

        # Test symswap.inc
        
        exporter.write_symswap_file(\
            writers.FortranWriter(pjoin(self.IOpath,'symswap.inc')),
            ident_perms)


        # Test symfact.dat
        
        exporter.write_symfact_file(\
            writers.FortranWriter(pjoin(self.IOpath,'symfact.inc')),
            symmetry)


        # Test matrix.f
        exporter.write_matrix_element_v4(\
            writers.FortranWriter(pjoin(self.IOpath,'matrix.f')),
            matrix_element,
            helas_call_writers.FortranUFOHelasCallWriter(mymodel))

        # Test auto_dsig,f
        exporter.write_auto_dsig_file(\
            writers.FortranWriter(pjoin(self.IOpath,"auto_dsig.f")),
            matrix_element)




class ExportV4IOTest(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
    created_files = ['test'
                    ]

    def setUp(self):

        test_file_writers.CheckFileCreate.clean_files
        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)
        self.myfortranmodel.downcase = False

    tearDown = test_file_writers.CheckFileCreate.clean_files
    
     
    def test_coeff_string(self):
        """Test the coeff string for JAMP lines"""

        process_exporter = export_v4.ProcessExporterFortran()

        self.assertEqual(process_exporter.coeff(1,
                                         fractions.Fraction(1),
                                         False, 0), '+')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(1),
                                         False, 0), '-')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(-3),
                                         False, 0), '+3D0*')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(3, 5),
                                         True, -2), '-1D0/15D0*imag1*')

    def test_export_group_decay_chains(self):
        """Test the result of exporting a subprocess group decay chain"""

        # Setup a model

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

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        # Z-Z-q-q~ 4-point vertex

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z,
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFVV1', 'FFVV2'],
                      'couplings':{(0, 0):'ZZQQ',(0, 1):'ZZQQ'},
                      'orders':{'QED':2}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Set parameters
        external_parameters = [\
            base_objects.ParamCardVariable('zero', 0.,'DUM', 1),
            base_objects.ParamCardVariable('MZ', 91.,'MASS', 23),
            base_objects.ParamCardVariable('WZ', 2.,'DECAY', 23)]
        couplings = [\
            base_objects.ModelVariable('GQQ', '1.', 'complex'),
            base_objects.ModelVariable('GQED', '0.1', 'complex'),
            base_objects.ModelVariable('G', '1.', 'complex'),
            base_objects.ModelVariable('GUZ1', '0.1', 'complex'),
            base_objects.ModelVariable('GUZ2', '0.1', 'complex'),
            base_objects.ModelVariable('GDZ1', '0.05', 'complex'),
            base_objects.ModelVariable('GDZ2', '0.05', 'complex'),
            base_objects.ModelVariable('ZZQQ', '0.01', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
                    
        procs = [[2,-2,23,23], [1,-1,23,23]]
        decays = [[23,1,-1,21], [23,2,-2]]
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
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            my_amplitude.set('has_mirror_process', True)
            coreamplitudes.append(my_amplitude)

        for proc in decays:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
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

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        # Check number of groups
        self.assertEqual(len(subproc_groups), 3)
        self.assertEqual([g.get('name') for g in subproc_groups],
                         ['qq_zz_z_qqg_z_qqg',
                          'qq_zz_z_qqg_z_qq',
                          'qq_zz_z_qq_z_qq'])

        subprocess_group = subproc_groups[0]
        matrix_elements = subprocess_group.get('matrix_elements')

        # Exporter
        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test amp2 lines
        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_elements[1],
                                          subprocess_group.get('diagram_maps')[1])
        #print '\n'.join(amp2_lines)

        self.assertEqual('\n'.join(amp2_lines),
"""AMP2(5)=AMP2(5)+AMP(5)*dconjg(AMP(5))
AMP2(6)=AMP2(6)+AMP(6)*dconjg(AMP(6))
AMP2(7)=AMP2(7)+AMP(7)*dconjg(AMP(7))
AMP2(8)=AMP2(8)+AMP(8)*dconjg(AMP(8))
AMP2(9)=AMP2(9)+AMP(9)*dconjg(AMP(9))
AMP2(10)=AMP2(10)+AMP(10)*dconjg(AMP(10))
AMP2(11)=AMP2(11)+AMP(11)*dconjg(AMP(11))
AMP2(12)=AMP2(12)+AMP(12)*dconjg(AMP(12))""")
        
        # Test configs.inc

        exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group,
            subprocess_group.get('diagrams_for_configs'))

        #print( open(self.give_pos('test')).read())
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/5,3/
      DATA (SPROP(I,-1,1),I=1,2)/1,1/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA (SPROP(I,-2,1),I=1,2)/23,23/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/8,6/
      DATA (SPROP(I,-3,1),I=1,2)/1,1/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/7,-3/
      DATA (SPROP(I,-4,1),I=1,2)/23,23/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/1,-2/
      DATA TPRID(-5,1)/2/
      DATA (SPROP(I,-5,1),I=1,2)/0,0/
      DATA (IFOREST(I,-6,1),I=1,2)/-5,-4/
      DATA TPRID(-6,1)/2/
      DATA (SPROP(I,-6,1),I=1,2)/0,0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,3/
      DATA (SPROP(I,-1,2),I=1,2)/1,1/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/4,-1/
      DATA (SPROP(I,-2,2),I=1,2)/23,23/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/8,7/
      DATA (SPROP(I,-3,2),I=1,2)/-1,-1/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/-3,6/
      DATA (SPROP(I,-4,2),I=1,2)/23,23/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/1,-2/
      DATA TPRID(-5,2)/2/
      DATA (SPROP(I,-5,2),I=1,2)/0,0/
      DATA (IFOREST(I,-6,2),I=1,2)/-5,-4/
      DATA TPRID(-6,2)/2/
      DATA (SPROP(I,-6,2),I=1,2)/0,0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/5,4/
      DATA (SPROP(I,-1,3),I=1,2)/-1,-1/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
      DATA (SPROP(I,-2,3),I=1,2)/23,23/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/8,6/
      DATA (SPROP(I,-3,3),I=1,2)/1,1/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/7,-3/
      DATA (SPROP(I,-4,3),I=1,2)/23,23/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/1,-2/
      DATA TPRID(-5,3)/2/
      DATA (SPROP(I,-5,3),I=1,2)/0,0/
      DATA (IFOREST(I,-6,3),I=1,2)/-5,-4/
      DATA TPRID(-6,3)/2/
      DATA (SPROP(I,-6,3),I=1,2)/0,0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/5,4/
      DATA (SPROP(I,-1,4),I=1,2)/-1,-1/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,3/
      DATA (SPROP(I,-2,4),I=1,2)/23,23/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/8,7/
      DATA (SPROP(I,-3,4),I=1,2)/-1,-1/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,6/
      DATA (SPROP(I,-4,4),I=1,2)/23,23/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/1,-2/
      DATA TPRID(-5,4)/2/
      DATA (SPROP(I,-5,4),I=1,2)/0,0/
      DATA (IFOREST(I,-6,4),I=1,2)/-5,-4/
      DATA TPRID(-6,4)/2/
      DATA (SPROP(I,-6,4),I=1,2)/0,0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/8,6/
      DATA (SPROP(I,-1,5),I=1,2)/1,1/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/7,-1/
      DATA (SPROP(I,-2,5),I=1,2)/23,23/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/5,3/
      DATA (SPROP(I,-3,5),I=1,2)/1,1/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/4,-3/
      DATA (SPROP(I,-4,5),I=1,2)/23,23/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/1,-2/
      DATA TPRID(-5,5)/2/
      DATA (SPROP(I,-5,5),I=1,2)/0,0/
      DATA (IFOREST(I,-6,5),I=1,2)/-5,-4/
      DATA TPRID(-6,5)/2/
      DATA (SPROP(I,-6,5),I=1,2)/0,0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/8,7/
      DATA (SPROP(I,-1,6),I=1,2)/-1,-1/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/-1,6/
      DATA (SPROP(I,-2,6),I=1,2)/23,23/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/5,3/
      DATA (SPROP(I,-3,6),I=1,2)/1,1/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/4,-3/
      DATA (SPROP(I,-4,6),I=1,2)/23,23/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/1,-2/
      DATA TPRID(-5,6)/2/
      DATA (SPROP(I,-5,6),I=1,2)/0,0/
      DATA (IFOREST(I,-6,6),I=1,2)/-5,-4/
      DATA TPRID(-6,6)/2/
      DATA (SPROP(I,-6,6),I=1,2)/0,0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA TSTRATEGY(7)/2/
      DATA (IFOREST(I,-1,7),I=1,2)/8,6/
      DATA (SPROP(I,-1,7),I=1,2)/1,1/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/7,-1/
      DATA (SPROP(I,-2,7),I=1,2)/23,23/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/5,4/
      DATA (SPROP(I,-3,7),I=1,2)/-1,-1/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/-3,3/
      DATA (SPROP(I,-4,7),I=1,2)/23,23/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/1,-2/
      DATA TPRID(-5,7)/2/
      DATA (SPROP(I,-5,7),I=1,2)/0,0/
      DATA (IFOREST(I,-6,7),I=1,2)/-5,-4/
      DATA TPRID(-6,7)/2/
      DATA (SPROP(I,-6,7),I=1,2)/0,0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA TSTRATEGY(8)/2/
      DATA (IFOREST(I,-1,8),I=1,2)/8,7/
      DATA (SPROP(I,-1,8),I=1,2)/-1,-1/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,6/
      DATA (SPROP(I,-2,8),I=1,2)/23,23/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/5,4/
      DATA (SPROP(I,-3,8),I=1,2)/-1,-1/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,3/
      DATA (SPROP(I,-4,8),I=1,2)/23,23/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/1,-2/
      DATA TPRID(-5,8)/2/
      DATA (SPROP(I,-5,8),I=1,2)/0,0/
      DATA (IFOREST(I,-6,8),I=1,2)/-5,-4/
      DATA TPRID(-6,8)/2/
      DATA (SPROP(I,-6,8),I=1,2)/0,0/
C     Number of configs
      DATA MAPCONFIG(0)/8/
""")

        # Test config_subproc_map.inc

        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'))

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      DATA (CONFSUB(I,1),I=1,2)/1,5/
      DATA (CONFSUB(I,2),I=1,2)/2,6/
      DATA (CONFSUB(I,3),I=1,2)/3,7/
      DATA (CONFSUB(I,4),I=1,2)/4,8/
      DATA (CONFSUB(I,5),I=1,2)/5,9/
      DATA (CONFSUB(I,6),I=1,2)/6,10/
      DATA (CONFSUB(I,7),I=1,2)/7,11/
      DATA (CONFSUB(I,8),I=1,2)/8,12/
""")

        # Test symfact.dat
        
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(subprocess_group)

        exporter.write_symfact_file(\
            writers.FortranWriter(self.give_pos('test')),
            symmetry)
        
        #print open(self.give_pos('test')).read()
        goal_symfact_dat = """ 1   1
 2  -1
 3  -1
 4  -1
 5  -1
 6  -1
 7  -1
 8  -1
"""
        self.assertFileContains('test', goal_symfact_dat)

        # Test symperms.inc
        
        exporter.write_symperms_file(\
            writers.FortranWriter(self.give_pos('test')),
            perms)
        goal_symperms_dat = """      DATA (PERMS(I,1),I=1,NEXTERNAL)/1,2,3,4,5,6,7,8/
      DATA (PERMS(I,2),I=1,NEXTERNAL)/1,2,3,4,5,7,6,8/
      DATA (PERMS(I,3),I=1,NEXTERNAL)/1,2,4,3,5,6,7,8/
      DATA (PERMS(I,4),I=1,NEXTERNAL)/1,2,4,3,5,7,6,8/
      DATA (PERMS(I,5),I=1,NEXTERNAL)/1,2,6,7,8,3,4,5/
      DATA (PERMS(I,6),I=1,NEXTERNAL)/1,2,6,7,8,4,3,5/
      DATA (PERMS(I,7),I=1,NEXTERNAL)/1,2,7,6,8,3,4,5/
      DATA (PERMS(I,8),I=1,NEXTERNAL)/1,2,7,6,8,4,3,5/
"""
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_symperms_dat)

        # Test processes.dat

        files.write_to_file(self.give_pos('test'),
                            exporter.write_processes_file,
                            subprocess_group)

        #print open(self.give_pos('test')).read()

        goal_processes = """1       u u~ > d d~ g d d~ g
mirror  u~ u > d d~ g d d~ g
2       d d~ > d d~ g d d~ g
mirror  d~ d > d d~ g d d~ g"""
        
        self.assertFileContains('test', goal_processes)

    def test_export_group_multidiagram_decay_chains(self):
        """Test export group_amplitudes for uu~>g>gogo, go>qqn1."""

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

        # A gluino
        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':8,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        go = mypartlist[-1]

        # A u squark
        mypartlist.append(base_objects.Particle({'name':'ul',
                      'antiname':'ul~',
                      'spin':1,
                      'color':3,
                      'mass':'MUL',
                      'width':'WUL',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':1000002,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        ul = mypartlist[-1]
        antiul = copy.copy(ul)
        antiul.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'dl',
                      'antiname':'dl~',
                      'spin':1,
                      'color':3,
                      'mass':'MDL',
                      'width':'WDL',
                      'texname':'d',
                      'antitexname':'\bar d',
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
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[-1]

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        g = mymodel.get_particle(21)
        d = mymodel.get_particle(1)
        antid = mymodel.get_particle(-1)
        u = mymodel.get_particle(2)
        antiu = mymodel.get_particle(-2)
        
        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Gluon couplings to gluino
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

        # Gluino couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             u, \
                                             antiul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             antiu, \
                                             ul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 13,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             d, \
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 14,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             antid, \
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Neutralino couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 15,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             u, \
                                             antiul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 16,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             antiu, \
                                             ul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 17,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             d, \
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 18,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             antid, \
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        mymodel.set('interactions', myinterlist)

        procs = [[2,-2,1000021,1000021], [1,-1,1000021,1000021]]
        decays = [[1000021,1,-1,1000022],[1000021,2,-2,1000022]]
        coreamplitudes = []
        coreamplitude2 = diagram_generation.AmplitudeList()
        decayamplitudes = diagram_generation.AmplitudeList()
        decayprocs = base_objects.ProcessList()
        proc_diags = [1,1]
        decay_diags = [2,2]
        
        for iproc, proc in enumerate(procs):
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'required_s_channels':[[21]]})
            my_amplitude = diagram_generation.Amplitude(my_process)
            coreamplitudes.append(diagram_generation.AmplitudeList([my_amplitude]))
            self.assertEqual(len(my_amplitude.get('diagrams')), proc_diags[iproc])

        for iproc, proc in enumerate(decays):
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'is_decay_chain': True})
                                               #'forbidden_particles':[2000001,2000002]})
            my_amplitude = diagram_generation.Amplitude(my_process)
            decayamplitudes.append(my_amplitude)
            decayprocs.append(my_process)
            self.assertEqual(len(my_amplitude.get('diagrams')), decay_diags[iproc])

        decays1 = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': copy.copy(decayamplitudes)})])
        decays2 = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': copy.copy(decayamplitudes)})])

        decay_chains1 = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes[0],
            'decay_chains': decays1})

        decay_chains2 = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes[1],
            'decay_chains': decays2})

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([decay_chains1,
                                                             decay_chains2]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 1)

        group_name = 'qq_gogo_go_qqn1_go_qqn1'
        me_len = 3

        subprocess_group = subproc_groups[0]
        self.assertEqual(subprocess_group.get('name'),
                         group_name)
        self.assertEqual(len(subprocess_group.get('matrix_elements')), me_len)

        self.assertEqual(len(subprocess_group.get('matrix_elements')[0].get('processes')), 2)

        # Exporter
        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test config_subproc_map.inc
        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'))

        self.assertFileContains('test',
"""      DATA (CONFSUB(I,1),I=1,3)/1,0,0/
      DATA (CONFSUB(I,2),I=1,3)/2,0,0/
      DATA (CONFSUB(I,3),I=1,3)/3,0,0/
      DATA (CONFSUB(I,4),I=1,3)/4,0,0/
      DATA (CONFSUB(I,5),I=1,3)/0,1,0/
      DATA (CONFSUB(I,6),I=1,3)/0,2,0/
      DATA (CONFSUB(I,7),I=1,3)/0,3,0/
      DATA (CONFSUB(I,8),I=1,3)/0,4,0/
      DATA (CONFSUB(I,9),I=1,3)/0,0,1/
      DATA (CONFSUB(I,10),I=1,3)/0,0,2/
      DATA (CONFSUB(I,11),I=1,3)/0,0,3/
      DATA (CONFSUB(I,12),I=1,3)/0,0,4/
""")

        # Test configs.inc

        exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group,
            subprocess_group.get('diagrams_for_configs'))

        #print open(self.give_pos('test')).read()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/8,6/
      DATA (SPROP(I,-1,1),I=1,3)/1000001,0,0/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/7,-1/
      DATA (SPROP(I,-2,1),I=1,3)/1000021,0,0/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,3/
      DATA (SPROP(I,-3,1),I=1,3)/1000001,0,0/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/4,-3/
      DATA (SPROP(I,-4,1),I=1,3)/1000021,0,0/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,1),I=1,3)/21,0,0/
      DATA TPRID(-5,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/8,7/
      DATA (SPROP(I,-1,2),I=1,3)/-1000001,0,0/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,6/
      DATA (SPROP(I,-2,2),I=1,3)/1000021,0,0/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/5,3/
      DATA (SPROP(I,-3,2),I=1,3)/1000001,0,0/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/4,-3/
      DATA (SPROP(I,-4,2),I=1,3)/1000021,0,0/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,2),I=1,3)/21,0,0/
      DATA TPRID(-5,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/8,6/
      DATA (SPROP(I,-1,3),I=1,3)/1000001,0,0/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/7,-1/
      DATA (SPROP(I,-2,3),I=1,3)/1000021,0,0/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/5,4/
      DATA (SPROP(I,-3,3),I=1,3)/-1000001,0,0/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,3/
      DATA (SPROP(I,-4,3),I=1,3)/1000021,0,0/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,3),I=1,3)/21,0,0/
      DATA TPRID(-5,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/8,7/
      DATA (SPROP(I,-1,4),I=1,3)/-1000001,0,0/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,6/
      DATA (SPROP(I,-2,4),I=1,3)/1000021,0,0/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/5,4/
      DATA (SPROP(I,-3,4),I=1,3)/-1000001,0,0/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,3/
      DATA (SPROP(I,-4,4),I=1,3)/1000021,0,0/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,4),I=1,3)/21,0,0/
      DATA TPRID(-5,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/8,6/
      DATA (SPROP(I,-1,5),I=1,3)/0,1000002,0/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/7,-1/
      DATA (SPROP(I,-2,5),I=1,3)/0,1000021,0/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/5,3/
      DATA (SPROP(I,-3,5),I=1,3)/0,1000001,0/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/4,-3/
      DATA (SPROP(I,-4,5),I=1,3)/0,1000021,0/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,5),I=1,3)/0,21,0/
      DATA TPRID(-5,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/8,7/
      DATA (SPROP(I,-1,6),I=1,3)/0,-1000002,0/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/-1,6/
      DATA (SPROP(I,-2,6),I=1,3)/0,1000021,0/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/5,3/
      DATA (SPROP(I,-3,6),I=1,3)/0,1000001,0/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/4,-3/
      DATA (SPROP(I,-4,6),I=1,3)/0,1000021,0/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,6),I=1,3)/0,21,0/
      DATA TPRID(-5,6)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA TSTRATEGY(7)/2/
      DATA (IFOREST(I,-1,7),I=1,2)/8,6/
      DATA (SPROP(I,-1,7),I=1,3)/0,1000002,0/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/7,-1/
      DATA (SPROP(I,-2,7),I=1,3)/0,1000021,0/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/5,4/
      DATA (SPROP(I,-3,7),I=1,3)/0,-1000001,0/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/-3,3/
      DATA (SPROP(I,-4,7),I=1,3)/0,1000021,0/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,7),I=1,3)/0,21,0/
      DATA TPRID(-5,7)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA TSTRATEGY(8)/2/
      DATA (IFOREST(I,-1,8),I=1,2)/8,7/
      DATA (SPROP(I,-1,8),I=1,3)/0,-1000002,0/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,6/
      DATA (SPROP(I,-2,8),I=1,3)/0,1000021,0/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/5,4/
      DATA (SPROP(I,-3,8),I=1,3)/0,-1000001,0/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,3/
      DATA (SPROP(I,-4,8),I=1,3)/0,1000021,0/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,8),I=1,3)/0,21,0/
      DATA TPRID(-5,8)/0/
C     Diagram 9
      DATA MAPCONFIG(9)/9/
      DATA TSTRATEGY(9)/2/
      DATA (IFOREST(I,-1,9),I=1,2)/8,6/
      DATA (SPROP(I,-1,9),I=1,3)/0,0,1000002/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/7,-1/
      DATA (SPROP(I,-2,9),I=1,3)/0,0,1000021/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/5,3/
      DATA (SPROP(I,-3,9),I=1,3)/0,0,1000002/
      DATA TPRID(-3,9)/0/
      DATA (IFOREST(I,-4,9),I=1,2)/4,-3/
      DATA (SPROP(I,-4,9),I=1,3)/0,0,1000021/
      DATA TPRID(-4,9)/0/
      DATA (IFOREST(I,-5,9),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,9),I=1,3)/0,0,21/
      DATA TPRID(-5,9)/0/
C     Diagram 10
      DATA MAPCONFIG(10)/10/
      DATA TSTRATEGY(10)/2/
      DATA (IFOREST(I,-1,10),I=1,2)/8,7/
      DATA (SPROP(I,-1,10),I=1,3)/0,0,-1000002/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/-1,6/
      DATA (SPROP(I,-2,10),I=1,3)/0,0,1000021/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/5,3/
      DATA (SPROP(I,-3,10),I=1,3)/0,0,1000002/
      DATA TPRID(-3,10)/0/
      DATA (IFOREST(I,-4,10),I=1,2)/4,-3/
      DATA (SPROP(I,-4,10),I=1,3)/0,0,1000021/
      DATA TPRID(-4,10)/0/
      DATA (IFOREST(I,-5,10),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,10),I=1,3)/0,0,21/
      DATA TPRID(-5,10)/0/
C     Diagram 11
      DATA MAPCONFIG(11)/11/
      DATA TSTRATEGY(11)/2/
      DATA (IFOREST(I,-1,11),I=1,2)/8,6/
      DATA (SPROP(I,-1,11),I=1,3)/0,0,1000002/
      DATA TPRID(-1,11)/0/
      DATA (IFOREST(I,-2,11),I=1,2)/7,-1/
      DATA (SPROP(I,-2,11),I=1,3)/0,0,1000021/
      DATA TPRID(-2,11)/0/
      DATA (IFOREST(I,-3,11),I=1,2)/5,4/
      DATA (SPROP(I,-3,11),I=1,3)/0,0,-1000002/
      DATA TPRID(-3,11)/0/
      DATA (IFOREST(I,-4,11),I=1,2)/-3,3/
      DATA (SPROP(I,-4,11),I=1,3)/0,0,1000021/
      DATA TPRID(-4,11)/0/
      DATA (IFOREST(I,-5,11),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,11),I=1,3)/0,0,21/
      DATA TPRID(-5,11)/0/
C     Diagram 12
      DATA MAPCONFIG(12)/12/
      DATA TSTRATEGY(12)/2/
      DATA (IFOREST(I,-1,12),I=1,2)/8,7/
      DATA (SPROP(I,-1,12),I=1,3)/0,0,-1000002/
      DATA TPRID(-1,12)/0/
      DATA (IFOREST(I,-2,12),I=1,2)/-1,6/
      DATA (SPROP(I,-2,12),I=1,3)/0,0,1000021/
      DATA TPRID(-2,12)/0/
      DATA (IFOREST(I,-3,12),I=1,2)/5,4/
      DATA (SPROP(I,-3,12),I=1,3)/0,0,-1000002/
      DATA TPRID(-3,12)/0/
      DATA (IFOREST(I,-4,12),I=1,2)/-3,3/
      DATA (SPROP(I,-4,12),I=1,3)/0,0,1000021/
      DATA TPRID(-4,12)/0/
      DATA (IFOREST(I,-5,12),I=1,2)/-2,-4/
      DATA (SPROP(I,-5,12),I=1,3)/0,0,21/
      DATA TPRID(-5,12)/0/
C     Number of configs
      DATA MAPCONFIG(0)/12/
""")


    def test_replace_make_opt_f_compiler(self):
        """check that the compiler in the Template is the default one for release"""

#        if os.path.exists(pjoin(MG5DIR, 'bin', 'create_release.py')):
        LO_text = open(pjoin(MG5DIR, 'Template','LO','Source','make_opts')).read()
        self.assertIn('F2PY=$(DEFAULT_F2PY_COMPILER)', LO_text)
        self.assertIn('FC=$(DEFAULT_F_COMPILER)', LO_text)
        if os.path.exists(pjoin(MG5DIR, 'bin', 'create_release.py')):
#            self.assertTrue('DEFAULT_F2PY_COMPILER=f2py' in LO_text)
            self.assertIn('DEFAULT_F_COMPILER=gfortran', LO_text)
#            NLO_text = open(pjoin(MG5DIR, 'Template','NLO','Source','make_opts.inc')).read()
#            self.assertTrue('DEFAULT_F2PY_COMPILER=f2py' in NLO_text)
#            self.assertTrue('FC=gfortran' in NLO_text)            
            
            
            
            
            
        
        
        
        
        
        
        
        


    def test_different_order_process_symmetry(self):
        """Test a process where different diagrams have different order props"""

        # Setup a model

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

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        # Coupling of Z to e+ e-
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             z]),
                      'color': [color.ColorString([])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GEZ1'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        proc = [21,2,23,23,2]
        decay = [23,11,-11]

        # Define the multiprocess
        my_leglist = base_objects.LegList([\
            base_objects.Leg({'id': id, 'state': True}) for id in proc])

        my_leglist[0].set('state', False)
        my_leglist[1].set('state', False)

        my_decaylegs = base_objects.LegList([\
            base_objects.Leg({'id': id, 'state': True}) for id in decay])

        my_decaylegs[0].set('state', False)
        my_process = base_objects.Process({'legs':my_leglist,
                                           'model':mymodel})
        my_decay_proc = base_objects.Process({'legs':my_decaylegs,
                                              'model':mymodel,
                                              'is_decay_chain': True})
        my_process.set('decay_chains', 
                       base_objects.ProcessList([my_decay_proc]))

        my_decay = diagram_generation.DecayChainAmplitude(my_process)
        helas_decay = helas_objects.HelasDecayChainProcess(my_decay)
        matrix_element = helas_decay.combine_decay_chain_processes()[0]

        # Exporter
        exporter = export_v4.ProcessExporterFortranME()
        exporter.opt['t_strategy'] = 2 # keep old ordering for this test

        # Test configs.inc

        mapconfigs, s_and_t_channels = exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            matrix_element)

        #print( open(self.give_pos('test')).read())
        self.assertFileContains('test',"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/23/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/7,-1/
      DATA (SPROP(I,-2,1),I=1,1)/2/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/6,5/
      DATA (SPROP(I,-3,1),I=1,1)/23/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/-3,-2/
      DATA (SPROP(I,-4,1),I=1,1)/2/
      DATA TPRID(-4,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/6,5/
      DATA (SPROP(I,-1,2),I=1,1)/23/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/7,-1/
      DATA (SPROP(I,-2,2),I=1,1)/2/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/4,3/
      DATA (SPROP(I,-3,2),I=1,1)/23/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/-2,-3/
      DATA (SPROP(I,-4,2),I=1,1)/2/
      DATA TPRID(-4,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/6,5/
      DATA (SPROP(I,-1,3),I=1,1)/23/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/4,3/
      DATA (SPROP(I,-2,3),I=1,1)/23/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/1,7/
      DATA TPRID(-3,3)/2/
      DATA (SPROP(I,-3,3),I=1,1)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,-1/
      DATA TPRID(-4,3)/2/
      DATA (SPROP(I,-4,3),I=1,1)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/-4,-2/
      DATA TPRID(-5,3)/2/
      DATA (SPROP(I,-5,3),I=1,1)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/4,3/
      DATA (SPROP(I,-1,4),I=1,1)/23/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/6,5/
      DATA (SPROP(I,-2,4),I=1,1)/23/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/1,7/
      DATA TPRID(-3,4)/2/
      DATA (SPROP(I,-3,4),I=1,1)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,-1/
      DATA TPRID(-4,4)/2/
      DATA (SPROP(I,-4,4),I=1,1)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/-4,-2/
      DATA TPRID(-5,4)/2/
      DATA (SPROP(I,-5,4),I=1,1)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/6,5/
      DATA (SPROP(I,-1,5),I=1,1)/23/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/7,-1/
      DATA (SPROP(I,-2,5),I=1,1)/2/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/4,3/
      DATA (SPROP(I,-3,5),I=1,1)/23/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/1,-2/
      DATA TPRID(-4,5)/2/
      DATA (SPROP(I,-4,5),I=1,1)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/-4,-3/
      DATA TPRID(-5,5)/2/
      DATA (SPROP(I,-5,5),I=1,1)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/4,3/
      DATA (SPROP(I,-1,6),I=1,1)/23/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/7,-1/
      DATA (SPROP(I,-2,6),I=1,1)/2/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/6,5/
      DATA (SPROP(I,-3,6),I=1,1)/23/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/1,-2/
      DATA TPRID(-4,6)/2/
      DATA (SPROP(I,-4,6),I=1,1)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/-4,-3/
      DATA TPRID(-5,6)/2/
      DATA (SPROP(I,-5,6),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/6/
""")

        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(matrix_element)

        # Test symfact.dat
        
        exporter.write_symfact_file(\
            writers.FortranWriter(self.give_pos('test')),
            symmetry)
        goal_symfact_dat = """ 1   2
 2  -1
 3   2
 4  -3
 5   2
 6  -5
"""
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_symfact_dat)

        # Test symperms.inc
        
        exporter.write_symperms_file(\
            writers.FortranWriter(self.give_pos('test')),
            perms)
        goal_symperms_dat = """      DATA (PERMS(I,1),I=1,NEXTERNAL)/1,2,3,4,5,6,7/
      DATA (PERMS(I,2),I=1,NEXTERNAL)/1,2,5,6,3,4,7/
      DATA (PERMS(I,3),I=1,NEXTERNAL)/1,2,3,4,5,6,7/
      DATA (PERMS(I,4),I=1,NEXTERNAL)/1,2,5,6,3,4,7/
      DATA (PERMS(I,5),I=1,NEXTERNAL)/1,2,3,4,5,6,7/
      DATA (PERMS(I,6),I=1,NEXTERNAL)/1,2,5,6,3,4,7/
"""
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_symperms_dat)

        # Test symswap.inc
        
        exporter.write_symswap_file(\
            writers.FortranWriter(self.give_pos('test')),
            ident_perms)
        goal_symswap_dat = """      DATA (ISYM(I,1),I=1,NEXTERNAL)/1,2,3,4,5,6,7/
      DATA (ISYM(I,2),I=1,NEXTERNAL)/1,2,5,6,3,4,7/
      DATA NSYM/2/
"""
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_symswap_dat)

#===============================================================================
# FullHelasOutputIOTest
#===============================================================================
class FullHelasOutputIOTest(IOTests.IOTestManager,
                            test_helas_call_writers.HelasModelTestSetup):

    def setUp(self):
        """Generate a simple model for the IOTests to use"""
        
        if hasattr(self,'IOTestModel'):
            return

        self.IOTestModel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Gluon self-couplings
        my_color_string = color.ColorString([color.f(0, 1, 2)])
        my_color_string.is_imaginary = True
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [my_color_string],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        self.IOTestModel.set('particles', mypartlist)
        self.IOTestModel.set('interactions', myinterlist)
        self.IOTestModel.set('couplings', ['QCD','QED'])        
        self.IOTestModel.set('order_hierarchy', {'QCD':1,'QED':2})

    @IOTests.createIOTest(groupName='SquaredOrder_IOTest')
    def testIO_sqso_uux_uuxuuxx(self):
        r""" target: [matrix(.*)\.f]
        """
    
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,'state':False}))
        myleglist.append(base_objects.Leg({'id':2,'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,'state':True}))
        myleglist.append(base_objects.Leg({'id':2,'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,'state':True}))

        fortran_model = helas_call_writers.FortranHelasCallWriter(self.IOTestModel)
        process_exporter = export_v4.ProcessExporterFortranSA()
        
        SO_tests = [({},{},{},[],'NoSQSO'),
                    ({},{'QCD':6},{'QCD':'=='},['QCD'],'QCDsq_eq_6'),
                    ({},{'QED':4},{'QED':'>'},['QCD'],'QEDsq_gt_4'),
                    ({},{'QED':6},{'QED':'<='},['QCD','QED'],'QCDsq_le_6'),
                    ({'QED':2},{'WEIGHTED':14,'QCD':2}, 
                     {'WEIGHTED':'<=','QCD':'>'},['WEIGHTED','QCD'],
                                    'ampOrderQED2_eq_2_WGTsq_le_14_QCDsq_gt_4')]
        
        for orders, sq_orders, sq_orders_type, split_orders, name in SO_tests:
            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.IOTestModel,
                                           'orders': orders,
                                           'squared_orders': sq_orders,
                                           'sqorders_types':sq_orders_type,
                                           'split_orders':split_orders})

            myamplitude = diagram_generation.Amplitude({'process': myproc})
            matrix_element = helas_objects.HelasMatrixElement(myamplitude)
            writer = writers.FortranWriter(pjoin(self.IOpath,'matrix_%s.f'%name))
            process_exporter.write_matrix_element_v4(
                                           writer,matrix_element,fortran_model)
        
#===============================================================================
# FullHelasOutputTest
#===============================================================================
class FullHelasOutputTest(test_helas_call_writers.HelasModelTestSetup,
                          test_file_writers.CheckFileCreate):
    """Test class for the output of various processes. In practice,
    tests both HelasObject generation and MG4 output."""

    created_files = ['leshouche'
                    ]


    def setUp(self):
        
        super(FullHelasOutputTest, self).setUp()
        helas_call_writers.HelasCallWriter.customize_argument_for_all_other_helas_object =\
            staticmethod(helas_call_writers.HelasCallWriter.default_customize_argument_for_all_other_helas_object)
        
    tearDown = test_file_writers.CheckFileCreate.clean_files



    def test_generate_helas_diagrams_ea_ae(self):
        """Testing the helas diagram generation e- a > a e-
        """

        # Test e- a > a e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(\
            helas_call_writers.FortranHelasCallWriter(self.mymodel).\
            get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL FVIXXX(W(1,1),W(1,2),MGVX12,me,zero,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,5),W(1,4),W(1,3),MGVX12,AMP(1))
CALL FVIXXX(W(1,1),W(1,3),MGVX12,me,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,5),W(1,4),W(1,2),MGVX12,AMP(2))""")

    def test_generate_helas_diagrams_uux_gepem_no_optimization(self):
        """Testing the helas diagram generation u u~ > g e+ e-
        """

        # Test u u~ > g e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            0)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual( helas_call_writers.FortranHelasCallWriter(self.mymodel).\
                                   get_matrix_element_calls(matrix_element),
                         """CALL IXXXXX(P(0,1),mu,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVIXXX(W(1,1),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,6),W(1,2),W(1,7),MGVX15,AMP(1))
CALL IXXXXX(P(0,1),mu,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVOXXX(W(1,2),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,1),W(1,6),W(1,7),MGVX15,AMP(2))""".split('\n'))

    def test_generate_helas_diagrams_uux_uuxuux(self):
        """Test calls for u u~ > u u~ u u~ and MadEvent files"""

        # Set up local model

        mybasemodel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Gluon self-couplings
        my_color_string = color.ColorString([color.f(0, 1, 2)])
        my_color_string.is_imaginary = True
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [my_color_string],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        # Test Helas calls

        fortran_model = helas_call_writers.FortranHelasCallWriter(mybasemodel)
        self.assertEqual(fortran_model.\
                                   get_matrix_element_calls(matrix_element),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL IXXXXX(P(0,6),zero,NHEL(6),-1*IC(6),W(1,6))
CALL JIOXXX(W(1,1),W(1,2),GG,zero,zero,W(1,7))
CALL JIOXXX(W(1,4),W(1,3),GG,zero,zero,W(1,8))
CALL FVOXXX(W(1,5),W(1,7),GG,zero,zero,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,6),W(1,9),W(1,8),GG,AMP(1))
CALL FVIXXX(W(1,6),W(1,7),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,10),W(1,5),W(1,8),GG,AMP(2))
CALL JIOXXX(W(1,6),W(1,5),GG,zero,zero,W(1,11))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,7),W(1,8),W(1,11),GG,AMP(3))
CALL JIOXXX(W(1,6),W(1,3),GG,zero,zero,W(1,12))
CALL FVIXXX(W(1,4),W(1,7),GG,zero,zero,W(1,13))
# Amplitude(s) for diagram number 4
CALL IOVXXX(W(1,13),W(1,5),W(1,12),GG,AMP(4))
# Amplitude(s) for diagram number 5
CALL IOVXXX(W(1,4),W(1,9),W(1,12),GG,AMP(5))
CALL JIOXXX(W(1,4),W(1,5),GG,zero,zero,W(1,9))
# Amplitude(s) for diagram number 6
CALL VVVXXX(W(1,7),W(1,12),W(1,9),GG,AMP(6))
CALL FVOXXX(W(1,3),W(1,7),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 7
CALL IOVXXX(W(1,6),W(1,14),W(1,9),GG,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOVXXX(W(1,10),W(1,3),W(1,9),GG,AMP(8))
# Amplitude(s) for diagram number 9
CALL IOVXXX(W(1,4),W(1,14),W(1,11),GG,AMP(9))
# Amplitude(s) for diagram number 10
CALL IOVXXX(W(1,13),W(1,3),W(1,11),GG,AMP(10))
CALL JIOXXX(W(1,1),W(1,3),GG,zero,zero,W(1,13))
CALL JIOXXX(W(1,4),W(1,2),GG,zero,zero,W(1,14))
CALL FVOXXX(W(1,5),W(1,13),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 11
CALL IOVXXX(W(1,6),W(1,10),W(1,14),GG,AMP(11))
CALL FVIXXX(W(1,6),W(1,13),GG,zero,zero,W(1,7))
# Amplitude(s) for diagram number 12
CALL IOVXXX(W(1,7),W(1,5),W(1,14),GG,AMP(12))
# Amplitude(s) for diagram number 13
CALL VVVXXX(W(1,13),W(1,14),W(1,11),GG,AMP(13))
CALL JIOXXX(W(1,6),W(1,2),GG,zero,zero,W(1,15))
CALL FVIXXX(W(1,4),W(1,13),GG,zero,zero,W(1,16))
# Amplitude(s) for diagram number 14
CALL IOVXXX(W(1,16),W(1,5),W(1,15),GG,AMP(14))
# Amplitude(s) for diagram number 15
CALL IOVXXX(W(1,4),W(1,10),W(1,15),GG,AMP(15))
# Amplitude(s) for diagram number 16
CALL VVVXXX(W(1,13),W(1,15),W(1,9),GG,AMP(16))
CALL FVOXXX(W(1,2),W(1,13),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 17
CALL IOVXXX(W(1,6),W(1,10),W(1,9),GG,AMP(17))
# Amplitude(s) for diagram number 18
CALL IOVXXX(W(1,7),W(1,2),W(1,9),GG,AMP(18))
# Amplitude(s) for diagram number 19
CALL IOVXXX(W(1,4),W(1,10),W(1,11),GG,AMP(19))
# Amplitude(s) for diagram number 20
CALL IOVXXX(W(1,16),W(1,2),W(1,11),GG,AMP(20))
CALL JIOXXX(W(1,1),W(1,5),GG,zero,zero,W(1,16))
CALL FVOXXX(W(1,3),W(1,16),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 21
CALL IOVXXX(W(1,6),W(1,10),W(1,14),GG,AMP(21))
CALL FVIXXX(W(1,6),W(1,16),GG,zero,zero,W(1,7))
# Amplitude(s) for diagram number 22
CALL IOVXXX(W(1,7),W(1,3),W(1,14),GG,AMP(22))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,16),W(1,14),W(1,12),GG,AMP(23))
# Amplitude(s) for diagram number 24
CALL IOVXXX(W(1,4),W(1,10),W(1,15),GG,AMP(24))
CALL FVIXXX(W(1,4),W(1,16),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 25
CALL IOVXXX(W(1,10),W(1,3),W(1,15),GG,AMP(25))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,16),W(1,15),W(1,8),GG,AMP(26))
CALL FVOXXX(W(1,2),W(1,16),GG,zero,zero,W(1,13))
# Amplitude(s) for diagram number 27
CALL IOVXXX(W(1,6),W(1,13),W(1,8),GG,AMP(27))
# Amplitude(s) for diagram number 28
CALL IOVXXX(W(1,7),W(1,2),W(1,8),GG,AMP(28))
# Amplitude(s) for diagram number 29
CALL IOVXXX(W(1,4),W(1,13),W(1,12),GG,AMP(29))
# Amplitude(s) for diagram number 30
CALL IOVXXX(W(1,10),W(1,2),W(1,12),GG,AMP(30))
CALL FVIXXX(W(1,1),W(1,14),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 31
CALL IOVXXX(W(1,10),W(1,5),W(1,12),GG,AMP(31))
CALL FVIXXX(W(1,1),W(1,12),GG,zero,zero,W(1,13))
# Amplitude(s) for diagram number 32
CALL IOVXXX(W(1,13),W(1,5),W(1,14),GG,AMP(32))
# Amplitude(s) for diagram number 33
CALL IOVXXX(W(1,10),W(1,3),W(1,11),GG,AMP(33))
CALL FVIXXX(W(1,1),W(1,11),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 34
CALL IOVXXX(W(1,10),W(1,3),W(1,14),GG,AMP(34))
CALL FVIXXX(W(1,1),W(1,15),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 35
CALL IOVXXX(W(1,14),W(1,5),W(1,8),GG,AMP(35))
CALL FVIXXX(W(1,1),W(1,8),GG,zero,zero,W(1,4))
# Amplitude(s) for diagram number 36
CALL IOVXXX(W(1,4),W(1,5),W(1,15),GG,AMP(36))
# Amplitude(s) for diagram number 37
CALL IOVXXX(W(1,14),W(1,3),W(1,9),GG,AMP(37))
CALL FVIXXX(W(1,1),W(1,9),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 38
CALL IOVXXX(W(1,14),W(1,3),W(1,15),GG,AMP(38))
# Amplitude(s) for diagram number 39
CALL IOVXXX(W(1,4),W(1,2),W(1,11),GG,AMP(39))
# Amplitude(s) for diagram number 40
CALL IOVXXX(W(1,10),W(1,2),W(1,8),GG,AMP(40))
# Amplitude(s) for diagram number 41
CALL IOVXXX(W(1,13),W(1,2),W(1,9),GG,AMP(41))
# Amplitude(s) for diagram number 42
CALL IOVXXX(W(1,14),W(1,2),W(1,12),GG,AMP(42))""".split('\n'))

        exporter = export_v4.ProcessExporterFortranME()

        numerator = [[27,    9,    9,    3,    3,    9],
                     [9,   27,    3,    9,    9,    3],
                     [9,    3,   27,    9,    9,    3],
                     [3,    9,    9,   27,    3,    9],
                     [3,    9,    9,    3,   27,    9],
                     [9,    3,    3,    9,    9,   27]]
        denom = 1

        i = 0
        for data in exporter.get_color_data_lines(\
                         matrix_element):
            #misc.sprint(data)
            if 'DATA' not in data:
                continue
            if 'denom' in data.lower():
                _, data, _ = data.split('/')
                self.assertEqual(int(data), denom) 
                continue 
            else:   
                _, data, _ = data.split('/') 
                number = [int(n) for n in data.split(',') ]
            for j,val in enumerate(number):
                coeff = 1 if j ==0 else 2 # symmetry factor
                self.assertAlmostEqual((coeff*numerator[i][i+j]), val)
            i+=1 



        out, nb = exporter.get_JAMP_lines(matrix_element)    
        self.assertEqual(nb,55)

        # Test JAMP (color amplitude) output
        if six.PY3:
            self.assertEqual(out, [' TMP_JAMP(39) = AMP(39) +  AMP(40) ! used 3 times', ' TMP_JAMP(38) = AMP(37) +  AMP(38) ! used 3 times', ' TMP_JAMP(37) = AMP(35) +  AMP(36) ! used 3 times', ' TMP_JAMP(36) = AMP(29) +  AMP(37) ! used 3 times', ' TMP_JAMP(35) = AMP(29) +  AMP(30) ! used 3 times', ' TMP_JAMP(34) = AMP(28) +  AMP(36) ! used 3 times', ' TMP_JAMP(33) = AMP(27) +  AMP(35) ! used 3 times', ' TMP_JAMP(32) = AMP(27) +  AMP(28) ! used 3 times', ' TMP_JAMP(31) = AMP(25) +  AMP(35) ! used 3 times', ' TMP_JAMP(30) = AMP(25) +  AMP(27) ! used 3 times', ' TMP_JAMP(29) = AMP(24) +  AMP(25) ! used 3 times', ' TMP_JAMP(28) = AMP(21) +  AMP(22) ! used 3 times', ' TMP_JAMP(27) = AMP(20) +  AMP(34) ! used 3 times', ' TMP_JAMP(26) = AMP(14) +  AMP(22) ! used 3 times', ' TMP_JAMP(25) = AMP(14) +  AMP(15) ! used 3 times', ' TMP_JAMP(24) = AMP(11) +  AMP(34) ! used 3 times', ' TMP_JAMP(23) = AMP(11) +  AMP(20) ! used 3 times', ' TMP_JAMP(22) = AMP(5) +  AMP(8) ! used 3 times', ' TMP_JAMP(21) = AMP(1) +  AMP(2) ! used 3 times', ' TMP_JAMP(20) = AMP(41) +  AMP(42) ! used 3 times', ' TMP_JAMP(19) = AMP(30) +  AMP(32) ! used 3 times', ' TMP_JAMP(18) = AMP(24) +  AMP(41) ! used 3 times', ' TMP_JAMP(17) = AMP(21) +  AMP(32) ! used 3 times', ' TMP_JAMP(16) = AMP(19) +  AMP(20) ! used 3 times', ' TMP_JAMP(15) = AMP(18) +  AMP(38) ! used 3 times', ' TMP_JAMP(14) = AMP(15) +  AMP(18) ! used 3 times', ' TMP_JAMP(13) = AMP(14) +  AMP(17) ! used 3 times', ' TMP_JAMP(12) = AMP(12) +  AMP(25) ! used 3 times', ' TMP_JAMP(11) = AMP(11) +  AMP(12) ! used 3 times', ' TMP_JAMP(10) = AMP(7) +  AMP(8) ! used 3 times', ' TMP_JAMP(9) = AMP(2) +  AMP(9) ! used 3 times', ' TMP_JAMP(8) = AMP(22) +  AMP(31) ! used 3 times', ' TMP_JAMP(7) = AMP(12) +  AMP(33) ! used 3 times', ' TMP_JAMP(6) = AMP(4) +  AMP(5) ! used 3 times', ' TMP_JAMP(5) = AMP(1) +  AMP(10) ! used 3 times', ' TMP_JAMP(4) = AMP(19) +  AMP(33) ! used 3 times', ' TMP_JAMP(3) = AMP(17) +  AMP(31) ! used 3 times', ' TMP_JAMP(2) = AMP(9) +  AMP(10) ! used 3 times', ' TMP_JAMP(1) = AMP(4) +  AMP(7) ! used 3 times', ' TMP_JAMP(52) = TMP_JAMP(38) + (3.000000000000000d+00) * AMP(42) ! used 2 times', ' TMP_JAMP(51) = TMP_JAMP(37) +  TMP_JAMP(32) ! used 2 times', ' TMP_JAMP(50) = TMP_JAMP(36) + (3.333333333333333d-01) * TMP_JAMP(22) ! used 2 times', ' TMP_JAMP(49) = TMP_JAMP(35) + (3.000000000000000d+00) * AMP(42) ! used 2 times', ' TMP_JAMP(48) = TMP_JAMP(27) + (3.000000000000000d+00) * TMP_JAMP(26) ! used 2 times', ' TMP_JAMP(47) = TMP_JAMP(22) + (3.333333333333333d-01) * TMP_JAMP(21) ! used 2 times', ' TMP_JAMP(46) = TMP_JAMP(18) + (3.333333333333333d-01) * TMP_JAMP(14) ! used 2 times', ' TMP_JAMP(45) = TMP_JAMP(18) + (3.333333333333333d-01) * TMP_JAMP(17) ! used 2 times', ' TMP_JAMP(44) = TMP_JAMP(17) + (3.333333333333333d-01) * TMP_JAMP(11) ! used 2 times', ' TMP_JAMP(43) = TMP_JAMP(4) + (3.000000000000000d+00) * TMP_JAMP(3) ! used 2 times', ' TMP_JAMP(42) = TMP_JAMP(4) + (3.333333333333333d-01) * TMP_JAMP(2) ! used 2 times', ' TMP_JAMP(41) = TMP_JAMP(3) + (3.333333333333333d-01) * TMP_JAMP(1) ! used 2 times', ' TMP_JAMP(40) = TMP_JAMP(2) + (3.000000000000000d+00) * TMP_JAMP(1) ! used 2 times', ' TMP_JAMP(55) = TMP_JAMP(48) +  TMP_JAMP(43) ! used 2 times', ' TMP_JAMP(54) = TMP_JAMP(47) + (3.333333333333333d-01) * TMP_JAMP(39) ! used 2 times', ' TMP_JAMP(53) = TMP_JAMP(40) +  TMP_JAMP(39) ! used 2 times', ' JAMP(1) = (-2.500000000000000d-01)*AMP(16)+(-2.500000000000000d-01)*AMP(23)+(2.500000000000000d-01)*TMP_JAMP(36)+(8.333333333333333d-02)*TMP_JAMP(47)+(8.333333333333333d-02)*TMP_JAMP(51)+(2.777777777777778d-02)*TMP_JAMP(53)+(8.333333333333333d-02)*TMP_JAMP(55)', ' JAMP(2) = (2.500000000000000d-01)*AMP(13)+(2.500000000000000d-01)*AMP(26)+(-2.500000000000000d-01)*TMP_JAMP(12)+(-8.333333333333333d-02)*TMP_JAMP(15)+(-8.333333333333333d-02)*TMP_JAMP(19)+(-2.777777777777778d-02)*TMP_JAMP(20)+(-8.333333333333333d-02)*TMP_JAMP(21)+(-2.500000000000000d-01)*TMP_JAMP(33)+(-8.333333333333333d-02)*TMP_JAMP(41)+(-2.500000000000000d-01)*TMP_JAMP(42)+(-8.333333333333333d-02)*TMP_JAMP(50)', ' JAMP(3) = (2.500000000000000d-01)*AMP(6)+(-2.500000000000000d-01)*AMP(26)+(-2.500000000000000d-01)*TMP_JAMP(34)+(-8.333333333333333d-02)*TMP_JAMP(44)+(-2.500000000000000d-01)*TMP_JAMP(46)+(-8.333333333333333d-02)*TMP_JAMP(53)+(-2.777777777777778d-02)*TMP_JAMP(55)', ' JAMP(4) = (2.500000000000000d-01)*AMP(3)+(2.500000000000000d-01)*AMP(16)+(8.333333333333333d-02)*AMP(28)+(2.500000000000000d-01)*AMP(38)+(2.500000000000000d-01)*AMP(40)+(2.500000000000000d-01)*TMP_JAMP(5)+(8.333333333333333d-02)*TMP_JAMP(6)+(8.333333333333333d-02)*TMP_JAMP(7)+(2.777777777777778d-02)*TMP_JAMP(8)+(2.500000000000000d-01)*TMP_JAMP(14)+(8.333333333333333d-02)*TMP_JAMP(24)+(8.333333333333333d-02)*TMP_JAMP(30)+(8.333333333333333d-02)*TMP_JAMP(45)+(2.777777777777778d-02)*TMP_JAMP(49)', ' JAMP(5) = (-2.500000000000000d-01)*AMP(3)+(2.500000000000000d-01)*AMP(23)+(2.500000000000000d-01)*AMP(30)+(8.333333333333333d-02)*AMP(36)+(2.500000000000000d-01)*AMP(39)+(2.500000000000000d-01)*TMP_JAMP(9)+(8.333333333333333d-02)*TMP_JAMP(10)+(2.777777777777778d-02)*TMP_JAMP(13)+(8.333333333333333d-02)*TMP_JAMP(16)+(8.333333333333333d-02)*TMP_JAMP(31)+(2.500000000000000d-01)*TMP_JAMP(44)+(8.333333333333333d-02)*TMP_JAMP(46)+(2.777777777777778d-02)*TMP_JAMP(52)', ' JAMP(6) = (-2.500000000000000d-01)*AMP(6)+(-2.500000000000000d-01)*AMP(13)+(-2.500000000000000d-01)*AMP(34)+(-2.500000000000000d-01)*TMP_JAMP(23)+(-8.333333333333333d-02)*TMP_JAMP(25)+(-8.333333333333333d-02)*TMP_JAMP(28)+(-2.777777777777778d-02)*TMP_JAMP(29)+(-8.333333333333333d-02)*TMP_JAMP(38)+(-8.333333333333333d-02)*TMP_JAMP(49)+(-2.777777777777778d-02)*TMP_JAMP(51)+(-2.500000000000000d-01)*TMP_JAMP(54)']
            )
            

        # Test configs.inc file
        writer = writers.FortranWriter(self.give_pos('test'))
        mapconfigs, (s_and_t_channels, nqcd_list) = exporter.write_configs_file(writer,
                                                                 matrix_element)
        writer.close()

        #print( open(self.give_pos('test')).read())

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/21/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/6,-1/
      DATA (SPROP(I,-2,1),I=1,1)/-2/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,-2/
      DATA (SPROP(I,-3,1),I=1,1)/21/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/4,3/
      DATA (SPROP(I,-1,2),I=1,1)/21/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/5,-1/
      DATA (SPROP(I,-2,2),I=1,1)/2/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/6,-2/
      DATA (SPROP(I,-3,2),I=1,1)/21/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/6,5/
      DATA (SPROP(I,-1,3),I=1,1)/21/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/4,3/
      DATA (SPROP(I,-2,3),I=1,1)/21/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/-1,-2/
      DATA (SPROP(I,-3,3),I=1,1)/21/
      DATA TPRID(-3,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/6,3/
      DATA (SPROP(I,-1,4),I=1,1)/21/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/5,-1/
      DATA (SPROP(I,-2,4),I=1,1)/2/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/4,-2/
      DATA (SPROP(I,-3,4),I=1,1)/21/
      DATA TPRID(-3,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/6,3/
      DATA (SPROP(I,-1,5),I=1,1)/21/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/4,-1/
      DATA (SPROP(I,-2,5),I=1,1)/-2/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/5,-2/
      DATA (SPROP(I,-3,5),I=1,1)/21/
      DATA TPRID(-3,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/6,3/
      DATA (SPROP(I,-1,6),I=1,1)/21/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/5,4/
      DATA (SPROP(I,-2,6),I=1,1)/21/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,-1/
      DATA (SPROP(I,-3,6),I=1,1)/21/
      DATA TPRID(-3,6)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA TSTRATEGY(7)/2/
      DATA (IFOREST(I,-1,7),I=1,2)/5,4/
      DATA (SPROP(I,-1,7),I=1,1)/21/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/6,-1/
      DATA (SPROP(I,-2,7),I=1,1)/-2/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/-2,3/
      DATA (SPROP(I,-3,7),I=1,1)/21/
      DATA TPRID(-3,7)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA TSTRATEGY(8)/2/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA (SPROP(I,-1,8),I=1,1)/21/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,3/
      DATA (SPROP(I,-2,8),I=1,1)/2/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/6,-2/
      DATA (SPROP(I,-3,8),I=1,1)/21/
      DATA TPRID(-3,8)/0/
C     Diagram 9
      DATA MAPCONFIG(9)/9/
      DATA TSTRATEGY(9)/2/
      DATA (IFOREST(I,-1,9),I=1,2)/6,5/
      DATA (SPROP(I,-1,9),I=1,1)/21/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/-1,4/
      DATA (SPROP(I,-2,9),I=1,1)/-2/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/-2,3/
      DATA (SPROP(I,-3,9),I=1,1)/21/
      DATA TPRID(-3,9)/0/
C     Diagram 10
      DATA MAPCONFIG(10)/10/
      DATA TSTRATEGY(10)/2/
      DATA (IFOREST(I,-1,10),I=1,2)/6,5/
      DATA (SPROP(I,-1,10),I=1,1)/21/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/-1,3/
      DATA (SPROP(I,-2,10),I=1,1)/2/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/4,-2/
      DATA (SPROP(I,-3,10),I=1,1)/21/
      DATA TPRID(-3,10)/0/
C     Diagram 11
      DATA MAPCONFIG(11)/11/
      DATA TSTRATEGY(11)/2/
      DATA (IFOREST(I,-1,11),I=1,2)/1,3/
      DATA TPRID(-1,11)/21/
      DATA (SPROP(I,-1,11),I=1,1)/0/
      DATA (IFOREST(I,-2,11),I=1,2)/-1,5/
      DATA TPRID(-2,11)/2/
      DATA (SPROP(I,-2,11),I=1,1)/0/
      DATA (IFOREST(I,-3,11),I=1,2)/-2,6/
      DATA TPRID(-3,11)/21/
      DATA (SPROP(I,-3,11),I=1,1)/0/
      DATA (IFOREST(I,-4,11),I=1,2)/-3,4/
      DATA TPRID(-4,11)/2/
      DATA (SPROP(I,-4,11),I=1,1)/0/
C     Diagram 12
      DATA MAPCONFIG(12)/12/
      DATA TSTRATEGY(12)/2/
      DATA (IFOREST(I,-1,12),I=1,2)/1,3/
      DATA TPRID(-1,12)/21/
      DATA (SPROP(I,-1,12),I=1,1)/0/
      DATA (IFOREST(I,-2,12),I=1,2)/-1,6/
      DATA TPRID(-2,12)/2/
      DATA (SPROP(I,-2,12),I=1,1)/0/
      DATA (IFOREST(I,-3,12),I=1,2)/-2,5/
      DATA TPRID(-3,12)/21/
      DATA (SPROP(I,-3,12),I=1,1)/0/
      DATA (IFOREST(I,-4,12),I=1,2)/-3,4/
      DATA TPRID(-4,12)/2/
      DATA (SPROP(I,-4,12),I=1,1)/0/
C     Diagram 13
      DATA MAPCONFIG(13)/13/
      DATA TSTRATEGY(13)/2/
      DATA (IFOREST(I,-1,13),I=1,2)/6,5/
      DATA (SPROP(I,-1,13),I=1,1)/21/
      DATA TPRID(-1,13)/0/
      DATA (IFOREST(I,-2,13),I=1,2)/1,3/
      DATA TPRID(-2,13)/21/
      DATA (SPROP(I,-2,13),I=1,1)/0/
      DATA (IFOREST(I,-3,13),I=1,2)/-2,-1/
      DATA TPRID(-3,13)/21/
      DATA (SPROP(I,-3,13),I=1,1)/0/
      DATA (IFOREST(I,-4,13),I=1,2)/-3,4/
      DATA TPRID(-4,13)/2/
      DATA (SPROP(I,-4,13),I=1,1)/0/
C     Diagram 14
      DATA MAPCONFIG(14)/14/
      DATA TSTRATEGY(14)/2/
      DATA (IFOREST(I,-1,14),I=1,2)/1,3/
      DATA TPRID(-1,14)/21/
      DATA (SPROP(I,-1,14),I=1,1)/0/
      DATA (IFOREST(I,-2,14),I=1,2)/-1,4/
      DATA TPRID(-2,14)/2/
      DATA (SPROP(I,-2,14),I=1,1)/0/
      DATA (IFOREST(I,-3,14),I=1,2)/-2,5/
      DATA TPRID(-3,14)/21/
      DATA (SPROP(I,-3,14),I=1,1)/0/
      DATA (IFOREST(I,-4,14),I=1,2)/-3,6/
      DATA TPRID(-4,14)/2/
      DATA (SPROP(I,-4,14),I=1,1)/0/
C     Diagram 15
      DATA MAPCONFIG(15)/15/
      DATA TSTRATEGY(15)/2/
      DATA (IFOREST(I,-1,15),I=1,2)/1,3/
      DATA TPRID(-1,15)/21/
      DATA (SPROP(I,-1,15),I=1,1)/0/
      DATA (IFOREST(I,-2,15),I=1,2)/-1,5/
      DATA TPRID(-2,15)/2/
      DATA (SPROP(I,-2,15),I=1,1)/0/
      DATA (IFOREST(I,-3,15),I=1,2)/-2,4/
      DATA TPRID(-3,15)/21/
      DATA (SPROP(I,-3,15),I=1,1)/0/
      DATA (IFOREST(I,-4,15),I=1,2)/-3,6/
      DATA TPRID(-4,15)/2/
      DATA (SPROP(I,-4,15),I=1,1)/0/
C     Diagram 16
      DATA MAPCONFIG(16)/16/
      DATA TSTRATEGY(16)/2/
      DATA (IFOREST(I,-1,16),I=1,2)/5,4/
      DATA (SPROP(I,-1,16),I=1,1)/21/
      DATA TPRID(-1,16)/0/
      DATA (IFOREST(I,-2,16),I=1,2)/1,3/
      DATA TPRID(-2,16)/21/
      DATA (SPROP(I,-2,16),I=1,1)/0/
      DATA (IFOREST(I,-3,16),I=1,2)/-2,-1/
      DATA TPRID(-3,16)/21/
      DATA (SPROP(I,-3,16),I=1,1)/0/
      DATA (IFOREST(I,-4,16),I=1,2)/-3,6/
      DATA TPRID(-4,16)/2/
      DATA (SPROP(I,-4,16),I=1,1)/0/
C     Diagram 17
      DATA MAPCONFIG(17)/17/
      DATA TSTRATEGY(17)/2/
      DATA (IFOREST(I,-1,17),I=1,2)/5,4/
      DATA (SPROP(I,-1,17),I=1,1)/21/
      DATA TPRID(-1,17)/0/
      DATA (IFOREST(I,-2,17),I=1,2)/6,-1/
      DATA (SPROP(I,-2,17),I=1,1)/-2/
      DATA TPRID(-2,17)/0/
      DATA (IFOREST(I,-3,17),I=1,2)/1,3/
      DATA TPRID(-3,17)/21/
      DATA (SPROP(I,-3,17),I=1,1)/0/
      DATA (IFOREST(I,-4,17),I=1,2)/-3,-2/
      DATA TPRID(-4,17)/2/
      DATA (SPROP(I,-4,17),I=1,1)/0/
C     Diagram 18
      DATA MAPCONFIG(18)/18/
      DATA TSTRATEGY(18)/2/
      DATA (IFOREST(I,-1,18),I=1,2)/5,4/
      DATA (SPROP(I,-1,18),I=1,1)/21/
      DATA TPRID(-1,18)/0/
      DATA (IFOREST(I,-2,18),I=1,2)/1,3/
      DATA TPRID(-2,18)/21/
      DATA (SPROP(I,-2,18),I=1,1)/0/
      DATA (IFOREST(I,-3,18),I=1,2)/-2,6/
      DATA TPRID(-3,18)/2/
      DATA (SPROP(I,-3,18),I=1,1)/0/
      DATA (IFOREST(I,-4,18),I=1,2)/-3,-1/
      DATA TPRID(-4,18)/2/
      DATA (SPROP(I,-4,18),I=1,1)/0/
C     Diagram 19
      DATA MAPCONFIG(19)/19/
      DATA TSTRATEGY(19)/2/
      DATA (IFOREST(I,-1,19),I=1,2)/6,5/
      DATA (SPROP(I,-1,19),I=1,1)/21/
      DATA TPRID(-1,19)/0/
      DATA (IFOREST(I,-2,19),I=1,2)/-1,4/
      DATA (SPROP(I,-2,19),I=1,1)/-2/
      DATA TPRID(-2,19)/0/
      DATA (IFOREST(I,-3,19),I=1,2)/1,3/
      DATA TPRID(-3,19)/21/
      DATA (SPROP(I,-3,19),I=1,1)/0/
      DATA (IFOREST(I,-4,19),I=1,2)/-3,-2/
      DATA TPRID(-4,19)/2/
      DATA (SPROP(I,-4,19),I=1,1)/0/
C     Diagram 20
      DATA MAPCONFIG(20)/20/
      DATA TSTRATEGY(20)/2/
      DATA (IFOREST(I,-1,20),I=1,2)/6,5/
      DATA (SPROP(I,-1,20),I=1,1)/21/
      DATA TPRID(-1,20)/0/
      DATA (IFOREST(I,-2,20),I=1,2)/1,3/
      DATA TPRID(-2,20)/21/
      DATA (SPROP(I,-2,20),I=1,1)/0/
      DATA (IFOREST(I,-3,20),I=1,2)/-2,4/
      DATA TPRID(-3,20)/2/
      DATA (SPROP(I,-3,20),I=1,1)/0/
      DATA (IFOREST(I,-4,20),I=1,2)/-3,-1/
      DATA TPRID(-4,20)/2/
      DATA (SPROP(I,-4,20),I=1,1)/0/
C     Diagram 21
      DATA MAPCONFIG(21)/21/
      DATA TSTRATEGY(21)/2/
      DATA (IFOREST(I,-1,21),I=1,2)/1,5/
      DATA TPRID(-1,21)/21/
      DATA (SPROP(I,-1,21),I=1,1)/0/
      DATA (IFOREST(I,-2,21),I=1,2)/-1,3/
      DATA TPRID(-2,21)/2/
      DATA (SPROP(I,-2,21),I=1,1)/0/
      DATA (IFOREST(I,-3,21),I=1,2)/-2,6/
      DATA TPRID(-3,21)/21/
      DATA (SPROP(I,-3,21),I=1,1)/0/
      DATA (IFOREST(I,-4,21),I=1,2)/-3,4/
      DATA TPRID(-4,21)/2/
      DATA (SPROP(I,-4,21),I=1,1)/0/
C     Diagram 22
      DATA MAPCONFIG(22)/22/
      DATA TSTRATEGY(22)/2/
      DATA (IFOREST(I,-1,22),I=1,2)/1,5/
      DATA TPRID(-1,22)/21/
      DATA (SPROP(I,-1,22),I=1,1)/0/
      DATA (IFOREST(I,-2,22),I=1,2)/-1,6/
      DATA TPRID(-2,22)/2/
      DATA (SPROP(I,-2,22),I=1,1)/0/
      DATA (IFOREST(I,-3,22),I=1,2)/-2,3/
      DATA TPRID(-3,22)/21/
      DATA (SPROP(I,-3,22),I=1,1)/0/
      DATA (IFOREST(I,-4,22),I=1,2)/-3,4/
      DATA TPRID(-4,22)/2/
      DATA (SPROP(I,-4,22),I=1,1)/0/
C     Diagram 23
      DATA MAPCONFIG(23)/23/
      DATA TSTRATEGY(23)/2/
      DATA (IFOREST(I,-1,23),I=1,2)/6,3/
      DATA (SPROP(I,-1,23),I=1,1)/21/
      DATA TPRID(-1,23)/0/
      DATA (IFOREST(I,-2,23),I=1,2)/1,5/
      DATA TPRID(-2,23)/21/
      DATA (SPROP(I,-2,23),I=1,1)/0/
      DATA (IFOREST(I,-3,23),I=1,2)/-2,-1/
      DATA TPRID(-3,23)/21/
      DATA (SPROP(I,-3,23),I=1,1)/0/
      DATA (IFOREST(I,-4,23),I=1,2)/-3,4/
      DATA TPRID(-4,23)/2/
      DATA (SPROP(I,-4,23),I=1,1)/0/
C     Diagram 24
      DATA MAPCONFIG(24)/24/
      DATA TSTRATEGY(24)/2/
      DATA (IFOREST(I,-1,24),I=1,2)/1,5/
      DATA TPRID(-1,24)/21/
      DATA (SPROP(I,-1,24),I=1,1)/0/
      DATA (IFOREST(I,-2,24),I=1,2)/-1,3/
      DATA TPRID(-2,24)/2/
      DATA (SPROP(I,-2,24),I=1,1)/0/
      DATA (IFOREST(I,-3,24),I=1,2)/-2,4/
      DATA TPRID(-3,24)/21/
      DATA (SPROP(I,-3,24),I=1,1)/0/
      DATA (IFOREST(I,-4,24),I=1,2)/-3,6/
      DATA TPRID(-4,24)/2/
      DATA (SPROP(I,-4,24),I=1,1)/0/
C     Diagram 25
      DATA MAPCONFIG(25)/25/
      DATA TSTRATEGY(25)/2/
      DATA (IFOREST(I,-1,25),I=1,2)/1,5/
      DATA TPRID(-1,25)/21/
      DATA (SPROP(I,-1,25),I=1,1)/0/
      DATA (IFOREST(I,-2,25),I=1,2)/-1,4/
      DATA TPRID(-2,25)/2/
      DATA (SPROP(I,-2,25),I=1,1)/0/
      DATA (IFOREST(I,-3,25),I=1,2)/-2,3/
      DATA TPRID(-3,25)/21/
      DATA (SPROP(I,-3,25),I=1,1)/0/
      DATA (IFOREST(I,-4,25),I=1,2)/-3,6/
      DATA TPRID(-4,25)/2/
      DATA (SPROP(I,-4,25),I=1,1)/0/
C     Diagram 26
      DATA MAPCONFIG(26)/26/
      DATA TSTRATEGY(26)/2/
      DATA (IFOREST(I,-1,26),I=1,2)/4,3/
      DATA (SPROP(I,-1,26),I=1,1)/21/
      DATA TPRID(-1,26)/0/
      DATA (IFOREST(I,-2,26),I=1,2)/1,5/
      DATA TPRID(-2,26)/21/
      DATA (SPROP(I,-2,26),I=1,1)/0/
      DATA (IFOREST(I,-3,26),I=1,2)/-2,-1/
      DATA TPRID(-3,26)/21/
      DATA (SPROP(I,-3,26),I=1,1)/0/
      DATA (IFOREST(I,-4,26),I=1,2)/-3,6/
      DATA TPRID(-4,26)/2/
      DATA (SPROP(I,-4,26),I=1,1)/0/
C     Diagram 27
      DATA MAPCONFIG(27)/27/
      DATA TSTRATEGY(27)/2/
      DATA (IFOREST(I,-1,27),I=1,2)/4,3/
      DATA (SPROP(I,-1,27),I=1,1)/21/
      DATA TPRID(-1,27)/0/
      DATA (IFOREST(I,-2,27),I=1,2)/6,-1/
      DATA (SPROP(I,-2,27),I=1,1)/-2/
      DATA TPRID(-2,27)/0/
      DATA (IFOREST(I,-3,27),I=1,2)/1,5/
      DATA TPRID(-3,27)/21/
      DATA (SPROP(I,-3,27),I=1,1)/0/
      DATA (IFOREST(I,-4,27),I=1,2)/-3,-2/
      DATA TPRID(-4,27)/2/
      DATA (SPROP(I,-4,27),I=1,1)/0/
C     Diagram 28
      DATA MAPCONFIG(28)/28/
      DATA TSTRATEGY(28)/2/
      DATA (IFOREST(I,-1,28),I=1,2)/4,3/
      DATA (SPROP(I,-1,28),I=1,1)/21/
      DATA TPRID(-1,28)/0/
      DATA (IFOREST(I,-2,28),I=1,2)/1,5/
      DATA TPRID(-2,28)/21/
      DATA (SPROP(I,-2,28),I=1,1)/0/
      DATA (IFOREST(I,-3,28),I=1,2)/-2,6/
      DATA TPRID(-3,28)/2/
      DATA (SPROP(I,-3,28),I=1,1)/0/
      DATA (IFOREST(I,-4,28),I=1,2)/-3,-1/
      DATA TPRID(-4,28)/2/
      DATA (SPROP(I,-4,28),I=1,1)/0/
C     Diagram 29
      DATA MAPCONFIG(29)/29/
      DATA TSTRATEGY(29)/2/
      DATA (IFOREST(I,-1,29),I=1,2)/6,3/
      DATA (SPROP(I,-1,29),I=1,1)/21/
      DATA TPRID(-1,29)/0/
      DATA (IFOREST(I,-2,29),I=1,2)/4,-1/
      DATA (SPROP(I,-2,29),I=1,1)/-2/
      DATA TPRID(-2,29)/0/
      DATA (IFOREST(I,-3,29),I=1,2)/1,5/
      DATA TPRID(-3,29)/21/
      DATA (SPROP(I,-3,29),I=1,1)/0/
      DATA (IFOREST(I,-4,29),I=1,2)/-3,-2/
      DATA TPRID(-4,29)/2/
      DATA (SPROP(I,-4,29),I=1,1)/0/
C     Diagram 30
      DATA MAPCONFIG(30)/30/
      DATA TSTRATEGY(30)/2/
      DATA (IFOREST(I,-1,30),I=1,2)/6,3/
      DATA (SPROP(I,-1,30),I=1,1)/21/
      DATA TPRID(-1,30)/0/
      DATA (IFOREST(I,-2,30),I=1,2)/1,5/
      DATA TPRID(-2,30)/21/
      DATA (SPROP(I,-2,30),I=1,1)/0/
      DATA (IFOREST(I,-3,30),I=1,2)/-2,4/
      DATA TPRID(-3,30)/2/
      DATA (SPROP(I,-3,30),I=1,1)/0/
      DATA (IFOREST(I,-4,30),I=1,2)/-3,-1/
      DATA TPRID(-4,30)/2/
      DATA (SPROP(I,-4,30),I=1,1)/0/
C     Diagram 31
      DATA MAPCONFIG(31)/31/
      DATA TSTRATEGY(31)/2/
      DATA (IFOREST(I,-1,31),I=1,2)/6,3/
      DATA (SPROP(I,-1,31),I=1,1)/21/
      DATA TPRID(-1,31)/0/
      DATA (IFOREST(I,-2,31),I=1,2)/5,-1/
      DATA (SPROP(I,-2,31),I=1,1)/2/
      DATA TPRID(-2,31)/0/
      DATA (IFOREST(I,-3,31),I=1,2)/1,-2/
      DATA TPRID(-3,31)/21/
      DATA (SPROP(I,-3,31),I=1,1)/0/
      DATA (IFOREST(I,-4,31),I=1,2)/-3,4/
      DATA TPRID(-4,31)/2/
      DATA (SPROP(I,-4,31),I=1,1)/0/
C     Diagram 32
      DATA MAPCONFIG(32)/32/
      DATA TSTRATEGY(32)/2/
      DATA (IFOREST(I,-1,32),I=1,2)/6,3/
      DATA (SPROP(I,-1,32),I=1,1)/21/
      DATA TPRID(-1,32)/0/
      DATA (IFOREST(I,-2,32),I=1,2)/1,-1/
      DATA TPRID(-2,32)/2/
      DATA (SPROP(I,-2,32),I=1,1)/0/
      DATA (IFOREST(I,-3,32),I=1,2)/-2,5/
      DATA TPRID(-3,32)/21/
      DATA (SPROP(I,-3,32),I=1,1)/0/
      DATA (IFOREST(I,-4,32),I=1,2)/-3,4/
      DATA TPRID(-4,32)/2/
      DATA (SPROP(I,-4,32),I=1,1)/0/
C     Diagram 33
      DATA MAPCONFIG(33)/33/
      DATA TSTRATEGY(33)/2/
      DATA (IFOREST(I,-1,33),I=1,2)/6,5/
      DATA (SPROP(I,-1,33),I=1,1)/21/
      DATA TPRID(-1,33)/0/
      DATA (IFOREST(I,-2,33),I=1,2)/-1,3/
      DATA (SPROP(I,-2,33),I=1,1)/2/
      DATA TPRID(-2,33)/0/
      DATA (IFOREST(I,-3,33),I=1,2)/1,-2/
      DATA TPRID(-3,33)/21/
      DATA (SPROP(I,-3,33),I=1,1)/0/
      DATA (IFOREST(I,-4,33),I=1,2)/-3,4/
      DATA TPRID(-4,33)/2/
      DATA (SPROP(I,-4,33),I=1,1)/0/
C     Diagram 34
      DATA MAPCONFIG(34)/34/
      DATA TSTRATEGY(34)/2/
      DATA (IFOREST(I,-1,34),I=1,2)/6,5/
      DATA (SPROP(I,-1,34),I=1,1)/21/
      DATA TPRID(-1,34)/0/
      DATA (IFOREST(I,-2,34),I=1,2)/1,-1/
      DATA TPRID(-2,34)/2/
      DATA (SPROP(I,-2,34),I=1,1)/0/
      DATA (IFOREST(I,-3,34),I=1,2)/-2,3/
      DATA TPRID(-3,34)/21/
      DATA (SPROP(I,-3,34),I=1,1)/0/
      DATA (IFOREST(I,-4,34),I=1,2)/-3,4/
      DATA TPRID(-4,34)/2/
      DATA (SPROP(I,-4,34),I=1,1)/0/
C     Diagram 35
      DATA MAPCONFIG(35)/35/
      DATA TSTRATEGY(35)/2/
      DATA (IFOREST(I,-1,35),I=1,2)/4,3/
      DATA (SPROP(I,-1,35),I=1,1)/21/
      DATA TPRID(-1,35)/0/
      DATA (IFOREST(I,-2,35),I=1,2)/5,-1/
      DATA (SPROP(I,-2,35),I=1,1)/2/
      DATA TPRID(-2,35)/0/
      DATA (IFOREST(I,-3,35),I=1,2)/1,-2/
      DATA TPRID(-3,35)/21/
      DATA (SPROP(I,-3,35),I=1,1)/0/
      DATA (IFOREST(I,-4,35),I=1,2)/-3,6/
      DATA TPRID(-4,35)/2/
      DATA (SPROP(I,-4,35),I=1,1)/0/
C     Diagram 36
      DATA MAPCONFIG(36)/36/
      DATA TSTRATEGY(36)/2/
      DATA (IFOREST(I,-1,36),I=1,2)/4,3/
      DATA (SPROP(I,-1,36),I=1,1)/21/
      DATA TPRID(-1,36)/0/
      DATA (IFOREST(I,-2,36),I=1,2)/1,-1/
      DATA TPRID(-2,36)/2/
      DATA (SPROP(I,-2,36),I=1,1)/0/
      DATA (IFOREST(I,-3,36),I=1,2)/-2,5/
      DATA TPRID(-3,36)/21/
      DATA (SPROP(I,-3,36),I=1,1)/0/
      DATA (IFOREST(I,-4,36),I=1,2)/-3,6/
      DATA TPRID(-4,36)/2/
      DATA (SPROP(I,-4,36),I=1,1)/0/
C     Diagram 37
      DATA MAPCONFIG(37)/37/
      DATA TSTRATEGY(37)/2/
      DATA (IFOREST(I,-1,37),I=1,2)/5,4/
      DATA (SPROP(I,-1,37),I=1,1)/21/
      DATA TPRID(-1,37)/0/
      DATA (IFOREST(I,-2,37),I=1,2)/-1,3/
      DATA (SPROP(I,-2,37),I=1,1)/2/
      DATA TPRID(-2,37)/0/
      DATA (IFOREST(I,-3,37),I=1,2)/1,-2/
      DATA TPRID(-3,37)/21/
      DATA (SPROP(I,-3,37),I=1,1)/0/
      DATA (IFOREST(I,-4,37),I=1,2)/-3,6/
      DATA TPRID(-4,37)/2/
      DATA (SPROP(I,-4,37),I=1,1)/0/
C     Diagram 38
      DATA MAPCONFIG(38)/38/
      DATA TSTRATEGY(38)/2/
      DATA (IFOREST(I,-1,38),I=1,2)/5,4/
      DATA (SPROP(I,-1,38),I=1,1)/21/
      DATA TPRID(-1,38)/0/
      DATA (IFOREST(I,-2,38),I=1,2)/1,-1/
      DATA TPRID(-2,38)/2/
      DATA (SPROP(I,-2,38),I=1,1)/0/
      DATA (IFOREST(I,-3,38),I=1,2)/-2,3/
      DATA TPRID(-3,38)/21/
      DATA (SPROP(I,-3,38),I=1,1)/0/
      DATA (IFOREST(I,-4,38),I=1,2)/-3,6/
      DATA TPRID(-4,38)/2/
      DATA (SPROP(I,-4,38),I=1,1)/0/
C     Diagram 39
      DATA MAPCONFIG(39)/39/
      DATA TSTRATEGY(39)/2/
      DATA (IFOREST(I,-1,39),I=1,2)/4,3/
      DATA (SPROP(I,-1,39),I=1,1)/21/
      DATA TPRID(-1,39)/0/
      DATA (IFOREST(I,-2,39),I=1,2)/6,5/
      DATA (SPROP(I,-2,39),I=1,1)/21/
      DATA TPRID(-2,39)/0/
      DATA (IFOREST(I,-3,39),I=1,2)/1,-1/
      DATA TPRID(-3,39)/2/
      DATA (SPROP(I,-3,39),I=1,1)/0/
      DATA (IFOREST(I,-4,39),I=1,2)/-3,-2/
      DATA TPRID(-4,39)/2/
      DATA (SPROP(I,-4,39),I=1,1)/0/
C     Diagram 40
      DATA MAPCONFIG(40)/40/
      DATA TSTRATEGY(40)/2/
      DATA (IFOREST(I,-1,40),I=1,2)/6,5/
      DATA (SPROP(I,-1,40),I=1,1)/21/
      DATA TPRID(-1,40)/0/
      DATA (IFOREST(I,-2,40),I=1,2)/4,3/
      DATA (SPROP(I,-2,40),I=1,1)/21/
      DATA TPRID(-2,40)/0/
      DATA (IFOREST(I,-3,40),I=1,2)/1,-1/
      DATA TPRID(-3,40)/2/
      DATA (SPROP(I,-3,40),I=1,1)/0/
      DATA (IFOREST(I,-4,40),I=1,2)/-3,-2/
      DATA TPRID(-4,40)/2/
      DATA (SPROP(I,-4,40),I=1,1)/0/
C     Diagram 41
      DATA MAPCONFIG(41)/41/
      DATA TSTRATEGY(41)/2/
      DATA (IFOREST(I,-1,41),I=1,2)/6,3/
      DATA (SPROP(I,-1,41),I=1,1)/21/
      DATA TPRID(-1,41)/0/
      DATA (IFOREST(I,-2,41),I=1,2)/5,4/
      DATA (SPROP(I,-2,41),I=1,1)/21/
      DATA TPRID(-2,41)/0/
      DATA (IFOREST(I,-3,41),I=1,2)/1,-1/
      DATA TPRID(-3,41)/2/
      DATA (SPROP(I,-3,41),I=1,1)/0/
      DATA (IFOREST(I,-4,41),I=1,2)/-3,-2/
      DATA TPRID(-4,41)/2/
      DATA (SPROP(I,-4,41),I=1,1)/0/
C     Diagram 42
      DATA MAPCONFIG(42)/42/
      DATA TSTRATEGY(42)/2/
      DATA (IFOREST(I,-1,42),I=1,2)/5,4/
      DATA (SPROP(I,-1,42),I=1,1)/21/
      DATA TPRID(-1,42)/0/
      DATA (IFOREST(I,-2,42),I=1,2)/6,3/
      DATA (SPROP(I,-2,42),I=1,1)/21/
      DATA TPRID(-2,42)/0/
      DATA (IFOREST(I,-3,42),I=1,2)/1,-1/
      DATA TPRID(-3,42)/2/
      DATA (SPROP(I,-3,42),I=1,1)/0/
      DATA (IFOREST(I,-4,42),I=1,2)/-3,-2/
      DATA TPRID(-4,42)/2/
      DATA (SPROP(I,-4,42),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/42/
""")

        # Test dummy config_subproc_map.inc file
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_config_subproc_map_file(writer, s_and_t_channels)
        writer.close()

        self.assertFileContains('test',
"""      DATA CONFSUB(1,1)/1/
      DATA CONFSUB(1,2)/1/
      DATA CONFSUB(1,3)/1/
      DATA CONFSUB(1,4)/1/
      DATA CONFSUB(1,5)/1/
      DATA CONFSUB(1,6)/1/
      DATA CONFSUB(1,7)/1/
      DATA CONFSUB(1,8)/1/
      DATA CONFSUB(1,9)/1/
      DATA CONFSUB(1,10)/1/
      DATA CONFSUB(1,11)/1/
      DATA CONFSUB(1,12)/1/
      DATA CONFSUB(1,13)/1/
      DATA CONFSUB(1,14)/1/
      DATA CONFSUB(1,15)/1/
      DATA CONFSUB(1,16)/1/
      DATA CONFSUB(1,17)/1/
      DATA CONFSUB(1,18)/1/
      DATA CONFSUB(1,19)/1/
      DATA CONFSUB(1,20)/1/
      DATA CONFSUB(1,21)/1/
      DATA CONFSUB(1,22)/1/
      DATA CONFSUB(1,23)/1/
      DATA CONFSUB(1,24)/1/
      DATA CONFSUB(1,25)/1/
      DATA CONFSUB(1,26)/1/
      DATA CONFSUB(1,27)/1/
      DATA CONFSUB(1,28)/1/
      DATA CONFSUB(1,29)/1/
      DATA CONFSUB(1,30)/1/
      DATA CONFSUB(1,31)/1/
      DATA CONFSUB(1,32)/1/
      DATA CONFSUB(1,33)/1/
      DATA CONFSUB(1,34)/1/
      DATA CONFSUB(1,35)/1/
      DATA CONFSUB(1,36)/1/
      DATA CONFSUB(1,37)/1/
      DATA CONFSUB(1,38)/1/
      DATA CONFSUB(1,39)/1/
      DATA CONFSUB(1,40)/1/
      DATA CONFSUB(1,41)/1/
      DATA CONFSUB(1,42)/1/
""")

        #print open(self.give_pos('test')).read()

        # Test coloramps.inc output
        self.assertEqual("\n".join(\
                       exporter.get_icolamp_lines(mapconfigs,
                                                   matrix_element, 1)),
                         """DATA(icolamp(i,1,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,2,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,3,1),i=1,6)/.false.,.false.,.false.,.true.,.true.,.false./
DATA(icolamp(i,4,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,5,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./
DATA(icolamp(i,6,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.true./
DATA(icolamp(i,7,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,8,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./
DATA(icolamp(i,9,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,10,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,11,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./
DATA(icolamp(i,12,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,13,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.true./
DATA(icolamp(i,14,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,15,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,16,1),i=1,6)/.true.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,17,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,18,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,19,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,20,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./
DATA(icolamp(i,21,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,22,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,23,1),i=1,6)/.true.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,24,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,25,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,26,1),i=1,6)/.false.,.true.,.true.,.false.,.false.,.false./
DATA(icolamp(i,27,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,28,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,29,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,30,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,31,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,32,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,33,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,34,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./
DATA(icolamp(i,35,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.false./
DATA(icolamp(i,36,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,37,1),i=1,6)/.true.,.false.,.false.,.false.,.false.,.false./
DATA(icolamp(i,38,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,39,1),i=1,6)/.false.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,40,1),i=1,6)/.false.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,41,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.false./
DATA(icolamp(i,42,1),i=1,6)/.false.,.false.,.false.,.false.,.false.,.true./"""
)

        # Test get_color.f output
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_colors_file(writer, matrix_element)
        writer.close()
        #print open(self.give_pos('test')).read()

        self.assertFileContains('test',
        """      FUNCTION GET_COLOR(IPDG)
      IMPLICIT NONE
      INTEGER GET_COLOR, IPDG

      IF(IPDG.EQ.-2)THEN
        GET_COLOR=-3
        RETURN
      ELSE IF(IPDG.EQ.2)THEN
        GET_COLOR=3
        RETURN
      ELSE IF(IPDG.EQ.21)THEN
        GET_COLOR=8
        RETURN
      ELSE IF(IPDG.EQ.1)THEN
C       This is dummy particle used in multiparticle vertices
        GET_COLOR=2
        RETURN
      ELSE
        WRITE(*,*)'Error: No color given for pdg ',IPDG
        GET_COLOR=0
        RETURN
      ENDIF
      END

""")

        # Test leshouche.inc output
        writer = writers.FortranWriter(self.give_pos('leshouche'))
        exporter.write_leshouche_file(writer, matrix_element)
        writer.close()

        self.assertFileContains('leshouche',
                         """      DATA (IDUP(I,1,1),I=1,6)/2,-2,2,-2,2,-2/
      DATA (MOTHUP(1,I),I=1, 6)/  0,  0,  1,  1,  1,  1/
      DATA (MOTHUP(2,I),I=1, 6)/  0,  0,  2,  2,  2,  2/
      DATA (ICOLUP(1,I,1,1),I=1, 6)/501,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,1,1),I=1, 6)/  0,501,  0,502,  0,503/
      DATA (ICOLUP(1,I,2,1),I=1, 6)/501,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,2,1),I=1, 6)/  0,501,  0,503,  0,502/
      DATA (ICOLUP(1,I,3,1),I=1, 6)/502,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,3,1),I=1, 6)/  0,501,  0,501,  0,503/
      DATA (ICOLUP(1,I,4,1),I=1, 6)/503,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,4,1),I=1, 6)/  0,501,  0,501,  0,502/
      DATA (ICOLUP(1,I,5,1),I=1, 6)/502,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,5,1),I=1, 6)/  0,501,  0,503,  0,501/
      DATA (ICOLUP(1,I,6,1),I=1, 6)/503,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,6,1),I=1, 6)/  0,501,  0,502,  0,501/
""")

        # Test pdf output (for auto_dsig.f)
        self.assertEqual(exporter.get_pdf_lines(matrix_element, 2),
        ('DOUBLE PRECISION u1\nDOUBLE PRECISION ux2', 'DATA u1/1*1D0/\nDATA ux2/1*1D0/', 'IF (ABS(LPP(1)) .GE. 1) THEN\n!LP=SIGN(1,LPP(1))\nu1=PDG2PDF(LPP(1),2, 1,XBK(1),DSQRT(Q2FACT(1)))\nENDIF\nIF (ABS(LPP(2)) .GE. 1) THEN\n!LP=SIGN(1,LPP(2))\nux2=PDG2PDF(LPP(2),-2, 2,XBK(2),DSQRT(Q2FACT(2)))\nENDIF\nPD(0) = 0d0\nIPROC = 0\nIPROC=IPROC+1 ! u u~ > u u~ u u~\nPD(IPROC)=u1*ux2\nPD(0)=PD(0)+DABS(PD(IPROC))', '')
                         )

        # Test mg.sym
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_mg_sym_file(writer, matrix_element)
        writer.close()
        
        self.assertFileContains('test',
                         """      2
      2
      3
      5
      2
      4
      6
""")

    def test_generate_helas_diagrams_gg_gg(self):
        """Test calls for g g > g g"""

        # Set up local model

        mybasemodel = base_objects.Model()
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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon self-couplings
        my_color_string = color.ColorString([color.f(0, 1, 2)])
        my_color_string.is_imaginary = True
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [my_color_string],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g,
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, -1),
                                                   color.f(2, 3, -1)]),
                                color.ColorString([color.f(2, 0, -1),
                                                   color.f(1, 3, -1)]),
                                color.ColorString([color.f(1, 2, -1),
                                                   color.f(0, 3, -1)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
                      'orders':{'QCD':2}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        # Test Helas calls

        fortran_model = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        self.assertEqual("\n".join(fortran_model.\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL GGGGXX(W(1,1),W(1,2),W(1,3),W(1,4),GG,AMP(1))
CALL GGGGXX(W(1,3),W(1,1),W(1,2),W(1,4),GG,AMP(2))
CALL GGGGXX(W(1,2),W(1,3),W(1,1),W(1,4),GG,AMP(3))
CALL JVVXXX(W(1,1),W(1,2),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,3),W(1,4),W(1,5),GG,AMP(4))
CALL JVVXXX(W(1,1),W(1,3),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,2),W(1,4),W(1,5),GG,AMP(5))
CALL JVVXXX(W(1,1),W(1,4),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,2),W(1,3),W(1,5),GG,AMP(6))""")

        exporter = export_v4.ProcessExporterFortranME()

        numerator = [[19,   -2,   -2,   -2,   -2,    4],
                     [-2,   19,   -2,    4,   -2,   -2],
                     [-2,   -2,   19,   -2,    4,   -2],
                     [-2,    4,   -2,   19,   -2,   -2],
                     [-2,   -2,    4,   -2,   19,   -2],
                     [4,   -2,   -2,   -2,   -2,   19]]
        denom = 6

        i = 0
        for data in exporter.get_color_data_lines(\
                         matrix_element):

            if 'DATA' not in data:
                continue
            if 'denom' in data.lower():
                _, data, _ = data.split('/') 
                self.assertEqual(int(data), denom) 
                continue
            else:
                _, data, _ = data.split('/') 
                number = [int(n) for n in data.split(',') ]
                for j,val in enumerate(number):
                    if j ==0: 
                        self.assertEqual((numerator[i][i+j]), val)
                    else:
                       # factor 2 due to symmetry
                       self.assertEqual((2*numerator[i][i+j]), val) 
                i+=1 

        # Test JAMP (color amplitude) output
        out, nb = exporter.get_JAMP_lines(matrix_element)
        self.assertEqual(nb, 6)
        self.assertEqual(out,
        [' TMP_JAMP(3) = AMP(3) - AMP(6) ! used 4 times', ' TMP_JAMP(2) = AMP(1) - AMP(4) ! used 4 times', ' TMP_JAMP(1) = AMP(2) +  AMP(5) ! used 4 times', ' TMP_JAMP(6) = TMP_JAMP(3) - TMP_JAMP(2) ! used 2 times', ' TMP_JAMP(5) = TMP_JAMP(3) - TMP_JAMP(1) ! used 2 times', ' TMP_JAMP(4) = TMP_JAMP(2) - TMP_JAMP(1) ! used 2 times', ' JAMP(1) = (2.000000000000000d+00)*TMP_JAMP(6)', ' JAMP(2) = (2.000000000000000d+00)*TMP_JAMP(4)', ' JAMP(3) = (-2.000000000000000d+00)*TMP_JAMP(5)', ' JAMP(4) = (2.000000000000000d+00)*TMP_JAMP(4)', ' JAMP(5) = (-2.000000000000000d+00)*TMP_JAMP(5)', ' JAMP(6) = (2.000000000000000d+00)*TMP_JAMP(6)']
        )
        

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(2)=AMP2(2)+AMP(4)*dconjg(AMP(4))',
                          'AMP2(3)=AMP2(3)+AMP(5)*dconjg(AMP(5))',
                          'AMP2(4)=AMP2(4)+AMP(6)*dconjg(AMP(6))'])
        
        # Test configs.inc file
        writer = writers.FortranWriter(self.give_pos('test'))
        nconfig, (s_and_t_channels, nqcd_list) = \
                 exporter.write_configs_file(writer, matrix_element)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""C     Diagram 2
      DATA MAPCONFIG(1)/2/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/21/
      DATA TPRID(-1,1)/0/
C     Diagram 3
      DATA MAPCONFIG(2)/3/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/1,3/
      DATA TPRID(-1,2)/21/
      DATA (SPROP(I,-1,2),I=1,1)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,4/
      DATA TPRID(-2,2)/21/
      DATA (SPROP(I,-2,2),I=1,1)/0/
C     Diagram 4
      DATA MAPCONFIG(3)/4/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/1,4/
      DATA TPRID(-1,3)/21/
      DATA (SPROP(I,-1,3),I=1,1)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
      DATA TPRID(-2,3)/21/
      DATA (SPROP(I,-2,3),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/3/
""")

        # Test dummy config_subproc_map.inc file
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_config_subproc_map_file(writer,
                                               s_and_t_channels)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      DATA CONFSUB(1,1)/1/
      DATA CONFSUB(1,2)/1/
      DATA CONFSUB(1,3)/1/
""")

    def test_generate_helas_diagrams_uu_susu(self):
        """Testing the helas diagram generation u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from the sign! (AMP 1,2)
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,2),W(1,5),W(1,4),MGVX575,AMP(1))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,2),W(1,5),W(1,3),MGVX575,AMP(2))""")

    def test_generate_helas_diagrams_zz_n1n1(self):
        """Testing the helas diagram generation z z > n1 n1 with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),zmas,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zmas,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),Mneu1,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mneu1,NHEL(4),-1*IC(4),W(1,4))
CALL FVOXXX(W(1,3),W(1,1),GZN11,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,4),W(1,5),W(1,2),GZN11,AMP(1))
CALL FVIXXX(W(1,4),W(1,1),GZN11,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,5),W(1,3),W(1,2),GZN11,AMP(2))""")

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(matrix_element)[0][0],
                         " JAMP(1) = (-1.000000000000000d+00)*AMP(1)+(-1.000000000000000d+00)*AMP(2)")

    def test_generate_helas_diagrams_gb_t1go_tttxn1x1m(self):
        """Testing the helas diagram generation g b > t1 go > t t t~ n1 x1-
        This diagrams get for mssm_v4 an inconsistency in the matrix.f in version
        lower than 1.5.7. This specific point is automatically check by the
        optimization routine.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        
        mypartlist.append(base_objects.Particle({
                    'name': 'g',
                    'antiname': 'g',
                    'spin': 3,
                    'color': 8,
                    'charge': 0.00,
                    'mass': 'ZERO',
                    'width': 'ZERO',
                    'pdg_code': 21,
                    'texname': '_',
                    'antitexname': '_',
                    'line': 'curly',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': True
                    }))
        p_g = mypartlist[-1]
                        
        mypartlist.append(base_objects.Particle({
                    'name': 'b',
                    'antiname': 'b~',
                    'spin': 2,
                    'color': 3,
                    'charge': 0.00,
                    'mass': 'BMASS',
                    'width': 'ZERO',
                    'pdg_code': 5,
                    'texname': 'b',
                    'antitexname': 'b',
                    'line': 'straight',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': False
                }))
        p_b = mypartlist[-1]
        p_bx = copy.copy(p_b)
        p_bx.set('is_part', False)
        
        mypartlist.append(base_objects.Particle({
                    'name': 't1',
                    'antiname': 't1~',
                    'spin': 1,
                    'color': 3,
                    'charge': 0.00,
                    'mass': 'MT1',
                    'width': 'WT1',
                    'pdg_code': 1000006,
                    'texname': 't1',
                    'antitexname': 't1',
                    'line': 'dashed',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': False
                }))
        p_t1 = mypartlist[-1]
        p_t1x = copy.copy(p_t1)
        p_t1x.set('is_part', False)
         
        mypartlist.append(base_objects.Particle({
                    'name': 'go',
                    'antiname': 'go',
                    'spin': 2,
                    'color': 8,
                    'charge': 0.00,
                    'mass': 'MGO',
                    'width': 'WGO',
                    'pdg_code': 1000021,
                    'texname': 'go',
                    'antitexname': 'go',
                    'line': 'straight',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': True
                }))
        p_go = mypartlist[-1]
                 
        mypartlist.append(base_objects.Particle({
                    'name': 't',
                    'antiname': 't~',
                    'spin': 2,
                    'color': 3,
                    'charge': 0.00,
                    'mass': 'TMASS',
                    'width': 'TWIDTH',
                    'pdg_code': 6,
                    'texname': 't',
                    'antitexname': 't',
                    'line': 'straight',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': False
                }))
        p_t = mypartlist[-1]
        p_tx = copy.copy(p_t)
        p_tx.set('is_part', False) 
        
        mypartlist.append(base_objects.Particle({
                    'name': 'n1',
                    'antiname': 'n1',
                    'spin': 2,
                    'color': 1,
                    'charge': 0.00,
                    'mass': 'MN1',
                    'width': 'WN1',
                    'pdg_code': 1000022,
                    'texname': 'N1',
                    'antitexname': 'N1',
                    'line': 'straight',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': True
                })) 
        p_n1 = mypartlist[-1]

        mypartlist.append(base_objects.Particle({
                    'name': 'x1-',
                    'antiname': 'x1+',
                    'spin': 2,
                    'color': 1,
                    'charge': 0.00,
                    'mass': 'MX1',
                    'width': 'WX1',
                    'pdg_code': -1000024,
                    'texname': 'X1',
                    'antitexname': 'X1',
                    'line': 'straight',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': False
                }))
        p_x1m = mypartlist[-1]
        p_x1p = copy.copy(p_x1m)
        p_x1p.set('is_part', False) 
        
        mypartlist.append(base_objects.Particle({
                    'name': 'b1',
                    'antiname': 'b1~',
                    'spin': 1,
                    'color': 3,
                    'charge': 0.00,
                    'mass': 'MB1',
                    'width': 'WB1',
                    'pdg_code': 1000005,
                    'texname': 'b1',
                    'antitexname': 'b1',
                    'line': 'dashed',
                    'propagating': True,
                    'is_part': True,
                    'self_antipart': False
                }))
        p_b1 = mypartlist[-1]
        p_b1x = copy.copy(p_b1)
        p_b1x.set('is_part', False)


        myinterlist.append(base_objects.Interaction({
                    'id': 5,
                    'particles': base_objects.ParticleList([p_bx, p_b, p_g]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GG'},
                    'orders': {'QCD': 1}}))
        
        myinterlist.append(base_objects.Interaction({'id': 362,
                    'particles': base_objects.ParticleList([p_bx, p_x1m, p_t1]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1X1M'},
                    'orders': {'QED': 1}}))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 363,
                    'particles': base_objects.ParticleList([p_x1p, p_b, p_t1x]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1X1P'},
                    'orders': {'QED': 1}}))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 109,
                    'particles': base_objects.ParticleList([p_go, p_t, p_t1x]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1GOP'},
                    'orders': {'QCD': 1}}))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 108,
                    'particles': base_objects.ParticleList([p_tx, p_go, p_t1]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1GOM'},
                    'orders': {'QCD': 1}}))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 250,
                    'particles': base_objects.ParticleList([p_tx, p_b1, p_t1]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1N1M'},
                    'orders': {'QED': 1}}))
        
        myinterlist.append(base_objects.Interaction({
                    'id': 251,
                    'particles': base_objects.ParticleList([p_n1, p_t, p_t1x]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GT1N1P'},
                    'orders': {'QED': 1}}))
        myinterlist.append(base_objects.Interaction({
                    'id': 105,
                    'particles': base_objects.ParticleList([p_go, p_b, p_b1x]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GB1GOP'},
                    'orders': {'QCD': 1}}))
        myinterlist.append(base_objects.Interaction({
                    'id': 104,
                    'particles': base_objects.ParticleList([p_bx, p_go, p_b1]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GB1GOM'},
                    'orders': {'QCD': 1}}))
        myinterlist.append(base_objects.Interaction({
                    'id': 354,
                    'particles': base_objects.ParticleList([p_tx, p_x1p, p_b1]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GB1X1M'},
                    'orders': {'QED': 1}}))
        myinterlist.append(base_objects.Interaction({
                    'id': 355,
                    'particles': base_objects.ParticleList([p_x1m, p_t, p_b1x]),
                    'color': [],
                    'lorentz': [''],
                    'couplings': {(0, 0): 'GB1X1P'},
                    'orders': {'QED': 1}}))
        
        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':5,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000024,
                                         'state':True}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        self.assertEqual(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element),
                                   """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),BMASS,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),TMASS,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),TMASS,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),TMASS,NHEL(5),-1*IC(5),W(1,5))
CALL IXXXXX(P(0,6),MN1,NHEL(6),-1*IC(6),W(1,6))
CALL OXXXXX(P(0,7),MX1,NHEL(7),+1*IC(7),W(1,7))
CALL FVIXXX(W(1,2),W(1,1),GG,BMASS,ZERO,W(1,8))
CALL HIOXXX(W(1,6),W(1,3),GT1N1P,MT1,WT1,W(1,2))
CALL HIOXXX(W(1,8),W(1,7),GT1X1M,MT1,WT1,W(1,9))
CALL FSIXXX(W(1,5),W(1,2),GT1GOM,MGO,WGO,W(1,7))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,4),W(1,9),GT1GOP,AMP(1))
CALL OXXXXX(P(0,5),TMASS,NHEL(5),+1*IC(5),W(1,7))
CALL IXXXXX(P(0,7),MX1,NHEL(7),-1*IC(7),W(1,10))
CALL HIOCXX(W(1,10),W(1,4),GB1X1P,MB1,WB1,W(1,11))
CALL FSIXXX(W(1,8),W(1,2),GT1X1M,MX1,WX1,W(1,12))
# Amplitude(s) for diagram number 2
CALL IOSCXX(W(1,12),W(1,7),W(1,11),GB1X1M,AMP(2))
CALL OXXXXX(P(0,2),BMASS,NHEL(2),-1*IC(2),W(1,12))
CALL FVOCXX(W(1,12),W(1,1),GG,BMASS,ZERO,W(1,13))
CALL FSOCXX(W(1,13),W(1,11),GB1GOM,MGO,WGO,W(1,12))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,5),W(1,12),W(1,2),GT1GOM,AMP(3))
CALL HIOCXX(W(1,10),W(1,3),GB1X1P,MB1,WB1,W(1,12))
CALL HIOXXX(W(1,6),W(1,4),GT1N1P,MT1,WT1,W(1,10))
CALL FSOCXX(W(1,13),W(1,12),GB1GOM,MGO,WGO,W(1,6))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,5),W(1,6),W(1,10),GT1GOM,AMP(4))
CALL FSIXXX(W(1,8),W(1,10),GT1X1M,MX1,WX1,W(1,6))
# Amplitude(s) for diagram number 5
CALL IOSCXX(W(1,6),W(1,7),W(1,12),GB1X1M,AMP(5))
CALL FSIXXX(W(1,5),W(1,10),GT1GOM,MGO,WGO,W(1,6))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,6),W(1,3),W(1,9),GT1GOP,AMP(6))""".split('\n'))
        
    
    def test_generate_helas_diagrams_epem_elpelmepem(self):
        """Testing the helas diagram generation e+ e- > sl2+ sl2- e+ e-
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl2-',
                      'antiname':'sl2+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist) - 1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':r'\chi_0^1',
                      'antitexname':r'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        #print myamplitude.get('process').nice_string()
        #print "\n".join(helas_call_writers.FortranHelasCallWriter().\
        #                get_matrix_element_calls(matrix_element))
        #print helas_call_writers.FortranHelasCallWriter().get_JAMP_line(matrix_element)


        # I have checked that the resulting Helas calls below give
        # identical result as MG4 (when fermionfactors are taken into
        # account)
        self.assertEqual(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element),
        """CALL OXXXXX(P(0,1),me,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),me,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL OXXXXX(P(0,6),me,NHEL(6),+1*IC(6),W(1,6))
CALL FSOXXX(W(1,1),W(1,3),MGVX350,Mneu1,Wneu1,W(1,7))
CALL FSIXXX(W(1,2),W(1,4),MGVX494,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,5),W(1,7),MGVX494,Msl2,Wsl2,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,8),W(1,6),W(1,9),MGVX350,AMP(1))
CALL OXXXXX(P(0,2),me,NHEL(2),-1*IC(2),W(1,9))
CALL FSOCXX(W(1,9),W(1,4),MGVX494,Mneu1,Wneu1,W(1,10))
CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,9))
CALL FSICXX(W(1,9),W(1,3),MGVX350,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,11),W(1,6),MGVX350,Msl2,Wsl2,W(1,9))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,5),W(1,10),W(1,9),MGVX494,AMP(2))
CALL FSIXXX(W(1,5),W(1,4),MGVX494,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,2),W(1,7),MGVX494,Msl2,Wsl2,W(1,12))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,11),W(1,6),W(1,12),MGVX350,AMP(3))
CALL OXXXXX(P(0,5),me,NHEL(5),+1*IC(5),W(1,12))
CALL FSOCXX(W(1,12),W(1,4),MGVX494,Mneu1,Wneu1,W(1,7))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,2),W(1,7),W(1,9),MGVX494,AMP(4))
CALL FSOXXX(W(1,6),W(1,3),MGVX350,Mneu1,Wneu1,W(1,9))
CALL HIOXXX(W(1,8),W(1,1),MGVX350,Msl2,Wsl2,W(1,6))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,5),W(1,9),W(1,6),MGVX494,AMP(5))
CALL IXXXXX(P(0,6),me,NHEL(6),-1*IC(6),W(1,6))
CALL FSICXX(W(1,6),W(1,3),MGVX350,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,8),W(1,1),MGVX350,Msl2,Wsl2,W(1,6))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,5),W(1,10),W(1,6),MGVX494,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,2),W(1,7),W(1,6),MGVX494,AMP(7))
CALL HIOXXX(W(1,11),W(1,1),MGVX350,Msl2,Wsl2,W(1,6))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,2),W(1,9),W(1,6),MGVX494,AMP(8))""".split('\n'))

        # Test find_outgoing_number
        goal_numbers = [1, 2, 3, 1, 2, 3, 2, 3, 1, 1, 3, 2, 3, 3]

        i = 0
        for wf in matrix_element.get_all_wavefunctions():
            if not wf.get('interaction_id'):
                continue
            self.assertEqual(wf.find_outgoing_number(), goal_numbers[i])
            i += 1
        # Test get_used_lorentz
        # Wavefunctions
        goal_lorentz_list = [(('',), (), 1), (('',), (), 2), (('',), (), 3), 
                             (('',), ('C1',), 1), (('',), ('C1',), 2), (('',), (), 3), 
                             (('',), (), 2), (('',), (), 3), (('',), ('C1',), 1), 
                             (('',), (), 1), (('',), (), 3), (('',), ('C1',), 2), 
                             (('',), (), 3), (('',), (), 3)]
        
        # Amplitudes
        goal_lorentz_list += [(('',), (), 0)] * 8
        self.assertEqual(matrix_element.get_used_lorentz(),
                         goal_lorentz_list)


    def test_generate_helas_diagrams_uu_susug(self):
        """Testing the helas diagram generation u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from sign! (AMP 1,2,5,6)
        self.assertEqual(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,6))
CALL FVIXXX(W(1,2),W(1,5),GG,mu,zero,W(1,7))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,6),W(1,4),MGVX575,AMP(1))
CALL HVSXXX(W(1,5),W(1,4),MGVX74,Musq2,Wusq2,W(1,8))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,2),W(1,6),W(1,8),MGVX575,AMP(2))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,6))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,7),W(1,6),W(1,3),MGVX575,AMP(3))
CALL HVSXXX(W(1,5),W(1,3),MGVX74,Musq2,Wusq2,W(1,7))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,2),W(1,6),W(1,7),MGVX575,AMP(4))
CALL FVOCXX(W(1,1),W(1,5),GG,mu,zero,W(1,6))
CALL FSIXXX(W(1,2),W(1,3),MGVX575,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 5
CALL IOSCXX(W(1,5),W(1,6),W(1,4),MGVX575,AMP(5))
CALL FSIXXX(W(1,2),W(1,4),MGVX575,Mneu1,Wneu1,W(1,9))
# Amplitude(s) for diagram number 6
CALL IOSCXX(W(1,9),W(1,6),W(1,3),MGVX575,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOSCXX(W(1,5),W(1,1),W(1,8),MGVX575,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSCXX(W(1,9),W(1,1),W(1,7),MGVX575,AMP(8))""".split('\n'))

    def test_generate_helas_diagrams_enu_enu(self):
        """Testing the helas diagram generation e- nubar > e- nubar
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A neutrino
        mypartlist.append(base_objects.Particle({'name':'ve',
                      'antiname':'ve~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\nu_e',
                      'antitexname':'\bar\nu_e',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':12,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        nu = mypartlist[len(mypartlist) - 1]
        nubar = copy.copy(nu)
        nubar.set('is_part', False)

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

        # Coupling of W- e+ nu_e

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             nu, \
                                             Wminus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [nubar, \
                                             eminus, \
                                             Wplus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),me,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL JIOXXX(W(1,1),W(1,2),MGVX27,MW,WW,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,4),W(1,3),W(1,5),MGVX27,AMP(1))""")

    def test_generate_helas_diagrams_WWWW(self):
        """Testing the helas diagram generation W+ W- > W+ W-
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Z
        mypartlist.append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Wplus,
                                             Wminus]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX6'},
            'orders':{'QED':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.  Note that this looks like it uses
        # incoming bosons instead of outgoing though
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL W3W3NX(W(1,2),W(1,1),W(1,3),W(1,4),MGVX6,DUM0,AMP(1))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,3),W(1,4),W(1,5),MGVX3,AMP(2))
CALL JVVXXX(W(1,2),W(1,1),MGVX5,MZ,WZ,W(1,5))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,3),W(1,4),W(1,5),MGVX5,AMP(3))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,5))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,2),W(1,4),W(1,5),MGVX3,AMP(4))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MZ,WZ,W(1,5))
# Amplitude(s) for diagram number 5
CALL VVVXXX(W(1,2),W(1,4),W(1,5),MGVX5,AMP(5))""")

    def test_generate_helas_diagrams_WWZA(self):
        """Testing the helas diagram generation W+ W- > Z A
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Z
        mypartlist.append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Wplus,
                                             Wminus]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX6'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX4'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX7'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX8'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MZ,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL W3W3NX(W(1,2),W(1,4),W(1,1),W(1,3),MGVX7,DUM0,AMP(1))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MW,WW,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,2),W(1,5),W(1,4),MGVX3,AMP(2))
CALL JVVXXX(W(1,1),W(1,4),MGVX3,MW,WW,W(1,5))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,5),W(1,2),W(1,3),MGVX5,AMP(3))""")


    def test_generate_helas_diagrams_WWWWA(self):
        """Testing the helas diagram generation W+ W- > W+ W- a
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Z
        mypartlist.append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Wplus,
                                             Wminus]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX6'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX4'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX7'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX8'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        #print myamplitude.get('process').nice_string()
        #print myamplitude.get('diagrams').nice_string()

        #print "Keys:"
        #for diagram in matrix_element.get('diagrams'):
        #    for wf in diagram.get('wavefunctions'):
        #        print wf.get_call_key()
        #    print diagram.get('amplitude').get_call_key()

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.
        self.assertEqual(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,6))
CALL JVVXXX(W(1,5),W(1,3),MGVX3,MW,WW,W(1,7))
# Amplitude(s) for diagram number 1
CALL VVVXXX(W(1,7),W(1,4),W(1,6),MGVX3,AMP(1))
CALL JVVXXX(W(1,1),W(1,2),MGVX5,MZ,WZ,W(1,8))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,4),W(1,7),W(1,8),MGVX5,AMP(2))
CALL JVVXXX(W(1,4),W(1,5),MGVX3,MW,WW,W(1,9))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,3),W(1,9),W(1,6),MGVX3,AMP(3))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,9),W(1,3),W(1,8),MGVX5,AMP(4))
# Amplitude(s) for diagram number 5
CALL W3W3NX(W(1,3),W(1,5),W(1,4),W(1,6),MGVX4,DUM0,AMP(5))
# Amplitude(s) for diagram number 6
CALL W3W3NX(W(1,3),W(1,5),W(1,4),W(1,8),MGVX7,DUM0,AMP(6))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,8))
CALL JVVXXX(W(1,5),W(1,2),MGVX3,MW,WW,W(1,6))
# Amplitude(s) for diagram number 7
CALL VVVXXX(W(1,6),W(1,4),W(1,8),MGVX3,AMP(7))
CALL JVVXXX(W(1,1),W(1,3),MGVX5,MZ,WZ,W(1,10))
# Amplitude(s) for diagram number 8
CALL VVVXXX(W(1,4),W(1,6),W(1,10),MGVX5,AMP(8))
# Amplitude(s) for diagram number 9
CALL VVVXXX(W(1,2),W(1,9),W(1,8),MGVX3,AMP(9))
# Amplitude(s) for diagram number 10
CALL VVVXXX(W(1,9),W(1,2),W(1,10),MGVX5,AMP(10))
# Amplitude(s) for diagram number 11
CALL W3W3NX(W(1,2),W(1,5),W(1,4),W(1,8),MGVX4,DUM0,AMP(11))
# Amplitude(s) for diagram number 12
CALL W3W3NX(W(1,2),W(1,5),W(1,4),W(1,10),MGVX7,DUM0,AMP(12))
CALL JVVXXX(W(1,1),W(1,5),MGVX3,MW,WW,W(1,10))
CALL JVVXXX(W(1,2),W(1,4),MGVX3,zero,zero,W(1,8))
# Amplitude(s) for diagram number 13
CALL VVVXXX(W(1,3),W(1,10),W(1,8),MGVX3,AMP(13))
CALL JVVXXX(W(1,4),W(1,2),MGVX5,MZ,WZ,W(1,9))
# Amplitude(s) for diagram number 14
CALL VVVXXX(W(1,10),W(1,3),W(1,9),MGVX5,AMP(14))
CALL JVVXXX(W(1,3),W(1,4),MGVX3,zero,zero,W(1,11))
# Amplitude(s) for diagram number 15
CALL VVVXXX(W(1,2),W(1,10),W(1,11),MGVX3,AMP(15))
CALL JVVXXX(W(1,4),W(1,3),MGVX5,MZ,WZ,W(1,12))
# Amplitude(s) for diagram number 16
CALL VVVXXX(W(1,10),W(1,2),W(1,12),MGVX5,AMP(16))
# Amplitude(s) for diagram number 17
CALL W3W3NX(W(1,2),W(1,4),W(1,3),W(1,10),MGVX6,DUM0,AMP(17))
# Amplitude(s) for diagram number 18
CALL VVVXXX(W(1,7),W(1,1),W(1,8),MGVX3,AMP(18))
# Amplitude(s) for diagram number 19
CALL VVVXXX(W(1,1),W(1,7),W(1,9),MGVX5,AMP(19))
# Amplitude(s) for diagram number 20
CALL VVVXXX(W(1,6),W(1,1),W(1,11),MGVX3,AMP(20))
# Amplitude(s) for diagram number 21
CALL VVVXXX(W(1,1),W(1,6),W(1,12),MGVX5,AMP(21))
CALL JW3WNX(W(1,3),W(1,1),W(1,2),MGVX6,DUM0,MW,WW,W(1,12))
# Amplitude(s) for diagram number 22
CALL VVVXXX(W(1,12),W(1,4),W(1,5),MGVX3,AMP(22))
CALL JW3WNX(W(1,1),W(1,2),W(1,4),MGVX6,DUM0,MW,WW,W(1,12))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,3),W(1,12),W(1,5),MGVX3,AMP(23))
CALL JW3WNX(W(1,1),W(1,5),W(1,2),MGVX4,DUM0,zero,zero,W(1,12))
# Amplitude(s) for diagram number 24
CALL VVVXXX(W(1,3),W(1,4),W(1,12),MGVX3,AMP(24))
CALL JW3WNX(W(1,2),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,12))
# Amplitude(s) for diagram number 25
CALL VVVXXX(W(1,4),W(1,3),W(1,12),MGVX5,AMP(25))
CALL JW3WNX(W(1,1),W(1,3),W(1,4),MGVX6,DUM0,MW,WW,W(1,12))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,2),W(1,12),W(1,5),MGVX3,AMP(26))
CALL JW3WNX(W(1,1),W(1,5),W(1,3),MGVX4,DUM0,zero,zero,W(1,12))
# Amplitude(s) for diagram number 27
CALL VVVXXX(W(1,2),W(1,4),W(1,12),MGVX3,AMP(27))
CALL JW3WNX(W(1,3),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,12))
# Amplitude(s) for diagram number 28
CALL VVVXXX(W(1,4),W(1,2),W(1,12),MGVX5,AMP(28))""".split('\n'))

    def test_helas_diagrams_gg_gogo_go_tt1x_t_wpb(self):
        """Testing g g > go go, (go > t t1~, t > w+ b)
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon and gluino
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

        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':8,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        go = mypartlist[-1]

        # A top quark and stop squark
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
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

        mypartlist.append(base_objects.Particle({'name':'t1',
                      'antiname':'t1~',
                      'spin':0,
                      'color':3,
                      'mass':'MT1',
                      'width':'WT1',
                      'texname':'t1',
                      'antitexname':'\bar t1',
                      'line':'dashed',
                      'charge':2. / 3.,
                      'pdg_code':1000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t1 = mypartlist[len(mypartlist) - 1]
        antit1 = copy.copy(t1)
        antit1.set('is_part', False)

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

        # b quark
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        # top-w-b coupling
        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([antib, t, Wminus]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFV2'],
            'couplings': {(0, 0): 'GC_108'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([antit, b, Wplus]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFV2'],
            'couplings': {(0, 0): 'GC_108'},
            'orders': {'QED': 1}
            }))

        # Gluon couplings to gluino
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

        # Gluino couplings to top and stop
        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             t, \
                                             antit1]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             go, \
                                             t1]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021}))
        myleglist.append(base_objects.Leg({'id':1000021}))

        core_proc = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.append(base_objects.Leg({'id':-1000006}))

        decay1_process = base_objects.Process({'legs':myleglist,
                                               'model':mybasemodel,
                                               'is_decay_chain': True})

        core_proc.get('decay_chains').append(decay1_process)
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':24}))

        decay2_process = base_objects.Process({'legs':myleglist,
                                               'model':mybasemodel,
                                               'is_decay_chain': True})

        decay1_process.get('decay_chains').append(decay2_process)

        amplitude = diagram_generation.DecayChainAmplitude(core_proc)

        self.assertEqual(len(amplitude.get('amplitudes')),1)

        matrix_elements = helas_objects.HelasDecayChainProcess(amplitude)

        matrix_element = matrix_elements.combine_decay_chain_processes()[0]

        #print "\n".join(helas_call_writers.FortranUFOHelasCallWriter().\
        #                get_matrix_element_calls(matrix_element))

        self.assertEqual("\n".join(helas_call_writers.FortranUFOHelasCallWriter().\
                                   get_matrix_element_calls(matrix_element)).split('\n'),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),+1*IC(4),W(1,4))
CALL FFV2_1(W(1,3),W(1,4),GC_108,MT,WT,W(1,5))
CALL SXXXXX(P(0,5),+1*IC(5),W(1,4))
CALL L1_1(W(1,5),W(1,4),GQQ,MGO,WGO,W(1,3))
CALL IXXXXX(P(0,6),zero,NHEL(6),-1*IC(6),W(1,4))
CALL VXXXXX(P(0,7),MW,NHEL(7),+1*IC(7),W(1,5))
CALL FFV2C1_2(W(1,4),W(1,5),GC_108,MT,WT,W(1,6))
CALL SXXXXX(P(0,8),+1*IC(8),W(1,5))
CALL L1C1_2(W(1,6),W(1,5),GQQ,MGO,WGO,W(1,4))
CALL L1_1(W(1,3),W(1,1),GQQ,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 1
CALL L1_0(W(1,4),W(1,5),W(1,2),GQQ,AMP(1))
CALL L1_2(W(1,4),W(1,1),-GQQ,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 2
CALL L1_0(W(1,5),W(1,3),W(1,2),GQQ,AMP(2))""".split('\n'))

        # Test get_used_lorentz
        goal_lorentz_list = [(('FFV2',), (), 1), (('L1',), (), 1), 
                             (('FFV2',), ('C1',), 2), (('L1',), ('C1',), 2), 
                             (('L1',), (), 1), (('L1',), (), 2), 
                             (('L1',), (), 0), (('L1',), (), 0)]

        self.assertEqual(matrix_element.get_used_lorentz(),
                         goal_lorentz_list)

    def test_multiple_lorentz_structures(self):
        """Testing multiple Lorentz structures for one diagram.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # An A particle
        mypartlist.append(base_objects.Particle({'name':'A',
                      'antiname':'A',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'A',
                      'antitexname':'A',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':45,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        A = mypartlist[len(mypartlist) - 1]

        # A particle self-couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [A, \
                                             A, \
                                             A]),
                      'color': [],
                      'lorentz':['CL1', 'L2'],
                      'couplings':{(0, 0):'G1', (0, 1):'G2'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':45,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, gen_color=False)

        myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel)
        self.assertEqual(myfortranmodel.get_matrix_element_calls(matrix_element),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL CL1_L2_1(W(1,1),W(1,2),G1,G2,zero,zero,W(1,5))
# Amplitude(s) for diagram number 1
CALL CL1_L2_0(W(1,3),W(1,4),W(1,5),G1,G2,AMP(1))
CALL CL1_L2_1(W(1,1),W(1,3),G1,G2,zero,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL CL1_L2_0(W(1,2),W(1,4),W(1,5),G1,G2,AMP(2))
CALL CL1_L2_1(W(1,1),W(1,4),G1,G2,zero,zero,W(1,5))
# Amplitude(s) for diagram number 3
CALL CL1_L2_0(W(1,2),W(1,3),W(1,5),G1,G2,AMP(3))""".split('\n'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))'])

        # Test configs.inc file
        writer = writers.FortranWriter(self.give_pos('test'))
        nconfig, (s_and_t_channels, nqcd_list) = \
                 exporter.write_configs_file(writer, matrix_element)
        writer.close()
        #print open(self.give_pos('test')).read()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/45/
      DATA TPRID(-1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/1,3/
      DATA TPRID(-1,2)/45/
      DATA (SPROP(I,-1,2),I=1,1)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,4/
      DATA TPRID(-2,2)/45/
      DATA (SPROP(I,-2,2),I=1,1)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/1,4/
      DATA TPRID(-1,3)/45/
      DATA (SPROP(I,-1,3),I=1,1)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
      DATA TPRID(-2,3)/45/
      DATA (SPROP(I,-2,3),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/3/
""")

    def test_multiple_lorentz_structures_with_fermion_flow_clash(self):
        """Testing process w+ w+ > z x1+ x1+.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # Z particle
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        z = mypartlist[len(mypartlist) - 1]

        # W particle
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W',
                      'antitexname':'W',
                      'line':'curly',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # n1 particle
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':'n1',
                      'antitexname':'n1',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        n1 = mypartlist[len(mypartlist) - 1]

        # x1+ particle
        mypartlist.append(base_objects.Particle({'name':'x1+',
                      'antiname':'x1-',
                      'spin':2,
                      'color':1,
                      'mass':'Mx1p',
                      'width':'Wx1p',
                      'texname':'x1+',
                      'antitexname':'x1-',
                      'line':'curly',
                      'charge':1.,
                      'pdg_code':1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        x1plus = mypartlist[len(mypartlist) - 1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([wminus,wplus,z]),
            'color': [],
            'lorentz': ['VVV1'],
            'couplings': {(0, 0): 'GC_214'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([n1,x1plus,wminus]),
            'color': [],
            'lorentz': ['FFV2', 'FFV3'],
            'couplings': {(0, 1): 'GC_628', (0, 0): 'GC_422'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList([n1,x1minus,wplus]),
            'color': [],
            'lorentz': ['FFV2', 'FFV3'],
            'couplings': {(0, 1): 'GC_628', (0, 0): 'GC_422'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList([n1,n1,z]),
            'color': [],
            'lorentz': ['FFV5'],
            'couplings': {(0, 0): 'GC_418'},
            'orders': {'QED': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 6)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, gen_color=False)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MZ,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mx1p,NHEL(4),-1*IC(4),W(1,4))
CALL OXXXXX(P(0,5),Mx1p,NHEL(5),+1*IC(5),W(1,5))
CALL VVV1_2(W(1,1),W(1,3),GC_214,MW,WW,W(1,6))
CALL FFV2_3C1_2(W(1,4),W(1,2),GC_422,GC_628,Mneu1,Wneu1,W(1,7))
# Amplitude(s) for diagram number 1
CALL FFV2_3_0(W(1,7),W(1,5),W(1,6),GC_422,GC_628,AMP(1))
CALL FFV2_3_1(W(1,5),W(1,2),GC_422,GC_628,Mneu1,Wneu1,W(1,8))
# Amplitude(s) for diagram number 2
CALL FFV2_3C1_0(W(1,4),W(1,8),W(1,6),GC_422,GC_628,AMP(2))
CALL FFV2_3C1_2(W(1,4),W(1,1),GC_422,GC_628,Mneu1,Wneu1,W(1,6))
CALL VVV1_2(W(1,2),W(1,3),GC_214,MW,WW,W(1,9))
# Amplitude(s) for diagram number 3
CALL FFV2_3_0(W(1,6),W(1,5),W(1,9),GC_422,GC_628,AMP(3))
# Amplitude(s) for diagram number 4
CALL FFV5_0(W(1,6),W(1,8),W(1,3),GC_418,AMP(4))
CALL FFV2_3_1(W(1,5),W(1,1),GC_422,GC_628,Mneu1,Wneu1,W(1,6))
# Amplitude(s) for diagram number 5
CALL FFV2_3C1_0(W(1,4),W(1,6),W(1,9),GC_422,GC_628,AMP(5))
# Amplitude(s) for diagram number 6
CALL FFV5_0(W(1,7),W(1,6),W(1,3),GC_418,AMP(6))""".split('\n') #end 7, 11,3

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))',
                          'AMP2(5)=AMP2(5)+AMP(5)*dconjg(AMP(5))',
                          'AMP2(6)=AMP2(6)+AMP(6)*dconjg(AMP(6))'])

    def test_four_fermion_vertex_normal_fermion_flow(self):
        """Testing process u u > t t g with fermion flow (u~t)(u~t)
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A t quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Four fermion vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antiu,
                                             t,
                                             antiu,
                                             t]),
                      'color': [color.ColorString([color.T(1, 0),
                                                   color.T(3, 2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0, 0):'GEFF'},
                      'orders':{'NP':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 4)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),zero,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FFV1_2(W(1,1),W(1,5),GG,zero,zero,W(1,6))
# Amplitude(s) for diagram number 1
CALL FFFF1_0(W(1,2),W(1,3),W(1,6),W(1,4),GEFF,AMP(1))
CALL FFFF1_2(W(1,1),W(1,2),W(1,3),GEFF,MT,WT,W(1,6))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,6),W(1,4),W(1,5),GG,AMP(2))
CALL FFFF1_2(W(1,1),W(1,2),W(1,4),GEFF,MT,WT,W(1,6))
# Amplitude(s) for diagram number 3
CALL FFV1_0(W(1,6),W(1,3),W(1,5),GG,AMP(3))
CALL FFFF1_1(W(1,3),W(1,1),W(1,4),GEFF,zero,zero,W(1,6))
# Amplitude(s) for diagram number 4
CALL FFV1_0(W(1,2),W(1,6),W(1,5),GG,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))'])

        # Check fermion factors
        self.assertEqual([d.get('amplitudes')[0].get('fermionfactor') \
                          for d in matrix_element.get('diagrams')],
                         [1, 1, -1, 1])
        
        writer = writers.FortranWriter(self.give_pos('test'))
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        # Test configs.inc file
        nconfig, (s_and_t_channels, nqcd_list) = \
                      exporter.write_configs_file(writer, matrix_element)
        writer.close()

        #print( open(self.give_pos('test')).read())
    
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/1/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/1,5/
      DATA TPRID(-2,1)/2/
      DATA (SPROP(I,-2,1),I=1,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/-2,-1/
      DATA TPRID(-3,1)/2/
      DATA (SPROP(I,-3,1),I=1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,4/
      DATA (SPROP(I,-1,2),I=1,1)/6/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,3/
      DATA (SPROP(I,-2,2),I=1,1)/1/
      DATA TPRID(-2,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/5,3/
      DATA (SPROP(I,-1,3),I=1,1)/6/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/4,-1/
      DATA (SPROP(I,-2,3),I=1,1)/1/
      DATA TPRID(-2,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/4,3/
      DATA (SPROP(I,-1,4),I=1,1)/1/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/1,-1/
      DATA TPRID(-2,4)/2/
      DATA (SPROP(I,-2,4),I=1,1)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,5/
      DATA TPRID(-3,4)/2/
      DATA (SPROP(I,-3,4),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/4/
""")

        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_props_file(writer, matrix_element,
                                   s_and_t_channels)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      PRMASS(-1,1)  = ZERO
      PRWIDTH(-1,1) = ZERO
      POW(-1,1) = 0
      PRMASS(-2,1)  = ZERO
      PRWIDTH(-2,1) = ZERO
      POW(-2,1) = 1
      PRMASS(-1,2)  = ABS(MT)
      PRWIDTH(-1,2) = ABS(WT)
      POW(-1,2) = 1
      PRMASS(-2,2)  = ZERO
      PRWIDTH(-2,2) = ZERO
      POW(-2,2) = 0
      PRMASS(-1,3)  = ABS(MT)
      PRWIDTH(-1,3) = ABS(WT)
      POW(-1,3) = 1
      PRMASS(-2,3)  = ZERO
      PRWIDTH(-2,3) = ZERO
      POW(-2,3) = 0
      PRMASS(-1,4)  = ZERO
      PRWIDTH(-1,4) = ZERO
      POW(-1,4) = 0
      PRMASS(-2,4)  = ZERO
      PRWIDTH(-2,4) = ZERO
      POW(-2,4) = 1
""")


    def test_four_fermion_vertex_strange_fermion_flow(self):
        """Testing process u u > t t g with fermion flow (u~u~)(tt)
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A t quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Four fermion vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antiu,
                                             antiu,
                                             t,
                                             t]),
                      'color': [color.ColorString([color.K6(-1, 1, 0),
                                                   color.K6Bar(-1,3, 2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0, 0):'GEFF'},
                      'orders':{'NP':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 4)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MT,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FFV1_2(W(1,1),W(1,5),GG,zero,zero,W(1,6))
# Amplitude(s) for diagram number 1
CALL FFFF1C1C2_0(W(1,6),W(1,2),W(1,3),W(1,4),GEFF,AMP(1))
CALL FFFF1C1C2_4(W(1,1),W(1,2),W(1,3),GEFF,MT,WT,W(1,6))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,6),W(1,4),W(1,5),GG,AMP(2))
CALL FFFF1C1C2_3(W(1,1),W(1,2),W(1,4),GEFF,MT,WT,W(1,6))
# Amplitude(s) for diagram number 3
CALL FFV1C1_0(W(1,3),W(1,6),W(1,5),GG,AMP(3))
CALL FFFF1C1C2_2(W(1,1),W(1,3),W(1,4),GEFF,zero,zero,W(1,6))
# Amplitude(s) for diagram number 4
CALL FFV1C1_0(W(1,6),W(1,2),W(1,5),GG,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))'])

        # Check fermion factors
        self.assertEqual([d.get('amplitudes')[0].get('fermionfactor') \
                          for d in matrix_element.get('diagrams')],
                         [1, 1, 1, 1])
        
    def test_multiple_lorentz_structures_with_decay_chain(self):
        """Testing process b~ t > wp+, wp+ > b~ t
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # bt sextet particle
        mypartlist.append(base_objects.Particle({'name':'wp+',
                      'antiname':'wp-',
                      'spin':3,
                      'color':1,
                      'mass':'Mwp',
                      'width':'Wwp',
                      'texname':'wp+',
                      'antitexname':'wp-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':9000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        wp = mypartlist[len(mypartlist) - 1]
        wpbar = copy.copy(wp)
        wpbar.set('is_part', False)

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
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

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([t, antib, wpbar]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([b, antit, wp]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-5,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':True}))
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-5}))
        myleglist.append(base_objects.Leg({'id':6}))
        mydecay = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myproc.set('decay_chains', base_objects.ProcessList([\
            mydecay]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('core_processes')[0])
        self.assertEqual(result,
                         """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),Mwp,NHEL(3),+1*IC(3),W(1,3))
# Amplitude(s) for diagram number 1
CALL FFS3_4_0(W(1,2),W(1,1),W(1,3),GC_108,GC_111,AMP(1))""".split('\n'))
        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('decay_chains')[0].get('core_processes')[0])
        self.assertEqual(result,
                         """CALL VXXXXX(P(0,1),Mwp,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
CALL FFS3_4_3(W(1,2),W(1,3),GC_108,GC_111,Mwp,Wwp,W(1,4))
# Amplitude(s) for diagram number 1
#""".split('\n'))

        matrix_elements = matrix_element.combine_decay_chain_processes()

        matrix_element = matrix_elements[0]

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL FFS3_4_3(W(1,3),W(1,4),GC_108,GC_111,Mwp,Wwp,W(1,5))
# Amplitude(s) for diagram number 1
CALL FFS3_4_0(W(1,2),W(1,1),W(1,5),GC_108,GC_111,AMP(1))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

    def test_multiple_lorentz_structures_with_decay_chain_and_fermion_flow(self):
        """Testing process b t > six1, six1 > b t
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # bt sextet particle
        mypartlist.append(base_objects.Particle({'name':'six1',
                      'antiname':'six1~',
                      'spin':1,
                      'color':6,
                      'mass':'Msix1',
                      'width':'Wsix1',
                      'texname':'six1',
                      'antitexname':'six1bar',
                      'line':'dashed',
                      'charge':1./3.,
                      'pdg_code':9000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        six1 = mypartlist[len(mypartlist) - 1]
        six1bar = copy.copy(six1)
        six1bar.set('is_part', False)

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
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

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([t, b, six1bar]),
            'color': [color.ColorString([color.K6Bar(2,1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([antib, antit, six1]),
            'color': [color.ColorString([color.K6(2,1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':5,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':True}))
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':6}))
        mydecay = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myproc.set('decay_chains', base_objects.ProcessList([\
            mydecay]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('core_processes')[0])
        self.assertEqual(result,
                         """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
# Amplitude(s) for diagram number 1
CALL FFS3_4C1_0(W(1,2),W(1,1),W(1,3),GC_108,GC_111,AMP(1))""".split('\n'))
        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('decay_chains')[0].get('core_processes')[0])
        self.assertEqual(result,
                         """CALL SXXXXX(P(0,1),-1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),+1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MT,NHEL(3),-1*IC(3),W(1,3))
CALL FFS3_4C1_3(W(1,3),W(1,2),GC_108,GC_111,Msix1,Wsix1,W(1,4))
# Amplitude(s) for diagram number 1
#""".split('\n'))

        matrix_elements = matrix_element.combine_decay_chain_processes()

        matrix_element = matrix_elements[0]
        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),MT,NHEL(4),-1*IC(4),W(1,4))
CALL FFS3_4C1_3(W(1,4),W(1,3),GC_108,GC_111,Msix1,Wsix1,W(1,5))
# Amplitude(s) for diagram number 1
CALL FFS3_4C1_0(W(1,2),W(1,1),W(1,5),GC_108,GC_111,AMP(1))""".split('\n')

        
        self.assertEqual(result, goal)

    def test_matrix_multistage_decay_chain_process(self):
        """Test matrix.f for multistage decay chain
        """

        # Set up local model

        mybasemodel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A mu and anti-mu
        mypartlist.append(base_objects.Particle({'name':'mu+',
                      'antiname':'mu-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\mu^+',
                      'antitexname':r'\mu^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':13,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        muminus = mypartlist[len(mypartlist) - 1]
        muplus = copy.copy(muminus)
        muplus.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GAL'},
                      'orders':{'QED':1}}))

        # Coupling of mu to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 16,
                      'particles': base_objects.ParticleList(\
                                            [muminus, \
                                             muplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GAL'},
                      'orders':{'QED':1}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)


        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mybasemodel})

        me_core = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mycoreproc))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay11 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me11 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay11))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay12 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me12 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay12))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':13,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-13,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mybasemodel})

        me2 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay2))

        mydecay11.set('decay_chains', base_objects.ProcessList([mydecay2]))
        mydecay12.set('decay_chains', base_objects.ProcessList([mydecay2]))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay11, mydecay12]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_elements = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()

        me = matrix_elements[0]

        # Check all ingredients in file here

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(me.get_nexternal_ninitial(), (10, 2))
        self.assertEqual(me.get_helicity_combinations(), 1024)
        self.assertEqual(len(exporter.get_helicity_lines(me).split("\n")), 1024)
        # This has been tested against v4
        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,5))
CALL VXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,6))
CALL FVOXXX(W(1,4),W(1,6),GAL,zero,zero,W(1,7))
CALL JIOXXX(W(1,5),W(1,7),GAL,zero,zero,W(1,8))
CALL FVOXXX(W(1,3),W(1,8),GAL,zero,zero,W(1,7))
CALL IXXXXX(P(0,7),zero,NHEL(7),-1*IC(7),W(1,8))
CALL OXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,9))
CALL IXXXXX(P(0,9),zero,NHEL(9),-1*IC(9),W(1,10))
CALL VXXXXX(P(0,10),zero,NHEL(10),+1*IC(10),W(1,11))
CALL FVOXXX(W(1,9),W(1,11),GAL,zero,zero,W(1,12))
CALL JIOXXX(W(1,10),W(1,12),GAL,zero,zero,W(1,13))
CALL FVIXXX(W(1,8),W(1,13),GAL,zero,zero,W(1,12))
CALL FVOXXX(W(1,7),W(1,1),GAL,zero,zero,W(1,13))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,12),W(1,13),W(1,2),GAL,AMP(1))
CALL FVIXXX(W(1,10),W(1,11),GAL,zero,zero,W(1,14))
CALL JIOXXX(W(1,14),W(1,9),GAL,zero,zero,W(1,11))
CALL FVIXXX(W(1,8),W(1,11),GAL,zero,zero,W(1,14))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,14),W(1,13),W(1,2),GAL,AMP(2))
CALL FVIXXX(W(1,5),W(1,6),GAL,zero,zero,W(1,13))
CALL JIOXXX(W(1,13),W(1,4),GAL,zero,zero,W(1,6))
CALL FVOXXX(W(1,3),W(1,6),GAL,zero,zero,W(1,13))
CALL FVOXXX(W(1,13),W(1,1),GAL,zero,zero,W(1,6))
# Amplitude(s) for diagram number 3
CALL IOVXXX(W(1,12),W(1,6),W(1,2),GAL,AMP(3))
# Amplitude(s) for diagram number 4
CALL IOVXXX(W(1,14),W(1,6),W(1,2),GAL,AMP(4))
CALL FVIXXX(W(1,12),W(1,1),GAL,zero,zero,W(1,6))
# Amplitude(s) for diagram number 5
CALL IOVXXX(W(1,6),W(1,7),W(1,2),GAL,AMP(5))
CALL FVIXXX(W(1,14),W(1,1),GAL,zero,zero,W(1,12))
# Amplitude(s) for diagram number 6
CALL IOVXXX(W(1,12),W(1,7),W(1,2),GAL,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOVXXX(W(1,6),W(1,13),W(1,2),GAL,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOVXXX(W(1,12),W(1,13),W(1,2),GAL,AMP(8))""".split('\n'))

        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_pmass_file(writer, me)
        writer.close()
        self.assertFileContains('test',"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ZERO
      PMASS(4)=ZERO
      PMASS(5)=ZERO
      PMASS(6)=ZERO
      PMASS(7)=ZERO
      PMASS(8)=ZERO
      PMASS(9)=ZERO
      PMASS(10)=ZERO\n""")


    def test_matrix_4g_decay_chain_process(self):
        """Test matrix.f for multistage decay chain
        """

        # Set up local model

        mybasemodel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

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

        g = mypartlist[len(mypartlist) - 1]

        # Gluon self-couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g,
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.f(2, 1, 0)]),
                                color.ColorString([color.f(1, 0, 2)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
                      'orders':{'QCD':2}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)


        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mybasemodel})

        me_core = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mycoreproc))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me1 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay1))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_elements = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()

        me = matrix_elements[0]

        # Check all ingredients in file here

        exporter = export_v4.ProcessExporterFortranME()


        goal = """16 82 [0, 0, 0]
16 83 [0, 1, 0]
16 84 [0, 2, 0]
16 85 [1, 0, 0]
16 86 [1, 1, 0]
16 87 [1, 2, 0]
16 88 [2, 0, 0]
16 89 [2, 1, 0]
16 90 [2, 2, 0]
16 91 [0, 0, 1]
16 92 [0, 1, 1]
16 93 [0, 2, 1]
16 94 [1, 0, 1]
16 95 [1, 1, 1]
16 96 [1, 2, 1]
16 97 [2, 0, 1]
16 98 [2, 1, 1]
16 99 [2, 2, 1]
16 100 [0, 0, 2]
16 101 [0, 1, 2]
16 102 [0, 2, 2]
16 103 [1, 0, 2]
16 104 [1, 1, 2]
16 105 [1, 2, 2]
16 106 [2, 0, 2]
16 107 [2, 1, 2]
16 108 [2, 2, 2]""".split("\n")

        diagram = me.get('diagrams')[15]

        for i, amp in enumerate(diagram.get('amplitudes')):
            if diagram.get('number') == 16:
                self.assertEqual("%d %d %s" % \
                                 (diagram.get('number'), amp.get('number'), \
                                  repr(amp.get('color_indices'))),
                                 goal[i])

        self.assertEqual(me.get_nexternal_ninitial(), (8, 2))
        self.assertEqual(me.get_helicity_combinations(), 256)
        self.assertEqual(len(exporter.get_helicity_lines(me).split("\n")), 256)
        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL JVVXXX(W(1,3),W(1,4),GG,zero,zero,W(1,6))
CALL JVVXXX(W(1,6),W(1,5),GG,zero,zero,W(1,7))
CALL VXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,6))
CALL VXXXXX(P(0,7),zero,NHEL(7),+1*IC(7),W(1,8))
CALL VXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,9))
CALL JVVXXX(W(1,6),W(1,8),GG,zero,zero,W(1,10))
CALL JVVXXX(W(1,10),W(1,9),GG,zero,zero,W(1,11))
# Amplitude(s) for diagram number 1
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,11),GG,AMP(1))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,11),GG,AMP(2))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,11),GG,AMP(3))
CALL JVVXXX(W(1,6),W(1,9),GG,zero,zero,W(1,10))
CALL JVVXXX(W(1,10),W(1,8),GG,zero,zero,W(1,12))
# Amplitude(s) for diagram number 2
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,12),GG,AMP(4))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,12),GG,AMP(5))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,12),GG,AMP(6))
CALL JVVXXX(W(1,8),W(1,9),GG,zero,zero,W(1,10))
CALL JVVXXX(W(1,6),W(1,10),GG,zero,zero,W(1,13))
# Amplitude(s) for diagram number 3
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,13),GG,AMP(7))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,13),GG,AMP(8))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,13),GG,AMP(9))
CALL JGGGXX(W(1,9),W(1,8),W(1,6),GG,W(1,10))
CALL JGGGXX(W(1,6),W(1,9),W(1,8),GG,W(1,14))
CALL JGGGXX(W(1,8),W(1,6),W(1,9),GG,W(1,15))
# Amplitude(s) for diagram number 4
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,10),GG,AMP(10))
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,14),GG,AMP(11))
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,15),GG,AMP(12))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,10),GG,AMP(13))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,14),GG,AMP(14))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,15),GG,AMP(15))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,10),GG,AMP(16))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,14),GG,AMP(17))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,15),GG,AMP(18))
CALL JVVXXX(W(1,3),W(1,5),GG,zero,zero,W(1,9))
CALL JVVXXX(W(1,9),W(1,4),GG,zero,zero,W(1,8))
# Amplitude(s) for diagram number 5
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,11),GG,AMP(19))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,11),GG,AMP(20))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,11),GG,AMP(21))
# Amplitude(s) for diagram number 6
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,12),GG,AMP(22))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,12),GG,AMP(23))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,12),GG,AMP(24))
# Amplitude(s) for diagram number 7
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,13),GG,AMP(25))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,13),GG,AMP(26))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,13),GG,AMP(27))
# Amplitude(s) for diagram number 8
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,10),GG,AMP(28))
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,14),GG,AMP(29))
CALL GGGGXX(W(1,1),W(1,2),W(1,8),W(1,15),GG,AMP(30))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,10),GG,AMP(31))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,14),GG,AMP(32))
CALL GGGGXX(W(1,8),W(1,1),W(1,2),W(1,15),GG,AMP(33))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,10),GG,AMP(34))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,14),GG,AMP(35))
CALL GGGGXX(W(1,2),W(1,8),W(1,1),W(1,15),GG,AMP(36))
CALL JVVXXX(W(1,4),W(1,5),GG,zero,zero,W(1,9))
CALL JVVXXX(W(1,3),W(1,9),GG,zero,zero,W(1,6))
# Amplitude(s) for diagram number 9
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,11),GG,AMP(37))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,11),GG,AMP(38))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,11),GG,AMP(39))
# Amplitude(s) for diagram number 10
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,12),GG,AMP(40))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,12),GG,AMP(41))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,12),GG,AMP(42))
# Amplitude(s) for diagram number 11
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,13),GG,AMP(43))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,13),GG,AMP(44))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,13),GG,AMP(45))
# Amplitude(s) for diagram number 12
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,10),GG,AMP(46))
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,14),GG,AMP(47))
CALL GGGGXX(W(1,1),W(1,2),W(1,6),W(1,15),GG,AMP(48))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,10),GG,AMP(49))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,14),GG,AMP(50))
CALL GGGGXX(W(1,6),W(1,1),W(1,2),W(1,15),GG,AMP(51))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,10),GG,AMP(52))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,14),GG,AMP(53))
CALL GGGGXX(W(1,2),W(1,6),W(1,1),W(1,15),GG,AMP(54))
CALL JGGGXX(W(1,5),W(1,4),W(1,3),GG,W(1,9))
CALL JGGGXX(W(1,3),W(1,5),W(1,4),GG,W(1,16))
CALL JGGGXX(W(1,4),W(1,3),W(1,5),GG,W(1,17))
# Amplitude(s) for diagram number 13
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,11),GG,AMP(55))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,11),GG,AMP(56))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,11),GG,AMP(57))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,11),GG,AMP(58))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,11),GG,AMP(59))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,11),GG,AMP(60))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,11),GG,AMP(61))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,11),GG,AMP(62))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,11),GG,AMP(63))
# Amplitude(s) for diagram number 14
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,12),GG,AMP(64))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,12),GG,AMP(65))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,12),GG,AMP(66))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,12),GG,AMP(67))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,12),GG,AMP(68))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,12),GG,AMP(69))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,12),GG,AMP(70))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,12),GG,AMP(71))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,12),GG,AMP(72))
# Amplitude(s) for diagram number 15
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,13),GG,AMP(73))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,13),GG,AMP(74))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,13),GG,AMP(75))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,13),GG,AMP(76))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,13),GG,AMP(77))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,13),GG,AMP(78))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,13),GG,AMP(79))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,13),GG,AMP(80))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,13),GG,AMP(81))
# Amplitude(s) for diagram number 16
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,10),GG,AMP(82))
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,14),GG,AMP(83))
CALL GGGGXX(W(1,1),W(1,2),W(1,9),W(1,15),GG,AMP(84))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,10),GG,AMP(85))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,14),GG,AMP(86))
CALL GGGGXX(W(1,1),W(1,2),W(1,16),W(1,15),GG,AMP(87))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,10),GG,AMP(88))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,14),GG,AMP(89))
CALL GGGGXX(W(1,1),W(1,2),W(1,17),W(1,15),GG,AMP(90))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,10),GG,AMP(91))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,14),GG,AMP(92))
CALL GGGGXX(W(1,9),W(1,1),W(1,2),W(1,15),GG,AMP(93))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,10),GG,AMP(94))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,14),GG,AMP(95))
CALL GGGGXX(W(1,16),W(1,1),W(1,2),W(1,15),GG,AMP(96))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,10),GG,AMP(97))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,14),GG,AMP(98))
CALL GGGGXX(W(1,17),W(1,1),W(1,2),W(1,15),GG,AMP(99))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,10),GG,AMP(100))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,14),GG,AMP(101))
CALL GGGGXX(W(1,2),W(1,9),W(1,1),W(1,15),GG,AMP(102))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,10),GG,AMP(103))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,14),GG,AMP(104))
CALL GGGGXX(W(1,2),W(1,16),W(1,1),W(1,15),GG,AMP(105))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,10),GG,AMP(106))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,14),GG,AMP(107))
CALL GGGGXX(W(1,2),W(1,17),W(1,1),W(1,15),GG,AMP(108))
CALL JVVXXX(W(1,1),W(1,2),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 17
CALL VVVXXX(W(1,7),W(1,11),W(1,5),GG,AMP(109))
# Amplitude(s) for diagram number 18
CALL VVVXXX(W(1,7),W(1,12),W(1,5),GG,AMP(110))
# Amplitude(s) for diagram number 19
CALL VVVXXX(W(1,7),W(1,13),W(1,5),GG,AMP(111))
# Amplitude(s) for diagram number 20
CALL VVVXXX(W(1,7),W(1,10),W(1,5),GG,AMP(112))
CALL VVVXXX(W(1,7),W(1,14),W(1,5),GG,AMP(113))
CALL VVVXXX(W(1,7),W(1,15),W(1,5),GG,AMP(114))
# Amplitude(s) for diagram number 21
CALL VVVXXX(W(1,8),W(1,11),W(1,5),GG,AMP(115))
# Amplitude(s) for diagram number 22
CALL VVVXXX(W(1,8),W(1,12),W(1,5),GG,AMP(116))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,8),W(1,13),W(1,5),GG,AMP(117))
# Amplitude(s) for diagram number 24
CALL VVVXXX(W(1,8),W(1,10),W(1,5),GG,AMP(118))
CALL VVVXXX(W(1,8),W(1,14),W(1,5),GG,AMP(119))
CALL VVVXXX(W(1,8),W(1,15),W(1,5),GG,AMP(120))
# Amplitude(s) for diagram number 25
CALL VVVXXX(W(1,6),W(1,11),W(1,5),GG,AMP(121))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,6),W(1,12),W(1,5),GG,AMP(122))
# Amplitude(s) for diagram number 27
CALL VVVXXX(W(1,6),W(1,13),W(1,5),GG,AMP(123))
# Amplitude(s) for diagram number 28
CALL VVVXXX(W(1,6),W(1,10),W(1,5),GG,AMP(124))
CALL VVVXXX(W(1,6),W(1,14),W(1,5),GG,AMP(125))
CALL VVVXXX(W(1,6),W(1,15),W(1,5),GG,AMP(126))
# Amplitude(s) for diagram number 29
CALL VVVXXX(W(1,9),W(1,11),W(1,5),GG,AMP(127))
CALL VVVXXX(W(1,16),W(1,11),W(1,5),GG,AMP(128))
CALL VVVXXX(W(1,17),W(1,11),W(1,5),GG,AMP(129))
# Amplitude(s) for diagram number 30
CALL VVVXXX(W(1,9),W(1,12),W(1,5),GG,AMP(130))
CALL VVVXXX(W(1,16),W(1,12),W(1,5),GG,AMP(131))
CALL VVVXXX(W(1,17),W(1,12),W(1,5),GG,AMP(132))
# Amplitude(s) for diagram number 31
CALL VVVXXX(W(1,9),W(1,13),W(1,5),GG,AMP(133))
CALL VVVXXX(W(1,16),W(1,13),W(1,5),GG,AMP(134))
CALL VVVXXX(W(1,17),W(1,13),W(1,5),GG,AMP(135))
# Amplitude(s) for diagram number 32
CALL VVVXXX(W(1,9),W(1,10),W(1,5),GG,AMP(136))
CALL VVVXXX(W(1,9),W(1,14),W(1,5),GG,AMP(137))
CALL VVVXXX(W(1,9),W(1,15),W(1,5),GG,AMP(138))
CALL VVVXXX(W(1,16),W(1,10),W(1,5),GG,AMP(139))
CALL VVVXXX(W(1,16),W(1,14),W(1,5),GG,AMP(140))
CALL VVVXXX(W(1,16),W(1,15),W(1,5),GG,AMP(141))
CALL VVVXXX(W(1,17),W(1,10),W(1,5),GG,AMP(142))
CALL VVVXXX(W(1,17),W(1,14),W(1,5),GG,AMP(143))
CALL VVVXXX(W(1,17),W(1,15),W(1,5),GG,AMP(144))
CALL JVVXXX(W(1,1),W(1,7),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 33
CALL VVVXXX(W(1,2),W(1,11),W(1,5),GG,AMP(145))
# Amplitude(s) for diagram number 34
CALL VVVXXX(W(1,2),W(1,12),W(1,5),GG,AMP(146))
# Amplitude(s) for diagram number 35
CALL VVVXXX(W(1,2),W(1,13),W(1,5),GG,AMP(147))
# Amplitude(s) for diagram number 36
CALL VVVXXX(W(1,2),W(1,10),W(1,5),GG,AMP(148))
CALL VVVXXX(W(1,2),W(1,14),W(1,5),GG,AMP(149))
CALL VVVXXX(W(1,2),W(1,15),W(1,5),GG,AMP(150))
CALL JVVXXX(W(1,1),W(1,8),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 37
CALL VVVXXX(W(1,2),W(1,11),W(1,5),GG,AMP(151))
# Amplitude(s) for diagram number 38
CALL VVVXXX(W(1,2),W(1,12),W(1,5),GG,AMP(152))
# Amplitude(s) for diagram number 39
CALL VVVXXX(W(1,2),W(1,13),W(1,5),GG,AMP(153))
# Amplitude(s) for diagram number 40
CALL VVVXXX(W(1,2),W(1,10),W(1,5),GG,AMP(154))
CALL VVVXXX(W(1,2),W(1,14),W(1,5),GG,AMP(155))
CALL VVVXXX(W(1,2),W(1,15),W(1,5),GG,AMP(156))
CALL JVVXXX(W(1,1),W(1,6),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 41
CALL VVVXXX(W(1,2),W(1,11),W(1,5),GG,AMP(157))
# Amplitude(s) for diagram number 42
CALL VVVXXX(W(1,2),W(1,12),W(1,5),GG,AMP(158))
# Amplitude(s) for diagram number 43
CALL VVVXXX(W(1,2),W(1,13),W(1,5),GG,AMP(159))
# Amplitude(s) for diagram number 44
CALL VVVXXX(W(1,2),W(1,10),W(1,5),GG,AMP(160))
CALL VVVXXX(W(1,2),W(1,14),W(1,5),GG,AMP(161))
CALL VVVXXX(W(1,2),W(1,15),W(1,5),GG,AMP(162))
CALL JVVXXX(W(1,1),W(1,9),GG,zero,zero,W(1,5))
CALL JVVXXX(W(1,1),W(1,16),GG,zero,zero,W(1,4))
CALL JVVXXX(W(1,1),W(1,17),GG,zero,zero,W(1,3))
# Amplitude(s) for diagram number 45
CALL VVVXXX(W(1,2),W(1,11),W(1,5),GG,AMP(163))
CALL VVVXXX(W(1,2),W(1,11),W(1,4),GG,AMP(164))
CALL VVVXXX(W(1,2),W(1,11),W(1,3),GG,AMP(165))
# Amplitude(s) for diagram number 46
CALL VVVXXX(W(1,2),W(1,12),W(1,5),GG,AMP(166))
CALL VVVXXX(W(1,2),W(1,12),W(1,4),GG,AMP(167))
CALL VVVXXX(W(1,2),W(1,12),W(1,3),GG,AMP(168))
# Amplitude(s) for diagram number 47
CALL VVVXXX(W(1,2),W(1,13),W(1,5),GG,AMP(169))
CALL VVVXXX(W(1,2),W(1,13),W(1,4),GG,AMP(170))
CALL VVVXXX(W(1,2),W(1,13),W(1,3),GG,AMP(171))
# Amplitude(s) for diagram number 48
CALL VVVXXX(W(1,2),W(1,10),W(1,5),GG,AMP(172))
CALL VVVXXX(W(1,2),W(1,14),W(1,5),GG,AMP(173))
CALL VVVXXX(W(1,2),W(1,15),W(1,5),GG,AMP(174))
CALL VVVXXX(W(1,2),W(1,10),W(1,4),GG,AMP(175))
CALL VVVXXX(W(1,2),W(1,14),W(1,4),GG,AMP(176))
CALL VVVXXX(W(1,2),W(1,15),W(1,4),GG,AMP(177))
CALL VVVXXX(W(1,2),W(1,10),W(1,3),GG,AMP(178))
CALL VVVXXX(W(1,2),W(1,14),W(1,3),GG,AMP(179))
CALL VVVXXX(W(1,2),W(1,15),W(1,3),GG,AMP(180))
CALL JVVXXX(W(1,1),W(1,11),GG,zero,zero,W(1,3))
# Amplitude(s) for diagram number 49
CALL VVVXXX(W(1,2),W(1,7),W(1,3),GG,AMP(181))
CALL JVVXXX(W(1,1),W(1,12),GG,zero,zero,W(1,11))
# Amplitude(s) for diagram number 50
CALL VVVXXX(W(1,2),W(1,7),W(1,11),GG,AMP(182))
CALL JVVXXX(W(1,1),W(1,13),GG,zero,zero,W(1,12))
# Amplitude(s) for diagram number 51
CALL VVVXXX(W(1,2),W(1,7),W(1,12),GG,AMP(183))
CALL JVVXXX(W(1,1),W(1,10),GG,zero,zero,W(1,13))
CALL JVVXXX(W(1,1),W(1,14),GG,zero,zero,W(1,10))
CALL JVVXXX(W(1,1),W(1,15),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 52
CALL VVVXXX(W(1,2),W(1,7),W(1,13),GG,AMP(184))
CALL VVVXXX(W(1,2),W(1,7),W(1,10),GG,AMP(185))
CALL VVVXXX(W(1,2),W(1,7),W(1,14),GG,AMP(186))
# Amplitude(s) for diagram number 53
CALL VVVXXX(W(1,2),W(1,8),W(1,3),GG,AMP(187))
# Amplitude(s) for diagram number 54
CALL VVVXXX(W(1,2),W(1,8),W(1,11),GG,AMP(188))
# Amplitude(s) for diagram number 55
CALL VVVXXX(W(1,2),W(1,8),W(1,12),GG,AMP(189))
# Amplitude(s) for diagram number 56
CALL VVVXXX(W(1,2),W(1,8),W(1,13),GG,AMP(190))
CALL VVVXXX(W(1,2),W(1,8),W(1,10),GG,AMP(191))
CALL VVVXXX(W(1,2),W(1,8),W(1,14),GG,AMP(192))
# Amplitude(s) for diagram number 57
CALL VVVXXX(W(1,2),W(1,6),W(1,3),GG,AMP(193))
# Amplitude(s) for diagram number 58
CALL VVVXXX(W(1,2),W(1,6),W(1,11),GG,AMP(194))
# Amplitude(s) for diagram number 59
CALL VVVXXX(W(1,2),W(1,6),W(1,12),GG,AMP(195))
# Amplitude(s) for diagram number 60
CALL VVVXXX(W(1,2),W(1,6),W(1,13),GG,AMP(196))
CALL VVVXXX(W(1,2),W(1,6),W(1,10),GG,AMP(197))
CALL VVVXXX(W(1,2),W(1,6),W(1,14),GG,AMP(198))
# Amplitude(s) for diagram number 61
CALL VVVXXX(W(1,2),W(1,9),W(1,3),GG,AMP(199))
CALL VVVXXX(W(1,2),W(1,16),W(1,3),GG,AMP(200))
CALL VVVXXX(W(1,2),W(1,17),W(1,3),GG,AMP(201))
# Amplitude(s) for diagram number 62
CALL VVVXXX(W(1,2),W(1,9),W(1,11),GG,AMP(202))
CALL VVVXXX(W(1,2),W(1,16),W(1,11),GG,AMP(203))
CALL VVVXXX(W(1,2),W(1,17),W(1,11),GG,AMP(204))
# Amplitude(s) for diagram number 63
CALL VVVXXX(W(1,2),W(1,9),W(1,12),GG,AMP(205))
CALL VVVXXX(W(1,2),W(1,16),W(1,12),GG,AMP(206))
CALL VVVXXX(W(1,2),W(1,17),W(1,12),GG,AMP(207))
# Amplitude(s) for diagram number 64
CALL VVVXXX(W(1,2),W(1,9),W(1,13),GG,AMP(208))
CALL VVVXXX(W(1,2),W(1,9),W(1,10),GG,AMP(209))
CALL VVVXXX(W(1,2),W(1,9),W(1,14),GG,AMP(210))
CALL VVVXXX(W(1,2),W(1,16),W(1,13),GG,AMP(211))
CALL VVVXXX(W(1,2),W(1,16),W(1,10),GG,AMP(212))
CALL VVVXXX(W(1,2),W(1,16),W(1,14),GG,AMP(213))
CALL VVVXXX(W(1,2),W(1,17),W(1,13),GG,AMP(214))
CALL VVVXXX(W(1,2),W(1,17),W(1,10),GG,AMP(215))
CALL VVVXXX(W(1,2),W(1,17),W(1,14),GG,AMP(216))""".split('\n'))


        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_pmass_file(writer, me)
        writer.close()

        self.assertFileContains('test',"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ZERO
      PMASS(4)=ZERO
      PMASS(5)=ZERO
      PMASS(6)=ZERO
      PMASS(7)=ZERO
      PMASS(8)=ZERO\n""")

    def test_vector_clash_majorana_process(self):
        """Test majorana process w+ w- > n2 n2
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # Neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'MN1',
                      'width':'WN1',
                      'texname':r'\chi_0^2',
                      'antitexname':r'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w-',
                      'antiname':'w+',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w-',
                      'antitexname':'w+',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':-24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # chargino+/-
        mypartlist.append(base_objects.Particle({'name':'x1-',
                      'antiname':'x1+',
                      'spin':2,
                      'color':1,
                      'mass':'MX1',
                      'width':'WX1',
                      'texname':'x1-',
                      'antitexname':'x1+',
                      'line':'straight',
                      'charge':1.,
                      'pdg_code':-1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        x1plus = mypartlist[len(mypartlist) - 1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # Coupling of n1 to w
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             x1minus, \
                                             wplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GWN1X1'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [x1plus, \
                                             n1, \
                                             wminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GWX1N1'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 2)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
        """CALL VXXXXX(P(0,1),WMASS,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),WMASS,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MN1,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MN1,NHEL(4),+1*IC(4),W(1,4))
CALL FVICXX(W(1,3),W(1,1),GWN1X1,MX1,WX1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVCXX(W(1,5),W(1,4),W(1,2),GWX1N1,AMP(1))
CALL FVOXXX(W(1,4),W(1,1),GWN1X1,MX1,WX1,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,3),W(1,5),W(1,2),GWX1N1,AMP(2))""")


    def test_export_majorana_decay_chain(self):
        """Test decay chain with majorana particles and MadEvent files
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl2-',
                      'antiname':'sl2+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist) - 1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':r'\chi_0^1',
                      'antitexname':r'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        # Coupling of sl2 to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [a, \
                                             seplus, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX56'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        # This has been checked against v4
        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOXXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,3))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,6))
CALL FSICXX(W(1,3),W(1,6),MGVX350,Mneu1,Wneu1,W(1,7))
CALL HIOXXX(W(1,1),W(1,5),MGVX494,Msl2,Wsl2,W(1,3))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,2),W(1,3),MGVX350,AMP(1))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))
CALL FSICXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,7))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,3))
CALL FSOXXX(W(1,3),W(1,6),MGVX350,Mneu1,Wneu1,W(1,4))
CALL HIOXXX(W(1,1),W(1,4),MGVX494,Msl2,Wsl2,W(1,3))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,7),W(1,2),W(1,3),MGVX350,AMP(2))""".split('\n'))

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(me)[0][0],
                         " JAMP(1) = AMP(1)+(-1.000000000000000d+00)*AMP(2)")

        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+, n1 > e+ sl2-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1, mydecay2]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        # This has been checked against v4
        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
        """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOXXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,3))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,6))
CALL FSIXXX(W(1,3),W(1,6),MGVX494,Mneu1,Wneu1,W(1,7))
CALL HIOXXX(W(1,1),W(1,5),MGVX494,Msl2,Wsl2,W(1,3))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,2),W(1,3),MGVX350,AMP(1))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))
CALL FSICXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,7))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,3))
CALL FSOCXX(W(1,3),W(1,6),MGVX494,Mneu1,Wneu1,W(1,4))
CALL HIOXXX(W(1,1),W(1,4),MGVX494,Msl2,Wsl2,W(1,3))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,7),W(1,2),W(1,3),MGVX350,AMP(2))""".split('\n'))

#        self.assertEqual(exporter.get_JAMP_lines(me)[0][0],
#                         " JAMP(1) = AMP(1)+(-1.0)*AMP(2)")


        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+ a $ sl2+

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay3 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel,
                                         'forbidden_onsh_s_channels':[-1000011]})

        me3 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay3))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay3]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        #print me.get_base_amplitude().nice_string()

        # This has been checked against v4
        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
        """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FVOXXX(W(1,3),W(1,5),MGVX12,zero,zero,W(1,6))
CALL FSOXXX(W(1,6),W(1,4),MGVX350,Mneu1,Wneu1,W(1,7))
CALL IXXXXX(P(0,6),zero,NHEL(6),-1*IC(6),W(1,6))
CALL SXXXXX(P(0,7),+1*IC(7),W(1,8))
CALL VXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,9))
CALL FVICXX(W(1,6),W(1,9),MGVX12,zero,zero,W(1,10))
CALL FSICXX(W(1,10),W(1,8),MGVX350,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,1),W(1,7),MGVX494,Msl2,Wsl2,W(1,10))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,11),W(1,2),W(1,10),MGVX350,AMP(1))
CALL HVSXXX(W(1,9),W(1,8),MGVX56,Msl2,Wsl2,W(1,7))
CALL FSICXX(W(1,6),W(1,7),MGVX350,Mneu1,Wneu1,W(1,12))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,12),W(1,2),W(1,10),MGVX350,AMP(2))
CALL HVSXXX(W(1,5),W(1,4),MGVX56,Msl2,Wsl2,W(1,10))
CALL FSOXXX(W(1,3),W(1,10),MGVX350,Mneu1,Wneu1,W(1,6))
CALL HIOXXX(W(1,1),W(1,6),MGVX494,Msl2,Wsl2,W(1,3))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,11),W(1,2),W(1,3),MGVX350,AMP(3))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,12),W(1,2),W(1,3),MGVX350,AMP(4))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))
CALL FVICXX(W(1,3),W(1,5),MGVX12,zero,zero,W(1,12))
CALL FSICXX(W(1,12),W(1,4),MGVX350,Mneu1,Wneu1,W(1,5))
CALL OXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,12))
CALL FVOXXX(W(1,12),W(1,9),MGVX12,zero,zero,W(1,4))
CALL FSOXXX(W(1,4),W(1,8),MGVX350,Mneu1,Wneu1,W(1,9))
CALL HIOXXX(W(1,1),W(1,9),MGVX494,Msl2,Wsl2,W(1,4))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,5),W(1,2),W(1,4),MGVX350,AMP(5))
CALL FSOXXX(W(1,12),W(1,7),MGVX350,Mneu1,Wneu1,W(1,9))
CALL HIOXXX(W(1,1),W(1,9),MGVX494,Msl2,Wsl2,W(1,12))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,5),W(1,2),W(1,12),MGVX350,AMP(6))
CALL FSICXX(W(1,3),W(1,10),MGVX350,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,5),W(1,2),W(1,4),MGVX350,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,5),W(1,2),W(1,12),MGVX350,AMP(8))""".split('\n'))

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(me)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))',
                          'AMP2(5)=AMP2(5)+AMP(5)*dconjg(AMP(5))',
                          'AMP2(6)=AMP2(6)+AMP(6)*dconjg(AMP(6))',
                          'AMP2(7)=AMP2(7)+AMP(7)*dconjg(AMP(7))',
                          'AMP2(8)=AMP2(8)+AMP(8)*dconjg(AMP(8))'])
        
        # Test jamp lines        
        self.assertEqual(exporter.get_JAMP_lines(me)[0][0],
                         " JAMP(1) = AMP(1)+AMP(2)+AMP(3)+AMP(4)+(-1.000000000000000d+00)*AMP(5)+(-1.000000000000000d+00)*AMP(6)+(-1.000000000000000d+00)*AMP(7)+(-1.000000000000000d+00)*AMP(8)")
                         

        writer = writers.FortranWriter(self.give_pos('test'))

        # Test configs.inc file
        mapconfigs, (s_and_t_channels, nqcd_list) = exporter.write_configs_file(writer,
                                     me)
        writer.close()
        #print( open(self.give_pos('test')).read())
        
        self.assertFileContains('test',
                         """C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/5,3/
      DATA (SPROP(I,-1,1),I=1,1)/11/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA (SPROP(I,-2,1),I=1,1)/1000022/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/8,6/
      DATA (SPROP(I,-3,1),I=1,1)/11/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/7,-3/
      DATA (SPROP(I,-4,1),I=1,1)/1000022/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/1,-2/
      DATA TPRID(-5,1)/1000011/
      DATA (SPROP(I,-5,1),I=1,1)/0/
      DATA (IFOREST(I,-6,1),I=1,2)/-5,-4/
      DATA TPRID(-6,1)/11/
      DATA (SPROP(I,-6,1),I=1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,3/
      DATA (SPROP(I,-1,2),I=1,1)/11/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/4,-1/
      DATA (SPROP(I,-2,2),I=1,1)/1000022/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/8,7/
      DATA (SPROP(I,-3,2),I=1,1)/-1000011/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/-3,6/
      DATA (SPROP(I,-4,2),I=1,1)/1000022/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/1,-2/
      DATA TPRID(-5,2)/1000011/
      DATA (SPROP(I,-5,2),I=1,1)/0/
      DATA (IFOREST(I,-6,2),I=1,2)/-5,-4/
      DATA TPRID(-6,2)/11/
      DATA (SPROP(I,-6,2),I=1,1)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/5,4/
      DATA (SPROP(I,-1,3),I=1,1)/-1000011/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
      DATA (SPROP(I,-2,3),I=1,1)/1000022/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/8,6/
      DATA (SPROP(I,-3,3),I=1,1)/11/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/7,-3/
      DATA (SPROP(I,-4,3),I=1,1)/1000022/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/1,-2/
      DATA TPRID(-5,3)/1000011/
      DATA (SPROP(I,-5,3),I=1,1)/0/
      DATA (IFOREST(I,-6,3),I=1,2)/-5,-4/
      DATA TPRID(-6,3)/11/
      DATA (SPROP(I,-6,3),I=1,1)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/5,4/
      DATA (SPROP(I,-1,4),I=1,1)/-1000011/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,3/
      DATA (SPROP(I,-2,4),I=1,1)/1000022/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/8,7/
      DATA (SPROP(I,-3,4),I=1,1)/-1000011/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,6/
      DATA (SPROP(I,-4,4),I=1,1)/1000022/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/1,-2/
      DATA TPRID(-5,4)/1000011/
      DATA (SPROP(I,-5,4),I=1,1)/0/
      DATA (IFOREST(I,-6,4),I=1,2)/-5,-4/
      DATA TPRID(-6,4)/11/
      DATA (SPROP(I,-6,4),I=1,1)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/8,6/
      DATA (SPROP(I,-1,5),I=1,1)/11/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/7,-1/
      DATA (SPROP(I,-2,5),I=1,1)/1000022/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/5,3/
      DATA (SPROP(I,-3,5),I=1,1)/11/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/4,-3/
      DATA (SPROP(I,-4,5),I=1,1)/1000022/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/1,-2/
      DATA TPRID(-5,5)/1000011/
      DATA (SPROP(I,-5,5),I=1,1)/0/
      DATA (IFOREST(I,-6,5),I=1,2)/-5,-4/
      DATA TPRID(-6,5)/11/
      DATA (SPROP(I,-6,5),I=1,1)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/8,7/
      DATA (SPROP(I,-1,6),I=1,1)/-1000011/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/-1,6/
      DATA (SPROP(I,-2,6),I=1,1)/1000022/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/5,3/
      DATA (SPROP(I,-3,6),I=1,1)/11/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/4,-3/
      DATA (SPROP(I,-4,6),I=1,1)/1000022/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/1,-2/
      DATA TPRID(-5,6)/1000011/
      DATA (SPROP(I,-5,6),I=1,1)/0/
      DATA (IFOREST(I,-6,6),I=1,2)/-5,-4/
      DATA TPRID(-6,6)/11/
      DATA (SPROP(I,-6,6),I=1,1)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA TSTRATEGY(7)/2/
      DATA (IFOREST(I,-1,7),I=1,2)/8,6/
      DATA (SPROP(I,-1,7),I=1,1)/11/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/7,-1/
      DATA (SPROP(I,-2,7),I=1,1)/1000022/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/5,4/
      DATA (SPROP(I,-3,7),I=1,1)/-1000011/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/-3,3/
      DATA (SPROP(I,-4,7),I=1,1)/1000022/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/1,-2/
      DATA TPRID(-5,7)/1000011/
      DATA (SPROP(I,-5,7),I=1,1)/0/
      DATA (IFOREST(I,-6,7),I=1,2)/-5,-4/
      DATA TPRID(-6,7)/11/
      DATA (SPROP(I,-6,7),I=1,1)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA TSTRATEGY(8)/2/
      DATA (IFOREST(I,-1,8),I=1,2)/8,7/
      DATA (SPROP(I,-1,8),I=1,1)/-1000011/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,6/
      DATA (SPROP(I,-2,8),I=1,1)/1000022/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/5,4/
      DATA (SPROP(I,-3,8),I=1,1)/-1000011/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,3/
      DATA (SPROP(I,-4,8),I=1,1)/1000022/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/1,-2/
      DATA TPRID(-5,8)/1000011/
      DATA (SPROP(I,-5,8),I=1,1)/0/
      DATA (IFOREST(I,-6,8),I=1,2)/-5,-4/
      DATA TPRID(-6,8)/11/
      DATA (SPROP(I,-6,8),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/8/
""")

        writer = writers.FortranWriter(self.give_pos('test'))

        # Test decayBW file
        exporter.write_decayBW_file(writer,
                                     s_and_t_channels)

        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
                         """      DATA GFORCEBW(-1,1)/0/
      DATA GFORCEBW(-2,1)/1/
      DATA GFORCEBW(-3,1)/0/
      DATA GFORCEBW(-4,1)/1/
      DATA GFORCEBW(-1,2)/0/
      DATA GFORCEBW(-2,2)/1/
      DATA GFORCEBW(-3,2)/2/
      DATA GFORCEBW(-4,2)/1/
      DATA GFORCEBW(-1,3)/2/
      DATA GFORCEBW(-2,3)/1/
      DATA GFORCEBW(-3,3)/0/
      DATA GFORCEBW(-4,3)/1/
      DATA GFORCEBW(-1,4)/2/
      DATA GFORCEBW(-2,4)/1/
      DATA GFORCEBW(-3,4)/2/
      DATA GFORCEBW(-4,4)/1/
      DATA GFORCEBW(-1,5)/0/
      DATA GFORCEBW(-2,5)/1/
      DATA GFORCEBW(-3,5)/0/
      DATA GFORCEBW(-4,5)/1/
      DATA GFORCEBW(-1,6)/2/
      DATA GFORCEBW(-2,6)/1/
      DATA GFORCEBW(-3,6)/0/
      DATA GFORCEBW(-4,6)/1/
      DATA GFORCEBW(-1,7)/0/
      DATA GFORCEBW(-2,7)/1/
      DATA GFORCEBW(-3,7)/2/
      DATA GFORCEBW(-4,7)/1/
      DATA GFORCEBW(-1,8)/2/
      DATA GFORCEBW(-2,8)/1/
      DATA GFORCEBW(-3,8)/2/
      DATA GFORCEBW(-4,8)/1/
""")

        fortran_model = helas_call_writers.FortranHelasCallWriter(mymodel)

        # Test dname.mg
        writer = writers.FileWriter(self.give_pos('test'))
        exporter.write_dname_file(writer,
                                  "P"+me.get('processes')[0].shell_string())
        writer.close()
        self.assertFileContains('test', "DIRNAME=P0_emep_n1n1_n1_emsl2pa_n1_emsl2pa\n")
        # Test iproc.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_iproc_file(writer, 0)
        writer.close()
        self.assertFileContains('test', "      1\n")
        # Test maxamps.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        # Extract ncolor
        ncolor = max(1, len(me.get('color_basis')))
        exporter.write_maxamps_file(writer,
                                     len(me.get_all_amplitudes()),
                                     ncolor,
                                     len(me.get('processes')),
                                     1)
        writer.close()
        self.assertFileContains('test',
                                "      INTEGER    MAXAMPS, MAXFLOW, " + \
                                "MAXPROC, MAXSPROC\n" + \
                                "      PARAMETER (MAXAMPS=8, MAXFLOW=1)\n" + \
                                "      PARAMETER (MAXPROC=1, MAXSPROC=1)\n")
        # Test mg.sym
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_mg_sym_file(writer, me)
        writer.close()
        self.assertFileContains('test', """      3
      2
      3
      6
      2
      4
      7
      2
      5
      8\n""")
        # Test ncombs.inc
        nexternal, ninitial = me.get_nexternal_ninitial()
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_ncombs_file(writer, nexternal)
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    N_MAX_CL
      PARAMETER (N_MAX_CL=256)\n""")
        # Test nexternal.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_nexternal_file(writer, nexternal, ninitial)
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=8)
      INTEGER    NINCOMING
      PARAMETER (NINCOMING=2)\n""")
        # Test ngraphs.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_ngraphs_file(writer, len(mapconfigs))
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    N_MAX_CG
      PARAMETER (N_MAX_CG=8)\n""")
        # Test props.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_props_file(writer, me, s_and_t_channels)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
                         """      PRMASS(-1,1)  = ZERO
      PRWIDTH(-1,1) = ZERO
      POW(-1,1) = 1
      PRMASS(-2,1)  = ABS(MNEU1)
      PRWIDTH(-2,1) = ABS(WNEU1)
      POW(-2,1) = 1
      PRMASS(-3,1)  = ZERO
      PRWIDTH(-3,1) = ZERO
      POW(-3,1) = 1
      PRMASS(-4,1)  = ABS(MNEU1)
      PRWIDTH(-4,1) = ABS(WNEU1)
      POW(-4,1) = 1
      PRMASS(-5,1)  = ABS(MSL2)
      PRWIDTH(-5,1) = ABS(WSL2)
      POW(-5,1) = 2
      PRMASS(-1,2)  = ZERO
      PRWIDTH(-1,2) = ZERO
      POW(-1,2) = 1
      PRMASS(-2,2)  = ABS(MNEU1)
      PRWIDTH(-2,2) = ABS(WNEU1)
      POW(-2,2) = 1
      PRMASS(-3,2)  = ABS(MSL2)
      PRWIDTH(-3,2) = ABS(WSL2)
      POW(-3,2) = 2
      PRMASS(-4,2)  = ABS(MNEU1)
      PRWIDTH(-4,2) = ABS(WNEU1)
      POW(-4,2) = 1
      PRMASS(-5,2)  = ABS(MSL2)
      PRWIDTH(-5,2) = ABS(WSL2)
      POW(-5,2) = 2
      PRMASS(-1,3)  = ABS(MSL2)
      PRWIDTH(-1,3) = ABS(WSL2)
      POW(-1,3) = 2
      PRMASS(-2,3)  = ABS(MNEU1)
      PRWIDTH(-2,3) = ABS(WNEU1)
      POW(-2,3) = 1
      PRMASS(-3,3)  = ZERO
      PRWIDTH(-3,3) = ZERO
      POW(-3,3) = 1
      PRMASS(-4,3)  = ABS(MNEU1)
      PRWIDTH(-4,3) = ABS(WNEU1)
      POW(-4,3) = 1
      PRMASS(-5,3)  = ABS(MSL2)
      PRWIDTH(-5,3) = ABS(WSL2)
      POW(-5,3) = 2
      PRMASS(-1,4)  = ABS(MSL2)
      PRWIDTH(-1,4) = ABS(WSL2)
      POW(-1,4) = 2
      PRMASS(-2,4)  = ABS(MNEU1)
      PRWIDTH(-2,4) = ABS(WNEU1)
      POW(-2,4) = 1
      PRMASS(-3,4)  = ABS(MSL2)
      PRWIDTH(-3,4) = ABS(WSL2)
      POW(-3,4) = 2
      PRMASS(-4,4)  = ABS(MNEU1)
      PRWIDTH(-4,4) = ABS(WNEU1)
      POW(-4,4) = 1
      PRMASS(-5,4)  = ABS(MSL2)
      PRWIDTH(-5,4) = ABS(WSL2)
      POW(-5,4) = 2
      PRMASS(-1,5)  = ZERO
      PRWIDTH(-1,5) = ZERO
      POW(-1,5) = 1
      PRMASS(-2,5)  = ABS(MNEU1)
      PRWIDTH(-2,5) = ABS(WNEU1)
      POW(-2,5) = 1
      PRMASS(-3,5)  = ZERO
      PRWIDTH(-3,5) = ZERO
      POW(-3,5) = 1
      PRMASS(-4,5)  = ABS(MNEU1)
      PRWIDTH(-4,5) = ABS(WNEU1)
      POW(-4,5) = 1
      PRMASS(-5,5)  = ABS(MSL2)
      PRWIDTH(-5,5) = ABS(WSL2)
      POW(-5,5) = 2
      PRMASS(-1,6)  = ABS(MSL2)
      PRWIDTH(-1,6) = ABS(WSL2)
      POW(-1,6) = 2
      PRMASS(-2,6)  = ABS(MNEU1)
      PRWIDTH(-2,6) = ABS(WNEU1)
      POW(-2,6) = 1
      PRMASS(-3,6)  = ZERO
      PRWIDTH(-3,6) = ZERO
      POW(-3,6) = 1
      PRMASS(-4,6)  = ABS(MNEU1)
      PRWIDTH(-4,6) = ABS(WNEU1)
      POW(-4,6) = 1
      PRMASS(-5,6)  = ABS(MSL2)
      PRWIDTH(-5,6) = ABS(WSL2)
      POW(-5,6) = 2
      PRMASS(-1,7)  = ZERO
      PRWIDTH(-1,7) = ZERO
      POW(-1,7) = 1
      PRMASS(-2,7)  = ABS(MNEU1)
      PRWIDTH(-2,7) = ABS(WNEU1)
      POW(-2,7) = 1
      PRMASS(-3,7)  = ABS(MSL2)
      PRWIDTH(-3,7) = ABS(WSL2)
      POW(-3,7) = 2
      PRMASS(-4,7)  = ABS(MNEU1)
      PRWIDTH(-4,7) = ABS(WNEU1)
      POW(-4,7) = 1
      PRMASS(-5,7)  = ABS(MSL2)
      PRWIDTH(-5,7) = ABS(WSL2)
      POW(-5,7) = 2
      PRMASS(-1,8)  = ABS(MSL2)
      PRWIDTH(-1,8) = ABS(WSL2)
      POW(-1,8) = 2
      PRMASS(-2,8)  = ABS(MNEU1)
      PRWIDTH(-2,8) = ABS(WNEU1)
      POW(-2,8) = 1
      PRMASS(-3,8)  = ABS(MSL2)
      PRWIDTH(-3,8) = ABS(WSL2)
      POW(-3,8) = 2
      PRMASS(-4,8)  = ABS(MNEU1)
      PRWIDTH(-4,8) = ABS(WNEU1)
      POW(-4,8) = 1
      PRMASS(-5,8)  = ABS(MSL2)
      PRWIDTH(-5,8) = ABS(WSL2)
      POW(-5,8) = 2
""")

        # Test symfact.dat
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(me)
        exporter.write_symfact_file(\
            writers.FortranWriter(self.give_pos('test')),
            symmetry)
        #print open(self.give_pos('test')).read()
        goal_symfact_dat = """ 1    2
 2    2
 3    2
 4    2
 5    -1
 6    -3
 7    -2
 8    -4
"""

        # Test reversed order of decay specifications
        # e- e+ > se+ se-, se- > e- n1, se+ > e+ n1

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay2, mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        self.assertEqual(myfortranmodel.get_matrix_element_calls(me),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mneu1,NHEL(4),-1*IC(4),W(1,4))
CALL HIOXXX(W(1,4),W(1,3),MGVX350,Msl2,Wsl2,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,4))
CALL OXXXXX(P(0,6),Mneu1,NHEL(6),+1*IC(6),W(1,3))
CALL HIOXXX(W(1,4),W(1,3),MGVX494,Msl2,Wsl2,W(1,6))
CALL JIOXXX(W(1,1),W(1,2),MGVX12,zero,zero,W(1,3))
# Amplitude(s) for diagram number 1
CALL VSSXXX(W(1,3),W(1,6),W(1,5),MGVX56,AMP(1))
CALL FSIXXX(W(1,1),W(1,5),MGVX494,Mneu1,Wneu1,W(1,3))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,3),W(1,2),W(1,6),MGVX350,AMP(2))""".split('\n'))

    def test_export_complicated_majorana_decay_chain(self):
        """Test complicated decay chain z e+ > n2 el+, n2 > e- e+ n1
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'el-',
                      'antiname':'el+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist) - 1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # Neutralinos
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'mn1',
                      'width':'zero',
                      'texname':r'\chi_0^1',
                      'antitexname':r'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        mypartlist.append(base_objects.Particle({'name':'n2',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'mn2',
                      'width':'wn2',
                      'texname':r'\chi_0^2',
                      'antitexname':r'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n2 = mypartlist[len(mypartlist) - 1]

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'zmass',
                      'width':'zwidth',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[len(mypartlist) - 1]

        # Coupling of e to Z
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZL'},
                      'orders':{'QED':1}}))

        # Coupling of n1 to n2 and z
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN12'},
                      'orders':{'QED':1}}))

        # Coupling of n1 and n2 to e and el
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN1M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN1P'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n2, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2P'},
                      'orders':{'QED':1}}))

        # Coupling of n2 to z
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN22'},
                      'orders':{'QED':1}}))

        # Coupling of el to z
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [z, \
                                             seminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZELEL'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel,
                                           'forbidden_particles':[1000022]})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        result = myfortranmodel.get_matrix_element_calls(me)
        goal = """CALL VXXXXX(P(0,1),zmass,NHEL(1),-1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),mn1,NHEL(5),-1*IC(5),W(1,5))
CALL JIOXXX(W(1,4),W(1,3),GZL,zmass,zwidth,W(1,6))
CALL FVIXXX(W(1,5),W(1,6),GZN12,mn2,wn2,W(1,7))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,6))
CALL FVOXXX(W(1,2),W(1,1),GZL,zero,zero,W(1,8))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,8),W(1,6),GELN2P,AMP(1))
CALL HIOXXX(W(1,5),W(1,3),GELN1P,Msl2,Wsl2,W(1,9))
CALL FSIXXX(W(1,4),W(1,9),GELN2M,mn2,wn2,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,5),W(1,8),W(1,6),GELN2P,AMP(2))
CALL OXXXXX(P(0,5),mn1,NHEL(5),+1*IC(5),W(1,9))
CALL HIOXXX(W(1,4),W(1,9),GELN1M,Msl2,Wsl2,W(1,3))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,9))
CALL FSICXX(W(1,9),W(1,3),GELN2P,mn2,wn2,W(1,4))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,4),W(1,8),W(1,6),GELN2P,AMP(3))
CALL FVIXXX(W(1,7),W(1,1),GZN22,mn2,wn2,W(1,8))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,8),W(1,2),W(1,6),GELN2P,AMP(4))
CALL FVIXXX(W(1,5),W(1,1),GZN22,mn2,wn2,W(1,8))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,8),W(1,2),W(1,6),GELN2P,AMP(5))
CALL FVIXXX(W(1,4),W(1,1),GZN22,mn2,wn2,W(1,8))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,8),W(1,2),W(1,6),GELN2P,AMP(6))
CALL HVSXXX(W(1,1),W(1,6),-GZELEL,Msl2,Wsl2,W(1,8))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,7),W(1,2),W(1,8),GELN2P,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,5),W(1,2),W(1,8),GELN2P,AMP(8))
# Amplitude(s) for diagram number 9
CALL IOSXXX(W(1,4),W(1,2),W(1,8),GELN2P,AMP(9))""".split('\n')


        self.assertEqual(result, goal)

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(me)[0][0],
                         " JAMP(1) = AMP(1)+(-1.000000000000000d+00)*AMP(2)+(-1.000000000000000d+00)*AMP(3)+AMP(4)+(-1.000000000000000d+00)*AMP(5)+(-1.000000000000000d+00)*AMP(6)+AMP(7)+(-1.000000000000000d+00)*AMP(8)+(-1.000000000000000d+00)*AMP(9)")
                         

    def test_duplicate_lorentz_structures(self):
        """Test duplicate Lorentz structure with only one color structure.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'mass':'zmass',
                      'width':'zwidth',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[len(mypartlist) - 1]

        # u ubar z coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             z]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L4', 'L7'],
                      'couplings':{(0,0):'GC_23',(0,1):'GC_24'},
                      'orders':{'QED':1}}))



        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 2)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=True)

        self.assertEqual(sum([len(diagram.get('amplitudes')) for diagram in \
                          me.get('diagrams')]), 2)

        for i, amp in enumerate(me.get_all_amplitudes()):
            self.assertEqual(amp.get('number'), i + 1)

        self.assertEqual(len(me.get('color_basis')), 2)

        exporter = export_v4.ProcessExporterFortranME()

#        self.assertEqual(exporter.get_JAMP_lines(me),
#                         ([' JAMP(1) = (-1.0)*AMP(1)', ' JAMP(2) = AMP(2)'], 0))
        

    def test_generate_helas_diagrams_gg_gogo(self):
        """Testing the v4 helas diagram generation g g > go go,
        where there is no extra sign.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # A gluino
        mypartlist.append(base_objects.Particle({'name': 'go',
                                                 'antiname': 'go',
                                                 'spin': 2,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'MGO',
                                                 'width': 'WGO',
                                                 'pdg_code': 1000021,
                                                 'texname': 'go',
                                                 'antitexname': 'go',
                                                 'line': 'straight',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))
        go = mypartlist[len(mypartlist) - 1]

        # Triple glue coupling
        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([g, g, g]),
            'lorentz': [''],
            'couplings': {(0, 0): 'G'},
            'orders': {'QCD': 1}
            }))

        # go-go-g coupling
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([go, go, g]),
            'lorentz': [''],
            'couplings': {(0, 0): 'GGI'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        goal_string = """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MGO,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MGO,NHEL(4),+1*IC(4),W(1,4))
CALL JVVXXX(W(1,1),W(1,2),G,ZERO,ZERO,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,3),W(1,4),W(1,5),GGI,AMP(1))
CALL FVIXXX(W(1,3),W(1,1),GGI,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,5),W(1,4),W(1,2),GGI,AMP(2))
CALL FVOXXX(W(1,4),W(1,1),GGI,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 3
CALL IOVXXX(W(1,3),W(1,5),W(1,2),GGI,AMP(3))""".split('\n')

        result = helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                 get_matrix_element_calls(matrix_element)
        for i in range(max(len(goal_string),len(result))):
            self.assertEqual(result[i], goal_string[i])

    def test_generate_ufo_helas_diagrams_gg_gogo(self):
        """Testing minus sign on the FFV_1 UFO helas go-go-g coupling.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # A gluino
        mypartlist.append(base_objects.Particle({'name': 'go',
                                                 'antiname': 'go',
                                                 'spin': 2,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'MGO',
                                                 'width': 'WGO',
                                                 'pdg_code': 1000021,
                                                 'texname': 'go',
                                                 'antitexname': 'go',
                                                 'line': 'straight',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))
        go = mypartlist[len(mypartlist) - 1]

        # Triple glue coupling
        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([g, g, g]),
            'lorentz': ['VVV1'],
            'couplings': {(0, 0): 'G'},
            'orders': {'QCD': 1}
            }))

        # go-go-g coupling
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([go, go, g]),
            'lorentz': ['FFV1'],
            'couplings': {(0, 0): 'GGI'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        goal_string = """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MGO,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MGO,NHEL(4),+1*IC(4),W(1,4))
CALL VVV1_1(W(1,1),W(1,2),G,ZERO,ZERO,W(1,5))
# Amplitude(s) for diagram number 1
CALL FFV1_0(W(1,3),W(1,4),W(1,5),GGI,AMP(1))
CALL FFV1_2(W(1,3),W(1,1),-GGI,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,5),W(1,4),W(1,2),GGI,AMP(2))
CALL FFV1_1(W(1,4),W(1,1),GGI,MGO,WGO,W(1,5))
# Amplitude(s) for diagram number 3
CALL FFV1_0(W(1,3),W(1,5),W(1,2),GGI,AMP(3))""".split('\n')

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                 get_matrix_element_calls(matrix_element)
        for i in range(max(len(goal_string),len(result))):
            self.assertEqual(result[i], goal_string[i])

    def test_majorana_conjugate_process(self):
        """Test process e+ e- > n1 n2 z, which needs conjugate wfs.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'el-',
                      'antiname':'el+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist) - 1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # Neutralinos
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'mn1',
                      'width':'zero',
                      'texname':r'\chi_0^1',
                      'antitexname':r'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        mypartlist.append(base_objects.Particle({'name':'n2',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'mn2',
                      'width':'wn2',
                      'texname':r'\chi_0^2',
                      'antitexname':r'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n2 = mypartlist[len(mypartlist) - 1]

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'zmass',
                      'width':'zwidth',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[len(mypartlist) - 1]

        # Coupling of n1 to n2 and z
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GZN12'},
                      'orders':{'QED':1}}))

        # Coupling of n1 and n2 to e and el
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':['FFS1'],
                      'couplings':{(0, 0):'GELN1M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':['FFS1'],
                      'couplings':{(0, 0):'GELN1P'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n2, \
                                             seminus]),
                      'color': [],
                      'lorentz':['FFS1'],
                      'couplings':{(0, 0):'GELN2M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':['FFS1'],
                      'couplings':{(0, 0):'GELN2P'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':1000023,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(mymodel)

        result = myfortranmodel.get_matrix_element_calls(matrix_element)

        #print "\n".join(result)

        goal = """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),zero,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),mn2,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),mn1,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zmass,NHEL(5),+1*IC(5),W(1,5))
CALL VXXXXX(P(0,6),zmass,NHEL(6),+1*IC(6),W(1,6))
CALL FFS1_3(W(1,1),W(1,3),GELN2M,Msl2,Wsl2,W(1,7))
CALL FFV1C1_1(W(1,4),W(1,5),GZN12,mn2,wn2,W(1,8))
CALL FFS1C1_2(W(1,2),W(1,7),GELN1P,mn1,zero,W(1,9))
# Amplitude(s) for diagram number 1
CALL FFV1_0(W(1,9),W(1,8),W(1,6),GZN12,AMP(1))
CALL FFV1C1_1(W(1,4),W(1,6),GZN12,mn2,wn2,W(1,7))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,9),W(1,7),W(1,5),GZN12,AMP(2))
CALL FFS1_3(W(1,1),W(1,4),GELN1M,Msl2,Wsl2,W(1,9))
CALL FFV1_1(W(1,3),W(1,5),GZN12,mn1,zero,W(1,10))
CALL FFS1C1_2(W(1,2),W(1,9),GELN2P,mn2,wn2,W(1,11))
# Amplitude(s) for diagram number 3
CALL FFV1C1_0(W(1,11),W(1,10),W(1,6),GZN12,AMP(3))
CALL FFV1_1(W(1,3),W(1,6),GZN12,mn1,zero,W(1,9))
# Amplitude(s) for diagram number 4
CALL FFV1C1_0(W(1,11),W(1,9),W(1,5),GZN12,AMP(4))
CALL FFS1C1_3(W(1,2),W(1,3),GELN2P,Msl2,Wsl2,W(1,11))
CALL FFS1_2(W(1,1),W(1,11),GELN1M,mn1,zero,W(1,3))
# Amplitude(s) for diagram number 5
CALL FFV1_0(W(1,3),W(1,8),W(1,6),GZN12,AMP(5))
# Amplitude(s) for diagram number 6
CALL FFV1_0(W(1,3),W(1,7),W(1,5),GZN12,AMP(6))
CALL FFS1C1_3(W(1,2),W(1,4),GELN1P,Msl2,Wsl2,W(1,3))
CALL FFS1_2(W(1,1),W(1,3),GELN2M,mn2,wn2,W(1,4))
# Amplitude(s) for diagram number 7
CALL FFV1C1_0(W(1,4),W(1,10),W(1,6),GZN12,AMP(7))
# Amplitude(s) for diagram number 8
CALL FFV1C1_0(W(1,4),W(1,9),W(1,5),GZN12,AMP(8))
CALL FFS1_3(W(1,1),W(1,10),GELN1M,Msl2,Wsl2,W(1,4))
# Amplitude(s) for diagram number 9
CALL FFS1C1_0(W(1,2),W(1,7),W(1,4),GELN2P,AMP(9))
CALL FFS1_3(W(1,1),W(1,7),GELN2M,Msl2,Wsl2,W(1,4))
# Amplitude(s) for diagram number 10
CALL FFS1C1_0(W(1,2),W(1,10),W(1,4),GELN1P,AMP(10))
CALL FFS1_3(W(1,1),W(1,9),GELN1M,Msl2,Wsl2,W(1,4))
# Amplitude(s) for diagram number 11
CALL FFS1C1_0(W(1,2),W(1,8),W(1,4),GELN2P,AMP(11))
CALL FFS1_3(W(1,1),W(1,8),GELN2M,Msl2,Wsl2,W(1,4))
# Amplitude(s) for diagram number 12
CALL FFS1C1_0(W(1,2),W(1,9),W(1,4),GELN1P,AMP(12))""".split('\n')

        for i in range(max(len(result), len(goal))):
             self.assertEqual(result[i], goal[i])

    def test_configs_ug_ttxz(self):
        """Test configs.inc which previously failed.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # u and t quarks
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

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
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

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'mass':'zmass',
                      'width':'zwidth',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[len(mypartlist) - 1]

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # t tbar z couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t, \
                                             antit, \
                                             z]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.append(base_objects.Leg({'id':-6}))
        myleglist.append(base_objects.Leg({'id':23}))
        myleglist.append(base_objects.Leg({'id':2}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, (s_and_t_channels, nqcd_list) = \
                 exporter.write_configs_file(writer, me)
        writer.close()
        #print( open(self.give_pos('test')).read())

        # 2 21 > 6 -6 23  2
        # 1  2   3  4  5  6
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/5,3/
      DATA (SPROP(I,-1,1),I=1,1)/6/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA (SPROP(I,-2,1),I=1,1)/21/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/6,-2/
      DATA (SPROP(I,-3,1),I=1,1)/2/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,4/
      DATA (SPROP(I,-1,2),I=1,1)/-6/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,3/
      DATA (SPROP(I,-2,2),I=1,1)/21/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/6,-2/
      DATA (SPROP(I,-3,2),I=1,1)/2/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/1,6/
      DATA TPRID(-1,3)/21/
      DATA (SPROP(I,-1,3),I=1,1)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,4/
      DATA TPRID(-2,3)/6/
      DATA (SPROP(I,-2,3),I=1,1)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/-2,5/
      DATA TPRID(-3,3)/6/
      DATA (SPROP(I,-3,3),I=1,1)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,3/
      DATA TPRID(-4,3)/21/
      DATA (SPROP(I,-4,3),I=1,1)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/5,4/
      DATA (SPROP(I,-1,4),I=1,1)/-6/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/1,6/
      DATA TPRID(-2,4)/21/
      DATA (SPROP(I,-2,4),I=1,1)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,-1/
      DATA TPRID(-3,4)/6/
      DATA (SPROP(I,-3,4),I=1,1)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,3/
      DATA TPRID(-4,4)/21/
      DATA (SPROP(I,-4,4),I=1,1)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/1,6/
      DATA TPRID(-1,5)/21/
      DATA (SPROP(I,-1,5),I=1,1)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,3/
      DATA TPRID(-2,5)/6/
      DATA (SPROP(I,-2,5),I=1,1)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/-2,5/
      DATA TPRID(-3,5)/6/
      DATA (SPROP(I,-3,5),I=1,1)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/-3,4/
      DATA TPRID(-4,5)/21/
      DATA (SPROP(I,-4,5),I=1,1)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA (SPROP(I,-1,6),I=1,1)/6/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/1,6/
      DATA TPRID(-2,6)/21/
      DATA (SPROP(I,-2,6),I=1,1)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,-1/
      DATA TPRID(-3,6)/6/
      DATA (SPROP(I,-3,6),I=1,1)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/-3,4/
      DATA TPRID(-4,6)/21/
      DATA (SPROP(I,-4,6),I=1,1)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA TSTRATEGY(7)/2/
      DATA (IFOREST(I,-1,7),I=1,2)/5,3/
      DATA (SPROP(I,-1,7),I=1,1)/6/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/1,6/
      DATA TPRID(-2,7)/21/
      DATA (SPROP(I,-2,7),I=1,1)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/-2,4/
      DATA TPRID(-3,7)/6/
      DATA (SPROP(I,-3,7),I=1,1)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/-3,-1/
      DATA TPRID(-4,7)/21/
      DATA (SPROP(I,-4,7),I=1,1)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA TSTRATEGY(8)/2/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA (SPROP(I,-1,8),I=1,1)/-6/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/1,6/
      DATA TPRID(-2,8)/21/
      DATA (SPROP(I,-2,8),I=1,1)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/-2,3/
      DATA TPRID(-3,8)/6/
      DATA (SPROP(I,-3,8),I=1,1)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,-1/
      DATA TPRID(-4,8)/21/
      DATA (SPROP(I,-4,8),I=1,1)/0/
C     Diagram 9
      DATA MAPCONFIG(9)/9/
      DATA TSTRATEGY(9)/2/
      DATA (IFOREST(I,-1,9),I=1,2)/5,3/
      DATA (SPROP(I,-1,9),I=1,1)/6/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/4,-1/
      DATA (SPROP(I,-2,9),I=1,1)/21/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/1,-2/
      DATA TPRID(-3,9)/2/
      DATA (SPROP(I,-3,9),I=1,1)/0/
      DATA (IFOREST(I,-4,9),I=1,2)/-3,6/
      DATA TPRID(-4,9)/21/
      DATA (SPROP(I,-4,9),I=1,1)/0/
C     Diagram 10
      DATA MAPCONFIG(10)/10/
      DATA TSTRATEGY(10)/2/
      DATA (IFOREST(I,-1,10),I=1,2)/5,4/
      DATA (SPROP(I,-1,10),I=1,1)/-6/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/-1,3/
      DATA (SPROP(I,-2,10),I=1,1)/21/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/1,-2/
      DATA TPRID(-3,10)/2/
      DATA (SPROP(I,-3,10),I=1,1)/0/
      DATA (IFOREST(I,-4,10),I=1,2)/-3,6/
      DATA TPRID(-4,10)/21/
      DATA (SPROP(I,-4,10),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/10/
""")
        
        # Test maxconfigs.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_maxconfigs_file(writer, [me])
        writer.close()
        self.assertFileContains('test',
                                """      INTEGER LMAXCONFIGS
      PARAMETER(LMAXCONFIGS=18)
""")

    def test_configs_4f_decay(self):
        """Test configs.inc for 4f decay process that failed in v. 1.3.27
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # u, b and t quarks
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

        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
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
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
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

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w+',
                      'antitexname':'w-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # t b w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t, \
                                             antib, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             b, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # t t u u 4F couplings

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antiu,
                                             t,
                                             antiu,
                                             t]),
                      'color': [color.ColorString([color.T(1,0),color.T(3,2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0,0):'GC_27'},
                      'orders':{'NP':2}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antit,
                                             u,
                                             antit,
                                             u]),
                      'color': [color.ColorString([color.T(1,0),color.T(3,2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0,0):'GC_27'},
                      'orders':{'NP':2}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.append(base_objects.Leg({'id':6}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-6}))
        myleglist.append(base_objects.Leg({'id':2}))
        myleglist.append(base_objects.Leg({'id':2}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-5}))
        myleglist.append(base_objects.Leg({'id':-24}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mydecay1.set('decay_chains', base_objects.ProcessList([mydecay2]))
        myproc.set('decay_chains', base_objects.ProcessList([mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        helas_decay_chain = helas_objects.HelasDecayChainProcess(myamplitude)
        me = helas_decay_chain.combine_decay_chain_processes()[0]

        myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(mymodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, (s_and_t_channels, nqcd_list) = \
                 exporter.write_configs_file(writer, me)
        writer.close()
        #print open(self.give_pos('test')).read()

        # 2  2 > -24 -5 2 2 -24 -5 2 2
        # 1  2     3  4 5 6   7  8 9 10
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/8,7/
      DATA (SPROP(I,-1,1),I=1,1)/-6/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/10,9/
      DATA (SPROP(I,-2,1),I=1,1)/1/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/-2,-1/
      DATA (SPROP(I,-3,1),I=1,1)/6/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/4,3/
      DATA (SPROP(I,-4,1),I=1,1)/-6/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/6,5/
      DATA (SPROP(I,-5,1),I=1,1)/1/
      DATA TPRID(-5,1)/0/
      DATA (IFOREST(I,-6,1),I=1,2)/-5,-4/
      DATA (SPROP(I,-6,1),I=1,1)/6/
      DATA TPRID(-6,1)/0/
      DATA (IFOREST(I,-7,1),I=1,2)/-3,-6/
      DATA (SPROP(I,-7,1),I=1,1)/1/
      DATA TPRID(-7,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/1/
""")
        
    def test_configs_long_decay(self):
        """Test configs.inc which previously failed.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
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

        # A w
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w+',
                      'spin':3,
                      'mass':'wmass',
                      'width':'wwidth',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wplus.set('is_part', False)

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # t b w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t, \
                                             antib, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             b, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antib, \
                                             b, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':21}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, (s_and_t_channels, nqcd_list) = \
                                      exporter.write_configs_file(writer, me)
        writer.close()

        #print open(self.give_pos('test')).read()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/3,2/
      DATA (SPROP(I,-1,1),I=1,1)/5/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA (SPROP(I,-2,1),I=1,1)/6/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,-2/
      DATA (SPROP(I,-3,1),I=1,1)/6/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA TSTRATEGY(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/3,2/
      DATA (SPROP(I,-1,2),I=1,1)/5/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/5,-1/
      DATA (SPROP(I,-2,2),I=1,1)/5/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/4,-2/
      DATA (SPROP(I,-3,2),I=1,1)/6/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA TSTRATEGY(3)/2/
      DATA (IFOREST(I,-1,3),I=1,2)/4,3/
      DATA (SPROP(I,-1,3),I=1,1)/6/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,2/
      DATA (SPROP(I,-2,3),I=1,1)/6/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/5,-2/
      DATA (SPROP(I,-3,3),I=1,1)/6/
      DATA TPRID(-3,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA TSTRATEGY(4)/2/
      DATA (IFOREST(I,-1,4),I=1,2)/4,3/
      DATA (SPROP(I,-1,4),I=1,1)/6/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/5,-1/
      DATA (SPROP(I,-2,4),I=1,1)/6/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,2/
      DATA (SPROP(I,-3,4),I=1,1)/6/
      DATA TPRID(-3,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA TSTRATEGY(5)/2/
      DATA (IFOREST(I,-1,5),I=1,2)/5,3/
      DATA (SPROP(I,-1,5),I=1,1)/5/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,2/
      DATA (SPROP(I,-2,5),I=1,1)/5/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/4,-2/
      DATA (SPROP(I,-3,5),I=1,1)/6/
      DATA TPRID(-3,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA TSTRATEGY(6)/2/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA (SPROP(I,-1,6),I=1,1)/5/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/4,-1/
      DATA (SPROP(I,-2,6),I=1,1)/6/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,2/
      DATA (SPROP(I,-3,6),I=1,1)/6/
      DATA TPRID(-3,6)/0/
C     Number of configs
      DATA MAPCONFIG(0)/6/
""")

    def test_configs_8fs(self):
        """Test configs.inc for 8fs process which previously failed.
        """

        diagrams = save_load_object.load_from_file(\
                                              os.path.join(_input_file_path,
                                                           'test_8fs.pkl'))

        goal_schannels = [[[8, 6, -1], [7, -1, -2], [-2, 5, -3],
                           [-3, 3, -4], [4, -4, -5]],
                          [],
                          [[6, 5, -1], [8, -1, -2], [7, -2, -3], [-3, 3, -4]],
                          [[6, 5, -1]]]
        goal_tchannels = [[[1, -5, -6]],
                          [[1, 4, -1], [-1, 7, -2], [-2, 3, -3],
                           [-3, 6, -4], [-4, 5, -5], [-5, 8, -6]],
                          [[1, 4, -5], [-5, -4, -6]],
                          [[1, 4, -2], [-2, 7, -3], [-3, 8, -4],
                           [-4, 3, -5], [-5, -1, -6]]]
                          

        import models.import_ufo
        mymodel = models.import_ufo.import_model('sm')

        wf_dict = {}
        for (idiag, diagram) in enumerate(diagrams):

            schannels, tchannels = diagram.get('amplitudes')[0].\
                                         get_s_and_t_channels(2, mymodel, 20)

            self.assertEqual([[l.get('number') for l in v.get('legs')] for v \
                              in schannels],
                             goal_schannels[idiag])
            self.assertEqual([[l.get('number') for l in v.get('legs')] for v \
                              in tchannels],
                             goal_tchannels[idiag]) 

    def test_get_color_pdg_antipdg(self):
        """Testing that both pdg and antipdg are included in get_color
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        # A neutrino
        mypartlist.append(base_objects.Particle({'name':'ve',
                      'antiname':'ve~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\nu_e',
                      'antitexname':'\bar\nu_e',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':12,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        nu = mypartlist[len(mypartlist) - 1]
        nubar = copy.copy(nu)
        nubar.set('is_part', False)

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

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        a = mypartlist[len(mypartlist) - 1]

        # Coupling of W- e+ nu_e

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             nu, \
                                             Wminus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [nubar, \
                                             eminus, \
                                             Wplus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':12,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, (s_and_t_channels, nqcd_list) = \
                 exporter.write_configs_file(writer, matrix_element)
        writer.close()
#        print open(self.give_pos('test')).read()
        self.assertFileContains('test',
        """C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA TSTRATEGY(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/1,4/
      DATA TPRID(-1,1)/11/
      DATA (SPROP(I,-1,1),I=1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/-1,3/
      DATA TPRID(-2,1)/24/
      DATA (SPROP(I,-2,1),I=1,1)/0/
C     Number of configs
      DATA MAPCONFIG(0)/1/
""")
        
        # Test get_color.f output
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_colors_file(writer, matrix_element)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
        """      FUNCTION GET_COLOR(IPDG)
      IMPLICIT NONE
      INTEGER GET_COLOR, IPDG

      IF(IPDG.EQ.-24)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.-12)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.-11)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.11)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.12)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.22)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.24)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.1)THEN
C       This is dummy particle used in multiparticle vertices
        GET_COLOR=2
        RETURN
      ELSE
        WRITE(*,*)'Error: No color given for pdg ',IPDG
        GET_COLOR=0
        RETURN
      ENDIF
      END

""")

class AlohaFortranWriterTest(unittest.TestCase):
    """ A basic test to see if the Aloha Fortran Writter is working """
    
    def setUp(self):
        """ check that old file are remove """
        try:
            os.remove('/tmp/FFV1_1.f')
        except:
            pass
    
    def test_header(self):
        """ test the header of a file """
        
        with misc.TMP_variable(sys, 'path', sys.path+[pjoin(MG5DIR,'models','sm')]):
            from models.sm.object_library import Lorentz
        import aloha.create_aloha as create_aloha
        
        FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')
        
        solution="""C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1_1(F2, V3, COUP, M1, W1,F1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(6)
      COMPLEX*16 F2(*)
      REAL*8 M1
      REAL*8 P1(0:3)
      COMPLEX*16 V3(*)
      REAL*8 W1
      COMPLEX*16 DENOM
      ENTRY FFV1_2(F2, V3, COUP, M1, W1,F1)

      F1(1) = +F2(1)+V3(1)
      F1(2) = +F2(2)+V3(2)
      P1(0) = -DBLE(F1(1))
      P1(1) = -DBLE(F1(2))
      P1(2) = -DIMAG(F1(2))
      P1(3) = -DIMAG(F1(1))
      DENOM = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI
     $ * W1))"""

        abstract_M = create_aloha.AbstractRoutineBuilder(FFV1).compute_routine(1)
        abstract_M.add_symmetry(2)
        abstract_M.write('/tmp','Fortran')
        
        self.assertTrue(os.path.exists('/tmp/FFV1_1.f'))
        textfile = open('/tmp/FFV1_1.f','r').read()
        split_sol = solution.split('\n')
        self.assertEqual(split_sol, textfile.split('\n')[:len(split_sol)])


class UFO_model_to_mg4_Test(unittest.TestCase):
    """Check the conversion model from UFO to MG4"""
    
    def setUp(self):
        import models.import_ufo
        self.mymodel = models.import_ufo.import_model('sm')
    
    
    def test_refactorize(self):
        """ test the separation of variable """
        
        mg4_model = export_v4.UFO_model_to_mg4(self.mymodel,'/dev/null')
        mg4_model.refactorize()
        
        # external parameters
        expected = ['aEWM1', 'mdl_Gf', 'aS', 'mdl_ymb', 'mdl_ymt', 'mdl_ymtau', 'mdl_MTA', 'mdl_MT', 'mdl_MB', 'mdl_MZ', 'mdl_MH', 'mdl_WT', 'mdl_WZ', 'mdl_WW', 'mdl_WH']
        expected.sort()
        solution = [param.name for param in mg4_model.params_ext]
        solution.sort()
        self.assertEqual(expected, solution)
        
        #  internal params
        self.assertEqual(len(mg4_model.params_dep), 2)
        self.assertEqual(len(mg4_model.params_indep), 34)
        
        # couplings
        self.assertEqual(len(mg4_model.coups_dep), 3)
        sol = ['GC_1', 'GC_2', 'GC_3', 'GC_5', 'GC_6', 'GC_7', 'GC_8', 'GC_15', 'GC_21', 'GC_31', 'GC_32', 'GC_33', 'GC_34', 'GC_35', 'GC_36', 'GC_37', 'GC_38', 'GC_50', 'GC_52', 'GC_53', 'GC_54', 'GC_55', 'GC_57', 'GC_58', 'GC_59', 'GC_60', 'GC_61', 'GC_62', 'GC_63', 'GC_64', 'GC_65', 'GC_66', 'GC_68', 'GC_69', 'GC_70', 'GC_71', 'GC_72', 'GC_74', 'GC_76', 'GC_77', 'GC_80', 'GC_81', 'GC_82', 'GC_83', 'GC_94', 'GC_95', 'GC_96', 'GC_98', 'GC_99', 'GC_100']

        self.assertEqual(sol, [ p.name for p in mg4_model.coups_indep])

        
        # MG4 use G and not aS as it basic object for alphas related computation
        # G is out of any list!
        self.assertNotIn('G', [p.name for p in mg4_model.params_dep])
        self.assertNotIn('G', [p.name for p in mg4_model.params_indep])
        # check that sqrt__aS is correctly set
        self.assertIn('mdl_sqrt__aS', [p.name for p in mg4_model.params_dep])
        self.assertNotIn('mdl_sqrt__aS', [p.name for p in mg4_model.params_indep])
        
        
    def test_case_sensitive(self):
        """ test that the case clash are dealt correctly """  
        
        mg4_model = export_v4.UFO_model_to_mg4(self.mymodel,'/dev/null')
        
        #check that they are no crash for normal model
        mg4_model.pass_parameter_to_case_insensitive()
        
        # edit model in order to add new parameter with name: CW / Cw / Mz / Mz2
        CW = base_objects.ParamCardVariable('CW', 100, 'MASS', 40)
        CWc = base_objects.ModelVariable( 'CW', 'Mz**2 * Mz2' , 'real')
        Cw = base_objects.ModelVariable( 'Cw', 'Mz**2 * Mz2 * CW' , 'real')
        Mz = base_objects.ParamCardVariable('MZ', 100, 'MASS', 41)
        Mzc = base_objects.ParamCardVariable('Mz', 100, 'MASS', 42)
        Mz2 = base_objects.ParamCardVariable('Mz2', 100, 'MASS', 43)

        mg4_model.model['parameters'][('external',)].append(CW)        
        mg4_model.model['parameters'][()].append(CWc)
        mg4_model.model['parameters'][()].append(Cw)
        mg4_model.model['parameters'][('external',)].append(Mz)
        mg4_model.model['parameters'][('external',)].append(Mzc)
        mg4_model.model['parameters'][('external',)].append(Mz2)


        mg4_model.pass_parameter_to_case_insensitive()
        self.assertEqual(CWc.name,'cw')
        self.assertEqual(CWc.expr,'mz__2**2 * Mz2')
        self.assertEqual(Cw.name,'cw__2')
        self.assertEqual(Cw.expr,'mz__2**2 * Mz2 * cw__3')
        self.assertEqual(CW.name,'cw__3')

        self.assertEqual(Mzc.name,'mz__2')
        


if __name__ == '__main__':
        """Write out pkl file with helas diagram for test_configs_8fs
        """

        import models.import_ufo
        mymodel = models.import_ufo.import_model('sm')
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':-5}))
        myleglist.append(base_objects.Leg({'id':21}))
        myleglist.append(base_objects.Leg({'id':21}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        import madgraph.iolibs.drawing_eps as draw
        filename = os.path.join('diagrams_' + \
                                myamplitude.get('process').shell_string() + ".eps")
        plot = draw.MultiEpsDiagramDrawer(myamplitude.get('diagrams'),
                                          filename,
                                          model=mymodel,
                                                amplitude='',
                                          legend=myamplitude.get('process').input_string())

        plot.draw()  



        me = save_load_object.save_to_file(\
                       os.path.join(_input_file_path, 'test_8fs.pkl'),
                       [me.get('diagrams')[323], me.get('diagrams')[954],
                        me.get('diagrams')[1123], me.get('diagrams')[1139]])
        
