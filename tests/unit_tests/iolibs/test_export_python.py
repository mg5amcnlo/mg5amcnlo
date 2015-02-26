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

"""Unit test library for the export Python format routines"""

import StringIO
import copy
import fractions
import os
import re

import tests.unit_tests as unittest

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.misc as misc

from madgraph import MG5DIR

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx

import tests.unit_tests.core.test_helas_objects as test_helas_objects

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# IOExportPythonTest
#===============================================================================
class IOExportPythonTest(unittest.TestCase):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()

    def setUp(self):

        # Set up model
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # u and c quarkd and their antiparticles
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'ZERO',
                      'width':'ZERO',
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

        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'MC',
                      'width':'ZERO',
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

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'ZERO',
                      'width':'ZERO',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        g = mypartlist[len(mypartlist) - 1]

        # A photon
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
        z = mypartlist[len(mypartlist) - 1]

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GC_10'},
                      'orders':{'QCD':1}}))

        # Gamma couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1, 0)])],
                      'lorentz':['FFV2', 'FFV5'],
                      'couplings':{(0,0): 'GC_35', (0,1): 'GC_47'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('name', 'sm')

        self.mypythonmodel = helas_call_writers.PythonUFOHelasCallWriter(self.mymodel)
    
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
                                       'model':self.mymodel})
        
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.mymatrixelement = helas_objects.HelasMultiProcess(myamplitude)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':4,
                                           'state':False,
                                           'number' : 1}))
        myleglist.append(base_objects.Leg({'id':-4,
                                         'state':False,
                                           'number' : 2}))
        myleglist.append(base_objects.Leg({'id':4,
                                         'state':True,
                                           'number' : 3}))
        myleglist.append(base_objects.Leg({'id':-4,
                                         'state':True,
                                           'number' : 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        self.mymatrixelement.get('matrix_elements')[0].\
                                               get('processes').append(myproc)

        self.exporter = export_python.ProcessExporterPython(\
            self.mymatrixelement, self.mypythonmodel)
        
    def test_python_export_functions(self):
        """Test functions used by the Python export"""

        # Test the exporter setup
        self.assertEqual(self.exporter.model, self.mymodel)
        self.assertEqual(self.exporter.matrix_elements, self.mymatrixelement.get('matrix_elements'))

    def test_get_python_matrix_methods(self):
        """Test getting the matrix methods for Python for a matrix element."""
        
        goal_method = (\
"""class Matrix_0_uux_uux(object):

    def __init__(self):
        \"\"\"define the object\"\"\"
        self.clean()

    def clean(self):
        self.jamp = []

    def smatrix(self,p, model):
        #  
        #  MadGraph5_aMC@NLO v. %(version)s, %(date)s
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: u u~ > u u~
        # Process: c c~ > c c~
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = 4
        ndiags = 4
        ncomb = 16
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \\
        [1,-1,-1,1],
        [1,-1,-1,-1],
        [1,-1,1,1],
        [1,-1,1,-1],
        [1,1,-1,1],
        [1,1,-1,-1],
        [1,1,1,1],
        [1,1,1,-1],
        [-1,-1,-1,1],
        [-1,-1,-1,-1],
        [-1,-1,1,1],
        [-1,-1,1,-1],
        [-1,1,-1,1],
        [-1,1,-1,-1],
        [-1,1,1,1],
        [-1,1,1,-1]]
        denominator = 36
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0.] * ndiags
        self.helEvals = []
        ans = 0.
        for hel in helicities:
            t = self.matrix(p, hel, model)
            ans = ans + t
            self.helEvals.append([hel, t.real / denominator ])
        ans = ans / denominator
        return ans.real

    def matrix(self, p, hel, model):
        #  
        #  MadGraph5_aMC@NLO v. %(version)s, %(date)s
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: u u~ > u u~
        # Process: c c~ > c c~
        #  
        #  
        # Process parameters
        #  
        ngraphs = 4
        nexternal = 4
        nwavefuncs = 5
        ncolor = 2
        ZERO = 0.
        #  
        # Color matrix
        #  
        denom = [1,1];
        cf = [[9,3],
        [3,9]];
        #
        # Model parameters
        #
        WZ = model.get('parameter_dict')["WZ"]
        MZ = model.get('parameter_dict')["MZ"]
        GC_47 = model.get('coupling_dict')["GC_47"]
        GC_35 = model.get('coupling_dict')["GC_35"]
        GC_10 = model.get('coupling_dict')["GC_10"]
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w[0] = ixxxxx(p[0],ZERO,hel[0],+1)
        w[1] = oxxxxx(p[1],ZERO,hel[1],-1)
        w[2] = oxxxxx(p[2],ZERO,hel[2],+1)
        w[3] = ixxxxx(p[3],ZERO,hel[3],-1)
        w[4]= FFV1_3(w[0],w[1],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[3],w[2],w[4],GC_10)
        w[4]= FFV2_5_3(w[0],w[1],GC_35,GC_47,MZ,WZ)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV2_5_0(w[3],w[2],w[4],GC_35,GC_47)
        w[4]= FFV1_3(w[0],w[2],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 3
        amp[2]= FFV1_0(w[3],w[1],w[4],GC_10)
        w[4]= FFV2_5_3(w[0],w[2],GC_35,GC_47,MZ,WZ)
        # Amplitude(s) for diagram number 4
        amp[3]= FFV2_5_0(w[3],w[1],w[4],GC_35,GC_47)

        jamp = [None] * ncolor

        jamp[0] = +1./6.*amp[0]-amp[1]+1./2.*amp[2]
        jamp[1] = -1./2.*amp[0]-1./6.*amp[2]+amp[3]

        self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        self.amp2[1]+=abs(amp[1]*amp[1].conjugate())
        self.amp2[2]+=abs(amp[2]*amp[2].conjugate())
        self.amp2[3]+=abs(amp[3]*amp[3].conjugate())
        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        return matrix
""" % misc.get_pkg_info()).split('\n')

        exporter = export_python.ProcessExporterPython(self.mymatrixelement,
                                                       self.mypythonmodel)

        matrix_methods = exporter.get_python_matrix_methods()["0_uux_uux"].\
                          split('\n')

        self.assertEqual(matrix_methods, goal_method)
        

    def test_run_python_matrix_element(self):
        """Test a complete running of a Python matrix element without
        writing any files"""

        # Import the SM
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':model})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)

        # Create only the needed aloha routines
        wanted_lorentz = mymatrixelement.get_used_lorentz()

        aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
        aloha_model.compute_subset(wanted_lorentz)

        # Write out the routines in Python
        aloha_routines = []
        for routine in aloha_model.values():
            aloha_routines.append(routine.write(output_dir = None,
                                                language = 'Python').\
                                  replace('import wavefunctions',
                                          'import aloha.template_files.wavefunctions as wavefunctions'))
        # Define the routines to be available globally
        for routine in aloha_routines:
            exec(routine, globals())

        # Write the matrix element(s) in Python
        mypythonmodel = helas_call_writers.PythonUFOHelasCallWriter(\
                                                             model)
        exporter = export_python.ProcessExporterPython(\
                                                     mymatrixelement,
                                                     mypythonmodel)
        matrix_methods = exporter.get_python_matrix_methods()

        # Calculate parameters and couplings
        full_model = model_reader.ModelReader(model)
        
        full_model.set_parameters_and_couplings()

        # Define a momentum
        p = [[0.5000000e+03, 0.0000000e+00,  0.0000000e+00,  0.5000000e+03,  0.0000000e+00],
             [0.5000000e+03,  0.0000000e+00,  0.0000000e+00, -0.5000000e+03,  0.0000000e+00],
             [0.4585788e+03,  0.1694532e+03,  0.3796537e+03, -0.1935025e+03,  0.6607249e-05],
             [0.3640666e+03, -0.1832987e+02, -0.3477043e+03,  0.1063496e+03,  0.7979012e-05],
             [0.1773546e+03, -0.1511234e+03, -0.3194936e+02,  0.8715287e+02,  0.1348699e-05]]

        # Evaluate the matrix element for the given momenta

        answer = 1.39189717257175028e-007
        for process in matrix_methods.keys():
            # Define Python matrix element for process
            exec(matrix_methods[process])
            # Calculate the matrix element for the momentum p
            value = eval("Matrix_0_epem_aaa().smatrix(p, full_model)")
            self.assertTrue(abs(value-answer)/answer < 1e-6,
                            "Value is: %.9e should be %.9e" % \
                            (abs(value), answer))


    def test_export_matrix_element_python_madevent_group(self):
        """Test the result of exporting a subprocess group matrix element"""

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

        # Gluon and photon couplings to quarks
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
                                            [antiu, \
                                             u, \
                                             a]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             a]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [],
                      'lorentz':['L1', 'L2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [],
                      'lorentz':['L1', 'L2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        

        procs = [[2,-2,21,21], [2,-2,2,-2]]
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

        subprocess_group = group_subprocs.SubProcessGroup.\
                           group_amplitudes(amplitudes, "madevent")[0]

        # Test amp2 lines
        helas_writer = helas_call_writers.PythonUFOHelasCallWriter(mymodel)
        python_exporter = export_python.ProcessExporterPython(subprocess_group,
                                                              helas_writer)

        amp2_lines = \
                 python_exporter.get_amp2_lines(subprocess_group.\
                                        get('matrix_elements')[0],
                                        subprocess_group.get('diagram_maps')[0])
        self.assertEqual(amp2_lines,
                         ['self.amp2[0]+=abs(amp[0]*amp[0].conjugate())', 'self.amp2[1]+=abs(amp[1]*amp[1].conjugate())', 'self.amp2[2]+=abs(amp[2]*amp[2].conjugate())'])

