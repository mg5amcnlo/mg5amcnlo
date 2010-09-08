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
import madgraph.iolibs.helas_call_writers as helas_call_writer
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
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

        self.mypythonmodel = helas_call_writer.PythonUFOHelasCallWriter(self.mymodel)
    
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

    def smatrix(self,p, model):
        #  
        #  MadGraph 5 v. %(version)s, %(date)s
        #  By the MadGraph Development Team
        #  Please visit us at https://launchpad.net/madgraph5
        # 
        # MadGraph StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: u u~ > u u~
        # Process: c c~ > c c~
        #  
        #  
        # CONSTANTS
        #  
        nexternal = 4
        ncomb = 16
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \\
        [-1,-1,-1,-1],
        [-1,-1,-1,1],
        [-1,-1,1,-1],
        [-1,-1,1,1],
        [-1,1,-1,-1],
        [-1,1,-1,1],
        [-1,1,1,-1],
        [-1,1,1,1],
        [1,-1,-1,-1],
        [1,-1,-1,1],
        [1,-1,1,-1],
        [1,-1,1,1],
        [1,1,-1,-1],
        [1,1,-1,1],
        [1,1,1,-1],
        [1,1,1,1]]
        denominator = 36
        # ----------
        # BEGIN CODE
        # ----------
        ans = 0.
        for hel in helicities:
            t = self.matrix(p, hel, model)
            ans = ans + t
        ans = ans / denominator
        return ans.real

    def matrix(self, p, hel, model):
        #  
        #  MadGraph 5 v. 0.4.3, 2010-07-21
        #  By the MadGraph Development Team
        #  Please visit us at https://launchpad.net/madgraph5
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
        ngraphs = 10
        nexternal = 4
        nwavefuncs = 10
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
        w[4] = FFV1_3(w[0],w[1],GC_10,ZERO, ZERO)
        # Amplitude(s) for diagram number 1
        amp[0] = FFV1_0(w[3],w[2],w[4],GC_10)
        w[5] = FFV2_3(w[0],w[1],GC_35,MZ, WZ)
        w[6] = FFV5_3(w[0],w[1],GC_47,MZ, WZ)
        # Amplitude(s) for diagram number 2
        amp[1] = FFV2_0(w[3],w[2],w[5],GC_35)
        amp[2] = FFV5_0(w[3],w[2],w[5],GC_47)
        amp[3] = FFV2_0(w[3],w[2],w[6],GC_35)
        amp[4] = FFV5_0(w[3],w[2],w[6],GC_47)
        w[7] = FFV1_3(w[0],w[2],GC_10,ZERO, ZERO)
        # Amplitude(s) for diagram number 3
        amp[5] = FFV1_0(w[3],w[1],w[7],GC_10)
        w[8] = FFV2_3(w[0],w[2],GC_35,MZ, WZ)
        w[9] = FFV5_3(w[0],w[2],GC_47,MZ, WZ)
        # Amplitude(s) for diagram number 4
        amp[6] = FFV2_0(w[3],w[1],w[8],GC_35)
        amp[7] = FFV5_0(w[3],w[1],w[8],GC_47)
        amp[8] = FFV2_0(w[3],w[1],w[9],GC_35)
        amp[9] = FFV5_0(w[3],w[1],w[9],GC_47)

        jamp = [None] * ncolor
        jamp[0] = +1./6.*amp[0]-amp[1]-amp[2]-amp[3]-amp[4]+1./2.*amp[5]
        jamp[1] = -1./2.*amp[0]-1./6.*amp[5]+amp[6]+amp[7]+amp[8]+amp[9]

        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   

        return matrix
""" % misc.get_pkg_info()).split('\n')

        exporter = export_python.ProcessExporterPython(self.mymatrixelement,
                                                       self.mypythonmodel)

        matrix_methods = exporter.get_python_matrix_methods()["0_uux_uux"].\
                          split('\n')

        for iline in range(len(goal_method)):
            self.assertEqual(matrix_methods[iline],
                             goal_method[iline])
        

    def test_run_python_matrix_element(self):
        """Test a complete running of a Python matrix element without
        writing any files"""

        # Import the SM
        model = import_ufo.import_model('sm')

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
            exec("\n".join(routine.split("\n")[:-1]), globals())

        # Write the matrix element(s) in Python
        mypythonmodel = helas_call_writer.PythonUFOHelasCallWriter(\
                                                             model)
        exporter = export_python.ProcessExporterPython(\
                                                     mymatrixelement,
                                                     mypythonmodel)
        matrix_methods = exporter.get_python_matrix_methods()

        # Define the routines (locally is enough)
        for matrix_method in matrix_methods.values():
            exec(matrix_method)

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
            value = eval("Matrix_%s().smatrix(p, full_model)" % process)
            self.assertTrue(abs(value-answer)/answer < 1e-6,
                            "Value is: %.9e should be %.9e" % \
                            (abs(value), answer))

