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
"""Unit test library for the export Pythia8 format routines"""

from __future__ import absolute_import
import copy
import fractions
import os
import re
import tests.IOTests as IOTests
from tests import test_manager

import tests.unit_tests as unittest

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_cpp as export_cpp
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.group_subprocs as group_subprocs

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.misc as misc

from madgraph import MG5DIR

import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers

pjoin = os.path.join

#===============================================================================
# IOExportPythia8Test
#===============================================================================
class IOExportPythia8Test(IOTests.IOTestManager, test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    created_files = ['test.h', 'test.cc'
                    ]

    def assertFileContains(self,*args,**opts):
        """Wrapper to make sure that the function assertFileContains, of
        test_file_writers is used. We cannot put IOTests.IOTestManager last
        in the hierarchy because the structure requires it to be first always."""
        return test_file_writers.CheckFileCreate.assertFileContains(
                                                               self,*args,**opts)

    def test_cpp_wavefunction_template_has_flavor_mask_placeholders(self):
        template = open(pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files',
                              'cpp_process_wavefunctions.inc')).read()
        self.assertIn('%(flavor_mask_decl)s', template)
        self.assertIn('%(flavor_mask_setup)s', template)

    def setUp(self):

        test_file_writers.CheckFileCreate.clean_files

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

        # A gluino
        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':8,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'go',
                      'antitexname':'go',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        go = mypartlist[len(mypartlist) - 1]

        # A sextet diquark
        mypartlist.append(base_objects.Particle({'name':'six',
                      'antiname':'six~',
                      'spin':1,
                      'color':6,
                      'mass':'MSIX',
                      'width':'WSIX',
                      'texname':'six',
                      'antitexname':'sixbar',
                      'line':'straight',
                      'charge':4./3.,
                      'pdg_code':6000001,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        six = mypartlist[len(mypartlist) - 1]
        antisix = copy.copy(six)
        antisix.set('is_part', False)
        

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

        # Gluon couplings to gluinos
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             go, \
                                             g]),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GC_8'},
                      'orders':{'QCD':1}}))

        # Sextet couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             u, \
                                             antisix]),
                      'color': [color.ColorString([color.K6Bar(2, 0, 1)])],
                      'lorentz':['FFS1'],
                      'couplings':{(0,0): 'GC_24'},
                      'orders':{'QSIX':1}}))

        myinterlist.append(base_objects.Interaction({
                       'id': 5,
                       'particles': base_objects.ParticleList(\
                                             [antiu, \
                                             antiu, \
                                             six]),
                      'color': [color.ColorString([color.K6(2, 0, 1)])],
                      'lorentz':['FFS1'],
                      'couplings':{(0,0): 'GC_24'},
                      'orders':{'QSIX':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('name', 'sm')

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
                                       'model':self.mymodel,
                                       'orders':{'QSIX':0}})
        
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
                                       'model':self.mymodel,
                                       'orders':{'QSIX':0}})

        self.mymatrixelement.get('matrix_elements')[0].\
                                               get('processes').append(myproc)

        self.mycppwriter = helas_call_writer.CPPUFOHelasCallWriter(self.mymodel)
    
        self.pythia8_exporter = export_cpp.OneProcessExporterPythia8(\
            self.mymatrixelement, self.mycppwriter,
            process_string = "q q~ > q q~")
        
        self.cpp_exporter = export_cpp.OneProcessExporterCPP(\
            self.mymatrixelement, self.mycppwriter,
            process_string = "q q~ > q q~")

    tearDown = test_file_writers.CheckFileCreate.clean_files




    def test_cppwriter_with_ifdefs(self):
        input_string = "hello"
        input_string ="""#include "CPPProcess.h"

  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess( bool verbose, /* clang-format off */
                          bool debug )
    : m_verbose( verbose )
    , m_debug( debug )
#ifndef MGONGPU_HARDCODE_PARAM
    , m_pars( 0 )
#endif
    , m_masses()
  {
    // Helicities for the process [NB do keep 'static' for this constexpr array, see issue #283]
    // *** NB There is no automatic check yet that these are in the same order as Fortran! #569 ***
    static constexpr short tHel[ncomb][npar] = {
      { -1, -1, -1, 1 },
      { -1, -1, -1, -1 },
      { -1, -1, 1, 1 },
      { -1, -1, 1, -1 },
      { -1, 1, -1, 1 },
      { -1, 1, -1, -1 },
      { -1, 1, 1, 1 },
      { -1, 1, 1, -1 },
      { 1, -1, -1, 1 },
      { 1, -1, -1, -1 },
      { 1, -1, 1, 1 },
      { 1, -1, 1, -1 },
      { 1, 1, -1, 1 },
      { 1, 1, -1, -1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, -1 } };
    static constexpr short tFlavors[nmaxflavor][npar] = {
      { 20, 20, 5, 5 } };
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( cHel, tHel, ncomb * npar * sizeof( short ) );
    gpuMemcpyToSymbol( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#else
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
    memcpy( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#endif

    // Enable SIGFPE traps for Floating Point Exceptions
#ifdef MGONGPUCPP_DEBUG
    fpeEnable();
#endif
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array (CUDA and C++)
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      //std::cout << "sigmaKin_setGoodHel ihel=" << ihel << ( isGoodHel[ihel] ? " true" : " false" ) << std::endl;
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( dcNGoodHel, &nGoodHel, sizeof( int ) );
    gpuMemcpyToSymbol( dcGoodHel, goodHel, ncomb * sizeof( int ) );
#endif
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  add_and_select_hel( int* allselhel,          // output: helicity selection[nevt]
                      const fptype* allrndhel, // input: random numbers[nevt] for helicity selection
                      fptype* ghelAllMEs,      // input/tmp: allMEs for nGoodHel <= ncomb individual/runningsum helicities (index is ighel)
                      fptype* allMEs,          // output: allMEs[nevt], final sum over helicities
                      const int nevt )         // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // Compute the sum of MEs over all good helicities (defer this after the helicity loop to avoid breaking streams parall>
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      allMEs[ievt] += ghelAllMEs[ighel * nevt + ievt];
      ghelAllMEs[ighel * nevt + ievt] = allMEs[ievt]; // reuse the buffer to store the running sum for helicity selection
    }
    // Event-by-event random choice of helicity #403
    //printf( "select_hel: ievt=%4d rndhel=%f\n", ievt, allrndhel[ievt] );
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      if( allrndhel[ievt] < ( ghelAllMEs[ighel * nevt + ievt] / allMEs[ievt] ) )
      {
        const int ihelF = dcGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allselhel[ievt] = ihelF;
        //printf( "select_hel: ievt=%4d ihel=%4d\n", ievt, ihelF );
        break;
      }
    }
    return;
  }
#endif

  //--------------------------------------------------------------------------"""

        writer = writers.CPPWriter(self.give_pos('cppprocess.cc'))
        writer.write(input_string)
        goal_string = """#include "CPPProcess.h"

  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess( bool verbose, /* clang-format off */
                          bool debug )
    : m_verbose( verbose )
    , m_debug( debug )
#ifndef MGONGPU_HARDCODE_PARAM
    , m_pars( 0 )
#endif
    , m_masses()
  {
    // Helicities for the process [NB do keep 'static' for this constexpr array, see issue #283]
    // *** NB There is no automatic check yet that these are in the same order as Fortran! #569 ***
    static constexpr short tHel[ncomb][npar] = {
      { -1, -1, -1, 1 },
      { -1, -1, -1, -1 },
      { -1, -1, 1, 1 },
      { -1, -1, 1, -1 },
      { -1, 1, -1, 1 },
      { -1, 1, -1, -1 },
      { -1, 1, 1, 1 },
      { -1, 1, 1, -1 },
      { 1, -1, -1, 1 },
      { 1, -1, -1, -1 },
      { 1, -1, 1, 1 },
      { 1, -1, 1, -1 },
      { 1, 1, -1, 1 },
      { 1, 1, -1, -1 },
      { 1, 1, 1, 1 },
      { 1, 1, 1, -1 } };
    static constexpr short tFlavors[nmaxflavor][npar] = {
      { 20, 20, 5, 5 } };
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( cHel, tHel, ncomb * npar * sizeof( short ) );
    gpuMemcpyToSymbol( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#else
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
    memcpy( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#endif

    // Enable SIGFPE traps for Floating Point Exceptions
#ifdef MGONGPUCPP_DEBUG
    fpeEnable();
#endif
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array (CUDA and C++)
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      //std::cout << "sigmaKin_setGoodHel ihel=" << ihel << ( isGoodHel[ihel] ? " true" : " false" ) << std::endl;
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( dcNGoodHel, &nGoodHel, sizeof( int ) );
    gpuMemcpyToSymbol( dcGoodHel, goodHel, ncomb * sizeof( int ) );
#endif
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  add_and_select_hel( int* allselhel,          // output: helicity selection[nevt]
                      const fptype* allrndhel, // input: random numbers[nevt] for helicity selection
                      fptype* ghelAllMEs,      // input/tmp: allMEs for nGoodHel <= ncomb individual/runningsum helicities (index is ighel)
                      fptype* allMEs,          // output: allMEs[nevt], final sum over helicities
                      const int nevt )         // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // Compute the sum of MEs over all good helicities (defer this after the helicity loop to avoid breaking streams parall>
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      allMEs[ievt] += ghelAllMEs[ighel * nevt + ievt];
      ghelAllMEs[ighel * nevt + ievt] = allMEs[ievt]; // reuse the buffer to store the running sum for helicity selection
    }
    // Event-by-event random choice of helicity #403
    //printf( "select_hel: ievt=%4d rndhel=%f
", ievt, allrndhel[ievt] );
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      if( allrndhel[ievt] < ( ghelAllMEs[ighel * nevt + ievt] / allMEs[ievt] ) )
      {
        const int ihelF = dcGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allselhel[ievt] = ihelF;
        //printf( "select_hel: ievt=%4d ihel=%4d
", ievt, ihelF );
        break;
      }
    }
    return;
  }
#endif

  //--------------------------------------------------------------------------"""

        self.assertFileContains('cppprocess.cc', goal_string)




    def disabled_test_write_process_files(self):
        """Test writing the .h  and .cc Pythia file for a matrix element"""

        export_cpp.generate_process_files_pythia8(self.mymatrixelement,
                                                      self.mycppwriter,
                                                      process_string = "q q~ > q q~",
                                                      path = "/tmp")
        
        print("Please try compiling the file /tmp/Sigma_sm_qqx_qqx.cc:")
        print("cd /tmp; g++ -c -I $PATH_TO_PYTHIA8/include Sigma_sm_qqx_qqx.cc.cc")


#===============================================================================
# ExportUFOModelPythia8Test
#===============================================================================




#===============================================================================
# IOExportPythia8Test
#===============================================================================
class IOExportMatchBox(unittest.TestCase,
                         test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    def setUp(self):

        if not hasattr(self, 'model'):
            self.mymodel = base_objects.Model()
            self.mymatrixelement = helas_objects.HelasMatrixElement()


        test_file_writers.CheckFileCreate.clean_files

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

        # A gluino
        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':8,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'go',
                      'antitexname':'go',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        go = mypartlist[len(mypartlist) - 1]

        # A sextet diquark
        mypartlist.append(base_objects.Particle({'name':'six',
                      'antiname':'six~',
                      'spin':1,
                      'color':6,
                      'mass':'MSIX',
                      'width':'WSIX',
                      'texname':'six',
                      'antitexname':'sixbar',
                      'line':'straight',
                      'charge':4./3.,
                      'pdg_code':6000001,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        six = mypartlist[len(mypartlist) - 1]
        antisix = copy.copy(six)
        antisix.set('is_part', False)
        

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

        # Gluon couplings to gluinos
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             go, \
                                             g]),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GC_8'},
                      'orders':{'QCD':1}}))

        # Sextet couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             u, \
                                             antisix]),
                      'color': [color.ColorString([color.K6Bar(2, 0, 1)])],
                      'lorentz':['FFS1'],
                      'couplings':{(0,0): 'GC_24'},
                      'orders':{'QSIX':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             antiu, \
                                             six]),
                      'color': [color.ColorString([color.K6(2, 0, 1)])],
                      'lorentz':['FFS1'],
                      'couplings':{(0,0): 'GC_24'},
                      'orders':{'QSIX':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('name', 'sm')

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
                                       'model':self.mymodel,
                                       'orders':{'QSIX':0}})
        
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
                                       'model':self.mymodel,
                                       'orders':{'QSIX':0}})

        self.mymatrixelement.get('matrix_elements')[0].\
                                               get('processes').append(myproc)

        self.mycppwriter = helas_call_writer.CPPUFOHelasCallWriter(self.mymodel)
    
#         self.pythia8_exporter = export_cpp.ProcessExporterMatchbox(\
#             self.mymatrixelement, self.mycppwriter,
#             process_string = "q q~ > q q~")
#         
#         self.cpp_exporter = export_cpp.ProcessExporterCPP(\
#             self.mymatrixelement, self.mycppwriter,
#             process_string = "q q~ > q q~")

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_fail_on_process_cc_file_uu_six(self):
        """Test writing the .cc Pythia file for u u > six"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number' : 1}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number' : 2}))
        myleglist.append(base_objects.Leg({'id':6000001,
                                           'number' : 3}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        mymatrixelement = helas_objects.HelasMultiProcess(myamplitude)

        exporter = export_cpp.OneProcessExporterMatchbox( mymatrixelement, self.mycppwriter, process_string="q q > six")
        
        self.assertRaises(export_cpp.OneProcessExporterCPP.ProcessExporterCPPError,
                          exporter.write_process_cc_file,
                          writers.CPPWriter(self.give_pos('test.cc')))

                          
    def test_write_match_go_process_cc_file(self):
        """Test writing the .cc C++ standalone file for u u~ > go go"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})
        
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMultiProcess(myamplitude)
        matrix_element.get('matrix_elements')[0].set('has_mirror_process',
                                                     True)

        exporter = export_cpp.OneProcessExporterMatchbox(matrix_element,
                                                 self.mycppwriter)

        exporter.write_process_cc_file(\
                  writers.CPPWriter(self.give_pos('test.cc')))

        goal_string = """int CPPProcess::colorstring(int i, int j) 
{
  static const double res[2][5] = {{3, 4, 2, 1, 0}, {4, 3, 2, 1, 0}}; 
  return res[i][j]; 
}"""

        #print open(self.give_pos('test.cc')).read()
        self.assertFileContains('test.cc', goal_string, partial=True)


class BrokenSymmetryCPPExportTest(unittest.TestCase):
    """Ensure C++ exporter receives decay-aware broken symmetry metadata."""

    def _make_process(self, model, initial_ids, final_ids, decays):
        legs = base_objects.LegList()
        for i, pid in enumerate(initial_ids + final_ids, 1):
            legs.append(base_objects.Leg({'id': pid,
                                          'state': i > len(initial_ids),
                                          'number': i}))
        process = base_objects.Process({'legs': legs,
                                        'model': model,
                                        'is_decay_chain': bool(decays)})
        decay_chains = base_objects.ProcessList()
        for parent_id, decay_finals in decays:
            decay_legs = base_objects.LegList([base_objects.Leg({'id': parent_id,
                                                                 'state': False,
                                                                 'number': 1})])
            for j, pid in enumerate(decay_finals, 2):
                decay_legs.append(base_objects.Leg({'id': pid,
                                                    'state': True,
                                                    'number': j}))
            decay_chains.append(base_objects.Process({'legs': decay_legs,
                                                      'model': model,
                                                      'is_decay_chain': True}))
        process.set('decay_chains', decay_chains)
        return process

    def test_cpp_export_decay_chain_broken_symmetry_metadata(self):
        model = import_ufo.import_model('sm')
        decay_process = self._make_process(model, [2, -2], [23, 23],
                                           [(23, [1, -1]), (23, [4, -4])])
        sym_data = export_v4.ProcessExporterFortran._get_broken_symmetry_data(decay_process, 2)
        replace_dict = {
            'process_lines': '',
            'model_name': 'sm',
            'initProc_lines': '',
            'reset_jamp_lines': '',
            'sigmaKin_lines': '',
            'sigmaHat_lines': 'return 0.;',
            'all_sigmaKin': '',
            'nexternal': 6,
            'nincoming': 2,
            'broken_sym_ncomponents': sym_data['ncomponents'],
            'broken_sym_nentries': sym_data['nentries'],
            'broken_sym_component_starts': ",".join(str(v) for v in sym_data['component_starts']),
            'broken_sym_component_ends': ",".join(str(v) for v in sym_data['component_ends']),
            'broken_sym_component_old_factors': ",".join(str(v) for v in sym_data['component_old_factors']),
            'broken_sym_pid_list': ",".join(str(v) for v in sym_data['pid_list']),
            'broken_sym_block_starts': ",".join(str(v) for v in sym_data['block_starts']),
            'broken_sym_block_lengths': ",".join(str(v) for v in sym_data['block_lengths'])
        }
        template_path = pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files',
                              'cpp_process_function_definitions.inc')
        with open(template_path) as stream:
            rendered = stream.read() % replace_dict
        self.assertIn('const int n_components = 3;', rendered)
        self.assertIn('const int comp_old[n_components] = {1,1,1};', rendered)
        self.assertIn('const int block_len[n_entries] = {2,2,1,1,1,1};', rendered)
