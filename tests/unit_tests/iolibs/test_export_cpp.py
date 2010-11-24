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

"""Unit test library for the export Pythia8 format routines"""

import StringIO
import copy
import fractions
import os
import re

import tests.unit_tests as unittest

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_cpp as export_cpp
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
from madgraph import MG5DIR

import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers

#===============================================================================
# IOExportPythia8Test
#===============================================================================
class IOExportPythia8Test(unittest.TestCase,
                         test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    created_files = ['test.h', 'test.cc'
                    ]

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

        self.mycppwriter = helas_call_writer.CPPUFOHelasCallWriter(self.mymodel)
    
        self.pythia8_exporter = export_cpp.ProcessExporterPythia8(\
            self.mymatrixelement, self.mycppwriter,
            process_string = "q q~ > q q~")
        
        self.cpp_exporter = export_cpp.ProcessExporterCPP(\
            self.mymatrixelement, self.mycppwriter,
            process_string = "q q~ > q q~")
        
    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_pythia8_export_functions(self):
        """Test functions used by the Pythia export"""

        # Test the exporter setup
        self.assertEqual(self.pythia8_exporter.model, self.mymodel)
        self.assertEqual(self.pythia8_exporter.matrix_elements, self.mymatrixelement.get('matrix_elements'))
        self.assertEqual(self.pythia8_exporter.process_string, "q q~ > q q~")
        self.assertEqual(self.pythia8_exporter.process_name, "Sigma_sm_qqx_qqx")
        self.assertEqual(self.pythia8_exporter.nexternal, 4)
        self.assertEqual(self.pythia8_exporter.ninitial, 2)
        self.assertEqual(self.pythia8_exporter.nfinal, 2)
        self.assertTrue(self.pythia8_exporter.single_helicities)
        self.assertEqual(self.pythia8_exporter.wavefunctions, self.mymatrixelement.get('matrix_elements')[0].get_all_wavefunctions())

        # Test get_process_influx
        processes = self.mymatrixelement.get('matrix_elements')[0].get('processes')
        self.assertEqual(self.pythia8_exporter.get_process_influx(), "qqbarSame")
        self.assertEqual(self.pythia8_exporter.get_id_masses(processes[0]), "")
        self.assertEqual(self.pythia8_exporter.get_id_masses(processes[1]), \
                        """int id3Mass() const {return 4;}
int id4Mass() const {return 4;}""")
        self.assertEqual(self.pythia8_exporter.get_resonance_lines(), \
                        "virtual int resonanceA() const {return 23;}")

    def test_write_process_h_file(self):
        """Test writing the .h Pythia file for a matrix element"""

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8
// by MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#ifndef Pythia8_Sigma_sm_qqx_qqx_H
#define Pythia8_Sigma_sm_qqx_qqx_H

#include <complex> 

#include "SigmaProcess.h"
#include "Parameters_sm.h"

using namespace std; 

namespace Pythia8 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u~ > u u~
// Process: c c~ > c c~
//--------------------------------------------------------------------------

class Sigma_sm_qqx_qqx : public Sigma2Process 
{
  public:

    // Constructor.
    Sigma_sm_qqx_qqx() {}

    // Initialize process.
    virtual void initProc(); 

    // Calculate flavour-independent parts of cross section.
    virtual void sigmaKin(); 

    // Evaluate sigmaHat(sHat).
    virtual double sigmaHat(); 

    // Select flavour, colour and anticolour.
    virtual void setIdColAcol(); 

    // Evaluate weight for decay angles.
    virtual double weightDecay(Event& process, int iResBeg, int iResEnd); 

    // Info on the subprocess.
    virtual string name() const {return "q q~ > q q~ (sm)";}

    virtual int code() const {return 10000;}

    virtual string inFlux() const {return "qqbarSame";}

    virtual int resonanceA() const {return 23;}
    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertM2() const {return true;}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int hel[]); 
    static const int nwavefuncs = 10; 
    std::complex<double> w[nwavefuncs][18]; 
    double matrix_uux_uux(); 

    // Constants for array limits
    static const int nexternal = 4; 
    static const int nprocesses = 1; 

    // Store the matrix element value from sigmaKin
    double matrix_element[nprocesses]; 

    // Color flows, used when selecting color
    double * jamp2[nprocesses]; 

    // Pointer to the model parameters
    Parameters_sm * pars; 

}; 

}  // end namespace Pythia

#endif  // Pythia8_Sigma_sm_qqx_qqx_H
""" % misc.get_pkg_info()

        self.pythia8_exporter.write_process_h_file(\
            writers.CPPWriter(self.give_pos('test.h')))

        #print open(self.give_pos('test.h')).read()

        self.assertFileContains('test.h', goal_string)

    def test_write_process_cc_file(self):
        """Test writing the .cc Pythia file for a matrix element"""

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
// by MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "Sigma_sm_qqx_qqx.h"
#include "hel_amps_sm.h"

using namespace Pythia8_sm; 

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > u u~
// Process: c c~ > c c~

//--------------------------------------------------------------------------
// Initialize process.

void Sigma_sm_qqx_qqx::initProc() 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  pars->setIndependentParameters(particleDataPtr, coupSMPtr); 
  pars->setIndependentCouplings(particleDataPtr, coupSMPtr); 
  // Set massive/massless matrix elements for c/b/mu/tau
  mcME = particleDataPtr->m0(4); 
  mbME = 0.; 
  mmuME = 0.; 
  mtauME = 0.; 
  jamp2[0] = new double[2]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void Sigma_sm_qqx_qqx::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(particleDataPtr, coupSMPtr, alpS); 
  pars->setDependentCouplings(particleDataPtr, coupSMPtr); 
  // Reset color flows
  for(int i = 0; i < 2; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 16; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  std::complex<double> * * wfs; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {-1, -1, -1, -1, -1, -1, -1,
      1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1,
      -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1,
      1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {36}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
  }

  // For now, call setupForME() here
  id1 = 2; 
  id2 = -2; 
  if( !setupForME())
  {
    return; 
  }

  if (sum_hel == 0 || ntry < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        calculate_wavefunctions(helicities[ihel]); 
        t[0] = matrix_uux_uux(); 
        double tsum = 0; 
        for(int iproc = 0; iproc < nprocesses; iproc++ )
        {
          matrix_element[iproc] += t[iproc]; 
          tsum += t[iproc]; 
        }
        // Store which helicities give non-zero result
        if (tsum != 0. && !goodhel[ihel])
        {
          goodhel[ihel] = true; 
          ngood++; 
          igood[ngood] = ihel; 
        }
      }
    }
    jhel = 0; 
    sum_hel = min(sum_hel, ngood); 
  }
  else
  {
    // Only use the "good" helicities
    for(int j = 0; j < sum_hel; j++ )
    {
      jhel++; 
      if (jhel >= ngood)
        jhel = 0; 
      double hwgt = double(ngood)/double(sum_hel); 
      int ihel = igood[jhel]; 
      calculate_wavefunctions(helicities[ihel]); 
      t[0] = matrix_uux_uux(); 
      for(int iproc = 0; iproc < nprocesses; iproc++ )
      {
        matrix_element[iproc] += t[iproc] * hwgt; 
      }
    }
  }

  for (int i = 0; i < nprocesses; i++ )
    matrix_element[i] /= denominators[i]; 



}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double Sigma_sm_qqx_qqx::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 4 && id2 == -4)
  {
    // Add matrix elements for processes with beams (4, -4)
    return matrix_element[0]; 
  }
  else if(id1 == 2 && id2 == -2)
  {
    // Add matrix elements for processes with beams (2, -2)
    return matrix_element[0]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void Sigma_sm_qqx_qqx::setIdColAcol() 
{
  if(id1 == 4 && id2 == -4)
  {
    // Pick one of the flavor combinations (4, -4)
    int flavors[1][2] = {4, -4}; 
    vector<double> probs; 
    double sum = matrix_element[0]; 
    probs.push_back(matrix_element[0]/sum); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
    id4 = flavors[choice][1]; 
  }
  else if(id1 == 2 && id2 == -2)
  {
    // Pick one of the flavor combinations (2, -2)
    int flavors[1][2] = {2, -2}; 
    vector<double> probs; 
    double sum = matrix_element[0]; 
    probs.push_back(matrix_element[0]/sum); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
    id4 = flavors[choice][1]; 
  }
  setId(id1, id2, id3, id4); 
  // Pick color flow
  int ncolor[1] = {2}; 
  if(id1 == 2 && id2 == -2 && id3 == 2 && id4 == -2 || id1 == 4 && id2 == -4 &&
      id3 == 4 && id4 == -4)
  {
    vector<double> probs; 
    double sum = jamp2[0][0] + jamp2[0][1]; 
    for(int i = 0; i < ncolor[0]; i++ )
      probs.push_back(jamp2[0][i]/sum); 
    int ic = rndmPtr->pick(probs); 
    static int col[2][8] = {1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1}; 
    setColAcol(col[ic][0], col[ic][1], col[ic][2], col[ic][3], col[ic][4],
        col[ic][5], col[ic][6], col[ic][7]);
  }
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process

double Sigma_sm_qqx_qqx::weightDecay(Event& process, int iResBeg, int iResEnd) 
{
  // Just use isotropic decay (default)
  return 1.; 
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void Sigma_sm_qqx_qqx::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  double p[nexternal][4]; 
  int i, j; 

  // Convert Pythia 4-vectors to double[]
  for(i = 0; i < nexternal; i++ )
  {
    p[i][0] = pME[i].e(); 
    p[i][1] = pME[i].px(); 
    p[i][2] = pME[i].py(); 
    p[i][3] = pME[i].pz(); 
  }

  // Calculate all wavefunctions
  ixxxxx(p[0], mME[0], hel[0], +1, w[0]); 
  oxxxxx(p[1], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[2], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[3], mME[3], hel[3], -1, w[3]); 
  FFV1_3(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[4]); 
  FFV2_3(w[0], w[1], pars->GC_35, pars->MZ, pars->WZ, w[5]); 
  FFV5_3(w[0], w[1], pars->GC_47, pars->MZ, pars->WZ, w[6]); 
  FFV1_3(w[0], w[2], pars->GC_10, pars->ZERO, pars->ZERO, w[7]); 
  FFV2_3(w[0], w[2], pars->GC_35, pars->MZ, pars->WZ, w[8]); 
  FFV5_3(w[0], w[2], pars->GC_47, pars->MZ, pars->WZ, w[9]); 


}
double Sigma_sm_qqx_qqx::matrix_uux_uux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 10; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> amp[ngraphs], jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {9, 3, 3, 9}; 
  // Calculate all amplitudes
  // Amplitude(s) for diagram number 1
  FFV1_0(w[3], w[2], w[4], pars->GC_10, amp[0]); 
  // Amplitude(s) for diagram number 2
  FFV2_0(w[3], w[2], w[5], pars->GC_35, amp[1]); 
  FFV5_0(w[3], w[2], w[5], pars->GC_47, amp[2]); 
  FFV2_0(w[3], w[2], w[6], pars->GC_35, amp[3]); 
  FFV5_0(w[3], w[2], w[6], pars->GC_47, amp[4]); 
  // Amplitude(s) for diagram number 3
  FFV1_0(w[3], w[1], w[7], pars->GC_10, amp[5]); 
  // Amplitude(s) for diagram number 4
  FFV2_0(w[3], w[1], w[8], pars->GC_35, amp[6]); 
  FFV5_0(w[3], w[1], w[8], pars->GC_47, amp[7]); 
  FFV2_0(w[3], w[1], w[9], pars->GC_35, amp[8]); 
  FFV5_0(w[3], w[1], w[9], pars->GC_47, amp[9]); 

  // Calculate color flows
  jamp[0] = +1./6. * amp[0] - amp[1] - amp[2] - amp[3] - amp[4] + 1./2. *
      amp[5];
  jamp[1] = -1./2. * amp[0] - 1./6. * amp[5] + amp[6] + amp[7] + amp[8] +
      amp[9];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}


}  // end namespace Pythia
""" % misc.get_pkg_info()

        exporter = export_cpp.ProcessExporterPythia8(self.mymatrixelement,
        self.mycppwriter, process_string = "q q~ > q q~")

        exporter.write_process_cc_file(\
        writers.CPPWriter(self.give_pos('test.cc')))

        self.assertFileContains('test.cc', goal_string)

    def test_write_cpp_go_process_cc_file(self):
        """Test writing the .cc C++ standalone file for g g > go go"""

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
                                       'model':self.mymodel})
        
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMultiProcess(myamplitude)

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
// by MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "CPPProcess.h"
#include "hel_amps_sm.h"

using namespace MG5_sm; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g g > go go

//--------------------------------------------------------------------------
// Initialize process.

void CPPProcess::initProc(string param_card_name) 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  SLHAReader slha(param_card_name); 
  pars->setIndependentParameters(slha); 
  pars->setIndependentCouplings(); 
  pars->printIndependentParameters(); 
  pars->printIndependentCouplings(); 
  // Set external particle masses for this matrix element
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->MGO); 
  mME.push_back(pars->MGO); 
  jamp2[0] = new double[6]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void CPPProcess::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  static bool firsttime = true; 
  if (firsttime)
  {
    pars->printDependentParameters(); 
    pars->printDependentCouplings(); 
    firsttime = false; 
  }

  // Reset color flows
  for(int i = 0; i < 6; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 16; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  std::complex<double> * * wfs; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {-1, -1, -1, -1, -1, -1, -1,
      1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1,
      -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1,
      1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {512}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
  }

  if (sum_hel == 0 || ntry < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        calculate_wavefunctions(helicities[ihel]); 
        t[0] = matrix_gg_gogo(); 
        double tsum = 0; 
        for(int iproc = 0; iproc < nprocesses; iproc++ )
        {
          matrix_element[iproc] += t[iproc]; 
          tsum += t[iproc]; 
        }
        // Store which helicities give non-zero result
        if (tsum != 0. && !goodhel[ihel])
        {
          goodhel[ihel] = true; 
          ngood++; 
          igood[ngood] = ihel; 
        }
      }
    }
    jhel = 0; 
    sum_hel = min(sum_hel, ngood); 
  }
  else
  {
    // Only use the "good" helicities
    for(int j = 0; j < sum_hel; j++ )
    {
      jhel++; 
      if (jhel >= ngood)
        jhel = 0; 
      double hwgt = double(ngood)/double(sum_hel); 
      int ihel = igood[jhel]; 
      calculate_wavefunctions(helicities[ihel]); 
      t[0] = matrix_gg_gogo(); 
      for(int iproc = 0; iproc < nprocesses; iproc++ )
      {
        matrix_element[iproc] += t[iproc] * hwgt; 
      }
    }
  }

  for (int i = 0; i < nprocesses; i++ )
    matrix_element[i] /= denominators[i]; 



}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double CPPProcess::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 21 && id2 == 21)
  {
    // Add matrix elements for processes with beams (21, 21)
    return matrix_element[0]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void CPPProcess::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  int i, j; 

  // Calculate all wavefunctions
  vxxxxx(p[0], mME[0], hel[0], -1, w[0]); 
  vxxxxx(p[1], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[2], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[3], mME[3], hel[3], -1, w[3]); 
  FFV1_1(w[2], w[0], pars->GC_8, pars->MGO, pars->WGO, w[4]); 
  FFV1_2(w[3], w[0], -pars->GC_8, pars->MGO, pars->WGO, w[5]); 


}
double CPPProcess::matrix_gg_gogo() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> amp[ngraphs], jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {6, 6, 6, 6, 6, 6}; 
  static const double cf[ncolor][ncolor] = {19, -2, -2, -2, -2, 4, -2, 19, -2,
      4, -2, -2, -2, -2, 19, -2, 4, -2, -2, 4, -2, 19, -2, -2, -2, -2, 4, -2,
      19, -2, 4, -2, -2, -2, -2, 19};
  // Calculate all amplitudes
  // Amplitude(s) for diagram number 1
  FFV1_0(w[3], w[4], w[1], pars->GC_8, amp[0]); 
  // Amplitude(s) for diagram number 2
  FFV1_0(w[5], w[2], w[1], pars->GC_8, amp[1]); 

  // Calculate color flows
  jamp[0] = +2. * (-amp[1]); 
  jamp[1] = +2. * (+amp[0]); 
  jamp[2] = +2. * (-amp[0] + amp[1]); 
  jamp[3] = +2. * (+amp[0]); 
  jamp[4] = +2. * (-amp[0] + amp[1]); 
  jamp[5] = +2. * (-amp[1]); 

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}



""" % misc.get_pkg_info()

        exporter = export_cpp.ProcessExporterCPP(matrix_element,
                                                 self.mycppwriter)

        exporter.write_process_cc_file(\
                  writers.CPPWriter(self.give_pos('test.cc')))

        self.assertFileContains('test.cc', goal_string)


    def disabled_test_write_process_files(self):
        """Test writing the .h  and .cc Pythia file for a matrix element"""

        export_cpp.generate_process_files_pythia8(self.mymatrixelement,
                                                      self.mycppwriter,
                                                      process_string = "q q~ > q q~",
                                                      path = "/tmp")
        
        print "Please try compiling the file /tmp/Sigma_sm_qqx_qqx.cc:"
        print "cd /tmp; g++ -c -I $PATH_TO_PYTHIA8/include Sigma_sm_qqx_qqx.cc.cc"


#===============================================================================
# ExportUFOModelPythia8Test
#===============================================================================
class ExportUFOModelPythia8Test(unittest.TestCase,
                                test_file_writers.CheckFileCreate):

    created_files = [
                    ]

    def disabled_setUp(self):

        model_pkl = os.path.join(MG5DIR, 'models','sm','model.pkl')
        if os.path.isfile(model_pkl):
            self.model = save_load_object.load_from_file(model_pkl)
        else:
            self.model = import_ufo.import_model('sm')
        self.model_builder = export_cpp.UFOModelConverterPythia8(\
                                                            self.model, "/tmp")
        
        test_file_writers.CheckFileCreate.clean_files

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def disabled_test_read_aloha_template_files(self):
        """Test reading the ALOHA template .h and .cc files"""

        template_h = self.model_builder.read_aloha_template_files("h")
        self.assertTrue(template_h)
        for file_lines in template_h:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1)
        template_cc = self.model_builder.read_aloha_template_files("cc")
        self.assertTrue(template_cc)
        for file_lines in template_cc:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1) 
       
    def disabled_test_write_aloha_functions(self):
        """Test writing function declarations and definitions"""

        template_h_files = []
        template_cc_files = []

        aloha_model = create_aloha.AbstractALOHAModel(\
                                         self.model_builder.model.get('name'))
        aloha_model.compute_all(save=False)
        for abstracthelas in dict(aloha_model).values():
            abstracthelas.write('/tmp', 'CPP')
            #abstracthelas.write('/tmp', 'Fortran')

        print "Please try compiling the files /tmp/*.cc and /tmp/*.f:"
        print "cd /tmp; g++ -c *.cc; gfortran -c *.f"
        

    def disabled_test_write_aloha_routines(self):
        """Test writing the aloha .h and.cc files"""

        self.model_builder.write_aloha_routines()
        print "Please try compiling the file /tmp/hel_amps_sm.cc:"
        print "cd /tmp; g++ -c hel_amps_sm.cc"

    def disabled_test_couplings_and_parameters(self):
        """Test generation of couplings and parameters"""

        self.assertTrue(self.model_builder.params_indep)
        self.assertTrue(self.model_builder.params_dep)
        self.assertTrue(self.model_builder.coups_indep)
        self.assertTrue(self.model_builder.coups_dep)

        g_expr = re.compile("G(?!f)")

        for indep_par in self.model_builder.params_indep:
            self.assertFalse(g_expr.search(indep_par.expr))
        for indep_coup in self.model_builder.coups_indep: 
            self.assertFalse(g_expr.search(indep_coup.expr))

    def disabled_test_write_parameter_files(self):
        """Test writing the model parameter .h and.cc files"""

        self.model_builder.write_parameter_class_files()        
        
        print "Please try compiling the file /tmp/Parameters_sm.cc:"
        print "cd /tmp; g++ -c -I $PATH_TO_PYTHIA8/include Parameters_sm.cc"


#===============================================================================
# ExportUFOModelCPPTest
#===============================================================================
class ExportUFOModelCPPTest(unittest.TestCase,
                                test_file_writers.CheckFileCreate):

    created_files = [
                    ]

    def setUp(self):

        model_pkl = os.path.join(MG5DIR, 'models','sm','model.pkl')
        if os.path.isfile(model_pkl):
            self.model = save_load_object.load_from_file(model_pkl)
        else:
            self.model = import_ufo.import_model('sm')
        self.model_builder = export_cpp.UFOModelConverterCPP(\
                                                            self.model, "/tmp")
        
        test_file_writers.CheckFileCreate.clean_files

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def disabled_test_read_aloha_template_files(self):
        """Test reading the ALOHA template .h and .cc files"""

        template_h = self.model_builder.read_aloha_template_files("h")
        self.assertTrue(template_h)
        for file_lines in template_h:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1)
        template_cc = self.model_builder.read_aloha_template_files("cc")
        self.assertTrue(template_cc)
        for file_lines in template_cc:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1) 
       
    def disabled_test_write_aloha_functions(self):
        """Test writing function declarations and definitions"""

        template_h_files = []
        template_cc_files = []

        aloha_model = create_aloha.AbstractALOHAModel(\
                                         self.model_builder.model.get('name'))
        aloha_model.compute_all(save=False)
        for abstracthelas in dict(aloha_model).values():
            abstracthelas.write('/tmp', 'CPP')
            #abstracthelas.write('/tmp', 'Fortran')

        print "Please try compiling the files /tmp/*.cc and /tmp/*.f:"
        print "cd /tmp; g++ -c *.cc; gfortran -c *.f"
        

    def disabled_test_write_aloha_routines(self):
        """Test writing the aloha .h and.cc files"""

        self.model_builder.write_aloha_routines()
        print "Please try compiling the file /tmp/hel_amps_sm.cc:"
        print "cd /tmp; g++ -c hel_amps_sm.cc"

    def disabled_test_couplings_and_parameters(self):
        """Test generation of couplings and parameters"""

        self.assertTrue(self.model_builder.params_indep)
        self.assertTrue(self.model_builder.params_dep)
        self.assertTrue(self.model_builder.coups_indep)
        self.assertTrue(self.model_builder.coups_dep)

        g_expr = re.compile("G(?!f)")

        for indep_par in self.model_builder.params_indep:
            self.assertFalse(g_expr.search(indep_par.expr))
        for indep_coup in self.model_builder.coups_indep: 
            self.assertFalse(g_expr.search(indep_coup.expr))

    def disabled_test_write_parameter_files(self):
        """Test writing the model parameter .h and.cc files"""

        self.model_builder.write_parameter_class_files()        
        
        print "Please try compiling the file /tmp/Parameters_sm.cc:"
        print "cd /tmp; g++ -c Parameters_sm.cc"

