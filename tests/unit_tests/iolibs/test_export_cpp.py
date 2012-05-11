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
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.misc as misc

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
// MadGraph 5 v. %(version)s, %(date)s
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
// Process: u u~ > u u~ QSIX=0
// Process: c c~ > c c~ QSIX=0
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
    void calculate_wavefunctions(const int perm[], const int hel[]); 
    static const int nwavefuncs = 8; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 4; 
    std::complex<double> amp[namplitudes]; 
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
// MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "Sigma_sm_qqx_qqx.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > u u~ QSIX=0
// Process: c c~ > c c~ QSIX=0

//--------------------------------------------------------------------------
// Initialize process.

void Sigma_sm_qqx_qqx::initProc() 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  pars->setIndependentParameters(particleDataPtr, couplingsPtr, slhaPtr); 
  pars->setIndependentCouplings(); 
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
  pars->setDependentParameters(particleDataPtr, couplingsPtr, slhaPtr, alpS); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 2; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 16; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {{-1, -1, -1, -1}, {-1, -1,
      -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1},
      {-1, 1, 1, -1}, {-1, 1, 1, 1}, {1, -1, -1, -1}, {1, -1, -1, 1}, {1, -1,
      1, -1}, {1, -1, 1, 1}, {1, 1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1,
      1, 1, 1}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {36}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
    t[i] = 0.; 
  }

  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
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
        calculate_wavefunctions(perm, helicities[ihel]); 
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
      calculate_wavefunctions(perm, helicities[ihel]); 
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
    int flavors[1][2] = {{4, -4}}; 
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
    int flavors[1][2] = {{2, -2}}; 
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
    static int colors[2][8] = {{1, 0, 0, 1, 2, 0, 0, 2}, {2, 0, 0, 1, 2, 0, 0,
        1}};
    setColAcol(colors[ic][0], colors[ic][1], colors[ic][2], colors[ic][3],
        colors[ic][4], colors[ic][5], colors[ic][6], colors[ic][7]);
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

void Sigma_sm_qqx_qqx::calculate_wavefunctions(const int perm[], const int
    hel[])
{
  // Calculate wavefunctions for all processes
  double p[nexternal][4]; 
  int i; 

  // Convert Pythia 4-vectors to double[]
  for(i = 0; i < nexternal; i++ )
  {
    p[i][0] = pME[i].e(); 
    p[i][1] = pME[i].px(); 
    p[i][2] = pME[i].py(); 
    p[i][3] = pME[i].pz(); 
  }

  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  FFV1_3(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[4]); 
  FFV2_5_3(w[0], w[1], pars->GC_35, pars->GC_47, pars->MZ, pars->WZ, w[5]); 
  FFV1_3(w[0], w[2], pars->GC_10, pars->ZERO, pars->ZERO, w[6]); 
  FFV2_5_3(w[0], w[2], pars->GC_35, pars->GC_47, pars->MZ, pars->WZ, w[7]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[3], w[2], w[4], pars->GC_10, amp[0]); 
  FFV2_5_0(w[3], w[2], w[5], pars->GC_35, pars->GC_47, amp[1]); 
  FFV1_0(w[3], w[1], w[6], pars->GC_10, amp[2]); 
  FFV2_5_0(w[3], w[1], w[7], pars->GC_35, pars->GC_47, amp[3]); 


}
double Sigma_sm_qqx_qqx::matrix_uux_uux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./6. * amp[0] - amp[1] + 1./2. * amp[2]; 
  jamp[1] = -1./2. * amp[0] - 1./6. * amp[2] + amp[3]; 

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

        #print open(self.give_pos('test.cc')).read()
        self.assertFileContains('test.cc', goal_string)

    def test_write_process_cc_file_uu_six(self):
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

        exporter = export_cpp.ProcessExporterPythia8(\
            mymatrixelement, self.mycppwriter,
            process_string = "q q > six")

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "Sigma_sm_qq_six.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > six

//--------------------------------------------------------------------------
// Initialize process.

void Sigma_sm_qq_six::initProc() 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  pars->setIndependentParameters(particleDataPtr, couplingsPtr, slhaPtr); 
  pars->setIndependentCouplings(); 
  // Set massive/massless matrix elements for c/b/mu/tau
  mcME = particleDataPtr->m0(4); 
  mbME = 0.; 
  mmuME = 0.; 
  mtauME = 0.; 
  jamp2[0] = new double[1]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void Sigma_sm_qq_six::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(particleDataPtr, couplingsPtr, slhaPtr, alpS); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 1; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 4; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {{-1, -1, 0}, {-1, 1, 0}, {1,
      -1, 0}, {1, 1, 0}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {36}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
    t[i] = 0.; 
  }

  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
  }

  // For now, call setupForME() here
  id1 = 2; 
  id2 = 2; 
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
        calculate_wavefunctions(perm, helicities[ihel]); 
        t[0] = matrix_uu_six(); 

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
      calculate_wavefunctions(perm, helicities[ihel]); 
      t[0] = matrix_uu_six(); 

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

double Sigma_sm_qq_six::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 2 && id2 == 2)
  {
    // Add matrix elements for processes with beams (2, 2)
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

void Sigma_sm_qq_six::setIdColAcol() 
{
  if(id1 == 2 && id2 == 2)
  {
    // Pick one of the flavor combinations (6000001,)
    int flavors[1][1] = {{6000001}}; 
    vector<double> probs; 
    double sum = matrix_element[0]; 
    probs.push_back(matrix_element[0]/sum); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
  }
  setId(id1, id2, id3); 
  // Pick color flow
  int ncolor[1] = {1}; 
  if(id1 == 2 && id2 == 2 && id3 == 6000001)
  {
    vector<double> probs; 
    double sum = jamp2[0][0]; 
    for(int i = 0; i < ncolor[0]; i++ )
      probs.push_back(jamp2[0][i]/sum); 
    int ic = rndmPtr->pick(probs); 
    static int colors[1][6] = {{1, 0, 2, 0, 1, -2}}; 
    setColAcol(colors[ic][0], colors[ic][1], colors[ic][2], colors[ic][3],
        colors[ic][4], colors[ic][5]);
  }
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process

double Sigma_sm_qq_six::weightDecay(Event& process, int iResBeg, int iResEnd) 
{
  // Just use isotropic decay (default)
  return 1.; 
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void Sigma_sm_qq_six::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  double p[nexternal][4]; 
  int i; 

  // Convert Pythia 4-vectors to double[]
  for(i = 0; i < nexternal; i++ )
  {
    p[i][0] = pME[i].e(); 
    p[i][1] = pME[i].px(); 
    p[i][2] = pME[i].py(); 
    p[i][3] = pME[i].pz(); 
  }

  // Calculate all wavefunctions
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  sxxxxx(p[perm[2]], +1, w[2]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFS1C1_0(w[1], w[0], w[2], pars->GC_24, amp[0]); 


}
double Sigma_sm_qq_six::matrix_uu_six() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 1; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{6}}; 

  // Calculate color flows
  jamp[0] = -amp[0]; 

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

        exporter.write_process_cc_file(\
                 writers.CPPWriter(self.give_pos('test.cc')))

        #print open(self.give_pos('test.cc')).read()
        self.assertFileContains('test.cc', goal_string)

    def test_write_cpp_go_process_cc_file(self):
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

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "CPPProcess.h"
#include "HelAmps_sm.h"

using namespace MG5_sm; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > go go

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
  jamp2[0] = new double[2]; 
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
  static const int helicities[ncomb][nexternal] = {{-1, -1, -1, -1}, {-1, -1,
      -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1},
      {-1, 1, 1, -1}, {-1, 1, 1, 1}, {1, -1, -1, -1}, {1, -1, -1, 1}, {1, -1,
      1, -1}, {1, -1, 1, 1}, {1, 1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1,
      1, 1, 1}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {72, 72}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
  }
  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
  }

  if (sum_hel == 0 || ntry < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        calculate_wavefunctions(perm, helicities[ihel]); 
        t[0] = matrix_uux_gogo(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_uux_gogo(); 
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
      calculate_wavefunctions(perm, helicities[ihel]); 
      t[0] = matrix_uux_gogo(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_uux_gogo(); 
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
  if(id1 == 2 && id2 == -2)
  {
    // Add matrix elements for processes with beams (2, -2)
    return matrix_element[0]; 
  }
  else if(id1 == -2 && id2 == 2)
  {
    // Add matrix elements for processes with beams (-2, 2)
    return matrix_element[1]; 
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

void CPPProcess::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  int i, j; 

  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  FFV1_3(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[4]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[2], w[3], w[4], pars->GC_8, amp[0]); 

}
double CPPProcess::matrix_uux_gogo() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 1; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {3, 3}; 
  static const double cf[ncolor][ncolor] = {{16, -2}, {-2, 16}}; 

  // Calculate color flows
  jamp[0] = -std::complex<double> (0, 1) * amp[0]; 
  jamp[1] = +std::complex<double> (0, 1) * amp[0]; 

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

        #print open(self.give_pos('test.cc')).read()
        self.assertFileContains('test.cc', goal_string)

    def test_write_cpp_four_fermion_vertex(self):
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

        matrix_element = helas_objects.HelasMultiProcess(myamplitude)
        matrix_element.get('matrix_elements')[0].set('has_mirror_process',
                                                     True)

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#ifndef Pythia8_Sigma__uu_ttg_H
#define Pythia8_Sigma__uu_ttg_H

#include <complex> 

#include "SigmaProcess.h"
#include "Parameters_.h"

using namespace std; 

namespace Pythia8 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u > t t g
//--------------------------------------------------------------------------

class Sigma__uu_ttg : public Sigma3Process 
{
  public:

    // Constructor.
    Sigma__uu_ttg() {}

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
    virtual string name() const {return "u u > t t g ()";}

    virtual int code() const {return 10000;}

    virtual string inFlux() const {return "qq";}
    int id3Mass() const {return 6;}
    int id4Mass() const {return 6;}
    virtual int resonanceA() const {return 6;}
    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertM2() const {return true;}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int perm[], const int hel[]); 
    static const int nwavefuncs = 9; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 4; 
    std::complex<double> amp[namplitudes]; 
    double matrix_uu_ttg(); 

    // Constants for array limits
    static const int nexternal = 5; 
    static const int nprocesses = 2; 

    // Store the matrix element value from sigmaKin
    double matrix_element[nprocesses]; 

    // Color flows, used when selecting color
    double * jamp2[nprocesses]; 

    // Pointer to the model parameters
    Parameters_ * pars; 

}; 

}  // end namespace Pythia

#endif  // Pythia8_Sigma__uu_ttg_H
""" % misc.get_pkg_info()

        exporter = export_cpp.ProcessExporterPythia8(matrix_element,
                                                     self.mycppwriter)

        exporter.write_process_h_file(\
                  writers.CPPWriter(self.give_pos('test.h')))

        #print open(self.give_pos('test.h')).read()
        self.assertFileContains('test.h', goal_string)

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph 5 v. 1.4.6, 2012-04-XX
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "Sigma__uu_ttg.h"
#include "HelAmps_.h"

using namespace Pythia8_; 

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > t t g

//--------------------------------------------------------------------------
// Initialize process.

void Sigma__uu_ttg::initProc() 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_::getInstance(); 
  pars->setIndependentParameters(particleDataPtr, couplingsPtr, slhaPtr); 
  pars->setIndependentCouplings(); 
  // Set massive/massless matrix elements for c/b/mu/tau
  mcME = 0.; 
  mbME = 0.; 
  mmuME = 0.; 
  mtauME = 0.; 
  jamp2[0] = new double[3]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void Sigma__uu_ttg::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(particleDataPtr, couplingsPtr, slhaPtr, alpS); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 3; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 32; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {{-1, -1, -1, -1, -1}, {-1,
      -1, -1, -1, 1}, {-1, -1, -1, 1, -1}, {-1, -1, -1, 1, 1}, {-1, -1, 1, -1,
      -1}, {-1, -1, 1, -1, 1}, {-1, -1, 1, 1, -1}, {-1, -1, 1, 1, 1}, {-1, 1,
      -1, -1, -1}, {-1, 1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, 1, -1, 1, 1},
      {-1, 1, 1, -1, -1}, {-1, 1, 1, -1, 1}, {-1, 1, 1, 1, -1}, {-1, 1, 1, 1,
      1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1,
      -1, 1, 1}, {1, -1, 1, -1, -1}, {1, -1, 1, -1, 1}, {1, -1, 1, 1, -1}, {1,
      -1, 1, 1, 1}, {1, 1, -1, -1, -1}, {1, 1, -1, -1, 1}, {1, 1, -1, 1, -1},
      {1, 1, -1, 1, 1}, {1, 1, 1, -1, -1}, {1, 1, 1, -1, 1}, {1, 1, 1, 1, -1},
      {1, 1, 1, 1, 1}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {72, 72}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
    t[i] = 0.; 
  }

  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
  }

  // For now, call setupForME() here
  id1 = 2; 
  id2 = 2; 
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
        calculate_wavefunctions(perm, helicities[ihel]); 
        t[0] = matrix_uu_ttg(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_uu_ttg(); 
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
      calculate_wavefunctions(perm, helicities[ihel]); 
      t[0] = matrix_uu_ttg(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_uu_ttg(); 
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

double Sigma__uu_ttg::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 2 && id2 == 2)
  {
    // Add matrix elements for processes with beams (2, 2)
    return matrix_element[0] + matrix_element[1]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void Sigma__uu_ttg::setIdColAcol() 
{
  if(id1 == 2 && id2 == 2)
  {
    // Pick one of the flavor combinations (6, 6, 21)
    int flavors[1][3] = {{6, 6, 21}}; 
    vector<double> probs; 
    double sum = matrix_element[0]; 
    probs.push_back(matrix_element[0]/sum); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
    id4 = flavors[choice][1]; 
    id5 = flavors[choice][2]; 
  }
  setId(id1, id2, id3, id4, id5); 
  // Pick color flow
  int ncolor[1] = {3}; 
  if(id1 == 2 && id2 == 2 && id3 == 6 && id4 == 6 && id5 == 21)
  {
    vector<double> probs; 
    double sum = jamp2[0][0] + jamp2[0][1] + jamp2[0][2]; 
    for(int i = 0; i < ncolor[0]; i++ )
      probs.push_back(jamp2[0][i]/sum); 
    int ic = rndmPtr->pick(probs); 
    static int colors[3][10] = {{3, 0, 1, 0, 1, 0, 2, 0, 3, 2}, {2, 0, 3, 0, 1,
        0, 2, 0, 3, 1}, {3, 0, 2, 0, 1, 0, 2, 0, 3, 1}};
    setColAcol(colors[ic][0], colors[ic][1], colors[ic][2], colors[ic][3],
        colors[ic][4], colors[ic][5], colors[ic][6], colors[ic][7],
        colors[ic][8], colors[ic][9]);
  }
  else if(id1 == 2 && id2 == 2 && id3 == 6 && id4 == 6 && id5 == 21)
  {
    vector<double> probs; 
    double sum = jamp2[0][0] + jamp2[0][1] + jamp2[0][2]; 
    for(int i = 0; i < ncolor[0]; i++ )
      probs.push_back(jamp2[0][i]/sum); 
    int ic = rndmPtr->pick(probs); 
    static int colors[3][10] = {{1, 0, 3, 0, 1, 0, 2, 0, 3, 2}, {3, 0, 2, 0, 1,
        0, 2, 0, 3, 1}, {2, 0, 3, 0, 1, 0, 2, 0, 3, 1}};
    setColAcol(colors[ic][0], colors[ic][1], colors[ic][2], colors[ic][3],
        colors[ic][4], colors[ic][5], colors[ic][6], colors[ic][7],
        colors[ic][8], colors[ic][9]);
  }
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process

double Sigma__uu_ttg::weightDecay(Event& process, int iResBeg, int iResEnd) 
{
  // Just use isotropic decay (default)
  return 1.; 
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void Sigma__uu_ttg::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  double p[nexternal][4]; 
  int i; 

  // Convert Pythia 4-vectors to double[]
  for(i = 0; i < nexternal; i++ )
  {
    p[i][0] = pME[i].e(); 
    p[i][1] = pME[i].px(); 
    p[i][2] = pME[i].py(); 
    p[i][3] = pME[i].pz(); 
  }

  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  vxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  FFV1_2(w[0], w[4], pars->GG, pars->zero, pars->zero, w[5]); 
  FFFF1_2(w[0], w[1], w[2], pars->GEFF, pars->MT, pars->WT, w[6]); 
  FFFF1_2(w[0], w[1], w[3], pars->GEFF, pars->MT, pars->WT, w[7]); 
  FFFF1_1(w[2], w[0], w[3], pars->GEFF, pars->zero, pars->zero, w[8]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFFF1_0(w[1], w[2], w[5], w[3], pars->GEFF, amp[0]); 
  FFV1_0(w[6], w[3], w[4], pars->GG, amp[1]); 
  FFV1_0(w[7], w[2], w[4], pars->GG, amp[2]); 
  FFV1_0(w[1], w[8], w[4], pars->GG, amp[3]); 


}
double Sigma__uu_ttg::matrix_uu_ttg() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 3; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 0, 4}, {0, 12, 4}, {4, 4,
      12}};

  // Calculate color flows
  jamp[0] = +amp[0] + amp[1]; 
  jamp[1] = +amp[3]; 
  jamp[2] = -amp[2]; 

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
"""

        exporter.write_process_cc_file(\
                  writers.CPPWriter(self.give_pos('test.cc')))

        #print open(self.give_pos('test.cc')).read()
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

    def setUp(self):

        model_pkl = os.path.join(MG5DIR, 'models','sm','model.pkl')
        if os.path.isfile(model_pkl):
            self.model = save_load_object.load_from_file(model_pkl)
        else:
            sm_path = import_ufo.find_ufo_path('sm')
            self.model = import_ufo.import_model(sm_path)
        self.model_builder = export_cpp.UFOModelConverterPythia8(\
                                             self.model, "/tmp")
        
        test_file_writers.CheckFileCreate.clean_files

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_write_pythia8_parameter_files(self):
        """Test writing the Pythia model parameter files"""

        goal_file_h = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8
#  MadGraph 5 v. %(version)s, %(date)s
#  By the MadGraph Development Team
#  Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#ifndef Pythia8_parameters_sm_H
#define Pythia8_parameters_sm_H

#include <complex>

#include "ParticleData.h"
#include "StandardModel.h"
#include "SusyLesHouches.h"

using namespace std;

namespace Pythia8 {

class Parameters_sm
{
public:

static Parameters_sm* getInstance();

// Model parameters independent of aS
double WTau,WH,WW,WZ,WT,MTA,MM,Me,MH,MZ,MB,MT,MC,ymtau,ymm,yme,ymt,ymb,ymc,etaWS,rhoWS,AWS,lamWS,Gf,aEWM1,ZERO,lamWS__exp__2,lamWS__exp__3,MZ__exp__2,MZ__exp__4,sqrt__2,MH__exp__2,aEW,MW,sqrt__aEW,ee,MW__exp__2,sw2,cw,sqrt__sw2,sw,g1,gw,v,v__exp__2,lam,yb,yc,ye,ym,yt,ytau,muH,gw__exp__2,cw__exp__2,ee__exp__2,sw__exp__2;
std::complex<double> CKM11,CKM12,complexi,CKM13,CKM21,CKM22,CKM23,CKM31,CKM32,CKM33,conjg__CKM11,conjg__CKM12,conjg__CKM13,conjg__CKM21,conjg__CKM22,conjg__CKM23,conjg__CKM31,conjg__CKM32,conjg__CKM33;
// Model parameters dependent on aS
double aS,sqrt__aS,G,G__exp__2;
// Model couplings independent of aS
std::complex<double> GC_1,GC_2,GC_3,GC_7,GC_8,GC_9,GC_10,GC_11,GC_12,GC_13,GC_14,GC_15,GC_16,GC_17,GC_18,GC_19,GC_20,GC_21,GC_22,GC_23,GC_24,GC_25,GC_26,GC_27,GC_28,GC_29,GC_30,GC_31,GC_32,GC_33,GC_34,GC_35,GC_36,GC_37,GC_38,GC_39,GC_40,GC_41,GC_42,GC_43,GC_44,GC_45,GC_46,GC_47;
// Model couplings dependent on aS
std::complex<double> GC_6,GC_5,GC_4;

// Set parameters that are unchanged during the run
void setIndependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr);
// Set couplings that are unchanged during the run
void setIndependentCouplings();
// Set parameters that are changed event by event
void setDependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr, double alpS);
// Set couplings that are changed event by event
void setDependentCouplings();

// Print parameters that are unchanged during the run
void printIndependentParameters();
// Print couplings that are unchanged during the run
void printIndependentCouplings();
// Print parameters that are changed event by event
void printDependentParameters();
// Print couplings that are changed event by event
void printDependentCouplings();


  private:
static Parameters_sm* instance;
};

} // end namespace Pythia8
#endif // Pythia8_parameters_sm_H
""" % misc.get_pkg_info()

        goal_file_cc = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
#  MadGraph 5 v. %(version)s, %(date)s
#  By the MadGraph Development Team
#  Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include <iostream>
#include "Parameters_sm.h"
#include "PythiaStdlib.h"

namespace Pythia8 {

    // Initialize static instance
    Parameters_sm* Parameters_sm::instance = 0;

    // Function to get static instance - only one instance per program
    Parameters_sm* Parameters_sm::getInstance(){
	if (instance == 0)
	    instance = new Parameters_sm();

	return instance;
    }

    void Parameters_sm::setIndependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr){
	WTau=pd->mWidth(15);
WH=pd->mWidth(25);
WW=pd->mWidth(24);
WZ=pd->mWidth(23);
WT=pd->mWidth(6);
MTA=pd->m0(15);
MM=pd->m0(13);
Me=pd->m0(11);
MH=pd->m0(25);
MZ=pd->m0(23);
MB=pd->m0(5);
MT=pd->m0(6);
MC=pd->m0(4);
ymtau=pd->mRun(15, pd->m0(24));
ymm=pd->mRun(13, pd->m0(24));
yme=pd->mRun(11, pd->m0(24));
ymt=pd->mRun(6, pd->m0(24));
ymb=pd->mRun(5, pd->m0(24));
ymc=pd->mRun(4, pd->m0(24));
if(!slhaPtr->getEntry<double>("wolfenstein", 4, etaWS)){
cout << "Warning, setting etaWS to 3.410000e-01" << endl;
etaWS = 3.410000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 3, rhoWS)){
cout << "Warning, setting rhoWS to 1.320000e-01" << endl;
rhoWS = 1.320000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 2, AWS)){
cout << "Warning, setting AWS to 8.080000e-01" << endl;
AWS = 8.080000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 1, lamWS)){
cout << "Warning, setting lamWS to 2.253000e-01" << endl;
lamWS = 2.253000e-01;}
Gf = M_PI*csm->alphaEM(pow(pd->m0(23),2))*pow(pd->m0(23),2)/(sqrt(2.)*pow(pd->m0(24),2)*(pow(pd->m0(23),2)-pow(pd->m0(24),2)));
aEWM1 = 1./csm->alphaEM(pow(pd->m0(23),2));
ZERO = 0.;
lamWS__exp__2 = pow(lamWS,2.);
CKM11 = 1.-lamWS__exp__2/2.;
CKM12 = lamWS;
complexi = std::complex<double>(0.,1.);
lamWS__exp__3 = pow(lamWS,3.);
CKM13 = AWS*lamWS__exp__3*(-(etaWS*complexi)+rhoWS);
CKM21 = -lamWS;
CKM22 = 1.-lamWS__exp__2/2.;
CKM23 = AWS*lamWS__exp__2;
CKM31 = AWS*lamWS__exp__3*(1.-etaWS*complexi-rhoWS);
CKM32 = -(AWS*lamWS__exp__2);
CKM33 = 1.;
MZ__exp__2 = pow(MZ,2.);
MZ__exp__4 = pow(MZ,4.);
sqrt__2 = sqrt(2.);
MH__exp__2 = pow(MH,2.);
conjg__CKM11 = conj(CKM11);
conjg__CKM12 = conj(CKM12);
conjg__CKM13 = conj(CKM13);
conjg__CKM21 = conj(CKM21);
conjg__CKM22 = conj(CKM22);
conjg__CKM23 = conj(CKM23);
conjg__CKM31 = conj(CKM31);
conjg__CKM32 = conj(CKM32);
conjg__CKM33 = conj(CKM33);
aEW = 1./aEWM1;
MW = sqrt(MZ__exp__2/2.+sqrt(MZ__exp__4/4.-(aEW*M_PI*MZ__exp__2)/(Gf*sqrt__2)));
sqrt__aEW = sqrt(aEW);
ee = 2.*sqrt__aEW*sqrt(M_PI);
MW__exp__2 = pow(MW,2.);
sw2 = 1.-MW__exp__2/MZ__exp__2;
cw = sqrt(1.-sw2);
sqrt__sw2 = sqrt(sw2);
sw = sqrt__sw2;
g1 = ee/cw;
gw = ee/sw;
v = (2.*MW*sw)/ee;
v__exp__2 = pow(v,2.);
lam = MH__exp__2/(2.*v__exp__2);
yb = (ymb*sqrt__2)/v;
yc = (ymc*sqrt__2)/v;
ye = (yme*sqrt__2)/v;
ym = (ymm*sqrt__2)/v;
yt = (ymt*sqrt__2)/v;
ytau = (ymtau*sqrt__2)/v;
muH = sqrt(lam*v__exp__2);
gw__exp__2 = pow(gw,2.);
cw__exp__2 = pow(cw,2.);
ee__exp__2 = pow(ee,2.);
sw__exp__2 = pow(sw,2.);
    }
    void Parameters_sm::setIndependentCouplings(){
	GC_1 = -(ee*complexi)/3.;
GC_2 = (2.*ee*complexi)/3.;
GC_3 = -(ee*complexi);
GC_7 = cw*complexi*gw;
GC_8 = -(complexi*gw__exp__2);
GC_9 = cw__exp__2*complexi*gw__exp__2;
GC_10 = (ee__exp__2*complexi)/(2.*sw__exp__2);
GC_11 = (ee*complexi)/(sw*sqrt__2);
GC_12 = (CKM11*ee*complexi)/(sw*sqrt__2);
GC_13 = (CKM12*ee*complexi)/(sw*sqrt__2);
GC_14 = (CKM13*ee*complexi)/(sw*sqrt__2);
GC_15 = (CKM21*ee*complexi)/(sw*sqrt__2);
GC_16 = (CKM22*ee*complexi)/(sw*sqrt__2);
GC_17 = (CKM23*ee*complexi)/(sw*sqrt__2);
GC_18 = (CKM31*ee*complexi)/(sw*sqrt__2);
GC_19 = (CKM32*ee*complexi)/(sw*sqrt__2);
GC_20 = (CKM33*ee*complexi)/(sw*sqrt__2);
GC_21 = -(cw*ee*complexi)/(2.*sw);
GC_22 = (cw*ee*complexi)/(2.*sw);
GC_23 = -(ee*complexi*sw)/(6.*cw);
GC_24 = (ee*complexi*sw)/(2.*cw);
GC_25 = complexi*gw*sw;
GC_26 = -2.*cw*complexi*gw__exp__2*sw;
GC_27 = complexi*gw__exp__2*sw__exp__2;
GC_28 = (cw*ee*complexi)/(2.*sw)+(ee*complexi*sw)/(2.*cw);
GC_29 = ee__exp__2*complexi+(cw__exp__2*ee__exp__2*complexi)/(2.*sw__exp__2)+(ee__exp__2*complexi*sw__exp__2)/(2.*cw__exp__2);
GC_30 = -6.*complexi*lam*v;
GC_31 = (ee__exp__2*complexi*v)/(2.*sw__exp__2);
GC_32 = ee__exp__2*complexi*v+(cw__exp__2*ee__exp__2*complexi*v)/(2.*sw__exp__2)+(ee__exp__2*complexi*sw__exp__2*v)/(2.*cw__exp__2);
GC_33 = -((complexi*yb)/sqrt__2);
GC_34 = -((complexi*yc)/sqrt__2);
GC_35 = -((complexi*ye)/sqrt__2);
GC_36 = -((complexi*ym)/sqrt__2);
GC_37 = -((complexi*yt)/sqrt__2);
GC_38 = -((complexi*ytau)/sqrt__2);
GC_39 = (ee*complexi*conjg__CKM11)/(sw*sqrt__2);
GC_40 = (ee*complexi*conjg__CKM12)/(sw*sqrt__2);
GC_41 = (ee*complexi*conjg__CKM13)/(sw*sqrt__2);
GC_42 = (ee*complexi*conjg__CKM21)/(sw*sqrt__2);
GC_43 = (ee*complexi*conjg__CKM22)/(sw*sqrt__2);
GC_44 = (ee*complexi*conjg__CKM23)/(sw*sqrt__2);
GC_45 = (ee*complexi*conjg__CKM31)/(sw*sqrt__2);
GC_46 = (ee*complexi*conjg__CKM32)/(sw*sqrt__2);
GC_47 = (ee*complexi*conjg__CKM33)/(sw*sqrt__2);
    }
    void Parameters_sm::setDependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr, double alpS){
	aS = alpS;
sqrt__aS = sqrt(aS);
G = 2.*sqrt__aS*sqrt(M_PI);
G__exp__2 = pow(G,2.);
    }
    void Parameters_sm::setDependentCouplings(){
	GC_6 = complexi*G__exp__2;
GC_5 = complexi*G;
GC_4 = -G;
    }

    // Routines for printing out parameters
    void Parameters_sm::printIndependentParameters(){
	cout << "sm model parameters independent of event kinematics:" << endl;
	cout << setw(20) << "WTau " << "= " << setiosflags(ios::scientific) << setw(10) << WTau << endl;
cout << setw(20) << "WH " << "= " << setiosflags(ios::scientific) << setw(10) << WH << endl;
cout << setw(20) << "WW " << "= " << setiosflags(ios::scientific) << setw(10) << WW << endl;
cout << setw(20) << "WZ " << "= " << setiosflags(ios::scientific) << setw(10) << WZ << endl;
cout << setw(20) << "WT " << "= " << setiosflags(ios::scientific) << setw(10) << WT << endl;
cout << setw(20) << "MTA " << "= " << setiosflags(ios::scientific) << setw(10) << MTA << endl;
cout << setw(20) << "MM " << "= " << setiosflags(ios::scientific) << setw(10) << MM << endl;
cout << setw(20) << "Me " << "= " << setiosflags(ios::scientific) << setw(10) << Me << endl;
cout << setw(20) << "MH " << "= " << setiosflags(ios::scientific) << setw(10) << MH << endl;
cout << setw(20) << "MZ " << "= " << setiosflags(ios::scientific) << setw(10) << MZ << endl;
cout << setw(20) << "MB " << "= " << setiosflags(ios::scientific) << setw(10) << MB << endl;
cout << setw(20) << "MT " << "= " << setiosflags(ios::scientific) << setw(10) << MT << endl;
cout << setw(20) << "MC " << "= " << setiosflags(ios::scientific) << setw(10) << MC << endl;
cout << setw(20) << "ymtau " << "= " << setiosflags(ios::scientific) << setw(10) << ymtau << endl;
cout << setw(20) << "ymm " << "= " << setiosflags(ios::scientific) << setw(10) << ymm << endl;
cout << setw(20) << "yme " << "= " << setiosflags(ios::scientific) << setw(10) << yme << endl;
cout << setw(20) << "ymt " << "= " << setiosflags(ios::scientific) << setw(10) << ymt << endl;
cout << setw(20) << "ymb " << "= " << setiosflags(ios::scientific) << setw(10) << ymb << endl;
cout << setw(20) << "ymc " << "= " << setiosflags(ios::scientific) << setw(10) << ymc << endl;
cout << setw(20) << "etaWS " << "= " << setiosflags(ios::scientific) << setw(10) << etaWS << endl;
cout << setw(20) << "rhoWS " << "= " << setiosflags(ios::scientific) << setw(10) << rhoWS << endl;
cout << setw(20) << "AWS " << "= " << setiosflags(ios::scientific) << setw(10) << AWS << endl;
cout << setw(20) << "lamWS " << "= " << setiosflags(ios::scientific) << setw(10) << lamWS << endl;
cout << setw(20) << "Gf " << "= " << setiosflags(ios::scientific) << setw(10) << Gf << endl;
cout << setw(20) << "aEWM1 " << "= " << setiosflags(ios::scientific) << setw(10) << aEWM1 << endl;
cout << setw(20) << "ZERO " << "= " << setiosflags(ios::scientific) << setw(10) << ZERO << endl;
cout << setw(20) << "lamWS__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << lamWS__exp__2 << endl;
cout << setw(20) << "CKM11 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM11 << endl;
cout << setw(20) << "CKM12 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM12 << endl;
cout << setw(20) << "complexi " << "= " << setiosflags(ios::scientific) << setw(10) << complexi << endl;
cout << setw(20) << "lamWS__exp__3 " << "= " << setiosflags(ios::scientific) << setw(10) << lamWS__exp__3 << endl;
cout << setw(20) << "CKM13 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM13 << endl;
cout << setw(20) << "CKM21 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM21 << endl;
cout << setw(20) << "CKM22 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM22 << endl;
cout << setw(20) << "CKM23 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM23 << endl;
cout << setw(20) << "CKM31 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM31 << endl;
cout << setw(20) << "CKM32 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM32 << endl;
cout << setw(20) << "CKM33 " << "= " << setiosflags(ios::scientific) << setw(10) << CKM33 << endl;
cout << setw(20) << "MZ__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << MZ__exp__2 << endl;
cout << setw(20) << "MZ__exp__4 " << "= " << setiosflags(ios::scientific) << setw(10) << MZ__exp__4 << endl;
cout << setw(20) << "sqrt__2 " << "= " << setiosflags(ios::scientific) << setw(10) << sqrt__2 << endl;
cout << setw(20) << "MH__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << MH__exp__2 << endl;
cout << setw(20) << "conjg__CKM11 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM11 << endl;
cout << setw(20) << "conjg__CKM12 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM12 << endl;
cout << setw(20) << "conjg__CKM13 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM13 << endl;
cout << setw(20) << "conjg__CKM21 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM21 << endl;
cout << setw(20) << "conjg__CKM22 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM22 << endl;
cout << setw(20) << "conjg__CKM23 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM23 << endl;
cout << setw(20) << "conjg__CKM31 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM31 << endl;
cout << setw(20) << "conjg__CKM32 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM32 << endl;
cout << setw(20) << "conjg__CKM33 " << "= " << setiosflags(ios::scientific) << setw(10) << conjg__CKM33 << endl;
cout << setw(20) << "aEW " << "= " << setiosflags(ios::scientific) << setw(10) << aEW << endl;
cout << setw(20) << "MW " << "= " << setiosflags(ios::scientific) << setw(10) << MW << endl;
cout << setw(20) << "sqrt__aEW " << "= " << setiosflags(ios::scientific) << setw(10) << sqrt__aEW << endl;
cout << setw(20) << "ee " << "= " << setiosflags(ios::scientific) << setw(10) << ee << endl;
cout << setw(20) << "MW__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << MW__exp__2 << endl;
cout << setw(20) << "sw2 " << "= " << setiosflags(ios::scientific) << setw(10) << sw2 << endl;
cout << setw(20) << "cw " << "= " << setiosflags(ios::scientific) << setw(10) << cw << endl;
cout << setw(20) << "sqrt__sw2 " << "= " << setiosflags(ios::scientific) << setw(10) << sqrt__sw2 << endl;
cout << setw(20) << "sw " << "= " << setiosflags(ios::scientific) << setw(10) << sw << endl;
cout << setw(20) << "g1 " << "= " << setiosflags(ios::scientific) << setw(10) << g1 << endl;
cout << setw(20) << "gw " << "= " << setiosflags(ios::scientific) << setw(10) << gw << endl;
cout << setw(20) << "v " << "= " << setiosflags(ios::scientific) << setw(10) << v << endl;
cout << setw(20) << "v__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << v__exp__2 << endl;
cout << setw(20) << "lam " << "= " << setiosflags(ios::scientific) << setw(10) << lam << endl;
cout << setw(20) << "yb " << "= " << setiosflags(ios::scientific) << setw(10) << yb << endl;
cout << setw(20) << "yc " << "= " << setiosflags(ios::scientific) << setw(10) << yc << endl;
cout << setw(20) << "ye " << "= " << setiosflags(ios::scientific) << setw(10) << ye << endl;
cout << setw(20) << "ym " << "= " << setiosflags(ios::scientific) << setw(10) << ym << endl;
cout << setw(20) << "yt " << "= " << setiosflags(ios::scientific) << setw(10) << yt << endl;
cout << setw(20) << "ytau " << "= " << setiosflags(ios::scientific) << setw(10) << ytau << endl;
cout << setw(20) << "muH " << "= " << setiosflags(ios::scientific) << setw(10) << muH << endl;
cout << setw(20) << "gw__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << gw__exp__2 << endl;
cout << setw(20) << "cw__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << cw__exp__2 << endl;
cout << setw(20) << "ee__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << ee__exp__2 << endl;
cout << setw(20) << "sw__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << sw__exp__2 << endl;
    }
    void Parameters_sm::printIndependentCouplings(){
	cout << "sm model couplings independent of event kinematics:" << endl;
	cout << setw(20) << "GC_1 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_1 << endl;
cout << setw(20) << "GC_2 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_2 << endl;
cout << setw(20) << "GC_3 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_3 << endl;
cout << setw(20) << "GC_7 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_7 << endl;
cout << setw(20) << "GC_8 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_8 << endl;
cout << setw(20) << "GC_9 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_9 << endl;
cout << setw(20) << "GC_10 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_10 << endl;
cout << setw(20) << "GC_11 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_11 << endl;
cout << setw(20) << "GC_12 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_12 << endl;
cout << setw(20) << "GC_13 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_13 << endl;
cout << setw(20) << "GC_14 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_14 << endl;
cout << setw(20) << "GC_15 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_15 << endl;
cout << setw(20) << "GC_16 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_16 << endl;
cout << setw(20) << "GC_17 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_17 << endl;
cout << setw(20) << "GC_18 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_18 << endl;
cout << setw(20) << "GC_19 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_19 << endl;
cout << setw(20) << "GC_20 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_20 << endl;
cout << setw(20) << "GC_21 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_21 << endl;
cout << setw(20) << "GC_22 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_22 << endl;
cout << setw(20) << "GC_23 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_23 << endl;
cout << setw(20) << "GC_24 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_24 << endl;
cout << setw(20) << "GC_25 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_25 << endl;
cout << setw(20) << "GC_26 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_26 << endl;
cout << setw(20) << "GC_27 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_27 << endl;
cout << setw(20) << "GC_28 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_28 << endl;
cout << setw(20) << "GC_29 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_29 << endl;
cout << setw(20) << "GC_30 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_30 << endl;
cout << setw(20) << "GC_31 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_31 << endl;
cout << setw(20) << "GC_32 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_32 << endl;
cout << setw(20) << "GC_33 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_33 << endl;
cout << setw(20) << "GC_34 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_34 << endl;
cout << setw(20) << "GC_35 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_35 << endl;
cout << setw(20) << "GC_36 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_36 << endl;
cout << setw(20) << "GC_37 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_37 << endl;
cout << setw(20) << "GC_38 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_38 << endl;
cout << setw(20) << "GC_39 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_39 << endl;
cout << setw(20) << "GC_40 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_40 << endl;
cout << setw(20) << "GC_41 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_41 << endl;
cout << setw(20) << "GC_42 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_42 << endl;
cout << setw(20) << "GC_43 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_43 << endl;
cout << setw(20) << "GC_44 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_44 << endl;
cout << setw(20) << "GC_45 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_45 << endl;
cout << setw(20) << "GC_46 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_46 << endl;
cout << setw(20) << "GC_47 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_47 << endl;
    }
    void Parameters_sm::printDependentParameters(){
	cout << "sm model parameters dependent on event kinematics:" << endl;
	cout << setw(20) << "aS " << "= " << setiosflags(ios::scientific) << setw(10) << aS << endl;
cout << setw(20) << "sqrt__aS " << "= " << setiosflags(ios::scientific) << setw(10) << sqrt__aS << endl;
cout << setw(20) << "G " << "= " << setiosflags(ios::scientific) << setw(10) << G << endl;
cout << setw(20) << "G__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << G__exp__2 << endl;
    }
    void Parameters_sm::printDependentCouplings(){
	cout << "sm model couplings dependent on event kinematics:" << endl;
	cout << setw(20) << "GC_6 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_6 << endl;
cout << setw(20) << "GC_5 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_5 << endl;
cout << setw(20) << "GC_4 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_4 << endl;
    }

} // end namespace Pythia8
""" % misc.get_pkg_info()

        file_h, file_cc = self.model_builder.generate_parameters_class_files()

        self.assertEqual(file_h, goal_file_h)
        self.assertEqual(file_cc, goal_file_cc)

