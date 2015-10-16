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

import StringIO
import copy
import fractions
import os
import re
import tests.IOTests as IOTests


import tests.unit_tests as unittest

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_cpp as export_cpp
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
// MadGraph5_aMC@NLO v. %(version)s, %(date)s
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef Pythia8_Sigma_sm_qqx_qqx_H
#define Pythia8_Sigma_sm_qqx_qqx_H

#include <complex> 

#include "Pythia8/SigmaProcess.h"
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

}  // end namespace Pythia8

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
// MadGraph5_aMC@NLO v. %(version)s, %(date)s
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
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
  if((id1 == 2 && id2 == -2 && id3 == 2 && id4 == -2) || (id1 == 4 && id2 == -4
      && id3 == 4 && id4 == -4))
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


}  // end namespace Pythia8
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
// MadGraph5_aMC@NLO v. %(version)s, %(date)s
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
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
  if((id1 == 2 && id2 == 2 && id3 == 6000001))
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


}  // end namespace Pythia8
""" % misc.get_pkg_info()

        exporter.write_process_cc_file(\
                 writers.CPPWriter(self.give_pos('test.cc')))

        #print open(self.give_pos('test.cc')).read()
        self.assertFileContains('test.cc', goal_string)

    @IOTests.createIOTest()
    def testIO_write_dec_multiprocess_files(self):
        """target: write_dec_multiprocess_files.h
           target: write_dec_multiprocess_files.cc
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

        # A quark S and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
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

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antis, \
                                             s, \
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

        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [antis, \
                                             s, \
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
            base_objects.ModelVariable('GDZ2', '0.05', 'complex'),
            base_objects.ModelVariable('ZZQQ', '0.01', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
        p = [21,1,2,3,-1,-2,-3]
        q = [1,2,-1,-2]
        procs = [[p,p,[23],p]]
        decays = [[[23],p,p]]
        my_processes = base_objects.ProcessDefinitionList()
        decayprocs = base_objects.ProcessDefinitionList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel})
            my_processes.append(my_process)

        for proc in decays:
            # Define the decays
            my_leglist = base_objects.MultiLegList([\
                base_objects.MultiLeg({'ids': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.ProcessDefinition({'legs':my_leglist,
                                                         'model':mymodel,
                                                         'is_decay_chain': True})
            decayprocs.append(my_process)

        for proc in my_processes:
            proc.set('decay_chains', decayprocs)
            
        decay_chains = diagram_generation.MultiProcess(my_processes,
                                                       collect_mirror_procs = True)

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(diagram_generation.DecayChainAmplitudeList(\
                               decay_chains.get('amplitudes')))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        # Check number of groups
        self.assertEqual(len(subproc_groups), 2)
        self.assertEqual([g.get('name') for g in subproc_groups],
                         ['gq_zq_z_qq','qq_zg_z_qq'])

        subprocess_group = subproc_groups[0]
        matrix_elements = subprocess_group.get('matrix_elements')

        exporter = export_cpp.ProcessExporterPythia8(matrix_elements,
                                                 self.mycppwriter)

        # Test .h file output
        exporter.write_process_h_file(\
         writers.CPPWriter(pjoin(self.IOpath,'write_dec_multiprocess_files.h')))

        # Test .cc file output
        text = exporter.get_process_function_definitions()
        my_writer = writers.CPPWriter(pjoin(
                                 self.IOpath,'write_dec_multiprocess_files.cc'))  
        my_writer.write(text)

    @IOTests.createIOTest()
    def testIO_write_cpp_go_process_cc_file(self):
        """ target: cpp_go_process.cc
        """
        #Test writing the .cc C++ standalone file for u u~ > go go

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

        exporter = export_cpp.ProcessExporterCPP(matrix_element,
                                                 self.mycppwriter)

        exporter.write_process_cc_file(\
               writers.CPPWriter(os.path.join(self.IOpath,'cpp_go_process.cc')))

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
                                             self.model, "/tmp",
                                             replace_dict={'include_prefix':'Pythia8/'})
        
        test_file_writers.CheckFileCreate.clean_files

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_write_pythia8_parameter_files(self):
        """Test writing the Pythia model parameter files"""

        goal_file_h = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8
#  MadGraph5_aMC@NLO v. %(version)s, %(date)s
#  By the MadGraph5_aMC@NLO Development Team
#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef Pythia8_parameters_sm_H
#define Pythia8_parameters_sm_H

#include <complex>

#include "Pythia8/ParticleData.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/SusyLesHouches.h"

using namespace std;

namespace Pythia8 {

class Parameters_sm
{
public:

static Parameters_sm* getInstance();

// Model parameters independent of aS
double mdl_WTau,mdl_WH,mdl_WT,mdl_WW,mdl_WZ,mdl_MTA,mdl_MM,mdl_Me,mdl_MH,mdl_MB,mdl_MT,mdl_MC,mdl_MZ,mdl_ymtau,mdl_ymm,mdl_yme,mdl_ymt,mdl_ymb,mdl_ymc,mdl_etaWS,mdl_rhoWS,mdl_AWS,mdl_lamWS,mdl_Gf,aEWM1,ZERO,mdl_lamWS__exp__2,mdl_lamWS__exp__3,mdl_MZ__exp__2,mdl_MZ__exp__4,mdl_sqrt__2,mdl_MH__exp__2,mdl_aEW,mdl_MW,mdl_sqrt__aEW,mdl_ee,mdl_MW__exp__2,mdl_sw2,mdl_cw,mdl_sqrt__sw2,mdl_sw,mdl_g1,mdl_gw,mdl_vev,mdl_vev__exp__2,mdl_lam,mdl_yb,mdl_yc,mdl_ye,mdl_ym,mdl_yt,mdl_ytau,mdl_muH,mdl_ee__exp__2,mdl_sw__exp__2,mdl_cw__exp__2;
std::complex<double> mdl_CKM1x1,mdl_CKM1x2,mdl_complexi,mdl_CKM1x3,mdl_CKM2x1,mdl_CKM2x2,mdl_CKM2x3,mdl_CKM3x1,mdl_CKM3x2,mdl_CKM3x3,mdl_conjg__CKM1x3,mdl_conjg__CKM2x3,mdl_conjg__CKM3x3,mdl_conjg__CKM2x1,mdl_conjg__CKM3x1,mdl_conjg__CKM2x2,mdl_conjg__CKM3x2,mdl_conjg__CKM1x1,mdl_conjg__CKM1x2,mdl_I1x31,mdl_I1x32,mdl_I1x33,mdl_I2x12,mdl_I2x13,mdl_I2x22,mdl_I2x23,mdl_I2x32,mdl_I2x33,mdl_I3x21,mdl_I3x22,mdl_I3x23,mdl_I3x31,mdl_I3x32,mdl_I3x33,mdl_I4x13,mdl_I4x23,mdl_I4x33;
// Model parameters dependent on aS
double aS,mdl_sqrt__aS,G,mdl_G__exp__2;
// Model couplings independent of aS
std::complex<double> GC_1,GC_2,GC_3,GC_4,GC_5,GC_6,GC_7,GC_8,GC_9,GC_13,GC_14,GC_15,GC_16,GC_17,GC_18,GC_19,GC_20,GC_21,GC_22,GC_23,GC_24,GC_25,GC_26,GC_27,GC_28,GC_29,GC_30,GC_31,GC_32,GC_33,GC_34,GC_35,GC_36,GC_37,GC_38,GC_39,GC_40,GC_41,GC_42,GC_43,GC_44,GC_45,GC_46,GC_47,GC_48,GC_49,GC_50,GC_51,GC_52,GC_53,GC_54,GC_55,GC_56,GC_57,GC_58,GC_59,GC_60,GC_61,GC_62,GC_63,GC_64,GC_65,GC_66,GC_67,GC_68,GC_69,GC_70,GC_71,GC_72,GC_73,GC_74,GC_75,GC_76,GC_77,GC_78,GC_79,GC_80,GC_81,GC_82,GC_83,GC_84,GC_85,GC_86,GC_87,GC_88,GC_89,GC_90,GC_91,GC_92,GC_93,GC_94,GC_95,GC_96,GC_97,GC_98,GC_99,GC_100,GC_101,GC_102,GC_103,GC_104,GC_105,GC_106,GC_107,GC_108;
// Model couplings dependent on aS
std::complex<double> GC_12,GC_11,GC_10;

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
"""% misc.get_pkg_info()


        goal_file_cc = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
#  MadGraph5_aMC@NLO v. %(version)s, %(date)s
#  By the MadGraph5_aMC@NLO Development Team
#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <iostream>
#include "Parameters_sm.h"
#include "Pythia8/PythiaStdlib.h"

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
    mdl_WTau=pd->mWidth(15);
mdl_WH=pd->mWidth(25);
mdl_WT=pd->mWidth(6);
mdl_WW=pd->mWidth(24);
mdl_WZ=pd->mWidth(23);
mdl_MTA=pd->m0(15);
mdl_MM=pd->m0(13);
mdl_Me=pd->m0(11);
mdl_MH=pd->m0(25);
mdl_MB=pd->m0(5);
mdl_MT=pd->m0(6);
mdl_MC=pd->m0(4);
mdl_MZ=pd->m0(23);
mdl_ymtau=pd->mRun(15, pd->m0(24));
mdl_ymm=pd->mRun(13, pd->m0(24));
mdl_yme=pd->mRun(11, pd->m0(24));
mdl_ymt=pd->mRun(6, pd->m0(24));
mdl_ymb=pd->mRun(5, pd->m0(24));
mdl_ymc=pd->mRun(4, pd->m0(24));
if(!slhaPtr->getEntry<double>("wolfenstein", 4, mdl_etaWS)){
cout << "Warning, setting mdl_etaWS to 3.410000e-01" << endl;
mdl_etaWS = 3.410000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 3, mdl_rhoWS)){
cout << "Warning, setting mdl_rhoWS to 1.320000e-01" << endl;
mdl_rhoWS = 1.320000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 2, mdl_AWS)){
cout << "Warning, setting mdl_AWS to 8.080000e-01" << endl;
mdl_AWS = 8.080000e-01;}
if(!slhaPtr->getEntry<double>("wolfenstein", 1, mdl_lamWS)){
cout << "Warning, setting mdl_lamWS to 2.253000e-01" << endl;
mdl_lamWS = 2.253000e-01;}
mdl_Gf = M_PI*csm->alphaEM(pow(pd->m0(23),2))*pow(pd->m0(23),2)/(sqrt(2.)*pow(pd->m0(24),2)*(pow(pd->m0(23),2)-pow(pd->m0(24),2)));
aEWM1 = 1./csm->alphaEM(pow(pd->m0(23),2));
ZERO = 0.;
mdl_lamWS__exp__2 = pow(mdl_lamWS,2.);
mdl_CKM1x1 = 1.-mdl_lamWS__exp__2/2.;
mdl_CKM1x2 = mdl_lamWS;
mdl_complexi = std::complex<double>(0.,1.);
mdl_lamWS__exp__3 = pow(mdl_lamWS,3.);
mdl_CKM1x3 = mdl_AWS*mdl_lamWS__exp__3*(-(mdl_etaWS*mdl_complexi)+mdl_rhoWS);
mdl_CKM2x1 = -mdl_lamWS;
mdl_CKM2x2 = 1.-mdl_lamWS__exp__2/2.;
mdl_CKM2x3 = mdl_AWS*mdl_lamWS__exp__2;
mdl_CKM3x1 = mdl_AWS*mdl_lamWS__exp__3*(1.-mdl_etaWS*mdl_complexi-mdl_rhoWS);
mdl_CKM3x2 = -(mdl_AWS*mdl_lamWS__exp__2);
mdl_CKM3x3 = 1.;
mdl_MZ__exp__2 = pow(mdl_MZ,2.);
mdl_MZ__exp__4 = pow(mdl_MZ,4.);
mdl_sqrt__2 = sqrt(2.);
mdl_MH__exp__2 = pow(mdl_MH,2.);
mdl_conjg__CKM1x3 = conj(mdl_CKM1x3);
mdl_conjg__CKM2x3 = conj(mdl_CKM2x3);
mdl_conjg__CKM3x3 = conj(mdl_CKM3x3);
mdl_conjg__CKM2x1 = conj(mdl_CKM2x1);
mdl_conjg__CKM3x1 = conj(mdl_CKM3x1);
mdl_conjg__CKM2x2 = conj(mdl_CKM2x2);
mdl_conjg__CKM3x2 = conj(mdl_CKM3x2);
mdl_conjg__CKM1x1 = conj(mdl_CKM1x1);
mdl_conjg__CKM1x2 = conj(mdl_CKM1x2);
mdl_aEW = 1./aEWM1;
mdl_MW = sqrt(mdl_MZ__exp__2/2.+sqrt(mdl_MZ__exp__4/4.-(mdl_aEW*M_PI*mdl_MZ__exp__2)/(mdl_Gf*mdl_sqrt__2)));
mdl_sqrt__aEW = sqrt(mdl_aEW);
mdl_ee = 2.*mdl_sqrt__aEW*sqrt(M_PI);
mdl_MW__exp__2 = pow(mdl_MW,2.);
mdl_sw2 = 1.-mdl_MW__exp__2/mdl_MZ__exp__2;
mdl_cw = sqrt(1.-mdl_sw2);
mdl_sqrt__sw2 = sqrt(mdl_sw2);
mdl_sw = mdl_sqrt__sw2;
mdl_g1 = mdl_ee/mdl_cw;
mdl_gw = mdl_ee/mdl_sw;
mdl_vev = (2.*mdl_MW*mdl_sw)/mdl_ee;
mdl_vev__exp__2 = pow(mdl_vev,2.);
mdl_lam = mdl_MH__exp__2/(2.*mdl_vev__exp__2);
mdl_yb = (mdl_ymb*mdl_sqrt__2)/mdl_vev;
mdl_yc = (mdl_ymc*mdl_sqrt__2)/mdl_vev;
mdl_ye = (mdl_yme*mdl_sqrt__2)/mdl_vev;
mdl_ym = (mdl_ymm*mdl_sqrt__2)/mdl_vev;
mdl_yt = (mdl_ymt*mdl_sqrt__2)/mdl_vev;
mdl_ytau = (mdl_ymtau*mdl_sqrt__2)/mdl_vev;
mdl_muH = sqrt(mdl_lam*mdl_vev__exp__2);
mdl_I1x31 = mdl_yb*mdl_conjg__CKM1x3;
mdl_I1x32 = mdl_yb*mdl_conjg__CKM2x3;
mdl_I1x33 = mdl_yb*mdl_conjg__CKM3x3;
mdl_I2x12 = mdl_yc*mdl_conjg__CKM2x1;
mdl_I2x13 = mdl_yt*mdl_conjg__CKM3x1;
mdl_I2x22 = mdl_yc*mdl_conjg__CKM2x2;
mdl_I2x23 = mdl_yt*mdl_conjg__CKM3x2;
mdl_I2x32 = mdl_yc*mdl_conjg__CKM2x3;
mdl_I2x33 = mdl_yt*mdl_conjg__CKM3x3;
mdl_I3x21 = mdl_CKM2x1*mdl_yc;
mdl_I3x22 = mdl_CKM2x2*mdl_yc;
mdl_I3x23 = mdl_CKM2x3*mdl_yc;
mdl_I3x31 = mdl_CKM3x1*mdl_yt;
mdl_I3x32 = mdl_CKM3x2*mdl_yt;
mdl_I3x33 = mdl_CKM3x3*mdl_yt;
mdl_I4x13 = mdl_CKM1x3*mdl_yb;
mdl_I4x23 = mdl_CKM2x3*mdl_yb;
mdl_I4x33 = mdl_CKM3x3*mdl_yb;
mdl_ee__exp__2 = pow(mdl_ee,2.);
mdl_sw__exp__2 = pow(mdl_sw,2.);
mdl_cw__exp__2 = pow(mdl_cw,2.);
    }
    void Parameters_sm::setIndependentCouplings(){
    GC_1 = -(mdl_ee*mdl_complexi)/3.;
GC_2 = (2.*mdl_ee*mdl_complexi)/3.;
GC_3 = -(mdl_ee*mdl_complexi);
GC_4 = mdl_ee*mdl_complexi;
GC_5 = mdl_ee__exp__2*mdl_complexi;
GC_6 = 2.*mdl_ee__exp__2*mdl_complexi;
GC_7 = -mdl_ee__exp__2/(2.*mdl_cw);
GC_8 = (mdl_ee__exp__2*mdl_complexi)/(2.*mdl_cw);
GC_9 = mdl_ee__exp__2/(2.*mdl_cw);
GC_13 = mdl_I1x31;
GC_14 = mdl_I1x32;
GC_15 = mdl_I1x33;
GC_16 = -mdl_I2x12;
GC_17 = -mdl_I2x13;
GC_18 = -mdl_I2x22;
GC_19 = -mdl_I2x23;
GC_20 = -mdl_I2x32;
GC_21 = -mdl_I2x33;
GC_22 = mdl_I3x21;
GC_23 = mdl_I3x22;
GC_24 = mdl_I3x23;
GC_25 = mdl_I3x31;
GC_26 = mdl_I3x32;
GC_27 = mdl_I3x33;
GC_28 = -mdl_I4x13;
GC_29 = -mdl_I4x23;
GC_30 = -mdl_I4x33;
GC_31 = -2.*mdl_complexi*mdl_lam;
GC_32 = -4.*mdl_complexi*mdl_lam;
GC_33 = -6.*mdl_complexi*mdl_lam;
GC_34 = (mdl_ee__exp__2*mdl_complexi)/(2.*mdl_sw__exp__2);
GC_35 = -((mdl_ee__exp__2*mdl_complexi)/mdl_sw__exp__2);
GC_36 = (mdl_cw__exp__2*mdl_ee__exp__2*mdl_complexi)/mdl_sw__exp__2;
GC_37 = -mdl_ee/(2.*mdl_sw);
GC_38 = -(mdl_ee*mdl_complexi)/(2.*mdl_sw);
GC_39 = (mdl_ee*mdl_complexi)/(2.*mdl_sw);
GC_40 = (mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_41 = (mdl_CKM1x1*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_42 = (mdl_CKM1x2*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_43 = (mdl_CKM1x3*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_44 = (mdl_CKM2x1*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_45 = (mdl_CKM2x2*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_46 = (mdl_CKM2x3*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_47 = (mdl_CKM3x1*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_48 = (mdl_CKM3x2*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_49 = (mdl_CKM3x3*mdl_ee*mdl_complexi)/(mdl_sw*mdl_sqrt__2);
GC_50 = -(mdl_cw*mdl_ee*mdl_complexi)/(2.*mdl_sw);
GC_51 = (mdl_cw*mdl_ee*mdl_complexi)/(2.*mdl_sw);
GC_52 = -((mdl_cw*mdl_ee*mdl_complexi)/mdl_sw);
GC_53 = (mdl_cw*mdl_ee*mdl_complexi)/mdl_sw;
GC_54 = -mdl_ee__exp__2/(2.*mdl_sw);
GC_55 = -(mdl_ee__exp__2*mdl_complexi)/(2.*mdl_sw);
GC_56 = mdl_ee__exp__2/(2.*mdl_sw);
GC_57 = (-2.*mdl_cw*mdl_ee__exp__2*mdl_complexi)/mdl_sw;
GC_58 = -(mdl_ee*mdl_complexi*mdl_sw)/(6.*mdl_cw);
GC_59 = (mdl_ee*mdl_complexi*mdl_sw)/(2.*mdl_cw);
GC_60 = -(mdl_cw*mdl_ee)/(2.*mdl_sw)-(mdl_ee*mdl_sw)/(2.*mdl_cw);
GC_61 = -(mdl_cw*mdl_ee*mdl_complexi)/(2.*mdl_sw)+(mdl_ee*mdl_complexi*mdl_sw)/(2.*mdl_cw);
GC_62 = (mdl_cw*mdl_ee*mdl_complexi)/(2.*mdl_sw)+(mdl_ee*mdl_complexi*mdl_sw)/(2.*mdl_cw);
GC_63 = (mdl_cw*mdl_ee__exp__2*mdl_complexi)/mdl_sw-(mdl_ee__exp__2*mdl_complexi*mdl_sw)/mdl_cw;
GC_64 = -(mdl_ee__exp__2*mdl_complexi)+(mdl_cw__exp__2*mdl_ee__exp__2*mdl_complexi)/(2.*mdl_sw__exp__2)+(mdl_ee__exp__2*mdl_complexi*mdl_sw__exp__2)/(2.*mdl_cw__exp__2);
GC_65 = mdl_ee__exp__2*mdl_complexi+(mdl_cw__exp__2*mdl_ee__exp__2*mdl_complexi)/(2.*mdl_sw__exp__2)+(mdl_ee__exp__2*mdl_complexi*mdl_sw__exp__2)/(2.*mdl_cw__exp__2);
GC_66 = -(mdl_ee__exp__2*mdl_vev)/(2.*mdl_cw);
GC_67 = (mdl_ee__exp__2*mdl_vev)/(2.*mdl_cw);
GC_68 = -2.*mdl_complexi*mdl_lam*mdl_vev;
GC_69 = -6.*mdl_complexi*mdl_lam*mdl_vev;
GC_70 = -(mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_71 = -(mdl_ee__exp__2*mdl_complexi*mdl_vev)/(4.*mdl_sw__exp__2);
GC_72 = (mdl_ee__exp__2*mdl_complexi*mdl_vev)/(2.*mdl_sw__exp__2);
GC_73 = (mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_74 = -(mdl_ee__exp__2*mdl_vev)/(2.*mdl_sw);
GC_75 = (mdl_ee__exp__2*mdl_vev)/(2.*mdl_sw);
GC_76 = -(mdl_ee__exp__2*mdl_vev)/(4.*mdl_cw)-(mdl_cw*mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_77 = (mdl_ee__exp__2*mdl_vev)/(4.*mdl_cw)-(mdl_cw*mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_78 = -(mdl_ee__exp__2*mdl_vev)/(4.*mdl_cw)+(mdl_cw*mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_79 = (mdl_ee__exp__2*mdl_vev)/(4.*mdl_cw)+(mdl_cw*mdl_ee__exp__2*mdl_vev)/(4.*mdl_sw__exp__2);
GC_80 = -(mdl_ee__exp__2*mdl_complexi*mdl_vev)/2.-(mdl_cw__exp__2*mdl_ee__exp__2*mdl_complexi*mdl_vev)/(4.*mdl_sw__exp__2)-(mdl_ee__exp__2*mdl_complexi*mdl_sw__exp__2*mdl_vev)/(4.*mdl_cw__exp__2);
GC_81 = mdl_ee__exp__2*mdl_complexi*mdl_vev+(mdl_cw__exp__2*mdl_ee__exp__2*mdl_complexi*mdl_vev)/(2.*mdl_sw__exp__2)+(mdl_ee__exp__2*mdl_complexi*mdl_sw__exp__2*mdl_vev)/(2.*mdl_cw__exp__2);
GC_82 = -(mdl_yb/mdl_sqrt__2);
GC_83 = -((mdl_complexi*mdl_yb)/mdl_sqrt__2);
GC_84 = -((mdl_complexi*mdl_yc)/mdl_sqrt__2);
GC_85 = mdl_yc/mdl_sqrt__2;
GC_86 = -mdl_ye;
GC_87 = mdl_ye;
GC_88 = -(mdl_ye/mdl_sqrt__2);
GC_89 = -((mdl_complexi*mdl_ye)/mdl_sqrt__2);
GC_90 = -mdl_ym;
GC_91 = mdl_ym;
GC_92 = -(mdl_ym/mdl_sqrt__2);
GC_93 = -((mdl_complexi*mdl_ym)/mdl_sqrt__2);
GC_94 = -((mdl_complexi*mdl_yt)/mdl_sqrt__2);
GC_95 = mdl_yt/mdl_sqrt__2;
GC_96 = -mdl_ytau;
GC_97 = mdl_ytau;
GC_98 = -(mdl_ytau/mdl_sqrt__2);
GC_99 = -((mdl_complexi*mdl_ytau)/mdl_sqrt__2);
GC_100 = (mdl_ee*mdl_complexi*mdl_conjg__CKM1x1)/(mdl_sw*mdl_sqrt__2);
GC_101 = (mdl_ee*mdl_complexi*mdl_conjg__CKM1x2)/(mdl_sw*mdl_sqrt__2);
GC_102 = (mdl_ee*mdl_complexi*mdl_conjg__CKM1x3)/(mdl_sw*mdl_sqrt__2);
GC_103 = (mdl_ee*mdl_complexi*mdl_conjg__CKM2x1)/(mdl_sw*mdl_sqrt__2);
GC_104 = (mdl_ee*mdl_complexi*mdl_conjg__CKM2x2)/(mdl_sw*mdl_sqrt__2);
GC_105 = (mdl_ee*mdl_complexi*mdl_conjg__CKM2x3)/(mdl_sw*mdl_sqrt__2);
GC_106 = (mdl_ee*mdl_complexi*mdl_conjg__CKM3x1)/(mdl_sw*mdl_sqrt__2);
GC_107 = (mdl_ee*mdl_complexi*mdl_conjg__CKM3x2)/(mdl_sw*mdl_sqrt__2);
GC_108 = (mdl_ee*mdl_complexi*mdl_conjg__CKM3x3)/(mdl_sw*mdl_sqrt__2);
    }
    void Parameters_sm::setDependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr, double alpS){
    aS = alpS;
mdl_sqrt__aS = sqrt(aS);
G = 2.*mdl_sqrt__aS*sqrt(M_PI);
mdl_G__exp__2 = pow(G,2.);
    }
    void Parameters_sm::setDependentCouplings(){
    GC_12 = mdl_complexi*mdl_G__exp__2;
GC_11 = mdl_complexi*G;
GC_10 = -G;
    }

    // Routines for printing out parameters
    void Parameters_sm::printIndependentParameters(){
    cout << "sm model parameters independent of event kinematics:" << endl;
    cout << setw(20) << "mdl_WTau " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_WTau << endl;
cout << setw(20) << "mdl_WH " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_WH << endl;
cout << setw(20) << "mdl_WT " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_WT << endl;
cout << setw(20) << "mdl_WW " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_WW << endl;
cout << setw(20) << "mdl_WZ " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_WZ << endl;
cout << setw(20) << "mdl_MTA " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MTA << endl;
cout << setw(20) << "mdl_MM " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MM << endl;
cout << setw(20) << "mdl_Me " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_Me << endl;
cout << setw(20) << "mdl_MH " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MH << endl;
cout << setw(20) << "mdl_MB " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MB << endl;
cout << setw(20) << "mdl_MT " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MT << endl;
cout << setw(20) << "mdl_MC " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MC << endl;
cout << setw(20) << "mdl_MZ " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MZ << endl;
cout << setw(20) << "mdl_ymtau " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ymtau << endl;
cout << setw(20) << "mdl_ymm " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ymm << endl;
cout << setw(20) << "mdl_yme " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_yme << endl;
cout << setw(20) << "mdl_ymt " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ymt << endl;
cout << setw(20) << "mdl_ymb " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ymb << endl;
cout << setw(20) << "mdl_ymc " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ymc << endl;
cout << setw(20) << "mdl_etaWS " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_etaWS << endl;
cout << setw(20) << "mdl_rhoWS " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_rhoWS << endl;
cout << setw(20) << "mdl_AWS " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_AWS << endl;
cout << setw(20) << "mdl_lamWS " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_lamWS << endl;
cout << setw(20) << "mdl_Gf " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_Gf << endl;
cout << setw(20) << "aEWM1 " << "= " << setiosflags(ios::scientific) << setw(10) << aEWM1 << endl;
cout << setw(20) << "ZERO " << "= " << setiosflags(ios::scientific) << setw(10) << ZERO << endl;
cout << setw(20) << "mdl_lamWS__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_lamWS__exp__2 << endl;
cout << setw(20) << "mdl_CKM1x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM1x1 << endl;
cout << setw(20) << "mdl_CKM1x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM1x2 << endl;
cout << setw(20) << "mdl_complexi " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_complexi << endl;
cout << setw(20) << "mdl_lamWS__exp__3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_lamWS__exp__3 << endl;
cout << setw(20) << "mdl_CKM1x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM1x3 << endl;
cout << setw(20) << "mdl_CKM2x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM2x1 << endl;
cout << setw(20) << "mdl_CKM2x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM2x2 << endl;
cout << setw(20) << "mdl_CKM2x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM2x3 << endl;
cout << setw(20) << "mdl_CKM3x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM3x1 << endl;
cout << setw(20) << "mdl_CKM3x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM3x2 << endl;
cout << setw(20) << "mdl_CKM3x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_CKM3x3 << endl;
cout << setw(20) << "mdl_MZ__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__2 << endl;
cout << setw(20) << "mdl_MZ__exp__4 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__4 << endl;
cout << setw(20) << "mdl_sqrt__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sqrt__2 << endl;
cout << setw(20) << "mdl_MH__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MH__exp__2 << endl;
cout << setw(20) << "mdl_conjg__CKM1x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM1x3 << endl;
cout << setw(20) << "mdl_conjg__CKM2x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM2x3 << endl;
cout << setw(20) << "mdl_conjg__CKM3x3 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM3x3 << endl;
cout << setw(20) << "mdl_conjg__CKM2x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM2x1 << endl;
cout << setw(20) << "mdl_conjg__CKM3x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM3x1 << endl;
cout << setw(20) << "mdl_conjg__CKM2x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM2x2 << endl;
cout << setw(20) << "mdl_conjg__CKM3x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM3x2 << endl;
cout << setw(20) << "mdl_conjg__CKM1x1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM1x1 << endl;
cout << setw(20) << "mdl_conjg__CKM1x2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_conjg__CKM1x2 << endl;
cout << setw(20) << "mdl_aEW " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_aEW << endl;
cout << setw(20) << "mdl_MW " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MW << endl;
cout << setw(20) << "mdl_sqrt__aEW " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sqrt__aEW << endl;
cout << setw(20) << "mdl_ee " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ee << endl;
cout << setw(20) << "mdl_MW__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_MW__exp__2 << endl;
cout << setw(20) << "mdl_sw2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sw2 << endl;
cout << setw(20) << "mdl_cw " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_cw << endl;
cout << setw(20) << "mdl_sqrt__sw2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sqrt__sw2 << endl;
cout << setw(20) << "mdl_sw " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sw << endl;
cout << setw(20) << "mdl_g1 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_g1 << endl;
cout << setw(20) << "mdl_gw " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_gw << endl;
cout << setw(20) << "mdl_vev " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_vev << endl;
cout << setw(20) << "mdl_vev__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_vev__exp__2 << endl;
cout << setw(20) << "mdl_lam " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_lam << endl;
cout << setw(20) << "mdl_yb " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_yb << endl;
cout << setw(20) << "mdl_yc " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_yc << endl;
cout << setw(20) << "mdl_ye " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ye << endl;
cout << setw(20) << "mdl_ym " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ym << endl;
cout << setw(20) << "mdl_yt " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_yt << endl;
cout << setw(20) << "mdl_ytau " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ytau << endl;
cout << setw(20) << "mdl_muH " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_muH << endl;
cout << setw(20) << "mdl_I1x31 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I1x31 << endl;
cout << setw(20) << "mdl_I1x32 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I1x32 << endl;
cout << setw(20) << "mdl_I1x33 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I1x33 << endl;
cout << setw(20) << "mdl_I2x12 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x12 << endl;
cout << setw(20) << "mdl_I2x13 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x13 << endl;
cout << setw(20) << "mdl_I2x22 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x22 << endl;
cout << setw(20) << "mdl_I2x23 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x23 << endl;
cout << setw(20) << "mdl_I2x32 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x32 << endl;
cout << setw(20) << "mdl_I2x33 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I2x33 << endl;
cout << setw(20) << "mdl_I3x21 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x21 << endl;
cout << setw(20) << "mdl_I3x22 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x22 << endl;
cout << setw(20) << "mdl_I3x23 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x23 << endl;
cout << setw(20) << "mdl_I3x31 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x31 << endl;
cout << setw(20) << "mdl_I3x32 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x32 << endl;
cout << setw(20) << "mdl_I3x33 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I3x33 << endl;
cout << setw(20) << "mdl_I4x13 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I4x13 << endl;
cout << setw(20) << "mdl_I4x23 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I4x23 << endl;
cout << setw(20) << "mdl_I4x33 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_I4x33 << endl;
cout << setw(20) << "mdl_ee__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_ee__exp__2 << endl;
cout << setw(20) << "mdl_sw__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sw__exp__2 << endl;
cout << setw(20) << "mdl_cw__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_cw__exp__2 << endl;
    }
    void Parameters_sm::printIndependentCouplings(){
    cout << "sm model couplings independent of event kinematics:" << endl;
    cout << setw(20) << "GC_1 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_1 << endl;
cout << setw(20) << "GC_2 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_2 << endl;
cout << setw(20) << "GC_3 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_3 << endl;
cout << setw(20) << "GC_4 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_4 << endl;
cout << setw(20) << "GC_5 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_5 << endl;
cout << setw(20) << "GC_6 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_6 << endl;
cout << setw(20) << "GC_7 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_7 << endl;
cout << setw(20) << "GC_8 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_8 << endl;
cout << setw(20) << "GC_9 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_9 << endl;
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
cout << setw(20) << "GC_48 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_48 << endl;
cout << setw(20) << "GC_49 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_49 << endl;
cout << setw(20) << "GC_50 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_50 << endl;
cout << setw(20) << "GC_51 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_51 << endl;
cout << setw(20) << "GC_52 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_52 << endl;
cout << setw(20) << "GC_53 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_53 << endl;
cout << setw(20) << "GC_54 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_54 << endl;
cout << setw(20) << "GC_55 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_55 << endl;
cout << setw(20) << "GC_56 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_56 << endl;
cout << setw(20) << "GC_57 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_57 << endl;
cout << setw(20) << "GC_58 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_58 << endl;
cout << setw(20) << "GC_59 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_59 << endl;
cout << setw(20) << "GC_60 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_60 << endl;
cout << setw(20) << "GC_61 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_61 << endl;
cout << setw(20) << "GC_62 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_62 << endl;
cout << setw(20) << "GC_63 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_63 << endl;
cout << setw(20) << "GC_64 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_64 << endl;
cout << setw(20) << "GC_65 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_65 << endl;
cout << setw(20) << "GC_66 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_66 << endl;
cout << setw(20) << "GC_67 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_67 << endl;
cout << setw(20) << "GC_68 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_68 << endl;
cout << setw(20) << "GC_69 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_69 << endl;
cout << setw(20) << "GC_70 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_70 << endl;
cout << setw(20) << "GC_71 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_71 << endl;
cout << setw(20) << "GC_72 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_72 << endl;
cout << setw(20) << "GC_73 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_73 << endl;
cout << setw(20) << "GC_74 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_74 << endl;
cout << setw(20) << "GC_75 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_75 << endl;
cout << setw(20) << "GC_76 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_76 << endl;
cout << setw(20) << "GC_77 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_77 << endl;
cout << setw(20) << "GC_78 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_78 << endl;
cout << setw(20) << "GC_79 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_79 << endl;
cout << setw(20) << "GC_80 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_80 << endl;
cout << setw(20) << "GC_81 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_81 << endl;
cout << setw(20) << "GC_82 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_82 << endl;
cout << setw(20) << "GC_83 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_83 << endl;
cout << setw(20) << "GC_84 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_84 << endl;
cout << setw(20) << "GC_85 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_85 << endl;
cout << setw(20) << "GC_86 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_86 << endl;
cout << setw(20) << "GC_87 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_87 << endl;
cout << setw(20) << "GC_88 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_88 << endl;
cout << setw(20) << "GC_89 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_89 << endl;
cout << setw(20) << "GC_90 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_90 << endl;
cout << setw(20) << "GC_91 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_91 << endl;
cout << setw(20) << "GC_92 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_92 << endl;
cout << setw(20) << "GC_93 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_93 << endl;
cout << setw(20) << "GC_94 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_94 << endl;
cout << setw(20) << "GC_95 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_95 << endl;
cout << setw(20) << "GC_96 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_96 << endl;
cout << setw(20) << "GC_97 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_97 << endl;
cout << setw(20) << "GC_98 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_98 << endl;
cout << setw(20) << "GC_99 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_99 << endl;
cout << setw(20) << "GC_100 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_100 << endl;
cout << setw(20) << "GC_101 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_101 << endl;
cout << setw(20) << "GC_102 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_102 << endl;
cout << setw(20) << "GC_103 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_103 << endl;
cout << setw(20) << "GC_104 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_104 << endl;
cout << setw(20) << "GC_105 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_105 << endl;
cout << setw(20) << "GC_106 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_106 << endl;
cout << setw(20) << "GC_107 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_107 << endl;
cout << setw(20) << "GC_108 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_108 << endl;
    }
    void Parameters_sm::printDependentParameters(){
    cout << "sm model parameters dependent on event kinematics:" << endl;
    cout << setw(20) << "aS " << "= " << setiosflags(ios::scientific) << setw(10) << aS << endl;
cout << setw(20) << "mdl_sqrt__aS " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_sqrt__aS << endl;
cout << setw(20) << "G " << "= " << setiosflags(ios::scientific) << setw(10) << G << endl;
cout << setw(20) << "mdl_G__exp__2 " << "= " << setiosflags(ios::scientific) << setw(10) << mdl_G__exp__2 << endl;
    }
    void Parameters_sm::printDependentCouplings(){
    cout << "sm model couplings dependent on event kinematics:" << endl;
    cout << setw(20) << "GC_12 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_12 << endl;
cout << setw(20) << "GC_11 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_11 << endl;
cout << setw(20) << "GC_10 " << "= " << setiosflags(ios::scientific) << setw(10) << GC_10 << endl;
    }

} // end namespace Pythia8
""" % misc.get_pkg_info()

        file_h, file_cc = self.model_builder.generate_parameters_class_files()
        self.assertEqual(file_h.split('\n'), goal_file_h.split('\n'))
        self.assertEqual(file_cc.replace('\t', '    ').split('\n'), goal_file_cc.replace('\t', '    ').split('\n'))



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

        exporter = export_cpp.ProcessExporterMatchbox( mymatrixelement, self.mycppwriter, process_string="q q > six")
        
        self.assertRaises(export_cpp.ProcessExporterCPP.ProcessExporterCPPError,
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

        exporter = export_cpp.ProcessExporterMatchbox(matrix_element,
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



