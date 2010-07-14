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
import unittest
import copy
import fractions

import madgraph.iolibs.misc as misc
import madgraph.iolibs.export_pythia8 as export_pythia8
import madgraph.iolibs.file_writers as writers
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_algebra as color
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
    mycppmodel = export_pythia8.UFOHelasCPPModel()
    created_files = ['test_h', 'test_cc'
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
                      'mass':'mu',
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

        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'mu',
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
                      'lorentz':['FFV'],
                      'couplings':{(0, 0):'QQG'},
                      'orders':{'QCD':1}}))

        # Gamma couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1, 0)])],
                      'lorentz':['FFV'],
                      'couplings':{(0, 0):'QQA'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('name', 'SM')

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

        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)

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

        self.mymatrixelement.get('processes').append(myproc)
        
    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_write_process_h_file(self):
        """Test writing the .h Pythia file for a matrix element"""

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8
// by MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#ifndef Pythia8_Sigma_uux_uux_H
#define Pythia8_Sigma_uux_uux_H

#include "SigmaProcess.h"

using namespace std; 

namespace Pythia8 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u~ > u u~
// Process: c c~ > c c~
//--------------------------------------------------------------------------

class Sigma_uux_uux : public Sigma2Process
{
  public:

    // Constructor.
    Sigma_uux_uux() {}

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
    virtual string name() const {return "u u~ > u u~ (SM)";}

    virtual int code() const {return 10000;}

    virtual string inFlux() const {return "qqbarSame";}

    int id3Mass() const {return 2;}
    int id4Mass() const {return 2;}

    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertME() const {return true;}

  private:

    // Private function to calculate the matrix element for given helicities
    double matrix(const int helicities[]); 

    // Private functions to set the couplings and parameters used in this
    // process
    void set_fixed_parameters(); 
    void set_variable_parameters(); 

    // Constants for array limits
    static const int nexternal = 4; 
    static const int ncolor = 2; 

    // Store the matrix element value from sigmaKin
    double matrix_element; 

    // Color flows, used when selecting color
    double jamp2[ncolor]; 

    // Other process-specific information, e.g. masses and couplings
    // Propagator masses
    double MZ; 
    // Propagator widths
    double WZ; 
    // Couplings
    complex QQG, QQA; 
}; 

}  // end namespace Pythia

#endif  // Pythia8_Sigma_uux_uux_H
""" % misc.get_pkg_info()

        export_pythia8.write_pythia8_process_h_file(\
            writers.CPPWriter(self.give_pos('test_h')),
            self.mymatrixelement)

        self.assertFileContains('test_h', goal_string)

    def test_write_process_cc_file(self):
        """Test writing the .cc Pythia file for a matrix element"""

        goal_string = \
"""//==========================================================================
// This file has been automatically generated for Pythia 8 by
// by MadGraph 5 v. %(version)s, %(date)s
// By the MadGraph Development Team
// Please visit us at https://launchpad.net/madgraph5
//==========================================================================

#include "Sigma_uux_uux.h"

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > u u~
// Process: c c~ > c c~

//--------------------------------------------------------------------------
// Initialize process.

void Sigma_uux_uux::initProc() 
{
  // Set all parameters that are fixed once and for all
  set_fixed_parameters(); 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void Sigma_uux_uux::sigmaKin() 
{
  // Local variables and constants
  const int ncomb = 16; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  double t; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {-1, -1, -1, -1, -1, -1, -1,
      1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1,
      -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1,
      1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  // Denominator: spins, colors and identical particles
  const int denominator = 36; 

  ntry = ntry + 1; 

  // Set the parameters which change event by event
  set_variable_parameters(); 

  // Reset color flows
  for(int i = 0; i < ncolor; i++ )
    jamp2[i] = 0.; 

  // Calculate the matrix element
  matrix_element = 0.; 

  if (sum_hel == 0 || ntry < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        t = matrix(helicities[ihel]); 
        matrix_element += t; 
        // Store which helicities give non-zero result
        if (t != 0. && !goodhel[ihel])
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
      t = matrix(helicities[ihel]); 
      matrix_element += t * hwgt; 
    }
    matrix_element /= denominator; 
  }

}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double Sigma_uux_uux::sigmaHat() 
{
  // Already calculated matrix_element in sigmaKin
  return matrix_element; 
}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void Sigma_uux_uux::setIdColAcol() 
{
  if(id1 == 4 && id2 == -4)
  {
    // Pick one of the flavor combinations [[4, -4]]
    int flavors[1][2] = {4, -4}; 
    vector<double> probs(1, 1./1.); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
    id4 = flavors[choice][1]; 
  }
  else if(id1 == 2 && id2 == -2)
  {
    // Pick one of the flavor combinations [[2, -2]]
    int flavors[1][2] = {2, -2}; 
    vector<double> probs(1, 1./1.); 
    int choice = rndmPtr->pick(probs); 
    id3 = flavors[choice][0]; 
    id4 = flavors[choice][1]; 
  }
  setId(id1, id2, id3, id4); 
  vector<double> probs; 
  double sum = jamp2[0] + jamp2[1]; 
  for(int i = 0; i < ncolor; i++ )
    probs.push_back(jamp2[i]/sum); 
  int ic = rndmPtr->pick(probs); 
  static int col[2][8] = {1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1}; 
  setColAcol(col[ic][0], col[ic][1], col[ic][2], col[ic][3], col[ic][4],
      col[ic][5], col[ic][6], col[ic][7]);
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process

double Sigma_uux_uux::weightDecay(Event& process, int iResBeg, int iResEnd) 
{
  // Just use isotropic decay (default)
  return 1.; 
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for a given helicity

double Sigma_uux_uux::matrix(const int hel[]) 
{
  // Local variables
  const int nwavefuncs = 8, ngraphs = 4; 
  const double zero = 0., ZERO = 0.; 
  int i, j; 
  complex ztemp; 
  complex amp[ngraphs], jamp[ncolor]; 
  complex w[nwavefuncs][18]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {9, 3, 3, 9}; 

  // Calculate all amplitudes
  ixxxxx(pME[0], mME[0], hel[0], +1, w[0]); 
  oxxxxx(pME[1], mME[1], hel[1], -1, w[1]); 
  oxxxxx(pME[2], mME[2], hel[2], +1, w[2]); 
  ixxxxx(pME[3], mME[3], hel[3], -1, w[3]); 
  FFV_110(w[0], w[1], QQG, zero, zero, w[4]); 
  // Amplitude(s) for diagram number 1
  FFV_111(w[3], w[2], w[4], QQG, amp[0]); 
  FFV_110(w[0], w[1], QQA, MZ, WZ, w[5]); 
  // Amplitude(s) for diagram number 2
  FFV_111(w[3], w[2], w[5], QQA, amp[1]); 
  FFV_110(w[0], w[2], QQG, zero, zero, w[6]); 
  // Amplitude(s) for diagram number 3
  FFV_111(w[3], w[1], w[6], QQG, amp[2]); 
  FFV_110(w[0], w[2], QQA, MZ, WZ, w[7]); 
  // Amplitude(s) for diagram number 4
  FFV_111(w[3], w[1], w[7], QQA, amp[3]); 

  // Calculate color flows
  jamp[0] = +1./6. * amp[0] - amp[1] + 1./2. * amp[2]; 
  jamp[1] = -1./2. * amp[0] - 1./6. * amp[2] + amp[3]; 

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 1; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 

}

//--------------------------------------------------------------------------
// Set couplings and other parameters that are fixed during the run

void Sigma_uux_uux::set_fixed_parameters() 
{
  // Propagator masses and widths
  MZ = ParticleData::m0(23); 
  WZ = ParticleData::mWidth(23); 
}

//--------------------------------------------------------------------------
// Set couplings and other parameters that vary event by event

void Sigma_uux_uux::set_variable_parameters() 
{
  // Couplings
  QQG = expression; 
  QQA = expression; 
}


}  // end namespace Pythia

""" % misc.get_pkg_info()

        color_amplitudes = self.mymatrixelement.get_color_amplitudes()

        export_pythia8.write_pythia8_process_cc_file(\
            writers.CPPWriter(self.give_pos('test_cc')),
            self.mymatrixelement, self.mycppmodel,
            color_amplitudes)

        self.assertFileContains('test_cc', goal_string)

