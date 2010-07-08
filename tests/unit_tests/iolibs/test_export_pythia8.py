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
        a = mypartlist[len(mypartlist) - 1]

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
                                             a]),
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

namespace Pythia8 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u~ > u u~
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
    virtual string name() const 
    {
      return "u u~ > u u~ (SM)"; 
    }

    virtual int code() const 
    {
      return 10000; 
    }

    virtual string inFlux() const 
    {
      return "qqbarSame"; 
    }

    virtual int id3Mass() const 
    {
      return 2; 
    }
    virtual int id4Mass() const 
    {
      return - 2; 
    }

    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertME() const 
    {
      return true; 
    }

  private:

    // Private function to calculate the matrix element for given helicities
    double matrix(int helicities[]); 

    // Constants for array limits
    const int nexternal = 4; 
    const int ncolor = 2; 

    // Store the matrix element value from sigmaKin
    double matrix_element; 

    // Color flows, used when selecting color
    double jamp2[ncolor]; 

    // Other process-specific information, e.g. couplings

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

#include < complex.h > 

#include "Sigma_uux_uux.h"

namespace Pythia8 
{

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > u u~

//--------------------------------------------------------------------------
// Initialize process.

void Sigma_uux_uux::initProc() 
{


}

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), part independent of incoming flavour.

void Sigma_uux_uux::sigmaKin() 
{

  const int _ncomb = 16; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  // Helicities for the process
  static const int helicity[ncomb][nexternal] = {-1, -1, -1, -1, -1, -1, -1, 1,
      -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1,
      -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1,
      1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  // Denominator: spins, colors and identical particles
  const int denominator = 36; 

  ntry = ntry + 1; 

  for(int i = 0; i < ngraphs; i++ )
    jamp2(i) = 0.; 

  matrix_element = 0.; 
  if (sum_hel == 0 || ntry < 10)
  {
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        double t = matrix(nhel[ihel]); 
        matrix_element += t; 
        if (t .ne. != 0. && !goodhel[ihel])
        {
          goodhel[ihel] = true; 
          ngood++; 
          igood[ngood] = ihel; 
        }
      }
    }
    jhel = 1; 
    sum_hel = min(sum_hel, ngood); 
  }
  else
    // random helicity
  {
    for(int j = 0; j < sum_hel; j++ )
    {
      jhel++; 
      if (jhel > ngood)
        jhel = 1; 
      double hwgt = double(ngood)/double(sum_hel); 
      int ihel = igood(jhel); 
      t = matrix(nhel[ihel]); 
      matrix_element += t * hwgt; 
    }
    matrix_element /= denominator; 
  }

}

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), including incoming flavour dependence.

double Sigma_uux_uux::sigmaHat() 
{


}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void Sigma_uux_uux::setIdColAcol() 
{


}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process

double Sigma_uux_uux::weightDecay(Event& process, int iResBeg, int iResEnd) 
{

  return 1.; 
}

//--------------------------------------------------------------------------
// Evaluate d(sigmaHat)/d(tHat), part independent of incoming flavour.

double Sigma_uux_uux::matrix(int nhel[]) 
{

  // Local variables
  const int nwavefuncs = 8, ngraphs = 4; 
  double zero = 0.; 
  double p[nexternal][4]; 
  int i, j; 
  complex ztemp; 
  complex amp[ngraphs], jamp[ncolor]; 
  complex w[nwavefuncs][18]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {9, 3, 3, 9}; 

  // Calculate all amplitudes
  ixxxxx(p[0], zero, nhel[0], +1, w[0])); 
  oxxxxx(p[1], zero, nhel[1], -1, w[1])); 
  oxxxxx(p[2], zero, nhel[2], +1, w[2])); 
  ixxxxx(p[3], zero, nhel[3], -1, w[3])); 
  FFV_110(w[0], w[1], QQG, zero, zero, w[4]); 
  // Amplitude(s) for diagram number 1
  FFV_111(w[3], w[2], w[4], QQG, amp[0]); 
  FFV_110(w[0], w[1], QQA, zero, zero, w[5]); 
  // Amplitude(s) for diagram number 2
  FFV_111(w[3], w[2], w[5], QQA, amp[1]); 
  FFV_110(w[0], w[2], QQG, zero, zero, w[6]); 
  // Amplitude(s) for diagram number 3
  FFV_111(w[3], w[1], w[6], QQG, amp[2]); 
  FFV_110(w[0], w[2], QQA, zero, zero, w[7]); 
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
    jamp2[i] += jamp[i] * conj(jamp[i]); 

  return matrix; 

}

}  // end namespace Pythia

""" % misc.get_pkg_info()

        color_amplitudes = self.mymatrixelement.get_color_amplitudes()

        export_pythia8.write_pythia8_process_cc_file(\
            writers.CPPWriter(self.give_pos('test_cc')),
            self.mymatrixelement, self.mycppmodel,
            color_amplitudes)

        self.assertFileContains('test_cc', goal_string)

