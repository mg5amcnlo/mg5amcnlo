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
import copy
import fractions
import os
import re

import tests.unit_tests as unittest

import aloha.aloha_writers as aloha_writers

import madgraph.iolibs.export_pythia8 as export_pythia8
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.helas_call_writers as helas_call_writer
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

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
                      'mass':'ZERO',
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

        self.mycppmodel = helas_call_writer.Pythia8UFOHelasCallWriter(self.mymodel)
    
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
#include "Parameters_sm.h"

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
    virtual string name() const {return "u u~ > u u~ (sm)";}

    virtual int code() const {return 10000;}

    virtual string inFlux() const {return "qqbarSame";}

    int id3Mass() const {return 2;}
    int id4Mass() const {return 2;}

    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertM2() const {return true;}

  private:

    // Private function to calculate the matrix element for given helicities
    double matrix(const int helicities[]); 

    // Constants for array limits
    static const int nexternal = 4; 
    static const int ncolor = 2; 

    // Store the matrix element value from sigmaKin
    double matrix_element; 

    // Color flows, used when selecting color
    double jamp2[ncolor]; 

    // Pointer to the model parameters
    Parameters_sm * pars; 

}; 

}  // end namespace Pythia

#endif  // Pythia8_Sigma_uux_uux_H
""" % misc.get_pkg_info()

        export_pythia8.write_pythia8_process_h_file(\
            writers.CPPWriter(self.give_pos('test.h')),
            self.mymatrixelement)

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

#include "Sigma_uux_uux.h"
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

void Sigma_uux_uux::initProc() 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  pars->setIndependentParameters(particleDataPtr, coupSMPtr); 
  pars->setIndependentCouplings(particleDataPtr, coupSMPtr); 

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
  pars->setDependentParameters(particleDataPtr, coupSMPtr, alpS); 
  pars->setDependentCouplings(particleDataPtr, coupSMPtr); 

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
  }
  matrix_element /= denominator; 

}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double Sigma_uux_uux::sigmaHat() 
{
  // Return 0 if not correct initial state assignment
  if( !((id1 == 4 && id2 == -4) || (id1 == 2 && id2 == -2)))
  {
    return 0; 
  }
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
  const int nwavefuncs = 10, ngraphs = 10; 
  int i, j; 
  complex ztemp; 
  complex amp[ngraphs], jamp[ncolor]; 
  complex w[nwavefuncs][18]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {9, 3, 3, 9}; 

  // Convert Pythia 4-vectors to double[]
  double p[nexternal][4]; 
  for(i = 0; i < nexternal; i++ )
  {
    p[i][0] = pME[i].e(); 
    p[i][1] = pME[i].px(); 
    p[i][2] = pME[i].py(); 
    p[i][3] = pME[i].pz(); 
  }

  // Calculate all amplitudes
  Pythia8_sm::ixxxxx(p[0], mME[0], hel[0], +1, w[0]); 
  Pythia8_sm::oxxxxx(p[1], mME[1], hel[1], -1, w[1]); 
  Pythia8_sm::oxxxxx(p[2], mME[2], hel[2], +1, w[2]); 
  Pythia8_sm::ixxxxx(p[3], mME[3], hel[3], -1, w[3]); 
  FFV1_3(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[4]); 
  // Amplitude(s) for diagram number 1
  FFV1_0(w[3], w[2], w[4], pars->GC_10, amp[0]); 
  FFV5_3(w[0], w[1], pars->GC_47, pars->MZ, pars->WZ, w[5]); 
  FFV2_3(w[0], w[1], pars->GC_35, pars->MZ, pars->WZ, w[6]); 
  // Amplitude(s) for diagram number 2
  FFV5_0(w[3], w[2], w[5], pars->GC_47, amp[1]); 
  FFV2_0(w[3], w[2], w[5], pars->GC_35, amp[2]); 
  FFV5_0(w[3], w[2], w[6], pars->GC_47, amp[3]); 
  FFV2_0(w[3], w[2], w[6], pars->GC_35, amp[4]); 
  FFV1_3(w[0], w[2], pars->GC_10, pars->ZERO, pars->ZERO, w[7]); 
  // Amplitude(s) for diagram number 3
  FFV1_0(w[3], w[1], w[7], pars->GC_10, amp[5]); 
  FFV5_3(w[0], w[2], pars->GC_47, pars->MZ, pars->WZ, w[8]); 
  FFV2_3(w[0], w[2], pars->GC_35, pars->MZ, pars->WZ, w[9]); 
  // Amplitude(s) for diagram number 4
  FFV5_0(w[3], w[1], w[8], pars->GC_47, amp[6]); 
  FFV2_0(w[3], w[1], w[8], pars->GC_35, amp[7]); 
  FFV5_0(w[3], w[1], w[9], pars->GC_47, amp[8]); 
  FFV2_0(w[3], w[1], w[9], pars->GC_35, amp[9]); 

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
    jamp2[i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 

}

}  // end namespace Pythia

""" % misc.get_pkg_info()

        color_amplitudes = self.mymatrixelement.get_color_amplitudes()

        export_pythia8.write_pythia8_process_cc_file(\
            writers.CPPWriter(self.give_pos('test.cc')),
            self.mymatrixelement, self.mycppmodel,
            color_amplitudes)

        self.assertFileContains('test.cc', goal_string)


    def test_write_process_files(self):
        """Test writing the .h  and .cc Pythia file for a matrix element"""

        export_pythia8.generate_process_files_pythia8(self.mymatrixelement,
                                                      self.mycppmodel,
                                                      "/tmp")
        
        print "Please try compiling the file /tmp/Sigma_uux_uux.cc:"
        print "cd /tmp; g++ -c -I $PATH_TO_PYTHIA8/include Sigma_uux_uux.cc"

        
#===============================================================================
# ExportUFOModelPythia8Test
#===============================================================================
class ExportUFOModelPythia8Test(unittest.TestCase,
                                test_file_writers.CheckFileCreate):

    created_files = [
                    ]

    def setUp(self):

        model_pkl = os.path.join('models','sm','model.pkl')
        if os.path.isfile(model_pkl):
            self.model = save_load_object.load_from_file(model_pkl)
        else:
            self.model = import_ufo.import_model('sm')
        self.model_builder = export_pythia8.UFO_model_to_pythia8(self.model,
                                                                 "/tmp")
        
        test_file_writers.CheckFileCreate.clean_files

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_read_aloha_template_files(self):
        """Test reading the ALOHA template .h and .cc files"""

        template_h = self.model_builder.read_aloha_template_files("h")
#        print ".h:"
#        print "\n".join(template_h)
        self.assertTrue(template_h)
        for file_lines in template_h:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1)
        template_cc = self.model_builder.read_aloha_template_files("cc")
#        print ".cc:"
#        print "\n".join(template_cc)
        self.assertTrue(template_cc)
        for file_lines in template_cc:
            self.assertFalse(file_lines.find('#include') > -1)
            self.assertFalse(file_lines.find('namespace') > -1) 
       
    def test_write_aloha_functions(self):
        """Test writing function declarations and definitions"""

        template_h_files = []
        template_cc_files = []

        for abstracthelas in self.model_builder.model.get('lorentz').values():
            abstracthelas.write('/tmp', 'CPP')
            abstracthelas.write('/tmp', 'Fortran')

        print "Please try compiling the files /tmp/*.cc and /tmp/*.f:"
        print "cd /tmp; g++ -c *.cc; gfortran -c *.f"
        

    def test_write_aloha_routines(self):
        """Test writing the aloha .h and.cc files"""

        self.model_builder.write_aloha_routines()
        print "Please try compiling the file /tmp/hel_amps_sm.cc:"
        print "cd /tmp; g++ -c hel_amps_sm.cc"

    def test_couplings_and_parameters(self):
        """Test generation of couplings and parameters"""

        self.assertTrue(self.model_builder.params_indep)
        self.assertTrue(self.model_builder.params_dep)
        self.assertTrue(self.model_builder.coups_indep)
        self.assertTrue(self.model_builder.coups_dep)

##        print "Dependent parameters: "
##        print "\n".join(["%s: %s" %(p.name, p.expr) for p in self.model_builder.params_dep])
##        print "Dependent couplings: "
##        print "\n".join(["%s: %s" %(p.name, p.expr) for p in self.model_builder.coups_dep.values()])

        g_expr = re.compile("G(?!f)")

##        print "Independent parameters: "
        for indep_par in self.model_builder.params_indep:
##            print "%s: %s" %(indep_par.name, indep_par.expr)
            self.assertFalse(g_expr.search(indep_par.expr))
##        print "Independent couplings: "
        for indep_coup in self.model_builder.coups_indep: 
##            print "%s: %s" %(indep_coup.name, indep_coup.expr)
            self.assertFalse(g_expr.search(indep_coup.expr))

    def test_write_parameter_files(self):
        """Test writing the model parameter .h and.cc files"""

        self.model_builder.write_parameter_class_files()        
        
        print "Please try compiling the file /tmp/Parameters_sm.cc:"
        print "cd /tmp; g++ -c -I $PATH_TO_PYTHIA8/include Parameters_sm.cc"

##    def test_write_parameters_and_couplings(self):
##        """Test generation of couplings and parameters"""

##        print "Independent parameters: "
##        print self.model_builder.write_parameters(self.model_builder.params_indep)
##        print "Dependent parameters: "
##        print self.model_builder.write_parameters(self.model_builder.params_dep)
##        print "Independent couplings: "
##        print self.model_builder.write_parameters(self.model_builder.coups_indep)
##        print "Independent couplings: "
##        print self.model_builder.write_parameters(self.model_builder.coups_indep)
##        print "Dependent couplings: "
##        print self.model_builder.write_parameters(self.model_builder.coups_dep.values())

##    def test_write_set_parameters_and_couplings(self):
##        """Test generation of couplings and parameters"""

##        print "Independent parameters: "
##        print self.model_builder.write_set_parameters(self.model_builder.params_indep)
##        print "Dependent parameters: "
##        print self.model_builder.write_set_parameters(self.model_builder.params_dep)
##        print "Independent couplings: "
##        print self.model_builder.write_set_parameters(self.model_builder.coups_indep)
##        print "Dependent couplings: "
##        print self.model_builder.write_set_parameters(self.model_builder.coups_dep.values())
