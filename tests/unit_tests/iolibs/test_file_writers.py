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

"""Unit test library for the writer classes"""

import StringIO
import re
import os

import tests.unit_tests as unittest

import madgraph.iolibs.file_writers as writers

#===============================================================================
# FortranWriterTest
#===============================================================================
class CheckFileCreate():
    """Check that the files are correctly created"""

    output_path = '/tmp/' # work only on LINUX but that's ok for the test routine
    created_files =[]

    def assertFileContains(self, filename, solution):
        """ Check the content of a file """
        
        current_value = open(self.give_pos(filename)).read()
        list_cur=current_value.split('\n')
        list_sol=solution.split('\n')
        while 1:
            if '' in list_sol:
                list_sol.remove('')
            else:
                break
        while 1:
            if '' in list_cur:
                list_cur.remove('')
            else:
                break            
        for a, b in zip(list_sol, list_cur):
            self.assertEqual(a,b)
        #for a, b in zip(current_value.split('\n'), solution.split('\n')):
        #    self.assertEqual(a,b)
        #self.assertEqual(current_value.split('\n'), solution.split('\n'))
        self.assertEqual(len(list_sol), len(list_cur))

    def give_pos(self, filename):
        """ take a name and a change it in order to have a valid path in the output directory """
        
        return os.path.join(self.output_path, filename)

    def clean_files(self):
        """ suppress all the files linked to this test """
        
        for filename in self.created_files:
            try:
                os.remove(self.give_pos(filename))
            except OSError:
                pass
    

class FortranWriterTest(unittest.TestCase, CheckFileCreate):
    """Test class for the Fortran writer object"""

    created_files = ['fortran_test'
                    ]

    # clean all the tested files before and after any test
    setUP = CheckFileCreate.clean_files
    tearDown = CheckFileCreate.clean_files

    def test_write_fortran_line(self):
        """Test writing a fortran line"""

        lines = []
        lines.append(" call aaaaaa(bbb, ccc, ddd, eee, fff, ggg, hhhhhhhhhhhhhh+asdasd, wspedfteispd)")

        lines.append('  include "test.inc"')
        lines.append(' print *, \'Hej \\"Da\\" Mo\'')
        lines.append("  IF (Test) then")
        lines.append(" if(mutt) call hej")
        lines.append(" else if(test) then")
        lines.append("c Test")
        lines.append("c = hej")
        lines.append(" Call hej")
        lines.append("# Test")
        lines.append("20 else")
        lines.append("bah=2")
        lines.append(" endif")
        lines.append("test")

        goal_string = """      CALL AAAAAA(BBB, CCC, DDD, EEE, FFF, GGG, HHHHHHHHHHHHHH
     $ +ASDASD, WSPEDFTEISPD)
      INCLUDE 'test.inc'
      PRINT *, 'Hej \\'Da\\' Mo'
      IF (TEST) THEN
        IF(MUTT) CALL HEJ
      ELSE IF(TEST) THEN
C       Test
        C = HEJ
        CALL HEJ
C       Test
 20   ELSE
        BAH=2
      ENDIF
      TEST\n"""

        writer = writers.FortranWriter(self.give_pos('fortran_test')).\
                 writelines(lines)

        # Check that the output stays the same
        self.assertFileContains('fortran_test',
                                 goal_string)

    def test_write_fortran_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.FortranWriter(os.devnull)
        for nonstring in non_strings:
            self.assertRaises(AssertionError,
                              writer.write_line,
                              nonstring)

#===============================================================================
# CPPWriterTest
#===============================================================================
class CPPWriterTest(unittest.TestCase, CheckFileCreate):
    """Test class for the C++ writer object"""

    created_files = ['cpp_test'
                    ]

    # clean all the tested files before and after any test
    setUP = CheckFileCreate.clean_files
    tearDown = CheckFileCreate.clean_files

    def test_write_cplusplus_line(self):
        """Test writing a cplusplus line"""

        fsock = StringIO.StringIO()

        lines = """#ifndef Pythia8_SigmaEW_H
#define Pythia8_SigmaEW_H

#include "PythiaComplex.h"
#include "SigmaProcess.h"

namespace Pythia8 {

 
/*==========================================================================

A derived class for q g -> q gamma (q = u, d, s, c, b).
Use massless approximation also for Q since no alternative.*/

class Sigma2qg2qgamma : public Sigma2Process {

public:

  // Constructor.
  Sigma2qg2qgamma() {   }

  // Calculate flavour-independent parts of cross section.
  virtual void sigmaKin();

  // Evaluate d(sigmaHat)/d(tHat). 
  virtual double sigmaHat();

  // Select flavour, colour and anticolour.
  virtual void setIdColAcol();

  // Info on the subprocess.
  virtual string name(  )   const {
      return "q g -> q gamma(udscb) test test test test test asasd as asd a dada djkl;sdf lkja sdfjkla;sdf l;kja+sdfkldf";}
  virtual int    code()   const {return 201e-3+2.E3+.01e+2+1E+3;}
  virtual string inFlux() const {return "qg";}

private:

  // Values stored for later use.
  double mNew, m2New, sigUS, sigma0; // Just qg

};

    // Select identity, colour and anticolour.

void Sigma2ff2fftgmZ::setIdColAcol() {

  // Trivial flavours: out = in.
  setId( id1, id2,    id1,id2);

  // Colour flow topologies. Swap when antiquarks.
  if (abs(id1)<9 && abs(id2)<9  &&  id1*id2>2/3.) 
                         setColAcol(1,0,2,0,1,0,2,0);
  else if (abs(id1)<9 &&abs(id2)< 9)
                         setColAcol(1,0,0,2,1,0,0,2); 
  else                   setColAcol(0,0,0,0,0,0,0,0);

  if ( (abs(id1)!=9&&id1<0)||(abs(id1 )==10 &&    id2 < 0) ) 
    swapColAcol( ) ;

  template< double > > hej;

}    """.split("\n")

        goal_string = """#ifndef Pythia8_SigmaEW_H
#define Pythia8_SigmaEW_H

#include "PythiaComplex.h"
#include "SigmaProcess.h"

namespace Pythia8 
{


//==========================================================================
// 
// A derived class for q g -> q gamma (q = u, d, s, c, b).
// Use massless approximation also for Q since no alternative.

class Sigma2qg2qgamma : public Sigma2Process 
{

  public:

    // Constructor.
    Sigma2qg2qgamma() {}

    // Calculate flavour-independent parts of cross section.
    virtual void sigmaKin(); 

    // Evaluate d(sigmaHat)/d(tHat).
    virtual double sigmaHat(); 

    // Select flavour, colour and anticolour.
    virtual void setIdColAcol(); 

    // Info on the subprocess.
    virtual string name() const 
    {
      return "q g -> q gamma(udscb) test test test test test asasd as asd a dada djkl;sdf lkja sdfjkla;sdf l;kja+sdfkldf"; 
    }
    virtual int code() const {return 201e-3 + 2.e3 + .01e+2 + 1e+3;}
    virtual string inFlux() const {return "qg";}

  private:

    // Values stored for later use.
    double mNew, m2New, sigUS, sigma0;  // Just qg

}; 

// Select identity, colour and anticolour.

void Sigma2ff2fftgmZ::setIdColAcol() 
{

  // Trivial flavours: out = in.
  setId(id1, id2, id1, id2); 

  // Colour flow topologies. Swap when antiquarks.
  if (abs(id1) < 9 && abs(id2) < 9 && id1 * id2 > 2/3.)
    setColAcol(1, 0, 2, 0, 1, 0, 2, 0); 
  else if (abs(id1) < 9 && abs(id2) < 9)
    setColAcol(1, 0, 0, 2, 1, 0, 0, 2); 
  else
    setColAcol(0, 0, 0, 0, 0, 0, 0, 0); 

  if ((abs(id1) != 9 && id1 < 0) || (abs(id1) == 10 && id2 < 0))
    swapColAcol(); 

  template<double> > hej; 

}
"""

        writer = writers.CPPWriter(self.give_pos('cpp_test')).\
                 writelines(lines)

        # Check that the output stays the same
        self.assertFileContains('cpp_test',
                                 goal_string)

    def test_write_cplusplus_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.CPPWriter(os.devnull)
        for nonstring in non_strings:
            self.assertRaises(AssertionError,
                              writer.write_line,
                              nonstring)

