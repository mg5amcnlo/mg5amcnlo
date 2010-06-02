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

"""Unit test library for the writer classes"""

import StringIO
import unittest
import re

import madgraph.iolibs.file_writers as writers

#===============================================================================
# FortranWriterTest
#===============================================================================
class FortranWriterTest(unittest.TestCase):
    """Test class for the Fortran writer object"""

    def test_write_fortran_line(self):
        """Test writing a fortran line"""

        fsock = StringIO.StringIO()

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
        lines.append("else")
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
      ELSE
        BAH=2
      ENDIF
      TEST\n"""

        writer = writers.FortranWriter()
        writer.writelines(fsock, lines)

        self.assertEqual(fsock.getvalue(),
                         goal_string)

    def test_write_fortran_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.FortranWriter()
        for nonstring in non_strings:
            self.assertRaises(writers.FortranWriter.FortranWriterError,
                              writer.write_line,
                              fsock, nonstring)

#===============================================================================
# CPPWriterTest
#===============================================================================
class CPPWriterTest(unittest.TestCase):
    """Test class for the C++ writer object"""

    def test_write_cplusplus_line(self):
        """Test writing a cplusplus line"""

        fsock = StringIO.StringIO()

        lines = """

#ifndef Pythia8_SigmaEW_H
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
      return "q g - > q gamma(udscb) test test test test test asasd as asd a dada djkl;sdf lkja sdfjkla;sdf l;kja sdfkldf";}
  virtual int    code()   const {return 201;}
  virtual string inFlux() const {return "qg";}

private:

  // Values stored for later use.
  double mNew, m2New, sigUS, sigma0;

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

}
    """.split("\n")

        goal_string = """#ifndef Pythia8_SigmaEW_H
#define Pythia8_SigmaEW_H
#include "PythiaComplex.h"
#include "SigmaProcess.h"
namespace Pythia8 
{
// ==========================================================================
// 
// A derived class for q g - > q gamma (q = u, d, s, c, b).
// Use massless approximation also for Q since no alternative.
class Sigma2qg2qgamma : public Sigma2Process 
{
  public:
    // Constructor.
    Sigma2qg2qgamma() {}
    // Calculate flavour - independent parts of cross section.
    virtual void sigmaKin(); 
    // Evaluate d(sigmaHat) / d(tHat). 
    virtual double sigmaHat(); 
    // Select flavour, colour and anticolour.
    virtual void setIdColAcol(); 
    // Info on the subprocess.
    virtual string name() const 
    {
      return "q g - > q gamma(udscb) test test test test test asasd as asd a dada djkl; sdf lkja sdfjkla; sdf l; kja sdfkldf"; 
    }
    virtual int code() const 
    {
      return 201; 
    }
    virtual string inFlux() const 
    {
      return "qg"; 
    }
  private:
    // Values stored for later use.
    double mNew, m2New, sigUS, sigma0; 
};
// Select identity, colour and anticolour.
void Sigma2ff2fftgmZ::setIdColAcol() 
{
  // Trivial flavours: out = in.
  setId(id1, id2, id1, id2); 
  // Colour flow topologies. Swap when antiquarks.
  if (abs(id1) < 9 && abs(id2) < 9 && id1 * id2 > 2 / 3.)
    setColAcol(1, 0, 2, 0, 1, 0, 2, 0); 
  else if (abs(id1) < 9 && abs(id2) < 9)
    setColAcol(1, 0, 0, 2, 1, 0, 0, 2); 
  else
    setColAcol(0, 0, 0, 0, 0, 0, 0, 0); 
  if ((abs(id1) != 9 && id1 < 0) || (abs(id1) == 10 && id2 < 0))
    swapColAcol(); 
}
"""

        writer = writers.CPPWriter()
        writer.writelines(fsock, lines)
        #print fsock.getvalue()

        self.assertEqual(fsock.getvalue(),
                         goal_string)

    def test_write_cplusplus_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.CPPWriter()
        for nonstring in non_strings:
            self.assertRaises(writers.CPPWriter.CPPWriterError,
                              writer.write_line,
                              fsock, nonstring)

