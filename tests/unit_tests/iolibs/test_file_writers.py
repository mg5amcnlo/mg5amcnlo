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

from __future__ import absolute_import
from io import StringIO
import re
import os

import tests.unit_tests as unittest

import madgraph.iolibs.file_writers as writers
from six.moves import zip
import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join

#===============================================================================
# FortranWriterTest
#===============================================================================
class CheckFileCreate():
    """Check that the files are correctly created"""

    output_path = '/tmp/' # work only on LINUX but that's ok for the test routine
    created_files =[]

    def assertFileContains(self, filename, solution, partial=False):
        """ Check the content of a file """

        if isinstance(filename, str):
            current_value = open(self.give_pos(filename)).read()
        else:
            current_value = filename.read()

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

        #full match expected
        if not partial:
            for a, b in zip(list_sol, list_cur):
                self.assertEqual(a.strip(),b.strip())
            #for a, b in zip(current_value.split('\n'), solution.split('\n')):
            #    self.assertEqual(a,b)
            #self.assertEqual(current_value.split('\n'), solution.split('\n'))
            self.assertEqual(len(list_sol), len(list_cur))
        else:
            #partial match expected
            old_start = 0
            len_sol = len(list_sol)
            while 1:
                start = list_cur[old_start:].index(list_sol[0])
                if start == -1:
                    raise
                old_start += start
                for a, b in zip(list_sol, list_cur[start:start+len_sol]):
                    try:
                        self.assertEqual(a,b)                
                    except Exception:
                        continue
                else:
                    break
            

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
        lines.append(" call aaaaaa(bbb, ccc, ddd, eee, fff, ggg, hhhhhhhhhhhhhhhh+asdasd, wspedfteispd)")
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
        lines.append("# this is a comment")
        lines.append('#ifdef not a comment')
        lines.append("c$$$C EXAMPLE: cut on top quark pT")

        goal_string = """      CALL AAAAAA(BBB, CCC, DDD, EEE, FFF, GGG, HHHHHHHHHHHHHHHH
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
      TEST\n
C     this is a comment
#ifdef not a comment
C     $$$C EXAMPLE: cut on top quark pT\n"""

        writer = writers.FortranWriter(self.give_pos('fortran_test')).\
                 writelines(lines)

        # Check that the output stays the same
        self.assertFileContains('fortran_test',
                                 goal_string)

    def test_write_fortran_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.FortranWriter(os.devnull)
        for nonstring in non_strings:
            self.assertRaises(AssertionError,
                              writer.write_line,
                              nonstring)

    def test_remove_routine(self):
      """test that the remove routine works
         and test that it does not complains if they are no routine to remove
      """

      input_string = """      Subroutine TO_RM(BBB, CCC, DDD, EEE, FFF, GGG, HHHHHHHHHHHHHHHH
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
      Return
      end

      Subroutine TO_KEEP(BBB)
        integer BBB
        BBB = 1
        return 
        end
    """

      writer = writers.FortranWriter(self.give_pos('fortran_test'))

      writer.remove_routine(input_string, ["TO_RM", "TO_RM2"])

      goal_string = """
      SUBROUTINE TO_KEEP(BBB)
      INTEGER BBB
      BBB = 1
      RETURN
      END"""

      # Check that the output stays the same
      self.assertFileContains('fortran_test',
                                 goal_string)
  
    def test_dummy_fcts(self):
        
      input = pjoin(MG5DIR, 'Template', 'NLO', 'SubProcesses', 'dummy_fct.f')
      text = open(input).read()

      writer = writers.FortranWriter(self.give_pos('fortran_test'))
      writer.write(text)
      
      for line in  open(self.give_pos('fortran_test')).readlines():
          if line.startswith(' '):
              self.assertFalse(line.strip().startswith(('c ','C ', 'c$', 'C$')))
      
      input = pjoin(MG5DIR, 'Template', 'LO', 'SubProcesses', 'dummy_fct.f')
      text = open(input).read()

      writer = writers.FortranWriter(self.give_pos('fortran_test'))
      writer.write(text)
      
      for line in  open(self.give_pos('fortran_test')).readlines():
          if line.startswith(' '):
              self.assertFalse(line.strip().startswith(('c ','C ','c$', 'C$')))      


      #self.assertFileContains('fortran_test', text)


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

        fsock = StringIO()

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
        
        
    def test_write_cplusplus_special_syntax(self):
        """ """
        
        line = """mapFinalStates[{5,-5,-13,11,-12,14}] =
{
{ &CPPProcess::matrix_gg_ttx_bbxmupemvexvm,
 false,
 {std::make_pair(21,21)},
 256,
 256
 }
,
{ &CPPProcess::matrix_uux_ttx_bbxmupemvexvm,
 true,
 {std::make_pair(2,-2),std::make_pair(4,-4),std::make_pair(1,-1),std::make_pair(3,-3)},
 256,
 36
 }
};"""
        
        goal_string = """mapFinalStates[{5, -5, -13, 11, -12, 14}] = 
{
  {
    &CPPProcess::matrix_gg_ttx_bbxmupemvexvm, 
    false, 
    {
      std::make_pair(21, 21)
    }, 
    256, 
    256
  }
  , 
  {
    &CPPProcess::matrix_uux_ttx_bbxmupemvexvm, 
    true, 
    {
      std::make_pair(2, -2), std::make_pair(4, -4), std::make_pair(1, -1),
          std::make_pair(3, -3)
    }, 
    256, 
    36
  }
}; """
        writer = writers.CPPWriter(self.give_pos('cpp_test')).\
                 writelines(line.split('\n'))

        # Check that the output stays the same
        self.assertFileContains('cpp_test',
                                 goal_string)   
    

    def test_write_cplusplus_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.CPPWriter(os.devnull)
        for nonstring in non_strings:
            self.assertRaises(AssertionError,
                              writer.write_line,
                              nonstring)


class FilePreprocessingTest(unittest.TestCase):
    """ Makes sure that the preprocessing tags '##' are correctly handled by the
    function preprocess_template()."""
    
    def test_single_if_statement(self):
        """ Test a single conditional statement """     
        tests = [\
("""
[...]
## if (TestVar) {
A
## } else {
B
## }
[...]
""".split('\n'),
"""
[...]
A
[...]
""".split('\n'),
{'TestVar':True}),
("""
[...]
## if (not TestVar) {
A
## } else {
B
## }
[...]
""".split('\n'),
"""
[...]
B
[...]
""".split('\n'),
{'TestVar':True}),
("""
[...]
## if (TestVar) {
A
## } else {
B
## }
[...]
""".split('\n'),
"""
[...]
B
[...]
""".split('\n'),
{'TestVar':False}),
]
        
        for tested_text, goal, context in tests:
            res = writers.FileWriter('tmp').preprocess_template(tested_text,
                                                                context=context)
            self.assertEqual(res,goal)
            
    def test_multiple_if_statement(self):
        """ Test multiple entangle conditional statement """     
        test_text = \
"""
[...]
## if (TestVar1) {
A
## if (TestVar2) {
B
## if (TestVar3) {
C
## } else {
D
## }
E
## if (TestVar4) {
F
## }
G
## }
H
## } else {
I
## }
[...]
""".split('\n')

        test_goals = \
[
("""
[...]
A
B
C
E
F
G
H
[...]
""".split('\n'),
{'TestVar1':True,'TestVar2':True,'TestVar3':True,'TestVar4':True}),
("""
[...]
A
B
C
E
G
H
[...]
""".split('\n'),
{'TestVar1':True,'TestVar2':True,'TestVar3':True,'TestVar4':False}),
("""
[...]
A
H
[...]
""".split('\n'),
{'TestVar1':True,'TestVar2':False,'TestVar3':True,'TestVar4':False}),
("""
[...]
I
[...]
""".split('\n'),
{'TestVar1':False,'TestVar2':True,'TestVar3':True,'TestVar4':True})
]
        for goal, context in test_goals:
            res = writers.FileWriter('tmp').preprocess_template(test_text,
                                                                context=context)
            self.assertEqual(res,goal)            
