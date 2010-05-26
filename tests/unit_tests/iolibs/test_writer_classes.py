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

import madgraph.iolibs.writer_classes as writers

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
        for line in lines:
            writer.write_fortran_line(fsock, line)

        self.assertEqual(fsock.getvalue(),
                         goal_string)

    def test_write_fortran_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.FortranWriter()
        for nonstring in non_strings:
            self.assertRaises(writers.FortranWriter.FortranWriterError,
                              writer.write_fortran_line,
                              fsock, nonstring)

#===============================================================================
# CPPWriterTest
#===============================================================================
class CPPWriterTest(unittest.TestCase):
    """Test class for the C++ writer object"""

    def test_write_cplusplus_line(self):
        """Test writing a cplusplus line"""

        fsock = StringIO.StringIO()

        lines = []
        goal_string = """\n"""

        writer = writers.CPPWriter()
        for line in lines:
            writer.write_cplusplus_line(fsock, line)

        self.assertEqual(fsock.getvalue(),
                         goal_string)

    def test_write_cplusplus_error(self):
        """Test that a non-string gives an error"""

        fsock = StringIO.StringIO()

        non_strings = [1.2, ["hej"]]

        writer = writers.CPPWriter()
        for nonstring in non_strings:
            self.assertRaises(writers.CPPWriter.CPPWriterError,
                              writer.write_cplusplus_line,
                              fsock, nonstring)

