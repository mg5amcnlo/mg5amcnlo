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
class MadGraph5Error(Exception):
    """Exception raised if an exception is find 
    Those Types of error will stop nicely in the cmd interface"""

class InvalidCmd(MadGraph5Error):
    """a class for the invalid syntax call"""

import os

#Look for basic file position MG5DIR and MG4DIR
MG5DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                os.path.pardir))
MG4DIR = MG5DIR
