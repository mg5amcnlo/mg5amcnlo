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
class MadGraph5Error(Exception):
    """Exception raised if an exception is find 
    Those Types of error will stop nicely in the cmd interface"""

class InvalidCmd(MadGraph5Error):
    """a class for the invalid syntax call"""

class aMCatNLOError(MadGraph5Error):
    """A MC@NLO error"""

import os
import logging
import time

#Look for basic file position MG5DIR and MG4DIR
MG5DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                os.path.pardir))
if ' ' in MG5DIR:
   logging.critical('''\033[1;31mpath to MG5: "%s" contains space. 
    This is likely to create code unstability. 
    Please consider changing the path location of the code\033[0m''' % MG5DIR)
   time.sleep(1)
MG4DIR = MG5DIR
ReadWrite = os.access(MG5DIR, os.W_OK) # W_OK is for writing

