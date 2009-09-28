##############################################################################
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
##############################################################################

"""A set of functions performing routine administrative I/O tasks."""

import madgraph
import os

def get_version():
    """Returns the current version of the MadGraph package, as written in
    the VERSION text file. If the file cannot be found, UNKNOWN is 
    returned"""

    version = "UNKNOWN"

    try:
        version_file = fopen(os.path.join(madgraph.__path__, "VERSION"), 'r')
        try:
            version = version_file.read()
        finally:
            version_file.close()
    except IOError:
        pass

    return version

