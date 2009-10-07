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

"""Methods and classes dealing with file access."""

import logging
import os

#===============================================================================
# act_on_file
#===============================================================================
def act_on_file(filename, myfunct, open_mode='r'):
    """Open a file, apply the function myfunct (with sock as an arg) 
    on its content and return the result. Deals properly with errors and
    returns None if something goes wrong. Open mode is r by default but 
    can be changed.
    """

    try:
        sock = open(filename, open_mode)
        try:
            ret_value = myfunct(sock)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        logging.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value
