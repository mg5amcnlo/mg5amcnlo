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

#===============================================================================
# read_from_file
#===============================================================================
def read_from_file(filename, myfunct, *args):
    """Open a file, apply the function myfunct (with sock as an arg) 
    on its content and return the result. Deals properly with errors and
    returns None if something goes wrong. 
    """

    try:
        sock = open(filename, 'r')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        logging.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value

#===============================================================================
# write_to_file
#===============================================================================
def write_to_file(filename, myfunct, *args):
    """Open a file for writing, apply the function myfunct (with sock as an arg) 
    on its content and return the result. Deals properly with errors and
    returns None if something goes wrong. 
    """

    try:
        sock = open(filename, 'w')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        logging.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value
