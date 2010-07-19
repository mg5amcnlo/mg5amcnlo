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


logger = logging.getLogger('madgraph.files')

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
        logger.error("I/O error (%s): %s" % (errno, strerror))
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
        logger.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value

#===============================================================================
# append_to_file
#===============================================================================
def append_to_file(filename, myfunct, *args):
    """Open a file for appending, apply the function myfunct (with
    sock as an arg) on its content and return the result. Deals
    properly with errors and returns None if something goes wrong.
    """

    try:
        sock = open(filename, 'a')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        logger.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value

#===============================================================================
# check piclke validity
#===============================================================================
def is_update(picklefile, path_list=None, min_time=1279550579):
    """Check if the pickle files is uptodate compare to a list of files. 
    If no files are given, the pickle files is checked against it\' current 
    directory"""
    
    if not os.path.exists(picklefile):
        return False
    
    if not path_list:
        dirpath = os.path.dirname(picklefile)
        path_list = [ os.path.join(dirpath, file) for file in \
                                                            os.listdir(dirpath)]
    
    assert type(path_list) == list, 'is_update expect a list of files'
      
    pickle_date = os.path.getctime(picklefile)
    if pickle_date < min_time:
        return False
    
    for path in path_list:
        if os.path.getmtime(path) > pickle_date:
            return False
    #all pass
    return True
    
