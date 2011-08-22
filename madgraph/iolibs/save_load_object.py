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

"""Function to save any Python object to file."""

import pickle

from . import files as files

class SaveObjectError(Exception):
    """Exception raised if an error occurs in while trying to save an
    object to file."""
    pass

def save_to_file(filename, object):
    """Save any Python object to file filename"""

    if not isinstance(filename, str):
        raise SaveObjectError, "filename must be a string"

    files.write_to_file(filename, pickle_object, object)

    return True
    
def load_from_file(filename):
    """Save any Python object to file filename"""

    if not isinstance(filename, str):
        raise SaveObjectError, "filename must be a string"

    return files.read_from_file(filename, unpickle_object)
    
def pickle_object(fsock, object):
    """Helper routine to pickle an object to file socket fsock"""

    pickle.dump(object, fsock)

def unpickle_object(fsock):
    """Helper routine to pickle an object to file socket fsock"""

    return pickle.load(fsock)

