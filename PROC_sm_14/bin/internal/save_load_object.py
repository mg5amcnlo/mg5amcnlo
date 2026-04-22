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

"""Function to save any Python object to file."""

from __future__ import absolute_import
import pickle
import six.moves.cPickle

from . import files as files
import six
import os

class SaveObjectError(Exception):
    """Exception raised if an error occurs in while trying to save an
    object to file."""
    pass

def save_to_file(filename, object, log=True, allow_fail=True):
    """Save any Python object to file filename"""

    if not isinstance(filename, six.string_types):
        raise SaveObjectError("filename must be a string")

    files.write_to_file(filename, pickle_object, object, log=log, binary=True,
                        bypass_error=True)

    return True
    
def load_from_file(filename,binary=True):
    """Save any Python object to file filename"""

    if not isinstance(filename, str):
        raise SaveObjectError("filename must be a string")
    return files.read_from_file(filename, unpickle_object, binary=binary)
    
def pickle_object(fsock, object, bypass_error=False, **opts):
    """Helper routine to pickle an object to file socket fsock"""

    try:
        six.moves.cPickle.dump(object, fsock, protocol=2)
    except Exception as error:
        if bypass_error:
            return
        else:
            raise 
        
        
class UnPickler(pickle.Unpickler):
    """Treat problem of librarie"""
    
    def __init__(self, *args, **opts):
        pickle.Unpickler.__init__(self, *args, **opts)
        self.basemod = os.path.dirname(args[0].name)
    
    def find_class(self, module, name):
        """Find the correct path for the given function.
           Due to ME call via MG some libraries might be messed up on the pickle
           This routine helps to find back which one we need. 
        """

        # A bit of an ugly hack, but it works and has no side effect.
        if module == 'loop_me_comparator':
            module = 'tests.parallel_tests.loop_me_comparator'
        import sys
        try:
            import madgraph.various.misc as misc
        except ImportError:
            import internal.misc as misc
            
        with misc.TMP_variable(sys, 'path', sys.path + [self.basemod]):
            try:
                return pickle.Unpickler.find_class(self, module, name)
            except ImportError as error:
                pass
        
        lerror = None
        for prefix in ['internal.%s', 'madgraph.iolibs.%s', 'madgraph.madevent.%s',
                       'madgraph.various.%s', 'internal.ufomodel.%s']:
        
            if '.' in module:
                newmodule = prefix % module.rsplit('.',1)[1]
            else:
                newmodule = prefix % module
        
            try:
                return pickle.Unpickler.find_class(self, newmodule , name)
            except Exception as error:
                lerror = error
                pass
        
        else:
            raise lerror
    

def unpickle_object(fsock):
    """Helper routine to pickle an object to file socket fsock"""

    p = UnPickler(fsock)
    return p.load()
    #return pickle.load(fsock)

