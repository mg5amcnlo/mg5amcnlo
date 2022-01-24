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
"""All models for MG5, in particular UFO models (by FeynRules)"""

from __future__ import absolute_import
import os
import sys
import madgraph.various.misc as misc
from madgraph import MG5DIR
import six
import logging

logger = logging.getLogger('madgraph.models')

pjoin = os.path.join

class UFOError(Exception): pass

def load_model(name, decay=False):
    
    # avoid final '/' in the path
    if name.endswith('/'):
        name = name[:-1]
    


    path_split = name.split(os.sep)
    if len(path_split) == 1:
        try:
            with misc.TMP_variable(sys, 'path', [pjoin(MG5DIR, 'models'), pjoin(MG5DIR, 'models', name), MG5DIR]):  
                model_pos = 'models.%s' % name
                __import__(model_pos)
            return sys.modules[model_pos]
        except Exception as error:
            pass
        if 'PYTHONPATH' in os.environ:
            for p in os.environ['PYTHONPATH'].split(':'):
                if not p:
                    continue
                new_name = os.path.join(p, name)
                try:
                    return load_model(new_name, decay)
                except Exception:
                    pass
                except ImportError:
                    pass
    elif path_split[-1] in sys.modules:
        model_path = os.path.realpath(os.sep.join(path_split))
        sys_path = os.path.realpath(os.path.dirname(sys.modules[path_split[-1]].__file__))
        if sys_path != model_path:
            raise Exception('name %s already consider as a python library cann\'t be reassigned(%s!=%s)' % \
                (path_split[-1], model_path, sys_path)) 

    # remove any link to previous model
    for name in ['particles', 'object_library', 'couplings', 'function_library', 'lorentz', 'parameters', 'vertices', 'coupling_orders', 'write_param_card',
                 'CT_couplings', 'CT_vertices', 'CT_parameters']:
        try:
            del sys.modules[name]
        except Exception:
            continue

    with misc.TMP_variable(sys, 'path', [os.sep.join(path_split[:-1]),os.sep.join(path_split)]):
        try:
            __import__(path_split[-1])
        except Exception as error:
            raise UFOError(str(error))
    output = sys.modules[path_split[-1]]
    if decay:
        dec_name = '%s.decays' % path_split[-1]
        try:
            __import__(dec_name)
        except ImportError:
            pass
        else:
            output.all_decays = sys.modules[dec_name].all_decays
        
    return sys.modules[path_split[-1]]
