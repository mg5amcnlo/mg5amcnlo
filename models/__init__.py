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
"""All models for MG5, in particular UFO models (by FeynRules)"""

import os
import sys

def load_model(name):
    
    # avoid final '/' in the path
    if name.endswith('/'):
        name = name[:-1]
    
    path_split = name.split(os.sep)
    if len(path_split) == 1:
        model_pos = 'models.%s' % name
        __import__(model_pos)
        return sys.modules[model_pos]
    else:
        sys.path.insert(0, os.sep.join(path_split[:-1]))
        __import__(path_split[-1])
        sys.path.pop(0)
        return sys.modules[path_split[-1]]
