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
"""Unit test library for the export Python format routines"""
from __future__ import division

import array
import copy
import fractions
import itertools
import logging
import math
import os
import re

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writer
import models.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.process_checks as process_checks

from madgraph import MG5DIR

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.diagram_symmetry')

#===============================================================================
# find_symmetry
#===============================================================================

def find_symmetry(matrix_element):
    """Find symmetries between amplitudes by comparing the squared
    amplitudes for all permutations of identical particles. Note that
    also multi-particle vertices are included here."""

    process = matrix_element.get('processes')[0]
    base_model = process.get('model')
    full_model = model_reader.ModelReader(base_model)
    full_model.set_parameters_and_couplings()
    # Get phase space point
    p, w_rambo = process_checks.get_momenta(process, full_model)
    
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

    # Writer for the Python matrix elements
    helas_writer = helas_call_writer.PythonUFOHelasCallWriter(base_model)

    # Check matrix element value for all permutations
    stored_quantities = {}
    amp2start = []
    symmetry = []
    me_values = []
    final_states = [l.get('id') for l in process.get('legs')[ninitial:]]
    nperm = 0
    ident_perms = []
    for perm in itertools.permutations(range(ninitial, nexternal)):
        if [process.get('legs')[i].get('id') for i in perm] != final_states:
            # Non-identical particles permutated
            continue
        ident_perms.append([0,1]+list(perm))
        nperm += 1
        new_p = p[:ninitial] + [p[i] for i in perm]
        # Reset matrix_elements, otherwise won't run again
        res = process_checks.evaluate_matrix_element(\
                                              matrix_element,stored_quantities,
                                              helas_writer, full_model, new_p,
                                              reuse = True)
        if not res:
            break
        me_value, amp2 = res
        # Make a list with (8-pos value, magnitude) to easily compare
        amp2mag = []
        for a in amp2:
            if a > 0:
                amp2mag.append(int(math.floor(math.log10(abs(a)))))
            else:
                amp2mag.append(0)
        
        amp2 = [(int(a*10**(8-am)), am) for (a, am) in zip(amp2, amp2mag)]
        if not symmetry:
            # This is the first iteration - initialize lists
            # Initiate symmetry with all 1:s
            symmetry = [1 for i in range(len(amp2))]
            # Store initial amplitudes
            amp2start = amp2
            # Initialize list of permutations
            perms = [range(nexternal) for i in range(len(amp2))]
            continue
        for i, val in enumerate(amp2):
            if val == (0,0):
                # If amp2 is 0, just set symmetry to 0
                symmetry[i] = 0
                continue
            # Only compare with diagrams below this one
            if val in amp2start[:i]:
                ind = amp2start.index(val)
                # Replace if 1) this amp is unmatched (symmetry[i] > 0) or
                # 2) this amp is matched but matched to an amp larger
                # than ind
                if symmetry[ind] > 0 and \
                   (symmetry[i] > 0 or \
                    symmetry[i] < 0 and -symmetry[i] > ind + 1):
                    symmetry[i] = -(ind+1)
                    perms[i] = [0, 1] + list(perm)
                    symmetry[ind] += 1

    return (symmetry, perms, ident_perms)
