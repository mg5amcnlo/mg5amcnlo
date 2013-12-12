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

"""Function to save model files."""

import logging
import os

import madgraph.iolibs.files as files
import madgraph.core.base_objects as base_objects

logger = logging.getLogger('madgraph.save_model')

def save_particles(fsock, part_list):
    """Save particle objects contained in part_list in the stream fsock"""

    if not isinstance(part_list, base_objects.ParticleList):
        raise ValueError, \
            "Object %s is not a valid ParticleList" % repr(part_list)

    fsock.write("particles = [\n")

    for part in part_list:
        if part_list.index(part) != len(part_list) - 1:
            fsock.write(str(part) + ',')
        else:
            fsock.write(str(part))

    fsock.write("]")

def save_interactions(fsock, inter_list):
    """Save interaction objects contained in inter_list in the stream fsock"""

    if not isinstance(inter_list, base_objects.InteractionList):
        raise ValueError, \
            "Object %s is not a valid InteractionList" % repr(inter_list)

    fsock.write("interactions = [\n")

    for inter in inter_list:
        if inter_list.index(inter) != len(inter_list) - 1:
            fsock.write(str(inter) + ',')
        else:
            fsock.write(str(inter))

    fsock.write("]")

def save_model(path, model):
    """Save a full model in directory path (try to create if necessary)."""

    if not isinstance(model, base_objects.Model):
        raise ValueError, \
            "Object %s is not a valid Model" % repr(model)

    if not isinstance(path, str):
        raise ValueError, \
            "Object %s is not a path string" % repr(path)

    if not os.path.isdir(path):
        logger.warning("Path %s does not exist, try to make it..." % str(path))
        try:
            os.mkdir(path)
        except IOError, (errno, strerror):
            logger.error("I/O error (%s): %s" % (errno, strerror))
            return None

    print "Saving particles...",
    files.write_to_file(os.path.join(path, 'particles.py'),
                        save_particles,
                        model['particles'])
    print "%i particles saved to %s" % (len(model['particles']),
                                      os.path.join(path, 'particles.py'))

    print "Saving interactions...",
    files.write_to_file(os.path.join(path, 'interactions.py'),
                        save_interactions,
                        model['interactions'])
    print "%i interactions saved to %s" % (len(model['interactions']),
                                      os.path.join(path, 'interactions.py'))
