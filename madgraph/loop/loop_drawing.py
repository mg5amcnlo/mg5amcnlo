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
"""All the routines to choose the position to each vertex and the 
direction for particles. All those class are not related to any output class.

This file contains 4 class:
    * FeynmanLine which extend the Leg with positioning information
    * VertexPoint which extend the vertex with position and line information.
        Coordinates belongs to [0,1] interval
    * FeynmanDiagram which
            1) Extends a diagram to have position information - load_diagram 
            2) Is able to structure the vertex in level - define_level 
                level are the number of s_channel line-initial particles
                separating the vertex from the initial particles starting point.
            3) Attributes position to each vertex - find_initial_vertex_position
    * FeynmanDiagramHorizontal
        is a child of FeynmanDiagram which assign position in a different way.

    The x-coordinate will proportional to the level, both in FeynmanDiagram and 
        in FeynmanDiagramHorizontal
    
    In FeynmanDiagram, the y-coordinate of external particles are put (if 
        possible and if option authorizes) to 0,1. all other y-coordinate are 
        assign such that the distance between two neighbor of the same level    
        are always the same. 
    
    In FeynmanDiagramHorizontal, an additional rules apply: if only one 
        S-channel is going from level to the next, then this S-channel should be
        horizontal."""

import madgraph.core.drawing as drawing

