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

#===============================================================================
# FeynmanDiagramLoop
#===============================================================================
class LoopFeynmanDiagram(drawing.FeynmanDiagram):
    """Object to compute the position of the different Vertex and Line associate
    to a diagram object with a presence of a Loop.
    
    This is the standard way to doing it [main]
    1) Creates the new structure needed for the diagram generation [load_diagram]
        This defines self.vertexList and self.lineList which are the list of     
        respectively all the vertex and all the line include in the diagram.
        Each line is associated to two vertex, so we have added new vertex
        compare to the diagram object (base_objects.Diagram). The two vertex are 
        named begin/end and represent the line direction. at this stage all line
        are going timelike. T-channel are going from particle 1 to particle 2
    2) Associate to each vertex a level. [define_level]
        This level is define has the number of non T-channel 
        particles needed to connect this particles to a initial state starting
        point. 
        The Loop is dispatched on only two channel. If some T-channel 
        started between the initial particles those are going in negative 
        directions (i.e. to negative level)
        
    3) Compute the position of each vertex [find_initial_vertex_position]
        The x-coordinate will proportional to the level. The most negative vertex
        will have x=0 coordinate (vertex associate with initial state particle)
        The vertex with the highest level value should be at x=1.
        
        If an external particles cann't be place at the border at the current 
        level. we will try to place it one level later, potentially up to last
        level. A option can force to place all external particles at x=1.
        
        the y-coordinate are chosen such that 
            - external particles try to have (y=0 or y=1) coordinates
                (if not move those vertex to next level)
            - other particles maximizes distances between themselves.
    4) Solve Fermion-flow and (anti)particle type [self.solve_line_direction]
        the way to solve the fermion-flow is basic and fail in general for
        majorana fermion. The basic idea is "particles are going timelike".
        This is sufficient in all cases but T-channel particles and Loop particles
        which are solve separately."""

    def __init__(self, diagram, fdstructures, model, opt=None):
        """Store the information concerning this diagram. This routines didn't
        perform any action at all.
        diagram: The diagram object to draw
        model: The model associate to the diagram
        opt: A DrawingOpt instance with all options for drawing the diagram.
        fdstructures:  list of  structure that might be connected to the loop.
        """
        
        # call the mother initialization
        super(LoopFeynmanDiagram, self).__init__(diagram, model, opt)
        self.fdstructures = fdstructures


    def load_diagram(self, contract=True):
        """Define all the object for the Feynman Diagram Drawing (Vertex and 
        Line) following the data include in 'self.diagram'
        'contract' defines if we contract to one point the non propagating line.
        Compare to usual load we glue the cutted propagator of the Loop.
        """ 
                 
        for pdg, list_struct_id, vertex_id in self.diagram['tag']:
            for structure_id in list_struct_id:
                for vertex in self.fdstructures[structure_id]['vertices']:
                     self.load_vertex(vertex)                        
        super(LoopFeynmanDiagram, self).load_diagram(contract)
        
        # select the lines present in the loop
        loop_line = [line for line in self.lineList if line.loop_line]
        
        # Fuse the cutted particles (the first and the last of the list)
        self.fuse_line(loop_line[0], loop_line[-1])

    def find_vertex_at_level(self, previous_level, level):
        """Returns a list of vertex such that all those vertex are one level 
        after the level of vertexlist and sorted in such way that the list 
        start with vertex connected with the first vertex of 'vertexlist' then 
        those connected to the second and so on."""
        started_loop = False

        vertex_at_level = []
        for vertex in previous_level:
            if  vertex.is_external() and  vertex.pos_y not in [0, 1]:
                # Move external vertex from one level to avoid external 
                #particles finishing inside the square. 
                vertex.def_level(vertex.level + 1)
                vertex_at_level.append(vertex)
                continue

            for line in vertex.lines:
                if line.begin is vertex and line.end.level == level:
                    if not line.loop_line:
                        vertex_at_level.append(line.end)
                    elif started_loop:
                        continue
                    else:
                        started_loop = True
                        vertex_at_level += self.find_all_loop_vertex(line.end)
                                        
        return vertex_at_level

   
        
    def find_vertex_position_at_level(self, vertexlist, level, direction=1):
        """Finds the vertex position for the particle at 'level' given the 
        ordering at previous level given by the vertexlist. 
        if direction !=0  pass in auto-recursive mode."""

        if level == 2:
            self.find_vertex_position_at_level(vertexlist, 0, -1)
        
        super(LoopFeynmanDiagram, self).find_vertex_position_at_level( \
                                                   vertexlist, level, direction)
 
    def find_all_loop_vertex(self, init_loop):
        """ Returns all the vertex associate at a given level. returns in a 
        logical ordinate way starting at init_loop """
        
        solution = []
        while init_loop:
            solution.append(init_loop)
            init_loop = self.find_next_loop_channel_vertex(init_loop, solution)
        return solution
 
    def find_next_loop_channel_vertex(self, loop_vertex, forbiden=[]):
        """Returns the next loop_vertex. i.e. the vertex following loop_vertex.
        """

        level = loop_vertex.level
        for line in loop_vertex.lines:
            if line.loop_line == False:
                continue
            
            if line.end is loop_vertex:
                if line.begin.level == level and line.begin not in forbiden: 
                    return line.begin
            else:
                assert line.begin is loop_vertex
                if line.end.level == level and line.end not in forbiden: 
                    return line.end              
            
    def fuse_line(self, line1, line2):
        """ make two lines to fuse in a single one. The final line will connect
        the two begin."""
        
        # remove line2 from lineList
        self.lineList.remove(line2)
        self.vertexList.remove(line1.end)
        self.vertexList.remove(line2.end)
        line2.begin.lines.remove(line2)
        
        # connect the line
        line1.def_end_point(line2.begin)
        
    def define_level(self):
        """ define level in a recursive way """
     
        #check what side of loop should be put on right side
        if self.need_to_flip():
            self.loop_flip()
        
        #add special attribute
        self.start_level_loop = None
                
        super(LoopFeynmanDiagram, self).define_level()
    
    def need_to_flip(self):
        """check if the T-channel of a loop diagram need to be flipped.
            This move from left to right the external particles linked to the 
            loop. 
        """

        left_side = 0
        right_side = 0
        side_weight = 0 # if side is positive need to switch
        
        binding_side = {}
        # Count the number of T-channel propagator
        for vertex in self.diagram.get('vertices'):
            nb_Tloop = len([line for line in vertex.get('legs') if line.get('loop_line') 
                            and line.get('state')])
            
            line = vertex['legs'][-1]
            if nb_Tloop % 2:
                continue
            if line.get('state'):
                right_side += 1
                left_direction = False
            else:
                left_side += 1
                left_direction = True

                
            for line in vertex['legs'][:-1]:
                if binding_side.has_key(line.get('number')):
                    pass
                binding_side[line.get('number')] = left_direction
        
        # See the depth of each side 
        for pdg, list_struct_id, vertex_id in self.diagram['tag']:
            for structure_id in list_struct_id:
                leg = self.fdstructures[structure_id].get('binding_leg')
                if leg.get('number') < 2:
                    pass # connecting to initial particles
                #compute the number of vertex in the structure
                nb_vertex = len(self.fdstructures[structure_id].get('vertices'))
                if not binding_side.has_key(leg.get('number')):
                    continue
                    
                if  binding_side[leg.get('number')]:
                    side_weight += nb_vertex **2
                else:
                    side_weight -= nb_vertex **2
        
        if side_weight == 0:
            return left_side > right_side
        elif right_side == left_side == 1:
            self.remove_T_channel()
            return False
        else:
            return side_weight > 0
    
    def loop_flip(self):
        """ switch t-channel information for the particle in the loop """
        
        for vertex in self.diagram.get('vertices'):
            leg = vertex['legs'][-1]
            leg.set('state', not leg.get('state'))
        
        for line in self.lineList:
            if not line.is_external() and line.loop_line:
                line.state = not line.state
 
    
    def remove_T_channel(self):
        """Remove T-channel information"""
        for vertex in self.diagram.get('vertices'):
            legs = vertex['legs'][-1]
            legs.set('state', True)
        
        for line in self.lineList:
            if not line.is_external() and line.loop_line:
                line.state = True
        

        
        
        
    def def_next_level_from(self, vertex, direction=1):
        """Define level for adjacent vertex.
        If those vertex is already defined do nothing
        Otherwise define as level+1 (at level 1 if T-channel)
        
        Special case for loop: 
        1) Loop are on two level max. so this saturates the level 
        2) If a branch starts from a Loop T-channel pass in negative number
           This is set by direction
        3) Treat T-channel first to avoid over-saturation of level 2
        This routine defines also self.max_level and self.min_level
        
        This routine is foreseen for an auto-recursive mode. So as soon as a 
        vertex have his level defined. We launch this routine for this vertex.
        """
         
        level = vertex.level
        if direction == -1:     
            nb_Tloop = len([line for line in vertex.lines if line.loop_line and \
                                                                   line.state])
            if nb_Tloop % 2:
                direction = 1
        
        def order(line1, line2):
            """ put T-channel first """
            if line1.state == line2.state:
                return 0
            if line2.state:
                return -1
            else:
                return 1
        
        vertex.lines.sort(order)
        for line in vertex.lines:
            if line.begin.level and line.end.level:
                continue # everything correctly define
            elif line.end is vertex:
                if line.loop_line and not line.state:
                    line.inverse_begin_end()
                    next = line.end
                else:
                    continue
            else:
                next = line.end       
            
            # Check if T-channel or not. Note that T-channel tag is wrongly 
            #define if only one particle in initial state.
            if line.state == False:
                # This is T vertex. => level is 1
                next.def_level(1)
                if line.loop_line:
                    direction = -1
                    nb_Tloop = len([l for l in vertex.lines 
                                    if l.loop_line and l.state])
                    if nb_Tloop % 2:
                        direction = 1
                    
            elif line.loop_line:
                direction = 1
                if self.start_level_loop is None:
                    next.def_level(level + 1)
                    self.start_level_loop = level
                    
                else:
                     next.def_level(self.start_level_loop + 1)
            else:
                # Define level
                next.def_level(level + direction)
                # Check for update in self.max_level
                self.max_level = max(self.max_level, level + direction)
                self.min_level = min(self.min_level, level + direction)
            # Launch the recursion
            self.def_next_level_from(next, direction)