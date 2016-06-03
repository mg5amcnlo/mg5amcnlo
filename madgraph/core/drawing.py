################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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

from __future__ import division

import math

import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_objects

#===============================================================================
# FeynmanLine
#===============================================================================
class FeynmanLine(object):
    """All the information about a line in a Feynman diagram
    i.e. begin-end/type/tag."""

    class FeynmanLineError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a Feynam_line."""

    def __init__(self, init_dict={}):
        """Initialize the FeynmanLine content."""

        # Add this attribute by default to have some tests with a hard-coded
        # dictionary for initialization passing.
        self.loop_line = False
        for key, value in init_dict.items():
            setattr(self, key, value)
        self.begin = 0
        self.end = 0

#    def is_valid_prop(self, name):
#        """Check if a given property name is valid."""
#
#        if name == 'pid':
#            return True
#        else:
#            return super(FeynmanLine, self).is_valid_prop(name)

    def def_model(self, model):
        """ 
        make a link between the  present object and the associate model 
        """
        
        assert isinstance(model, base_objects.Model), ' try to assign a non model obect'

        self.model = model

    def def_begin_point(self, vertex):
        """-Re-Define the starting point of the line."""

        assert isinstance(vertex, VertexPoint), 'The begin point should be a ' + \
                 'Vertex_Point object'

        self.begin = vertex
        vertex.add_line(self)
        return

    def def_end_point(self, vertex):
        """-Re-Define the starting point of the line. with check"""

        assert isinstance(vertex, VertexPoint), 'The end point should be a ' + \
                 'Vertex_Point object'

        self.end = vertex
        vertex.add_line(self)
        return

    def add_vertex(self, vertex):
        """Associate the vertex to the line at the correct position.
        line.begin should be closer of the lower right corner than line.end.
        
        This is achieved in the following way:
        * We don't care about external particles.Those one will be perform
         easily in a second step. In the mean time we apply this method anyway.
         Internal particles are created from a combination of particles.
        * S-channel either are create from number [X,Y,Z are strictly bigger 
            than two and A,B,C are strictly bigger than one).
                         (1 A [X Y]> 1) =>forward
                         (X Y [Z]> X) => backward 
        * T-channel are also produce either by 
                        (1 X> 1) =>forward
                        (2 X >2) => backward        
        So the common rule is to check if the number is one or not.
        """
        
        # Look if we already resolve this problem
        if self.begin:
            self.def_end_point(vertex)
        elif self.end:
            self.def_begin_point(vertex)
        #If not: resolve it. Treat external particle separately
        else:
            number = self.number
            if number == 1:
                self.def_begin_point(vertex)
            else:
                self.def_end_point(vertex)

    def define_line_orientation(self):
        """Define the line orientation. Use the following rules:
        Particles move timelike when anti-particles move anti-timelike.
        """

        if (self.id < 0):
            self.inverse_begin_end()

    def inverse_pid_for_type(self, inversetype='straight'):
        """Change the particle in his anti-particle if this type is 
        equal to 'inversetype'."""

        drawtype = self.get_info('line')
        if drawtype == inversetype:
            self.inverse_part_antipart()

    def inverse_part_antipart(self):
        """Pass particle into an anti-particle. This is needed for initial state
        particles (usually wrongly defined) and for some fermion flow resolution
        problem."""

        self.id = -1 * self.id

    def inverse_begin_end(self):
        """Invert the orientation of the line. This is needed to have correct 
        fermion flow."""

        self.begin, self.end = self.end, self.begin

    def get_info(self, name):
        """Return the model information 'name'  associated to the line."""

        pid = abs(self.id)
        return self.model.get_particle(pid).get(name)


    def get_name(self, name='name'):
        """Return the name associate to the particle."""

        pid = self.id
        model_info = self.model.get_particle(pid)
        
        if pid > 0:
            return model_info.get(name)
        elif model_info:
            return model_info.get('anti' + name)
        else:
            # particle is self anti particle
            return self.model.get_particle(-1 * pid).get(name)
            
    def get_length(self):
        """ return the length of the line """
        
        return math.sqrt((self.end.pos_x - self.begin.pos_x) ** 2 + \
                         (self.end.pos_y - self.begin.pos_y) ** 2)
        
        
    def is_fermion(self):
        """Returns True if the particle is a fermion."""

        model_info = self.model.get_particle(abs(self.id))
        if model_info.get('spin') % 2 == 0:
            return True

    def is_external(self):
        """Check if this line represent an external particles or not."""

        return self.end.is_external() or self.begin.is_external()

    def __eq__(self, other):
        """Define that two line are equal when they have the same pointer"""

        return self is other

    def __ne__(self, other):
        """Define that two line are different when they have different 
        pointer."""

        return self is not other


    # Debugging Routines linked to FeynmanLine ---------------------------------

    def has_intersection(self, line):
        """Check if the two line intersects and returns status. A common vertex 
        is not consider as an intersection.
        This routine first check input validity. 
        
        At current status this is use for test/debugging only."""

        assert self.check_position_exist()
        assert line.check_position_exist()

        return self._has_intersection(line)

    def _has_intersection(self, line):
        """Check if the two line intersects and returns status. A common vertex 
        is not consider as an intersection.
        
        At current status this is only use for test/debugging only."""

        #tolerance for equality of two number
        sys_error = 1e-7

        # Find the x-range where both line are defined  
        min, max = self._domain_intersection(line)

        # No x-value where both line are defined => No possible intersection
        if min == None:
            return False

        # Only one x value is common for both line
        if min == max :
            # Check if self is normal line (not vertical)
            if abs(self.begin.pos_x - self.end.pos_x) > sys_error:
                # Check if line is normal line (not vertical)
                if abs(line.begin.pos_x - line.end.pos_x) > sys_error:
                    # No vertical line => one vertex in common
                    return False
                # line is vertical but not self:
                return self._intersection_with_vertical_line(line)

            # Check if line is not vertical)    
            elif (abs(line.begin.pos_x - line.end.pos_x) > sys_error):
                # self is vertical but not line
                return line._intersection_with_vertical_line(self)

            # both vertical case
            else:
                # Find the y-range where both line are defined
                min, max = self._domain_intersection(line, 'y')
                if min == None or min == max:
                    return False
                else:
                    return True

        # No vertical line -> resolve angular coefficient
        xS0 = self.begin.pos_x
        yS0 = self.begin.pos_y
        xS1 = self.end.pos_x
        yS1 = self.end.pos_y

        xL0 = line.begin.pos_x
        yL0 = line.begin.pos_y
        xL1 = line.end.pos_x
        yL1 = line.end.pos_y

        coef1 = (yS1 - yS0) / (xS1 - xS0)
        coef2 = (yL1 - yL0) / (xL1 - xL0)

        # Check if the line are parallel
        if abs(coef1 - coef2) < sys_error:
            # Check if one point in common in the domain
            if abs(line._has_ordinate(min) - self._has_ordinate(min)) < \
                                                                      sys_error:
                return True
            else:
                return False

        # Intersecting line -> find point of intersection (commonX, commonY)
        commonX = (yS0 - yL0 - coef1 * xS0 + coef2 * xL0) / (coef2 - coef1)

        #check if the intersection is in the x-domain
        if (commonX >= min) == (commonX >= max):
            return False

        commonY = self._has_ordinate(commonX)

        #check if intersection is a common vertex
        if self.is_end_point(commonX, commonY):
            if line.is_end_point(commonX, commonY):
                return False
            else:
                return True
        else:
            return True

    def is_end_point(self, x, y):
        """Check if 'x','y' are one of the end point coordinates of the line.
        
        At current status this is use for test/debugging only."""

        #authorize error machine
        gap = 1e-9

        if abs(x - self.begin.pos_x) < gap and abs(y - self.begin.pos_y) < gap:
            return True
        elif abs(x - self.end.pos_x) < gap and abs(y - self.end.pos_y) < gap:
            return True
        else:
            return False

    def domain_intersection(self, line, axis='x'):
        """Returns x1,x2 where both line and self are defined. 
        Returns None, None if this domain is empty.
        This routine contains self consistency check
        
        At current status this is use for test/debugging only."""

        assert isinstance(line, FeynmanLine), ' domain intersection are between ' + \
                'Feynman_line object only and not {0} object'.format(type(line))

        # Check consistency
        self.check_position_exist()
        line.check_position_exist()

        # Launch routine
        return self._domain_intersection(line, axis)

    def _domain_intersection(self, line, axis='x'):
        """Returns x1,x2 where both line and self are defined. 
        Returns None, None if this domain is empty.
        This routine doesn't contain self consistency check.
        
        At current status this is use for debugging only."""

        #find domain for each line
        min_self, max_self = self.border_on_axis(axis)
        min_line, max_line = line.border_on_axis(axis)

        #find intersection
        start = max(min_self, min_line)
        end = min(max_self, max_line)
        if start <= end:
            return start, end
        else:
            return None, None

    def border_on_axis(self, axis='x'):
        """ Returns the two value of the domain interval for the given axis.
        
        At current status this is use for test/debugging only."""

        data = [getattr(self.begin, 'pos_' + axis), \
                                            getattr(self.end, 'pos_' + axis)]
        data.sort()
        return data

    def _intersection_with_vertical_line(self, line):
        """Checks if line intersect self. Line SHOULD be a vertical line and 
        self COULDN'T. No test are done to check those conditions.
        
        At current status this is use for test/debugging only."""

        # Find the y coordinate for the x-value corresponding to line x-position
        y_self = self._has_ordinate(line.begin.pos_x)

        # Find the y range for line. This is done in order to check that the 
        #intersection point is not a common vertex
        ymin, ymax = line.border_on_axis('y')

        # Search intersection status
        if (ymin == y_self or ymax == y_self):
            if self.is_end_point(line.begin.pos_x, y_self):
                return False
            else:
                return True
        elif (y_self > ymin) and (y_self < ymax):
            return True
        else:
            return False

    def check_position_exist(self):
        """Check that the begin-end position are defined.
        
        At current status this is use for debugging only."""

        try:
            self.begin.pos_x
            self.end.pos_y
        except Exception:
            raise self.FeynmanLineError, 'No vertex in begin-end position ' + \
                        ' or no position attach at one of those vertex '
        return True

    def has_ordinate(self, x):
        """Returns the y associate to the x value in the line
        Raises FeynmanLineError if point outside interval or result not unique.
        This routines contains check consistency.
        
        At current status this is use for debugging only."""

        if __debug__:
            self.check_position_exist()
            min = self.begin.pos_x
            max = self.end.pos_x
            if max < min:
                min, max = max, min

            if min == max:
                raise self.FeynmanLineError, 'Vertical line: no unique solution'
            if(not(min <= x <= max)):
                raise self.FeynmanLineError, 'point outside interval invalid ' + \
                    'invalid order {0:3}<={1:3}<={2:3}'.format(min, x, max)

        return self._has_ordinate(x)

    def _has_ordinate(self, x):
        """Returns the y associate to the x value in the line
        This routines doesn't contain check consistency.
        
        At current status this is use for debugging only."""

        #calculate the angular coefficient
        x_0 = self.begin.pos_x
        y_0 = self.begin.pos_y
        x_1 = self.end.pos_x
        y_1 = self.end.pos_y

        alpha = (y_1 - y_0) / (x_1 - x_0) #x1 always diff of x0


        ordinate_fct = lambda X: y_0 + alpha * (X - x_0)
        return ordinate_fct(x)



#===============================================================================
# VertexPoint
#===============================================================================
class VertexPoint(object):
    """Extension of the class Vertex in order to store the information 
    linked to the display of a FeynmanDiagram, as position
    """

    class VertexPointError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a VertexPoint."""


    def __init__(self, vertex):
        """Update a vertex to a VertexPoint with additional information about
        positioning and link with other vertex/line of the diagram."""

        # Check the validity of the parameter should be Vertex
        assert(isinstance(vertex, base_objects.Vertex))

        # Copy data and add new entry                    
        for key, value in vertex.items():
            setattr(self, key, value)
        self.lines = []
        self.level = None
        self.pos_x = 0
        self.pos_y = 0

    def def_position(self, x, y):
        """-Re-Define the position of the vertex in a square [0, 1]^2"""

        # Coordinate should b
        assert  0 <= x <= 1 and 0 <= y <= 1 ,  'vertex coordinate should be' + \
                    ' in  0,1 interval introduce value ({0},{1})'.format(x, y)

        self.pos_x = x
        self.pos_y = y
        return

    def fuse_vertex(self, vertex, common_line=''):
        """Import the line of the second vertex in the first one
            this means 
            A) change the 'line' of this vertex
            B) change the start-end position of line to point on this vertex
            C) remove common_line (if defined)."""

        for line in vertex.lines:
            # Remove common line. They are shrink to a single point
            if line is common_line:
                self.lines.remove(line)
                continue

            # Re-define the begin-end vertex of the line to point on this vertex
            #and not on the old one. self.lines is automatically updated.
            if line.begin is vertex:
                line.def_begin_point(self)
            else:
                line.def_end_point(self)
        return


    def add_line(self, line):
        """Add the line in the list keeping line connected to this vertex :
        self.lines. This routine avoid duplication of entry."""

        assert isinstance(line, FeynmanLine), \
                           'Trying to add in a Vertex a non FeynmanLine Object'

        for oldline in self.lines:
            if oldline is line:
                return

        self.lines.append(line)

    def remove_line(self, line_to_del):
        """Remove the line from the lineList. Didn't touch to vertex associate
        to begin/end point. This happens only if we fuse two vertex together. 
        (Then the line will be completely drop out, such that we dont't care
        about those vertex point."""

        assert isinstance(line_to_del, FeynmanLine), \
                   'trying to remove in a Vertex_Point a non FeynmanLine Object'

        # Find the first item in the list and remove it. note that we cann't use
        #standard delete as remove because it's use '==' and not 'is'. 
        for i, line in enumerate(self.lines):
            if line is line_to_del:
                del self.lines[i]
                return # only one data to remove!

        raise self.VertexPointError, 'trying to remove in a ' + \
                            'Vertex_Point a non present Feynman_Line'


    def def_level(self, level):
        """Define the Vertex level at 'level'. The level represents the 
        distance between the initial vertex and the current vertex. This
        distance is define has the number of non T-channel particles needed to 
        connect this particle to initial states starting point."""

        assert isinstance(level, int), 'Trying to attribute non integer level'

        self.level = level

    def is_external(self):
        """Check if this vertex is  external , i.e is related to a single 
        (external) particles."""

        #the termination has only one line.
        if len(self.lines) == 1:
            return True
        else:
            return False

    def has_the_same_line_content(self, other):
        """Check if the line associate to the two vertex are equivalent. 
        This means that they have the same number of particles with the same pid
        and that the external particles have the same number.
        
        This is a backup function, this is not use for the moment."""

        # Check the number of line
        if len(self.lines) != len(other.lines):
            return False

        # Store the information of the pid content of the vertex other
        other_line_pid = [line.id for line in other.lines]
        # Sort the information of the number content for external particle
        other_line_number = [line.number for line in other.lines if \
                                                            line.is_external()]
        
        # Now look at the self vertex and destroy the information store in the
        #two above variable. If an error raise, this means that the content is 
        #not the same. 
        for s_line in self.lines:
            try:
                other_line_pid.remove(s_line.id)
            except ValueError:
                return False
            if s_line.is_external():
                try:
                    other_line_number.remove(s_line.number)
                except ValueError:
                    return False
        
        # The lines have all their equivalent, so returns True 
        return True

    def get_uid(self):
        """Provide a unique id for the vertex"""

        tag = 0
        for i, line in enumerate(self.lines):
            tag += line.number / 10 ** (-i)
        tag = tag * 10 ** (len(self.lines) + 1)
        return tag

    def __eq__(self, other):
        """Define equality with pointeur equality."""

        return self is other

#===============================================================================
# FeynmanDiagram
#===============================================================================
class FeynmanDiagram(object):
    """Object to compute the position of the different Vertex and Line associate
    to a diagram object.
    
    This is the standard way to doing it [main]
    1) Creates the new structure needed for the diagram generation [load_diagram]
        This defines self.vertexList and self.lineList which are the list of     
        respectively all the vertex and all the line include in the diagram.
        Each line is associated to two vertex, so we have added new vertex
        compare to the diagram object (base_objects.Diagram). The two vertex are 
        named begin/end and represent the line direction. at this stage all line
        are going timelike. T-channel are going from particle 1 to particle 2
    2) Associate to each vertex a level. [define_level]
        The level represents the distance between the initial vertex and the 
        current vertex. This distance is define has the number of non T-channel 
        particles needed to connect this particles to a initial state starting
        point.
    3) Compute the position of each vertex [find_initial_vertex_position]
        The x-coordinate will proportional to the level. The vertex at level=0.
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
        This is sufficient in all cases but T-channel particles which are solve 
        separately."""

    class FeynamDiagramError(Exception):
        """Class for internal error."""

    def __init__(self, diagram, model, amplitude=False, opt=None):
        """Store the information concerning this diagram. This routines didn't
        perform any action at all.
        diagram: The diagram object to draw
        model: The model associate to the diagram
        amplitude: tell if the diagram has already fixed the I/O state of the fermion
        opt: A DrawingOpt instance with all options for drawing the diagram."""

        # Check if input are what we are expecting 
        assert isinstance(diagram, base_objects.Diagram), \
                           'first argument should derivate from Diagram object'
        assert isinstance(model, base_objects.Model), \
                            'second argument should derivate from Model object, get %s' % type(model)
        
       
        self.diagram = diagram
        self.model = model
        self.amplitude = amplitude
        
        if opt is None:
            self.opt = DrawOption()
        else:
            assert isinstance(opt, DrawOption), 'third argument should derivates' + \
                                          ' from DrawOption object'
            self.opt = opt

        # Initialize other value to void.
        self.vertexList = [] # List of vertex associate to the diagram 
        self.initial_vertex = [] # vertex associate to initial particles
        self.lineList = []  # List of line present in the diagram
        self.min_level = 0
        self.max_level = 1
        
        #internal parameter
        self._treated_legs = [] # List of leg, in the same order as lineList
        self._available_legs = {} # List of line which can/should be reuse.
        self._ext_distance_up = self.opt.external
        self._ext_distance_down = self.opt.external
        

    def main(self):
        """This routine will compute all the vertex position and line 
        orientation needed to draw the diagram."""
        
        # Define all the vertex/line 
        # Define self.vertexList,self.lineList
        self.load_diagram(contract=self.opt.contract_non_propagating)
        # Define the level of each vertex
        self.define_level()
        # Define position for each vertex
        self.find_initial_vertex_position()
        # Adjust some 'not beautifull' position
        self.adjust_position()
        # Flip the particle orientation such that fermion-flow is correct
        self.solve_line_direction()
    
    #fake vertex for end point particle use in load_diagram
    fake_vertex = base_objects.Vertex({'id':0, 'legs':base_objects.LegList([])})
    
    def load_diagram(self, contract=True):
        """Define all the object for the Feynman Diagram Drawing (Vertex and 
        Line) following the data include in 'self.diagram'
        'contract' defines if we contract to one point the non propagating line.
        """

        for vertex in self.diagram.get('vertices'):
            self.load_vertex(vertex)
        # The last vertex is particular
        last_vertex = self.vertexList[-1]

        for line in last_vertex.lines:
            self.deal_last_line(line)
        
        if contract:
            # Contract the non propagating particle and fuse vertex associated
            self._fuse_non_propa_particule()

        # External particles have only one vertex attach to the line. (by 
        #construction). So we will add a new vertex object in order that all 
        #line are associated to two vertex. Those additional vertex will be 
        #place, later, at the border of the square.
        for line in self.lineList:
            if line.end == 0 or line.begin == 0:
                # Create a new vertex. update the list, assign at the line.
                vertex_point = VertexPoint(self.fake_vertex)
                self.vertexList.append(vertex_point)
                # If initial state particle, we will need to flip begin-end
                if line.state== False:
                    if line.begin:
                        line.inverse_begin_end()
                    line.def_begin_point(vertex_point)
                    vertex_point.def_level(0)
                    self.initial_vertex.append(vertex_point)
                else:
                    if line.end:
                        line.inverse_begin_end()
                    line.def_end_point(vertex_point)
        
        if len(self.initial_vertex) == 2:
            if self.initial_vertex[0].lines[0].number == 2:
                self.initial_vertex.reverse()
        else:
            # Remove wrongly define T-channel
            self.remove_t_channel()
        
        return

    def find_leg_id(self, leg, equal=0, end=0):
        """Find the position of leg in self._treated_legs
            
            if equal=0 returns the last position of number in the list
            otherwise check that leg is the item in self._treated_legs
            
            the two methods provides the same result if they provide a result.
            But some times equal=0 mode provides result when equal=1 doesn't.
            To my understanding equal=1 is suppose to be sufficient in all cases
            but gg> 7x( g ) fails with using equal=1 only.
                        
            'end' removes the last 'end' element of the list, before looking at
            the id in the list. (the list is not modify)"""

        if equal:
            return self.find_leg_id2(leg, end=end)

        for i in range(len(self.lineList) - 1 - end, -1, -1):
            if leg.get('number') == self.lineList[i].number:
                return i

        return None

    def find_leg_id2(self, leg, end=0):
        """Find the position of leg in self._treated_legs. Use object equality 
        to find the position."""

        for i in range(len(self.lineList) - 1 - end, -1, -1):
            if  (self._treated_legs[i] is leg):
                return i

    def find_leg_id3(self, gen_id):
        """Find the position of leg in self._treated_legs but only if this b
        belongs to an available particles"""
        
        try:
            return self._available_legs[gen_id]
        except Exception:
            return None

    def load_vertex(self, vertex):
        """1) Extend the vertex to a VertexPoint. 
        2) Add this vertex in vertexList of the diagram
        3) Update vertex.lines list. (first update the leg into line if needed)

        4) assign line.start[end] to this vertex. (in end if start is already
                assigned to another vertex). the start-end will be flip later
                if needed.
        5) if the fermion flow is correctly set by the diagram (amplitude=True)
           Then change the particles/anti-particles states accordingly.
        """

        #1) Extend to a vertex point
        vertex_point = VertexPoint(vertex)

        #2) Add to the vertexList of the diagram
        self.vertexList.append(vertex_point)
        
        # Loop over the leg associate to the diagram
        for i, leg in enumerate(vertex.get('legs')):
            gen_id = leg.get('number')
            # Search if leg exist: two case exist corresponding if it is the 
            #line of vertex or not. Corresponding to that change mode to find
            #if the leg exist or not.
            mg_id = self.find_leg_id3(gen_id)

            # Define-recover the line associate to this leg                  
            if mg_id:
                del self._available_legs[gen_id]
                line = self.lineList[mg_id]
            else:
                line = self.load_leg(leg)
                if i + 1 == len(vertex.get('legs')):
                    self._available_legs[gen_id] = len(self.lineList) - 1

            # Associate the vertex to the line at the correct place
            line.add_vertex(vertex_point)

        # Change particule to anti-particule for last entry of vertex.lines
        #doing this modification only if the vertex is the type 1 X....Z>1
        #since in this case either the last particles will be a T-channel 
        #and will be resolve latter (so we don't care) or we have to flip
        #particle to antiparticle.
        if line.number == 1 == vertex.get('legs')[0].get('number'):
            line.inverse_part_antipart()
        elif self.amplitude and line.number == 1:
            nb = [l.get('number') for l in vertex.get('legs')]
            if nb.count(1) == 2: 
                line.inverse_part_antipart()           

    def load_leg(self, leg):
        """Extend the leg to Feynman line. Associate the line to the diagram.
        """

        # Extend the leg to FeynmanLine Object
        line = FeynmanLine(leg)
        line.def_model(self.model)

        # Assign line and leg to the diagram. Storing leg is done in order to be
        #able to check if a leg was already treated or not.
        self._treated_legs.append(leg)
        self.lineList.append(line)

        return line

    def deal_last_line(self, last_line):
        """The line of the last vertex breaks the rules that line before
        '>' exist previously and the one after don't. The last one can also
        already exist and for the one before the '>' sometimes they arrive 
        with a second object which is equivalent to another one but not 
        the same object. discover those case and treat this properly."""

        # Check if the line has two vertex associate to it, if not correct.  
        if last_line.end == 0 or last_line.begin == 0:
            # Find the position of the line in self._treated_legs
            id1 = self.find_leg_id(self._treated_legs[-1])
            # Find if they are a second call to this line
            id2 = self.find_leg_id(self._treated_legs[-1], end=len(self._treated_legs) - id1)
            if id2 is not None:
                # Line is duplicate in linelist => remove this duplication
                line = self.lineList[id2]
                # Connect correctly the lines. The two lines are a continuation
                #one of the other with a common vertex. We want to delete 
                #last_line. In consequence we replace in line the common vertex
                #by the second vertex present in last_line, such that the new 
                #line is the sum of the two lines.
                if last_line.begin == 0:
                    if line.end == 0 :
                        line.def_end_point(last_line.end)
                    else:
                        line.def_begin_point(last_line.end)
                    # Remove last_line from the vertex    
                    last_line.end.remove_line(last_line)
                else:
                    if line.end == 0 :
                        line.def_end_point(last_line.begin)
                    else:
                        line.def_begin_point(last_line.begin)
                    # Remove last_line from the vertex    
                    last_line.begin.remove_line(last_line)

                # Remove last_line
                self.lineList.remove(last_line)
            else:
                return #this is an external line => everything is ok

    def _fuse_non_propa_particule(self):
        """Fuse all the non propagating line
            step:
            1) find those line
            2) fuse the vertex
            3) remove one vertex from self.vertexList
            4) remove the line/leg from self.lineList/self._treated_leg
        """

        # Look for all line in backward mode in order to delete entry in the 
        #same time (as making th loop) without creating trouble 
        for i in range(len(self.lineList)).__reversed__():
            if self.lineList[i].get_info('propagating'):
                continue
            else:
                line = self.lineList[i]
                line.begin.fuse_vertex(line.end, common_line=line)
                self.vertexList.remove(line.end)
                del self._treated_legs[i]
                del self.lineList[i]

    def define_level(self):
        """Assign to each vertex a level:
        the level correspond to the number of visible particles and S-channel 
        needed in order to reach the initial particles vertex.
        
        This is computing by search level by level starting at level 0.
        """

        for vertex in self.initial_vertex:
            self.def_next_level_from(vertex) #auto recursive operation

        self.nb_level = self.max_level - self.min_level

    def def_next_level_from(self, vertex):
        """Define level for adjacent vertex.
        If those vertex is already defined do nothing
        Otherwise define as level+1 (at level 1 if T-channel) 
        
        This routine defines also self.max_level.
        
        This routine is foreseen for an auto-recursive mode. So as soon as a 
        vertex have his level defined. We launch this routine for this vertex.
        """
        
        level = vertex.level
        for line in vertex.lines:
            if line.end.level is not None:
                continue
            # Check if T-channel or not. Note that T-channel tag is wrongly 
            #define if only one particle in initial state.
            if line.state == False:
                # This is T vertex. => level is 1
                line.end.def_level(1)
            else:
                # Define level
                line.end.def_level(level + 1)
                # Check for update in self.max_level
                self.max_level = max(self.max_level, level + 1)
            # Launch the recursion
            self.def_next_level_from(line.end)


    def find_t_channel_vertex(self):
        """Returns the vertex (T-vertex authorize) associate to level 1.
        We start with the vertex associate to first entry of previous_level
        and then following the T-line."""

        vertex_at_level = []
        try:
            t_vertex = self.initial_vertex[-2]
        except Exception:
            return [] #only one particle in initial state => no T-channel

        while 1:
            # search next vertex and the connection line leading to this vertex
            t_vertex = self.find_next_t_channel_vertex(t_vertex)

            #add it to the list
            if t_vertex:
                vertex_at_level.append(t_vertex)
            else:
                return vertex_at_level

    def find_next_t_channel_vertex(self, t_vertex):
        """Returns the next t_vertex. i.e. the vertex following t_vertex. t_line
        indicates the 'wrong' T-direction. This routines returns also the 'good'
        evolution direction (which will be the wrong one at the next step)."""

        for line in t_vertex.lines:
            if line.state == False and line.begin is t_vertex:
                return line.end

    def find_vertex_at_level(self, previous_level, level):
        """Returns a list of vertex such that all those vertex are one level 
        after the level of vertexlist and sorted in such way that the list 
        start with vertex connected with the first vertex of 'vertexlist' then 
        those connected to the second and so on."""

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
                    vertex_at_level.append(line.end)

        return vertex_at_level

    def find_initial_vertex_position(self):
        """Find a position to each vertex. All the vertex with the same level
        will have the same x coordinate. All external particles will be on the
        border of the square."""

        if len(self.initial_vertex) == 2:
            self.initial_vertex[0].def_position(0, 0)
            self.initial_vertex[1].def_position(0, 1)
            # Initial state are wrongly consider as outgoing-> solve:
            self.initial_vertex[0].lines[0].inverse_part_antipart()
            self.initial_vertex[1].lines[0].inverse_part_antipart()
            # Associate position to T-vertex  
            t_vertex = self.find_vertex_position_tchannel()
            # Associatie position to level 2 and following (auto-recursive fct)
            self.find_vertex_position_at_level(t_vertex, 2)
            self.initial_vertex[0].def_position(0, 0)
            self.initial_vertex[1].def_position(0, 1)
        elif len(self.initial_vertex) == 1:
            #No T-Channel
            self.initial_vertex[0].def_position(0, 0.5)
            #initial state are wrongly consider as outgoing -> solve:
            init_line = self.initial_vertex[0].lines[0]
            init_line.inverse_part_antipart()          
            # Associate position to level 1
            init_line.end.def_position(1 / self.nb_level, 0.5)
            # Associate position to level 2 and following (auto-recursive fct)
            self.find_vertex_position_at_level([init_line.end], 2)
        else:
            raise self.FeynamDiagramError, \
                                'only for one or two initial particles not %s' \
                                % (len(self.initial_vertex))


    def find_vertex_position_tchannel(self):
        """Finds the vertex position for level one, T channel are authorize"""

        # Find the T-vertex in correct order 
        t_vertex = self.find_t_channel_vertex()
        # Assign position at those vertex
        self.assign_pos(t_vertex, 1)
        return t_vertex


    def find_vertex_position_at_level(self, vertexlist, level, direction=1):
        """Finds the vertex position for the particle at 'level' given the 
        ordering at previous level given by the vertexlist. 
        if direction != 0 pass in auto-recursive mode."""

        if level > self.max_level or level < self.min_level:
            return

        # Find the order of vertex at next-level. if some external particle
        #are in vertexlist. They are replace in vertex_at_level. Such case 
        #happen if the options forbids to an external particles to end at x!=1
        #coordinates or if it's not possible to put the vertex on the border.
        vertex_at_level = self.find_vertex_at_level(vertexlist, level)

        if not vertex_at_level:
            return
        # Assign position to vertex_at_level. In order to deal with external 
        #particles the vertex_at_level is modify. If an external vertex has
        #position on border it will be remove of vertex_at_level.    
        self.assign_pos(vertex_at_level, level)

        # Recursive mode
        if direction and vertex_at_level:
            self.find_vertex_position_at_level(vertex_at_level, 
                                                   level + direction, direction)


    def assign_pos(self, vertex_at_level, level, min=0, max=1):
        """Assign the position to each vertex of vertex_at_level.
        
        The x-coordinate will the ratio of the current level with the maximum
        level of the diagram.
        
        If the first_vertex of vertex_at_level is an outgoing particle. Put it 
        at y=0 if possible (this could be prevented by min>0 or by drawing 
        option). if you put it at y=0 delete the vertex of the list to avoid 
        duplications.
        
        Do the symmetric case for the last entry of vertex_at_level.
        
        The y-value for the other point is computed such that the distance 
        between two vertex of the list are the same. the distance between min 
        (resp. max) and the first vertex is also equal but if min=0 (resp.
        max=1) then this distance counts half.
        
        the option self.opt.external is used
        if    equals 0, the external lines are authorizes to end only 
                   at the end of the diagram (in x=1 axis) so this will forbid
                   to put any vertex at y=0-1 (except if x=1)
        if bigger than 0, minimal distance in before putting a external line
                  on the border of the diagram.
        
        
        The computation of y is done in this way
        first compute the distance [dist] between two vertex and assign the point.
        begin_gap and end_gap are the ratio of the compute distance to put
        between min and first vertex.
        """

        if not vertex_at_level:
            return []

        assert self.min_level <= level <= self.max_level , \
                    'Incorrect value of min/max level: %s <= %s <= %s' % \
                    (self.min_level, level, self.max_level)  
                

        # At final level we should authorize min=0 and max=1 position    
        if level == self.max_level:
            ext_dist_up = 1
            ext_dist_down = 1
            # Treat special 2 > 1 case
            if len(vertex_at_level) == 1 and min == 0 and max == 1:
                vertex_at_level[0].def_position(1, 0.5)
                return []
        else:
            # else follow option
            ext_dist_up = self._ext_distance_up
            ext_dist_down = self._ext_distance_down 
        # Set default gap in dist unity
        begin_gap, end_gap = 1, 1
        # Check the special case when min is 0 -> border
        if min == 0:
            if ext_dist_down and vertex_at_level[0].is_external():
                line = vertex_at_level[0].lines[0]
                if line.end.level - line.begin.level >= ext_dist_down:
                    # Assign position at the border and update option
                    self.define_vertex_at_border(vertex_at_level[0], level, 0)
                    # Remove the vertex to avoid that it will pass to next level
                    del vertex_at_level[0]
                    # 
                    if not vertex_at_level:
                        return []
                else:
                    begin_gap = 0.5
            else:
                begin_gap = 0.5

        # Check the special case when max is 1 -> border    
        if max == 1:
            if ext_dist_up and vertex_at_level[-1].is_external():
                line = vertex_at_level[-1].lines[0]
                if line.end.level - line.begin.level >= ext_dist_up:
                    # Assign position at the border 
                    self.define_vertex_at_border(vertex_at_level[-1], level, 1)
                    # Remove the vertex to avoid that it will pass to next level
                    del vertex_at_level[-1]
                    if not vertex_at_level:
                        return []
                else:
                    end_gap = 0.5
            else:
                end_gap = 0.5

        # Compute the distance between two vertex
        dist = (max - min) / (begin_gap + end_gap + len(vertex_at_level) - 1)

        # Assign position to each vertex
        for i, vertex in enumerate(vertex_at_level):
            vertex.def_position((level - self.min_level) / self.nb_level, 
                                min + dist * (begin_gap + i))

        return vertex_at_level

    def define_vertex_at_border(self, vertex, level, pos_y):
        """Define the position of the vertex considering the distance required
        in the Drawing Options. Update the option if needed."""
        
        # find the minimal x distance and update this distance for the future
        if pos_y == 1:
            dist = self._ext_distance_up
            self._ext_distance_up += self.opt.add_gap
        else:
            dist = self._ext_distance_down
            self._ext_distance_down += self.opt.add_gap
        
        # Find the position and switch integer and not integer case
        if dist % 1:
            # Check that we have to move forward the line
            if level < self.max_level: 
                pos_x = (level - 1 + (dist % 1)) / self.max_level
            elif (1 - vertex.lines[0].begin.pos_x) * self.max_level > dist:
                pos_x = (level - 1 + (dist % 1)) / self.max_level
            else:
                pos_x = 1
        else:
            pos_x = level / self.max_level

        vertex.def_position(pos_x, pos_y)
         

    def remove_t_channel(self):
        """Removes all T-channel in a diagram and convert those in S-channel.
        This occur for 1>X diagram where T-channel are wrongly define."""
        
        for line in self.lineList:
            if line.state == False:
                line.state = True


    def solve_line_direction(self):
        """Computes the directions of the lines of the diagrams.
        first use simple rules as particles move in time directions (to right).
        - define_line_orientation -. Then flip T-channel particles to 
        correct fermion flow in T-channel. Majorana case not deal correctly 
        at this stage."""

        # Use the basic rules. Assigns correctly but for T-channel
        # This methods fails if the creation of wavefunctions modify the 
        # particle content.

        for line in self.lineList:
            if line.state == True:
                line.define_line_orientation()
                                  
        # The define line orientation use level information and in consequence 
        #fails on T-Channel. So in consequence we still have to fix T-channel
        #line.

        # Make a loop on T-channel particles
        try:
            t_vertex = self.initial_vertex[-2]
        except Exception:
            return # No T-channel for 1 > X diagram

        t_vertex = self.find_next_t_channel_vertex(t_vertex)
        self.initial_vertex[0].lines[0].define_line_orientation()
        
        t_old = self.initial_vertex[0].lines[0]
        while 1:
            # Look the total flow of the vertex the other
            ver_flow = 0 # Current flow status for the vertex 
            t_next = None  # Next T-channel line. with unfix fermion flow
            for line in t_vertex.lines:

                # Identify the next T-channel particles
                if line.state == False and t_old is not line and \
                    line.begin is t_vertex:
                    t_next = line
                    
                    #import sys
                    #sys.exit()

                # If not fermion, no update of the fermion flow
                if not line.is_fermion():
                    continue

                # Update the fermion_flow    
                if (line.begin is t_vertex):
                    ver_flow += 1
                elif line.end is t_vertex:
                    ver_flow -= 1

            # End of the loop on the line of the vertex. 
            if t_next:
                t_old = t_next
                t_vertex = t_next.end
                # Check the vertex_flow=0, we were lucky, else correct the flow.
                if ver_flow:
                    t_next.inverse_begin_end()
            else:
                if ver_flow:
                    self.initial_vertex[1].lines[0].inverse_begin_end()
                return


    def adjust_position(self):
        """Modify the position of some particles in order to improve the final
        diagram look. This routines use one option
        1) max_size which forbids external particles to be longer than max_size.
            This is in level unit. If a line is too long we contract it to 
            max_size preserving the orientation.
        2) external indicating the minimal x-gap for an external line. This 
            constraints is already take into account in previous stage. But that
            stage cann't do non integer gap. So this routines correct this."""
        
        finalsize = self.opt.max_size
        
        # Check if we need to do something
        if not finalsize:
            return 

        # Select all external line
        for line in self.lineList:
            if line.is_external():
                # Check the size of final particles to restrict to the max_size
                #constraints.
                if line.state == False or not line.is_external():
                    continue 
                size = line.get_length() * self.max_level
                if size > finalsize:
                    ratio = finalsize / size
                    new_x = line.begin.pos_x + ratio * (line.end.pos_x - 
                                                               line.begin.pos_x)
                    new_y = line.begin.pos_y + ratio * (line.end.pos_y - 
                                                               line.begin.pos_y)
                    line.end.def_position(new_x, new_y)

    def _debug_load_diagram(self):
        """Return a string to check to conversion of format for the diagram. 
        
        This is a debug function."""

        text = 'line content :\n'
        for i in range(0, len(self.lineList)):
            line = self.lineList[i]
            try:
                begin = self.vertexList.index(line.begin)
            except Exception:
                begin = -1
            try:
                end = self.vertexList.index(line.end)
            except Exception:
                end = -1
            try:
                external = line.is_external()
            except Exception:
                external = '?'
            text += 'pos, %s ,id: %s, number: %s, external: %s, S-channel: %s, loop : %s \
                    begin at %s, end at %s \n' % (i, line.id, \
                    line.number, external, line.state, line.loop_line, begin, end)
        text += 'vertex content : \n'
        for i in range(0, len(self.vertexList)):
            vertex = self.vertexList[i]
            text += 'pos, %s, id: %s, external: %s, uid: %s ' % \
                       (i, vertex.id, vertex.is_external(), \
                         vertex.get_uid())
            text += 'line: ' + ','.join([str(self.lineList.index(line)) \
                                                for line in vertex.lines]) + '\n'
        text += '%s' % [(l.number,) for l in self.lineList if l.state==False]
        return text


    def _debug_level(self, text=1):
        """Returns a string to check the level of each vertex. 
        
        This is a debug function."""

        for line in self.lineList:
            if line.begin.level > line.end.level:
                if text == 0:
                    raise self.FeynamDiagramError('invalid level order')


        text = ''
        for vertex in self.vertexList:
            text += 'vertex : ' + str(int(vertex.get_uid())) 
            text += 'line : '
            text += ','.join([str(line['id']) for line in vertex.legs])
            text += ' level : ' + str(vertex.level)
            text += '\n'
        if text:
            return text

    def _debug_position(self):
        """Returns a string to check the position of each vertex. 
        
        This is a debug function."""

        text = ''
        for vertex in self.vertexList:
            text += 'line : '
            text += ','.join([str(line.id) for line in vertex.lines])
            text += ' level : ' + str(vertex.level)
            text += ' pos : ' + str((vertex.pos_x, vertex.pos_y))
            text += '\n'
        return text

    def _debug_has_intersection(self):
        """Returns if some line cross are crossing each other.
        
        This is a debug Function and is used for the test routine."""

        #loop on all pair combination
        for i, line in enumerate(self.lineList):
            for j in range(i + 1, len(self.lineList)):
                line2 = self.lineList[j]
                #check if they are a unvalid intersection
                if line.begin == line2.end or line.begin == line2.begin:
                    continue
                elif line.has_intersection(line2):
                    import logging
                    logger = logging.getLogger('test')
                    logger.info('intersection for %s %s' % (i, j))
                    logger.info('line %s (%s,%s),(%s,%s)' % (i, line.begin.pos_x, line.begin.pos_y,line.end.pos_x, line.end.pos_y))
                    logger.info('line %s (%s,%s),(%s,%s)' % (j, line2.begin.pos_x, line2.begin.pos_y,line2.end.pos_x, line2.end.pos_y))
                    
                    return True
        return False

    def __eq__(self, other):
        """Check if two diagrams are equivalent. (same structure-same particle)
        
        This function is not used for the moment. The initial purpose was the
        avoid duplication of identical diagram in the output (these could happen
        if we contract non propagating line). But the number of such comparaison
        rise as the number of diagram square such that the total time needed for
        this feature was consider as too (time-)expansive."""
        
        if other==None:
            return self.__class__==type(None)
        
        # Check basic globals (this is done to fastenize the check
        if self.max_level != other.max_level:
            return False
        elif len(self.lineList) != len(other.lineList):
            return False
        
        # Then compare vertex by vertex. As we didn't want to use order 
        #information, we first select two vertex with the same position and then
        #compare then.
        other_pos = [(vertex.pos_x, vertex.pos_y) for vertex in other.vertexList]
        for vertex_self in self.vertexList:
            try:
                i = other_pos.index((vertex_self.pos_x, vertex_self.pos_y))
            except Exception:
                # This vertex doesn't have equivalent => They are different.
                return False
            else:
                vertex_other = other.vertexList[i]
 
            # So now we have the 'vertex_self' and 'vertex_other' which are 
            #vertex at the same position. Now we check if they have the same 
            #line content.
            if not vertex_self.has_the_same_line_content(vertex_other):
                return False
 
        # All the vertex and the associate line are equivalent. So the two 
        #diagrams are consider as identical.
        return True


#===============================================================================
# FeynmanDiagramHorizontal
#===============================================================================
class FeynmanDiagramHorizontal(FeynmanDiagram):
    """Object to compute the position of the different Vertex and Line associate
    to a diagram object. This routines is quite similar to FeynmanDiagram. 
    The only differences concerns the rules for the y-coordinate of each vertex.
    
    In case of vertex with one and only one S-channel going to the next level. 
    Then force this line to be horizontal. This creates sub-interval where other
    vertex can be place following the same rule as before  (equal distance 
    between vertex) but this time sub-interval by sub-interval."""

    def find_vertex_position_at_level(self, vertexlist, level, direction=1):
        """Finds the vertex position for the particle at 'level' given the 
        ordering at previous level given by the vertexlist. 
        if auto=True pass in autorecursive mode.
        
        Compare to the function of FeynmanDiagram, this check the number of 
        S-channel particles going out of each vertex. If the result is one:
        1) Fix the associate vertex at the same y as the original vertex
            -> horizontal line
        2) Assign non fix vertex below the fix one in the current interval. 
        3) Continue to the next vertex."""

        # If only final-initial particles no S-channel to fix => old routine
        if level == 1 or level == self.max_level :
            FeynmanDiagram.find_vertex_position_at_level(self, vertexlist, \
                                                           level, direction)
            return
        elif level > self.max_level or level < self.min_level:
            return

        # Find the order of vertex at next-level. if some external particle
        #are in vertexlist. They are replace in vertex_at_level. Such case 
        #happen if the options forbids to an external particles to end at x!=1
        #coordinates or if it's not possible to put the vertex on the border
        #of a previous level.
        vertex_at_level = self.find_vertex_at_level(vertexlist, level)
        vertex_at_level2 = [] # Will be the same list as vertex_at level but 
                              #with a potential different order and whitout some
                              #(already fixed) external particles

        min_pos = 0              # Starting of the current interval
        list_unforce_vertex = [] # Vertex which fit in this interval

        # Loop at level-1 in order to check the number of S-channel going from
        #level-1 to level.
        for vertex in vertexlist:

            s_vertex = []   # List of s vertex going to level
            ext_vertex = [] # List of external particle vertex 
            v_pos = vertex.pos_y

            # Assign the vertex linked to current vertex in the associate 
            #category (S-channel or external)
            for line in vertex.lines:

                # Find the vertex
                if line.end in vertex_at_level:
                    new_vertex = line.end
                elif line.begin in vertex_at_level:
                    new_vertex = line.begin
                else:
                    # The line goes to level-2
                    continue

                # Assign in the correct list (external/s-channel)
                if line.is_external():
                    ext_vertex.append(new_vertex)
                else:
                    s_vertex.append(new_vertex)

            # Check the number of S-channel
            if len(s_vertex) != 1:
                # Udate the list_unforce_vertex. The complex way to do is a 
                #naive attempt of improving the look of the diagram.
                if len(ext_vertex) <= 1:
                    if vertex.pos_y >= 0.5:
                        list_unforce_vertex += (s_vertex + ext_vertex)
                    else:
                        list_unforce_vertex += (ext_vertex + s_vertex)
                else:
                    list_unforce_vertex += ext_vertex[:-1] + s_vertex + \
                                                                ext_vertex[-1:]
                continue

            # Only One S-Channel => force to be horizontal                   
            force_vertex = s_vertex[0]
            force_vertex.def_position(level / self.max_level, v_pos)

            list_unforce_vertex += ext_vertex

            # Assign position to unforce list with some naive try of improvement
            if (len(ext_vertex) == 1 and v_pos >= 0.5) or len(ext_vertex) > 1:
                vertex_at_level2 += self.assign_pos(list_unforce_vertex[:-1], \
                                                        level, min_pos, v_pos)

                list_unforce_vertex = [list_unforce_vertex[-1]]
            else:
                vertex_at_level2 += self.assign_pos(list_unforce_vertex, level, \
                                                                 min_pos, v_pos)
                list_unforce_vertex = []

            # Update value for the next interval
            min_pos = v_pos
            vertex_at_level2.append(force_vertex)

        # End of the loop assign the position of unforce vertex remaining
        if list_unforce_vertex:
            vertex_at_level2 += self.assign_pos(list_unforce_vertex, level, \
                                                min_pos, 1)

        if direction and vertex_at_level2:
            self.find_vertex_position_at_level(vertex_at_level2, 
                                              level + direction, direction)


#===============================================================================
# DiagramDrawer
#===============================================================================
class DiagramDrawer(object):
    """In principle ALL routines representing diagram in ANY format SHOULD 
    derive from this class.
     
    This is a (nearly empty) frameworks to draw a diagram in any type format  

    This frameworks defines in particular 
        - function to convert the input diagram (create by the generation step)
            in the correct object. [convert_diagram]
        - main loop to draw a diagram in a line-by-line method
            [draw - draw_diagram - draw_line] 
        - name of routine (routine are empty) in order to fit with the framework
            [ associate_name - associate_number - draw_straight ]
        - some basic definition of routines
            [conclude - initialize]
    
    This framework is base on the idea that we can create the diagram line after
    line. Indeed all line object (FeynmanLine) contains the full information 
    needed to be drawed independently of the rest of the diagram. 
    
    In order to create a class with this framework you should start to write the
    draw_straight, draw_curly, ... method which are called by the framework.
    
    If you want to write a file, you can store his content in self.text variable
    the routine conclude will then automatically write the file.
    
    The main routine to draw a diagram is 'draw' which call
    1) initialize: setup things for the diagram (usually open a file).
    2) convert_diagram : Update the diagram in the correct format if needed.
    3) draw_diagram : Build the diagram line after line.
    4) conclude : finish the operation. 
    """

    class DrawDiagramError(Exception):
        """Standard error for error occuring to create output of a Diagram."""

    def __init__(self, diagram=None, filename=None, model=None, amplitude=None, \
                                                                    opt=None):
        """Define basic variables and store some global information.
        All argument are optional:
        diagram : is the object to  'diagram' should inherit from either 
                base_objects.Diagram  or drawing_lib.FeynmanDiagram.
        filename: file's name of the file to write.
        model: model associate to the diagram. In principle use only if diagram
            inherit from base_objects.Diagram (for conversion).
        amplitude: amplitude associates to the diagram. NOT USE for the moment.
            In future you could pass the amplitude associate to the object in 
            order to adjust fermion flow in case of Majorana fermion.
        opt: should be a valid DrawOption object."""

        # Check the parameter value
        #No need to test Diagram class, it will be tested before using it anyway
        try:
            assert(not model or isinstance(model, base_objects.Model))
            assert(not filename or isinstance(filename, basestring))
        except AssertionError:
            raise self.DrawDiagramError('No valid model provide to convert ' + \
                                        'diagram in appropriate format')
        
        assert opt is None or isinstance(opt, DrawOption) , \
                        'The Option to draw the diagram are in a invalid format'

        # A Test of the Amplitude should be added when this one will be 
        #use.

        # Store the parameter in the object variable
        self.diagram = diagram
        self.filename = filename
        self.model = model         # use for automatic conversion of graph
        self.amplitude = amplitude # will be use for conversion of graph
        self.opt = opt
        
        # Set variable for storing text        
        self.text = ''
        # Do we have to write a file? -> store in self.file
        if file:
            self.file = True # Note that this variable will be overwritten. THis
                             #will be the object file. [initialize]
        else:
            self.file = False
            
            
    def draw(self, opt=None):
        """Main routine to draw a single diagram.
        opt is DrawOption object use for the conversion of the 
        base_objects.Diagram in one of the Diagram object."""

        # Check if we need to upgrade the diagram.
        self.convert_diagram(amplitude=self.amplitude, opt=opt)
        # Initialize some variable before starting to draw the diagram
        # This is just for frameworks capabilities (default: open file in 
        #write mode if a filename was provide.
        self.initialize()
        # Call the instruction to draw the diagram line by line.
        self.draw_diagram(self.diagram)
        # Finish the creation of the file/object (default: write object if a 
        #filename was provide).
        self.conclude()


    def convert_diagram(self, diagram=None, model=None, amplitude=None, \
                                                opt=None):
        """If diagram is a basic diagram (inherit from base_objects.Diagram)
        convert him to a FeynmanDiagram one. 'opt' keeps track of possible 
        option of drawing. 'amplitude' is not use for the moment. But, later,
        if defined will authorize to adjust the fermion-flow of Majorana 
        particles. opt is a DrawOption object containing all option on the way
        to draw the diagram (see this class for more details)
        
        
        This is the list of recognize options:
            external [True] : authorizes external particles to finish on 
                horizontal limit of the square
            horizontal [True]: if on true use FeynmanDiagramHorizontal to 
                convert the diagram. otherwise use FeynmanDiagram (Horizontal 
                forces S-channel to be horizontal)
            non_propagating [True] : removes the non propagating particles 
                present in the diagram."""

        if diagram is None:
            diagram = self.diagram

        #if already a valid diagram. nothing to do
        if isinstance(diagram, FeynmanDiagram):
            return diagram
        
        if amplitude is None:
            amplitude = self.amplitude
        
        try:
            loop_structure = amplitude.get('structure_repository')
        except Exception:
            loop_structure = None

        # assign default for model and check validity (if not default)
        if model is None:
            model = self.model
        elif not isinstance(model, base_objects.Model):
            raise self.DrawDiagramError('No valid model provide to convert ' + \
                                        'diagram in appropriate format')

        # Test on Amplitude should be enter here, when we will use this 
        #information
        if opt is None:
            if self.opt:
                opt = self.opt
            else:
                opt = DrawOption()
        elif not isinstance(opt, DrawOption):
            raise self.DrawDiagramError('The Option to draw the diagram are' + \
                                        ' in a invalid format')

        # Upgrade diagram to FeynmanDiagram or FeynmanDiagramHorizontal 
        
        #following option choice type is zero for the born and negative for R2
        if isinstance(diagram, loop_objects.LoopDiagram) and diagram.get('type') > 0:
            diagram = LoopFeynmanDiagram(diagram, 
                                    loop_structure,
                                    model, 
                                    opt=opt)
        elif isinstance(diagram, loop_objects.LoopUVCTDiagram) or \
             (isinstance(diagram, loop_objects.LoopDiagram) and \
              diagram.get('type') < 0):
            return None
        else:
            if opt.horizontal:
                diagram = FeynmanDiagramHorizontal(diagram, model, \
                                                   amplitude=amplitude, opt=opt)
            else:
                diagram = FeynmanDiagram(diagram, model, \
                                         amplitude=amplitude, opt=opt)        
        
        #following option choice
        # if opt.horizontal:
        #    diagram = FeynmanDiagramHorizontal(diagram, model, \
        #                                           amplitude=amplitude, opt=opt)
        #else:
        #    diagram = FeynmanDiagram(diagram, model, \
        #                                           amplitude=amplitude, opt=opt)

        # Find the position of all vertex and all line orientation
        assert isinstance(diagram, FeynmanDiagram)
        diagram.main()

        # Store-return information
        self.diagram = diagram
        return diagram

    def initialize(self):
        """Initialization of object-file before starting in order to draw the
        diagram correctly. By default, we just check if we are in writing mode.
        And open the output file if we are."""

        # self.file is set on True/False in __init__. This defines if a filename
        #was provide in the __init__ step. 
        if self.file:
            self.file = open(self.filename, 'w')


    def draw_diagram(self, diagram=None, number=0):
        """Building the diagram Line after Line. 
        This is the key routine of 'draw'."""
        
        # If No diagram set, use the one use at init
        if diagram is None:
            diagram = self.diagram

        # drawing the vertex
        [self.draw_vertex(vertex) for vertex in diagram.vertexList]

        # check if we need curved loop particles
        curved_for_loop = False
        circled_for_loop = False
        
        if isinstance(diagram, LoopFeynmanDiagram):
            # If only 2 particle in the loop require that those lines are
            # curved
            if len([l for l in diagram.lineList if l.loop_line]) == 2:
                curved_for_loop = True
                self.curved_part_start = (0, 0)
        
            # for tadpole DOES NOT CRASH BUT STILL NEED FIXING
            elif len([l for l in diagram.lineList if l.loop_line]) == 1:
                circled_for_loop = True
                self.curved_part_start = (0, 0)

        # drawing the particles
        for line in diagram.lineList:
            if (not curved_for_loop and not circled_for_loop) or not line.loop_line:
                self.draw_line(line)
            elif circled_for_loop:
                self.draw_circled_line(line)
            else:
                self.draw_curved_line(line)
                
        # Finalize information related to the graph. First, associate a diagram
        #position to the diagram representation.
        self.put_diagram_number(number)
        

        # Then If a file exist write the text in it                 
        if self.file:
            self.file.writelines(self.text)
            self.text = ""

    def conclude(self):
        """Final operation of the draw method. By default, this end to write the
        
        file (if this one exist)
        """

        # self.file is set on True/False in __init__. If it is on True
        #the Initialize method change it in a file object
        if self.file:
            self.file.writelines(self.text)
            self.file.close()
        return

    def draw_line(self, line):
        """Draw the line information.
        First, call the method associate the line type [draw_XXXXXX]
        Then finalize line representation by adding his name and, if it's an 
        external particle, the MadGraph5_aMC@NLO number associate to it."""

        # Find the type line of the particle [straight, wavy, ...]
        line_type = line.get_info('line')
        # Call the routine associate to this type [self.draw_straight, ...]
        if hasattr(self, 'draw_' + line_type):
            getattr(self, 'draw_' + line_type)(line)
        else:
            self.draw_straight(line)
            
        # Finalize the line representation with adding the name of the particle
        name = line.get_name()
        self.associate_name(line, name)
        # And associate the MadGraph5_aMC@NLO Number if it is an external particle
        if line.is_external():
            number = line.number
            self.associate_number(line, number)

    # To draw tadpole
    def draw_circled_line(self, line):
        """Draw the line information.
        First, call the method associate the line type [draw_circled_XXXXXX]
        Then finalize line representation by adding his name."""

        # if 4 point (or more interaction at the beginning/end of the loop
        # need to reduce curvature
        if len(line.begin.lines) > 3 or len(line.end.lines) > 3 :
            cercle = False
        else:
            cercle = True

        # Find the type line of the particle [straight, wavy, ...]
        line_type = line.get_info('line')
        # Call the routine associate to this type [self.draw_straight, ...]
        if hasattr(self, 'draw_circled_' + line_type):
            getattr(self, 'draw_circled_' + line_type)(line, cercle)
        else:
            self.draw_circled_straight(line, reduce)
            
        # Finalize the line representation with adding the name of the particle
        name = line.get_name()
        self.associate_name(line, name)
        
        #store begin for helping future curving
        self.curved_part_start = (line.begin.pos_x, line.begin.pos_y*1.2)

    def draw_curved_line(self, line):
        """Draw the line information.
        First, call the method associate the line type [draw_curved_XXXXXX]
        Then finalize line representation by adding his name."""


        # if 4 point (or more interaction at the beginning/end of the loop
        # need to reduce curvature
        if len(line.begin.lines) > 3 or len(line.end.lines) > 3 :
            cercle = False
        else:
            cercle = True

        # Find the type line of the particle [straight, wavy, ...]
        line_type = line.get_info('line')
        # Call the routine associate to this type [self.draw_straight, ...]
        if hasattr(self, 'draw_curved_' + line_type):
            getattr(self, 'draw_curved_' + line_type)(line, cercle)
        else:
            self.draw_curved_straight(line, reduce)
            
        # Finalize the line representation with adding the name of the particle
        name = line.get_name()
        if self.curved_part_start == (line.begin.pos_x, line.begin.pos_y):
            self.associate_name(line, name, loop=True, reverse=True)
            self.curved_part_start = (line.end.pos_x, line.end.pos_y)
        else:
            self.associate_name(line, name, loop=True)
            #store begin for helping future curving
            self.curved_part_start = (line.begin.pos_x, line.begin.pos_y)

    def draw_vertex(self, vertex):
        """default vertex style"""
        pass


    def draw_straight(self, line):
        """Example of routine for drawing the line 'line' in a specific format.
        straight is an example and can be replace by other type of line as 
        dashed, wavy, curly, ..."""

        raise self.DrawDiagramError, 'DrawDiagram.draw_straight should be ' + \
                'overwritten by Inherited Class'

    draw_curved_straight = draw_straight

    def associate_name(self, line, name):
        """Method to associate a name to a the given line. 
        The default action of this framework doesn't do anything"""
        pass


    def associate_number(self, line, number):
        """Method to associate a number to 'line'. By default this method is 
        call only for external particles and the number is the MadGraph5_aMC@NLO number 
        associate to the particle. The default routine doesn't do anything"""
        pass
    
class DrawOption(object):
    """Dealing with the different option of the drawing method.
     This is the list of recognize attributes:
           horizontal [False]: force S-channel to be horizontal
           external [0]: authorizes external particles to end
                     at top or bottom of diagram. If bigger than zero
                     this tune the length of those line.
           add_gap [0]: make external rising after each positioning.
           max_size [0]: this forbids external line bigger than 
                     max_size.
           non_propagating [True]:contracts non propagating lines"""    

    class DrawingOptionError(Exception):
        """Error raising if an invalid entry is set in a option."""

    def __init__(self, opt=''):
        """Fullfill option with standard value."""
        
        #define default
        self.external = 0
        self.add_gap = 0
        self.horizontal = False
        self.max_size = 1.5
        self.contract_non_propagating = True

        if isinstance(opt, dict):
            for key, value in opt.items():
                self.set(key, value)
        else:    
            for value in ['external','add_gap','horizontal','max_size',
                                                    'contract_non_propagating']:
                if hasattr(opt, value):
                    self.set(value, getattr(opt, value))

    def set(self, key, value):
        """Check and attribute the given value."""
        
        if key in ['horizontal', 'contract_non_propagating']:
            value = self.pass_to_logical(value)
            setattr(self, key, value)
        elif(key in ['external', 'max_size', 'add_gap']):
            try:
                value = self.pass_to_number(value)
            except Exception:
                raise self.DrawingOptionError('%s is not a numerical when %s \
                                requires one' % (value, key))
            setattr(self, key, value)
                
        else:
            raise self.DrawingOptionError('%s is not a valid property for  \
                                        drawing object' % key)
            
    def pass_to_logical(self, value):
        """convert the value in a logical"""
        
        if value in [0, False, '0', 'False', 'false']:
            return False
        else:
            return True
         
    def pass_to_number(self, value):
        """Convert the value in a number"""
        
        return float(value)
 
#===============================================================================
# FeynmanDiagramLoop
#===============================================================================
class LoopFeynmanDiagram(FeynmanDiagram):
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

        if self.diagram['tag'] and not self.fdstructures is None:
            for pdg, list_struct_id, vertex_id in self.diagram['tag']:
                for structure_id in list_struct_id:
                    for vertex in self.fdstructures[structure_id]['vertices']:
                        self.load_vertex(vertex)                        
            super(LoopFeynmanDiagram, self).load_diagram(contract)
        else:
            super(LoopFeynmanDiagram, self).load_diagram(contract)
        
        # select the lines present in the loop
        loop_line = [line for line in self.lineList if line.loop_line]
        
        
        # Fuse the cutted particles (the first and the last but one of the list)
        fake_line = loop_line[-1]
        self.fuse_line(loop_line[0], loop_line[-2])
        # delete the fake line:
        self.vertexList.remove(fake_line.end)
        self.lineList.remove(fake_line)

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

            tmp = []
            loop_tmp = []
            for line in vertex.lines:
                if line.begin is vertex and line.end.level == level:
                    if not line.loop_line:
                        tmp.append(line.end)
                    elif started_loop:
                        continue
                    else:
                        started_loop = True
                        loop_tmp = self.find_all_loop_vertex(line.end)
                elif line.end is vertex and line.begin.level == level:
                    if not line.loop_line:
                        tmp.append(line.begin)
                    elif started_loop:
                        continue
                    else:
                        started_loop = True
                        loop_tmp = self.find_all_loop_vertex(line.begin)            
            #order tmp to put external on the bottom/top
            if loop_tmp:
                    vertex_at_level += tmp
                    vertex_at_level += loop_tmp
            else:
                vertex_at_level += tmp
                         
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
                          
        #if not any([True for l in self.lineList if l.loop_line and l.state == False]):
            
        #    return False

        left_side = 0
        right_side = 0
        side_weight = 0 # if side is positive need to switch
        nb_T_channel = 0
        nb_S_channel = 0 

        
        binding_side = {}
        
        # Count the number of T-channel propagator
        for i,vertex in enumerate(self.diagram.get('vertices')):
            if len([l.get('id') for l in vertex.get('legs')]) < 3:
                continue            
            nb_T_channel += len([line for line in vertex.get('legs') if line.get('loop_line') 
                            and line.get('state') == False])
            
            
            
            nb_Tloop = len([line for line in vertex.get('legs') if line.get('loop_line') 
                            and line.get('state')]) 
            nb_S_channel += nb_Tloop


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
        
        if not nb_T_channel:
            return False
        # Note that the number of T_channel/S_channel has a factor 2 compare to  
        # the number of particles in the loop.
        
        # Ensure that the triangle are always correct:
        if nb_S_channel == 2:
            return True 
        elif nb_T_channel == 2:
            return False 
        
        # See the depth of each side 
        if not self.fdstructures is None:
            for pdg, list_struct_id, vertex_id in self.diagram['tag']:
                for structure_id in list_struct_id:
                    leg = self.fdstructures[structure_id].get('binding_leg')
                    if leg.get('number') < 3:
                        continue # connecting to initial particles
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
        else:
            return side_weight > 0
    
    def loop_flip(self):
        """ switch t-channel information for the particle in the loop """

        #for vertex in self.diagram.get('vertices'):
        #    leg = vertex['legs'][-1]
        #    if leg.get('loop_line'):
        #        leg.set('state', not leg.get('state'))
        
        for line in self.lineList:
            if not line.is_external() and line.loop_line:
                line.state = not line.state
 
    
    def remove_t_channel(self):
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
            if line.begin.level is not None and line.end.level is not None:
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
