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
from __future__ import division
import madgraph.core.base_objects as base_objects



class Feynman_line(base_objects.Leg):
    """ all the information about a line in a feynman diagram
        i.e. begin-end/type/tag
    """
    
    class FeynmanLineError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a Feynam_line."""
        pass

    def __init__(self, pid, init_dict={}):
        """ initialize the Feynam line"""        
        base_objects.Leg.__init__(self, init_dict)
        self['pid'] = pid
        self.ordinate_fct = 0 #no function assign
        self.start = 0
        self.end = 0
        
    def def_begin_point(self, vertex):
        """ (re)define the starting point of the line """

        if not isinstance(vertex, Vertex_Point):
            raise self.FeynmanLineError, 'The begin point should be a ' + \
                 'Vertex_Point object'
        
        self.start = vertex
        vertex.add_line(self)
        return 
    
    def def_end_point(self, vertex):
        """ (re)define the starting point of the line """
 
        if not isinstance(vertex, Vertex_Point):
            raise self.FeynmanLineError, 'The end point should be a ' + \
                 'Vertex_Point object'
        
        self.end = vertex
        vertex.add_line(self)
        return 
    
    def has_type(self):
        """ return the spin of the feynman line """
        pass

    def has_intersection(self, line):
        """ check if the two line intersects 
            return True/False
            contains the check consistency
        """
        
        self.check_position_exist()
        line.check_position_exist()
        
        return self._has_intersection(line)

    def _has_intersection(self, line):
        """ check if the two line intersects 
            return True/False
        """
        
        min, max = self._domain_intersection(line)
        if min == None:
            return False
        
        if min == max :
            if self.start['pos_x'] != self.end['pos_x']:
                if line.start['pos_x'] != line.end['pos_x']: #no vertical line
                    return False
                #line is vertical but not self:
                return self._intersection_with_vertical_line(line)           
                
            elif (line.start['pos_x'] != line.end['pos_x']):
                # self is vertical but not line
                return line._intersection_with_vertical_line(self)
            else:
                #both vertical line
                min, max = self._domain_intersection(line, 'y')
                if min == None or min == max:
                    return False
                else:
                    return True
        
        xS0 = self.start['pos_x']
        yS0 = self.start['pos_y']
        xS1 = self.end['pos_x']
        yS1 = self.end['pos_y']

        xL0 = line.start['pos_x']        
        yL0 = line.start['pos_y']
        xL1 = line.end['pos_x']  
        yL1 = line.end['pos_y']
                
        coef1 = (yS1 - yS0) / (xS1 - xS0)
        coef2 = (yL1 - yL0) / (xL1 - xL0)
        
        if coef1 == coef2: #parralel line
            if line._has_ordinate(min) == self._has_ordinate(min):
                return True
            else:
                return False
        commonX = (yS0 - yL0 - coef1 * xS0 + coef2 * xL0) / (coef2 - coef1)
        if (commonX >= min) == (commonX >= max):
            return False
        
        commonY = self._has_ordinate(commonX)
        if self._is_end_point(commonX, commonY):
            if line._is_end_point(commonX, commonY):
                return False
            else:
                return True 
        else:
            return True
        
    def _is_end_point(self, x, y):
        """ check if this is the end point coordinate """

        if x == self.start['pos_x'] and y == self.start['pos_y']:
            return True
        elif x == self.end['pos_x'] and y == self.end['pos_y']:
            return True
        else:
            return False

    def domain_intersection(self, line, axis='x'):
        """ return x1,x2 where both line and self are defined 
            return None,None if no such domain 
        """
        
        if not isinstance(line, Feynman_line):
            raise self.FeynmanLineError, ' domain intersection are between ' + \
                'Feynman_line object only and not {0} object'.format(type(line))
               
        self.check_position_exist()
        line.check_position_exist()
        return self._domain_intersection(line, axis)
    
    def _domain_intersection(self, line, axis='x'):
        """ return x1,x2 where both line and self are defined 
            return None,None if no such domain 
            whithout existence test
        """
        
        min_self, max_self = self._border_on_axis(axis)
        min_line, max_line = line._border_on_axis(axis)
        
        start = max(min_self, min_line)
        end = min(max_self, max_line)
        if start <= end:
            return start, end
        else:
            return None, None
        
    def _border_on_axis(self, axis='x'):
        """ 
            return the axis coordinate for the begin-end point in a order way 
        """
  
        data = [self.start['pos_' + axis], self.end['pos_' + axis]] 
        data.sort()
        return data
    
    def _intersection_with_vertical_line(self, line): 
        """ deal with case where line is vertical but self is not \
            vertical (no check of that)"""
                
        y_self = self._has_ordinate(line.start['pos_x'])
        ymin, ymax = line._border_on_axis('y')
        if (ymin == y_self or ymax == y_self):
            if self._is_end_point(line.start['pos_x'], y_self):
                return False
            else:
                return True
        elif (y_self > ymin) and (y_self < ymax):
            return True
        else:
            return False            
   
    def check_position_exist(self):
        """ check if the begin-end position are defined """
 
        try:
            min = self.start['pos_x']
            max = self.end['pos_y']
        except:
            raise self.FeynmanLineError, 'No vertex in begin-end position ' + \
                        ' or no position attach at one of those vertex '       
        return
            
    def has_ordinate(self, x):
        """ return the y associate to the x """
        
        self.check_position_exist()
        min = self.start['pos_x']
        max = self.end['pos_x']

        if min == max:
            raise self.FeynmanLineError, 'Vertical line: no unique solution'
        if(not(min <= x <= max)):
            raise self.FeynmanLineError, 'point outside interval invalid ' + \
                    'order {0:3}<={1:3}<={2:3}'.format(min, x, max)
        
        return self._has_ordinate(x)

    def _has_ordinate(self, x):
        """ hidden routine whithout checks validation """

        if self.ordinate_fct:
            return self.ordinate_fct(x)
        
        x_0 = self.start['pos_x']
        y_0 = self.start['pos_y']
        x_1 = self.end['pos_x']
        y_1 = self.end['pos_y']
        
        alpha = (y_1 - y_0) / (x_1 - x_0) #x1 always diff of x0
        
        self.ordinate_fct = lambda X: y_0 + alpha * (X - x_0)
        return self.ordinate_fct(x)

class Vertex_Point(base_objects.Vertex):
    """ extension of the class Vertex in order to store the information 
        linked to the display of the feynman diagram
    """
  
    class VertexPointError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a VertexPoint."""
        pass
       
    def __init__(self, vertex):
        """ include the vertex information """
        #copy data
        if not isinstance(vertex, base_objects.Vertex):
            raise self.VertexPointError, 'cannot extend non VertexObject to' + \
               ' Vertex_Point Object.\n type introduce {0}'.format(type(vertex))
                    
        base_objects.Vertex.__init__(self, vertex)
        self['line'] = []
        self['level'] = -1
        self['pos_x'] = 0
        self['pos_y'] = 0
        
    def def_position(self, x, y):
        """ (re)define the position of the vertex in a square (1,1) """
        if(not(0 <= x <= 1 and 0 <= y <= 1)):
            raise self.VertexPointError, 'vertex coordinate should be in' + \
                    '0,1 interval introduce value ({0:4},{0:4})'.format(x, y)

        return self._def_position(x, y)
    
    def _def_position(self, x, y):
        """ (re)define the position of the vertex in a square (1,1) 
            whithout test
        """        
        self['pos_x'] = x
        self['pos_y'] = y
        
        for line in self['line']:
            line.ordinate_fct = 0    
    
    def add_line(self, line):
        """add the line in linelist """
        
        if not isinstance(line, Feynman_line):
            raise self.VertexPointError, ' trying to add in a Vertex a non' + \
                            ' FeynmanLine Object'

        if line not in self['line']:
            self['line'].append(line)
            
    def remove_line(self, line):
        """ remove a line from the linelist"""

        if not isinstance(line, Feynman_line):
            raise self.VertexPointError, 'trying to remove in a Vertex_Point' + \
                            ' a non FeynmanLine Object'
                            
        try:
            pos = self['line'].index(line)
        except ValueError:
            raise self.VertexPointError, 'trying to remove in a Vertex_Point' + \
                            ' a non present Feynman_Line'
        del self['line'][pos]
        
    def def_level(self, level):
        """ define the level at level """
        
        if not isinstance(level, int):
            raise self.VertexPointError, 'trying to attribute non integer level'
        
        self['level'] = level
    
class Feynman_Diagram:
    """ object which compute the position of the vertex/line for a given
        Diagram object
    """
    
    def __init__(self, diagram):
        """ compute the position of the vertex/line for this diagram """
        
        self.diagram = diagram
        self.VertexList = []
        self.LineList = []
        self._treated_legs = []
        self._vertex_assigned_to_level =[] 
        
    def main(self):
        #define all the vertex/line 
        #define self.VertexList,self.LineList
        self.charge_diagram()
        #define the level of each vertex
        self.define_level()
        #define a first position for each vertex
        self.find_initial_vertex_position()
        #avoid any crossing
        self.avoid_crossing()
        #additional update
        #self.optimize()
        
    def charge_diagram(self):
        """ define all the object for the Feynman Diagram Drawing (Vertex and 
            Line) following the data include in 'diagram' 
        """
        
        for vertex in self.diagram['vertices']: #treat the last separately     
            self._load_vertex(vertex)
        
        last_vertex = self.VertexList[-1]
        if len(last_vertex['legs']) == 2:
            #vertex say that two line are identical
            self._deal_special_vertex(last_vertex)

        #external particles have only one vertex attach to the line
        vertex = base_objects.Vertex({'id':0, 'legs':base_objects.LegList([])})
        for line in self.LineList:
            if line.end == 0:
                vertex_point = Vertex_Point(vertex)
                self.VertexList.append(vertex_point)
                if line['state'] == 'initial':
                    line.def_end_point(line.start)    #assign at the wrong pos
                    line.def_begin_point(vertex_point)
                    vertex_point.def_level(0)
                else:
                    line.def_end_point(vertex_point)
                
                
    def _load_vertex(self, vertex):
        """
        extend the vertex to a VertexPoint
        add this one in self.VertexList
        assign the leg to the vertex (always in first available position
        position for initial particle will be chage later
        """
        
        vertex_point = Vertex_Point(vertex)
        self.VertexList.append(vertex_point)
        for leg in vertex['legs']:
            try:
                position = self._treated_legs.index(leg)
            except ValueError:
                line = self._load_leg(leg)
            else:    
                line = self.LineList[position]

            if line.start == 0:
                line.def_begin_point(vertex_point)
            else:
                line.def_end_point(vertex_point)         
    
    def _load_leg(self, leg):
        """ 
        extend the leg to Feynman line
        add the leg-line in Feynman diagram object
        """
        
        #extend the leg to FeynmanLine Object
        line = Feynman_line(leg['id'], base_objects.Leg(leg)) 
        self._treated_legs.append(leg)
        self.LineList.append(line)
        return line
 
    def _deal_special_vertex(self,last_vertex):
        """ """
        pos1 = self._treated_legs.index(last_vertex['legs'][0])
        pos2 = self._treated_legs.index(last_vertex['legs'][1])
        line1 = self.LineList[pos1]
        line2 = self.LineList[pos2]
        
        #case 1 one of the particule is a external particule
        if line1.end or line2.end:
            internal=line1
            external=line2
            if line2.end:
                external,internal=internal,external
            external.def_begin_point(internal.start)
            internal.start.remove_line(internal)
            internal.end.remove_line(internal)
            self.LineList.remove(internal)
            self.VertexList.remove(last_vertex)
        else:
            line1.def_end_point(line2.start)
            line2.start.remove_line(line2)
            del self.LineList[pos2]
            self.VertexList.remove(last_vertex)       
        
    
    def define_level(self):
        """ assign to each vertex a level:
            the level correspond to the number of visible particles and 
            S-channel needed in order to reach the initial particules vertex
        """
        
        initial_vertex=[vertex for vertex in self.VertexList if\
                                                         vertex['level'] == 0 ]
        #print 'initial vertex', initial_vertex
        for vertex in initial_vertex:
            self._def_next_level_from(vertex)
            #auto recursive operation
        
    def _def_next_level_from(self,vertex):
        
        level=vertex['level']
        outgoing_line=[line for line in vertex['line'] if line.start == vertex]
        for line in outgoing_line:
            if line.end['level']!=-1:
                continue
            if line['state']=='initial':
                line.end.def_level(max(level,1))
            else:
                line.end.def_level(level+1)
            self._def_next_level_from(line.end)
            
    def find_initial_vertex_position(self):
        """ find a position to each vertex. all the vertex with the same level
            will have the same x coordinate. All external particules will be
            on the border of the square. The implementation of the code is 
            such that some external particles lines cuts sometimes some 
            propagator. This will be resolve in a second step 
        """
        pass
    
    def find_vertex_position_at_level(self, level):
        """ find the vertex for the next level """
     
    def avoid_crossing(self):
        """  modify the position of any vertex in order to avoid any line 
            crossing.
        """
        pass
    
    
if __name__ == '__main__':
    
    from madgraph.interface.cmd_interface import MadGraphCmd
    cmd = MadGraphCmd()
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/particles.dat')
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/interactions.dat')
    cmd.do_generate(' g g > g u u~ g')
    
    i = 0
    while i < 300:
        i += 1
        data = cmd._MadGraphCmd__curr_amp['diagrams'][i]['vertices']
        if [leg['number'] for leg in data[0]['legs'] if leg['number'] not in [1, 2]]:
            break
    for info in data:
        for leg in info['legs']:
            print dict.__str__(leg)
    
    
    
    
    
