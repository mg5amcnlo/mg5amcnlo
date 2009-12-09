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



class Feynman_line(base_objects.PhysicsObject):
    """ all the information about a line in a feynman diagram
        i.e. begin-end/type/tag
    """
    
    class FeynmanLineError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a Feynam_line."""
        pass

    def __init__(self, pid, init_dict={}):
        """ initialize the Feynam line"""        
        base_objects.PhysicsObject.__init__(self, init_dict)
        self['pid'] = pid
        self.ordinate_fct=0 #no function assign
        
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

    def has_intersection(self,line):
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
                min, max = self._domain_intersection(line,'y')
                if min == None or min==max:
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
                
        coef1=(yS1-yS0)/(xS1-xS0)
        coef2=(yL1-yL0)/(xL1-xL0)
        
        if coef1 == coef2: #parralel line
            if line._has_ordinate(min) == self._has_ordinate(min):
                return True
            else:
                return False
        commonX=(yS0-yL0-coef1*xS0+coef2*xL0)/(coef2-coef1)
        if ( commonX>=min) == (commonX>=max):
            return False
        
        commonY=self._has_ordinate(commonX)
        if self._is_end_point(commonX, commonY):
            if line._is_end_point(commonX, commonY):
                return False
            else:
                return True 
        else:
            return True
        
    def _is_end_point(self,x,y):
        """ check if this is the end point coordinate """

        if x == self.start['pos_x'] and y == self.start['pos_y']:
            return True
        elif x == self.end['pos_x'] and y == self.end['pos_y']:
            return True
        else:
            return False

    def domain_intersection(self,line,axis='x'):
        """ return x1,x2 where both line and self are defined 
            return None,None if no such domain 
        """
        
        if not isinstance(line, Feynman_line):
            raise self.FeynmanLineError, ' domain intersection are between ' + \
                'Feynman_line object only and not {0} object'.format(type(line))
               
        self.check_position_exist()
        line.check_position_exist()
        return self._domain_intersection(line,axis)
    
    def _domain_intersection(self,line,axis='x'):
        """ return x1,x2 where both line and self are defined 
            return None,None if no such domain 
            whithout existence test
        """
        
        min_self, max_self = self._border_on_axis(axis)
        min_line, max_line = line._border_on_axis(axis)
        
        start=max(min_self,min_line)
        end=min(max_self,max_line)
        if start<=end:
            return start,end
        else:
            return None,None
        
    def _border_on_axis(self,axis='x'):
        """ return the axis coordinate for the begin-end point in a order way """
  
        data=[self.start['pos_'+axis],self.end['pos_'+axis]] 
        data.sort()
        return data
    
    def _intersection_with_vertical_line(self,line): 
        """ deal with case where line is vertical but self is not \
            vertical (no check of that)"""
                
        y_self=self._has_ordinate(line.start['pos_x'])
        ymin,ymax= line._border_on_axis('y')
        if (ymin == y_self or ymax == y_self):
            if self._is_end_point(line.start['pos_x'], y_self):
                return False
            else:
                return True
        elif (y_self>ymin) and (y_self<ymax):
            return True
        else:
            return False            
   
    def check_position_exist(self):
        """ check if the begin-end position are defined """
 
        try:
            min=self.start['pos_x']
            max=self.end['pos_y']
        except:
            raise self.FeynmanLineError, 'No vertex in begin-end position ' + \
                        ' or no position attach at one of those vertex '       
        return
            
    def has_ordinate(self, x):
        """ return the y associate to the x """
        
        self.check_position_exist()
        min = self.start['pos_x']
        max = self.end['pos_x']

        if min==max:
            raise self.FeynmanLineError, 'Vertical line: no unique solution'
        if(not(min<=x<=max)):
            raise self.FeynmanLineError, 'point outside interval'
        
        return self._has_ordinate(x)

    def _has_ordinate(self, x):
        """ hidden routine whithout checks validation """

        if self.ordinate_fct:
            return self.ordinate_fct(x)
        
        x_0 = self.start['pos_x']
        y_0 = self.start['pos_y']
        x_1 = self.end['pos_x']
        y_1 = self.end['pos_y']
        
        alpha=(y_1-y_0)/(x_1-x_0) #x1 always diff of x0
        
        self.ordinate_fct=lambda X: y_0+alpha*(X-x_0)
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
        self['line'] = []
        self['level'] = 0
        self['pos_x'] = 0
        self['pos_y'] = 0
        
    def def_position(self, x, y):
        """ (re)define the position of the vertex in a square (1,1) """
        self['pos_x']=x
        self['pos_y']=y
        pass
    
    def add_line(self, line):
        """add the line in linelist """
        
        if not isinstance(line, Feynman_line):
            raise self.VertexPointError, ' trying to add in a Vertex a non' + \
                            ' FeynmanLine Object'
        
        if line not in self['line']:
            self['line'].append(line)
        pass
        
    def def_level(self, level):
        """ define the level at level """
        pass
    
class Feynman_Diagram:
    """ object which compute the position of the vertex/line for a given
        Diagram object
    """
    
    def __init__(self, diagram):
        """ compute the position of the vertex/line for this diagram """
        
        self.diagram = diagram
        self.VertexList = []
        self.LineList = []
        
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
        pass
        
    def define_level(self):
        """ assign to each vertex a level:
            the level correspond to the number of visible particles and 
            S-channel needed in order to reach the initial particules vertex
        """
        pass
    
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
            print [leg['number'] for leg in data[0]['legs'] if leg['number'] not in [1, 2]] 
            break
    print type(data)
    for info in data:
        print 'id ', info['id']
        for leg in info['legs']:
            print leg['id'],
        print 'end'
        for leg in info['legs']:
            print leg['number'],
        print leg
    
    
    
    
    
    
