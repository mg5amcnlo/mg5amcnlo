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
    
    def get_info(self, name):
        """ return the model information  associated to the line """
        pid = self['pid']
        
        result = self.model.get_particle(pid)
        if result:
            return result[name]
        else:
            pid = -1 * pid #in case of auto anti-particle
            return self.model.get_particle(pid)[name]

    def get_name(self, name='name'):
        """ return the name associate to the particle """
        
        pid = self['pid']
        model_info = self.model.get_particle(pid)
 
        if pid > 0:
            return model_info[name]
        elif model_info:
            return model_info['anti' + name]
        else:
            # particle is self anti particle
            return self.model.get_particle(-1 * pid)[name]
 
      
        

         
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
    
    def _inverse_pid_for_type(self,inversetype='straight'):
        """ change the particle in his anti-particle if this is inversetype """
        
        type=self.get_info('line')
        if type == inversetype:
            self._inverse_part_antipart()

    def _inverse_part_antipart(self):
        """ pass particle into an anti-particule this is needed for 
            initial state particles (usually wrongly defined) and for some
            fermion flow resolution problem
        """
        
        self['pid'] = -1*self['pid']
        
    def _inverse_begin_end(self):
        """ invert the orientation of the line
            this is needed to have correct fermion flow
        """
        
        self.start, self.end = self.end, self.start
        
    def _is_fermion(self):
        """ return true if the particle is a fermion """
        
        model_info = self.model.get_particle(abs(self['pid']))
        if model_info['line'] == 'straight':
            return True

        
    def _define_line_orientation(self,mode=0):
        """
            define the line orientation
            if mode == 0: use the following rules:
                particles move to right, anti-particules to left
        """
        
        if mode == 0: 
            if (self['pid']>0 and self.start['level'] > self.end['level']) or \
                    (self['pid']<0 and self.start['level'] < self.end['level']):
                self._inverse_begin_end()

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
    
    def is_external(self):
        """ check if this particles is an external particles or not """

        if self.end:
            return self.end.is_external() or self.start.is_external()
        else:
            return self.start.is_external()
        
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
                    '0,1 interval introduce value ({0},{1})'.format(x, y)

        return self._def_position(x, y)
    
    def _def_position(self, x, y):
        """ (re)define the position of the vertex in a square (1,1) 
            whithout test
        """        
        self['pos_x'] = x
        self['pos_y'] = y
        
        for line in self['line']:
            line.ordinate_fct = 0    
            
    def _fuse_vertex(self,vertex,common_line=''):
        """ 
            import the line of the second vertex in the first one
            this mean 
            A) change the 'line' of this vertex
            B) change the start-end position of line to point on this vertex
            C) remove common_line (if defined)
        """
        for line in vertex['line']:
            if line is common_line:
                self['line'].remove(line)
                continue
            self['line'].append(line)
            if line.start is vertex:
                line.def_begin_point(self)
            else:
                line.def_end_point(self)
        return
        
    
    def add_line(self, line):
        """add the line in linelist """
        
        if not isinstance(line, Feynman_line):
            raise self.VertexPointError, ' trying to add in a Vertex a non' + \
                            ' FeynmanLine Object'

        for oldline in self['line']:
            if oldline is line:
                return 
#        if line not in self['line']:
        self['line'].append(line)
            
    def remove_line(self, line):
        """ remove a line from the linelist"""

        if not isinstance(line, Feynman_line):
            raise self.VertexPointError, 'trying to remove in a ' + \
                            'Vertex_Point a non FeynmanLine Object'
        
        suceed = 0     
        for i in range(0, len(self['line'])):
            if self['line'][i] is line:
                del self['line'][i]
                suceed = 1
                break # only one recurence!
        if not suceed:    
            raise self.VertexPointError, 'trying to remove in a ' + \
                            'Vertex_Point a non present Feynman_Line'
        
    def def_level(self, level):
        """ define the level at level """
        
        if not isinstance(level, int):
            raise self.VertexPointError, 'trying to attribute non integer level'
        
        self['level'] = level
        
    def is_external(self):
        """ 
            check if this vertex is a vertex associate to an external particle
        """
        
        if len(self['line']) <= 2:
            return True
        else:
            return False
    
class Feynman_Diagram:
    """ object which compute the position of the vertex/line for a given
        Diagram object
    """
    
    class FeynamDiagramError(Exception):
        """ class for internal error """
        pass
    
    def __init__(self, diagram, model, mode=1):
        """ compute the position of the vertex/line for this diagram """
        
        if isinstance(diagram, base_objects.Diagram):
            self.diagram = diagram
        else:
            raise self.FeynamDiagramError('first arg should derivates' + \
                                          ' form Diagram object')
        if isinstance(model, base_objects.Model):
            self.model = model
        else:
            raise self.FeynamDiagramError('second arg should derivates' + \
                                          ' form Modela object')
        
        self.mode = 1 #0 forbids particles to end in 
        
        self.vertexList = base_objects.PhysicsObjectList()
        self.lineList = base_objects.PhysicsObjectList()
        self._treated_legs = []
        self._unpropa_legs = []
        self._vertex_assigned_to_level = [] 
        self.max_level = 0
        
    def main(self):
        #define all the vertex/line 
        #define self.vertexList,self.lineList
        self.charge_diagram()
        #define the level of each vertex
        self.define_level()
        #define a first position for each vertex
        self.find_initial_vertex_position()
        # flip the particule such that fermion-flow is correct
        self.solve_line_direction()
        #avoid any crossing
        self.avoid_crossing()
        #additional update
        #self.optimize()
        
    def charge_diagram(self, fuseUnPropa=True):
        """ define all the object for the Feynman Diagram Drawing (Vertex and 
            Line) following the data include in 'self.diagram'
            if fuseUnPropa=1 
        """
        
        for vertex in self.diagram['vertices']: #treat the last separately     
            self._load_vertex(vertex)
        
        last_vertex = self.vertexList[-1]
        if len(last_vertex['legs']) == 2:
            # vertex say that two line are identical
            self._deal_special_vertex(last_vertex)
        else:
            # check that the last line is not a fake
            for line in last_vertex['line']:
                self._deal_last_line(line)
                
        if fuseUnPropa:
            #contract the unpropagating particle and fuse vertex
            self._fuse_non_propa_particule()
            

        #external particles have only one vertex attach to the line
        vertex = base_objects.Vertex({'id':0, 'legs':base_objects.LegList([])})
        for line in self.lineList:
            if line.end == 0:
                vertex_point = Vertex_Point(vertex)
                self.vertexList.append(vertex_point)
                if line['state'] == 'initial':
                    line.def_end_point(line.start)    #assign at the wrong pos
                    line.def_begin_point(vertex_point)
                    vertex_point.def_level(0)
                else:
                    line.def_end_point(vertex_point)      
           
        #self.assign_model_to_line() #associate the model obj to all line
        #put line in correct direction
        #self._solve_line_direction()
            
        
    def _find_leg_id(self, leg, equal=0, end=0):
        """ find the position of leg in self._treated_legs
            
            if equal=0 returns the last occurence of number in the list
            otherwise check that leg is the occurence in self._treated_legs
            
            the two methods provides if leg is not in self._treated_legs
            (equal=0 send sometimes a result when equal=1 doesn't)
            both output are needed in different part of the code
            
            To my understanding equal=1 is suppose to be sufficient but 
            gg> 7x( g ) forces to use equal 0 mode as well
            
            end removes the comparaison for the 'end' last element of 
            self._treated_legs
        
        """
        
        if equal:
            return self._find_leg_id2(leg, end=end)

        for i in range(len(self.lineList)-1-end,-1,-1):
            if leg['number'] == self.lineList[i]['number']:

                return i 

        return None
                      
    def _find_leg_id2(self,leg,end=0):
        """ find the position of leg in self._treated_legs
            use object equality to find the position 
        """
        for i in range(len(self.lineList)-1-end,-1,-1):
            if  (self._treated_legs[i] is leg):
                return i
                
    def _load_vertex(self, vertex):
        """
        extend the vertex to a VertexPoint
        add this one in self.vertexList
        assign the leg to the vertex (always in first available position
        position for initial particle will be change later
        """ 
        
        vertex_point = Vertex_Point(vertex)
        self.vertexList.append(vertex_point)
        for i in range(0,len(vertex['legs'])):
            leg=vertex['legs'][i] 
            
            #check legs status (old/new)
            if i+1 != len(vertex['legs']):
                # find last item in self._treated_legs with same number
                # returns the position in that list 
                id = self._find_leg_id(leg)
            else:
                # find if leg is in self._treated_legs
                # returns the position in that list
                id = self._find_leg_id(leg, equal=1) 
            
            #define the line associate to this leg                  
            if id:
                line = self.lineList[id]
            else:
                line = self._load_leg(leg)
 
            #associate the vertex to the line
            if line.start == 0:
                line.def_begin_point(vertex_point)
            else:
                line.def_end_point(vertex_point)
        #flip the last entry
        if line['number'] ==1 :# in [1,2]:
            line._inverse_part_antipart()
            
    
    def _load_leg(self, leg):
        """ 
        extend the leg to Feynman line
        add the leg-line in Feynman diagram object
        """
        
        #extend the leg to FeynmanLine Object
        line = Feynman_line(leg['id'], base_objects.Leg(leg)) 
        line._def_model(self.model)
        
        self._treated_legs.append(leg)
        self.lineList.append(line)

        line._inverse_pid_for_type(inversetype='wavy')
        return line
        
 
    def _deal_special_vertex(self, last_vertex):
        """ 
        remove connection vertex and reconnect correctly the different pieces
        """
        
        pos1 = self._find_leg_id(last_vertex['legs'][0],equal=0)
        pos2 = self._find_leg_id(last_vertex['legs'][1],equal=0)
                             
        line1 = self.lineList[pos1]
        line2 = self.lineList[pos2]
        
        #case 1 one of the particle is a external particle
        if line1.end == 0 or line2.end == 0:
            #one external particles detected
            internal = line1
            external = line2
            if line2.end:
                external, internal = internal, external
            external.def_begin_point(internal.start)
            internal.start.remove_line(internal)
            internal.end.remove_line(internal)
            self.lineList.remove(internal)
            self.vertexList.remove(last_vertex)
        else:
            line2.def_end_point(line1.start)
            line1.start.remove_line(line1)
            del self.lineList[pos1]
            self.vertexList.remove(last_vertex)     
            
    def _deal_last_line(self, last_line):
        """ 
        remove connection vertex and reconnect correctly the different pieces
        """
                     
        if last_line.end==0:
            # find the position of the line in self._treated_legs
            id1 = self._find_leg_id(last_line)
            # find if they are a second call to this line
            id2 = self._find_leg_id(last_line, end=len(self._treated_legs)-id1)
            
            #old method
            number=last_line['number']
            is_internal = [i for i in range(0,len(self.lineList)-3) \
                                        if self.lineList[i]['number']==number]
            
            if id2 is not None:
                #line is duplicate in linelist => remove this duplication
                line = self.lineList[id2]
                line.def_end_point(last_line.start)
                last_line.start.remove_line(last_line)
                pos=self.pos_to_line(last_line)
                del self.lineList[pos]              
            else:
                return #this is an external line => everything is ok
    
    def _fuse_non_propa_particule(self):
        """ fuse all the non propagating line
            step:
            1) find those line
            2) fuse the vertex
            3) remove one vertex from self.vertexList
            4) remove the line/leg from self.lineList/self._treated_leg
        """
        
        # look for all line in backward mode in order to delete entry in the 
        # same time without creating trouble 
        for i in range(len(self.lineList)-1,-1,-1):
            if self.lineList[i].get_info('propagating'):
                continue
            else:
                line=self.lineList[i]
                first_vertex, second_vertex = line.start, line.end
                              
                first_vertex._fuse_vertex(second_vertex,common_line=line)
                self.vertexList.remove(second_vertex)
                del self._treated_legs[i]
                del self.lineList[i]
            
        
        
        
        
        
    def assign_model_to_line(self):
        return
        
        for line in self.lineList:
            line._def_model(self.model)
    
    def define_level(self):
        """ assign to each vertex a level:
            the level correspond to the number of visible particles and 
            S-channel needed in order to reach the initial particles vertex
        """
        
        initial_vertex = [vertex for vertex in self.vertexList if \
                                                         vertex['level'] == 0 ]

        #print 'initial vertex', initial_vertex
        for vertex in initial_vertex:
            self._def_next_level_from(vertex)
            #auto recursive operation
            
        
    def _def_next_level_from(self, vertex, data=[]):
        
        #data.append('')
        level = vertex['level']
        #print 'search form level',level
        #outgoing_line = [line for line in vertex['line']]
        #print 'number of outgoing line from this vertex', len(outgoing_line)
        #print 'start external ?'+ ' '.join([ str(line.start.is_external()) for line in outgoing_line])
        #print 'end external ?'+ ' '.join([str(line.end.is_external()) for line in outgoing_line])
        #print 'status ?'+ ' '.join([line['state'] for line in outgoing_line])
        #print self._debug_level()
        for line in vertex['line']:
            if line.end['level'] != -1 and line.start['level'] != -1:
                continue
            direction = 'end'
            if line.end['level'] != -1:
                direction = 'start'
            if line['state'] == 'initial' and \
               len([1 for vertex in self.vertexList if vertex['level']==0]) == 2:
                #print 'find a T vertex'
                getattr(line, direction).def_level(1)# T channel => level 1
            else:
                #print 'find one vertex at level ',level+1
                getattr(line, direction).def_level(level + 1)
                self.max_level = max(self.max_level, level + 1)
            self._def_next_level_from(getattr(line, direction))
            
    def find_initial_vertex_position(self):
        """ find a position to each vertex. all the vertex with the same level
            will have the same x coordinate. All external particules will be
            on the border of the square. The implementation of the code is 
            such that some external particles lines cuts sometimes some 
            propagator. This will be resolve in a second step 
        """
        initial_vertex = [vertex for vertex in self.vertexList if\
                                                         vertex['level'] == 0 ]
        
        if len(initial_vertex) == 2:
            initial_vertex[0].def_position(0, 1)
            initial_vertex[1].def_position(0, 0)
            initial_vertex.reverse() #change order in order to draw from bottom
            #initial state are wrongly consider as outgoing for fermion-> solve:
            initial_vertex[0]['line'][0]._inverse_part_antipart()
            initial_vertex[1]['line'][0]._inverse_part_antipart()       
            t_vertex=self.find_vertex_position_tchannel(initial_vertex)
            self.find_vertex_position_at_level(t_vertex,2)
        elif len(initial_vertex) == 1:
            initial_vertex[0].def_position(0, 0.5)
            #initial state are wrongly consider as outgoing -> solve:
            initial_vertex[0]['line'][0]._inverse_part_antipart()
            self.find_vertex_position_at_level(initial_vertex, 1)
        else:
            raise self.Feynman_DiagramError, \
                                'only for one or two initial particles'


    def find_vertex_position_tchannel(self, vertexlist):
        """ find the vertex position for level one, T channel are authorize """

        t_vertex = self.find_t_channel_vertex(vertexlist)
        self._assign_position_for_level(t_vertex, 1) 
        return t_vertex      
        
    def find_vertex_position_at_level(self, vertexlist, level, auto=True):
        """ find the vertex for the next level """
        
        vertex_at_level = self.find_vertex_at_level(vertexlist)

        if not vertex_at_level:
            return    
        self._assign_position_for_level(vertex_at_level, level)
        # vertex at level is modified (remove external particle
        
        #recursive mode
        if auto and vertex_at_level:
            self.find_vertex_position_at_level(vertex_at_level, level + 1)
            
    def find_vertex_at_level(self, previous_level):
        """
        return the vertex at next level starting from the lowest to the highest 
        previous level should be ordinate in the exact same way
        and call for solving line orientation
        """   
 
        
        vertex_at_level = []          
        level=previous_level[0]['level'] + 1
        for i in range(0,len(previous_level)):
            vertex = previous_level[i]
            if  vertex.is_external() and level-1 != self.max_level and \
                                    level != 1 and vertex['pos_y'] not in [0,1]:
                #move external vertex from one level to avoid external 
                # particles finishing inside the square. 
                vertex.def_level(level)
                
            for line in vertex['line']:
                #if (line['state'] == 'initial'):
                #    continue
                if(line.start['level'] == level):
                    vertex_at_level.append(line.start)
                elif(line.end['level'] == level):
                    vertex_at_level.append(line.end)
                    
        return vertex_at_level
 
    def find_t_channel_vertex(self,previous_level):
        """
        return the vertex at next level starting from the lowest to the highest 
        previous level should be ordinate in the exact same way
        """            
        
        vertex_at_level = []
        t_vertex = previous_level[0]
        tline = ''
        while 1:
            tline,t_vertex=self._find_next_t_channel_vertex(t_vertex, tline)
            if t_vertex:
                vertex_at_level.append(t_vertex)
            else:
                return vertex_at_level        
    
    def _find_next_t_channel_vertex(self,t_vertex,t_line):
        """
            returns the next t_vertex,t_line in the Tvertex-line serie
        """

        #find the next t-channel particle
        t_line = [line for line in t_vertex['line'] if \
                                line['state'] == 'initial' and \
                                line is not t_line ][0]   
        
        #find the new vertex associate to it
        t_vertex = [vertex for vertex in[t_line.start, t_line.end] if\
                                      vertex != t_vertex  ][0]
                                      
        if t_vertex['level']==1:
            return t_line,t_vertex
        else:
            return None, None
                       
    def _assign_position_for_level(self, vertex_at_level, level, mode=''):
        """ 
            assign position in ascendant order for the vertex at level given 
        """
        if mode == '':
            mode=self.mode
        
        if level == self.max_level:
            mode=0
        
        begin_gap, end_gap = 0.5, 0.5
        if vertex_at_level[0].is_external():
            if mode:
                vertex_at_level[0].def_position(level / self.max_level, 0)
                del vertex_at_level[0]
            else:
                begin_gap = 0
            
        if vertex_at_level[-1].is_external():
            if mode:
                vertex_at_level[-1].def_position(level / self.max_level, 1)
                del vertex_at_level[-1]
            else:
                end_gap = 0
            
        den = (begin_gap + end_gap + len(vertex_at_level) - 1)
        if den:
            gap = 1 / den
        else:
            begin_gap = 0.5
            gap = 1
        
        for i in range(0, len(vertex_at_level)):
            vertex = vertex_at_level[i]
            if level > self.max_level:
                raise self.FeynamDiagramError, 'self.max_level not correctly ' + \
                    'assigned %s is lower than level %s' % (self.max_level,level) 
            vertex.def_position(level / self.max_level, gap * (begin_gap + i))
            
    def solve_line_direction(self):
        """ solve the direction of the line. some line need to be flipped 
            in order to agreed with fermion flow- 
        """ 
        for line in self.lineList:
                line._define_line_orientation()
    
        # the define line orientation use level information and in consequence 
        # fails on T-Channel. So in consequence we still have to fix T-channel
        # line
        t_vertex = [vertex for vertex in self.vertexList if vertex['level'] == 0]
        
        t_vertex = t_vertex[1]
        t_line = ''
        #t_line, t_vertex = self._find_next_t_channel_vertex(t_vertex,t_line)
        while 1:
            t_line, t_vertex = self._find_next_t_channel_vertex(t_vertex,t_line)
            
            if t_vertex == None:
                return
            #look the total for the other
            incoming = 0  
            t_dir= 0
            t_next=''
            for line in t_vertex['line']:
                direction=0
                if line['state']=='internal' and line is not t_line:
                    t_next = line 
                if not line._is_fermion():
                    continue
                if (line.start is t_vertex):
                    direction+=1
                elif line.end is t_vertex:
                    direction-=1
                if line['state']=='initial' and line is not t_line:
                    t_next = line 
                    t_dir = direction
                else:
                    incoming += direction
                
            if t_next == '':
                continue   


            if incoming == t_dir:
                t_next._inverse_begin_end()
        
        return
             
    def _debug_charge_diagram(self):
        """ return a string to check to conversion of format for the diagram 
            this is a debug function
        """
        text = 'line content :\n'
        for i in range(0,len(self.lineList)):
            line = self.lineList[i]
            begin = self.pos_to_vertex(line.start)
            end = self.pos_to_vertex(line.end)
            text += 'pos, %s ,id: %s, number: %s, external: %s, \
                    begin at %s, end at %s \n' % (i, line['id'], \
                    line['number'], line.is_external(), begin, end )
        text += 'vertex content : \n'
        for i in range(0,len(self.vertexList)):
            vertex = self.vertexList[i]
            text += 'pos, %s, id: %s, external: %s, ' % (i, vertex['id'], \
                                                         vertex.is_external())
            text += 'line: '+','.join([str(self.pos_to_line(line)) for line in 
                                                        vertex['line']])+'\n'
        return text
        
    def pos_to_vertex(self, vertex):
        """ find the position of 'vertex' in self.vertexList """
        
        for i in range(0,len(self.vertexList)):
            if vertex is self.vertexList[i]:
                return i 
        return None

    def pos_to_line(self, line):
        """ find the position of 'line' in self.lineList """
        
        for i in range(0,len(self.lineList)):
            if line is self.lineList[i]:
                return i 
        return None    
        
            
    def _debug_level(self, text=1):
        """ return a string to check the level of each vertex 
            this is a debug function
        """
        
        for line in self.lineList:
            if line.start['level'] > line.end['level']:
                if text==0:
                    raise self.FeynamDiagramError('invalid level order')
        
        
        text = ''
        for vertex in self.vertexList:
            text += 'line : '
            text += ','.join([str(line['id']) for line in vertex['line']])  
            text += ' level : ' + str(vertex['level'])
            text += '\n'
        if text:
            return text
    
    def _debug_position(self):
        """ return a string to check the position of each vertex 
            this is a debug function
        """
        
        text = ''
        for vertex in self.vertexList:
            text += 'line : '
            text += ','.join([str(line['id']) for line in vertex['line']])  
            text += ' level : ' + str(vertex['level'])
            text += ' pos : ' + str((vertex['pos_x'], vertex['pos_y']))
            text += '\n'
        return text
         
            
                
    def avoid_crossing(self):
        """  modify the position of any vertex in order to avoid any line 
            crossing.
        """
        pass
    

class Feynman_Diagram_horizontal(Feynman_Diagram):
    """ 
        second class for computing point position, force to have horizontal 
        line when only S-channel propagator
    """ 
    
    def find_vertex_position_at_level(self, vertexlist, level, auto=True):
        """ find the vertex for the next level """        
    
        vertex_at_level = self.find_vertex_at_level(vertexlist)
        if level == 1 or level == self.max_level :
            if not vertex_at_level:
                return   
            self._assign_position_for_level(vertex_at_level, level)
            #recursive mode
            if auto :
                self.find_vertex_position_at_level(vertex_at_level, level + 1)
            return
        min_pos=0
        list_unforce_vertex=[]
        #level>1
        for vertex in vertexlist:
            force_vertex=''
            pos=''
            s_prop = 0
            new_vertex = 0
            for line in vertex['line']:
                
                if line.start == vertex and line.end in vertex_at_level :
                    new_vertex += 1
                    if not line.end.is_external():
                        s_prop = -1 * s_prop + 1
                    if s_prop == 1:
                            force_vertex = line.end
                            pos = vertex['pos_y']
                            s_prop = - s_prop
                    elif s_prop == 2:
                            list_unforce_vertex+=[force_vertex,line.end]
                            force_vertex = ''
                            pos = ''
                            s_prop = -s_prop
                    else:
                        list_unforce_vertex.append(line.end)                         
                elif line.end == vertex and (line.start in vertex_at_level):
                    new_vertex += 1
                    if not line.start.is_external():
                        s_prop = -1 * s_prop + 1
                    if s_prop == 1:
                            force_vertex = line.start
                            pos = vertex['pos_y']
                            s_prop = - 1
                    elif s_prop == 2:
                            list_unforce_vertex+=[force_vertex,line.start]
                            force_vertex = ''
                            pos = ''
                            s_prop= -2
                    else:
                        list_unforce_vertex.append(line.start)
                elif line.end == vertex and (vertex in vertex_at_level):
                    #case for external particles moves from one level
                    list_unforce_vertex.append(vertex)
          
            if force_vertex:
                force_vertex.def_position(level / self.max_level, pos)
                if new_vertex > 2 or (new_vertex ==2 and vertex['pos_y']>=0.5):
                    self.assign_position_between(list_unforce_vertex[:-1],min_pos,pos,level)
                    list_unforce_vertex = [list_unforce_vertex[-1]]
                else:
                    self.assign_position_between(list_unforce_vertex,min_pos,pos,level)
                    list_unforce_vertex = []
                min_pos = pos
        if list_unforce_vertex:
            self.assign_position_between(list_unforce_vertex,min_pos, 1 ,level)

        if auto and vertex_at_level:
            self.find_vertex_position_at_level(vertex_at_level, level+1)
            
        
        
    def assign_position_between(self,vertex_at_level,min,max,level,mode=''):
        """
        assign the position to vertex at vertex_at_level. 
        the x coordination is linked to the level
        while the y coordinate is between min and max and depend of the number
        of vertex as well as the type of those one.
        """
        
        if not vertex_at_level:
            return
        
        if mode == '':
            mode = self.mode
            
        if level == self.max_level:
            mode=0
        
        begin_gap, end_gap = 0.5, 0.5
        if min == 0 and vertex_at_level[0].is_external():
            if mode:
                vertex_at_level[0].def_position(level / self.max_level, 0)
                #del self.vertex_at_level[0]
                del vertex_at_level[0]
                if not vertex_at_level:
                    return
            else:
                begin_gap = 0
            
        if max == 1 and vertex_at_level[-1].is_external():
            if mode:
                vertex_at_level[-1].def_position(level / self.max_level, 1)
                #del self.vertex_at_level[-1]
                del vertex_at_level[-1]
            else:
                end_gap = 0
            
        den = (begin_gap + end_gap + len(vertex_at_level) - 1)
        if den:
            gap = (max-min) / den
        else:
            begin_gap = 0.5
            gap = (max-min)
        
        for i in range(0, len(vertex_at_level)):
            vertex = vertex_at_level[i]
            if level > self.max_level:
                raise self.FeynamDiagramError, 'self.max_level not correctly ' + \
                    'assigned %s is lower than level %s' % (self.max_level,level) 
            vertex.def_position(level / self.max_level, min+gap * (begin_gap + i))
        self._debug_position()
                   
                       



    
if __name__ == '__main__':
    
    from madgraph.interface.cmd_interface import MadGraphCmd
    cmd = MadGraphCmd()
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/particles.dat')
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/interactions.dat')
    cmd.do_generate(' g g > g u u~ g')
    
    i = 0
    while i < 300:
        i += 1
        data = cmd.curr_amp['diagrams'][i]['vertices']
        if [leg['number'] for leg in data[0]['legs'] if leg['number'] not in [1, 2]]:
            break
#    for info in data:
#        for leg in info['legs']:
#            print dict.__str__(leg)
            
    print cmd.curr_model.get_particle(-2)
    
    
