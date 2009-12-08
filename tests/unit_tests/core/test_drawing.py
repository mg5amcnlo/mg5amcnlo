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
"""Unit test library for the routine creating the points position for the 
    diagram drawing"""
from __future__ import division
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.drawing as drawing

#===============================================================================
# TestTestFinder
#===============================================================================
class TestFeynman_line(unittest.TestCase):
    """The TestCase class for the test the Feynman_line"""
    
    def setUp(self):
        """ basic building of the class to test """
        
        self.my_line = drawing.Feynman_line(11)
        myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
                                      'from_group':False})] * 10)
        self.mydict = {'id':3,
                      'legs':myleglist}
        self.vertex = base_objects.Vertex(self.mydict)        
        self.my_vertex = drawing.Vertex_Point(self.vertex) #extend class
        self.my_vertex2 = drawing.Vertex_Point(self.vertex)

    @staticmethod
    def def_line(begin, end):
        """ fast way to have line with begin-end (each are list) """

        myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
                                      'from_group':False})] * 10)
        mydict = {'id':3,
                      'legs':myleglist}
        vertex = base_objects.Vertex(mydict)
                    
        my_line = drawing.Feynman_line(11)
        my_vertex = drawing.Vertex_Point(vertex)
        my_vertex.def_position(begin[0], begin[1])
        my_line.def_begin_point(my_vertex)
        
        my_vertex = drawing.Vertex_Point(vertex)
        my_vertex.def_position(end[0], end[1])
        my_line.def_end_point(my_vertex)
        return my_line




        
    def  test_def_begin_end_point(self):
        """ test if you can correctly assign/reassign begin vertex to a line """
        
        #test begin point 
        self.my_line.def_begin_point(self.my_vertex)
        self.assertTrue(self.my_line.start is self.my_vertex)
        self.my_line.def_begin_point(self.my_vertex2)
        self.assertTrue(self.my_line.start is self.my_vertex2)
        
        #test end point
        self.my_line.def_end_point(self.my_vertex)
        self.assertTrue(self.my_line.end is self.my_vertex)
        self.my_line.def_end_point(self.my_vertex2)
        self.assertTrue(self.my_line.end is self.my_vertex2)
        
        #test if the vertex references the line correctly
        self.assertTrue(self.my_line in self.my_vertex['line'])
        self.assertTrue(self.my_line in self.my_vertex2['line'])        
        
    def test_begin_end_wrong_input(self):
        """ test that begin/end routines forbids wrong entry """
    
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          self.my_line.def_begin_point([0, 0]))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          self.my_line.def_end_point([0, 0]))
        
    def test_has_type(self):
        """ test if we found the correct type of line for some basic line """
        
        #need to load SM?
        for id in [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]:
            my_line = drawing.Feynman_line(id)
            self.assertTrue(my_line.has_type() == 0.5)
            
        for id in [25]:
            my_line = drawing.Feynman_line(id)
            self.assertTrue(my_line.has_type() == 0)        
        
        for id in [21, 22, 23, 24]:
            my_line = drawing.Feynman_line(id)
            self.assertTrue(my_line.has_type() == 1)               
    
    def test_hasintersection(self):
        """ check if two lines cross-each-other works correctly"""
        
        #def a set of line
        my_line1 = self.def_line([0, 0], [1, 1])         #diagonal
        my_line2 = self.def_line([0.5, 0.5], [0.9, 0.9]) # part of the diagonal
        my_line3 = self.def_line([0.1, 0.1], [0.4, 0.4]) # other part of the diagonal
        my_line4 = self.def_line([0, 0.5], [0.5, 1])     # parallel to the diagonal
        my_line5 = self.def_line([0.0, 0.0], [0.0, 1.0]) # y axis 
        my_line6 = self.def_line([0, 1], [1, 0])         # second diagonal
        my_line7 = self.def_line([0, 1], [0.6, 0.4])     # part of the second 
        my_line8 = self.def_line([0.6, 0.4], [0, 1])     # same part but inverse order
        my_line9 = self.def_line([0, 0.5], [0.9, 1])     # other

        #Line 1 intersection
        self.assertTrue(my_line1.has_intersection(my_line1))
        self.assertTrue(my_line1.has_intersection(my_line2))
        self.assertTrue(my_line1.has_intersection(my_line3))
        self.assertFalse(my_line1.has_intersection(my_line4))
        self.assertFalse(my_line1.has_intersection(my_line5)) #cross=begin point
        self.assertTrue(my_line1.has_intersection(my_line6))
        self.assertTrue(my_line1.has_intersection(my_line7))
        self.assertTrue(my_line1.has_intersection(my_line8))
        self.assertFalse(my_line1.has_intersection(my_line9))        
        
        #Line2 intersection
        self.assertTrue(my_line2.has_intersection(my_line1))      
        self.assertFalse(my_line2.has_intersection(my_line3))
        self.assertFalse(my_line2.has_intersection(my_line4))
        self.assertFalse(my_line2.has_intersection(my_line5))
        self.assertTrue(my_line2.has_intersection(my_line6))
        self.assertTrue(my_line2.has_intersection(my_line7))
        self.assertTrue(my_line2.has_intersection(my_line8))
        self.assertFalse(my_line2.has_intersection(my_line9))
                 
        #Line3 intersection
        self.assertFalse(my_line3.has_intersection(my_line4))
        self.assertFalse(my_line3.has_intersection(my_line5))
        self.assertFalse(my_line3.has_intersection(my_line6))
        self.assertFalse(my_line3.has_intersection(my_line7))
        self.assertFalse(my_line3.has_intersection(my_line8))   
        self.assertFalse(my_line3.has_intersection(my_line9))
           
        # Line 4 intersection
        self.assertTrue(my_line4.has_intersection(my_line5))
        self.assertTrue(my_line4.has_intersection(my_line6))
        self.assertTrue(my_line4.has_intersection(my_line7))
        self.assertTrue(my_line4.has_intersection(my_line8))
        self.assertFalse(my_line4.has_intersection(my_line9)) 
        
        # Line 5 intersection
        self.assertFalse(my_line5.has_intersection(my_line6))
        self.assertFalse(my_line5.has_intersection(my_line7))
        self.assertFalse(my_line5.has_intersection(my_line8))
        self.assertTrue(my_line1.has_intersection(my_line9))         
        
        # Line 6 intersection
        self.assertTrue(my_line6.has_intersection(my_line7))
        self.assertTrue(my_line6.has_intersection(my_line8))
        self.assertTrue(my_line6.has_intersection(my_line9))         
        
        #Line 7-8 intersection
        self.assertTrue(my_line7.has_intersection(my_line8))
        self.assertTrue(my_line8.has_intersection(my_line7))
        self.assertTrue(my_line7.has_intersection(my_line9))
  
    def test_domainintersection(self):
        """ check if two lines cross-each-other works correctly"""
        
        #def a set of line
        my_line1 = self.def_line([0, 0], [1, 1])         #diagonal
        my_line2 = self.def_line([0.5, 0.5], [0.9, 0.9]) # part of the diagonal
        my_line3 = self.def_line([0.1, 0.1], [0.4, 0.4]) # other part of the diagonal
        my_line4 = self.def_line([0, 0.5], [0.5, 1])     # parallel to the diagonal
        my_line5 = self.def_line([0.0, 0.0], [0.0, 1.0]) # y axis 
        my_line6 = self.def_line([0, 1], [1, 0])         # second diagonal
        my_line7 = self.def_line([0, 1], [0.6, 0.4])     # part of the second 
        my_line8 = self.def_line([0.6, 0.4], [0, 1])     # same part but inverse order
        my_line9 = self.def_line([0, 0.5], [0.9, 1])     # other

        #with line1 ->return line 2 domain
        self.assertEquals(my_line1.domain_intersection(my_line1), (0, 1))        
        self.assertEquals(my_line1.domain_intersection(my_line2), (0.5, 0.9))
        self.assertEquals(my_line1.domain_intersection(my_line3), (0.1, 0.4))
        self.assertEquals(my_line1.domain_intersection(my_line4), (0.0, 0.5))
        self.assertEquals(my_line1.domain_intersection(my_line5), (0.0, 0.0))
        self.assertEquals(my_line1.domain_intersection(my_line6), (0.0, 1.0))
        self.assertEquals(my_line1.domain_intersection(my_line7), (0, 0.6))
        self.assertEquals(my_line1.domain_intersection(my_line8), (0, 0.6))
        self.assertEquals(my_line1.domain_intersection(my_line9), (0, 0.9))
        
        #with line 5 =>(0,0) at max
        self.assertEquals(my_line5.domain_intersection(my_line2), (None, None))
        self.assertEquals(my_line5.domain_intersection(my_line3), (None, None))
        self.assertEquals(my_line5.domain_intersection(my_line4), (0.0, 0.0))
        self.assertEquals(my_line5.domain_intersection(my_line5), (0.0, 0.0))
        self.assertEquals(my_line5.domain_intersection(my_line6), (0.0, 0.0))
        self.assertEquals(my_line5.domain_intersection(my_line7), (0, 0.0))
        self.assertEquals(my_line5.domain_intersection(my_line8), (0, 0.0))
        self.assertEquals(my_line5.domain_intersection(my_line9), (0, 0.0))        
                
      
    def test_hasordinate(self):
        """ test if the return of ordinate works correctly """
        
        my_line1 = self.def_line([0.1, 0.1], [0.4, 0.1]) #horizontal
        my_line3 = self.def_line([0.1, 0.1], [0.4, 0.4]) #normal
        
    
        #returns correct value in other case
        self.assertEquals(my_line1.has_ordinate(0.2), 0.1)
        self.assertEquals(my_line1.has_ordinate(0.1), 0.1)
        self.assertEquals(my_line3.has_ordinate(0.2), 0.2)
        self.assertEquals(my_line3.has_ordinate(0.1), 0.1)
        
    def test_hasordinate_wronginput(self):
        """ check the error raises in case of incorrect input """
        
        my_line1 = self.def_line([0.1, 0.1], [0.4, 0.2]) #random
        my_line2 = self.def_line([0.1, 0.1], [0.1, 0.4]) #vertical 
        
        #fail if asked outside range
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line1.has_ordinate(-2))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line1.has_ordinate(1.2))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line1.has_ordinate(0.05))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line1.has_ordinate(0.5))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(-2))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(1.2))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(0.05))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(0.5))
        
        #fails for vertical line
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(0.1))
        
        #fails if not real value
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate('a'))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate([0, 0.2]))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.has_ordinate(my_line1))


    def test_failure_if_no_vertex(self):
        """ check that the correct is raise if no vertex
            are assigned before doing position related operation
        """
        print type(self.my_line)
        print type(self.my_line.has_ordinate)    

        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          self.my_line.has_ordinate(0.5))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          self.my_line.intersection(self.my_line))
        
        #check intersection if one is valid
        my_line2 = self.def_line([0.1, 0.1], [0.4, 0.2]) #random
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          my_line2.intersection(self.my_line))
        self.assertRaises(drawing.Feynman_line.FeynmanLineError, \
                          self.my_line.intersection(my_line2))        

#===============================================================================
# TestVertex
#===============================================================================
class TestVertexPoint(unittest.TestCase):
    """The TestCase class for testing Vertex_Point"""
    
    def setUp(self):
        """ basic building of the class to test """
        
        self.line1 = drawing.Feynman_line(11)
        self.line2 = drawing.Feynman_line(11)
        self.line3 = drawing.Feynman_line(11)
        self.line4 = drawing.Feynman_line(11)
        self.myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
                                      'from_group':False})] * 10)
        self.mydict = {'id':3,
                      'legs':self.myleglist}
        self.vertex = base_objects.Vertex(self.mydict)
        
    def testbuilding(self):
        """ test the correct creation of object """
        
        my_vertex = drawing.Vertex_Point(self.vertex)
        self.assertTrue(isinstance(my_vertex, base_objects.Vertex))
        self.assertTrue(isinstance(my_vertex, drawing.Vertex_Point))
        
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          drawing.Vertex_Point({'data':''}))
        
 
 
    def testdef_position(self):
        """ test the position definition are set correctly"""
        
        my_vertex = drawing.Vertex_Point(self.vertex)
        my_vertex.def_position(0.1, 0.3)
        self.assertEqual(my_vertex['pos_x'], 0.1)
        self.assertEqual(my_vertex['pos_y'], 0.3)
        #check border are corectly define (no error raises
        my_vertex.def_position(0, 0.3)
        my_vertex.def_position(0, 0)
        my_vertex.def_position(0, 1)
        my_vertex.def_position(1, 0)
        my_vertex.def_position(1, 0.3)
        my_vertex.def_position(1, 1)
        my_vertex.def_position(0.3, 0)
        my_vertex.def_position(0.3, 1)      
        
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          my_vertex.def_position(1.4, 0.2))
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          my_vertex.def_position(-1.0, 0.2))
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          my_vertex.def_position(0.4, 1.2))
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          my_vertex.def_position(0, -0.2))
  
  
        
    def testredef_position(self):
        """ test the position definition are set correctly"""        
        #check that lambda function linked to Line are correctly remo
        
        my_vertex = drawing.Vertex_Point(self.vertex)
        my_vertex.def_position(0.1, 0.3)
        my_vertex2 = drawing.Vertex_Point(self.vertex)
        my_vertex2.def_position(0.4, 0.5)        
        self.line1.def_begin_point(my_vertex)
        self.line1.def_end_point(my_vertex2)
        self.line1.has_ordinate(0.2)
        my_vertex2.begin_position(0.3, 0.5)
        self.assertEquals(self.line1.ordinate_fct,0)
        self.assertEquals(self.line1.has_ordinate(0.35),0.5)
                
    def test_add_line(self):
        """add the line in linelist """
        
        my_vertex = drawing.Vertex_Point(self.vertex)
        my_vertex.add_line(self.line1)
        
        self.assertTrue(self.line1 in my_vertex['line'])       
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                                                    my_vertex.add_line('data'))
         
    def testdef_level(self):
        """ define the level at level """
        
        my_vertex = drawing.Vertex_Point(self.vertex)
        my_vertex.def_level(3)
        self.assertEquals(my_vertex['level'], 3)
        
        self.assertRaises(drawing.Vertex_Point.VertexPointError, \
                          my_vertex.def_level('3'))
                          
                          
#===============================================================================
# TestVertex
#===============================================================================
class TestFeynman_Diagram(unittest.TestCase):
    """ Test the object which compute the position of the vertex/line 
        for a given Diagram object
    """
    
    #test diagram gg>g(g>jj)g via a T-channel   
    leg1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                            'from_group':False})
    leg2 = base_objects.Leg({'id':22, 'number':2, 'state':'initial',
                            'from_group':False})
    leg3 = base_objects.Leg({'id':22, 'number':3, 'state':'final',
                            'from_group':False})
    leg4 = base_objects.Leg({'id':2, 'number':4, 'state':'final',
                            'from_group':False})    
    leg5 = base_objects.Leg({'id':-2, 'number':5, 'state':'final',
                            'from_group':False})
    leg6 = base_objects.Leg({'id':22, 'number':6, 'state':'final',
                            'from_group':False})        
    
    #intermediate particle +vertex associate
    leg_t1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                        'from_group':True}) 
    vertex1 = base_objects.Vertex({'id':1, \
                        'legs':base_objects.LegList([leg1, leg3, leg_t1])})
    
    leg_t2 = base_objects.Leg({'id':22, 'number':2, 'state':'initial',
                        'from_group':True})    
    vertex2 = base_objects.Vertex({'id':2, \
                        'legs':base_objects.LegList([leg2, leg6, leg_t2])})
 
    leg_s1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                        'from_group':True}) 
    vertex3 = base_objects.Vertex({'id':3, \
                        'legs':base_objects.LegList([leg_t1, leg_t2, leg_s1])})
    
    vertex4 = base_objects.Vertex({'id':4, \
                        'legs':base_objects.LegList([leg_s1, leg3, leg4])})

    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3, vertex4])
    
    mix_diagram_dict = {'vertices':vertexlist}
    
    #test diagram gg>gg via a T-channel
    leg1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                            'from_group':False})
    leg2 = base_objects.Leg({'id':22, 'number':2, 'state':'initial',
                            'from_group':False})
    leg3 = base_objects.Leg({'id':22, 'number':3, 'state':'final',
                            'from_group':False})
    leg4 = base_objects.Leg({'id':22, 'number':4, 'state':'final',
                            'from_group':False})    
   
    #intermediate particle +vertex associate
    leg_t1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                        'from_group':True}) 
    vertex1 = base_objects.Vertex({'id':1, \
                        'legs':base_objects.LegList([leg1, leg3, leg_t1])})
    
    leg_t2 = base_objects.Leg({'id':22, 'number':2, 'state':'initial',
                        'from_group':True})    
    vertex2 = base_objects.Vertex({'id':2, \
                        'legs':base_objects.LegList([leg2, leg6, leg_t2])})
    
    vertex3 = base_objects.Vertex({'id':3, \
                        'legs':base_objects.LegList([leg_t1, leg_t2])})
    
    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3])
    t_diagram_dict = {'vertices':vertexlist}
    t_diagram = base_objects.Diagram(t_diagram_dict)
    
    #test diagram gg>gg via a S-channel
    leg1 = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                            'from_group':False})
    leg2 = base_objects.Leg({'id':22, 'number':2, 'state':'initial',
                            'from_group':False})
    leg3 = base_objects.Leg({'id':22, 'number':3, 'state':'final',
                            'from_group':False})
    leg4 = base_objects.Leg({'id':22, 'number':4, 'state':'final',
                            'from_group':False})    
   
    #intermediate particle +vertex associate
    leg_s = base_objects.Leg({'id':22, 'number':1, 'state':'initial',
                        'from_group':True}) 
    vertex1 = base_objects.Vertex({'id':1, \
                        'legs':base_objects.LegList([leg1, leg2, leg_s])})
    
    leg_temp = base_objects.Leg({'id':22, 'number':1, 'state':'final',
                            'from_group':False})    
       
    vertex2 = base_objects.Vertex({'id':2, \
                        'legs':base_objects.LegList([leg_s, leg3, leg_temp])})
    
    vertex3 = base_objects.Vertex({'id':3, \
                        'legs':base_objects.LegList([leg_temp, leg4])})
    
    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3])
    s_diagram_dict = {'vertices':vertexlist}
    s_diagram = base_objects.Diagram(s_diagram_dict)   

    
    def setUp(self):
        """ basic building of the object needed to build the test """
            
        # gg>g(g>uux)g (via a T channel)  
        mix_diagram = base_objects.Diagram(self.mix_diagram_dict)
        self.mix_drawing = drawing.Feynman_Diagram(mix_diagram)
        
        # gg>gg (via a T channel)
        t_diagram = base_objects.Diagram(self.t_diagram_dict)
        self.t_drawing = drawing.Feynman_Diagram(t_diagram)
        
        # gg>gg (via a S channel)
        s_diagram = base_objects.Diagram(self.s_diagram_dict)
        self.s_diagram = drawing.Feynman_Diagram(s_diagram)
              
    def test_charge_diagram(self):
        """ define all the object for the Feynman Diagram Drawing (Vertex and 
            Line) following the data include in 'diagram' 
        """
        
        # check len of output
        self.mix_drawing.charge_diagram()
        self.assertEqual(len(self.mix_drawing.VertexList), 10)
        self.assertEqual(len(self.mix_drawing.LineList), 9)
        
        self.t_drawing.charge_diagram()
        self.assertEqual(len(self.t_drawing.VertexList), 6)
        self.assertEqual(len(self.t_drawing.LineList), 5)
        
        self.s_drawing.charge_diagram()
        self.assertEqual(len(self.s_drawing.VertexList), 6)
        self.assertEqual(len(self.s_drawing.LineList), 5) 
        
        #check type of object
        for obj in self.mix_drawing.VertexList:
            self.assertTrue(isinstance(obj, drawing.Vertex_Point))
        for obj in self.mix_drawing.LineList:
            self.assertTrue(isinstance(obj, drawing.Feynman_line))
            
                
    def test_define_level(self):
        """ assign to each vertex a level:
            the level correspond to the number of visible particles and 
            S-channel needed in order to reach the initial particules vertex
        """
        self.mix_drawing.charge_diagram()
        self.mix_drawing.define_level()
        
        level_solution = [0, 0, 3, 3, 3, 3, 1, 1, 1, 2] #order: initial-external-vertex \ 
                                             # in diagram order                                 
        for i in range(0, 10):
            self.assertEquals(self.mix_drawing.VertexList[i]['level'], \
                              level_solution[i])
            
        self.s_drawing.charge_diagram()
        self.s_drawing.define_level()

        level_solution = [0, 0, 3, 3, 1, 2] #order: initial-external-vertex in \
                                     #   diagram order                                 
        for i in range(0, 6):
            self.assertEquals(self.s_drawing.VertexList[i]['level'], \
                              level_solution[i])        
        
        
        self.t_drawing.charge_diagram()       
        self.t_drawing.define_level()
        
        level_solution = [0, 0, 2, 2, 1, 1] #order: initial-external-vertex in \
                                     #   diagram order                                 
        for i in range(0, 6):
            self.assertEquals(self.t_drawing.VertexList[i]['level'], \
                              level_solution[i])    
    
    
    def test_find_initial_vertex_position(self):
        """ find a position to each vertex. all the vertex with the same level
            will have the same x coordinate. All external particules will be
            on the border of the square. The implementation of the code is 
            such that some external particles lines cuts sometimes some 
            propagator. This will be resolve in a second step 
        """
        
        self.mix_drawing.charge_diagram()
        self.mix_drawing.define_level()        
        self.mix_drawing.find_initial_vertex_position()
        
        x_position = [0, 0, 1, 1, 1, 1, 1 / 3, 1 / 3, 1 / 3, 2 / 3]
        y_position = [0, 1, 0, 1 / 3, 2 / 3, 1, 1 / 4, 1 / 2, 0.75, 0.5]
        for i in range(0, 10):
            self.assertEquals(self.mix_drawing.VertexList[i]['pos_x'], \
                              x_position[i])
            self.assertEquals(self.mix_drawing.VertexList[i]['pos_y'], \
                              y_position[i])                     
                            
                          
                          
                          
