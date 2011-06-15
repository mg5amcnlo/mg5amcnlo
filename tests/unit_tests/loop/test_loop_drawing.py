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

"""Unit test library for the various properties of objects in 
   loop_helas_objects.py"""

import copy
import itertools
import logging
import math
import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest


import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_drawing as loop_drawing
import madgraph.iolibs.save_load_object as save_load_object
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

#===============================================================================
# LoopDiagramDrawer Test
#===============================================================================
class LoopDiagramDrawerTest(unittest.TestCase):
    """Test class for all functions related to the LoopDiagramDrawer"""
    
    myloopmodel = loop_base_objects.LoopModel()
    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myproc = base_objects.Process()

    def setUp(self):
        """ Setup a toy-model with gluon and down-quark only """

        # A gluon
        self.mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark D and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'dmass',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[1])
        antid.set('is_part', False)

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon coupling to the down-quark
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)
        self.myproc.set('model',self.mymodel)
        
        self.myloopmodel = save_load_object.load_from_file(os.path.join(_input_file_path,\
                                                            'test_toyLoopModel.pkl'))

        box_diagram, box_struct = self.def_box()
        pent_diagram, pent_struct = self.def_pent()
        
        self.box_drawing = loop_drawing.LoopFeynmanDiagram(
                                box_diagram, box_struct, self.myloopmodel)

    def test_loop_load_diagram(self):
        """ check that we can load a NLO diagram """
        
        self.box_drawing.load_diagram()
        self.assertEqual(len(self.box_drawing.vertexList), 8)
        self.assertEqual(len(self.box_drawing.lineList), 8)
#        
#        self.triangle_drawing.load_diagram()
#        self.assertEqual(len(self.triangle_drawing.vertexList), 8)
#        self.assertEqual(len(self.triangle_drawing.lineList), 8)
#    
#        # Check T-channel information
#        is_t_channel = lambda line: line.state == False
#        self.assertEqual(len([1 for line in self.box_drawing.lineList if is_t_channel(line)]),3)
#        self.assertEqual(len([1 for line in self.triangle_drawing.lineList if is_t_channel(line)]),4)

    def test_fuse_line(self):
        """ check that we fuse line correctly """
        
        self.box_drawing.load_diagram()
        #avoid that element are erase from memory
        line1 = self.box_drawing.lineList[0]
        line2 = self.box_drawing.lineList[1]
        vertex1 = line1.begin
        vertex2 = line1.end
        vertex3 = line2.begin
        vertex4 = line2.end
        
        # fuse line1 and line2
        self.box_drawing.fuse_line(line1, line2)
        
        # check that all link to line1 are ok
        self.assertEqual(line1.begin, vertex1)
        self.assertEqual(line1.end, vertex3)
        self.assertTrue(line1 in vertex1.lines)
        self.assertTrue(line1 in vertex3.lines)
        #self.assertTrue(vertex1 in self.box_drawing.vertexList)
        #self.assertTrue(vertex4 in self.box_drawing.vertexList)

        
        #check that all info to line2 are deleted
        self.assertFalse(line2 in self.box_drawing.lineList)
        self.assertFalse(line2 in vertex1.lines)
        self.assertFalse(line2 in vertex3.lines)
        self.assertFalse(vertex2 in self.box_drawing.vertexList)
        self.assertFalse(vertex3 in self.box_drawing.vertexList)
        
    def test_define_level_nlo(self):
        """ test define level in the NLO case """
        
        # Check for the Box diagram
        self.box_drawing.load_diagram()
        self.box_drawing.define_level()
        #order: initial-external-vertex in diagram order                                 
        level_solution = [1, 1, 2, 2, 0, 0, 3, 3]
        number_of_line = [3, 3, 3, 3, 1, 1, 1, 1]
        # the ordering is not important but we test it anyway in order 
        # to ensure that we don't have any wrong permutation
        
        self.assertEqual(self.box_drawing.max_level, 3)
        self.assertEqual(self.box_drawing.min_level, 0)
        for i in range(0,8):
            #continue
            self.assertEquals(self.box_drawing.vertexList[i].level, \
                                                            level_solution[i])
            self.assertEquals(len(self.box_drawing.vertexList[i].lines), \
                                                            number_of_line[i])
        
        
        # Check for the triangle diagram
#        self.triangle_drawing.load_diagram()
#        self.triangle_drawing.define_level()
#        #order: initial-external-vertex in diagram order                                 
#        level_solution = [1, 2, 1, 1, 0, 3, 0, 0]
#        number_of_line = [3, 3, 3, 3, 1, 1, 1, 1]
#        # the ordering is not important but we test it anyway in order 
#        # to ensure that we don't have any wrong permutation
#        
#        self.assertEqual(self.triangle_drawing.max_level, 3)
#        self.assertEqual(self.triangle_drawing.min_level, 0)
#        for i in range(0, 8):
#            self.assertEquals(self.triangle_drawing.vertexList[i].level, \
#                                                            level_solution[i])
#            self.assertEquals(len(self.triangle_drawing.vertexList[i].lines), \
#                                                            number_of_line[i])
#        
#        # build an extension of the triangle
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        neg_diagram = base_objects.Diagram(self.neg_diagram_dict)  
#        neg_drawing = drawing.FeynmanDiagramNLO(neg_diagram, _model, opt)
#        neg_drawing.load_diagram()
#        neg_drawing.define_level()
#        self.assertEqual(neg_drawing.max_level, 4)
#        self.assertEqual(neg_drawing.min_level, -1)
#        level_solution = [1, 3, 2, 0, 1, 1, 0, 4, 4, -1, -1, 0]
#        number_of_line = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1]
#        for i in range(0, 11):
#            self.assertEquals(neg_drawing.vertexList[i].level, \
#                                                            level_solution[i])
#            self.assertEquals(len(neg_drawing.vertexList[i].lines), \
#                                                            number_of_line[i])
#        
#        # Check that the begin-end order is coherent for the negative particles
#        for line in neg_drawing.lineList:
#            if line.end.level > 0:
#                self.assertTrue(line.begin.level <= line.end.level, 
#                    'wrong level organization begin is level %s and end is level %s' \
#                    % (line.begin.level, line.end.level))
#            else:
#                self.assertTrue(line.begin.level >= line.end.level, 
#                    'wrong level organization begin is level %s and end is level %s' \
#                    % (line.begin.level, line.end.level))






    def def_box(self):
        """ Test the drawing of a simple loop box """
        
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num, 'state':True,
                                              'loop_line':False}) \
                                              for num in range(1, 5)])
        myleglist.append(base_objects.Leg({'id':1,'number':5,'loop_line':True}))
        myleglist.append(base_objects.Leg({'id':-1,'number':6,'loop_line':True}))                         
        l1=myleglist[0]
        l1.set('state',False)
        l2=myleglist[1]
        l2.set('state',False)
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]

        
        # One way of constructing this diagram, with a three-point amplitude
        l15 = base_objects.Leg({'id':1,'number':1,'loop_line':True, 'state':False})
        l12 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l13 = base_objects.Leg({'id':1,'number':1,'loop_line':True}) 
        

        vx15 = base_objects.Vertex({'legs':base_objects.LegList([l1, l5, l15]), 'id': 3})
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l15, l2, l12]), 'id': 3})
        vx13 = base_objects.Vertex({'legs':base_objects.LegList([l12, l3, l13]), 'id': 3})
        vx164 = base_objects.Vertex({'legs':base_objects.LegList([l13, l6, l4]), 'id': 3})
        ctvx = base_objects.Vertex({'legs':base_objects.LegList([l1, l2, l3, l4]), 'id': 666})

        myVertexList1=base_objects.VertexList([vx15,vx12,vx13,vx164])
        myCTVertexList=base_objects.VertexList([ctvx,])
        myPentaDiag1=loop_base_objects.LoopDiagram({'vertices':myVertexList1,'type':1,\
                                                    'CT_vertices':myCTVertexList})
        
        return myPentaDiag1, []

    def def_pent(self):       
        """ Test the gg>gggg d*dx* tagging of a quark pentagon which is tagged"""

        # Five gluon legs with two initial states
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 7)])
        myleglist.append(base_objects.Leg({'id':1,'number':7,'loop_line':True}))
        myleglist.append(base_objects.Leg({'id':-1,'number':8,'loop_line':True}))                         
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]
        l7=myleglist[6]
        l8=myleglist[7]

        # One way of constructing this diagram, with a three-point amplitude
        l17 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l12 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l68 = base_objects.Leg({'id':-1,'number':6,'loop_line':True}) 
        l56 = base_objects.Leg({'id':-1,'number':5,'loop_line':True})
        l34 = base_objects.Leg({'id':21,'number':3,'loop_line':False})

        self.myproc.set('legs',myleglist)

        vx17 = base_objects.Vertex({'legs':base_objects.LegList([l1, l7, l17]), 'id': 3})
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l17, l2, l12]), 'id': 3})
        vx68 = base_objects.Vertex({'legs':base_objects.LegList([l6, l8, l68]), 'id': 3})
        vx56 = base_objects.Vertex({'legs':base_objects.LegList([l5, l68, l56]), 'id': 3})
        vx34 = base_objects.Vertex({'legs':base_objects.LegList([l3, l4, l34]), 'id': 1})
        vx135 = base_objects.Vertex({'legs':base_objects.LegList([l12, l56, l34]), 'id': 3})

        myVertexList1=base_objects.VertexList([vx17,vx12,vx68,vx56,vx34,vx135])

        myPentaDiag1=loop_base_objects.LoopDiagram({'vertices':myVertexList1,'type':1})

        myStructRep=loop_base_objects.FDStructureList()
        
        myPentaDiag1.tag(myStructRep,7,8,self.myproc)
        
        return myPentaDiag1,myStructRep
        # test the drawing of myPentaDiag with its loop vertices and those in the 
        # structures of myStructRep
        

    def def_diagrams_epemddx(self):
        """ Test the drawing of diagrams from the loop process e+e- > dd~ """
    
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        # Now the drawing test on myloopamplitude['loop_diagrams']
        return myloopamplitude['loop_diagrams']
    
    def test_need_to_flip(self):
        """ test define level in the NLO case """
        
        # In some case, the diagram can be better if we permutte the T-channel 
        #part of the loop with the non-T channel part of the loop. Check the 
        #routine deciding if we need to flip or not.
        raise NotImplemented, 'Not know yet the structure of this routine'    

    def test_find_all_loop_particles(self):
        """ check if we can find the loop particles at a given position """
        
        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
        penta_diagram = base_objects.Diagram(self.penta_diagram_dict)  
        penta_drawing = drawing.FeynmanDiagramNLO(penta_diagram, _model, opt)
        
        penta_drawing.load_diagram()
        penta_drawing.define_level()
        
        #start position
        vertex_at_2 = [vertex for vertex in penta_drawing.vertexList \
                                                             if vertex.level==2]
        
        self.assertEqual(penta_drawing.find_all_loop_vertex(vertex_at_2[0]), \
                                                                    vertex_at_2)
        vertex_at_2.reverse()
        self.assertEqual(penta_drawing.find_all_loop_vertex(vertex_at_2[0]), \
                                                                    vertex_at_2)        
        
        
#    def do_draw(self):
#        """draw the diagrams for producing the plot associated to 
#        those tests"""
#        
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        penta_diagram = base_objects.Diagram(self.neg_diagram_dict)  
#        penta_drawing = drawing.FeynmanDiagramNLO(penta_diagram, _model, opt)
#        
#        penta_drawing.load_diagram()
#        penta_drawing.define_level()
#        penta_drawing.find_initial_vertex_position()
#        #diaglist = base_objects.DiagramList([penta_drawing])
#        plot = draw_eps.EpsDiagramDrawer(penta_drawing, \
#                                        '__testdiag3__.eps', model=_model, \
#                                         amplitude='')
#        plot.draw(opt)


    
##===============================================================================
## TestFeynmanDiagramLoop
##===============================================================================
#class TestFeynmanDiagramNLO(unittest.TestCase):
#    """Test the object which compute the position of the vertex/line 
#        for a given Diagram object with Loop
#    """
#
#    #test diagram gg>gg via a box Loop
#    leg1 = base_objects_nlo.LegNLO({'id':21, 'number':1, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg2 = base_objects_nlo.LegNLO({'id':21, 'number':2, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg3 = base_objects_nlo.LegNLO({'id':21, 'number':3, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg4 = base_objects_nlo.LegNLO({'id':21, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg5 = base_objects_nlo.LegNLO({'id':21, 'number':5, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg6 = base_objects_nlo.LegNLO({'id':21, 'number':6, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg7 = base_objects_nlo.LegNLO({'id':21, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg8 = base_objects_nlo.LegNLO({'id':21, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
#    leg9 = base_objects_nlo.LegNLO({'id':21, 'number':1, 'state':True,
#                            'inloop':True, 'from_group':True})
#    leg10 = base_objects_nlo.LegNLO({'id':21, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
##((1(21),3(21)>1(21),id:1),
##(4(21),5(21)>4(21),id:1),
##(1(21),2(21)>1(21),id:1),
##(4(21),6(21)>4(21),id:1),
##(1(21),4(21),id:0))
#
#    vertex1 = base_objects.Vertex({'id':1, \
#                        'legs':base_objects.LegList([leg1, leg3, leg7])})
#
#    vertex2 = base_objects.Vertex({'id':2, \
#                        'legs':base_objects.LegList([leg4, leg5, leg8])})
#
#    vertex3 = base_objects.Vertex({'id':3, \
#                        'legs':base_objects.LegList([leg7, leg2, leg9])})
#
#    vertex4 = base_objects.Vertex({'id':4, \
#                        'legs':base_objects.LegList([leg4, leg6, leg10])})
#    
#    vertex5 = base_objects.Vertex({'id':0, \
#                        'legs':base_objects.LegList([leg9, leg10])})
#
#    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3, vertex4, \
#                                                                      vertex5])
#    box_diagram_dict = {'vertices':vertexlist}
#
## Info for triangle box with backward
##40  ((1(21),3(21)>1(21),id:1),(4(21),5(21)>4(21),id:1),
##(1(21),6(21)>1(21),id:1),(2(21),4(21)>2(21),id:1),
##(1(21),2(21),id:0)) (QCD=4)
#
#    leg1 = base_objects_nlo.LegNLO({'id':1, 'number':1, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg2 = base_objects_nlo.LegNLO({'id':2, 'number':2, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg3 = base_objects_nlo.LegNLO({'id':3, 'number':3, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg4 = base_objects_nlo.LegNLO({'id':4, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg5 = base_objects_nlo.LegNLO({'id':5, 'number':5, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg6 = base_objects_nlo.LegNLO({'id':6, 'number':6, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg7 = base_objects_nlo.LegNLO({'id':-1, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg8 = base_objects_nlo.LegNLO({'id':-2, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
#    leg9 = base_objects_nlo.LegNLO({'id':-3, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg10 = base_objects_nlo.LegNLO({'id':-4, 'number':2, 'state':False,
#                            'inloop':True, 'from_group':True})
#    
#    vertex1 = base_objects.Vertex({'id':1, \
#                        'legs':base_objects.LegList([leg1, leg3, leg7])})    
#   
#    vertex2 = base_objects.Vertex({'id':2, \
#                        'legs':base_objects.LegList([leg4, leg5, leg8])})
#
#    vertex3 = base_objects.Vertex({'id':3, \
#                        'legs':base_objects.LegList([leg7, leg6, leg9])})
#
#    vertex4 = base_objects.Vertex({'id':4, \
#                        'legs':base_objects.LegList([leg2, leg8, leg10])})
#    
#    vertex5 = base_objects.Vertex({'id':0, \
#                        'legs':base_objects.LegList([leg9, leg10])})
#
#    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3, vertex4, \
#                                                                      vertex5])
#    triangle_diagram_dict = {'vertices':vertexlist} 
#    
#    # Doulby Extended (both outgoing particles decays)
#    # Check the possibility to go to negative number
#    leg1 = base_objects_nlo.LegNLO({'id':1, 'number':1, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg2 = base_objects_nlo.LegNLO({'id':2, 'number':2, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg3 = base_objects_nlo.LegNLO({'id':3, 'number':3, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg4 = base_objects_nlo.LegNLO({'id':4, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg5 = base_objects_nlo.LegNLO({'id':5, 'number':5, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg6 = base_objects_nlo.LegNLO({'id':6, 'number':6, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg7 = base_objects_nlo.LegNLO({'id':11, 'number':7, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg8 = base_objects_nlo.LegNLO({'id':12, 'number':8, 'state':True,
#                            'inloop':False, 'from_group':False})
#
#    leg9 = base_objects_nlo.LegNLO({'id':-1, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg10 = base_objects_nlo.LegNLO({'id':-2, 'number':5, 'state':True,
#                            'inloop':False, 'from_group':True})
#    leg11 = base_objects_nlo.LegNLO({'id':-3, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
#    leg12 = base_objects_nlo.LegNLO({'id':-4, 'number':6, 'state':True,
#                            'inloop':False, 'from_group':True})
#    leg13 = base_objects_nlo.LegNLO({'id':-5, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg14 = base_objects_nlo.LegNLO({'id':-5, 'number':2, 'state':False,
#                            'inloop':True, 'from_group':True})
#    
#    vertex1 = base_objects.Vertex({'id':1, \
#                        'legs':base_objects.LegList([leg1, leg3, leg9])})    
#    vertex2 = base_objects.Vertex({'id':2, \
#                        'legs':base_objects.LegList([leg5, leg8, leg10])})
#    vertex3 = base_objects.Vertex({'id':3, \
#                        'legs':base_objects.LegList([leg10, leg4, leg11])})
#    vertex4 = base_objects.Vertex({'id':4, \
#                        'legs':base_objects.LegList([leg7, leg6, leg12])})   
#    vertex5 = base_objects.Vertex({'id':5, \
#                        'legs':base_objects.LegList([leg9, leg12, leg13])})
#    vertex6 = base_objects.Vertex({'id':6, \
#                        'legs':base_objects.LegList([leg2, leg11, leg14])})
#    vertex7 = base_objects.Vertex({'id':0, \
#                        'legs':base_objects.LegList([leg13, leg14])})
#
#    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3, vertex4, \
#                                                     vertex5, vertex6, vertex7])
#    neg_diagram_dict = {'vertices':vertexlist}
#    
#    
#    #The pentagone 
#    leg1 = base_objects_nlo.LegNLO({'id':1, 'number':1, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg2 = base_objects_nlo.LegNLO({'id':2, 'number':2, 'state':False,
#                            'inloop':False, 'from_group':False})
#    leg3 = base_objects_nlo.LegNLO({'id':3, 'number':3, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg4 = base_objects_nlo.LegNLO({'id':4, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':False})
#    leg5 = base_objects_nlo.LegNLO({'id':5, 'number':5, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg6 = base_objects_nlo.LegNLO({'id':6, 'number':6, 'state':True,
#                            'inloop':False, 'from_group':False})
#    leg7 = base_objects_nlo.LegNLO({'id':11, 'number':7, 'state':True,
#                            'inloop':False, 'from_group':False})
#    
#    #((1(21),3(21)>1(21),id:1),(4(21),5(21)>4(21),id:1),(1(21),2(21)>1(21),id:1),
#    #(4(21),7(21)>4(21),id:1),(1(21),4(21),6(21),id:1)) (QCD=5)  
#    leg8 = base_objects_nlo.LegNLO({'id':-1, 'number':1, 'state':False,
#                            'inloop':True, 'from_group':True})
#    leg9 = base_objects_nlo.LegNLO({'id':-2, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
#    leg10 = base_objects_nlo.LegNLO({'id':-2, 'number':1, 'state':True,
#                            'inloop':True, 'from_group':True})     
#    leg11 = base_objects_nlo.LegNLO({'id':-3, 'number':4, 'state':True,
#                            'inloop':True, 'from_group':True})
#
#    #
#    vertex1 = base_objects.Vertex({'id':1, \
#                        'legs':base_objects.LegList([leg1, leg3, leg8])})    
#    vertex2 = base_objects.Vertex({'id':2, \
#                        'legs':base_objects.LegList([leg4, leg5, leg9])})
#    vertex3 = base_objects.Vertex({'id':3, \
#                        'legs':base_objects.LegList([leg8, leg2, leg10])})
#    vertex4 = base_objects.Vertex({'id':4, \
#                        'legs':base_objects.LegList([leg9, leg7, leg11])})   
#    vertex5 = base_objects.Vertex({'id':5, \
#                        'legs':base_objects.LegList([leg10, leg11, leg6])})
#    
#    vertexlist = base_objects.VertexList([vertex1, vertex2, vertex3, vertex4, \
#                                                                       vertex5])
#    penta_diagram_dict = {'vertices':vertexlist}
#       
#    def setUp(self):
#        """ basic construction """
#        
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        # gg>g(g>uux)g (via a T channel)
#        box_diagram = base_objects.Diagram(self.box_diagram_dict)  
#        self.box_drawing = drawing.FeynmanDiagramNLO(box_diagram, _model, opt)
#
#        triangle_diagram = base_objects.Diagram(self.triangle_diagram_dict)  
#        self.triangle_drawing = drawing.FeynmanDiagramNLO(triangle_diagram, _model, opt)    
#    
#    
#    def test_find_initial_vertex_position_for_neg(self):
#        """Test if we can correctly set the position with loop"""
#        
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        neg_diagram = base_objects.Diagram(self.neg_diagram_dict)  
#        neg_drawing = drawing.FeynmanDiagramNLO(neg_diagram, _model, opt)
#        neg_drawing.load_diagram()
#        neg_drawing.define_level()
#        neg_drawing.find_initial_vertex_position()
#        
#        level = [1, 3, 2, 0, 1, 1, -1, 4, 4, -1, -1, -1]
#        x_position = [(l+1)/5 for l in level]
#        y_position = [1/6, 1/2, 1/2, 1/2, 1/2, 5/6, 0.0, 0.0, 1.0, 1/4, 3/4, 1.0]
#                                                    
#
#        for i in range(len(level)):
#            self.assertAlmostEquals(neg_drawing.vertexList[i].pos_x, \
#                              x_position[i])
#            self.assertAlmostEquals(neg_drawing.vertexList[i].pos_y, \
#                              y_position[i])
#            
#    def test_find_initial_vertex_position_for_s_loop(self):
#        """Test if we can correctly set the position with loop"""
#        
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        penta_diagram = base_objects.Diagram(self.penta_diagram_dict)  
#        penta_drawing = drawing.FeynmanDiagramNLO(penta_diagram, _model, opt)
#        
#        penta_drawing.load_diagram()
#        penta_drawing.define_level()
#        penta_drawing.find_initial_vertex_position()
#
#        level = [1, 2, 1, 2, 2, 0, 3, 0, 3, 3]
#        x_position = [(l)/3 for l in level]
#        y_position = [0.25, 1/6, 0.75, 0.5, 5/6, 0, 0, 1, 0.5, 1]
#        
#        for i in range(len(level)):
#            self.assertAlmostEquals(penta_drawing.vertexList[i].pos_x, \
#                              x_position[i])
#            self.assertAlmostEquals(penta_drawing.vertexList[i].pos_y, \
#                              y_position[i])
#    
     
#        
#        
#    def do_draw(self):
#        """draw the diagrams for producing the plot associated to 
#        those tests"""
#        
#        opt = drawing.DrawOption({'external':1, 'horizontal':0, 'max_size':0})
#        penta_diagram = base_objects.Diagram(self.neg_diagram_dict)  
#        penta_drawing = drawing.FeynmanDiagramNLO(penta_diagram, _model, opt)
#        
#        penta_drawing.load_diagram()
#        penta_drawing.define_level()
#        penta_drawing.find_initial_vertex_position()
#        #diaglist = base_objects.DiagramList([penta_drawing])
#        plot = draw_eps.EpsDiagramDrawer(penta_drawing, \
#                                        '__testdiag3__.eps', model=_model, \
#                                         amplitude='')
#        plot.draw(opt)
        