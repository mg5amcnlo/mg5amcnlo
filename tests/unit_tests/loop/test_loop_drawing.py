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

"""Unit test library for the various properties of objects in 
   loop_helas_objects.py"""
from __future__ import division
import copy
import itertools
import logging
import math
import os
import pickle
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest


import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.iolibs.save_load_object as save_load_object
from madgraph.interface.master_interface import MasterCmd
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
pjoin = os.path.join

#===============================================================================
# LoopDiagramDrawer Test
#===============================================================================
class TestLoopDrawer(unittest.TestCase):
    """ Test class for all functions related to the LoopDiagramDrawer """

    def setUp(self):
        if not hasattr(self, 'cmd'):
            TestLoopDrawer.cmd = MasterCmd()
            model_path = pjoin(_file_path, '../../input_files','LoopSMTest')
            TestLoopDrawer.cmd.do_import('model %s' % model_path)
            TestLoopDrawer.model = TestLoopDrawer.cmd._curr_model
            try:
                TestLoopDrawer.store_diagram = pickle.load(open(os.path.join(_file_path, \
                                            '../../input_files/test_draw_nlo.obj'), 'r'))
            except Exception, error:
                print error
                pass

    class FakeAMP(dict):
        
        def get(self, name):
            return self[name]
    
    def assertnozerolength(self, diagram):
        """check that all line have a non zero length"""
        
        for line in diagram.lineList:
            a = (line.begin.pos_x, line.begin.pos_y ) 
            b= (line.end.pos_x, line.end.pos_y )
            self.assertNotEqual(a,b)

    def assertnocrossing(self, diagram):
        """check that all line have a non zero length"""
        
        for i, line1 in enumerate(diagram.lineList):
            if i == len(diagram.lineList):
                continue
            for line2 in diagram.lineList[i+1:]:
                if line1.begin in [line2.end, line2.begin] and line1.end in [line2.end, line2.begin]:
                    continue
                
                
                self.assertFalse(line1.has_intersection(line2),'(%s,%s),(%s,%s) with (%s,%s),(%s,%s)' %\
                                 (line1.begin.pos_x, line1.begin.pos_y,
                                  line1.end.pos_x, line1.end.pos_y,
                                  line2.begin.pos_x, line2.begin.pos_y,
                                  line2.end.pos_x, line2.end.pos_y) )

            

    def test_loop_convert_diagram(self):
        """check that the drawer assign the correct Drawing-class"""
        
        lo_diagram = self.store_diagram['g g > g g'][0]
        amp = self.FakeAMP({'structure_repository': self.store_diagram['g g > g g']['structure']})
        drawer = draw_lib.DiagramDrawer()
        new_diagram = drawer.convert_diagram(diagram=lo_diagram, 
                               model=self.model, 
                               amplitude=amp)
        
        self.assertEqual(new_diagram.__class__.__name__, 'FeynmanDiagram')
        
        nlo_diagram = self.store_diagram['g g > g g'][12]
            
        new_diagram = drawer.convert_diagram(diagram=nlo_diagram, 
                               model=self.model, 
                               amplitude=amp)
        
        self.assertEqual(new_diagram.__class__.__name__, 'LoopFeynmanDiagram')        
        
    def test_LO_draw_with_NLO_generation(self):
        """ check if we can make the drawing """
    
        # test that LO is still fine when generate via NLO: 4 point
        diagram = self.store_diagram['g g > g g'][0]
        diagram = draw_lib.FeynmanDiagram(diagram, self.model)

        diagram.load_diagram()
        diagram.define_level()
        level_solution = [1, 0, 0, 2, 2]
        found = [v.level for v in diagram.vertexList]
        self.assertEqual(level_solution, found)

        diagram.find_initial_vertex_position()
        level_solution = [1, 0, 0, 2, 2]
        x_position = [1 / 2, 0, 0, 1, 1]
        y_position = [1 / 2, 0, 1, 0, 1]
        self.assertEquals(len(diagram.vertexList), 5)
        for i in range(0, 5):
            self.assertEquals(diagram.vertexList[i].level, \
                              level_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_x, \
                              x_position[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_y, \
                              y_position[i])
        for line in diagram.lineList:
            self.assertNotEquals(line.begin, None)
            self.assertNotEquals(line.end, None)

        # test that LO is still fine when generate via NLO: S-channel
        diagram = self.store_diagram['g g > g g'][1]
        diagram = draw_lib.FeynmanDiagram(diagram, self.model)

        diagram.load_diagram()
        diagram.define_level()
        level_solution = [1, 2, 0, 0, 3, 3]
        found = [v.level for v in diagram.vertexList]
        self.assertEqual(level_solution, found)

        diagram.find_initial_vertex_position()
        level_solution = [1, 2, 0, 0, 3, 3]
        x_position = [1/3, 2/3, 0, 0, 1, 1]
        y_position = [1/2, 1/2, 0, 1, 0, 1]
        self.assertEquals(len(diagram.vertexList), 6)


    def test_NLO_draw(self):
        """ check if we can make the drawing """
    
        # test that NLO DRAW is fine first check for diagram
        # +        +
        #  +   + 
        #   +
        #   I
        #   I              +
        #   +             +
        #   I +          +
        #   I  +        +
        #   I   +------
        #   I  +        +
        #   I +          +
        #   +              +
        #  +                +
        # +                  +
        
        diagram = self.store_diagram['g g > g g'][12]
        structure = self.store_diagram['g g > g g']['structure']
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        

        diagram.load_diagram()
        diagram.define_level()


        level_solution = [1, 1, 1, 2, 0, 2, 0, 3]
        level = [v.level for v in diagram.vertexList]
        level_solution.sort()
        level.sort()
        self.assertEqual(level, level_solution)        
        diagram.find_initial_vertex_position()
        
        level_solution = [1, 1, 2, 1, 0, 3, 0, 3]
        level = [v.level for v in diagram.vertexList]
        self.assertEqual(level, level_solution)
        x_solution = [1/3, 1/3, 2/3, 1/3, 0, 1, 0, 1]
        y_solution = [1/6, 5/6, 3/4, 1/2, 0, 0, 1, 1]
        
        self.assertEquals(len(diagram.vertexList), 8)
        
        for i in range(0, 8):
            self.assertEquals(diagram.vertexList[i].level, \
                              level_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_x, \
                              x_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_y, \
                              y_solution[i])
        for line in diagram.lineList:
            self.assertNotEquals(line.begin, None)
            self.assertNotEquals(line.end, None)

    def test_flipping(self):
        """ check if the flipping of loop-line work"""
        
        
        ### Box cut at the T-channel
        #diagram = copy.deepcopy(self.store_diagram['g g > g g'][60])
        structure = self.store_diagram['g g > g g']['structure']
        #diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
            
        #self.assertTrue(diagram.need_to_flip())
        #diagram.load_diagram()
        #diagram.loop_flip()
        #nb_t = len([1 for l in diagram.lineList if l.state and l.loop_line])
        #self.assertEqual(nb_t,3)
        #self.assertFalse(diagram.need_to_flip())
        
        ### Triangle
        diagram = copy.deepcopy(self.store_diagram['g g > g g'][8])
        
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
#        diagram.load_diagram()
        self.assertTrue(diagram.need_to_flip())
        
        ### Triangle
        diagram = copy.deepcopy(self.store_diagram['FULL g g > g g'][202])
        
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
#        diagram.load_diagram()
        self.assertFalse(diagram.need_to_flip())        
        # Same test but for d d~ > e+ e- mu+ mu-:
        diagram = copy.deepcopy(self.store_diagram['FULL d d~ > e+ e- mu+ mu-'][42])
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        diagram.load_diagram()
        diagram.define_level()
        self.assertTrue(diagram.need_to_flip())
        
    def test_level_with_flipping_triangle(self):

        diagram = copy.deepcopy(self.store_diagram['g g > g g'][8])
        structure = self.store_diagram['g g > g g']['structure']
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        diagram.load_diagram()
        self.assertTrue(diagram.need_to_flip())

        # check the position for this diagram
        diagram.define_level()
        level_solution = [1, 1, 2, 1, 0, 2, 0, 3]
        level = [v.level for v in diagram.vertexList]
        self.assertEqual(level, level_solution)        
        
        diagram.find_initial_vertex_position()
        
        level_solution = [1, 1, 2, 1, 0, 3, 0, 3]
        level = [v.level for v in diagram.vertexList]
        self.assertEqual(level, level_solution)
        x_solution = [1/3, 1/3, 2/3, 1/3, 0, 1, 0, 1]
        y_solution = [1/6, 5/6, 3/4, 1/2, 0, 0, 1, 1]
        
        self.assertEquals(len(diagram.vertexList), 8)
        for i in range(0, 8):
            self.assertEquals(diagram.vertexList[i].level, \
                              level_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_x, \
                              x_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_y, \
                              y_solution[i])
        for line in diagram.lineList:
            self.assertNotEquals(line.begin, None)
            self.assertNotEquals(line.end, None)
        
        

        

    # The test below is 'commented' out as it is no longer compatible with 
    # the diagram ordering obtained from the new loop model and the old format
    # is no longer supported. I therefore need to be updated.
    def no_test_special_gg_gg(self):
        
        diagram = self.store_diagram['g g > g g'][75]
        structure = self.store_diagram['g g > g g']['structure']
        diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        
        diagram.load_diagram()
        self.assertFalse(diagram.need_to_flip())
        
        # check the position for this diagram
        diagram.define_level()
        level_solution = [2, 1, 1, 3, 3, 0, 0]
        level = [v.level for v in diagram.vertexList]
        self.assertEqual(level, level_solution)        
        
        diagram.find_initial_vertex_position()
        
        level = [v.level for v in diagram.vertexList]
        self.assertEqual(level, level_solution)
        x_solution = [2/3, 1/3, 1/3, 1, 1, 0, 0]
        y_solution = [1/2, 3/4, 1/4, 0, 1, 1, 0]
        
        self.assertEquals(len(diagram.vertexList), 7)
        for i in range(0, 7):
            self.assertEquals(diagram.vertexList[i].level, \
                              level_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_x, \
                              x_solution[i])
            self.assertAlmostEquals(diagram.vertexList[i].pos_y, \
                              y_solution[i])
        for line in diagram.lineList:
            self.assertNotEquals(line.begin, None)
            self.assertNotEquals(line.end, None)

        # check the associate position
        
        diagram.find_initial_vertex_position()
                
    def test_NLO_draw_all_gg_gg(self):
        for i in range(5,85):
            diagram = copy.deepcopy(self.store_diagram['g g > g g'][i])
            structure = self.store_diagram['g g > g g']['structure']
            diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        

            diagram.load_diagram()
            diagram.define_level()
            diagram.find_initial_vertex_position()
            self.assertnozerolength(diagram)

    def test_NLO_draw_gg_ggg(self):
        for i in range(1500,1600):
            diagram = copy.deepcopy(self.store_diagram['g g > g g g'][i])
            structure = self.store_diagram['g g > g g g']['structure']
            diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        
            diagram.load_diagram()
            diagram.define_level()
            diagram.find_initial_vertex_position()
            self.assertnozerolength(diagram)
            self.assertFalse(diagram._debug_has_intersection())

        

    def test_NLO_draw_all_reconstructed_gg_gg(self):
               
        for i in range(200,230):
            diagram = copy.deepcopy(self.store_diagram['FULL g g > g g'][i])
            Drawer = draw_lib.DiagramDrawer()
            diagram = Drawer.convert_diagram(diagram, self.model) 
            if diagram is None:
                continue # counter term are not drawn
            self.assertnozerolength(diagram)
            self.assertnocrossing(diagram)
            
    def test_NLO_draw_all_reconstructed_dd_eemm(self):
               
        for i in range(72):
            diagram = copy.deepcopy(self.store_diagram['FULL d d~ > e+ e- mu+ mu-'][i])
            Drawer = draw_lib.DiagramDrawer()
            diagram = Drawer.convert_diagram(diagram, self.model) 
            if diagram is None:
                continue # counter term are not drawn
            self.assertnozerolength(diagram)
            self.assertnocrossing(diagram)

    def test_NLO_draw_uux_guux(self):
        for i in range(51,52):
            diagram = copy.deepcopy(self.store_diagram['u u~ > u u~ g'][i])
            structure = self.store_diagram['u u~ > u u~ g']['structure']
            diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        
            diagram.load_diagram()
            diagram.define_level()
            diagram.find_initial_vertex_position()
            self.assertnozerolength(diagram)

    def test_NLO_draw_uux_uuxddx(self):
        for i in range(139,140):
            diagram = copy.deepcopy(self.store_diagram['u u~ > u u~ d d~'][i])
            structure = self.store_diagram['u u~ > u u~ d d~']['structure']
            diagram = draw_lib.LoopFeynmanDiagram(diagram, structure, self.model)
        
            diagram.load_diagram()
            diagram.define_level()
            diagram.find_initial_vertex_position()
            self.assertnozerolength(diagram)
                        

class LoopDiagramDrawerTest(unittest.TestCase):
    """Test class for all functions related to the LoopDiagramDrawer
        diagram made by hand
    """
    
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
       
        self.box_drawing = draw_lib.LoopFeynmanDiagram(
                                box_diagram, box_struct, self.myloopmodel)

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
        lfake = base_objects.Leg({'id':1,'number':1,'loop_line':True})         

        vx15 = base_objects.Vertex({'legs':base_objects.LegList([l1, l5, l15]), 'id': 3})
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l15, l2, l12]), 'id': 3})
        vx13 = base_objects.Vertex({'legs':base_objects.LegList([l12, l3, l13]), 'id': 3})
        vx164 = base_objects.Vertex({'legs':base_objects.LegList([l13, l6, l4]), 'id': 3})
        fakevx = base_objects.Vertex({'legs':base_objects.LegList([l13, lfake]), 'id': 0})
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
        
        myPentaDiag1.tag(myStructRep, self.myproc['model'],7,8)
        
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
        
if __name__ == '__main__':

    # For debugging it's interesting to store problematic diagram in one file.
    #Those one are generated with cmd and store in files with pickle module.
    model_path = pjoin(_file_path, '../../input_files','LoopSMTest')
    process_diag = {}
    process_diag['g g > g g'] = range(85)#[0, 12]
    process_diag['g g > g g g'] = range(1500, 1600)
    process_diag['u u~ > u u~ g'] =[51]
    process_diag['u u~ > u u~ d d~'] = [139]
    cmd = MasterCmd()
    cmd.do_import('model %s' % model_path )
    # Create the diagrams
    diag_content = {}
    for gen_line, pos_list in process_diag.items():
        print gen_line, ':',
        gen_line_with_order = gen_line + ' [virt=QCD]'
        cmd.do_generate(gen_line_with_order)
        #Look for decay chains
        amplitude = cmd._curr_amps[0]
        print len(amplitude.get('diagrams'))
        diag_content[gen_line] = {}
        diag_content[gen_line]['structure'] = amplitude.get('structure_repository')
        for pos in pos_list:
            diag_content[gen_line][pos] = amplitude.get('diagrams')[pos]

    # register the full amplitutde
    process_diag = {}
    process_diag['g g > g g'] = range(200,230)#[0, 12]
    process_diag['d d~ > e+ e- mu+ mu-'] = range(72)
    cmd = MasterCmd()
    cmd.do_import('model %s' % model_path )
    # Create the diagrams
    for gen_line, pos_list in process_diag.items():
        print gen_line, ':',
        gen_line_with_order = gen_line + ' [virt=QCD]'
        cmd.do_generate(gen_line_with_order)
        #Look for decay chains
        amplitude = cmd._curr_amps[0]
        print len(amplitude.get('diagrams'))
        matrix_elements = loop_helas_objects.LoopHelasProcess(cmd._curr_amps)
        matrix_element = matrix_elements.get('matrix_elements')[0]
        diag = matrix_element.get_base_amplitude().get('diagrams')
        diag_content['FULL %s' %gen_line] = {}
        for pos in pos_list:
            diag_content['FULL %s' %gen_line][pos] = diag[pos]

        

    # Store the diagrams  
    file_test_diagram = open(os.path.join(_file_path , \
                                    '../../input_files/test_draw_nlo.obj'), 'w')
    pickle.dump(diag_content, file_test_diagram)
    print 'done'
        
        
        
        
        
