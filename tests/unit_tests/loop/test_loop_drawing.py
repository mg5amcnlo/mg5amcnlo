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
import madgraph.loop.loop_helas_objects as loop_helas_objects
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

    def test_draw_box(self):
        """ Test the drawing of a simple loop box """
        
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 5)])
        myleglist.append(base_objects.Leg({'id':1,'number':5,'loop_line':True}))
        myleglist.append(base_objects.Leg({'id':-1,'number':6,'loop_line':True}))                         
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]

        
        # One way of constructing this diagram, with a three-point amplitude
        l15 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
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
        
        options = draw_lib.DrawOption()
        filename = os.path.join('/Users/Spooner/Documents/PhD/MG5/NLO', 'loopdiagTrial.jpg')
        # plot = draw.MultiEpsDiagramDrawer(myPentaDiag1,
        #                                  filename,
        #                                  amplitude='',
        #                                  model=self.myloopmodel,
        #                                  legend='loopTest')
        # plot.draw(opt=options)
        
        # Now the drawing test on myPentaDiag
        pass

    def test_draw_pentagon(self):    
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

        # test the drawing of myPentaDiag with its loop vertices and those in the 
        # structures of myStructRep
        

    def test_draw_diagrams_epemddx(self):
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
        pass