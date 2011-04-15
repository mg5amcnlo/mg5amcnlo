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

"""Testing modules for FKS_process class"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.core.fks as fks
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import copy


class TestFKSProcess(unittest.TestCase):
    """a class to test FKS Processes"""
    myleglist = MG.LegList()
    # PROCESS: u u~ > d d~ g g 
    mylegs = [{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': -2,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 1,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': -1,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 6,\
    'state': True,\
   # 'from_group': True\
}
]

    for i in mylegs:
        myleglist.append(MG.Leg(i))

    myleglist2 = MG.LegList()
    # PROCESS: u u~ > u u~ g  
    mylegs2 = [{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': -2,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': -2,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
   # 'from_group': True\
}
]
    for i in mylegs2:
        myleglist2.append(MG.Leg(i))

    
    mypartlist = MG.ParticleList()
    mypartlistbad = MG.ParticleList()
    myinterlist = MG.InteractionList()
    myinterlistbad = MG.InteractionList()
    mypartlist.append(MG.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  #'is_part': True,
                  'propagating':True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'g',
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

    mypartlist.append(MG.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
    
    mypartlist.append(MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
        
    antiu = MG.Particle({'name':'u~',
                  'antiname':'u',
                  'spin':2,
                  'color': 3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':  2. / 3.,
                  'pdg_code': 2,
                  'propagating':True,
                  'is_part':False,
                  'self_antipart':False})
  #  mypartlist.append(antiu)

    
    antid = MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'is_part': False,
                  'propagating':True,
                  'self_antipart':False})
    
    antit = MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  'is_part': False,
                  'self_antipart':False})
    
    
    myinterlist.append(MG.Interaction({\
                      'id':1,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
        
    myinterlist.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    myinterlist.append(MG.Interaction({\
                      'id':3,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[2]] *3 \
                                             ),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
 #   mymodel.set('couplings', {(0, 0):'GQQ'})
 #   mymodel.set('lorentz', ['L1'])

    #mymodel.set('couplings',{(0, 0):'GQQ'} )

    
    dict = {'legs' : myleglist, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2 = {'legs' : myleglist2, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    myproc = MG.Process(dict)

    myproc2 = MG.Process(dict2)
    

#    print str(myproc)
    
    def test_fks_process(self):

        pass
    
    def test_combine_ij(self):
        """tests wether leg i and j are correctly merged into leg ij 
        (if exists)"""
        list_i = []
        list_j = []
        list_ij = []
        
        ###configurations that should get an output
        # final u u~ should combine to a final gluon
        list_i.append(MG.Leg({'id' : -1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 21, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
       
        # initial and final u should combine to an initial gluon
        list_i.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':False,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 21, 
                              'number': 4, 
                              'state':False,
                              'from_group': True}))

        # initial and final u~ should combine to an initial gluon
        list_i.append(MG.Leg({'id' : -1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : -1, 
                              'number': 5, 
                              'state':False,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 21, 
                              'number': 4, 
                              'state':False,
                              'from_group': True}))
        
        # final g and final u should combine to a final u
        list_i.append(MG.Leg({'id' : 21, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        
        # final g and initial u should combine to an initial u
        list_i.append(MG.Leg({'id' : 21, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':False,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':False,
                              'from_group': True}))
        
        # final u and initial g should combine to an initial u
        list_i.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 21, 
                              'number': 5, 
                              'state':False,
                              'from_group': True}))
        list_ij.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':False,
                              'from_group': True}))
        
        #configurations that should NOT get an output
        # final u d~ 
        list_i.append(MG.Leg({'id' : -1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 2, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(None)
        
        # final u u 
        list_i.append(MG.Leg({'id' : 1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(None)
        
        # final massive pair (t t~) 
        list_i.append(MG.Leg({'id' : 6, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : -6, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(None)

        # any colorless particle
        list_i.append(MG.Leg({'id' : 22, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 5, 
                              'state':True,
                              'from_group': True}))
        list_ij.append(None)           
        
        #add your favourite configuration here...     
        fks_proc = fks.FKSProcess(self.myproc)
        for i in list_i:
            j = list_j[list_i.index(i)]
            ij = list_ij[list_i.index(i)]
            my_ij = fks_proc.combine_ij(i,j)
            self.assertEqual(ij, my_ij)
    
    
    def test_reduce_real_process(self):
        """tests whether the process is correctly reduced, removing leg i and j
        and inserting leg ij"""
        fksproc = fks.FKSProcess(self.myproc)
        
        list_i = []
        list_j = []
        list_LEGS_reduced = []
        
        list_i.append(MG.Leg({'id' : -1, 
                              'number': 4, 
                              'state':True,
                              'from_group': True}))
        list_j.append(MG.Leg({'id' : 1, 
                              'number': 3, 
                              'state':True,
                              'from_group': True}))
        list_LEGS_reduced.append([{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
    'from_group': True \
}, \
{ \
    'id': -2,\
    'number': 2,\
    'state': False,\
    'from_group': True\
},\
{\
    'id': 21,\
    'number': 3,\
    'state': True,\
    'from_group': True\
},\
{\
    'id': 21,\
    'number': 4,\
    'state': True,\
    'from_group': True\
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
    'from_group': True\
}])
        
        for i in list_i:
            j = list_j[list_i.index(i)]
            red = list_LEGS_reduced[list_i.index(i)]
            ij = fksproc.combine_ij(i,j)
            leglist_orig = copy.deepcopy(self.myproc.get('legs'))
            my_red = fksproc.reduce_real_process(i,j, ij, 1)
# check thar the replacement occours
            self.assertEqual(red, my_red['process']['legs'])
# check that the original process is left as it was
            self.assertEqual(leglist_orig, self.myproc['legs'])
# just an extra (in)-sanity check
            self.assertEqual(len(my_red['process']['legs']) + 1, \
                             len(leglist_orig))
#check that the QCD order is lowered
            self.assertEqual(self.myproc['orders']['QCD'],\
                             my_red['process']['orders']['QCD'] +1)
            self.assertEqual(my_red['i_fks'], 4)
            self.assertEqual(my_red['j_fks'], 3)
            self.assertEqual(my_red['ij_fks'], ij)
            self.assertEqual(my_red['conf_num'], 1)
            
    def test_find_directorites(self): 
        """test whether all the born subdirectories for a fks process are found"""
        # test u u~ > d d~ g g  
        fksproc = fks.FKSProcess(self.myproc)
        #test u u ~ > u u~ g
        fksproc2 = fks.FKSProcess(self.myproc2)
        # arrange all the FKS subprocesses in a dictionary of the form
        #  (i,j): process string
        subproc_dict = {}
        subproc_dict2 = {}
        for proc in fksproc.reduced_processes:
            subproc_dict[(proc['i_fks'], proc['j_fks'])] = \
            proc['process'].input_string()
        for proc in fksproc2.reduced_processes:
            subproc_dict2[(proc['i_fks'], proc['j_fks'])] = \
            proc['process'].input_string()
        
        check_dict = {}
        check_dict[(5,1)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(6,1)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(5,2)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(6,2)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(4,3)] = 'u u~ > g g g QED=0 QCD=9'
        check_dict[(5,3)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(6,3)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(5,4)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(6,4)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(5,6)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict[(6,5)] = 'u u~ > d d~ g QED=0 QCD=9'
        check_dict2 = {}
        check_dict2[(3,1)] = 'g u~ > u~ g QED=0 QCD=9'
        check_dict2[(5,1)] = 'u u~ > u u~ QED=0 QCD=9'
        check_dict2[(4,2)] = 'u g > u g QED=0 QCD=9'
        check_dict2[(5,2)] = 'u u~ > u u~ QED=0 QCD=9'
        check_dict2[(4,3)] = 'u u~ > g g QED=0 QCD=9'
        check_dict2[(5,3)] = 'u u~ > u u~ QED=0 QCD=9'
        check_dict2[(5,4)] = 'u u~ > u u~ QED=0 QCD=9'
        self.assertEqual(check_dict, subproc_dict)
        self.assertEqual(check_dict2, subproc_dict2)
    
            

class TestFKSDirectory(unittest.TestCase):
    """a class to test FKS directories"""
    
    myleglist = MG.LegList()
    # PROCESS: u u~ > d d~ g  
    mylegs = [{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
 #   'from_group': True \
}, \
{ \
    'id': -2,\
    'number': 2,\
    'state': False,\
    #'from_group': True\
},\
{\
    'id': 1,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': -1,\
    'number': 4,\
    'state': True,\
   # 'from_group': True\
},\
{\
    'id': 21,\
    'number': 5,\
    'state': True,\
   # 'from_group': True\
}
]

    for i in mylegs:
        myleglist.append(MG.Leg(i))
    
    mypartlist = MG.ParticleList()
    mypartlistbad = MG.ParticleList()
    myinterlist = MG.InteractionList()
    myinterlistbad = MG.InteractionList()
    mypartlist.append(MG.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  #'is_part': True,
                  'propagating':True,
                  'self_antipart':False}))
    mypartlist.append(MG.Particle({'name':'g',
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

    mypartlist.append(MG.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
    
    mypartlist.append(MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  #'is_part': True,
                  'self_antipart':False}))
        
    antiu = MG.Particle({'name':'u~',
                  'antiname':'u',
                  'spin':2,
                  'color': 3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\\overline{u}',
                  'line':'straight',
                  'charge':  2. / 3.,
                  'pdg_code': 2,
                  'propagating':True,
                  'is_part':False,
                  'self_antipart':False})
  #  mypartlist.append(antiu)

    
    antid = MG.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\\overline{d}',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'is_part': False,
                  'propagating':True,
                  'self_antipart':False})
    
    antit = MG.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'tmass',
                  'width':'twidth',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  'is_part': False,
                  'self_antipart':False})
    
    
    myinterlist.append(MG.Interaction({\
                      'id':1,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
        
    myinterlist.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    myinterlist.append(MG.Interaction({\
                      'id':3,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[2]] *3 \
                                             ),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
 #   mymodel.set('couplings', {(0, 0):'GQQ'})
 #   mymodel.set('lorentz', ['L1'])

    #mymodel.set('couplings',{(0, 0):'GQQ'} )

    
    dict = {'legs' : myleglist, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    
    myproc = MG.Process(dict)
    myamp = diagram_generation.Amplitude(myproc)
    
    def test_color_repr(self):
        """test if the correct color representation is assigned to a given leg"""
        leg = {\
               'id': 21,\
               'number': 5,\
               'state': True,\
               'from_group': True\
               }
        myfksdir = fks.FKSDirectory(self.myamp, leg)        
        
        legs = []
        cols = []
        
        #initial quark > antitriplet
        legs.append({ \
                     'id': 2,\
                     'number': 1,\
                     'state': False,\
                     'from_group': True \
                     })
        cols.append(-3)
        #initial antiquark > triplet
        legs.append({ \
                     'id': -2,\
                     'number': 1,\
                     'state': False,\
                     'from_group': True \
                     })
        cols.append(3)        
        #final quark > triplet
        legs.append({ \
                     'id': 2,\
                     'number': 1,\
                     'state': True,\
                     'from_group': True \
                     })
        cols.append(3) 
        #final antiquark > antitriplet
        legs.append({ \
                     'id': -2,\
                     'number': 1,\
                     'state': True,\
                     'from_group': True \
                     })
        cols.append(-3)   
        #initial gluon > octet
        legs.append({ \
                     'id': 21,\
                     'number': 1,\
                     'state': False,\
                     'from_group': True \
                     })
        cols.append(8)      
        #final gluon > octet
        legs.append({ \
                     'id': 21,\
                     'number': 1,\
                     'state': True,\
                     'from_group': True \
                     })
        cols.append(8)    
        #initial photon > singlet
        legs.append({ \
                     'id': 22,\
                     'number': 1,\
                     'state': False,\
                     'from_group': True \
                     })
        cols.append(1)      
        #final photon > singlet
        legs.append({ \
                     'id': 22,\
                     'number': 1,\
                     'state': True,\
                     'from_group': True \
                     })
        cols.append(1)
        
        for leg in legs:
            mycol = cols[legs.index(leg)]
            self.assertEqual(myfksdir.color_repr(leg), mycol)     

    
    def test_define_link(self):
        """test if the link informations created by define_link are correct"""
        leg = {\
               'id': 21,\
               'number': 5,\
               'state': True,\
               'from_group': True\
               }
        myfksdir = fks.FKSDirectory(self.myamp, leg) 
        leg1 =  {\
               'id': -2,\
               'number': 2,\
               'state': False,\
               'from_group': True\
               }   
        leg2 =  {\
               'id': 1,\
               'number': 3,\
               'state': True,\
               'from_group': True\
               }  
        self.assertEqual(myfksdir.define_link(leg1, leg2),
                         [{'number': 2, 'color': 3}, {'number': 3, 'color': 3}]
                         )


    def test_insert_link(self):
        """test to check the correct insertion of the color link"""
        col_dicts = []
        links = []
        ins_dicts = []
        
        col_dicts.append([{(0,0): [color.T(-1001,4,1),
                                  color.T(-1001,2,3)]}])
        links.append([{'number' : 1, 'color': -3},{'number' :3, 'color': -3}])
        ins_dicts.append([{(0,0):[color.T(-1001,4,-1002),
                                 color.T(-1001,2,-1003),
                                 color.T(-2002,-1002,1),
                                 color.T(-2002,-1003,3)]
                         }])
        
        
        for dict in col_dicts:
            link = links[col_dicts.index(dict)]
            ins_dict = ins_dicts[col_dicts.index(dict)]
        self.assertEqual(ins_dict, fks.insert_link(dict, link) )
    
