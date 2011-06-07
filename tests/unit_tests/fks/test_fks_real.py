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
import madgraph.fks.fks_real as fks
import madgraph.fks.fks_common as fks_common
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import copy
import string

class TestFKSProcessFromReals(unittest.TestCase):
    """a class to test FKS Processes initiated from real process"""
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
        
    antiu = MG.Particle({'name':'u',
                  'antiname':'u~',
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
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))    
    
    myinterlist.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

    myinterlist.append(MG.Interaction({\
                      'id':5,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[4], \
                                             antit, \
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
    
    myinterlist.append(MG.Interaction({\
                      'id':4,\
                      'particles': MG.ParticleList([mypartlist[1], \
                                             antid, \
                                             mypartlist[3]]
                                             ),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'ADD'},
                      'orders':{'QED':1}}))
    myinterlist.append(MG.Interaction({\
                      'id':6,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'AUU'},
                      'orders':{'QED':1}}))
    

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
 #   mymodel.set('couplings', {(0, 0):'GQQ'})
 #   mymodel.set('lorentz', ['L1'])

    #mymodel.set('couplings',{(0, 0):'GQQ'} )

    
    dict = {'legs' : fks_common.to_fks_legs(myleglist, mymodel), 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2 = {'legs' : fks_common.to_fks_legs(myleglist2, mymodel), 'orders':{'QCD':10, 'QED':0},
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
    
    fks1 = fks.FKSProcessFromReals(myproc)
    fks2 = fks.FKSProcessFromReals(myproc2)
    
    def test_get_fks_inc_string(self):
        """check if the fks.inc string is correct, for u u~ > u u~ g"""
        lines = self.fks2.get_fks_inc_string().split('\n')
                
        goallines = """INTEGER FKS_CONFIGS, IPOS, JPOS
      DATA FKS_CONFIGS / 7 /
      INTEGER FKS_I(7), FKS_J(7)
      INTEGER FKS_IPOS(0:NEXTERNAL)
      INTEGER FKS_J_FROM_I(NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE(NEXTERNAL), PDG_TYPE(NEXTERNAL)

C     FKS configuration number  1 
      DATA FKS_I(  1  ) /  3  /
      DATA FKS_J(  1  ) /  1  /

C     FKS configuration number  2 
      DATA FKS_I(  2  ) /  4  /
      DATA FKS_J(  2  ) /  2  /

C     FKS configuration number  3 
      DATA FKS_I(  3  ) /  4  /
      DATA FKS_J(  3  ) /  3  /

C     FKS configuration number  4 
      DATA FKS_I(  4  ) /  5  /
      DATA FKS_J(  4  ) /  1  /

C     FKS configuration number  5 
      DATA FKS_I(  5  ) /  5  /
      DATA FKS_J(  5  ) /  2  /

C     FKS configuration number  6 
      DATA FKS_I(  6  ) /  5  /
      DATA FKS_J(  6  ) /  3  /

C     FKS configuration number  7 
      DATA FKS_I(  7  ) /  5  /
      DATA FKS_J(  7  ) /  4  /

      DATA (FKS_IPOS(IPOS), IPOS = 0, 3)  / 3, 3, 4, 5 /

      DATA (FKS_J_FROM_I(3, JPOS), JPOS = 0, 1)  / 1, 1 /
      DATA (FKS_J_FROM_I(4, JPOS), JPOS = 0, 2)  / 2, 2, 3 /
      DATA (FKS_J_FROM_I(5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /
C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE(IPOS), IPOS=1, NEXTERNAL) / 3, -3, 3, -3, 8 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE(IPOS), IPOS=1, NEXTERNAL) / 2, -2, 2, -2, 21 /
        """.split('\n')
        
        for l1, l2 in zip(lines, goallines):
            self.assertEqual(string.strip(l1.upper()), string.strip(l2.upper())) 
        

    
    def test_FKS_born_process(self):
        """tests the correct initialization of a FKSBornProcess object, which
        takes as input a real process, FKSlegs corresponding to i/j and ij fks
        in particular check i
        --born amplitude
        --i/j fks
        --ijglu
        also the correctness of reduce_real_leglist is (implicitly) tested here"""
        model = self.mymodel
        
        ## u u~ > d d~ g g, combining d and d~ to glu
        born1 = fks.FKSBornProcess(self.myproc, 
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': -1,\
                                        'number': 4,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 1,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 21,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model) )
    
        mylegsborn1 = MG.LegList()
    # PROCESS: u u~ > g g g 
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
    'id': 21,\
    'number': 3,\
    'state': True,\
  #  'from_group': True\
},\
{\
    'id': 21,\
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
            mylegsborn1.append(MG.Leg(i))
        
        born_amp1 = diagram_generation.Amplitude(MG.Process(
                            {'legs' : fks_common.to_fks_legs(mylegsborn1, model), 'orders':{'QCD':10, 'QED':0},
                           'model': model,
                           'id': 1,
                           'required_s_channels':[],
                           'forbidden_s_channels':[],
                           'forbidden_particles':[],
                           'is_decay_chain': False,
                           'decay_chains': MG.ProcessList(),
                           'overall_orders': {}} ))
        
        self.assertEqual(born1.i_fks, 4)
        self.assertEqual(born1.j_fks, 3)
        self.assertEqual(born1.ijglu, 3)
        self.assertEqual(born1.amplitude, born_amp1)
        self.assertEqual(born1.need_color_links, False)
        
        
    def test_find_color_links(self):
        """tests the find_color_link function of a FKSBornProcess object. This
        function calls the one in fks_common, so this tests just checks that
        the list color_links has the correct size"""
        model = self.mymodel
        
        ## u u~ > d d~ g g, combining d and d~ to glu
        born1 = fks.FKSBornProcess(self.myproc, 
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': -1,\
                                        'number': 4,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 1,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 21,\
                                        'number': 3,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model) )
        born1.find_color_links()
        #no color links for this process
        self.assertEqual(len(born1.color_links), 0)
        
        ## u u~ > d d~ g g, combining g and d~ to d~
        born1 = fks.FKSBornProcess(self.myproc, 
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': 21,\
                                        'number': 5,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': -1,\
                                        'number': 4,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model),
                                        fks_common.to_fks_leg(MG.Leg({\
                                        'id': -1,\
                                        'number': 4,\
                                        'state': True,\
                                       # 'from_group': True\
                                    }), model) )
        born1.find_color_links()
        #no color links for this process
        self.assertEqual(len(born1.color_links), 20)

    
    def test_FKS_process_from_reals(self):
        """tests the correct initialization of a FKSProcessFromReals object 
        either using an input process or using an input amplitude
        in particular it checks the correct initialization of the following 
        internal variables:
        --real_proc/real_amp
        --model
        --leglist, nlegs
        --pdg codes, colors ###, ipos, j_from_i to be written in fks.inc
        --borns"""
        
        #u u~ > d d~ g g
        proc1 = self.myproc
        amp1 = diagram_generation.Amplitude(proc1)
        fks1 = self.fks1
        fks2 = fks.FKSProcessFromReals(amp1)
        
        self.assertEqual(fks1.leglist, 
                         fks_common.to_fks_legs(self.myleglist, self.mymodel))
        self.assertEqual(len(fks1.borns), 11)
        self.assertEqual(fks1.pdg_codes, [2,-2, 1, -1, 21, 21])
        self.assertEqual(fks1.colors, [3,-3,3,-3,8,8])
        self.assertEqual(fks1.nlegs, 6)
        self.assertEqual(fks1.real_proc, proc1)
  #      self.assertEqual(fks1.real_amp, amp1)
        self.assertEqual(fks1.model, self.mymodel)
        
        self.assertEqual(fks2.leglist, 
                         fks_common.to_fks_legs(self.myleglist, self.mymodel))
        self.assertEqual(len(fks2.borns), 11)
        self.assertEqual(fks2.pdg_codes, [2,-2, 1, -1, 21, 21])
        self.assertEqual(fks2.colors, [3,-3,3,-3,8,8])
        self.assertEqual(fks2.nlegs, 6)
        self.assertEqual(fks2.real_proc, proc1)
  #      self.assertEqual(fks1.real_amp, amp1)
        self.assertEqual(fks2.model, self.mymodel)
        
        
    def test_find_borns(self):
        """checks if, for a given process, all the underlying born processes are
        found"""
        #u u~ > d d~ g g 
        fks1 = self.fks1
        
        target_borns1 = [[2,-2,21,21,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21],
                         [2,-2,1,-1,21]
                         ]

        #u u~ > u u~ g 
        fks2 = self.fks2
       # fks2 = fks.FKSProcessFromReals(self.myproc2)
        target_borns2 = [[21,-2,-2,21],
                         [2,21,2,21],
                         [2,-2,21,21],
                         [2,-2,2,-2],
                         [2,-2,2,-2]
                         ]

        for b1, b2 in zip(fks1.borns, target_borns1):
            self.assertEqual([l.get('id') for l in  b1.process.get('legs')], b2)
        for b1, b2 in zip(fks2.borns, target_borns2):
            self.assertEqual([l.get('id') for l in  b1.process.get('legs')], b2)


