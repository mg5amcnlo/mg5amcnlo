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
import models.import_ufo as import_ufo
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
    
    fks1 = fks.FKSProcessFromReals(myproc, False)
    fks1_rem = fks.FKSProcessFromReals(myproc)
    fks2 = fks.FKSProcessFromReals(myproc2, False)
    
        

    
    def test_FKS_born_process(self):
        """tests the correct initialization of a FKSBornProcess object, which
        takes as input a real process, FKSlegs corresponding to i/j and ij fks
        in particular check
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
                            {'legs' : mylegsborn1, 'orders':{'QCD':10, 'QED':0},
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
        self.assertEqual(born1.is_nbody_only, False)
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
        --pdg codes, colors 
        -- fks_j_from_i to be written in fks.inc
        --borns"""
        
        #u u~ > d d~ g g
        proc1 = self.myproc
        amp1 = diagram_generation.Amplitude(proc1)
        fks1 = self.fks1
        fks2 = fks.FKSProcessFromReals(amp1, False)
        
        self.assertEqual(fks1.leglist, 
                         fks_common.to_fks_legs(self.myleglist, self.mymodel))
        self.assertEqual(len(fks1.borns), 11)
        self.assertEqual(fks1.fks_j_from_i, 
                {1:[], 2:[], 3:[], 4:[3], 5:[1,2,3,4,6], 6:[1,2,3,4,5]})
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

    
    def test_find_borns_to_integrate(self):
        """tests if the double genereted configurations are set as not to integrate
        and if they are removed from the born list when asked.
        it uses self.fks1 and self.fks1_rem (u u~ > d d~ g g). The configurations
        for which i_fks is = 5 should be set not to integrate"""
        to_int = 0
        not_to_int = 0
        for born in self.fks1.borns:
            if born.i_fks == 5:
                self.assertEqual(born.is_to_integrate, False)
                not_to_int += 1
            else:
                self.assertEqual(born.is_to_integrate, True)
                self.assertEqual(born.i_fks, self.fks1_rem.borns[to_int].i_fks)
                self.assertEqual(born.j_fks, self.fks1_rem.borns[to_int].j_fks)
                self.assertEqual(born.amplitude, self.fks1_rem.borns[to_int].amplitude)
                to_int += 1

        self.assertEqual(to_int, 6)
        self.assertEqual(not_to_int, 5)
        
        self.assertEqual(len(self.fks1_rem.borns), 6)

    def test_fks_j_from_i(self):
        """tests the correct filling of the fks_j_from_i dictionary.
        In particular check that it is filled also with j/i corresponding
        to configurations that are removed because not to be integrated"""
        self.assertEqual(self.fks1.fks_j_from_i, 
                {1:[], 2:[], 3:[], 4:[3], 5:[1,2,3,4,6], 6:[1,2,3,4,5]})
        self.assertEqual(self.fks1.fks_j_from_i, self.fks1_rem.fks_j_from_i)

    
    def test_find_nbodyonly(self):
        """tests if the is_nbody_only variable is set to true only in the 
        "last" soft-singular born, i.e. the one for which i/j are the largest"""
        
        #fks1 -> i should be 6, j should be 5
        for born in self.fks1.borns:
            if born.i_fks == 6 and born.j_fks ==5:
                self.assertEqual(born.is_nbody_only, True)
            else:
                self.assertEqual(born.is_nbody_only, False)

        #fks2 -> i should be 5, j should be 4
        for born in self.fks2.borns:
            if born.i_fks == 5 and born.j_fks == 4:
                self.assertEqual(born.is_nbody_only, True)
            else:
                self.assertEqual(born.is_nbody_only, False)        
        
        
    def test_sort_fks_proc_from_real(self):
        """tests that two FKSProcessesFromReal with different legs order in the
        input process/amplitude are returned as equal"""
        model = import_ufo.import_model('sm')

# sorted leglist for e+ e- > u u~ g g
        myleglist_s = MG.LegList()
        myleglist_s.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':-2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':21, 'state':True}))
        myleglist_s.append(MG.Leg({'id':21, 'state':True}))

# unsorted leglist: e+ e- > u g u~
        myleglist_u = MG.LegList()
        myleglist_u.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_u.append(MG.Leg({'id':11, 'state':False}))
        myleglist_u.append(MG.Leg({'id':21, 'state':True}))
        myleglist_u.append(MG.Leg({'id':2, 'state':True}))
        myleglist_u.append(MG.Leg({'id':21, 'state':True}))
        myleglist_u.append(MG.Leg({'id':-2, 'state':True}))

# define (un)sorted processes:
        proc_s = MG.Process({'model':model, 'legs':myleglist_s,\
                             'orders':{'QED':2, 'QCD':1}})
        proc_u = MG.Process({'model':model, 'legs':myleglist_u,\
                             'orders':{'QED':2, 'QCD':1}})
# define (un)sorted amplitudes:
        amp_s = diagram_generation.Amplitude(proc_s)
        amp_u = diagram_generation.Amplitude(proc_u)

        fks_p_s = fks.FKSProcessFromReals(proc_s)
        fks_p_u = fks.FKSProcessFromReals(proc_u)

        self.assertEqual(fks_p_s.real_proc, fks_p_u.real_proc)
        self.assertEqual(fks_p_s.real_amp, fks_p_u.real_amp)

        fks_a_s = fks.FKSProcessFromReals(amp_s)
        fks_a_u = fks.FKSProcessFromReals(amp_u)

        self.assertEqual(fks_a_s.real_proc, fks_a_u.real_proc)
        self.assertEqual(fks_a_s.real_amp, fks_a_u.real_amp)
