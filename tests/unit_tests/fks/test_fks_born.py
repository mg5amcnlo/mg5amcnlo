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

"""Testing modules for FKS_process_born class"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_born as fks_born
import madgraph.fks.fks_common as fks_common
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
import copy
import array


class TestFKSProcess(unittest.TestCase):
    """a class to test FKS Processes"""
    myleglist = MG.LegList()
    # PROCESS: u g > u g 
    mylegs = [{ \
    'id': 2,\
    'number': 1,\
    'state': False,\
}, \
{ \
    'id': 21,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': 21,\
    'number': 4,\
    'state': True,\
}
]

    for i in mylegs:
        myleglist.append(MG.Leg(i))

    myleglist2 = MG.LegList()
    # PROCESS: d d~ > u u~
    mylegs2 = [{ \
    'id': 1,\
    'number': 1,\
    'state': False,\
}, \
{ \
    'id': -1,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 2,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': -2,\
    'number': 4,\
    'state': True,\
}
]
    for i in mylegs2:
        myleglist2.append(MG.Leg(i))
        
        myleglist3 = MG.LegList()
    # PROCESS: d d~ > a a
    mylegs3 = [{ \
    'id': 1,\
    'number': 1,\
    'state': False,\
}, \
{ \
    'id': -1,\
    'number': 2,\
    'state': False,\
},\
{\
    'id': 22,\
    'number': 3,\
    'state': True,\
},\
{\
    'id': 22,\
    'number': 4,\
    'state': True,\
}
]
    for i in mylegs3:
        myleglist3.append(MG.Leg(i))

    
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
    
    expected_qcd_inter = MG.InteractionList()
        
    expected_qcd_inter.append(MG.Interaction({\
                      'id':1,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[1], \
                                             antid, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.append(MG.Interaction({\
                      'id':2,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[0], \
                                             antiu, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.append(MG.Interaction({\
                      'id':3,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[2]] *3 \
                                             ),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    expected_qcd_inter.append(MG.Interaction({\
                      'id':5,\
                      'particles': MG.ParticleList(\
                                            [mypartlist[4], \
                                             antit, \
                                             mypartlist[2]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))
    
    expected_qcd_inter.sort()

    mymodel = MG.Model()
    mymodel.set('particles', mypartlist)
    mymodel.set('interactions', myinterlist)
    
    dict = {'legs' : myleglist, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict2 = {'legs' : myleglist2, 'orders':{'QCD':10, 'QED':0},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}

    dict3 = {'legs' : myleglist3, 'orders':{'QCD':0, 'QED':2},
                       'model': mymodel,
                       'id': 1,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_particles':[],
                       'is_decay_chain': False,
                       'perturbation_couplings':['QCD'],
                       'decay_chains': MG.ProcessList(),
                       'overall_orders': {}}
    
    myproc = MG.Process(dict)
    myproc2 = MG.Process(dict2)
    myprocaa= MG.Process(dict3)
    
    
    def test_FKSMultiProcessFromBorn(self):
        """tests the correct initializiation of a FKSMultiProcess. In particular
        checks that the correct number of borns is found"""
        
        p = [1, 21]

        my_multi_leg = MG.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = MG.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        my_process_definition = MG.ProcessDefinition({\
                        'legs': my_multi_leglist,
                        'perturbation_couplings': ['QCD'],
                        'NLO_mode': 'real',
                        'model': self.mymodel})
        my_process_definitions = MG.ProcessDefinitionList(\
            [my_process_definition])

        my_multi_process = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions':my_process_definitions})
        
        self.assertEqual(len(my_multi_process.get('born_processes')),4)
        #check the total numbers of reals 11 11 6 16
        totreals = 0
        for born in my_multi_process.get('born_processes'):
            for reals in born.reals:
                totreals += len(reals)
        self.assertEqual(totreals, 44)
    

    def test_FKSRealProcess_init(self):
        """tests the correct initialization of the FKSRealProcess class. 
        In particular checks that
        --fks_info
        --amplitude (also the generate_real_amplitude function is tested)
        --leg_permutation <<REMOVED
        are set to the correct values"""
        #u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        #take the first real for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][0]
        realproc = fks_born.FKSRealProcess(fksproc.born_proc, leglist, 1,0)

        self.assertEqual(realproc.fks_infos, [{'i' : 5,
                                               'j' : 1,
                                               'ij' : 1,
                                               'ij_glu' : 0,
                                               'need_color_links': True}])

        sorted_legs = fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel )
        

        sorted_real_proc = MG.Process({'legs':sorted_legs, 'model':self.mymodel,
            'orders':{'QCD':3, 'QED':0, 'WEIGHTED': 3}, 'id':1})
## an emplty amplitude is generted so far...
        self.assertEqual(diagram_generation.Amplitude(), realproc.amplitude)

## now generate the amplitude
        realproc.generate_real_amplitude()

        self.assertEqual(sorted_real_proc, realproc.amplitude.get('process'))
        amp = diagram_generation.Amplitude(sorted_real_proc)
        self.assertEqual(amp,realproc.amplitude)
        self.assertEqual(array.array('i',[2,21,2,21,21]), realproc.pdgs)
        self.assertEqual([3,8,3,8,8], realproc.colors)
 ##       self.assertEqual(realproc.permutation, [1,2,4,5,3])


    def test_find_fks_j_from_i(self):
        """tests that the find_fks_j_from_i function of a FKSRealProcess returns the
        correct result"""
        #u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        #take the first real for this process 2j 21 >2 21 21i
        leglist = fksproc.reals[0][0]
        realproc = fks_born.FKSRealProcess(fksproc.born_proc, leglist, 1,0)
        target = {1:[], 2:[], 3:[1,2], 4:[1,2,3,5], 5:[1,2,3,4] }
        self.assertEqual(target, realproc.find_fks_j_from_i())


    def test_fks_real_process_get_leg_i_j(self):
        """test the correct output of the FKSRealProcess.get_leg_i/j() function"""
        #u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        #take the first real for this process 2j 21 > 2 21 21i
        leglist = fksproc.reals[0][0]
        realproc = fks_born.FKSRealProcess(fksproc.born_proc, leglist,1,0)
        self.assertEqual(realproc.get_leg_i(), leglist[4])
        self.assertEqual(realproc.get_leg_j(), leglist[0])
    
    def test_generate_reals_no_combine(self):
        """tests the generate_reals function, if all the needed lists
        -- amplitudes
        -- real amps
        have the correct number of elements
        checks also the find_reals_to_integrate, find_real_nbodyonly functions
        that are called by generate_reals"""
        
        #process u g > u g 
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        fksproc.generate_reals(False)
        
        #there should be 11 real processes for this born
        self.assertEqual(len(fksproc.real_amps), 11)


    def test_generate_reals_no_combine(self):
        """tests the generate_reals function, if all the needed lists
        -- amplitudes
        -- real amps
        have the correct number of elements. 
        Check also that real emissions with the same m.e. are combined together"""
        
        #process u g > u g 
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        fksproc.generate_reals()
        
        #there should be 8 real processes for this born
        self.assertEqual(len(fksproc.real_amps), 8)
        # the process u g > u g g should correspond to 4 possible fks_confs:
        amp_ugugg = [amp for amp in fksproc.real_amps \
                if amp.pdgs == array.array('i', [2, 21, 2, 21, 21])]
        self.assertEqual(len(amp_ugugg), 1)
        self.assertEqual(len(amp_ugugg[0].fks_infos), 4)
        self.assertEqual(amp_ugugg[0].fks_infos,
                [{'i':5, 'j':1, 'ij':1, 'ij_glu':0, 'need_color_links':True},
                 {'i':5, 'j':2, 'ij':2, 'ij_glu':2, 'need_color_links':True},
                 {'i':5, 'j':3, 'ij':3, 'ij_glu':0, 'need_color_links':True},
                 {'i':5, 'j':4, 'ij':4, 'ij_glu':4, 'need_color_links':True}])

        
    def test_find_reals(self):
        """tests if all the real processes are found for a given born"""
        #process is u g > u g
        fksproc = fks_born.FKSProcessFromBorn(self.myproc)
        target = []
        
        #leg 1 can split as u g > u g g or  g g > u u~ g 
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel ),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                         fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)]
                                        )
        
        #leg 2 can split as u d > d u g OR u d~ > d~ u g OR
        #                   u u > u u g OR u u~ > u u~ g OR
        #                   u g > u g g 
        
        target.append([fks_common.to_fks_legs([#ug>ugg
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                       fks_common.to_fks_legs([#ud>dug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ],self.mymodel ),
                    fks_common.to_fks_legs([#ud~>d~ug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu>uug
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([#uu~>uu~g
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)
                    ])
        
        #leg 3 can split as u g > u g g
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        ]
                                        )
        #leg 4 can split as u g > u g g or u g > u u u~ or u g > u d d~ 
        target.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                        fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :1,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                        fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :2,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)
                                        
                                        ]
                                        )
                       
        for i in range(len(fksproc.reals)):
            for j in range(len(fksproc.reals[i])):
                self.assertEqual(fksproc.reals[i][j], target[i][j]) 
        
        #process is d d~ > u u~
        fksproc2 = fks_born.FKSProcessFromBorn(self.myproc2)
        target2 = []
        #leg 1 can split as d d~ > u u~ g or  g d~ > d~ u u~ 
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)]
                                        )
        
        #leg 2 can split as d d~ > u u~ g or  d g > d u u~ 
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'i'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :-2,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'n'})
                                        ], self.mymodel)]
                                        )
        
        #leg 3 can split as d d~ > u u~ g  
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)])

        #leg 4 can split as d d~ > u u~ g  
        target2.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :2,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-2,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)])

        for i in range(len(fksproc2.reals)):
            self.assertEqual(fksproc2.reals[i], target2[i])       
    
        #d d~ > a a
        fksproc3 = fks_born.FKSProcessFromBorn(self.myprocaa)
        target3 = []
        #leg 1 can split as d d~ > g a a or  g d~ > d~ a a 
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :21,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)]
                                        )
        #leg 2 can split as d d~ > g a a  or  d g > d a a 
        target3.append( [fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :-1,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' :22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' :21,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel),
                    fks_common.to_fks_legs([
                                        fks_common.FKSLeg(
                                        {'id' :1,
                                         'number' :1,
                                         'state' :False,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' :21,
                                         'number' :2,
                                         'state' :False,
                                         'fks' : 'j'}),
                                        fks_common.FKSLeg(
                                         {'id' : 22,
                                         'number' :3,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg( 
                                         {'id' : 22,
                                         'number' :4,
                                         'state' :True,
                                         'fks' : 'n'}),
                                        fks_common.FKSLeg(
                                         {'id' : 1,
                                         'number' :5,
                                         'state' :True,
                                         'fks' : 'i'})
                                        ], self.mymodel)]
                                        )        

        for real, res in zip(fksproc3.reals, target3):
            self.assertEqual(real, res) 


    def test_sort_fks_proc_from_born(self):
        """tests that two FKSProcessesFromBorn with different legs order in the
        input process/amplitude are returned as equal"""
        model = import_ufo.import_model('sm')

# sorted leglist for e+ e- > u u~ g
        myleglist_s = MG.LegList()
        myleglist_s.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':11, 'state':False}))
        myleglist_s.append(MG.Leg({'id':2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':-2, 'state':True}))
        myleglist_s.append(MG.Leg({'id':21, 'state':True}))

# unsorted leglist: e+ e- > u g u~
        myleglist_u = MG.LegList()
        myleglist_u.append(MG.Leg({'id':-11, 'state':False}))
        myleglist_u.append(MG.Leg({'id':11, 'state':False}))
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

        fks_p_s = fks_born.FKSProcessFromBorn(proc_s)
        fks_p_u = fks_born.FKSProcessFromBorn(proc_u)

        self.assertEqual(fks_p_s.born_proc, fks_p_u.born_proc)
        self.assertEqual(fks_p_s.born_amp, fks_p_u.born_amp)

        fks_a_s = fks_born.FKSProcessFromBorn(amp_s)
        fks_a_u = fks_born.FKSProcessFromBorn(amp_u)

        self.assertEqual(fks_a_s.born_proc, fks_a_u.born_proc)
        self.assertEqual(fks_a_s.born_amp, fks_a_u.born_amp)
