################################################################################
#
# Copyright (c) 2013 The MadGraph Development team and Contributors
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

import copy
import shutil
import os
import unittest
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
from madgraph import MG5DIR
from cStringIO import StringIO
import tests.IOTests as IOTests
import tempfile

pjoin = os.path.join

class TestMadWeight(IOTests.IOTestManager):
    """ """
    
    def test_ungrouping_lepton(self):
        """check that the routines which ungroup the process for the muon/electron processes
        works as expected"""
        
        # Setup A simple model
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        u = mypartlist[-1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # A quark C and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':4,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        c = mypartlist[-1]
        antic = copy.copy(c)
        antic.set('is_part', False)

        # electron/positront
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':1,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        e = mypartlist[-1]
        antie = copy.copy(e)
        mu = copy.copy(e)
        antie.set('is_part', False)
        mu.set('name', 'mu-')
        mu.set('antiname', 'mu+')
        mu.set('pdg_code', 13)
        mypartlist.append(mu)
        antimu = copy.copy(mu)
        antimu.set('is_part', False)        

        # A Z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[-1]

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))
        
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 1):'GDZ2'},
                      'orders':{'QED':1}})) 
        
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antic, \
                                             c, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))       

        # Coupling of Z to leptons
        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antie, \
                                             e, \
                                             z]),
                      'color': [color.ColorString()],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GLZ1', (0, 1):'GLZ2'},
                      'orders':{'QED':1}})) 
        
        # Coupling of Z to leptons
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [antimu, \
                                             mu, \
                                             z]),
                      'color': [color.ColorString()],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GLZ1', (0, 1):'GLZ2'},
                      'orders':{'QED':1}})) 

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')
        
        procs = [[1,-1,23], [2,-2,23], [4,-4,23]]
        decays = [[23,11,-11], [23,13,-13]]
        coreamplitudes = diagram_generation.AmplitudeList()
        decayamplitudes = diagram_generation.AmplitudeList()
        decayprocs = base_objects.ProcessList()

        #Building the basic objects
        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            my_amplitude.set('has_mirror_process', True)
            coreamplitudes.append(my_amplitude)

        for proc in decays:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'is_decay_chain': True})
            my_amplitude = diagram_generation.Amplitude(my_process)
            decayamplitudes.append(my_amplitude)
            decayprocs.append(my_process)

        decays = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': decayamplitudes})])

        decay_chains = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes,
            'decay_chains': decays})

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
              group_amplitudes(\
                 diagram_generation.DecayChainAmplitudeList([decay_chains]))

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()
        
        ######################
        ##  Make the test!! ##
        ######################
        self.assertEqual(len(subproc_groups), 1)
        subproc_groups = subproc_groups.split_lepton_grouping()
        self.assertEqual(len(subproc_groups), 2)        
        
        # check that indeed 
        for group in subproc_groups:
            self.assertEqual(len(group['matrix_elements']), 2)
            has_muon=any(abs(l['id'])==13 for l in group['matrix_elements'][0]['processes'][0]['decay_chains'][0]['legs'])
            for me in group['matrix_elements']:
                if abs(me['processes'][0]['legs'][0]['id']) == 1:
                    self.assertEqual(len(me['processes']), 1)
                else:
                    self.assertEqual(len(me['processes']), 2)
                for proc in me['processes']:
                    for dec in proc['decay_chains']:
                        if has_muon:
                            self.assertFalse(any(abs(l['id'])==11 for l in dec['legs']))
                        else:
                            self.assertFalse(any(abs(l['id'])==13 for l in dec['legs']))
            
            self.assertNotEqual(group['name'], 'qq_z_z_ll')
            if has_muon:
                self.assertEqual(group['name'], 'qq_z_z_mummup')
            else:
                self.assertEqual(group['name'], 'qq_z_z_emep')
            group['name']

    @IOTests.createIOTest()
    def testIO_modification_to_cuts(self):
        """ target: cuts.f"""
        
        exporter = export_v4.ProcessExporterFortranMW(MG5DIR, self.IOpath)
        strfile = StringIO()
        exporter.get_mw_cuts_version(strfile)
        
        text = strfile.getvalue()
        
        # first ensure that the file seems coherent
        self.assertFalse(' call initcluster' in  text)
        self.assertFalse('ickkw' in text )
        self.assertFalse('genps.inc' in text )
        self.assertTrue('logical function cut_bw' in text)
        self.assertTrue('maxparticles.inc' in text )
        
        open(pjoin(self.IOpath, 'cuts.f'),'w').write(text)
        # But force manual inspection at each change of the file

        
        