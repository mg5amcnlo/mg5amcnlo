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

import copy
import itertools
import logging
import math
import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import tests.unit_tests.loop.test_loop_diagram_generation as looptest
import madgraph.core.drawing as draw_lib
import aloha.create_aloha as create_aloha
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_amp as color_amp
import madgraph.loop.loop_color_amp as loop_color_amp
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.iolibs.save_load_object as save_load_object
import models.import_ufo as models
from madgraph import MadGraph5Error


_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')

#===============================================================================
# LoopDiagramGeneration Test
#===============================================================================
class LoopHelasMatrixElementTest(unittest.TestCase):
    """Test class for all functions related to the LoopHelasMatrixElement"""
    
    myloopmodel = loop_base_objects.LoopModel()
    
    def setUp(self):
        """load the NLO toy model"""

#        self.myloopmodel = models.import_full_model(os.path.join(\
#            _input_file_path,'LoopModelTest'))
        self.myloopmodel = models.import_full_model(os.path.join(\
            _input_file_path,'LoopSMTest'))
        
    def test_get_aloha_input(self):
        """ Check that the function aloha_get_input in the class 
        HelasWavefunction and HelasAmplitude behaves as expected """
        
        d_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(1),
                                'number':1,
                                'is_loop':False,
                                'state':'initial'})
        antid_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(-1),
                                'number':2,
                                'is_loop':True,
                                'state':'initial'})
        g_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(21),
                                'number':3,
                                'is_loop':True,
                                'lorentz':['FFV1','FFV2','FFV3'],
                                'state':'final',
                                'pdg_codes':[1,-1,21],
                                'interaction_id':18})
        
        g_helas_wf.set('mothers',helas_objects.HelasWavefunctionList(\
                                                   [antid_helas_wf,d_helas_wf]))
        
        a_opt_input = g_helas_wf.get_aloha_info(optimized_output=True)
        a_def_input = g_helas_wf.get_aloha_info(optimized_output=False)
        
        self.assertEqual(a_opt_input,(('FFV1', 'FFV2', 'FFV3'), ('L1', 'P0'), 3))
        self.assertEqual(a_def_input,(('FFV1', 'FFV2', 'FFV3'), ('L', 'P0'), 3))
        
    def test_get_analytic_info(self):
        """ Check that the function get_analytic_info and compute_analytic
        info in the class HelasWavefunction behaves as expected """

        alohaModel = create_aloha.AbstractALOHAModel(self.myloopmodel.get('name'))
        alohaModel.add_Lorentz_object(self.myloopmodel.get('lorentz'))
        
        d_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(1),
                                'number':1,
                                'is_loop':False,
                                'state':'initial'})
        antid_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(21),
                                'number':2,
                                'is_loop':True,
                                'state':'initial'})
        g_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(1),
                                'number':5,
                                'is_loop':True,
                                'lorentz':['FFV1','FFV2'],
                                'state':'intermediate',
                                'pdg_codes':[1,-1,21],
                                'interaction_id':18})
        
        g_helas_wf.set('mothers',helas_objects.HelasWavefunctionList(\
                                                   [antid_helas_wf,d_helas_wf]))
        
        final_d_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(1),
                                'number':3,
                                'is_loop':False,
                                'state':'final'})
        final_antid_helas_wf = helas_objects.HelasWavefunction({
                                'particle':self.myloopmodel.get_particle(21),
                                'number':4,
                                'is_loop':True,
                                'pdg_codes':[1,-1,21],
                                'lorentz':['FFV1','FFV2'],
                                'interaction_id':18,
                                'state':'final'})
        
        final_antid_helas_wf.set('mothers',helas_objects.HelasWavefunctionList(\
                                                 [g_helas_wf,final_d_helas_wf]))

        
        # First make sure that, alone, with an alohaModel, the wavefunction can
        # return some analytic information.
        for (lwf, trgt_interaction_rank, trgt_wavefunction_rank) in \
            [(antid_helas_wf,0,0),(g_helas_wf,1,1),(final_antid_helas_wf,0,1)]:
            
            wf_rank = lwf.get_analytic_info('wavefunction_rank', \
                                                        alohaModel = alohaModel)
            self.assertEqual(wf_rank,trgt_wavefunction_rank)
            self.assertTrue('wavefunction_rank' in lwf['analytic_info'].keys())
            inter_rank = lwf.get_analytic_info('interaction_rank', \
                                                        alohaModel = alohaModel)
            self.assertEqual(inter_rank,trgt_interaction_rank)
            self.assertTrue('interaction_rank' in lwf['analytic_info'].keys())
            
            # Now make sure it is recovered without the alohaModel (so that the
            # caching works
            wf_rank = lwf.get_analytic_info('wavefunction_rank')
            self.assertEqual(wf_rank,trgt_wavefunction_rank)        
            inter_rank = lwf.get_analytic_info('interaction_rank')
            self.assertEqual(inter_rank,trgt_interaction_rank)
            
        for lwf in [antid_helas_wf,g_helas_wf,final_antid_helas_wf]:
            # Clean up for the next tests
            lwf['analytic_info'] = {}
        
        myProcess = base_objects.Process({'model':self.myloopmodel})
        
        
        # Build a loopME by hand. I know, it looks very complicated and...
        # ... in fact, it is very involved.
        
        myLoopHelasAmp = loop_helas_objects.LoopHelasAmplitude({
                        'amplitudes':helas_objects.HelasAmplitudeList([
                            helas_objects.HelasAmplitude({
                                'mothers':helas_objects.HelasWavefunctionList(
                                                  [final_antid_helas_wf,])})])})
        myLoopME = loop_helas_objects.LoopHelasMatrixElement({
            'diagrams':helas_objects.HelasDiagramList([
                loop_helas_objects.LoopHelasDiagram({
                    'amplitudes':helas_objects.HelasAmplitudeList([
                        myLoopHelasAmp]),
                    'wavefunctions':helas_objects.HelasWavefunctionList([
                        d_helas_wf,final_d_helas_wf]),
                    'loop_wavefunctions':helas_objects.HelasWavefunctionList([
                        antid_helas_wf,g_helas_wf,final_antid_helas_wf])
                    }
                )]),
            'processes':base_objects.ProcessList([myProcess])})

        # Now make sure that the compute_all_analytic information works indeed
        myLoopME.compute_all_analytic_information()
        for (lwf, trgt_interaction_rank, trgt_wavefunction_rank) in \
            [(antid_helas_wf,0,0),(g_helas_wf,1,1),(final_antid_helas_wf,0,1)]:
            wf_rank = lwf.get_analytic_info('wavefunction_rank')
            self.assertEqual(wf_rank,trgt_wavefunction_rank)        
            inter_rank = lwf.get_analytic_info('interaction_rank')
            self.assertEqual(inter_rank,trgt_interaction_rank)
            
        # Check that it works for the loop amplitude too
        wf_rank = myLoopHelasAmp.get_analytic_info('wavefunction_rank')
        self.assertEqual(wf_rank,1)
        inter_rank = myLoopHelasAmp.get_analytic_info('interaction_rank')
        self.assertEqual(inter_rank,0)        
            
        for lwf in [antid_helas_wf,g_helas_wf,final_antid_helas_wf]:
            # Clean up for the test that follows            
            lwf['analytic_info'] = {}

        # Also check the the compute_all_analytic_information works when one 
        # provides its own aloha model
        myLoopME.compute_all_analytic_information(alohaModel)
        for (lwf, trgt_interaction_rank, trgt_wavefunction_rank) in \
            [(antid_helas_wf,0,0),(g_helas_wf,1,1),(final_antid_helas_wf,0,1)]:
            wf_rank = lwf.get_analytic_info('wavefunction_rank')
            self.assertEqual(wf_rank,trgt_wavefunction_rank)        
            inter_rank = lwf.get_analytic_info('interaction_rank')
            self.assertEqual(inter_rank,trgt_interaction_rank)

        # Check that it works for the loop amplitude too
        wf_rank = myLoopHelasAmp.get_analytic_info('wavefunction_rank')
        self.assertEqual(wf_rank,1)
        inter_rank = myLoopHelasAmp.get_analytic_info('interaction_rank')
        self.assertEqual(inter_rank,0)

    def check_HME_individual_diag_sanity(self,Amplitude, process,\
          mode='collective', selection=None, verbose=False, checkColor=True):
        """ Check that the HelasDiagrams are correctly generated in
        HelasMatrixElement (HME) by generate_diagram() when initiated with
        the Amplitude given in argument. It does so by basically checking
        that the reconstructed (loop_)base_objects.(Loop)Diagram correctly match
        the original one used. The optional user arguments are:
        selection: list specifying what diagrams should be checked 
        mode: In 'individual' mode, a new matrix element is created for each
              diagram tested while in 'collective' mode, a single matrix element
              creates all the HelasDiagrams for the selection so that the dynamic
              effects of wavefunction recycling are tested.
        verbose: To display some informations on the diagram checked.
        checkColor: To also check the color generation"""
       
        alldiags=Amplitude['diagrams']
        # By default, try them all
        if not selection:
            if mode=='individual':
                DiagProcessed=range(len(alldiags))
            elif mode=='collective':
                DiagProcessed=[len(alldiags)-1]                
            else:
                raise Error,"Mode can only be 'individual' or 'collective'"
        else:
            DiagProcessed=selection
            
        for idiag in DiagProcessed:
            diagSelection=base_objects.DiagramList()
            selectionStart=0
            if mode=='individual':
                diagSelection=base_objects.DiagramList([alldiags[idiag]])
                selectionStart=idiag
            elif mode=='collective':
                if isinstance(idiag,int):
                    diagSelection=base_objects.DiagramList(alldiags[:idiag+1])
                elif isinstance(idiag,tuple):
                    diagSelection=base_objects.DiagramList(alldiags[idiag[0]:idiag[1]+1])
                    selectionStart=idiag[0]
                else:
                    raise Error,"Selection must be a list of integers or 2-tuple of integers."                                        
            else:
                raise Error,"Mode can only be 'individual' or 'collective'"
            Amplitude.set('diagrams',diagSelection)
            myME=helas_objects.HelasMatrixElement(Amplitude,gen_color=False)
            # Keep in mind that the loop diagram is always placed before its
            # corresponding CT diagrams.
            AllReconstructedDiags=myME.get_base_amplitude()['diagrams']
            colorize_obj=[]
            if checkColor:
                col_basis = color_amp.ColorBasis()
                newamp=myME.get_base_amplitude()
                myME.set('base_amplitude',newamp)
                colorize_obj = col_basis.create_color_dict_list(myME.get('base_amplitude'))
                self.assertEqual(len(colorize_obj),len(diagSelection))
                col_basis.build(myME.get('base_amplitude'))
                myME.set('color_basis',col_basis)
                col_matrix = color_amp.ColorMatrix(col_basis)
                if verbose:
                    print "color matrix =",col_matrix
                self.assertEqual(len(col_matrix),len(col_basis)**2)
                myME.set('color_matrix',col_matrix)
                color_amplitudes=myME.get_color_amplitudes()
                self.assertEqual(len(color_amplitudes),len(col_basis))
                # Create list of amplitude number appearing
                amp_number_apparition=[]
                for jamp in color_amplitudes:
                    amp_number_apparition.extend([a[1] for a in jamp])

            diagIndex=0
            for i, diag in enumerate(diagSelection):
                if verbose: 
                    print "============"
                    print "Checking diag",selectionStart+i,"with type :",\
                      diag.nice_string()
                    print "============"
                if mode=='individual':
                    reconstructedDiags=AllReconstructedDiags
                else:
                    reconstructedDiags=AllReconstructedDiags[diagIndex:\
                                                                 diagIndex+1]
                    diagIndex+=1

                if verbose: print "Reconstructed diag :",\
                  reconstructedDiags[0].nice_string()
                # Initialize some quantities for the tests
                origDiagNVertices=len(diag['vertices'])

                # Check of the loop diagram    
                self.assertEqual(len(reconstructedDiags[0]['vertices']),\
                             origDiagNVertices)
                
                if checkColor:
                    numEntries=1
                    for vert in diag['vertices']:
                        numColorKeys=\
        len(process['model'].get('interaction_dict')[vert['id']].get('color'))
                        if numColorKeys>1:
                             numEntries*=numColorKeys
                    if verbose: print "Diag colorization :",colorize_obj[i].keys()
                    self.assertEqual(len(colorize_obj[i].keys()),numEntries)
                    # Commented because it is only true when there is only one
                    # lorentz structure per vertex.
                    #self.assertEqual(len(colorize_obj[i].keys()),\
                    #  len(myME['diagrams'][i]['amplitudes']))
                    for amp in myME['diagrams'][i]['amplitudes']:
                        # Check that the amplitude 'color_indices' is in the
                        # color keys tuples generated
                        self.assertTrue(tuple(amp['color_indices']) in \
                                        colorize_obj[i].keys())
                        # Check that this amplitude number appear in the result
                        # from get_color_amplitude
                        self.assertTrue(amp['number'] in amp_number_apparition)                  
                        # For the verbose mode, show to which color basis element each
                        # amplitude contributes to
                        contribution_list=[]
                        for j, a in enumerate(color_amplitudes):
                            if amp['number'] in [b[1] for b in a]:
                                contribution_list.append(j)
                        if verbose:
                            print "Amplitude number",amp['number'],\
                            "contributes to the following color basis elements",contribution_list
        return myME

    def check_LHME_individual_diag_sanity(self,loopAmplitude, process,\
            mode='collective', selection=None, verbose=False, checkColor=True):
        """ Check that the HelasDiagrams are correctly generated in
        LoopHelasMatrixElement (LHME) by generate_diagram() when initiated with
        the (loop)Amplitude given in argument. It does so by basically checking
        that the reconstructed (loop_)base_objects.(Loop)Diagram correctly match
        the original one used. The optional user arguments are:
        selection: list specifying what diagrams should be checked 
        mode: In 'individual' mode, a new matrix element is created for each
              diagram tested while in 'collective' mode, a single matrix element
              creates all the HelasDiagrams for the selection so that the dynamic
              effects of wavefunction recycling are tested.
        verbose: To display some informations on the diagram checked.
        checkColor: To also check the color generation of the tested diagrams"""
       
        alldiags=loopAmplitude['born_diagrams']
        alldiags.extend(loopAmplitude['loop_diagrams'])
        alldiags.extend(loopAmplitude['loop_UVCT_diagrams'])
        # By default, try them all
        if not selection:
            if mode=='individual':
                DiagProcessed=range(len(alldiags))
            elif mode=='collective':
                DiagProcessed=[len(alldiags)-1]                
            else:
                raise Error,"Mode can only be 'individual' or 'collective'"
        else:
            DiagProcessed=selection
            
        for idiag in DiagProcessed:
            diagSelection=base_objects.DiagramList()
            selectionStart=0
            if mode=='individual':
                diagSelection=base_objects.DiagramList([alldiags[idiag]])
                selectionStart=idiag
            elif mode=='collective':
                if isinstance(idiag,int):
                    diagSelection=base_objects.DiagramList(alldiags[:idiag+1])
                elif isinstance(idiag,tuple):
                    diagSelection=base_objects.DiagramList(alldiags[idiag[0]:idiag[1]+1])
                    selectionStart=idiag[0]
                else:
                    raise Error,"Selection must be a list of integers or 2-tuple of integers."                                        
            else:
                raise Error,"Mode can only be 'individual' or 'collective'"
            
  #          diagSelection = base_objects.DiagramList(\
  #            [diag for diag in diagSelection if not isinstance(diag,\
  #             loop_base_objects.LoopUVCTDiagram) and diag['type']!=0])
            bornDiagSelected=base_objects.DiagramList(\
              [diag for diag in diagSelection if not isinstance(diag,\
               loop_base_objects.LoopUVCTDiagram) and diag['type']==0])
            loopDiagSelected=base_objects.DiagramList(\
              [diag for diag in diagSelection if not isinstance(diag,\
               loop_base_objects.LoopUVCTDiagram) and diag['type']!=0])
            loopUVCTDiagSelected=base_objects.DiagramList(\
              [diag for diag in diagSelection if isinstance(diag, \
                loop_base_objects.LoopUVCTDiagram)])
            loopAmplitude.set('diagrams',diagSelection)
            myloopME=loop_helas_objects.LoopHelasMatrixElement(loopAmplitude,
                                                                gen_color=False)
            # Keep in mind that the loop diagram is always placed before its
            # corresponding CT diagrams.
            AllReconstructedDiags=myloopME.get_base_amplitude().get('diagrams')

            loop_colorize_obj=[]
            born_colorize_obj=[]
            nBornRecDiags = len(bornDiagSelected)
            nLoopUVCTDiags = len(loopUVCTDiagSelected)
            nLoopDiags = 0
            for diag in loopDiagSelected:
                # We don't want to count twice the counterterms with same ID
                nLoopDiags += (1+len(set(vert.get('id') for vert in \
                                                      diag.get('CT_vertices'))))
            if checkColor:
                if (loopDiagSelected or loopUVCTDiagSelected):
                    loop_col_basis = loop_color_amp.LoopColorBasis()
                    newloopamp=myloopME.get_base_amplitude()
                    myloopME.set('base_amplitude',newloopamp)
                    loop_colorize_obj = loop_col_basis.create_loop_color_dict_list(\
                                          myloopME.get('base_amplitude'))
                    self.assertEqual(len(loop_colorize_obj),nLoopDiags+nLoopUVCTDiags)
                    loop_col_basis.build()
                    myloopME.set('loop_color_basis',loop_col_basis)
                    loop_color_amplitudes=myloopME.get_loop_color_amplitudes()
                    self.assertEqual(len(loop_color_amplitudes),len(loop_col_basis))
                    loop_amp_number_apparition=[]
                    for jamp in loop_color_amplitudes:
                        loop_amp_number_apparition.extend([a[1] for a in jamp])
                if bornDiagSelected:
                    born_col_basis = loop_color_amp.LoopColorBasis()
                    born_colorize_obj = born_col_basis.create_born_color_dict_list(\
                                          myloopME.get('base_amplitude'))
                    self.assertEqual(len(born_colorize_obj),nBornRecDiags)
                    born_col_basis.build()
                    myloopME.set('born_color_basis',born_col_basis)
                    born_color_amplitudes=myloopME.get_born_color_amplitudes()
                    self.assertEqual(len(born_color_amplitudes),len(born_col_basis))
                    born_amp_number_apparition=[]
                    for jamp in born_color_amplitudes:
                        born_amp_number_apparition.extend([a[1] for a in jamp])
                
                if bornDiagSelected and (loopDiagSelected or loopUVCTDiagSelected):
                    col_matrix=color_amp.ColorMatrix(loop_col_basis,\
                                                       born_col_basis)
                    if verbose:
                        print "color matrix virt_vs_born =",col_matrix
                    self.assertEqual(len(col_matrix),len(loop_col_basis)*len(born_col_basis))
                elif (loopDiagSelected or loopUVCTDiagSelected) and not bornDiagSelected:
                    col_matrix=color_amp.ColorMatrix(loop_col_basis)
                    if verbose:
                        print "color matrix virt_vs_virt =",col_matrix
                    self.assertEqual(len(col_matrix),len(loop_col_basis)**2)
                elif bornDiagSelected and not (loopDiagSelected or loopUVCTDiagSelected):
                    col_matrix=color_amp.ColorMatrix(born_col_basis)
                    if verbose:
                        print "color matrix born_vs_born =",col_matrix
                    self.assertEqual(len(col_matrix),len(born_col_basis)**2)
                myloopME.set('color_matrix',col_matrix)
                
            diagIndex=0
            bornColorDiagIndex=0
            loopColorDiagIndex=0            
            for i, diag in enumerate(diagSelection):
                if verbose: 
                    print "============"
                    if isinstance(diag,loop_base_objects.LoopUVCTDiagram):
                        print "Checking diag #",selectionStart+i," with type UVCT and interaction ids ",\
                        [coupl for coupl in diag.get('UVCT_couplings')]
                    else:
                        print "Checking diag #",selectionStart+i,"with type",diag['type'],":",\
                          diag.nice_string(loopAmplitude['structure_repository'])
                    print "============"
                if mode=='individual':
                    reconstructedDiags=AllReconstructedDiags
                    if checkColor:
                        this_born_colorize_obj=born_colorize_obj
                        this_loop_colorize_obj=loop_colorize_obj
                else:
                    if not isinstance(diag,loop_base_objects.LoopUVCTDiagram) and diag['type']==0:
                        reconstructedDiags=AllReconstructedDiags[diagIndex:\
                                                                 diagIndex+1]
                        diagIndex+=1
                        if checkColor:
                            this_born_colorize_obj=born_colorize_obj[bornColorDiagIndex:
                                                        bornColorDiagIndex+1]
                            this_loop_colorize_obj=[]
                            bornColorDiagIndex+=1
                    else:
                        # We assume here the number of base_object.Diagram
                        # generated is correct.
                        if not isinstance(diag,loop_base_objects.LoopUVCTDiagram):
                            # We don't want to count twice the counterterms with
                            # same ID
                            shift=len(set(vert.get('id') for vert in \
                                                       diag.get('CT_vertices')))
                        else:
                            shift=0
                        reconstructedDiags=AllReconstructedDiags[diagIndex:\
                                    diagIndex+shift+1]
                        diagIndex=diagIndex+shift+1
                        if checkColor:
                            this_loop_colorize_obj=loop_colorize_obj[loopColorDiagIndex:
                              loopColorDiagIndex+shift+1]
                            this_born_colorize_obj=[]
                            loopColorDiagIndex+=shift+1                         
                
                if verbose: print "Reconstructed loop diag :",\
                  reconstructedDiags[0].nice_string()
                # Initialize some quantities for the tests
                origDiagNVertices=len(diag['vertices'])
                structNVertices=0
                if isinstance(diag,loop_base_objects.LoopDiagram):
                    for tagElem in diag['tag']:
                        for structID in tagElem[1]:
                            # Just taking out possible vertices with id=0 for safety
                            structNVertices=structNVertices+\
                  len([vert for vert in loopAmplitude['structure_repository'][structID]['vertices']\
                       if vert['id']!=0])
                
                # Check of the loop diagram    
                self.assertEqual(len(reconstructedDiags[0]['vertices']),\
                             origDiagNVertices+structNVertices)
            
                origStructListlength=len(loopAmplitude['structure_repository'])
                if not isinstance(reconstructedDiags[0], loop_base_objects.LoopUVCTDiagram) \
                   and reconstructedDiags[0]['type']!=0:
                    start_leg=reconstructedDiags[0].get_starting_loop_line()
                    finish_leg=reconstructedDiags[0].get_finishing_loop_line()
                    reconstructedDiags[0].tag(loopAmplitude['structure_repository'],\
                      process['model'],start_leg.get('number'),finish_leg.get('number'))
                    # Then make sure it leads to the same canonical tag
                    self.assertEqual(diag['canonical_tag'],\
                                 reconstructedDiags[0]['canonical_tag'])
                    # And that the structure repository is left untouched
                    self.assertEqual(origStructListlength,\
                                 len(loopAmplitude['structure_repository']))
            
                # Check CT diagrams
                if not isinstance(diag, loop_base_objects.LoopUVCTDiagram) \
                   and diag.get('CT_vertices'):
                    # Check that the right number of counter-diagrams is produced
                    # We don't want to count twice the counterterms with same ID
                    self.assertEqual(len(reconstructedDiags)-1,
                      len(set(vert.get('id') for vert in diag.get('CT_vertices'))))
                    for ct_number in range(1,len(reconstructedDiags)):
                        if verbose: print 'reconstructed CT diag',ct_number,':',\
                          reconstructedDiags[ct_number].nice_string()
                        # Superficial check of the sanity of the Counter-terms diagrams, based
                        # on the number of vertices building them.
                        self.assertEqual(len(reconstructedDiags[ct_number]['vertices']),\
                             1+structNVertices)

                if checkColor:
                    numEntries=1
                    for vert in [v for v in diag['vertices'] if v['id']!=-1]:
                        numColorKeys=\
          len(process['model'].get('interaction_dict')[vert['id']].get('color'))
                        if numColorKeys>1:
                            numEntries*=numColorKeys
                    if not isinstance(diag, loop_base_objects.LoopUVCTDiagram) and diag['type']==0:
                        # Check born
                        if verbose:
                            print "Born diag colorization :",\
                              this_born_colorize_obj[0].keys()
                        self.assertEqual(len(this_born_colorize_obj[0]),
                          numEntries)

                        # Commented because it is only true when there is only one
                        # lorentz structure per vertex.
                        #self.assertEqual(len(myloopME['diagrams'][i]['amplitudes']),
                        #                 len(this_born_colorize_obj[0].keys()))
                        # Check that the amplitude attribute 'color_indices' is in the
                        # color keys tuples generated
                        for amp in myloopME['diagrams'][i]['amplitudes']:
                            self.assertTrue(tuple(amp['color_indices']) in \
                                        this_born_colorize_obj[0].keys())
                            # Make sure the born amplitude number appear in the color key
                            # generated
                            self.assertTrue(amp['number'] in born_amp_number_apparition)
                            # For the verbose mode, show to which color basis element each
                            # born amplitude contributes to
                            contribution_list=[]
                            for j, a in enumerate(born_color_amplitudes):
                                if amp['number'] in [b[1] for b in a]:
                                    contribution_list.append(j)
                            if verbose:
                                print "Born amplitude number",amp['number'],\
                                "contributes to the following born color basis elements",contribution_list
                    elif isinstance(diag, loop_base_objects.LoopUVCTDiagram):
                        # Check UVCT diagrams
                        if verbose:
                            print "UVCT diag colorization :",\
                              this_loop_colorize_obj[0].keys()
                        self.assertEqual(len(this_loop_colorize_obj[0]),
                          numEntries)

                        # Commented because it is only true when there is only one
                        # lorentz structure per vertex.
                        #self.assertEqual(len(myloopME['diagrams'][i]['amplitudes']),
                        #                 len(this_born_colorize_obj[0].keys()))
                        # Check that the amplitude attribute 'color_indices' is in the
                        # color keys tuples generated
                        for amp in myloopME['diagrams'][i]['amplitudes']:
                            self.assertTrue(tuple(amp['color_indices']) in \
                                        this_loop_colorize_obj[0].keys())
                            # Make sure the born amplitude number appear in the color key
                            # generated
                            self.assertTrue(amp['number'] in loop_amp_number_apparition)
                            # For the verbose mode, show to which color basis element each
                            # born amplitude contributes to
                            contribution_list=[]
                            for j, a in enumerate(loop_color_amplitudes):
                                if amp['number'] in [b[1] for b in a]:
                                    contribution_list.append(j)
                            if verbose:
                                print "UVCT amplitude number",amp['number'],\
                                "contributes to the following born color basis elements",contribution_list                        
                    else:
                        structEntries=1
                        for tagElem in diag['canonical_tag']:
                            for structID in tagElem[1]:
                                # Just taking out possible vertices with id=0
                                # for safety
                                for vert in [v for v in \
          loopAmplitude['structure_repository'][structID]['vertices'] if v['id']!=0]:
                                    numColorKeys=\
          len(process['model'].get('interaction_dict')[vert['id']].get('color'))
                                    if numColorKeys>1:
                                        structEntries*=numColorKeys
                        # Check the number of key entries for the loop diagram
                        if verbose:
                            print "Loop diag colorization :",\
                              this_loop_colorize_obj[0].keys()
                        self.assertEqual(len(this_loop_colorize_obj[0].keys()),
                          numEntries*structEntries)
                        loop_amps=[]
                        for loopamp in myloopME['diagrams'][i].get_loop_amplitudes():
                            loop_amps.extend(loopamp['amplitudes'])
                        # Commented because it is only true when there is only one
                        # lorentz structure per vertex.                        
                        #self.assertEqual(len(loop_amps),\
                        #          len(this_loop_colorize_obj[0].keys()))
                        # Check that the loop amplitudes attributes 'color_indices' are in the
                        # color keys tuples generated
                        for amp in loop_amps:
                            self.assertTrue(tuple(amp['color_indices']) in \
                              this_loop_colorize_obj[0].keys())
                            self.assertTrue(amp['number'] in loop_amp_number_apparition)
                            # For the verbose mode, show to which color basis element each
                            # loop amplitude contributes to
                            contribution_list=[]
                            for j, a in enumerate(loop_color_amplitudes):
                                if amp['number'] in [b[1] for b in a]:
                                    contribution_list.append(j)
                            if verbose:
                                print "Loop amplitude number",amp['number'],\
                                "contributes to the following loop color basis elements",contribution_list
                        # Check the number of key entries for the counter-terms                        
                        if diag.get('CT_vertices'):
                            # We don't want to count twice the counterterms with same ID
                            self.assertEqual(len(set(vert.get('id') for vert in \
                              diag.get('CT_vertices'))),len(this_loop_colorize_obj)-1)
                            # Don't forget that in get_base_amplitude the CT are given sorted
                            # according to their vertex id
                            ctverts=copy.copy(diag.get('CT_vertices'))
                            ctverts=sorted(ctverts, key=lambda vert: vert['id'])
                            for ct_number in range(1,len(reconstructedDiags)):
                                ct_vert=ctverts[ct_number-1]                                
                                nCTColor=len(process['model'].get('interaction_dict')[ct_vert['id']].get('color'))
                                if nCTColor==0: nCTColor=1
                                if verbose:
                                    print "CT diag",ct_number,"colorization :",\
                                      this_loop_colorize_obj[ct_number].keys()
                                self.assertEqual(len(this_loop_colorize_obj[ct_number].keys()),\
                                  structEntries*nCTColor)
                                ct_amps=[a for a in myloopME['diagrams'][i].get_ct_amplitudes() \
                                  if a['interaction_id']==ct_vert['id']]
                                # Commented because it is only true when there is only one
                                # lorentz structure per vertex.      
                                #self.assertEqual(len(ct_amps),\
                                #  len(this_loop_colorize_obj[ct_number].keys()))
                                # Check that the ct amplitudes attributes 'color_indices' are in the
                                # color keys tuples generated
                                for amp in ct_amps:
                                    self.assertTrue(tuple(amp['color_indices']) in \
                                    this_loop_colorize_obj[ct_number].keys())
                                    self.assertTrue(amp['number'] in loop_amp_number_apparition)
                                    # For the verbose mode, show to which color basis element each
                                    # CT amplitude contributes to
                                    contribution_list=[]
                                    for j, a in enumerate(loop_color_amplitudes):
                                        if amp['number'] in [b[1] for b in a]:
                                            contribution_list.append(j)
                                    if verbose:
                                        print "CT amplitude number",amp['number'],\
                                        "contributes to the following loop color basis elements",contribution_list
        return myloopME                                                                              
    
    def test_helas_diagrams_ddx_uux(self):
        """Test the generation of the helas diagrams for the process dd~>uu
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        self.check_HME_individual_diag_sanity(myamplitude,myproc)
                
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{'QED':0},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)

    def test_helas_diagrams_gg_gg(self):
        """Test the generation of the helas diagrams for the loop process g g > g g
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        self.check_HME_individual_diag_sanity(myamplitude,myproc)
        
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        #print "CT interaction considered=",self.myloopmodel.get_interaction(8)
        #self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc,selection=[(152,155)],verbose=True)
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)
        
    def test_helas_diagrams_gg_ggg(self):
        """Test the generation of all the helas diagrams for the loop process 
           gg > ggg. This test is quite time consuming (30 sec) so it is
           commented out by default.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        
        self.check_HME_individual_diag_sanity(myamplitude,myproc)
        
        # Skip the lengthy check for the equivalent NLO process
        return
    
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)

    def test_helas_diagrams_gg_wpwmttx(self):
        """Test the generation of all the helas diagrams for the loop process 
           gg > w+w-tt~ for which Fabio got an error in the color keys at
           helas generation time.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))

        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{'WEIGHTE':6},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        self.check_HME_individual_diag_sanity(myamplitude,myproc)
        
        # Skip the lengthy check for the equivalent NLO process
        return
        
        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{'WEIGHTED':6},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        #print "CT interaction considered=",self.myloopmodel.get_interaction(8)
        #self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc,selection=[(152,155)],verbose=True)
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)

    def test_helas_diagrams_gd_gd(self):
        """Test the generation of all the helas diagrams for the loop process 
           gd > gd.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        
        self.check_HME_individual_diag_sanity(myamplitude,myproc)

        # Skip the lengthy check for the equivalent NLO process
        #return

        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        ME=self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)
        target_lorentz=[(('VVVV1',), ('L','P0'), 1), (('FFV1',), (), 1), 
                        (('R2_GG_1', 'R2_GG_3'), (), 0), (('VVV1',), ('L','P0'), 1),
                        (('FFV1',), ('L',), 1), (('FFV1',), (), 0),
                        (('GHGHG',), ('L',), 1), (('VVV1',), (), 0), 
                        (('R2_GG_1',), (), 0), (('FFV1',), (), 2), 
                        (('GHGHG',), ('L',), 2), 
                        (('R2_GG_1', 'R2_GG_2'), (), 0), (('VVV1',), ('P0',), 1),
                        (('FFV1',), ('L','P0'), 3), (('FFV1',), ('P0',), 3),
                        (('FFV1',), ('L',), 2), (('VVVV4',), ('L','P0'), 1), 
                        (('R2_QQ_1',), (), 0), (('VVVV3',), ('L','P0'), 1)]
        self.assertEqual(set(target_lorentz),set(ME.get_used_lorentz()))

    def test_helas_diagrams_gd_ggd(self):

        """Test the generation of all the helas diagrams for the loop process 
           gd > ggd. This test is quite time consuming (30 sec) so it is
           commented out by default.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        
        self.check_HME_individual_diag_sanity(myamplitude,myproc)

        # Skip the lengthy check for the equivalent NLO process
        return

        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)

    def test_helas_diagrams_ud_ggdu(self):

        """Test the generation of all the helas diagrams for the loop process 
           ud > ggdu. This test is quite time consuming (1min) so it is
           commented out by default.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        
        self.check_HME_individual_diag_sanity(myamplitude,myproc)

        # Skip the lengthy check for the equivalent NLO process
        return

        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)
        
    def test_helas_diagrams_dxd_gz(self):

        """Test the generation of all the helas diagrams for the loop process 
           d~d > gz.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'squared_orders':{}})
    
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)
        myamplitude.generate_diagrams()
        
        self.check_HME_individual_diag_sanity(myamplitude,myproc)

        # Skip the lengthy check for the equivalent NLO process
        #return

        myloopproc = base_objects.Process({'legs':myleglist,
                                        'model':self.myloopmodel,
                                        'orders':{},
                                        'perturbation_couplings':['QCD',],
                                        'squared_orders':{}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myloopproc)
        myloopamplitude.generate_diagrams()
        
        self.check_LHME_individual_diag_sanity(myloopamplitude,myloopproc)
