################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import unittest
import tempfile
import madgraph.various.systematics as systematicsmod
import os
import models
import six
StringIO = six
import sys


_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

pjoin = os.path.join


class TestExecutors(unittest.TestCase):
    """ A class to test the executors functionality """
    
    
    def test_serial(self):


        #just test accumuate
        executor = systematicsmod.SerialExecutor()
        executor.set_functions(
            event_updater = lambda e: e,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a + e,
            finish_accumulate = lambda b,a: a)

        sum = 0
        for i in range(0,10):
            sum +=i
            executor.process(i)
        accumulate = executor.finish(True)

        self.assertEqual( sum, accumulate)

        #update the event
        executor = systematicsmod.SerialExecutor()
        executor.set_functions(
            event_updater = lambda e: e+1,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a + e,
            finish_accumulate = lambda b,a: a)

        sum = 0
        for i in range(0,10):
            sum +=i + 1
            executor.process(i)
        accumulate = executor.finish(True)

        self.assertEqual( sum, accumulate)
        
        #test finish accumulate
        executor = systematicsmod.SerialExecutor()
        executor.set_functions(
            event_updater = lambda e: e,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a+e ,
            finish_accumulate = lambda b,a: a if b else a+10)

        sum = 0
        for i in range(0,10):
            sum +=i
            executor.process(i)
        accumulate = executor.finish(False)
        sum +=10

    def test_parallel(self):

        #just test accumuate
        executor = systematicsmod.ParallelExecutor(1)
        executor.set_functions(
            event_updater = lambda e: e,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a + e,
            finish_accumulate = lambda b,a: a)

        sum = 0
        for i in range(0,10):
            sum +=i
            executor.process(i)
        accumulate = executor.finish(True)

        self.assertEqual( sum, accumulate)

        #update the event
        executor = systematicsmod.ParallelExecutor(1)
        executor.set_functions(
            event_updater = lambda e: e+1,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a + e,
            finish_accumulate = lambda b,a: a)

        sum = 0
        for i in range(0,10):
            sum +=i + 1
            executor.process(i)
        accumulate = executor.finish(True)

        self.assertEqual( sum, accumulate)
        
        #test finish accumulate
        executor = systematicsmod.ParallelExecutor(1)
        executor.set_functions(
            event_updater = lambda e: e,
            init_accumulate = lambda : 0,
            accumulator = lambda e,a : a+e ,
            finish_accumulate = lambda b,a: a if b else a+10)

        sum = 0
        for i in range(0,10):
            sum +=i
            executor.process(i)
        accumulate = executor.finish(False)
        sum +=10
        self.assertEqual( sum, accumulate)

    def test_failing_parallel(self):
        def failing_updator(e):
            raise RuntimeError("ERROR")

        executor = systematicsmod.ParallelExecutor(1)
        executor.set_functions(
            failing_updator,
            lambda : 0,
            lambda e, a : a+e,
            lambda b, a : a)

        
        for i in range(0,10):
            executor.process(i)
        accumulate = executor.finish(True)
        self.assertEqual(0, accumulate)


        def failed_accumulator(e,a):
            raise RuntimeError("ERROR")

        executor = systematicsmod.ParallelExecutor(1)
        executor.set_functions(
            lambda e : e,
            lambda : 0,
            failed_accumulator,
            lambda b, a : a)

        for i in range(0,10):
            executor.process(i)
        accumulate = executor.finish(True)
        self.assertEqual(0, accumulate)

class TestWeightsController(unittest.TestCase):
    """ A class to test the WeightsController functionality """
    def test_empty(self):
        wc = systematicsmod.WeightsController([], [], None)
        self.assertTrue(wc.get_start_wgt_id() is None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertFalse(wc.will_remove_wgts())

    def test_keep_all(self):
        wc = systematicsmod.WeightsController(remove_wgts=[], keep_wgts=['all'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertFalse(wc.will_remove_wgts())


        #keep all trumps any remove_wgts
        wc = systematicsmod.WeightsController(remove_wgts=['dummy'], keep_wgts=['all'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertTrue(wc.will_remove_wgts())

    def test_remove_all(self):
        wc = systematicsmod.WeightsController(remove_wgts=['all'], keep_wgts=[], start_id=None)
        self.assertFalse(wc.is_wgt_kept('dummy'))
        self.assertTrue(wc.will_remove_wgts())

        #keep specific trumps remove all
        wc = systematicsmod.WeightsController(remove_wgts=['all'], keep_wgts=['dummy'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertFalse(wc.is_wgt_kept('dropped'))
        self.assertTrue(wc.will_remove_wgts())

    def test_specific_keep_and_drop(self):
        # if do not explicitly remove, it is kept
        wc = systematicsmod.WeightsController(remove_wgts=[], keep_wgts=['dummy'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertTrue(wc.is_wgt_kept('dropped'))
        self.assertFalse(wc.will_remove_wgts())

        wc = systematicsmod.WeightsController(remove_wgts=['dropped'], keep_wgts=['dummy'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertFalse(wc.is_wgt_kept('dropped'))
        self.assertTrue(wc.will_remove_wgts())

        #keep trumps remove
        wc = systematicsmod.WeightsController(remove_wgts=['dummy'], keep_wgts=['dummy'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertTrue(wc.will_remove_wgts())

    def test_pattern_keep_and_drop(self):
        wc = systematicsmod.WeightsController(remove_wgts=['.ummy'], keep_wgts=['^d.*'], start_id=None)
        self.assertTrue(wc.is_wgt_kept('dummy'))
        self.assertFalse(wc.is_wgt_kept('gummy'))
        self.assertFalse(wc.is_wgt_kept('rummy'))
        self.assertFalse(wc.is_wgt_kept('summyd'))
        self.assertTrue(wc.is_wgt_kept('foo'))
        self.assertTrue(wc.will_remove_wgts())

class DummyPDF(object):
    def __init__(self):
        self.lhapdfID = 1
    def xfxQ(*k):
        return 1.
    def xfxQ2(*k):
        return 2.
        
class TestWeightCalculator(unittest.TestCase):
    """ A class to test the WeightCalculator functionality """

    def test_nb(self):
        wc = systematicsmod.WeightCalculator(0,0,
                                             None,
                                             {},
                                             0,
                                             False, 1, True,
                                             {"nb_proton1":1.0,
                                              "nb_proton2":2.0,
                                              "nb_neutron1":-1.0,
                                              "nb_neutron2":-2.0,
                                              "pdlabel":"other"},
                                             False)
        self.assertEqual(wc.get_nb_p(1),1.0)
        self.assertEqual(wc.get_nb_p(2),2.0)
        self.assertEqual(wc.get_nb_n(1),-1.0)
        self.assertEqual(wc.get_nb_n(2),-2.0)

    def test_pdfQ(self):
        wc = systematicsmod.WeightCalculator(b1=0,
                                             b2=0,
                                             alpsrunner=None,
                                             pdfsets={},
                                             only_beam=0,
                                             ion_scaling=False, 
                                             orig_pdf=1, orig_ion_pdf=False,
                                             run_card ={"nb_proton1":1.0,
                                                        "nb_proton2":2.0,
                                                        "nb_neutron1":-1.0,
                                                        "nb_neutron2":-2.0,
                                                        "pdlabel":"other"},
                                             False)
        pdf = DummyPDF()
        self.assertEqual(1 , wc.get_pdfQ(pdf, 0, 1., 1.))

        pdg =21
        x = 5.
        scale = 1.
        self.assertEqual(pdf.xfxQ(pdg, x, scale)/x , wc.get_pdfQ(pdf, pdg, x, scale))


    def test_pdfQ2(self):
        wc = systematicsmod.WeightCalculator(b1=0,
                                             b2=0,
                                             alpsrunner=None,
                                             pdfsets={},
                                             only_beam=0,
                                             ion_scaling=False, 
                                             orig_pdf=1, orig_ion_pdf=False,
                                             run_card ={"nb_proton1":1.0,
                                                        "nb_proton2":2.0,
                                                        "nb_neutron1":-1.0,
                                                        "nb_neutron2":-2.0,
                                                        "pdlabel":"other"},
                                             False)
        pdf = DummyPDF()
        self.assertEqual(1 , wc.get_pdfQ2(pdf, 0, 1., 1.))

        pdg =21
        x = 5.
        scale = 1.
        self.assertEqual(pdf.xfxQ2(pdg, x, scale)/x , wc.get_pdfQ2(pdf, pdg, x, scale))

class TestEventUpdater(unittest.TestCase):
    """ A class to test the EventUpdater functionality """
    def test_update(self):
        class DummyWeightCalculator(object):
            def __init__(self):
                self.lo_used = False
                self.nlo_used  =False
                self.cacheWasReset = False
            def get_lo_wgt(self,event,*arg):
                self.lo_used = True
                return arg[0]
            def get_nlo_wgt(self,event,*arg):
                self.nlo_used = True
                return arg[0]
            def resetCache(self):
                self.cacheWasReset = True
                self.lo_used = False
                self.nlo_used  =False

        class DummyWeightsController(object):
            def __init__(self):
                self.remove_called = False
            def remove_old_wgts(self,event):
                event.reweight_order = []
                self.remove_called = True
                
        class DummyEvent(object):
            def __init__(self):
                self.reweight_order = []
                self.id_weights = []
                self.wgt = 1
            def parse_reweight(self):
                class DummyUpdater(object):
                    def __init__(self, id_weights):
                        self.id_weights = id_weights
                    def update(self, u):
                        self.id_weights.extend(u)
                return DummyUpdater(self.id_weights)
        
        wcontroller = DummyWeightsController()
        wcalc = DummyWeightCalculator()

        #LO
        eu = systematicsmod.EventUpdater(wcontroller, wcalc, True, [[1],[2],[3]], lambda n:str(n))
        event = DummyEvent()
        eu.update_event(event)

        self.assertTrue(wcontroller.remove_called)
        self.assertTrue(wcalc.cacheWasReset)
        self.assertTrue(wcalc.lo_used)
        self.assertFalse(wcalc.nlo_used)
        self.assertEqual(event.reweight_order, ['0','1'])
        self.assertEqual(event.id_weights, [('0',2), ('1',3)])

        #NLO
        wcalc.cacheWasReset = False
        eu = systematicsmod.EventUpdater(wcontroller, wcalc, False, [[1],[2],[3]], lambda n:str(n))
        event = DummyEvent()
        eu.update_event(event)

        self.assertTrue(wcontroller.remove_called)
        self.assertTrue(wcalc.cacheWasReset)
        self.assertFalse(wcalc.lo_used)
        self.assertTrue(wcalc.nlo_used)
        self.assertEqual(event.reweight_order, ['0','1'])
        self.assertEqual(event.id_weights, [('0',2), ('1',3)])
