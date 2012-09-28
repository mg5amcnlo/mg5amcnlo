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
"""A Test suite in order to compare the width computed by MG5 to those provided
by FR in the decays.py files of the model.
"""
from __future__ import division

import logging
import os
import shutil
import unittest
import subprocess
import time

pjoin = os.path.join
# Get the grand parent directory (mg5 root) of the module real path 
# (tests/acceptance_tests) and add it to the current PYTHONPATH to allow
# for easy import of MG5 tools

_file_path = os.path.dirname(os.path.realpath(__file__))

import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.files as files

import madgraph.interface.master_interface as cmd_interface
import madgraph.interface.madevent_interface as me_interface
import models.check_param_card as card_reader

import madgraph.various.misc as misc
import madgraph.various.misc as misc
from madgraph import MadGraph5Error, MG5DIR


class DecayComparator(object):
    """base object to run comparaison test"""
    
    def __init__(self, model):
        
        self.model = model
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.exec_cmd('import model %s --modelname' % model)
        
        self.particles_id = dict([(p.get('name'), p.get('pdg_code'))
                                for p in self.cmd._curr_model.get('particles')])
        
    def has_same_decay(self, particle):
        """create mg5 directory and then use fr to compare. Returns the ratio of
        the decay or None if this ratio is not constant for all channel.
        """
        enter_time = time.time()
        def error_text(mg5_info, fr_info, pid):
            """get error text"""
            text = "MG5 INFORMATION:\n"
            text += 'total: %s ' % mg5_info['decay'].get((pid,)).value 
            if mg5_info['decay'].decay_table.has_key(pid):
                text += str(mg5_info['decay'].decay_table[pid])+'\n'
            text += "FR INFORMATION\n"
            text += 'total: %s ' % fr_info['decay'].get((pid,)).value 
            if fr_info['decay'].decay_table.has_key(pid):
                text += str(fr_info['decay'].decay_table[pid])+'\n'
            print text
            return text
        
        dir_name = 'TEST_DECAY_%s_%s' % (self.model,particle)       
        pid = self.particles_id[particle]
        
        # clean previous run
        os.system('rm -rf %s >/dev/null' % dir_name)
        
        #
        # RUN MG5
        #
        start1= time.time()
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('generate %s > all all' % particle)

        self.cmd.exec_cmd('output %s -f' % dir_name)
        
        
        files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                 '%s/Cards/run_card.dat' % dir_name, log=True)
        
        self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'MG5 Running time: %s s ' % (stop_mg5 -start1)
        mg5_info = card_reader.ParamCard(pjoin(dir_name,'Events','run_01','param_card.dat'))
        
        #
        # RUN FR DECAY
        #
                
        me_cmd = me_interface.MadEventCmd(dir_name)
        start3 = time.time()
        me_cmd.exec_cmd('compute_widths %s -f' % particle)
        stop_fr = time.time()
        fr_info = card_reader.ParamCard(pjoin(dir_name, 'Cards', 'param_card.dat'))
        print 'FR Running time: %s s ' % (stop_fr -start3)
        
        
        # check the total width
        mg5_width = mg5_info['decay'].get((pid,)).value
        fr_width = fr_info['decay'].get((pid,)).value
        print mg5_width, fr_width
        
        if mg5_width == fr_width == 0:
            return 'True'
        elif (mg5_width - fr_width) / (mg5_width + fr_width) > 1e-4:
            text = error_text(mg5_info, fr_info, pid)
            return text + '\n%s has not the same total width: ratio of %s' % \
                (particle, (mg5_width - fr_width) / (mg5_width + fr_width))
        
        mg5_info_partial = {}
        for partial_width in mg5_info['decay'].decay_table[pid]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            mg5_info_partial[lha_code] = partial_width.value
            
        for partial_width in fr_info['decay'].decay_table[pid]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            mg5_value = mg5_info_partial[lha_code]
            fr_value = partial_width.value
            if mg5_value == fr_value == 0:
                continue
            elif (mg5_value - fr_value) / (mg5_value + fr_value) > 1e-3 and \
                mg5_value / mg5_width > 1e-5:
                text = error_text(mg5_info, fr_info, pid)
                return text + '\n%s has not the same partial width for %s: ratio of %s' % \
                (particle, lha_code, (mg5_value - fr_value) / (mg5_value + fr_value))
                return False            
            
        os.system('rm -rf %s >/dev/null' % dir_name)
        return 'True'
        
        
        
        
class TestFRDecay(unittest.TestCase):
    
    def test_decay_mssm(self):
        decay_framework = DecayComparator('mssm')
        
        for name in decay_framework.particles_id.keys():
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
        
    def test_decay_nmssm1(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[:17]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
    
    def test_decay_nmssm2(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[17:34]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
    
    def test_decay_nmssm3(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[34:]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
        
    def test_decay_heft(self):
        decay_framework = DecayComparator('Higgs_Effective_Couplings_UFO')

        for name in decay_framework.particles_id.keys():
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)        
        
    def test_decay_triplet_diquarks(self):
        decay_framework = DecayComparator('triplet_diquarks')

        for name in decay_framework.particles_id.keys():
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)         
        
