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
import models.import_ufo as import_ufo

import madgraph.various.misc as misc
import madgraph.various.misc as misc
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd


class DecayComparator(object):
    """base object to run comparaison test"""
    
    def __init__(self, model):
        
        self.model = model
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.exec_cmd('import model %s --modelname' % model)
        self.cmd._curr_model = import_ufo.import_model(model, decay=True)
        
        self.particles_id = dict([(p.get('name'), p.get('pdg_code'))
                                for p in self.cmd._curr_model.get('particles')])

    def compare(self, card1, card2, pid, name1, name2):
        
        def error_text(mg5_info, fr_info, pid):
            """get error text"""
            text = "%s INFORMATION:\n" % name2
            text += 'total: %s \n' % mg5_info['decay'].get((pid,)).value 
            if mg5_info['decay'].decay_table.has_key(pid):
                text += str(mg5_info['decay'].decay_table[pid])+'\n'
            text += "%s INFORMATION\n" % name1
            text += 'total: %s \n' % fr_info['decay'].get((pid,)).value 
            if fr_info['decay'].decay_table.has_key(pid):
                text += str(fr_info['decay'].decay_table[pid])+'\n'
            print text
            return text
        
        if os.path.exists(card1):
            card1 = card_reader.ParamCard(card1)
            width1 = card1['decay'].get((pid,)).value
        else:
            width1 = 0

        if os.path.exists(card2):             
            card2 = card_reader.ParamCard(card2)
            width2 = card2['decay'].get((pid,)).value
        else:
            width2 = 0
        print width1, width2
        
        
        if width1 == width2 == 0:
            return 'True'
        if (width1 - width2) / (width1 + width2) > 1e-4:
            text = error_text(card1, card2, pid)
            return text + '\n%s has not the same total width: ratio of %s' % \
                (pid, (width1 - width2) / (width1 + width2))
        
        info_partial1 = {}
        for partial_width in card1['decay'].decay_table[pid]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            info_partial1[lha_code] = partial_width.value
            
        for partial_width in card2['decay'].decay_table[pid]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            try:
                value1 = info_partial1[lha_code]
            except:
                value1 = 0
            value2 = partial_width.value
            if value1 == value2 == 0:
                continue
            elif value1 == 0 and value2/width2 < 1e-6:
                continue
            elif (value1 - value2) / (value1 + value2) > 1e-3 and \
                value2 / width2 > 1e-5:
                text = error_text(card1, card2, pid)
                return text + '\n%s has not the same partial width for %s: ratio of %s' % \
                (pid, lha_code, (value1 - value2) / (value1 + value2))
        return 'True'

        
     
    def has_same_decay(self, particle, run_fr=True):
        """create mg5 directory and then use fr to compare. Returns the ratio of
        the decay or None if this ratio is not constant for all channel.
        """
        enter_time = time.time()
        
        dir_name = 'TEST_DECAY_%s_%s' % (self.model,particle)       
        pid = self.particles_id[particle]
        
        # clean previous run
        os.system('rm -rf %s >/dev/null' % dir_name)
        os.system('rm -rf %s_dec >/dev/null' % dir_name)        
        #
        # RUN MG5
        #
        start1= time.time()
        self.cmd.exec_cmd('set automatic_html_opening False')
        try:
            self.cmd.exec_cmd('generate %s > all all --optimize' % particle)
        except InvalidCmd:
            return 'True'
        if self.cmd._curr_amps: 
            self.cmd.exec_cmd('output %s -f' % dir_name)
            
            
            files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                     '%s/Cards/run_card.dat' % dir_name, log=True)
            
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'MG5 Running time: %s s ' % (stop_mg5 -start1)
                
        #
        # Run MG Decay module
        #
        start4= time.time()
        self.cmd.exec_cmd('calculate_width %s 2' % particle)
        if self.cmd._curr_amps:  
            self.cmd.exec_cmd('output %s_dec -f' % dir_name)
            files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                     '%s_dec/Cards/run_card.dat' % dir_name, log=True)
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'DECAY Running time: %s s ' % (stop_mg5 -start4)
        
        #
        # RUN FR DECAY
        #
        if run_fr:    
            me_cmd = me_interface.MadEventCmd(dir_name)
            start3 = time.time()
            me_cmd.model_name = self.model
            me_cmd.do_compute_widths(' %s -f --nbody=2' % particle)
            stop_fr = time.time()
            print 'FR Running time: %s s ' % (stop_fr -start3)
            out1 = self.compare(pjoin(dir_name, 'Cards', 'param_card.dat'),
                         pjoin(dir_name,'Events','run_01','param_card.dat'), 
                         pid, 'FR', 'MG5')         
            out2 = self.compare(pjoin(dir_name, 'Cards', 'param_card.dat'),
                         pjoin('%s_dec' % dir_name, 'Events','run_01','param_card.dat'), 
                         pid, 'FR', 'DECAY')
              
    
            if out1 == out2 == 'True':
                os.system('rm -rf %s >/dev/null' % dir_name)
                os.system('rm -rf %s_dec >/dev/null' % dir_name)
                
                return 'True'
            else:
                return out1 + out2
        else:
            return self.compare(
                         pjoin(dir_name,'Events','run_01','param_card.dat'), 
                         pjoin('%s_dec' % dir_name, 'Events','run_01','param_card.dat'), 
                         pid, 'MG5', 'DECAY') 
        
        
        
        
        
class TestFRDecay(unittest.TestCase):
    
    def test_decay_mssm(self):
        decay_framework = DecayComparator('mssm')
        
        for i, name in enumerate(decay_framework.particles_id.keys()):
            import time
            start = time.time()
            print 'comparing decay for %s %s' % (i, name)
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
        
    def test_decay_nmssm1(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[:17]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name, False))
            print 'done in %s s' % (time.time() - start)
    
    def test_decay_nmssm2(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[17:34]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name, False))
            print 'done in %s s' % (time.time() - start)
    
    def test_decay_nmssm3(self):
        decay_framework = DecayComparator('nmssm')

        for name in decay_framework.particles_id.keys()[34:]:
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name, False))
            print 'done in %s s' % (time.time() - start)
        
    def test_decay_heft(self):
        decay_framework = DecayComparator('heft')

        for name in decay_framework.particles_id.keys():
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name, run_fr=False))
            print 'done in %s s' % (time.time() - start)        
        
    def test_decay_triplet_diquarks(self):
        decay_framework = DecayComparator('triplet_diquarks')

        for i, name in enumerate(decay_framework.particles_id.keys()):
            import time
            start = time.time()
            print 'comparing decay for %s %s' % (i, name)
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)         
        
    def test_decay_sm(self):
        decay_framework = DecayComparator('sm')

        for name in decay_framework.particles_id.keys():
            import time
            start = time.time()
            print 'comparing decay for %s' % name
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start) 