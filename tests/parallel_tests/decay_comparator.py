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
        self.cmd.exec_cmd('set automatic_html_opening False')
        self.cmd.exec_cmd('import model %s --modelname' % model)
        self.cmd._curr_model = import_ufo.import_model(model, decay=True)
        
        self.particles_id = dict([(p.get('name'), p.get('pdg_code'))
                                for p in self.cmd._curr_model.get('particles')])

    def compare(self, card1, card2, pid, name1, name2):
        
        def error_text(info1, info2, pid, name1, name2):
            """get error text"""
            text = "%s INFORMATION:\n" % name1
            text += 'total: %s \n' % info1['decay'].get((pid,)).value 
            if info1['decay'].decay_table.has_key(pid):
                text += str(info1['decay'].decay_table[pid])+'\n'
            text += "%s INFORMATION\n" % name2
            text += 'total: %s \n' % info2['decay'].get((pid,)).value 
            if info2['decay'].decay_table.has_key(pid):
                text += str(info2['decay'].decay_table[pid])+'\n'
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
            text = error_text(card1, card2, pid, name1, name2)
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
            elif abs(value1 - value2) / (value1 + value2) > 1e-3 and \
                value2 / width2 > 1e-5:
                text = error_text(card1, card2, pid, name1, name2)
                return text + '\n%s has not the same partial width for %s: ratio of %s' % \
                (pid, lha_code, (value1 - value2) / (value1 + value2))
        return 'True'

    def compare3(self, card1, card2, pid, name1, name2, twobody_decay, log,
                 raiseerror=False):
        
        fail = False
        if twobody_decay:
            twobody_key = [list(p.lhacode)[1:] for p in twobody_decay if len(p.lhacode) ==3]
            [t.sort() for t in twobody_key]
        else:
            twobody_key = []
            
        def error_text(info1, info2, pid, name1, name2):
            """get error text"""
            text = "%s INFORMATION:\n" % name1
            text += 'total: %s \n' % info1['decay'].get((pid,)).value 
            if info1['decay'].decay_table.has_key(pid):
                text += str(info1['decay'].decay_table[pid])+'\n'
            text += "%s INFORMATION\n" % name2
            text += 'total: %s \n' % info2['decay'].get((pid,)).value 
            if info2['decay'].decay_table.has_key(pid):
                text += str(info2['decay'].decay_table[pid])+'\n'
            if not raiseerror:
                print text
            else:
                raise Exception, text
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
           
        if width1 == width2 == 0:
            return 'True'
        orig_width1 = width1
        
        info_partial2 = {}
        for partial_width in card2['decay'].decay_table[pid]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            info_partial2[lha_code] = partial_width.value * width2
        
        
        
        for partial_width in card1['decay'].decay_table[pid][:]:
            lha_code = list(partial_width.lhacode)
            lha_code.sort()
            lha_code = tuple(lha_code)
            try:
                value2 = info_partial2[lha_code]
            except:
                value2 = 0
            value1 = partial_width.value * orig_width1
            
            if value1 == value2 == 0:
                continue
            elif value2 == 0 and len(lha_code) == 4:
                # check if this is a radiation
                ids = list(partial_width.lhacode)[1:]
                ids.sort()
                id1, id2, id3 = ids
                if [id1, id2] in twobody_key or \
                   [id1, id3] in twobody_key or \
                   [id2, id3] in twobody_key:
                    card1['decay'].decay_table[pid].remove(partial_width.lhacode)
                    card1['decay'].get((pid,)).value -= value1
                    width1 -= value1
                    continue
                if value2 == 0 and value1/orig_width1 < 1e-6:
                    continue
                print '\n%s has not the same partial width for %s: ratio of %s' % \
                (pid, lha_code, (value1 - value2) / (value1 + value2))
                fail = True     
            elif abs(value1 - value2) / (value1 + value2) < 5e-2 or \
                value1 / orig_width1 < 1e-5:
                continue
            elif abs(value1 - value2) / (value1 + value2) < 1e-3 or \
                value1 / orig_width1 < 1e-3:
                continue                         
            else:
                print 'fail for %s %s %s' % (lha_code,value1,value2)
                fail = True
        
        if fail:
            text = error_text(card1, card2, pid, name1, name2)
            if not raiseerror:
                return text
            else:
                raise Exception, text
        if abs(width1 - width2) / width1 + width2 > 5e-2 or \
             abs(width1 - width2) / orig_width1 + width2 > 3e-2:
            print width1, width2, abs(width1 - width2) / width1 + width2, abs(width1 - width2) / orig_width1 + width2
            print 'fail for the total cross section (orig: %s)' % orig_width1
            text = error_text(card1, card2, pid, name1, name2)
            if not raiseerror:
                return text + '\n%s has not the same total width: ratio of %s' % \
                (pid, (width1 - width2) / (width1 + width2))
            else:
                raise text
        return 'True'


        
     
    def has_same_decay(self, particle, run_fr=True):
        """create mg5 directory and then use fr to compare. Returns the ratio of
        the decay or None if this ratio is not constant for all channel.
        Check only 2 body.
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
        self.cmd.run_cmd("set automatic_html_opening False --no-save")
        try:
            self.cmd.exec_cmd('generate %s > all all --optimize' % particle)
        except InvalidCmd:
            return 'True'
        if self.cmd._curr_amps: 
            self.cmd.exec_cmd('output %s -f' % dir_name)
            
            
            files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                     '%s/Cards/run_card.dat' % dir_name, log=True)
            self.cmd.exec_cmd("set automatic_html_opening False --no-save")
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'MG5 Running time: %s s ' % (stop_mg5 -start1)
                
        #
        # Run MG Decay module
        #
        start4= time.time()
        self.cmd.run_cmd("set automatic_html_opening False --no-save")
        self.cmd.exec_cmd('calculate_width %s 2' % particle)
        if self.cmd._curr_amps:  
            self.cmd.exec_cmd('output %s_dec -f' % dir_name)
            files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                     '%s_dec/Cards/run_card.dat' % dir_name, log=True)
            print "279 launch"
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'DECAY Running time: %s s ' % (stop_mg5 -start4)
        
        #
        # RUN FR DECAY
        #
        if run_fr:    
            me_cmd = me_interface.MadEventCmd(dir_name)
            self.cmd.define_child_cmd_interface(me_cmd, False)
            start3 = time.time()
            me_cmd.model_name = self.model
            me_cmd.run_cmd("set automatic_html_opening False --no-save")
            me_cmd.do_compute_widths(' %s -f --body_decay=2' % particle)
            stop_fr = time.time()
            print 'FR Running time: %s s ' % (stop_fr -start3)
            out1 = self.compare(pjoin(dir_name, 'Cards', 'param_card.dat'),
                         pjoin(dir_name,'Events','run_01','param_card.dat'), 
                         pid, 'FR', 'MG5')         
            out2 = self.compare(pjoin(dir_name, 'Cards', 'param_card.dat'),
                         pjoin('%s_dec' % dir_name, 'Events','run_01','param_card.dat'), 
                         pid, 'FR', 'DECAY')
            me_cmd.do_quit('')
    
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
            
            
            
        
    def check_3body(self, part, multi1='all', multi2='all', multi3='all', log=None,
                    raiseerror=False):
        """checking the 3body between the decay module and standard MG5"""
        try:
            pid = self.particles_id[part]
        except:
            return 'True'
        if log:
            log = open(log,'w')
        
        to_avoid, twobody_decay = self.get_2body(part)
        to_avoid.update(['a','g', part])#, 'u','u~', 'd', 'd~','c', 'c~','s', 's~'])
        
        # Generate the MG comparison point:
        start= time.time()
        dir_name = 'TEST_DECAY3_%s_%s' % (self.model,part)
        os.system('rm -rf %s >/dev/null' % dir_name)
        os.system('rm -rf %s_dec >/dev/null' % dir_name)
        self.cmd.run_cmd('set automatic_html_opening False --no-save')
        self.cmd.exec_cmd('generate %s > %s %s %s $ all $$ %s --optimize' % 
                          (part, multi1, multi2, multi3, ' '.join(to_avoid)))
        print 'generate %s > %s %s %s $ all $$ %s --optimize' % \
                          (part, multi1, multi2, multi3, ' '.join(to_avoid))
        self.cmd.exec_cmd('history hist.cmd')
        if self.cmd._curr_amps:
            print dir_name  
            self.cmd.exec_cmd('output %s -f' % dir_name)
            #files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
            #             '%s/Cards/run_card.dat' % dir_name, log=True)
                
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'MG5 Running time: %s s ' % (stop_mg5 -start)
                
        #
        # Run MG Decay module
        #
        start4= time.time()
        self.cmd.do_compute_widths('%s --body_decay=3' % part, do2body=True)
        if self.cmd._curr_amps:  
            self.cmd.exec_cmd('output %s_dec -f' % dir_name)
            files.cp(pjoin(_file_path, 'input_files/run_card_decay.dat'),
                     '%s_dec/Cards/run_card.dat' % dir_name, log=True)
            self.cmd.exec_cmd('launch -f')
        stop_mg5 = time.time()
        print 'DECAY Running time: %s s ' % (stop_mg5 -start4)
        
        pid = self.particles_id[part]
        return self.compare3(
                         pjoin(dir_name,'Events','run_01','param_card.dat'), 
                         pjoin('%s_dec' % dir_name, 'Events','run_01','param_card.dat'), 
                         pid, 'MG5', 'DECAY', twobody_decay, log, raiseerror=raiseerror) 
        
        
    def get_2body(self, particle):
        """use FR to get the list of particles being in the 1->2 final state"""    

       
        dir_name = 'TEST_DECAY3_%s_%s' % (self.model,particle) 
        os.system('rm -rf %s >/dev/null' % dir_name)
         
        pid = self.particles_id[particle] 
        #make a fake output
        self.cmd._curr_model.write_param_card()
        self.cmd.exec_cmd('generate z > e+ e-')
        self.cmd.exec_cmd('output %s -f' % dir_name)
        me_cmd = me_interface.MadEventCmd(dir_name)
        self.cmd.define_child_cmd_interface(me_cmd, False)
        me_cmd.model_name = self.model
        me_cmd.do_compute_widths(' %s -f --body_decay=2' % particle)
        # now get the results:
        pids = set()
        path = pjoin(dir_name, 'Cards', 'param_card.dat')
        card = card_reader.ParamCard(path)
        if pid not in card['decay'].decay_table:
            return set(), {}
        for partial_width in card['decay'].decay_table[pid]:
            pids.update(list(partial_width.lhacode)[1:])
    
        labels = set()
        for id in pids:
            part = self.cmd._curr_model.get_particle(id)
            
            #if part.get('mass').lower() == 'zero':
            #    continue
            #if part.get('width').lower() == 'zero':
            #    continue            
            
            labels.add(part.get('name'))
            labels.add(part.get('antiname'))

        os.system('rm -rf %s >/dev/null' % dir_name)  
        me_cmd.do_quit('') 
        return labels, card['decay'].decay_table[pid]
    
    
        
        
class TestFRDecay(unittest.TestCase):
    
    def test_decay_mssm(self):
        decay_framework = DecayComparator('mssm')
        
        for i, name in enumerate(decay_framework.particles_id.keys()):
            import time
            start = time.time()
            print 'comparing decay for %s %s' % (i, name)
            self.assertEqual('True', decay_framework.has_same_decay(name))
            print 'done in %s s' % (time.time() - start)
        
    def notest_3body_mssm(self):
        #just for comparison
        decay_framework = DecayComparator('mssm')
        
        #add multiparticle
        decay_framework.cmd.do_define('mssm = go su1 su2 su3 su4 su5 su6 sd1 sd2 sd3 sd4 sd5 sd6 su1~ su2~ su3~ su4~ su5~ su6~ sd1~ sd2~ sd3~ sd4~ sd5~ sd6~ h02 a0 h+ sv1 sv2 sv3 sl1- sl2- sl3- sl4- sl5- sl6- w- h- sv1~ sv2~ sv3~ sl1+ sl2+ sl3+ sl4+ sl5+ sl6+ n1 n2 n3 n4 x1+ x2+ x1- x2-')
        decay_framework.cmd.do_define('sm = g u c d s u~ c~ d~ s~ a ve vm vt e- mu- ve~ vm~ vt~ e+ mu+ t b t~ b~ z w+ h01 tau+ tau-')
        mssm = 'go su1 su2 su3 su4 su5 su6 sd1 sd2 sd3 sd4 sd5 sd6 h02 a0 h+ sv1 sv2 sv3 sl1- sl2- sl3- sl4- sl5- sl6- w- h- n2 n3 n4 x1+ x2+ x1- x2-'
        
        for i, name in enumerate(mssm.split()[:]):
            import time
            start = time.time()
            print '\n****** comparing decay for %s %s *********' % (i, name)
            decay_framework.check_3body(name, 'mssm', 'sm', 'sm', log='all_mssm.log')
            print 'done in %s s' % (time.time() - start)
        
    def notest_3body_one_mssm(self):
        decay_framework = DecayComparator('mssm')
        
        #add multiparticle
        decay_framework.cmd.do_define('mssm = go su1 su2 su3 su4 su5 su6 sd1 sd2 sd3 sd4 sd5 sd6 su1~ su2~ su3~ su4~ su5~ su6~ sd1~ sd2~ sd3~ sd4~ sd5~ sd6~ h02 a0 h+ sv1 sv2 sv3 sl1- sl2- sl3- sl4- sl5- sl6- w- h- sv1~ sv2~ sv3~ sl1+ sl2+ sl3+ sl4+ sl5+ sl6+ n1 n2 n3 n4 x1+ x2+ x1- x2-')
        decay_framework.cmd.do_define('sm = g u c d s u~ c~ d~ s~ a ve vm vt e- mu- ve~ vm~ vt~ e+ mu+ t b t~ b~ z w+ h01 tau+ tau-')
        mssm = 'su1 su2 su3 su4 su5 su6 sd1 sd2 sd3 sd4 sd5 sd6 h02 a0 h+ sv1 sv2 sv3 sl1- sl2- sl3- sl4- sl5- sl6- w- h- n1 n2 n3 n4 x1+ x2+ x1- x2-'
        

        import time
        start = time.time()
        #print 'comparing decay for %s %s' % (i, name)
        decay_framework.check_3body('n3', 'mssm', 'sm', 'sm', log='one_mssm.log')
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