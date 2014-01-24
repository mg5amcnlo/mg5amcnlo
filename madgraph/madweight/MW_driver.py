#! /usr/bin/env python
################################################################################
# Copyright (c) 2012 The MadGraph Development team and Contributors             
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
from __future__ import division
import math
import os 
import sys
import subprocess

class RunningMW(object):

    def __init__(self, card_nb, first_event, nb_events, evt_file, mw_int_points, \
                 log_level, sample_nb):
        """store the data"""
        
        self.card_nb = int(card_nb)
        self.first_event = int(first_event)
        self.evtfile = evt_file
        self.nb_events = int(nb_events) # number of events to run
        self.mw_int_points = int(mw_int_points)
        self.log_level = log_level # weight | permutation | channel | iteration | full_log
        if log_level == 'debug':
            self.log_level = 'iteration'
            self.debug = True
        else:
            self.debug = False
        self.sample_nb = int(sample_nb)
        
        self.current_event = -1
        self.last_line = ''
        self.nb_line_by_event = 0
        
        restrict_path = evt_file.replace('verif','restrict%i' % self.card_nb).replace('.lhco','.dat')
        if os.path.exists(restrict_path):
            allow = map(int, open(restrict_path).read().split())
            self.allow_event = lambda x: int(x) in allow
        else:
            self.allow_event = lambda x: True
    
    def run(self):
        """Run the computation"""

        fsock = open('param.dat','w')
        fsock.writelines('param_card_'+str(self.card_nb)+'.dat\n')
        fsock.writelines(str(self.mw_int_points)+'\n')
        fsock.close()

        self.fsock = open('output_%s_%s.xml' % (self.card_nb, self.sample_nb), 'w')
        self.fsock.write('<card id=\'%s\'>\n' % self.card_nb)
        while self.get_next_event(create=True):
            if not self.debug:
                subprocess.call('./comp_madweight', stdout=open('log.txt','w'))
            else:
                print 'submit in debug mode'
                
                os.system('echo "./comp_madweight" > log.txt')
                os.system('bash log.txt')
            self.get_one_job_result()
        self.fsock.write('</card>')
    
    def get_next_event(self, create=True, update_event_nb=True):
        """prepare the verif.lhco"""
    


        if self.current_event == -1:
            self.input_file = open(self.evtfile)
            self.current_event +=1
            for i in range(self.first_event):
                self.get_next_event(False)
        
        if update_event_nb:
            self.current_event +=1
            if self.current_event >= self.first_event + self.nb_events + 1:
                return False
        
        evt = self.last_line
        self.last_line = '' 
        if evt:
            nb_line = 1
        else:
            nb_line = 0    
        for line in self.input_file:
            nb_line +=1
            if not self.nb_line_by_event:
                if len(line.split()) == 3 and nb_line > 1:
                    self.last_line = line
                    self.nb_line_by_event = nb_line -1
                    break
                else:
                    evt += line
            else:
                evt += line
            if nb_line == self.nb_line_by_event:
                break


        if not evt:
            return False    
        

        try:
            self.lhco_number = int(evt.split('\n')[0].split()[1])
        except ValueError:
            self.lhco_number = evt.split('\n')[0].split()[1]
            evt = evt.split('\n')
            id, nblhco, trigger = evt[0].split()
            if '.' in nblhco:
                nblhco, _ = nblhco.split('.',1)
            elif ',' in nblhco:
                nblhco, _ = nblhco.split(',',1)
            nblhco = ''.join(i for i in nblhco if i.isdigit())
            if not nblhco:
                nblhco = '1'
            
            evt[0] = ' '.join([id, nblhco,trigger])
            evt = '\n'.join(evt)
        if self.allow_event(self.lhco_number):
            # now write the verif.lhco event:
            if create:
                fsock = open('verif.lhco', 'w')
                fsock.write(evt)
                fsock.close()
        else:
            return self.get_next_event(create, update_event_nb=False)
            
        return evt

    def get_one_job_result(self):
        """collect the associate result and update the final output file"""
        
        #fsock = open('output_%s_%s.xml' % (self.card_nb, self.sample_nb), 'a')
        
        weight = Weight(self.lhco_number, log_level)
        weight.get()
        weight.write(self.fsock)


class TFsets(dict):
    """ """
    nb_space=4
    def __init__(self, tf_set):
        self.value = 0
        self.error = 0
        self.tf_set = tf_set
        
        dict.__init__(self)
    
    def add(self, perm_id, channel_id, value, error, perm_order):
        
        if perm_id in self:
            perm_obj = self[perm_id]
        else:
            perm_obj = Permutation(perm_id, perm_order)
            self[perm_id] = perm_obj
        perm_obj.add(channel_id, value, error)
                
    def write(self, fsock, log_level):
        """ """
                
        self.value, self.error = self.calculate_total()
        fsock.write('%s<tfset id=\'%s\' value=\'%s\' error=\'%s\'>' % \
                        (' '*self.nb_space,self.tf_set, self.value, self.error))
        
        if log_level in ['permutation','channel', 'iterations', 'full']:
            fsock.write('\n')
            perm_ids = self.keys()
            perm_ids.sort()
            for perm_id in perm_ids:
                obj = self[perm_id]
                obj.write(fsock, log_level)
            fsock.write('%s</tfset>' % (' ' * self.nb_space))
        else:
            fsock.write('</tfset>\n')
            
    def calculate_total(self):
        
        if self.value:
            return self.value, self.error 
        total = 0
        total_error = 0
        if '0' in self.keys():
            self.value, self.error =  self['0'].calculate_total()
            return self.value, self.error
        else:
            for perm in self.values():
                value, error =  perm.calculate_total() 
                total += value
                total_error += error**2
        self.value = total / len(self)
        self.error = math.sqrt(total_error) / len(self) 
        
        return self.value, self.error

class Weight(dict):
    
    def __init__(self, lhco_number, log_level):
        self.log_level = log_level
        self.value = 0
        self.error = 0
        self.lhco_number = lhco_number
        dict.__init__(self)
        self.log = ''        
    
    def get(self):
        
        #1. get the weight, error for this object
        try:
            ff=open('weights.out','r')
        except Exception:
            return
        for line in ff:
            line = line.strip()
            if not line:
                continue
            value, error = line.split()
            self.value = float(value)
            self.error = float(error)
            break
        os.remove('weights.out')
        #2. details
        self.get_details()
            
        #3 full log
        if self.log_level == 'full':
            self.log = open('log.txt').read().replace('<','!>')
        
    def get_details(self):
        """ """
        try:
            ff=open('details.out', 'r')
        except Exception:
            return

        for line in ff:
            split = line.split()
            perm_id, channel_id, tf_id, value, error = split[:5]
            perm_order = split[5:]
            value = float(value)
            error = float(error)
            if tf_id not in self:
                tfsets = TFsets(tf_id)
                self[tf_id] = tfsets
            else:
                tfsets = self[tf_id] 
            tfsets.add(perm_id, channel_id, value, error, perm_order)
                    
    def write(self, fsock):
        """ """ 
        
        fsock.write('<event id=\'%s\' value=\'%s\' error=\'%s\'>\n' % \
                    (self.lhco_number, self.value, self.error))
        tfsets = self.keys()
        tfsets.sort()
        for tf_id in tfsets:
            self[tf_id].write(fsock, self.log_level)
        if 'full' == self.log_level:
            fsock.write('\n    <log>\n%s\n</log>\n' % self.log)#.replace('\n','\n<br></br>')) 
        fsock.write('</event>\n')        
    
    def __str__(self):
        return 'Weight(%s)' % self.value
    
    def __repr__(self):
        return 'Weight(%s)' % self.value
            
class Permutation(dict):
    nb_space=8
    def __init__(self, perm_id, perm_order):
        self.value = 0
        self.error = 0
        self.error2 = 0
        self.id = perm_id
        self.perm_order = ' '.join(perm_order)
        
        dict.__init__(self)
    
    def add(self, channel_id, value, error):
        
        self[channel_id] = Channel(channel_id, value, error) 
                
    def write(self, fsock, log_level):
        """ """
        
        self.value, self.error = self.calculate_total()
        if self.id =='0':
            tag = 'all'
        else:
            tag = self.id
        
        fsock.write('%s<permutation id=\'%s\' value=\'%s\' error=\'%s\'>\n%s%s' % \
            (' '*self.nb_space, tag, self.value, self.error,
             ' '*(self.nb_space+2), self.perm_order))
        
        if log_level in ['channel', 'iterations', 'full']:
            fsock.write('\n')
            ids = self.keys()
            ids.sort()
            for pid in ids:
                channel = self[pid]
                channel.write(fsock, log_level)
                fsock.write('\n')
            fsock.write('%s</permutation>\n' % (' '*self.nb_space))
        else:
            fsock.write('</permutation>\n')
            
    def calculate_total(self):
        
        if self.value:
            self.error = math.sqrt(self.error2)
            return self.value, self.error
        total = 0
        error = 0
        for channel in self.values():
            total += channel.value
            error += channel.error**2
        self.value = total
        self.error2 = error
        self.error = math.sqrt(self.error2)
        return total, self.error   
    
class Channel(object):
    """ """
    nb_space=12
    def __init__(self, channel_id, value, error):
        """ """
        self.channel_id = channel_id
        self.value = float(value)
        self.error = float(error)
    
    def write(self, fsock, log_level):
        
        fsock.write('%s<channel id=\'%s\' value=\'%s\' error=\'%s\'></channel>' %
                    (' '*self.nb_space,self.channel_id, self.value, self.error))

if __name__ == '__main__':
    try:                                                                                                                                                                                                   
        card_nb, first_event, nb_event, evt, mw_int_points, log_level, sample_nb = sys.argv[1:]                                                                                                            
    except:                                                                                                                                                                                                
        card_nb, first_event, nb_event, evt, mw_int_points, log_level, sample_nb = open('arguments').read().split() 
    else:
        fsock = open('arguments', 'w')
        fsock.write(' '.join(sys.argv[1:]))
        fsock.close()
    running_mw = RunningMW(card_nb, first_event, nb_event, evt, mw_int_points, log_level, sample_nb)
    running_mw.run()
