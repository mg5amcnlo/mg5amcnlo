#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##   email:  olivier.mattelaer@uclouvain.be                             ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:25/03/08                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Note                                                               ##
##   ----                                                               ##
##                                                                      ##
##     1) All local  routines must be launched from schedular           ##
##     2) All global routines must be launched from bin                 ## 
##     3) All routines wich start in a directory must finish in the     ##
##            same directory                                            ## 
##                                                                      ##
##########################################################################
##
## BEGIN INCLUDE
##
from __future__ import division
import string
import os
import sys
import re,math
import time
from math import log,exp,sqrt
from MW_param import MW_info,go_to_main_dir
from clean import Clean_event
import progressbar
##
## END INCLUDE
##

##
## Object to collect the information
##
class One_Cross_Result(dict):
    """One cross section result"""
    conv=1/(389379.66*1000) #conv pb -> Gev^-2
    
    def __init__(self,card=0, value=0, error=0):
        """ Save the data"""
        
        self['card'] = int(card)
        self['value'] = float(value) * self.conv
        self['error'] = float(error) * self.conv
        self['event'] = -1
        
    def get_key(self):
        """return a unique identifier of the event"""
        return "%s" % (self['card'])
        
    def update_value(self, other):
        
        assert self.get_key() == other.get_key()
        
        self['value'] += other['value']
        self['error'] = math.sqrt(self['error']**2 + other['error']**2)

    def __str__(self, format='%(card)s\t%(value)s\t%(error)s\n'):
        return format % self

class One_Weight_Result(One_Cross_Result):
    
    conv = 1
     
    def __init__(self,card=0, value=0, error=0, event=0, lhco_event=0, trigger=0):
        """ Save the data"""
        
        One_Cross_Result.__init__(self, card=card, value=value, error=error)
        self['event'] = int(event)
        self['lhco_event'] = int(lhco_event)
        self['trigger'] = int(trigger)
        
    def get_key(self):
        """return a unique identifier of the event"""
        return "%s %s %s" % (self['card'], self['event'], self['lhco_event'])
    
    def normalize(self, cross):
        """ divide result by cross-section result """
        
        
        
        weight = self['value'] + 1e-99
        error = self['error']
        xsec = cross['value']  + 1e-99
        xsec_err = cross['error']
            
        self['value'] /= (xsec) 
        self['error'] = weight*math.sqrt(xsec_err**2/xsec**2+error**2/weight**2) #uncorrelated error

    def __str__(self, format='%(card)s.%(event)s\t%(value)s\t%(error)s\n'):
        return format % self


class Weight_results(dict):
    One_Result = One_Weight_Result
            
    def add(self, *arg, **opt):

        result = self.One_Result(*arg, **opt)
        self.add_result(result)
    
    def add_result(self, result):
        key = result.get_key()
        if key in self:
            self[key].update_value(result)
        else:
            self[key] = result       
            
    def normalize(self, cross):
        """ normalize result"""

        for result in self.values():
            result.normalize(cross['%s' % result['card']])
  
  
    def sorting(self, first, second):
        card1, event1, lhco1 = first.split()
        card2, event2, lhco2 = second.split()
        card1, card2 = int(card1), int(card2)
        if card1 < card2:
            return -1
        elif card1 > card2:
            return 1
        event1, event2 = int(event1), int(event2)
        if event1 < event2:
            return -1
        elif event1 > event2:
            return 1
        if lhco1 < lhco2:
            return -1
        else:
            return 1  
            
    def write(self, outputpath, format):
        """ """
        ff = open(outputpath,'w')
        keys = [key for key in self.keys()]
        keys.sort(self.sorting)
        for key in keys:
            ff.writelines(self[key].__str__(format))
            
class Cross_results(Weight_results):
    One_Result = One_Cross_Result

    def sorting(self, first, second):
        card1, card2 = int(first), int(second)
        if card1 < card2:
            return -1
        elif card1 > card2:
            return 1

################################################################################
##  Collect Weight information for a single run
################################################################################
class collect_weight(One_Weight_Result):
    """Collect Weight information for a single run"""

    ############################################################################
    def __init__(self, path, card, event, mode=1):
        """ collect information in a given path return One_Weight_Result"""

        self.path = path
        if mode == 1:
            value, error = self.read_from_output()
        elif mode == 2:
            #select a particular permutation [default the first]
            value, error = self.select_perm(1)
        elif mode == 3:
            # select the best permutation only as the weight
            value, error = self.choose_bestperm()
        elif mode == 4: 
            #recreate the weights.out from details.out
            value, error = self.find_from_details()
        
        if (float(error) < 0):
            print "instabality in %s resolve at" % path, 
            value, error = self.extract_from_vegas_value()
        

        lhco_event, trigger = self.get_lhco_information()
            
        One_Weight_Result.__init__(self, card, value, error, event, lhco_event, trigger)
    
    ############################################################################
    def get_lhco_information(self):
        """ read the lhco event number and associate trigger """
        for line in open(os.path.join(self.path,'verif.lhco')):
            data = line.split()
            if data[0] == '0':
                return int(data[1]), int(data[2])
        return 0,0

    ############################################################################
    def read_from_output(self):
        """ find the weight/error from the file weights.out
        if not exist find from the permutation (they are problems with weights.out in some case)
        """
        try:
            ff=open(os.path.join(self.path,'weights.out'),'r')
        except:
            input=list(self.find_from_details())
            return input
        
        #read weight
        line=ff.readline()
        input=line.split()
        while 1:
            try:
                input.remove('')
            except:
                break
        return input

    ############################################################################
    def choose_bestperm(self):
        """ select the best permutation for the weight"""

        ff=open(os.path.join(self.path,'details.out','r'))
        pat=re.compile(r'''^\s*\d+\s*\|\|\s*\d+\s*\|\|\s*(?P<value>[\de.+-]+)\s*\|\|\s*(?P<error>[\deE.+-]+)\s*$''',re.I)
    
        max=0
        i=0
        for line in ff:
            if pat.search(line):
                i+=1
                value=float(pat.search(line).group('value'))
                if value>max:
                    max=value
                    error=pat.search(line).group('error')
                
        return max/i,error/i

    ############################################################################
    def find_from_details(self):
        """ recompute the weights.out from details.out """

        try:
            ff=open(os.path.join(self.path,'details.out','r'))
        except:
            return 0, 0
        
        pat=re.compile(r'''^\s*\d+\s*\|\|\s*\d+\s*\|\|\s*(?P<value>[\de.+-na]+)\s*\|\|\s*(?P<error>[\de.+-na]+)\s*$''',re.I)

        i=0
        total=0
        total_err=0
        for line in ff:
            if pat.search(line):
                i+=1
                value=pat.search(line).group('value').lower()
                err=float(pat.search(line).group('error'))
                if value in ['nan','NAN'] or float(value)==0:
                    value=0
                    err=0
                        

                total+=float(value)
                total_err+=(err)**2
        ff.close()
        gg=open(os.path.join(self.path,'weights.out'),'w')
        gg.writelines(str(total/i)+'\t'+str(sqrt(total_err)/i))
                    
        return total/i,sqrt(total_err)/i

    #2########################################################################                
    def select_perm(self,perm):
        """ select the permutation value for the weight"""

        ff=open(os.path.join(self.path,'details.out'),'r')
        pat=re.compile(r'''^\s*\d+\s*\|\|\s*\d+\s*\|\|\s*(?P<value>[\de.+-]+)\s*\|\|\s*(?P<error>[\deE.+-]+)\s*$''',re.I)
    
        i=0
        for line in ff:
            if pat.search(line):
                i+=1
                if i==perm:
                    value=float(pat.search(line).group('value'))
                    error=pat.search(line).group('error')
                
        return max/i,error/i        

    #2########################################################################
    def extract_from_vegas_value(self):

        def compare_precision(data):
            """ data: list of double (cross,err) """

            rel_err=[data[i][1]/data[i][0] for i in range(0,len(data)) if data[i][0]!=0.0]
            data2=[ [data[i][0], data[i][1]] for i in range(0,len(data)) if data[i][0]!=0.0]
            if rel_err:
                best= [data2[i] for i in range(0,len(data2)) if min(rel_err)==rel_err[i]]
            else:
                return [0.0,0.0]

            if best:
                return best[0]
            else:
                return [0.0,0.0]

        def find_cross(data):
            """ data: list of double (cross,err) """
            Soversig=0
            inv_sq_err=0
            data_out=[]
            for i in range(0,len(data)):
                cross,err=data[i][0],data[i][1]
                if err==0:
                    continue
                Soversig+=cross/err**2
                inv_sq_err+=1.0/err**2
                data_out.append([Soversig/inv_sq_err,math.sqrt(1.0/inv_sq_err)])
            return compare_precision(data_out)
        
        #### START ROUTINE
        out=[{},{},{},{},{}] #one entry by step
        data_store=0
        for line in open(os.path.join(self.path,'vegas_value.out')):
            data=line.split()
            if len(data)==8:            
                step,perm_pos,config,calls,it,err_it,cross,sd=data
            elif data_store:
                data=data_store+data
                step,perm_pos,config,calls,it,err_it,cross,sd=data
                data_store=[]
            else:
                data_store=list(data)
                continue
           
            if out[int(step)-1].has_key(int(perm_pos)):
                out[int(step)-1][int(perm_pos)].append([float(it),float(err_it)])
            else:
                out[int(step)-1][int(perm_pos)]=[[float(it),float(err_it)]]

        total=0
        total_err=0                     
        out_value={}
        for perm in out[0].keys():
            data=[]
            data_out=[]
            for i in range(4,-1,-1):
                try:
                    data+=out[i][perm]
                except:
                    continue
                data_out.append(list(find_cross(data)))
            end_result=compare_precision(data_out)
            out_value[perm]=end_result
            total+=end_result[0]
            total_err+=end_result[1]**2


        num_per=len(out[0])
        print sqrt(total_err)/total*100,'%'
        return total/num_per,sqrt(total_err)/num_per

################################################################################
##  Collect Weight information for a single run
################################################################################
class collect_cross(One_Cross_Result):
    """Collect Weight information for a single run"""

    ############################################################################
    def __init__(self, path, card):
        """ collect information in a given path return One_Weight_Result"""

        try:
            line = open(os.path.join(path, 'results.dat')).readline()
        except:
            One_Cross_Result.__init__(self,card, value=0, error=0)
            return
    
        data=line.split()
        value = data[0]
        error = sqrt(float(data[1])**2+float(data[2])**2)
                
        One_Cross_Result.__init__(self, card, value, error)


################################################################################
##  Multi_Collector_Class
################################################################################
class Multi_Collector(dict):
    collector_class = None # Should be overwrite
    collector_routine = None # Should be overwrite
    
    def __init__(self,*arg,**opt):
        dict.__init__(self, *arg, **opt)
        self.inactive = {}
    
    def add_collector(self, name):
        if name in self:
            raise Exception('name already use')
        self[name] = self.collector_class()
    
    def put_has_inactive(self, name):
        """ """
        self.inactive[name] = self[name]
        del self[name]
    
    def write_result(self, name, output_path, format=None):
        format = format.decode('string_escape')
        self[name].write(output_path, format)
    
    def collect_one(self, *arg, **opt):
        """ """

        self.collector_routine(*arg)
        out = self.collector_routine(*arg, **opt)
        for obj in self.values():
            obj.add_result(out)
        return
    
    ############################################################################ 
    def define_fail(self, name):
        """ define the list of job to relaunch """
       
        minimal_precision = self.MWparam.run_opt['refine']
        failed = []
        for result in self[name].values():
            if  not result['value']:
                failed.append([result['card'],result['event']])
        if len(failed):
            print "WARNING: %s zero result detected in %s" % (len(failed), name) 
       
       
        if minimal_precision:
            for result in self[name].values():
                if result['value'] and result['error']/result['value'] > float(minimal_precision):
                    failed.append([result['card'],result['event']])
            print "%s result to refine in %s" % (len(failed), name)
        
        self.write_failed(name, failed)
        return failed
    
    def write_failed(self, name, failed):
        """ write in a file the failed subprocess """

        path = './SubProcesses/'+name+'/'+self.run_name+'/failed_job.dat'
        ff=open(path,'w')
        for card, event in failed:
            ff.writelines('%s %s\n' %(card, event))
        ff.close()
    
#################################################################################
###  Collect All MW Dir
#################################################################################    
class Collect_All_MW(Multi_Collector):
    """A Class for collecting all Weights"""
    
    collector_class = Weight_results
    collector_routine = collect_weight

    ############################################################################ 
    def __init__(self, MWparam, keep=None):
            
        Multi_Collector.__init__(self)
        
        self.MWparam=MWparam #MW_info object
        self.run_name=self.MWparam.name

        self.add_collector('all')
        old = None

        for MW_dir in self.MWparam.MW_listdir:
            # treat collector
            if keep and old:
                self.put_has_inactive(old)
            elif old:
                del self[old]
                
            old = MW_dir
            self.add_collector(MW_dir)

            # Make the collection
            path = './SubProcesses/'+MW_dir+'/'+self.run_name+'/'
            self.collect_dir(path)
            
            # Write the Local dir result
            output_path = os.path.join(path, self.run_name+'_weights.out')
            self.write_result(MW_dir, output_path, self.MWparam['mw_run']['weight_format'])
            self.define_fail(MW_dir) # write failing job
    

        output_path = os.path.join('./Events',self.run_name, self.run_name+'_weights.out')
        self.write_result('all', output_path, self.MWparam['mw_run']['weight_format'])

    ############################################################################ 
    def collect_dir(self, path):
        """ collect_one_directory """
# Pierre: change the tag MW -> P
        print [name for name in path.split('/') if name.startswith('P')]
        dir_name = [name for name in path.split('/') if name.startswith('P')][-1]
        pbar = progressbar.progbar('Collect %s' % dir_name, 
                    len(self.MWparam.actif_param) * 
                    (self.MWparam.nb_event_MW[dir_name] - self.MWparam.startevent))
        for nb_card in self.MWparam.actif_param:
            for nb_event in range(self.MWparam.startevent, self.MWparam.nb_event_MW[dir_name]):
                cur_path = path + '/card_'+str(nb_card)+'/event_'+str(nb_event)
                pbar.update()

                if not os.path.exists(cur_path):
                    print 'WARNING: no directory: ',cur_path
                    print 'stop to collect for this SubProcesses'
                    pbar.finish()
                    return
                self.collect_one(cur_path, nb_card, nb_event, mode=self.MWparam['mw_perm']['combine_mode'])
        pbar.finish()
################################################################################
##  Collect All MadEvent Dir
################################################################################    
class Collect_All_ME(Multi_Collector):
    """A class for colecting All cross-sections results"""
    
    collector_class = Cross_results
    collector_routine = collect_cross

    ############################################################################ 
    def __init__(self, MWparam, keep=False):

        Multi_Collector.__init__(self)
        self.MWparam=MWparam #MW_info object
        self.run_name=self.MWparam.name

        self.add_collector('all')
        old = None
        for P_dir in self.MWparam.P_listdir:
            # treat collector
            if keep and old:
                self.put_has_inactive(old)
            elif old:
                del self[old]
            old = P_dir
            self.add_collector(P_dir)

            # Make the collection
            path = './SubProcesses/'+P_dir+'/'+self.run_name+'/'
            self.collect_dir(path)
            
            # Write the Local dir result
            output_path = os.path.join(path, 'cross.out')
            self.write_result(P_dir, output_path, self.MWparam['mw_run']['cross_format'])
            self.define_fail(P_dir) # write failing job
    
        output_path = os.path.join('./Events', self.run_name, self.run_name+'_cross_weights.out')
        self.write_result('all', output_path, self.MWparam['mw_run']['cross_format'])
    
    ############################################################################ 
    def collect_dir(self, path):
        """ collect_one_directory """
#Pierre
        dir_name = [name for name in path.split('/') if name.startswith('XXX')][-1]
        pbar = progressbar.progbar('Collect %s' % dir_name, 
                    len(self.MWparam.actif_param))

        for nb_card in self.MWparam.actif_param:
            cur_path = path + '/card_'+str(nb_card)
            pbar.update()
            if not os.path.exists(path):
                print 'WARNING: no directory: ',path
                print 'stop to collect for this SubProcesses'
                pbar.finish()
                break
            self.collect_one(cur_path, nb_card)
        pbar.finish()

################################################################################
##  Collect All Directory
################################################################################    
class Collect_All(Multi_Collector):
    """A class for colecting All results"""
    
    def __init__(self, MWparam):

        weight = Collect_All_MW(MWparam)
        if not MWparam.norm_with_cross:
            return

        # Pierre
        print "Total cross sections are not evaluated"
        print "You need to normalize the weights yourself! "        
        #cross = Collect_All_ME(MWparam)
        #weight['all'].normalize(cross['all']) 
        #outpath = './Events/%(run)s/%(run)s_norm_weights.out' % {'run':MWparam.name} 
        #weight.write_result('all', outpath, MWparam['mw_run']['weight_format'])

##########################################################################
###                    START CODE
##########################################################################
def collect_schedular(MW_param):

    go_to_main_dir()
    Collect_All(MW_param)




############################################################################
####################             TEST                      ##################
#############################################################################           
#if __name__=='__main__':
#
#    print os.getcwd()
#    go_to_main_dir()
#
#    MW_content=MW_info('MadWeight_card.dat')
#    #collect_schedular(MW_content)
#    Collect_All_dir(MW_content)
#
#    
#    
#        
