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
"""Program to combine results from channels that have been
     split into multiple jobs. Multi-job channels are identified
     by local file mjobs.dat in the channel directory.
"""
from __future__ import division
from __future__ import absolute_import
import math
import os
import shutil
import re
import logging
from six.moves import range
import random
import time


try:
    import madgraph
except ImportError:
    import internal.sum_html as sum_html
    import internal.misc as misc
    from internal import InvalidCmd, MadGraph5Error    
else:
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.misc as misc
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR

    
logger = logging.getLogger('madevent.combine_run') # -> stdout

#usefull shortcut
pjoin = os.path.join

   
def get_inc_file(path):
    """read the information of fortran inc files and returns
       the definition in a dictionary format.
       This catch PARAMETER (NAME = VALUE)"""
       
    pat = re.compile(r'''PARAMETER\s*\((?P<name>[_\w]*)\s*=\s*(?P<value>[\+\-\ded]*)\)''',
                     re.I)
        
    out = {}   
    for name, value in pat.findall(open(path).read()):
        orig_value = str(value)
        try:
            out[name.lower()] = float(value.replace('d','e'))
        except ValueError:
            out[name] = orig_value
    return out

class CombineRuns(object):
    
    def __init__(self, me_dir, subproc=None):
        
        self.me_dir = me_dir
        
        if not subproc:
            subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        self.subproc = subproc
        maxpart = get_inc_file(pjoin(me_dir, 'Source', 'maxparticles.inc'))
        self.maxparticles = maxpart['max_particles']
    
    
        for procname in self.subproc:
            path = pjoin(self.me_dir,'SubProcesses', procname)
            channels = self.get_channels(path)
            for channel in channels:
                self.sum_multichannel(channel)
    
    def sum_multichannel(self, channel):
        """Looks in channel to see if there are multiple runs that
        need to be combined. If so combines them into single run"""
       
        start = time.time()
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        if os.path.exists(pjoin(channel, 'multijob.dat')):
            njobs = int(open(pjoin(channel, 'multijob.dat')).read())
        else:
            return
        results = sum_html.Combine_results(channel)
        if njobs:
            logger.debug('find %s multijob in %s' % (njobs, channel))
        else:
            return
        for i in range(njobs):
            if channel.endswith(os.path.pathsep):
                path = channel[:-1] + alphabet[i % 26] + str((i+1)//26) 
            else:
                path = channel + alphabet[i % 26] + str((i+1)//26) 
            results.add_results(name=alphabet[i % 26] + str((i+1)//26) , 
                                filepath=pjoin(path, 'results.dat'))
        
        results.compute_average()
        if results.xsec:
            results.write_results_dat(pjoin(channel, 'results.dat'))
        else:
            return
        ### Adding information in the log file
        fsock = open(pjoin(channel, 'log.txt'), 'a')
        fsock.write('--------------------- Multi run with %s jobs. ---------------------\n'
                    % njobs)
        for r in results:
            
            fsock.write('job %s : %s %s +- %s %s\n' % (r.name, r.xsec, r.axsec,\
                                                       r.xerru, r.nunwgt))  
            
        #Now read in all of the events and write them
        #back out with the appropriate scaled weight
        to_clean = []
        fsock = open(pjoin(channel, 'events.lhe'), 'w')
        #wgt = results.axsec / results.nunwgt
        maxwgt = results.axsec / results.nunwgt 
        tot_nevents, nb_file = 0, 0
        for result in results:  
            #misc.sprint('target:', result.axsec/result.nunwgt)
            #misc.sprint('job %s : %s %s +- %s: %s' % (result.name, result.xsec, result.axsec,\
            #                                           result.xerru, result.nunwgt))
            

            ratio = result.nunwgt/results.nunwgt
            i = result.name
            if channel.endswith(os.path.pathsep):
                path = channel[:-1] + i 
            else:
                path = channel + i
            nw = self.copy_events(fsock, pjoin(path,'events.lhe'), ratio, maxwgt)
            tot_nevents += nw
            nb_file += 1
            to_clean.append(path)
        logger.debug("Combined %s file generating %s events for %s (%.1f%%): (%fs) " , nb_file, tot_nevents, channel, 100*tot_nevents/results.nunwgt, time.time()-start)
        for path in to_clean:
            try:
                shutil.rmtree(path)
            except Exception as error:
                pass
            
    @staticmethod
    def get_fortran_str(nb):
        data = '%E' % nb
        nb, power = data.split('E')
        nb = abs(float(nb)) /10
        power = int(power) + 1
        return '%.7fE%+03i' %(nb,power)    


    def copy_events(self, fsock, input, scale_wgt, max_wgt):
        """ Copy events from separate runs into one file w/ appropriate wgts"""
        
        import collections
        wgts = collections.defaultdict(int)


        do_unweight = True
        #tmp_max_wgt = 0
        #new_wgt = self.get_fortran_str(new_wgt)
        old_line = ""
        nb_evt =0 
        skip = False
        nb_read = 0
        for line in open(input):
            if old_line.startswith("<event>"):
                nb_read +=1
                data = line.split()
                if not len(data) == 6:
                    raise MadGraph5Error("Line after <event> should have 6 entries")

                new_wgt = float(data[2]) * scale_wgt
                wgts[new_wgt] +=1
                if new_wgt < 0:
                    sign = '-'
                else:
                    sign = ''
                new_wgt = abs(new_wgt)
                skip = False
                #if new_wgt > tmp_max_wgt:
                    #misc.sprint("Found event with wgt %s higher than max wgt %s. uwgt to %s " % (new_wgt, tmp_max_wgt, max_wgt))
                #    tmp_max_wgt = new_wgt
                if do_unweight and abs(new_wgt) < random.random() * max_wgt:
                    skip = True 
                else:
                    nb_evt+=1
                    if do_unweight:
                        new_wgt = max(max_wgt, new_wgt) 

                    new_wgt = self.get_fortran_str(new_wgt)
                    line= ' %s  %s%s  %s\n' % ('   '.join(data[:2]),
                                           sign, new_wgt, '  '.join(data[3:]))
            if not skip and old_line:
                fsock.write(old_line)
            old_line = line
        if not skip and old_line:
                fsock.write(old_line) 
        #misc.sprint("Read %s events, wrote %s events: %s%%" % (nb_read, nb_evt, 100*nb_evt/nb_read if nb_read else 0))
        #misc.sprint(wgts)
        return nb_evt
    
    def get_channels(self, proc_path):
        """Opens file symfact.dat to determine all channels"""
        sympath = os.path.join(proc_path, 'symfact.dat')
        
        #ncode is number of digits needed for the bw coding
        
        ncode = int(math.log10(3)*(self.maxparticles-3))+1
        channels = []
        for line in open(sympath):
            try:
                xi, j = line.split()
            except Exception:
                break
            xi, j  = float(xi), int(j)
            
            if j > 0:
                k = int(xi) 
                npos = int(math.log10(k))+1
                #Write with correct number of digits
                if xi == k:
                    dirname = 'G%i' % k
                else:
                    dirname = 'G%.{0}f'.format(ncode) % xi
                channels.append(os.path.join(proc_path,dirname))
        return channels
    
        
        
              
