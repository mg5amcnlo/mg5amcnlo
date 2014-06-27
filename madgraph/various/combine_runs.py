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
import math
import os
import re
import logging

try:
    import madgraph.various.sum_html as sum_html
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR
except ImportError:
    import internal.sum_html as sum_html
    from internal import InvalidCmd, MadGraph5Error
    
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
                path = channel[:-1] + alphabet[i] 
            else:
                path = channel + alphabet[i] 
            results.add_results(name=alphabet[i], 
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
            fsock.write('job %s : %s %s %s\n' % (r.name, r.xsec, r.axsec, r.nunwgt))  
            
        #Now read in all of the events and write them
        #back out with the appropriate scaled weight
        fsock = open(pjoin(channel, 'events.lhe'), 'w')
        wgt = results.axsec / results.nunwgt
        tot_nevents=0
        for result in results:  
            i = result.name
            if channel.endswith(os.path.pathsep):
                path = channel[:-1] + i 
            else:
                path = channel + i 
            nw = self.copy_events(fsock, pjoin(path,'events.lhe'), wgt)
            #tot_events += nw
        #logger.debug("Combined %s events to %s " % (tot_events, channel))


    def copy_events(self, fsock, input, new_wgt):
        """ Copy events from separate runs into one file w/ appropriate wgts"""
        
        def get_fortran_str(nb):
            data = '%E' % nb
            nb, power = data.split('E')
            nb = abs(float(nb)) /10
            power = int(power) + 1
            return '%.7fE%+03i' %(nb,power)
        new_wgt = get_fortran_str(new_wgt)
        old_line = ""
        for line in open(input):
            if old_line.startswith("<event>"):
                data = line.split()
                if not len(data) == 6:
                    raise MadGraph5Error, "Line after <event> should have 6 entries"
                if float(data[2]) > 0:
                    sign = ''
                else:
                    sign = '-'
                    
                line= ' %s  %s%s  %s\n' % ('   '.join(data[:2]), sign,
                                           new_wgt, '  '.join(data[3:]))
            fsock.write(line)
            old_line = line
           
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
    
        
        
              
