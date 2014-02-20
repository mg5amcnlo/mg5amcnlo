################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import division
import os
import math
import logging
import re
logger = logging.getLogger('madevent.stdout') # -> stdout

pjoin = os.path.join
try:
    import madgraph.various.cluster as cluster
except ImportError:
    import internal.cluster as cluster

class OneResult(object):
    
    def __init__(self, name):
        """Initialize all data """
        
        self.name = name
        self.axsec = 0  # Absolute cross section = Sum(abs(wgt))
        self.xsec = 0 # Real cross section = Sum(wgt)
        self.xerru = 0  # uncorrelated error
        self.xerrc = 0  # correlated error
        self.nevents = 0
        self.nw = 0     # Don't know
        self.maxit = 0  # 
        self.nunwgt = 0  # number of unweighted events
        self.luminosity = 0
        self.mfactor = 1 # number of times that this channel occur (due to symmetry)
        self.ysec_iter = []
        self.yerr_iter = []
        self.yasec_iter = []
        self.eff_iter = []
        self.maxwgt_iter = []
        return
    
    @cluster.multiple_try(nb_try=5,sleep=20)
    def read_results(self, filepath):
        """read results.dat and fullfill information"""
        
        i=0
        for line in open(filepath):
            i+=1
            if i == 1:
                def secure_float(d):
                    try:
                        return float(d)
                    except ValueError:
                        m=re.search(r'''([+-]?[\d.]*)([+-]\d*)''', d)
                        if m:
                            return float(m.group(1))*10**(float(m.group(2)))
                        return 
                    
                data = [secure_float(d) for d in line.split()]
                self.axsec, self.xerru, self.xerrc, self.nevents, self.nw,\
                         self.maxit, self.nunwgt, self.luminosity, self.wgt, self.xsec = data[:10]
                if self.mfactor > 1:
                    self.luminosity /= self.mfactor
                    #self.ysec_iter.append(self.axsec)
                    #self.yerr_iter.append(0)
                continue
            try:
                l, sec, err, eff, maxwgt, asec = line.split()
            except:
                return
            self.ysec_iter.append(secure_float(sec))
            self.yerr_iter.append(secure_float(err))
            self.yasec_iter.append(secure_float(asec))
            self.eff_iter.append(secure_float(eff))
            self.maxwgt_iter.append(secure_float(maxwgt))
        # this is for amcatnlo: the number of events has to be read from another file
        if self.nevents == 0 and self.nunwgt == 0 and \
                os.path.exists(pjoin(os.path.split(filepath)[0], 'nevts')): 
            nevts = int(open(pjoin(os.path.split(filepath)[0], 'nevts')).read())
            self.nevents = nevts
            self.nunwgt = nevts
        
        
    def set_mfactor(self, value):
        self.mfactor = int(value)    
        
    def change_iterations_number(self, nb_iter):
        """Change the number of iterations for this process"""
            
        if len(self.ysec_iter) <= nb_iter:
            return
        
        # Combine the first iterations into a single bin
        nb_to_rm =  len(self.ysec_iter) - nb_iter
        ysec = [0]
        yerr = [0]
        for i in range(nb_to_rm):
            ysec[0] += self.ysec_iter[i]
            yerr[0] += self.yerr_iter[i]**2
        ysec[0] /= (nb_to_rm+1)
        yerr[0] = math.sqrt(yerr[0]) / (nb_to_rm + 1)
        
        for i in range(1, nb_iter):
            ysec[i] = self.ysec_iter[nb_to_rm + i]
            yerr[i] = self.yerr_iter[nb_to_rm + i]
        
        self.ysec_iter = ysec
        self.yerr_iter = yerr


class Combine_results(list, OneResult):
    
    def __init__(self, name):
        
        list.__init__(self)
        OneResult.__init__(self, name)
    
    def add_results(self, name, filepath, mfactor=1):
        """read the data in the file"""
        oneresult = OneResult(name)
        oneresult.set_mfactor(mfactor)
        oneresult.read_results(filepath)
        self.append(oneresult)
    
    
    def compute_values(self):
        """compute the value associate to this combination"""

        self.compute_iterations()
        self.axsec = sum([one.axsec for one in self])
        self.xsec = sum([one.xsec for one in self])
        self.xerrc = sum([one.xerrc for one in self])
        self.xerru = math.sqrt(sum([one.xerru**2 for one in self]))

        self.nevents = sum([one.nevents for one in self])
        self.nw = sum([one.nw for one in self])
        self.maxit = len(self.yerr_iter)  # 
        self.nunwgt = sum([one.nunwgt for one in self])  
        self.wgt = 0
        self.luminosity = min([0]+[one.luminosity for one in self])
        
        
        
        
    def compute_average(self):
        """compute the value associate to this combination"""

        nbjobs = len(self)
        if not nbjobs:
            return
        self.axsec = sum([one.axsec for one in self]) / nbjobs
        self.xsec = sum([one.xsec for one in self]) /nbjobs
        self.xerrc = sum([one.xerrc for one in self]) /nbjobs
        self.xerru = math.sqrt(sum([one.xerru**2 for one in self])) /nbjobs

        self.nevents = sum([one.nevents for one in self])
        self.nw = 0#sum([one.nw for one in self])
        self.maxit = 0#len(self.yerr_iter)  # 
        self.nunwgt = sum([one.nunwgt for one in self])  
        self.wgt = 0
        self.luminosity = sum([one.luminosity for one in self])
        self.ysec_iter = []
        self.yerr_iter = []
        for result in self:
            self.ysec_iter+=result.ysec_iter
            self.yerr_iter+=result.yerr_iter
            self.yasec_iter += result.yasec_iter
            self.eff_iter += result.eff_iter
            self.maxwgt_iter += result.maxwgt_iter

    
    def compute_iterations(self):
        """Compute iterations to have a chi-square on the stability of the 
        integral"""

        nb_iter = min([len(a.ysec_iter) for a in self], 0)
        # syncronize all iterations to a single one
        for oneresult in self:
            oneresult.change_iterations_number(nb_iter)
            
        # compute value error for each iteration
        for i in range(nb_iter):
            value = [one.ysec_iter[i] for one in self]
            error = [one.yerr_iter[i]**2 for one in self]
            
            # store the value for the iteration
            self.ysec_iter.append(sum(value))
            self.yerr_iter.append(math.sqrt(sum(error)))
    
       
    template_file = \
"""  
%(diagram_link)s
 <BR>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>s= %(cross).5g &#177 %(error).3g (%(unit)s)</b><br><br>
<table class="sortable" id='tablesort'>
<tr><th>Graph</th>
    <th> %(result_type)s</th>
    <th>Error</th>
    <th>Events (K)</th>
    <th>Unwgt</th>
    <th>Luminosity</th>
</tr>
%(table_lines)s
</table>
</center>
<br><br><br>
"""    
    table_line_template = \
"""
<tr><td align=right>%(P_title)s</td>
    <td align=right><a id="%(P_link)s" href=%(P_link)s onClick="check_link('%(P_link)s','%(mod_P_link)s','%(P_link)s')"> %(cross)s </a> </td>
    <td align=right>  %(error)s</td>
    <td align=right>  %(events)s</td>
    <td align=right>  %(unweighted)s</td>
    <td align=right>  %(luminosity)s</td>
</tr>
"""

    def get_html(self,run, unit, me_dir = []):
        """write html output"""
        
        # store value for global cross-section
        P_grouping = {}

        tables_line = ''
        for oneresult in self:
            if oneresult.name.startswith('P'):
                title = '<a href=../../SubProcesses/%(P)s/diagrams.html>%(P)s</a>' \
                                                          % {'P':oneresult.name}
                P = oneresult.name.split('_',1)[0]
                if P in P_grouping:
                    P_grouping[P] += float(oneresult.xsec)
                else:
                    P_grouping[P] = float(oneresult.xsec)
            else:
                title = oneresult.name
            
            if not isinstance(oneresult, Combine_results):
                # this is for the (aMC@)NLO logs
                if os.path.exists(pjoin(me_dir, 'Events', run, 'alllogs_1.html')):
                    link = '../../Events/%(R)s/alllogs_1.html#/%(P)s/%(G)s' % \
                                        {'P': self.name,
                                         'G': oneresult.name,
                                         'R': run}
                    mod_link = link
                elif os.path.exists(pjoin(me_dir, 'Events', run, 'alllogs_0.html')):
                    link = '../../Events/%(R)s/alllogs_0.html#/%(P)s/%(G)s' % \
                                        {'P': self.name,
                                         'G': oneresult.name,
                                         'R': run}
                    mod_link = link
                else:
                    # this is for madevent runs
                    link = '../../SubProcesses/%(P)s/%(G)s/%(R)s_log.txt' % \
                                            {'P': self.name,
                                             'G': oneresult.name,
                                             'R': run}
                    mod_link = '../../SubProcesses/%(P)s/%(G)s/log.txt' % \
                                            {'P': self.name,
                                             'G': oneresult.name}
            else:
                link = '#%s' % oneresult.name
                mod_link = link
            
            dico = {'P_title': title,
                    'P_link': link,
                    'mod_P_link': mod_link,
                    'cross': '%.4g' % oneresult.xsec,
                    'error': '%.3g' % oneresult.xerru,
                    'events': oneresult.nevents,
                    'unweighted': oneresult.nunwgt,
                    'luminosity': '%.3g' % oneresult.luminosity
                   }
    
            tables_line += self.table_line_template % dico
        
        for P_name, cross in P_grouping.items():
            dico = {'P_title': '%s sum' % P_name,
                    'P_link': './results.html',
                    'mod_P_link':'',
                    'cross': cross,
                    'error': '',
                    'events': '',
                    'unweighted': '',
                    'luminosity': ''
                   }
            tables_line += self.table_line_template % dico

        if self.name.startswith('P'):
            title = '<dt><a  name=%(P)s href=../../SubProcesses/%(P)s/diagrams.html>%(P)s</a></dt><dd>' \
                                                          % {'P':self.name}
        else:
            title = ''
            
        dico = {'cross': self.xsec,
                'abscross': self.axsec,
                'error': self.xerru,
                'unit': unit,
                'result_type': 'Cross-Section',
                'table_lines': tables_line,
                'diagram_link': title
                }

        html_text = self.template_file % dico
        return html_text
    
    def write_results_dat(self, output_path):
        """write a correctly formatted results.dat"""

        def fstr(nb):
            data = '%E' % nb
            nb, power = data.split('E')
            nb = float(nb) /10
            power = int(power) + 1
            return '%.5fE%+03i' %(nb,power)

        line = '%s %s %s %i %i %i %i %s %s %s\n' % (fstr(self.axsec), fstr(self.xerru), 
                fstr(self.xerrc), self.nevents, self.nw, self.maxit, self.nunwgt,
                 fstr(self.luminosity), fstr(self.wgt), fstr(self.xsec))        
        fsock = open(output_path,'w') 
        fsock.writelines(line)
        for i in range(len(self.ysec_iter)):
            line = '%s %s %s %s %s %s\n' % (i+1, self.ysec_iter[i], self.yerr_iter[i], 
                      self.eff_iter[i], self.maxwgt_iter[i], self.yasec_iter[i]) 
            fsock.writelines(line)
        


results_header = """
<head>
    <title>Process results</title>
    <script type="text/javascript" src="../sortable.js"></script>
    <link rel=stylesheet href="../mgstyle.css" type="text/css">
</head>
<body>
<script type="text/javascript">
function UrlExists(url) {
  var http = new XMLHttpRequest();
  http.open('HEAD', url, false);
  try{
     http.send()
     }
  catch(err){
   return 1==2;
  }
  return http.status!=404;
}
function check_link(url,alt, id){
    var obj = document.getElementById(id);
    if ( ! UrlExists(url)){
        if ( ! UrlExists(alt)){
         obj.href = alt;
         return true;
        }
       obj.href = alt;
       return false;
    }
    obj.href = url;
    return 1==1;
}
</script>
""" 






def make_all_html_results(cmd, folder_names = []):
    """ folder_names has been added for the amcatnlo runs """
    run = cmd.results.current['run_name']
    if not os.path.exists(pjoin(cmd.me_dir, 'HTML', run)):
        os.mkdir(pjoin(cmd.me_dir, 'HTML', run))
    
    unit = cmd.results.unit
            
    all = Combine_results(run)
    P_text = ""
    
    for Pdir in open(pjoin(cmd.me_dir, 'SubProcesses','subproc.mg')):
        Pdir = Pdir.strip()
        P_comb = Combine_results(Pdir)
        
        P_path = pjoin(cmd.me_dir, 'SubProcesses', Pdir)
        G_dir = [G for G in os.listdir(P_path) if G.startswith('G') and 
                                                os.path.isdir(pjoin(P_path,G))]
        
        for line in open(pjoin(P_path, 'symfact.dat')):
            name, mfactor = line.split()
            if float(mfactor) < 0:
                continue
            if os.path.exists(pjoin(P_path, 'ajob.no_ps.log')):
                continue
                                  
            if not folder_names:
                name = 'G' + name
                P_comb.add_results(name, pjoin(P_path,name,'results.dat'), mfactor)
            else:
                for folder in folder_names:
                    if 'G' in folder:
                        dir = folder.replace('*', name)
                    else:
                        dir = folder.replace('*', '_G' + name)
                    P_comb.add_results(dir, pjoin(P_path,dir,'results.dat'), mfactor)

        P_comb.compute_values()
        P_text += P_comb.get_html(run, unit, cmd.me_dir)
        P_comb.write_results_dat(pjoin(P_path, '%s_results.dat' % run))
        all.append(P_comb)
    all.compute_values()
    all.write_results_dat(pjoin(cmd.me_dir,'SubProcesses', 'results.dat'))

    fsock = open(pjoin(cmd.me_dir, 'HTML', run, 'results.html'),'w')
    fsock.write(results_header)
    fsock.write('%s <dl>' % all.get_html(run, unit, cmd.me_dir))
    fsock.write('%s </dl></body>' % P_text)


          
    return all.xsec, all.xerru
