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
import xml.dom.minidom as minidom

logger = logging.getLogger('madevent.stdout') # -> stdout

pjoin = os.path.join
try:
    import madgraph
except ImportError:
    import internal.cluster as cluster
    import internal.misc as misc
    from internal import MadGraph5Error
else:
    import madgraph.various.cluster as cluster
    import madgraph.various.misc as misc
    from madgraph import MadGraph5Error

class RunStatistics(dict):
    """ A class to store statistics about a MadEvent run. """
    
    def __init__(self, *args, **opts):
        """ Initialize the run dictionary. For now, the same as a regular
        dictionary, except that we specify some default statistics. """
        
        madloop_statistics = {
          'unknown_stability'  : 0,
          'stable_points'      : 0,
          'unstable_points'    : 0,
          'exceptional_points' : 0,
          'DP_usage'           : 0,
          'QP_usage'           : 0,
          'DP_init_usage'      : 0,
          'QP_init_usage'      : 0,
          'CutTools_DP_usage'  : 0,
          'CutTools_QP_usage'  : 0,          
          'PJFry_usage'        : 0,
          'Golem_usage'        : 0,
          'IREGI_usage'        : 0,
          'max_precision'      : 1.0e99,
          'min_precision'      : 0.0,
          'averaged_timing'    : 0.0,
          'n_madloop_calls'    : 0,
          'cumulative_timing'  : 0.0,
          'skipped_subchannel' : 0 # number of times that a computation have been 
                                    # discarded due to abnormal weight.
          }
        
        for key, value in madloop_statistics.items():
            self[key] = value

        super(dict,self).__init__(*args, **opts)
    
    def aggregate_statistics(self, new_stats):
        """ Update the current statitistics with the new_stats specified."""
        
        if isinstance(new_stats,RunStatistics):
            new_stats = [new_stats, ]
        elif isinstance(new_stats,list):
            if any(not isinstance(_,RunStatistics) for _ in new_stats):
                raise MadGraph5Error, "The 'new_stats' argument of the function "+\
                        "'updtate_statistics' must be a (possibly list of) "+\
                                                       "RunStatistics instance."
 
        keys = set([])
        for stat in [self,]+new_stats:
            keys |= set(stat.keys())

        new_stats = new_stats+[self,]
        for key in keys:
            # Define special rules
            if key=='max_precision':
                # The minimal precision corresponds to the maximal value for PREC
                self[key] = min( _[key] for _ in new_stats if key in _)
            elif key=='min_precision':
                # The maximal precision corresponds to the minimal value for PREC
                self[key] = max( _[key] for _ in new_stats if key in _)
            elif key=='averaged_timing':
                n_madloop_calls = sum(_['n_madloop_calls'] for _ in new_stats if
                                                         'n_madloop_calls' in _)
                if n_madloop_calls > 0 :
                    self[key] = sum(_[key]*_['n_madloop_calls'] for _ in 
                      new_stats if (key in _ and 'n_madloop_calls' in _) )/n_madloop_calls
            else:
                # Now assume all other quantities are cumulative
                self[key] = sum(_[key] for _ in new_stats if key in _)
    
    def load_statistics(self, xml_node):
        """ Load the statistics from an xml node. """
        
        def getData(Node):
            return Node.childNodes[0].data
        
        u_return_code = xml_node.getElementsByTagName('u_return_code')
        u_codes = [int(_) for _ in getData(u_return_code[0]).split(',')]
        self['CutTools_DP_usage'] = u_codes[1]
        self['PJFry_usage']       = u_codes[2]
        self['IREGI_usage']       = u_codes[3]
        self['Golem_usage']       = u_codes[4]
        self['CutTools_QP_usage'] = u_codes[9]
        t_return_code = xml_node.getElementsByTagName('t_return_code')
        t_codes = [int(_) for _ in getData(t_return_code[0]).split(',')]
        self['DP_usage']          = t_codes[1]
        self['QP_usage']          = t_codes[2]
        self['DP_init_usage']     = t_codes[3]
        self['DP_init_usage']     = t_codes[4]
        h_return_code = xml_node.getElementsByTagName('h_return_code')
        h_codes = [int(_) for _ in getData(h_return_code[0]).split(',')]
        self['unknown_stability']  = h_codes[1]
        self['stable_points']      = h_codes[2]
        self['unstable_points']    = h_codes[3]
        self['exceptional_points'] = h_codes[4]
        average_time = xml_node.getElementsByTagName('average_time')
        avg_time = float(getData(average_time[0]))
        self['averaged_timing']    = avg_time 
        cumulated_time = xml_node.getElementsByTagName('cumulated_time')
        cumul_time = float(getData(cumulated_time[0]))
        self['cumulative_timing']  = cumul_time 
        max_prec = xml_node.getElementsByTagName('max_prec')
        max_prec = float(getData(max_prec[0]))
        # The minimal precision corresponds to the maximal value for PREC
        self['min_precision']      = max_prec  
        min_prec = xml_node.getElementsByTagName('min_prec')
        min_prec = float(getData(min_prec[0]))
        # The maximal precision corresponds to the minimal value for PREC
        self['max_precision']      = min_prec              
        n_evals = xml_node.getElementsByTagName('n_evals')
        n_evals = int(getData(n_evals[0]))
        self['n_madloop_calls']    = n_evals
    
    def nice_output(self,G, no_warning=False):
        """Returns a one-line string summarizing the run statistics 
        gathered for the channel G."""
        
        # Do not return anythign for now if there is no madloop calls. This can
        # change of course if more statistics are gathered, unrelated to MadLoop.
        if self['n_madloop_calls']==0:
            return ''

        stability = [
          ('tot#',self['n_madloop_calls']),
          ('unkwn#',self['unknown_stability']),
          ('UPS%',float(self['unstable_points'])/self['n_madloop_calls']),
          ('EPS#',self['exceptional_points'])]

        stability = [_ for _ in stability if _[1] > 0 or _[0] in ['UPS%','EPS#']]
        stability = [(_[0],'%i'%_[1]) if isinstance(_[1], int) else
                     (_[0],'%.3g'%(100.0*_[1])) for _ in stability]
        
        tools_used = [
          ('CT_DP',float(self['CutTools_DP_usage'])/self['n_madloop_calls']),
          ('CT_QP',float(self['CutTools_QP_usage'])/self['n_madloop_calls']),
          ('PJFry',float(self['PJFry_usage'])/self['n_madloop_calls']),
          ('Golem',float(self['Golem_usage'])/self['n_madloop_calls']),
          ('IREGI',float(self['IREGI_usage'])/self['n_madloop_calls'])]

        tools_used = [(_[0],'%.3g'%(100.0*_[1])) for _ in tools_used if _[1] > 0.0 ]

        to_print = [('%s statistics:'%(G if isinstance(G,str) else
                                                    str(os.path.join(list(G))))\
          +(' %s,'%misc.format_time(int(self['cumulative_timing'])) if
                                     int(self['cumulative_timing']) > 0 else '')
          +((' Avg. ML timing = %i ms'%int(1.0e3*self['averaged_timing'])) if
            self['averaged_timing'] > 0.001 else
            (' Avg. ML timing = %i mus'%int(1.0e6*self['averaged_timing']))) \
          +', Min precision = %.2e'%self['min_precision'])
          ,'   -> Stability %s'%dict(stability)
          ,'   -> Red. tools usage in %% %s'%dict(tools_used)
#         I like the display above better after all
#          ,'Stability %s'%(str([_[0] for _ in stability]),
#                             str([_[1] for _ in stability]))
#          ,'Red. tools usage in %% %s'%(str([_[0] for _ in tools_used]),
#                                      str([_[1] for _ in tools_used]))
        ]

        if self['skipped_subchannel'] > 0 and not no_warning:
            to_print.append("WARNING: Some event with large weight have been "+\
               "discarded. This happened %s times." % self['skipped_subchannel'])

        return ('\n'.join(to_print)).replace("'"," ")
    
    def has_warning(self):
        """return if any stat needs to be reported as a warning
           When this is True, the print_warning doit retourner un warning
        """
    
        if self['n_madloop_calls'] > 0:
            fraction = self['exceptional_points']/float(self['n_madloop_calls'])
        else:
            fraction = 0.0
            
        if self['skipped_subchannel'] > 0:
            return True
        elif fraction > 1.0e-4:
            return True
        else:
            return False

    def get_warning_text(self):
        """get a string with all the identified warning"""
        
        to_print = []
        if self['skipped_subchannel'] > 0:
            to_print.append("Some event with large weight have been discarded."+\
                         " This happens %s times." % self['skipped_subchannel'])
        if self['n_madloop_calls'] > 0:
            fraction = self['exceptional_points']/float(self['n_madloop_calls'])
            if fraction > 1.0e-4:
                to_print.append("Some PS with numerical instability have been set "+\
                   "to a zero matrix-element (%.3g%%)" % (100.0*fraction))
        
        return ('\n'.join(to_print)).replace("'"," ") 

class OneResult(object):
    
    def __init__(self, name):
        """Initialize all data """
        
        self.run_statistics = RunStatistics()
        self.name = name
        self.parent_name = ''
        self.axsec = 0  # Absolute cross section = Sum(abs(wgt))
        self.xsec = 0 # Real cross section = Sum(wgt)
        self.xerru = 0  # uncorrelated error
        self.xerrc = 0  # correlated error
        self.nevents = 0
        self.nw = 0     # number of events after the primary unweighting
        self.maxit = 0  # 
        self.nunwgt = 0  # number of unweighted events
        self.luminosity = 0
        self.mfactor = 1 # number of times that this channel occur (due to symmetry)
        self.ysec_iter = []
        self.yerr_iter = []
        self.yasec_iter = []
        self.eff_iter = []
        self.maxwgt_iter = []
        self.maxwgt = 0 # weight used for the secondary unweighting.
        self.th_maxwgt= 0 # weight that should have been use for secondary unweighting
                          # this can happen if we force maxweight
        self.th_nunwgt = 0 # associated number of event with th_maxwgt 
                           #(this is theoretical do not correspond to a number of written event)

        return
    
    #@cluster.multiple_try(nb_try=5,sleep=20)
    def read_results(self, filepath):
        """read results.dat and fullfill information"""
        
        if isinstance(filepath, str):
            finput = open(filepath)
        elif isinstance(filepath, file):
            finput = filepath
        else:
            raise Exception, "filepath should be a path or a file descriptor"
        
        i=0
        found_xsec_line = False
        for line in finput:            
            # Exit as soon as we hit the xml part. Not elegant, but the part
            # below should eventually be xml anyway.
            if '<' in line:
                break
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
                         self.maxit, self.nunwgt, self.luminosity, self.wgt, \
                         self.xsec = data[:10]
                if len(data) > 10:
                    self.maxwgt = data[10]
                if len(data) >12:
                    self.th_maxwgt, self.th_nunwgt = data[11:13]
                if self.mfactor > 1:
                    self.luminosity /= self.mfactor
                continue
            try:
                l, sec, err, eff, maxwgt, asec = line.split()
                found_xsec_line = True
            except:
                break
            self.ysec_iter.append(secure_float(sec))
            self.yerr_iter.append(secure_float(err))
            self.yasec_iter.append(secure_float(asec))
            self.eff_iter.append(secure_float(eff))
            self.maxwgt_iter.append(secure_float(maxwgt))

        finput.seek(0)
        xml = []
        for line in finput:
            if re.match('^.*<.*>',line):
                xml.append(line)
                break
        for line in finput:
            xml.append(line)

        if xml:
            self.parse_xml_results('\n'.join(xml))        
        
        # this is for amcatnlo: the number of events has to be read from another file
        if self.nevents == 0 and self.nunwgt == 0 and isinstance(filepath, str) and \
                os.path.exists(pjoin(os.path.split(filepath)[0], 'nevts')): 
            nevts = int(open(pjoin(os.path.split(filepath)[0], 'nevts')).read())
            self.nevents = nevts
            self.nunwgt = nevts
        
    def parse_xml_results(self, xml):
        """ Parse the xml part of the results.dat file."""

        dom = minidom.parseString(xml)
                    
        statistics_node = dom.getElementsByTagName("run_statistics")
        
        if statistics_node:
            try:
                self.run_statistics.load_statistics(statistics_node[0])
            except ValueError, IndexError:
                logger.warning('Fail to read run statistics from results.dat')

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
    
    def get(self, name):
        
        if name in ['xsec', 'xerru','xerrc']:
            return getattr(self, name) * self.mfactor
        elif name in ['luminosity']:
            #misc.sprint("use unsafe luminosity definition")
            #raise Exception
            return getattr(self, name) #/ self.mfactor
        elif (name == 'eff'):
            return self.xerr*math.sqrt(self.nevents/(self.xsec+1e-99))
        elif name == 'xerr':
            return math.sqrt(self.xerru**2+self.xerrc**2)
        elif name == 'name':
            return pjoin(self.parent_name, self.name)
        else:
            return getattr(self, name)

class Combine_results(list, OneResult):
    
    def __init__(self, name):
        
        list.__init__(self)
        OneResult.__init__(self, name)
    
    def add_results(self, name, filepath, mfactor=1):
        """read the data in the file"""
        try:
            oneresult = OneResult(name)
            oneresult.set_mfactor(mfactor)
            oneresult.read_results(filepath)
            oneresult.parent_name = self.name
            self.append(oneresult)
            return oneresult
        except Exception:
            logger.critical("Error when reading %s" % filepath)
            raise
        
    
    def compute_values(self, update_statistics=False):
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
        if update_statistics:
            self.run_statistics.aggregate_statistics([_.run_statistics for _ in self])

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
        self.th_maxwgt = 0.0
        self.th_nunwgt = 0 
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
                    'events': oneresult.nevents/1000.0,
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
            if data == 'NAN':
                nb, power = 0,0
            else:
                nb, power = data.split('E')
                nb = float(nb) /10
            power = int(power) + 1
            return '%.5fE%+03i' %(nb,power)

        line = '%s %s %s %i %i %i %i %s %s %s %s %s %i\n' % (fstr(self.axsec), fstr(self.xerru), 
                fstr(self.xerrc), self.nevents, self.nw, self.maxit, self.nunwgt,
                 fstr(self.luminosity), fstr(self.wgt), fstr(self.xsec), fstr(self.maxwgt),
                 fstr(self.th_maxwgt), self.th_nunwgt)        
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

def collect_result(cmd, folder_names):
    """ """ 

    run = cmd.results.current['run_name']
    all = Combine_results(run)
    
    for Pdir in open(pjoin(cmd.me_dir, 'SubProcesses','subproc.mg')):
        Pdir = Pdir.strip()
        P_comb = Combine_results(Pdir)
        
        P_path = pjoin(cmd.me_dir, 'SubProcesses', Pdir)
        G_dir = [G for G in os.listdir(P_path) if G.startswith('G') and 
                                                os.path.isdir(pjoin(P_path,G))]

        try:
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
        except IOError:
            continue
        P_comb.compute_values()
        all.append(P_comb)
    all.compute_values()
    return all


def make_all_html_results(cmd, folder_names = []):
    """ folder_names has been added for the amcatnlo runs """
    run = cmd.results.current['run_name']
    if not os.path.exists(pjoin(cmd.me_dir, 'HTML', run)):
        os.mkdir(pjoin(cmd.me_dir, 'HTML', run))
    
    unit = cmd.results.unit
    P_text = ""      
    Presults = collect_result(cmd, folder_names=folder_names)
            
    
    for P_comb in Presults:
        P_text += P_comb.get_html(run, unit, cmd.me_dir) 
        P_comb.compute_values()
        P_comb.write_results_dat(pjoin(cmd.me_dir, 'SubProcesses', P_comb.name,
                                        '%s_results.dat' % run))

    
    Presults.write_results_dat(pjoin(cmd.me_dir,'SubProcesses', 'results.dat'))   
    
    fsock = open(pjoin(cmd.me_dir, 'HTML', run, 'results.html'),'w')
    fsock.write(results_header)
    fsock.write('%s <dl>' % Presults.get_html(run, unit, cmd.me_dir))
    fsock.write('%s </dl></body>' % P_text)         
    
    return Presults.xsec, Presults.xerru   
            

