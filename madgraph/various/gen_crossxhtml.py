################################################################################
#
# Copyright (c) 2011 The MadGraph Development team and Contributors
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
""" Create gen_crossxhtml """


import os
import pickle
try:
    import internal.files as files
    import internal.save_load_object as save_load_object
except:
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object

pjoin = os.path.join
exists = os.path.exists

crossxhtml_template = """
<HTML> 
<HEAD> 
    <META HTTP-EQUIV="Refresh" CONTENT="10" > 
    <META HTTP-EQUIV="EXPIRES" CONTENT="20" > 
    <TITLE>Online Event Generation</TITLE>
</HEAD>
<BODY>
    <H2 align=center> Results in the %(model)s for %(process)s </H2> 
    <HR>
    %(status)s
    <br>
    <br>
    <H2 align="center"> Available Results </H2>
        <TABLE BORDER=2 align="center" >  
            <TR align="center">
                <TH>Links</TH> 
                <TH>Events</TH> 
                <TH NOWRAP> Tag </TH>
                <TH NOWRAP> Run </TH> 
                <TH>Collider</TH> 
                <TH> Cross section (pb) </TH> 
                <TH> Events  </TH> 
            </TR>     
            %(old_run)s
        </TABLE>
    <H3 align=center><A HREF="../index.html"> Main Page </A></H3>
</BODY> 
</HTML> 
"""

status_template = """
<H2 ALIGN=CENTER> Currently Running </H2>
<TABLE BORDER=2 ALIGN=CENTER>
    <TR ALIGN=CENTER>
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Run Name </TH>
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Cards </TH>   
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Results </TH> 
        <TH nowrap ROWSPAN=1 COLSPAN=4 font color="#0000FF"> Status/Jobs
        <TR> 
            <TH>   Queued </TH>
            <TH>  Running </TH>
            <TH> Done  </TH>
            <TH> Total </TH>
        </TR>
    </TR>
    <TR ALIGN=CENTER> 
        <TD nowrap> %(run_name)s </TD>
        <TD nowrap> <a href="../Cards/param_card.dat">param_card</a><BR>
                    <a href="../Cards/run_card.dat">run_card</a><BR>
                    %(plot_card)s
                    %(pythia_card)s
                    %(pgs_card)s
                    %(delphes_card)s
                    
        </TD>
        <TD nowrap> <A HREF="../SubProcesses/results.html">%(cross).4g <font face=symbol>&#177</font> %(error).4g (%(unit)s)</A> </TD> 
        %(status)s
 </TR></TABLE>
"""

class AllResults(dict):
    """Store the results for all the run of a given directory"""
    
    web = False 
    
    def __init__(self, model, process, path):
        
        dict.__init__(self)
        self.order = []
        self.process = ' <br> '.join(process)
        self.path = path
        self.model = model
    
    def def_current(self, name):
        """define the name of the current run"""
        assert name in self
        self.current = self[name]

    def def_web_mode(self, web):
        """define if we are in web mode or not """
        if web is True:
            try:
                web = os.environ['SERVER_NAME']
            except:
                web = 'my_computer'
        self['web'] = web
        self.web = web
        
    def add_run(self, name, run_card, current=True):
        """ Adding a run to this directory"""
        
        new = OneRunResults(name, run_card, self.process, self.path)
        self[name] = new
        self.def_current(name)
        if name in self.order:
            self.order.remove(name)
        self.order.append(name)
        if current:
            self.def_current(name)
        
        
    def update(self, status, level):
        """update the current run status"""
        self.current.update_status(level)
        self.status = status
        self.output()
    
    def save(self):
        """Save the results of this directory in a pickle file"""
        filename = pjoin(self.path, 'HTML', 'results.pkl')
        save_load_object.save_to_file(filename, self)

    def add_detail(self, name, value):
        """ add information to current run (cross/error/event)"""
        
        assert name in ['cross', 'error', 'nb_event']        
        self.current[name] = value
        
    def output(self):
        """ write the output file """
        
        # 1) Create the text for the status directory
        if isinstance(self.status, str):
            status = '<td colspan=4>%s</td>' %  self.status
        else:
            status ='<td> %s </td> <td> %s </td> <td> %s </td> <td> %s </td>' % \
                (tuple(self.status)+ (sum(self.status),))
        status_dict = {'status': status,
                        'cross': self.current['cross'],
                        'error': self.current['error'],
                        'unit': self.current['unit'],
                        'run_name': self.current['run_name']}
        if exists(pjoin(self.path, 'Cards', 'plot_card.dat')):
            status_dict['plot_card'] = """ <a href="../Cards/plot_card.dat">plot_card</a><BR>"""
        else:
            status_dict['plot_card'] = ""
        if exists(pjoin(self.path, 'Cards', 'pythia_card.dat')):
            status_dict['pythia_card'] = """ <a href="../Cards/pythia_card.dat">pythia_card</a><BR>"""
        else:
            status_dict['pythia_card'] = ""
        if exists(pjoin(self.path, 'Cards', 'pgs_card.dat')):
            status_dict['pgs_card'] = """ <a href="../Cards/pgs_card.dat">pgs_card</a><BR>"""
        else:
            status_dict['pgs_card'] = ""
        if exists(pjoin(self.path, 'Cards', 'delphes_card.dat')):
            status_dict['delphes_card'] = """ <a href="../Cards/delphes_card.dat">delphes_card</a><BR>"""
        else:
            status_dict['delphes_card'] = ""
        
        status = status_template % status_dict
        
        # 2) Create the text for the old run:
        old_run = ''
        for key in self.order:
            old_run += self[key].info_html(self.path, self.web)
        
        text_dict = {'process': self.process,
                     'model': self.model,
                     'status': status,
                     'old_run': old_run}
        
        text = crossxhtml_template % text_dict
        open(pjoin(self.path,'HTML','crossx.html'),'w').write(text)
        
        
class OneRunResults(dict):
    """ Store the results of a specific run """
    
    def __init__(self, run_name, run_card, process, path):
        """initialize the object"""
        
        # define at run_result
        self['run_name'] = run_name
        self['tag'] = run_card['run_tag']
        self.event_path = pjoin(path,'Events')
        self.me_dir = path
        
        # Set the collider information
        data = process.split('>',1)[0].split()
        if len(data) == 2:
            name1,name2 = data
            if run_card['lpp1'] == '-1':
                name1 += '~'
            elif run_card['lpp1'] == '2':
                name1 = ' a'
            if run_card['lpp2'] == '-1':
                name2 += '~'
            elif run_card['lpp2'] == '2':
                name2 = ' a'                
            self['collider'] = '''%s %s <br> %s x %s  GeV''' % \
                    (name1, name2, run_card['ebeam1'], run_card['ebeam2'])
            self['unit'] = 'pb'                       
        else:
            self['collider'] = 'decay'
            self['unit'] = 'GeV'
        
        # Default value
        self['nb_event'] = None
        self['cross'] = 0
        self['error'] = 0
        self.parton = [] 
        self.pythia = []
        self.pgs = []
        self.delphes = []
        
        # data 
        self.status = ''
        
    def update_status(self, level='all'):
        """update the status of the current run """

        exists = os.path.exists
        run = self['run_name']
        path = self.event_path
        # Check if the output of the last status exists
        
        if level in ['parton','all']:
        
            if 'lhe' not in self.parton and \
                        exists(pjoin(path,"%s_unweighted_events.lhe.gz" % run)):
                self.parton.append('lhe')
        
            if 'root' not in self.parton and \
                          exists(pjoin(path,"%s_unweighted_events.root" % run)):
                self.parton.append('root')
            
            if 'plot' not in self.parton and \
                                      exists(pjoin(path,"%s_plots.html" % run)):
                self.parton.append('plot')

        if level in ['pythia', 'all']:
            
            if 'plot' not in self.pythia and \
                               exists(pjoin(path,"%s_plots_pythia.html" % run)):
                self.pythia.append('plot')
            
            if 'lhe' not in self.pythia and \
                            exists(pjoin(path,"%s_pythia_events.lhe.gz" % run)):
                self.pythia.append('lhe')

            if 'hep' not in self.pythia and \
                            exists(pjoin(path,"%s_pythia_events.hep.gz" % run)):
                self.pythia.append('hep')
            
            if 'root' not in self.pythia and \
                              exists(pjoin(path,"%s_pythia_events.root" % run)):
                self.pythia.append('root')
                
            if 'lheroot' not in self.pythia and \
                          exists(pjoin(path,"%s_pythia_lhe_events.root" % run)):
                self.pythia.append('lheroot')            

        if level in ['pgs', 'all']:
            
            if 'plot' not in self.pgs and \
                         exists(pjoin(path,"%s_plots_pgs.html" % run)):
                self.pgs.append('plot')
            
            if 'lhco' not in self.pgs and \
                              exists(pjoin(path,"%s_pgs_events.lhco.gz" % run)):
                self.pgs.append('lhco')
                
            if 'root' not in self.pgs and \
                                 exists(pjoin(path,"%s_pgs_events.root" % run)):
                self.pgs.append('root') 
    
        if level in ['delphes', 'all']:
            
            if 'plot' not in self.delphes and \
                              exists(pjoin(path,"%s_plots_delphes.html" % run)):
                self.delphes.append('plot')
            
            if 'lhco' not in self.delphes and \
                 exists(pjoin(path,"%s_delphes_events.lhco.gz" % run)):
                self.delphes.append('lhco')
                
            if 'root' not in self.delphes and \
                             exists(pjoin(path,"%s_delphes_events.root" % run)):
                self.delphes.append('root')     
        
        

    def info_html(self, path, web=False):
        """ Return the line of the table containing run info for this run """

        if web:
            self['web'] = web
            self['me_dir'] = path
            
        out = "<tr>"
        # Links Events Tag Run Collider Cross Events
        
        #Links:
        out += '<td>'
        out += """<a href="../SubProcesses/%(run_name)s_results.html">results</a>"""
        out += """<br><a href="../Events/%(run_name)s_banner.txt">banner</a>"""
        out += '</td>'
        # Events
        out += '<td>'
        out += self.get_html_event_info(web)
        out += '</td>'
        # Tag
        out += '<td> %(tag)s </td>'
        # Run
        out += '<td> %(run_name)s </td>'
        # Collider
        out += '<td> %(collider)s </td>'
        # Cross
        out += '<td> %(cross).4g <font face=symbol>&#177</font> %(error).4g  </td>'
        # Events
        out += '<td> %(nb_event)s </td>' 

        return out % self
    
    def get_html_event_info(self, web=False):
        """return the events information"""
        
        # Events
        out = '<table border=1>'
        if self.parton:
            out += '<tr><td> Parton Events : </td><td>'
            if 'lhe' in self.parton:
                out += ' <a href="../Events/%(run_name)s_unweighted_events.lhe.gz">LHE</a>'
            if 'root' in self.parton:
                out += ' <a href="../Events/%(run_name)s_unweighted_events.root">rootfile</a>'
            if 'plot' in self.parton:
                out += ' <a href="../Events/%(run_name)s_plots.html">plots</a>'
            out += '<td></tr>'
        if self.pythia:
            out += '<tr><td> Pythia Events : </td><td>'

            if 'hep' in self.pythia:
                out += """ <a href="../Events/%(run_name)s_pythia_events.hep.gz">STDHEP</a>"""
            if 'lhe' in self.pythia:
                out += """ <a href="../Events/%(run_name)s_pythia_events.lhe.gz">LHE</a>"""
            if 'root' in self.pythia:
                out += """ <a href="../Events/%(run_name)s_pythia_events.root">rootfile (LHE)</a>"""
            if 'lheroot' in self.pythia:
                out += """ <a href="../Events/%(run_name)s_pythia_lhe_events.root">rootfile (LHE)</a>"""
            if 'plot' in self.pythia:
                out += ' <a href="../Events/%(run_name)s_plots_pythia.html">plots</a>'
            out += '</td></tr>'
        elif web and self['nb_event']:
            out += """<tr><td> Pythia Events : </td><td><center>
                       <FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
                       <INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s"> 
                       <INPUT TYPE=HIDDEN NAME=whattodo VALUE="pythia"> 
                       <INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s"> 
                       <INPUT TYPE=SUBMIT VALUE="Run Pythia"></FORM><center></td>
                       </table>"""
            return out 

        if self.pgs:
            out += '<tr><td>Reco. Objects. (PGS) : </td><td>'
            if 'lhco' in self.pgs:
                out += """ <a href="../Events/%(run_name)s_pgs_events.lhco.gz">LHCO</a>"""
            if 'root' in self.pgs:
                out += """ <a href="../Events/%(run_name)_pgs_events.root">rootfile</a>"""    
            if 'plot' in self.pgs:
                out += """ <a href="../Events/%(run_name)s_plots_pgs.html">plots</a>"""
            out += '</td></tr>'
        if self.delphes:
            out += '<tr><td>Reco. Objects. (Delphes) : </td><td>'
            if 'lhco' in self.delphes:
                out += """ <a href="../Events/%(run_name)s_delphes_events.lhco.gz">LHCO</a>"""
            if 'root' in self.delphes:
                out += """ <a href="../Events/%(run_name)_delphes_events.root">rootfile</a>"""    
            if 'plot' in self.delphes:
                out += """ <a href="../Events/%(run_name)s_plots_delphes.html">plots</a>"""            
            out += '</td></tr>'
        
        if not (self.pgs or self.delphes) and web and self['nb_event']:
            out += """<tr><td> Reco. Objects: </td><td><center>
                       <FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
                       <INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s"> 
                       <INPUT TYPE=HIDDEN NAME=whattodo VALUE="pgs"> 
                       <INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s"> 
                       <INPUT TYPE=SUBMIT VALUE="Run Pythia"></FORM></center></td>
                       </table>"""
            return out             
        
        
        out += '</table>'
        return out
        
    
    
    
    
    




