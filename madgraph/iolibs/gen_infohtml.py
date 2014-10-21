#! /usr/bin/env python
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

import string
import os
import re
import sys

import madgraph.various.misc as misc

template_text= string.Template("""
<HTML> 
<HEAD> 
<TITLE>Detail on the Generation</TITLE> 
<META $meta ></HEAD> 

<style type="text/css">

table.processes { border-collapse: collapse;
                  border: solid}

.processes td {
padding: 2 5 2 5;
border: solid thin;
}

th{
border-top: solid;
border-bottom: solid;
}

.first td{
border-top: solid;
}




</style>

<BODY> 
    <P> <H2 ALIGN=CENTER> SubProcesses and Feynman diagrams </H2>
    
    <TABLE BORDER=2 ALIGN=CENTER class=processes> 
        <TR>
           <TH>Directory</TH> 
           <TH NOWRAP># Diagrams </TH>
           <TH NOWRAP># Subprocesses </TH>
           <TH>FEYNMAN DIAGRAMS</TH> 
           <TH> SUBPROCESS </TH>
        </TR> 
        $info_lines
    </TABLE><BR> 
    <CENTER> $nb_diag diagrams ($nb_gen_diag independent).</CENTER>
    <br><br><br>
    <TABLE ALIGN=CENTER>
    $log
    <TR> 
        <TD ALIGN=CENTER> <A HREF="../Cards/proc_card_mg5.dat">proc_card_mg5.dat</A> </TD>
        <TD> Input file used for code generation.
    $model_info
    </TABLE><br>
    <center>
    <H3>Back to <A HREF="../index.html">Process main page</A></H3>
    </center>
 </BODY> 

</HTML>""")


template_text_nlo= string.Template("""
<HTML> 
<HEAD> 
<TITLE>Detail on the Generation</TITLE> 
<META $meta ></HEAD> 

<style type="text/css">

table.processes { border-collapse: collapse;
                  border: solid}

.processes td {
padding: 2 5 2 5;
border: solid thin;
}

th{
border-top: solid;
border-bottom: solid;
}

.first td{
border-top: solid;
}




</style>

<BODY> 
    <P> <H2 ALIGN=CENTER> SubProcesses and Feynman diagrams </H2>
    
    <TABLE BORDER=2 ALIGN=CENTER class=processes> 
        <TR>
           <TH>Directory</TH> 
           <TH NOWRAP>Type</TH>
           <TH NOWRAP># Diagrams </TH>
           <TH NOWRAP># Subprocesses </TH>
           <TH>FEYNMAN DIAGRAMS</TH> 
           <TH> SUBPROCESS </TH>
        </TR> 
        $info_lines
    </TABLE><BR> 
    <CENTER> $nb_diag diagrams ($nb_gen_diag independent).</CENTER>
    <br><br><br>
    <TABLE ALIGN=CENTER>
    $log
    <TR> 
        <TD ALIGN=CENTER> <A HREF="../Cards/proc_card_mg5.dat">proc_card_mg5.dat</A> </TD>
        <TD> Input file used for code generation.
    $model_info
    </TABLE><br>
    <center>
    <H3>Back to <A HREF="../index.html">Process main page</A></H3>
    </center>
 </BODY> 

</HTML>""")


class make_info_html:

    def __init__(self, cur_dir='./'):

        self.dir = cur_dir
        
        
        self.rep_rule = {'nb_diag': 0, 'nb_gen_diag': 0}
        
        self.define_meta()
        self.rep_rule['info_lines'] = self.define_info_tables()
        self.rep_rule['model_info']= self.give_model_info()
        self.rep_rule['log'] = self.check_log() 
        self.write()
        
        
    def give_model_info(self):
        """find path for the model"""
        
        path = os.path.join(self.dir, 'Source','MODEL','particles.dat')
        if os.path.exists(path):
            return """<TR> 
        <TD ALIGN=CENTER> <A HREF="../Source/MODEL/particles.dat">particles</A></TD> 
        <TD> Particles file used for code generation.</TD>
    </TR>
    <TR> 
        <TD ALIGN=CENTER> <A HREF="../Source/MODEL/interactions.dat">interactions</A></TD> 
        <TD> Interactions file used for code generation.</TD>
    </TR>"""
        else:
            return ''
        
        
    def define_meta(self):
        """add the meta in the replacement rule"""
        
        if os.path.exists(os.path.join(self.dir,'SubProcesses','done')):
            self.rep_rule['meta'] = 'HTTP-EQUIV=\"REFRESH\" CONTENT=\"30\"'
        else:
            self.rep_rule['meta'] = "HTTP-EQUIV=\"EXPIRES\" CONTENT=\"20\""
        

    def define_info_tables(self):
        """define the information table"""
        
        line_template = string.Template("""
        <TR class=$class> $first 
<TD> $diag </TD> 
<TD> $subproc </TD> 
<TD> <A HREF="../SubProcesses/$processdir/diagrams.html#$id" >html</A> $postscript
</TD><TD class=$class>
<SPAN style="white-space: nowrap;"> $subprocesslist</SPAN>
</TD></TR>""")
        
        #output text
        text = ''
        # list of valid P directory
        subproc = [content for content in os.listdir(os.path.join(self.dir,'SubProcesses'))
                                if content.startswith('P') and 
                                os.path.isdir(os.path.join(self.dir,'SubProcesses',content))
                                and os.path.exists(os.path.join(self.dir,'SubProcesses',content,'auto_dsig.f'))]
        
        for proc in subproc:
            
            idnames = self.get_subprocesses_info(proc)
               
            for id in range(1,len(idnames)+1):

                if id == 1:
                    
                    line_dict = {'processdir': proc,
                                 'class': 'first'}
                    line_dict['first']= '<TD class=$class rowspan=%s> %s </TD>' % (len(idnames), proc)
                else:
                    line_dict = {'processdir': proc,
                                 'class': 'second'}
                    line_dict['first'] = ''
                try:
                    names = idnames[id]
                except Exception:
                    names = idnames['']
                    id = ''
                line_dict['id'] = str(id)     
                line_dict['diag'] = self.get_diagram_nb(proc, id)
                line_dict['subproc'] = sum([len(data) for data in names])
                self.rep_rule['nb_diag'] += line_dict['diag'] * line_dict['subproc']
                self.rep_rule['nb_gen_diag'] += line_dict['diag']
                line_dict['subprocesslist'] = ', <br>'.join([' </SPAN> , <SPAN style="white-space: nowrap;"> '.join(info) for info in names])
                line_dict['postscript'] = self.check_postcript(proc, id)
                
                text += line_template.substitute(line_dict)
        return text
    
    def get_diagram_nb(self, proc, id):
        
        path = os.path.join(self.dir, 'SubProcesses', proc, 'matrix%s.f' % id)
        nb_diag = 0
                
        pat = re.compile(r'''Amplitude\(s\) for diagram number (\d+)''' )
       
        text = open(path).read()
        for match in re.finditer(pat, text):
            pass
        nb_diag += int(match.groups()[0])
        
        return nb_diag
            
            
    def get_subprocesses_info(self, proc):
        """ return the list of processes with their name"""    
        
        #no python information available (should not happen anymore)
        path = os.path.join(self.dir, 'SubProcesses', proc)        
        nb_sub = 0
        names = {}
        old_main = ''
        
        if not os.path.exists(os.path.join(path,'processes.dat')):
            return self.get_subprocess_info_v4(proc)
        
        for line in open(os.path.join(path,'processes.dat')):
            main = line[:8].strip()
            if main == 'mirror':
                main = old_main
                if line[8:].strip() == 'none':
                    continue 
            else:
                main = int(main)
                old_main = main

            sub_proccess = line[8:]
            nb_sub += sub_proccess.count(',') + 1
            if main in names:
                names[main] += [sub_proccess.split(',')]
            else: 
                names[main]= [sub_proccess.split(',')]
    

        return names

    def get_subprocess_info_v4(self, proc):
        """ return the list of processes with their name in case without grouping """
        
        nb_sub = 0
        names = {'':[[]]}
        path = os.path.join(self.dir, 'SubProcesses', proc,'auto_dsig.f')
        found = 0
        for line in open(path):
            if line.startswith('C     Process:'):
                found += 1
                names[''][0].append(line[15:])
            elif found >1:
                break    
        return names    
    
    def check_postcript(self, proc, id):
        """ check if matrix.ps is defined """
        path = os.path.join(self.dir, 'SubProcesses', proc,'matrix%s.f' % id) 
        if os.path.exists(path):
            return "<A HREF=\"../SubProcesses/%s/matrix%s.ps\" >postscript </A>" % \
                    (proc, id)
        else:
            return ''

    def check_log(self):
        path = os.path.join(self.dir, 'proc_log.txt') 
        if os.path.exists(path):
            return """<TR> 
        <TD ALIGN=CENTER> <A HREF="../proc_log.txt">proc_log.txt</A> </TD>
        <TD> Log file from MadGraph code generation. </TD>
        </TR>"""
        else:
            return ''
        
    def write(self):
        """write the info.html file"""
        
        fsock = open(os.path.join(self.dir,'HTML','info.html'),'w')
        text = template_text.substitute(self.rep_rule)
        fsock.write(text)

    

class make_info_html_nlo(make_info_html):


    def define_info_tables(self):
        """define the information table"""
        
        # we do not export the diagrams.html so far
        line_template = string.Template("""
        <TR class=$class> $first 
<TD> $type </TD>
<TD> $diag </TD> 
<TD> $subproc </TD> 
<TD>$postscript </TD>
<TD class=$class>
<SPAN style="white-space: nowrap;"> $subprocesslist</SPAN>
</TD></TR>""")
        

        #output text
        text = ''
        # list of valid P directory
        subproc = [content for content in os.listdir(os.path.join(self.dir,'SubProcesses'))
                                if content.startswith('P') and 
                                os.path.isdir(os.path.join(self.dir,'SubProcesses',content))
                                and os.path.islink(os.path.join(self.dir,'SubProcesses',content,'fks_singular.f'))]
        
        for proc in subproc:
            files_dict = {'born': ['born.f'], 
                          'virt': [os.path.join('V' + proc[1:], 'loop_matrix.f')],
                          'real': [file for file in os.listdir(os.path.join(self.dir,'SubProcesses', proc)) if 
                                   file.startswith('matrix_') and file.endswith('.f')]}
            
            for type in ['born', 'virt', 'real']:
                for file in files_dict[type]:
                    idnames = self.get_subprocesses_info_from_file(proc, file)
               
                    for id in range(1,len(idnames)+1):

                        if type == 'born':
                            line_dict = {'processdir': proc,
                                         'class': 'first'}
                            line_dict['first']= '<TD class=$class rowspan=%s> %s </TD>' % (len(idnames), proc)
                        else:
                            line_dict = {'processdir': 'proc',
                                         'class': 'second'}
                            line_dict['first'] = '<TD class=$class rowspan=%s>  </TD>' % (len(idnames))
                        try:
                            names = idnames[id]
                        except Exception:
                            names = idnames['']
                            id = ''
                        line_dict['type'] = type
                        line_dict['id'] = str(id)     
                        line_dict['diag'] = self.get_diagram_nb_from_file(proc, file.replace('.f', '.ps'))
                        line_dict['subproc'] = sum([len(data) for data in names])
                        self.rep_rule['nb_diag'] += line_dict['diag'] * line_dict['subproc']
                        self.rep_rule['nb_gen_diag'] += line_dict['diag']
                        line_dict['subprocesslist'] = ', <br>'.join([' </SPAN> , <SPAN style="white-space: nowrap;"> '.join(info) for info in names])
                        line_dict['postscript'] = self.check_postcript_from_file(proc, file)
                       
                        text += line_template.substitute(line_dict)
        return text


    def get_subprocesses_info_from_file(self, proc, filename):
        """ return the list of processes with their name in case without grouping
         type can be 0 for born, i > 0 for ith real and -1 for virtual"""

        nb_sub = 0
        names = {'':[[]]}
        path = os.path.join(self.dir, 'SubProcesses', proc, filename)
        if not os.path.exists(path):
            return []
        found = 0
        start= 0
        for line in open(path):
            if line.startswith('C     Process:'):
                found += 1
                names[''][0].append(line[15:-1])
                start =1
            elif found >0 and 'IMPLICIT NONE' in line:
                break    
            elif start:
                names[''][0][-1] += line[2:-1].strip()
        return names    


    def get_diagram_nb_from_file(self, proc, filename):
        
        path = os.path.join(self.dir, 'SubProcesses', proc, filename)
        nb_diag = 0
                
        pat = re.compile(r'''diagram (\d+)''' )
       
        text = open(path).read()
        for match in re.finditer(pat, text):
            pass
        try:
            nb_diag += int(match.groups()[0])
        except Exception:
            pass
        
        return nb_diag


    def check_postcript_from_file(self, proc, filename):
        """ check if matrix.ps is defined """
        psname = filename[:-1] + 'ps'
        path = os.path.join(self.dir, 'SubProcesses', proc, psname) 
        if os.path.exists(path):
            return "<A HREF=\"../SubProcesses/%s/%s\" >postscript </A>" % \
                    (proc, psname)
        else:
            return ''
            

    def write(self):
        """write the info.html file"""
        
        fsock = open(os.path.join(self.dir,'HTML','info.html'),'w')
        text = template_text_nlo.substitute(self.rep_rule)
        fsock.write(text)
