################################################################################
#
# Copyright (c) 2016 The MadGraph5_aMC@NLO Development team and Contributors
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

if __name__ == "__main__":
    import sys
    import os
    root = os.path.dirname(__file__)
    if os.path.basename(root) == 'internal':
        sys.path.append(os.path.dirname(root))
    else:
        sys.path.append(os.path.dirname(os.path.dirname(root)))
        
import lhe_parser
import banner
import banner as banner_mod
import itertools
import misc
import math
import os 
import re
import sys
import time
import StringIO

pjoin = os.path.join
root = os.path.dirname(__file__)

class SystematicsError(Exception):
    pass

class Systematics(object):
    
    def __init__(self, input_file, output_file,
                 start_event=0, stop_event=sys.maxint, write_banner=False,
                 mur=[0.5,1,2],
                 muf=[0.5,1,2],
                 alps=[1],
                 pdf='errorset', #[(id, subset)]
                 dyn=[-1,1,2,3,4],
                 together=[('mur', 'muf', 'dyn')],
                 remove_wgts=[],
                 keep_wgts=[],
                 start_id=None,
                 lhapdf_config=misc.which('lhapdf-config'),
                 log=lambda x: sys.stdout.write(str(x)+'\n'),
                 only_beam=False,
                 ion_scaling=True,
                 ):

        # INPUT/OUTPUT FILE
        if isinstance(input_file, str):
            self.input = lhe_parser.EventFile(input_file)
        else:
            self.input = input_file
        self.output_path = output_file
        if output_file != None:
            if isinstance(output_file, str):
                if output_file == input_file:
                    directory,name = os.path.split(output_file)
                    new_name = pjoin(directory, '.tmp_'+name)
                    self.output =  lhe_parser.EventFile(new_name, 'w')
                else:
                    self.output =  lhe_parser.EventFile(output_file, 'w')
            else:
                self.output = output_file
        self.log = log
        
        #get some information from the run_card.
        self.banner = banner_mod.Banner(self.input.banner)  
        self.force_write_banner = bool(write_banner)
        self.orig_dyn = self.banner.get('run_card', 'dynamical_scale_choice')
        self.orig_pdf = self.banner.run_card.get_lhapdf_id()
        matching_mode = self.banner.get('run_card', 'ickkw')

        #check for beam
        beam1, beam2 = self.banner.get_pdg_beam()
        if abs(beam1) != 2212 and abs(beam2) != 2212:
            self.b1 = 0
            self.b2 = 0 
            pdf = 'central'
            #raise SystematicsError, 'can only reweight proton beam'
        elif abs(beam1) != 2212:
            self.b1 = 0
            self.b2 = beam2//2212
        elif abs(beam2) != 2212:
            self.b1 = beam1//2212
            self.b2 = 0
        else:             
            self.b1 = beam1//2212
            self.b2 = beam2//2212
    
        self.orig_ion_pdf = False
        self.ion_scaling = ion_scaling
        self.only_beam = only_beam 
        if isinstance(self.banner.run_card, banner_mod.RunCardLO):
            self.is_lo = True
            if not self.banner.run_card['use_syst']:
                raise SystematicsError, 'The events have not been generated with use_syst=True. Cannot evaluate systematics error on these events.'
            
            if self.banner.run_card['nb_neutron1'] != 0 or \
               self.banner.run_card['nb_neutron2'] != 0 or \
               self.banner.run_card['nb_proton1'] != 1 or \
               self.banner.run_card['nb_proton2'] != 1:
                self.orig_ion_pdf = True
        else:
            self.is_lo = False
            if not self.banner.run_card['store_rwgt_info']:
                raise SystematicsError, 'The events have not been generated with store_rwgt_info=True. Cannot evaluate systematics error on these events.'

        # MUR/MUF/ALPS PARSING
        if isinstance(mur, str):
            mur = mur.split(',')
        self.mur=[float(i) for i in mur]   
        if isinstance(muf, str):
            muf = muf.split(',')
        self.muf=[float(i) for i in muf]
        
        if isinstance(alps, str):
            alps = alps.split(',')
        self.alps=[float(i) for i in alps]

        # DYNAMICAL SCALE PARSING + together
        if isinstance(dyn, str):
            dyn = dyn.split(',')
        self.dyn=[int(i) for i in dyn]
        # For FxFx only mode -1 makes sense
        if matching_mode == 3:
            self.dyn = [-1]
        # avoid sqrts at NLO if ISR is possible
        if 4 in self.dyn and self.b1 and self.b2 and not self.is_lo:
            self.dyn.remove(4)

        if isinstance(together, str):
            self.together = together.split(',')
        else:
            self.together = together
            
        # START/STOP EVENT                                   
        self.start_event=int(start_event)
        self.stop_event=int(stop_event)
        if start_event != 0:
            self.log( "#starting from event #%s" % start_event)
        if stop_event != sys.maxint:
            self.log( "#stopping at event #%s" % stop_event)
        
        # LHAPDF set 
        if isinstance(lhapdf_config, list):
            lhapdf_config = lhapdf_config[0]
        lhapdf = misc.import_python_lhapdf(lhapdf_config)
        if not lhapdf:
            log('fail to load lhapdf: doe not perform systematics')
            return
        lhapdf.setVerbosity(0)
        self.pdfsets = {}  
        if isinstance(pdf, str):
            pdf = pdf.split(',')
            
        if isinstance(pdf,list) and isinstance(pdf[0],(str,int)):
            self.pdf = []
            for data in pdf:
                if data == 'errorset':
                    data = '%s' % self.orig_pdf
                if data == 'central':
                    data = '%s@0' % self.orig_pdf
                if '@' in data:
                    #one particular dataset
                    name, arg = data.rsplit('@',1)
                    if int(arg) == 0:
                        if name.isdigit():
                            self.pdf.append(lhapdf.mkPDF(int(name)))
                        else:
                            self.pdf.append(lhapdf.mkPDF(name))
                    elif name.isdigit():
                        try:
                            self.pdf.append(lhapdf.mkPDF(int(name)+int(arg)))
                        except:
                            raise Exception, 'Individual error sets need to be called with LHAPDF NAME not with LHAGLUE NUMBER'
                    else:
                        self.pdf.append(lhapdf.mkPDF(name, int(arg)))
                else:
                    if data.isdigit():
                        pdfset = lhapdf.mkPDF(int(data)).set()
                    else:
                        pdfset = lhapdf.mkPDF(data).set()
                    self.pdfsets[pdfset.lhapdfID] = pdfset 
                    self.pdf += pdfset.mkPDFs()
        else:
            self.pdf = pdf
            
        for p in self.pdf:
            if p.lhapdfID == self.orig_pdf:
                self.orig_pdf = p
                break
        else:
            self.orig_pdf = lhapdf.mkPDF(self.orig_pdf)
        if not self.b1 == 0 == self.b2: 
            self.log( "# events generated with PDF: %s (%s)" %(self.orig_pdf.set().name,self.orig_pdf.lhapdfID ))
        # create all the function that need to be called
        self.get_all_fct() # define self.fcts and self.args
        
        # For e+/e- type of collision initialise the running of alps
        if self.b1 == 0 == self.b2:
            try:
                from models.model_reader import Alphas_Runner
            except ImportError:
                root_path = pjoin(root, os.pardir, os.pardir)
                try:
                    import internal.madevent_interface as me_int
                    cmd = me_int.MadEventCmd(root_path,force_run=True)
                except ImportError:
                    import internal.amcnlo_run_interface as me_int
                    cmd = me_int.Cmd(root_path,force_run=True)                
                if 'mg5_path' in cmd.options and cmd.options['mg5_path']:
                    sys.path.append(cmd.options['mg5_path'])
                from models.model_reader import Alphas_Runner
                
            if not hasattr(self.banner, 'param_card'):
                param_card = self.banner.charge_card('param_card')
            else:
                param_card = self.banner.param_card
            
            asmz = param_card.get_value('sminputs', 3, 0.13)
            nloop =2
            zmass = param_card.get_value('mass', 23, 91.188)
            cmass = param_card.get_value('mass', 4, 1.4)
            if cmass == 0:
                cmass = 1.4
            bmass = param_card.get_value('mass', 5, 4.7)
            if bmass == 0:
                bmass = 4.7
            self.alpsrunner = Alphas_Runner(asmz, nloop, zmass, cmass, bmass)
        
        # Store which weight to keep/removed
        self.remove_wgts = []
        for id in remove_wgts:
            if id == 'all':
                self.remove_wgts = ['all']
                break
            elif ',' in id:
                min_value, max_value = [int(v) for v in id.split(',')]
                self.remove_wgts += [i for i in range(min_value, max_value+1)]
            else:
                self.remove_wgts.append(id)
        self.keep_wgts = []
        for id in keep_wgts:
            if id == 'all':
                self.keep_wgts = ['all']
                break
            elif ',' in id:
                min_value, max_value = [int(v) for v in id.split(',')]
                self.remove_wgts += [i for i in range(min_value, max_value+1)]
            else:
                self.remove_wgts.append(id)  
                
        # input to start the id in the weight
        self.start_wgt_id = int(start_id[0]) if (start_id is not None) else None
        self.has_wgts_pattern = False # tag to check if the pattern for removing
                                      # the weights was computed already
        
    def is_wgt_kept(self, name):
        """ determine if we have to keep/remove such weight """
        
        if 'all' in self.keep_wgts or not self.remove_wgts:
            return True

        #start by checking what we want to keep        
        if name in self.keep_wgts: 
            return True
        
        # check for regular expression
        if not self.has_wgts_pattern:
            pat = r'|'.join(w for w in self.keep_wgts if any(letter in w for letter in '*?.([+\\'))
            if pat:
                self.keep_wgts_pattern = re.compile(pat)
            else:
                self.keep_wgts_pattern = None
            pat = r'|'.join(w for w in self.remove_wgts if any(letter in w for letter in '*?.([+\\'))
            if pat:
                self.rm_wgts_pattern = re.compile(pat)
            else:
                self.rm_wgts_pattern = None                
            self.has_wgts_pattern=True
            
        if self.keep_wgts_pattern and re.match(self.keep_wgts_pattern,name):
            return True

        #check what we want to remove
        if 'all' in self.remove_wgts:
            return False
        elif name in self.remove_wgts:
            return False
        elif self.rm_wgts_pattern and re.match(self.rm_wgts_pattern, name):
            return False
        else:
            return True

    def remove_old_wgts(self, event):
        """remove the weight as requested by the user"""
        
        rwgt_data = event.parse_reweight()
        for name in rwgt_data.keys():
            if not self.is_wgt_kept(name):
                del rwgt_data[name]
                event.reweight_order.remove(name)
        
        
    def run(self, stdout=sys.stdout):
        """ """
        start_time = time.time()
        if self.start_event == 0 or self.force_write_banner:
            lowest_id = self.write_banner(self.output)
        else:
            lowest_id = self.get_id()        

        ids = [lowest_id+i for i in range(len(self.args)-1)]
        all_cross = [0 for i in range(len(self.args))]
        
        self.input.parsing = False
        for nb_event,event in enumerate(self.input):
            if nb_event < self.start_event:
                continue
            elif nb_event == self.start_event:
                self.input.parsing = True
                event = lhe_parser.Event(event)
            elif nb_event >= self.stop_event:
                if self.force_write_banner:
                    self.output.write('</LesHouchesEvents>\n')
                break
            
            if self.is_lo:
                if (nb_event-self.start_event)>=0 and (nb_event-self.start_event) % 2500 ==0:
                    self.log( '# currently at event %s [elapsed time: %.2g s]' % (nb_event, time.time()-start_time))
            else:
                if (nb_event-self.start_event)>=0 and (nb_event-self.start_event) % 1000 ==0:
                    self.log( '# currently at event %i [elapsed time: %.2g s]' % (nb_event, time.time()-start_time))
                    
            self.new_event() #re-init the caching of alphas/pdf
            self.remove_old_wgts(event)
            if self.is_lo:
                wgts = [self.get_lo_wgt(event, *arg) for arg in self.args]
            else:
                wgts = [self.get_nlo_wgt(event, *arg) for arg in self.args]
            
            if wgts[0] == 0:
                print wgts
                print event
                raise Exception
            
            wgt = [event.wgt*wgts[i]/wgts[0] for i in range(1,len(wgts))]
            all_cross = [(all_cross[j] + event.wgt*wgts[j]/wgts[0]) for j in range(len(wgts))]
            
            rwgt_data = event.parse_reweight()
            rwgt_data.update(zip(ids, wgt))
            event.reweight_order += ids
            # order the 
            self.output.write(str(event))
        else:
            self.output.write('</LesHouchesEvents>\n')
        self.output.close()
        self.print_cross_sections(all_cross, min(nb_event,self.stop_event)-self.start_event+1, stdout)
        
        if self.output.name != self.output_path:
            #check problem for .gz missinf in output.name
            if not os.path.exists(self.output.name) and os.path.exists('%s.gz' % self.output.name):
                to_check = '%s.gz' % self.output.name
            else:
                to_check = self.output.name
            
            if to_check != self.output_path:
                if '%s.gz' % to_check == self.output_path:
                    misc.gzip(to_check) 
                else:
                    import shutil
                    shutil.move(self.output.name, self.output_path)
        
        return all_cross
        
    def print_cross_sections(self, all_cross, nb_event, stdout):
        """print the cross-section."""
        
        norm = self.banner.get('run_card', 'event_norm', default='sum')
        #print "normalisation is ", norm
        #print "nb_event is ", nb_event
    
        max_scale, min_scale = 0,sys.maxint
        max_alps, min_alps = 0, sys.maxint
        max_dyn, min_dyn = 0,sys.maxint
        pdfs = {}
        dyns = {} # dyn : {'max': , 'min':}

        if norm == 'sum':
            norm = 1
        elif norm in ['average', 'bias']:
            norm = 1./nb_event
        elif norm == 'unity':
            norm = 1
            
        all_cross = [c*norm for c in all_cross]
        stdout.write("# mur\t\tmuf\t\talpsfact\tdynamical_scale\tpdf\t\tcross-section\n")
        for i,arg in enumerate(self.args):
            
            to_print = list(arg)
            to_print[4] = to_print[4].lhapdfID
            to_print.append(all_cross[i])  
            to_report = []  
            stdout.write('%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % tuple(to_print)) 
            
            mur, muf, alps, dyn, pdf = arg[:5]
            if pdf == self.orig_pdf and (dyn==self.orig_dyn or dyn==-1)\
                                              and (mur!=1 or muf!=1 or alps!=1):
                max_scale = max(max_scale,all_cross[i])
                min_scale = min(min_scale,all_cross[i])
            if pdf == self.orig_pdf and mur==1 and muf==1 and \
                                    (dyn==self.orig_dyn or dyn==-1) and alps!=1:
                max_alps = max(max_alps,all_cross[i])
                min_alps = min(min_alps,all_cross[i]) 
            
            if pdf == self.orig_pdf and mur==1 and muf==1 and alps==1:
                max_dyn = max(max_dyn,all_cross[i])
                min_dyn = min(min_dyn,all_cross[i])
                                            
            if pdf == self.orig_pdf and (alps!=1 or mur!=1 or muf!=1) and \
                                                (dyn!=self.orig_dyn or dyn!=-1):
                if dyn not in dyns:
                    dyns[dyn] = {'max':0, 'min':sys.maxint,'central':0}
                curr = dyns[dyn]
                curr['max'] = max(curr['max'],all_cross[i])
                curr['min'] = min(curr['min'],all_cross[i])
            if pdf == self.orig_pdf and (alps==1 and mur==1 and muf==1) and \
                                                (dyn!=self.orig_dyn or dyn!=-1):
                if dyn not in dyns:
                    dyns[dyn] = {'max':0, 'min':sys.maxint,'central':all_cross[i]}
                else:
                    dyns[dyn]['central'] = all_cross[i]          
                
            if alps==1 and mur==1 and muf==1 and (dyn==self.orig_dyn or dyn==-1):
                pdfset = pdf.set()
                if pdfset.lhapdfID in self.pdfsets:
                    if pdfset.lhapdfID not in pdfs :
                        pdfs[pdfset.lhapdfID] = [0] * pdfset.size
                    pdfs[pdfset.lhapdfID][pdf.memberID] = all_cross[i]
                else:
                    to_report.append('# PDF %s : %s\n' % (pdf.lhapdfID, all_cross[i]))
  
        stdout.write('\n') 
                
        resume = StringIO.StringIO()
                
        resume.write( '#***************************************************************************\n')
        resume.write( "#\n")
        resume.write( '# original cross-section: %s\n' % all_cross[0])
        if max_scale:
            resume.write( '#     scale variation: +%2.3g%% -%2.3g%%\n' % ((max_scale-all_cross[0])/all_cross[0]*100,(all_cross[0]-min_scale)/all_cross[0]*100))
        if max_alps:
            resume.write( '#     emission scale variation: +%2.3g%% -%2.3g%%\n' % ((max_alps-all_cross[0])/all_cross[0]*100,(max_alps-min_scale)/all_cross[0]*100))
        if max_dyn and (max_dyn!= all_cross[0] or min_dyn != all_cross[0]):
            resume.write( '#     central scheme variation: +%2.3g%% -%2.3g%%\n' % ((max_dyn-all_cross[0])/all_cross[0]*100,(all_cross[0]-min_dyn)/all_cross[0]*100))
        if self.orig_pdf.lhapdfID in pdfs:
            lhapdfid = self.orig_pdf.lhapdfID
            values = pdfs[lhapdfid]
            pdfset = self.pdfsets[lhapdfid]
            try:
                pdferr =  pdfset.uncertainty(values)
            except RuntimeError:
                resume.write( '# PDF variation: missing combination\n')
            else:
                resume.write( '# PDF variation: +%2.3g%% -%2.3g%%\n' % (pdferr.errplus*100/all_cross[0], pdferr.errminus*100/all_cross[0]))       
        # report error/central not directly linked to the central
        resume.write( "#\n")        
        for lhapdfid,values in pdfs.items():
            if lhapdfid == self.orig_pdf.lhapdfID:
                continue
            if len(values) == 1 :
                continue
            pdfset = self.pdfsets[lhapdfid]

            if pdfset.errorType == 'unknown' :
                # Don't know how to determine uncertainty for 'unknown' errorType :
                # File "lhapdf.pyx", line 329, in lhapdf.PDFSet.uncertainty (lhapdf.cpp:6621)
                # RuntimeError: "ErrorType: unknown" not supported by LHAPDF::PDFSet::uncertainty.
                continue
            try:
                pdferr =  pdfset.uncertainty(values)
            except RuntimeError:
                # the same error can happend to some other type of error like custom.
                pass
            else:
                resume.write( '#PDF %s: %g +%2.3g%% -%2.3g%%\n' % (pdfset.name, pdferr.central,pdferr.errplus*100/all_cross[0], pdferr.errminus*100/all_cross[0]))

        dyn_name = {1: '\sum ET', 2:'\sum\sqrt{m^2+pt^2}', 3:'0.5 \sum\sqrt{m^2+pt^2}',4:'\sqrt{\hat s}' }
        for key, curr in dyns.items():
            if key ==-1:
                continue
            central, maxvalue, minvalue = curr['central'], curr['max'], curr['min']
            if central == 0:
                continue
            if maxvalue == 0:
                resume.write("# dynamical scheme # %s : %g # %s\n" %(key, central, dyn_name[key]))
            else:
                resume.write("# dynamical scheme # %s : %g +%2.3g%% -%2.3g%% # %s\n" %(key, central, (maxvalue-central)/central*100,(central-minvalue)/central*100, dyn_name[key]))
      
        resume.write('\n'.join(to_report))

        resume.write( '#***************************************************************************\n')
    
        stdout.write(resume.getvalue())
        self.log(resume.getvalue())
    
    
    def write_banner(self, fsock):
        """create the new banner with the information of the weight"""

        cid = self.get_id()
        lowest_id = cid
        
        in_scale = False
        in_pdf = False
        in_alps = False
        
        text = ''
        
        default = self.args[0]
        for arg in self.args[1:]:
            mur, muf, alps, dyn, pdf = arg[:5]
            if pdf == self.orig_pdf and alps ==1 and (mur!=1 or muf!=1 or dyn!=-1):
                if not in_scale:
                    text += "<weightgroup name=\"Central scale variation\" combine=\"envelope\">\n"
                    in_scale=True
            elif in_scale:
                if (pdf == self.orig_pdf and alps ==1) and arg != default:
                    pass
                else:
                    text += "</weightgroup> # scale\n"
                    in_scale = False
                
            if pdf == self.orig_pdf and mur == muf == 1 and dyn==-1 and alps!=1:
                if not in_alps:
                    text += "<weightgroup name=\"Emission scale variation\" combine=\"envelope\">\n"
                    in_alps=True
            elif in_alps:
                text += "</weightgroup> # ALPS\n"
                in_alps=False
            
            if mur == muf == 1 and dyn==-1 and alps ==1:
                if pdf.lhapdfID < 0:
                    for central,sets in self.pdfsets.items():
                        if pdf in sets.set():
                            misc.sprint(central)
                
                if pdf.lhapdfID in self.pdfsets:
                    if in_pdf:
                        text += "</weightgroup> # PDFSET -> PDFSET\n"
                    pdfset = self.pdfsets[pdf.lhapdfID]
                    descrip = pdfset.description.replace('=>',';').replace('>','.gt.').replace('<','.lt.')
                    text +="<weightgroup name=\"%s\" combine=\"%s\"> # %s: %s\n" %\
                            (pdfset.name, pdfset.errorType,pdfset.lhapdfID, descrip)
                    in_pdf=pdf.lhapdfID 
                elif pdf.memberID == 0 and (pdf.lhapdfID - pdf.memberID) in self.pdfsets:
                    if in_pdf:
                        text += "</weightgroup> # PDFSET -> PDFSET\n"
                    pdfset = self.pdfsets[pdf.lhapdfID - 1]
                    descrip = pdfset.description.replace('=>',';').replace('>','.gt.').replace('<','.lt.')
                    text +="<weightgroup name=\"%s\" combine=\"%s\"> # %s: %s\n" %\
                            (pdfset.name, pdfset.errorType,pdfset.lhapdfID, descrip)
                    in_pdf=pdfset.lhapdfID 
                elif in_pdf and pdf.lhapdfID - pdf.memberID != in_pdf:
                    misc.sprint(pdf.lhapdfID)
                    text += "</weightgroup> # PDFSET -> PDF\n"
                    in_pdf = False 
            elif in_pdf:
                text += "</weightgroup> PDF \n"
                in_pdf=False                
                 
                
                
            tag, info = '',''
            if mur!=1.:
                tag += 'MUR="%s" ' % mur
                info += 'MUR=%s ' % mur
            else:
                tag += 'MUR="%s" ' % mur
            if muf!=1.:
                tag += 'MUF="%s" ' % muf
                info += 'MUF=%s ' % muf
            else:
                tag += 'MUF="%s" ' % muf
                
            if alps!=1.:
                tag += 'ALPSFACT="%s" ' % alps
                info += 'alpsfact=%s ' % alps
            if dyn!=-1.:
                tag += 'DYN_SCALE="%s" ' % dyn
                info += 'dyn_scale_choice=%s ' % {1:'sum pt', 2:'HT',3:'HT/2',4:'sqrts'}[dyn]
                                           
            if pdf != self.orig_pdf:
                tag += 'PDF="%s" ' % pdf.lhapdfID
                info += 'PDF=%s MemberID=%s' % (pdf.lhapdfID-pdf.memberID, pdf.memberID)
            else:
                tag += 'PDF="%s" ' % pdf.lhapdfID
                
            text +='<weight id="%s" %s> %s </weight>\n' % (cid, tag, info)
            cid+=1
        
        if in_scale or in_alps or in_pdf:
             text += "</weightgroup>\n"
            
        if 'initrwgt' in self.banner:
            if not self.remove_wgts:
                self.banner['initrwgt'] += text
            else:
                # remove the line which correspond to removed weight
                # removed empty group.
                wgt_in_group=0
                tmp_group_txt =[]
                out =[]
                keep_last = False
                for line in self.banner['initrwgt'].split('\n'):
                    sline = line.strip()
                    if sline.startswith('</weightgroup'):
                        if wgt_in_group:
                            out += tmp_group_txt
                            out.append('</weightgroup>')
                        if '<weightgroup' in line:
                            wgt_in_group=0
                            tmp_group_txt = [line[line.index('<weightgroup'):]]                            
                    elif sline.startswith('<weightgroup'):
                        wgt_in_group=0
                        tmp_group_txt = [line]   
                    elif sline.startswith('<weight'):
                        name = re.findall(r'\bid=[\'\"]([^\'\"]*)[\'\"]', sline)
                        if self.is_wgt_kept(name[0]):
                            tmp_group_txt.append(line)
                            keep_last = True
                            wgt_in_group +=1
                        else:
                            keep_last = False
                    elif keep_last:
                        tmp_group_txt.append(line)
                out.append(text)
                self.banner['initrwgt'] = '\n'.join(out) 
        else:
            self.banner['initrwgt'] = text
            
        
        self.banner.write(self.output, close_tag=False)
        
        return lowest_id
        

    def get_id(self):
        
        if self.start_wgt_id is not None:
            return int(self.start_wgt_id)
        
        if 'initrwgt' in self.banner:
            pattern = re.compile('<weight id=(?:\'|\")([_\w]+)(?:\'|\")', re.S+re.I+re.M)
            matches =  pattern.findall(self.banner['initrwgt'])
            matches.append('0') #ensure to have a valid entry for the max 
            return  max([int(wid) for wid in  matches if wid.isdigit()])+1
        else:
            return 1
        
        
    def get_all_fct(self):
        
        all_args = []
        default = [1.,1.,1.,-1,self.orig_pdf]
        #all_args.append(default)
        pos = {'mur':0, 'muf':1, 'alps':2, 'dyn':3, 'pdf':4}
        done = set()
        for one_block in self.together:
            for name in one_block:
                done.add(name)
            for together in itertools.product(*[getattr(self,name) for name in one_block]):
                new_args = list(default)
                for name,value in zip(one_block, together):
                    new_args[pos[name]] = value
                all_args.append(new_args)
        for name in pos:
            if name in done:
                continue
            for value in getattr(self, name):
                new_args = list(default)
                new_args[pos[name]] = value
                all_args.append(new_args)
        
        self.args = [default] + [arg for arg in all_args if arg!= default]

        # add the default before the pdf scan to have a full grouping
        pdfplusone = [pdf for pdf in self.pdf if pdf.lhapdfID == self.orig_pdf.lhapdfID+1]
        if pdfplusone:
            pdfplusone = default[:-1] + [pdfplusone[0]] 
            index = self.args.index(pdfplusone)
            self.args.insert(index, default)

        self.log( "#Will Compute %s weights per event." % (len(self.args)-1))
        return
    
    def new_event(self):
        self.alphas = {}
        self.pdfQ2 = {}
        
            
    def get_pdfQ(self, pdf, pdg, x, scale, beam=1):
        
        if pdg in [-21,-22]:
            pdg = abs(pdg)
        elif pdg == 0:
            return 1
        
        if self.only_beam and self.only_beam!= beam and pdf.lhapdfID != self.orig_pdf:
            return self.getpdfQ(self.pdfsets[self.orig_pdf], pdg, x, scale, beam)
        
        if self.orig_ion_pdf and (self.ion_scaling or pdf.lhapdfID == self.orig_pdf):
            nb_p = self.banner.run_card["nb_proton%s" % beam]
            nb_n = self.banner.run_card["nb_neutron%s" % beam]


            if pdg in [1,2]:
                pdf1 =  pdf.xfxQ(1, x, scale)/x
                pdf2 =  pdf.xfxQ(2, x, scale)/x
                if pdg == 1:
                    f = nb_p * pdf1 + nb_n * pdf2
                else:
                    f = nb_p * pdf2 + nb_n * pdf1
            elif pdg in [-1,-2]:
                pdf1 =  pdf.xfxQ(-1, x, scale)/x
                pdf2 =  pdf.xfxQ(-2, x, scale)/x
                if pdg == -1:
                    f = nb_p * pdf1 + nb_n * pdf2
                else:
                    f = nb_p * pdf2 + nb_n * pdf1                    
            else: 
                f = (nb_p + nb_n) * pdf.xfxQ(pdg, x, scale)/x
                
            f = f * (nb_p+nb_n) 
        else:
            f = pdf.xfxQ(pdg, x, scale)/x
#        if f == 0 and pdf.memberID ==0:
#            pdfset = pdf.set()
#            allnumber= [p.xfxQ(pdg, x, scale) for p in pdfset.mkPDFs()]
#            f = pdfset.uncertainty(allnumber).central /x
        return f

    def get_pdfQ2(self, pdf, pdg, x, scale, beam=1):

        if pdg in [-21,-22]:
            pdg = abs(pdg)
        elif pdg == 0:
            return 1
        
        if (pdf, pdg,x,scale, beam) in self.pdfQ2:
            return self.pdfQ2[(pdf, pdg,x,scale,beam)]
        
        if self.orig_ion_pdf and (self.ion_scaling or pdf.lhapdfID == self.orig_pdf):
            nb_p = self.banner.run_card["nb_proton%s" % beam]
            nb_n = self.banner.run_card["nb_neutron%s" % beam]


            if pdg in [1,2]:
                pdf1 =  pdf.xfxQ2(1, x, scale)/x
                pdf2 =  pdf.xfxQ2(2, x, scale)/x
                if pdg == 1:
                    f = nb_p * pdf1 + nb_n * pdf2
                else:
                    f = nb_p * pdf2 + nb_n * pdf1
            elif pdg in [-1,-2]:
                pdf1 =  pdf.xfxQ2(-1, x, scale)/x
                pdf2 =  pdf.xfxQ2(-2, x, scale)/x
                if pdg == -1:
                    f = nb_p * pdf1 + nb_n * pdf2
                else:
                    f = nb_p * pdf2 + nb_n * pdf1                    
            else: 
                f = (nb_p + nb_n) * pdf.xfxQ2(pdg, x, scale)/x
                
            f = f * (nb_p+nb_n)      
        else:
            f = pdf.xfxQ2(pdg, x, scale)/x
        self.pdfQ2[(pdf, pdg,x,scale,beam)] = f
        return f        
        
        
        
        #one method to handle the nnpd2.3 problem -> now just move to central
        if f == 0 and pdf.memberID ==0:
            # to avoid problem with nnpdf2.3 in lhapdf6.1.6
            #print 'central pdf returns 0', pdg, x, scale
            #print self.pdfsets
            pdfset = pdf.set()
            allnumber= [0] + [self.get_pdfQ2(p, pdg, x, scale) for p in pdfset.mkPDFs()[1:]]
            f = pdfset.uncertainty(allnumber).central
        self.pdfQ2[(pdf, pdg,x,scale)] = f
        return f
                
    def get_lo_wgt(self,event, Dmur, Dmuf, Dalps, dyn, pdf):
        """ 
        pdf is a lhapdf object!"""
        
        loinfo = event.parse_lo_weight()

        if dyn == -1:
            mur = loinfo['ren_scale']
            muf1 = loinfo['pdf_q1'][-1]
            muf2 = loinfo['pdf_q2'][-1]
        else:
            if dyn == 1: 
                mur = event.get_et_scale(1.)
            elif dyn == 2:
                mur = event.get_ht_scale(1.)
            elif dyn == 3:
                mur = event.get_ht_scale(0.5)
            elif dyn == 4:
                mur = event.get_sqrts_scale(1.)
            muf1 = mur
            muf2 = mur
            loinfo = dict(loinfo)
            loinfo['pdf_q1'] = loinfo['pdf_q1'] [:-1] + [mur]
            loinfo['pdf_q2'] = loinfo['pdf_q2'] [:-1] + [mur]
            
        
        
        # MUR part
        if self.b1 == 0 == self.b2:
            if loinfo['n_qcd'] != 0:
                wgt = self.alpsrunner(Dmur*mur)**loinfo['n_qcd']
            else:
                wgt = 1.0
        else:
            wgt = pdf.alphasQ(Dmur*mur)**loinfo['n_qcd']
        # MUF/PDF part
        wgt *= self.get_pdfQ(pdf, self.b1*loinfo['pdf_pdg_code1'][-1], loinfo['pdf_x1'][-1], Dmuf*muf1, beam=1) 
        wgt *= self.get_pdfQ(pdf, self.b2*loinfo['pdf_pdg_code2'][-1], loinfo['pdf_x2'][-1], Dmuf*muf2, beam=2) 

        for scale in loinfo['asrwt']:
            if self.b1 == 0 == self.b2:
                wgt = self.alpsrunner(Dalps*scale)
            else:
                wgt *= pdf.alphasQ(Dalps*scale)
        
        # ALS part
        for i in range(loinfo['n_pdfrw1']-1):
            scale = min(Dalps*loinfo['pdf_q1'][i], Dmuf*muf1)
            wgt *= self.get_pdfQ(pdf, self.b1*loinfo['pdf_pdg_code1'][i], loinfo['pdf_x1'][i], scale, beam=1)
            wgt /= self.get_pdfQ(pdf, self.b1*loinfo['pdf_pdg_code1'][i], loinfo['pdf_x1'][i+1], scale, beam=1)

        for i in range(loinfo['n_pdfrw2']-1):
            scale = min(Dalps*loinfo['pdf_q2'][i], Dmuf*muf2)
            wgt *= self.get_pdfQ(pdf, self.b2*loinfo['pdf_pdg_code2'][i], loinfo['pdf_x2'][i], scale, beam=2)
            wgt /= self.get_pdfQ(pdf, self.b2*loinfo['pdf_pdg_code2'][i], loinfo['pdf_x2'][i+1], scale, beam=2)            
        
        return wgt

    def get_nlo_wgt(self,event, Dmur, Dmuf, Dalps, dyn, pdf):
        """return the new weight for NLO event --with weight information-- """
        
        wgt = 0 
        nloinfo = event.parse_nlo_weight(real_type=(1,11,12,13))
        for cevent in nloinfo.cevents:
            if dyn == 1: 
                mur2 = max(1.0, cevent.get_et_scale(1.)**2) 
            elif dyn == 2:
                mur2 = max(1.0, cevent.get_ht_scale(1.)**2)
            elif dyn == 3:
                mur2 = max(1.0, cevent.get_ht_scale(0.5)**2)
            elif dyn == 4:
                mur2 = cevent.get_sqrts_scale(event,1)**2
            else:
                mur2 = 0
            muf2 = mur2
            
            for onewgt in cevent.wgts:
                if not __debug__ and (dyn== -1 and Dmur==1 and Dmuf==1 and pdf==self.orig_pdf):
                    wgt += onewgt.ref_wgt 
                
                if dyn == -1:
                    mur2 = onewgt.scales2[1]
                    muf2 = onewgt.scales2[2]
                Q2 = onewgt.scales2[0] # Ellis-Sexton scale
                
                wgtpdf = self.get_pdfQ2(pdf, self.b1*onewgt.pdgs[0], onewgt.bjks[0],
                                      Dmuf**2 * muf2)
                wgtpdf *= self.get_pdfQ2(pdf, self.b2*onewgt.pdgs[1], onewgt.bjks[1],
                                      Dmuf**2 * muf2)
                
                tmp = onewgt.pwgt[0]
                tmp += onewgt.pwgt[1] * math.log(Dmur**2 * mur2/ Q2)
                tmp += onewgt.pwgt[2] * math.log(Dmuf**2 * muf2/ Q2)
                
                if self.b1 == 0 == self.b2:
                    alps = self.alpsrunner(Dmur*math.sqrt(mur2))
                else:
                    alps = pdf.alphasQ2(Dmur**2*mur2)
                
                tmp *= math.sqrt(4*math.pi*alps)**onewgt.qcdpower
                
                if wgtpdf == 0: #happens for nn23pdf due to wrong set in lhapdf
                    key = (self.b1*onewgt.pdgs[0], self.b2*onewgt.pdgs[1], onewgt.bjks[0],onewgt.bjks[1], muf2)
                    if dyn== -1 and Dmuf==1 and Dmur==1 and pdf==self.orig_pdf:
                        wgtpdf = onewgt.ref_wgt / tmp
                        self.pdfQ2[key] = wgtpdf
                    elif key in self.pdfQ2:
                        wgtpdf = self.pdfQ2[key]
                    else:
                        # real zero!
                        wgtpdf = 0

                tmp *= wgtpdf                
                wgt += tmp
                
                
                if __debug__ and dyn== -1 and Dmur==1 and Dmuf==1 and pdf==self.orig_pdf:
                    if not misc.equal(tmp, onewgt.ref_wgt, sig_fig=2):
                        misc.sprint(tmp, onewgt.ref_wgt, (tmp-onewgt.ref_wgt)/tmp)
                        misc.sprint(onewgt)
                        misc.sprint(cevent)
                        misc.sprint(mur2,muf2)
                        raise Exception, 'not enough agreement between stored value and computed one'
                
        return wgt
                            

def call_systematics(args, result=sys.stdout, running=True,
                     log=lambda x:sys.stdout.write(str(x)+'\n')):
    """calling systematics from a list of arguments"""            

    input, output = args[0:2]
    opts = {}
    for arg in args[2:]:
        if '=' in arg:
            key,values= arg.split('=')
            key = key.replace('-','')
            values = values.strip()
            if values[0] in ["'",'"'] and values[-1]==values[0]:
                values = values[1:-1]
            values = values.split(',')
            if key == 'together':
                if key in opts:
                    opts[key].append(tuple(values))
                else:
                    opts[key]=[tuple(values)]
            elif key == 'result':
                result = open(values[0],'w')
            elif key in ['start_event', 'stop_event', 'only_beam']:
                opts[key] = banner_mod.ConfigFile.format_variable(values[0], int, key)
            elif key in ['write_banner', 'ion_scalling']:
                opts[key] = banner_mod.ConfigFile.format_variable(values[0], bool, key)
            else:
                if key in opts:
                    opts[key] += values
                else:
                    opts[key] = values
        else:
            raise SystematicsError, "unknow argument %s" % arg

    #load run_card and extract parameter if needed.
    if 'from_card' in opts:
        if opts['from_card'] != ['internal']:
            card = banner.RunCard(opts['from_card'][0])
        else:
            for i in range(10):
                try:
                    lhe = lhe_parser.EventFile(input)
                    break
                except OSError,error:
                    print error
                    time.sleep(15*(i+1))
            else:
                raise
                    
            card = banner.RunCard(banner.Banner(lhe.banner)['mgruncard'])
            lhe.close()
            
        if isinstance(card, banner.RunCardLO):
            # LO case
            opts['mur'] = [float(x) for x in card['sys_scalefact'].split()]
            opts['muf'] = opts['mur']
            if card['sys_alpsfact'] != 'None':
                opts['alps'] = [float(x) for x in card['sys_alpsfact'].split()]
            else:
                opts['alps'] = [1.0]
            opts['together'] = [('mur','muf','alps','dyn')]
            if '&&' in card['sys_pdf']:
                pdfs =  card['sys_pdf'].split('&&')
            else:
                data = card['sys_pdf'].split()
                pdfs = []
                for d in data:
                    if not d.isdigit():
                        pdfs.append(d)
                    elif int(d) > 500:
                        pdfs.append(d)
                    else:
                        pdfs[-1] = '%s %s' % (pdfs[-1], d)

            opts['dyn'] = [-1,1,2,3,4]
            opts['pdf'] = []
            for pdf in pdfs:
                split = pdf.split()
                if len(split)==1:
                    opts['pdf'].append('%s' %pdf)
                else:
                    pdf,nb = split
                    for i in range(int(nb)):
                        opts['pdf'].append('%s@%s' % (pdf, i))
            if not opts['pdf']:
                opts['pdf'] = 'central'
        else:
            #NLO case
            raise Exception
        del opts['from_card']
    

    obj = Systematics(input, output, log=log, **opts)
    if running and obj:
        obj.run(result)  
    return obj

if __name__ == "__main__":
    sys_args = sys.argv[1:]
    for i, arg in enumerate(sys_args) :
        if arg.startswith('--lhapdf_config=') :
            lhapdf = misc.import_python_lhapdf(arg[len('--lhapdf_config='):])
            del sys_args[i]
            break

    if 'lhapdf' not in globals():
        lhapdf = misc.import_python_lhapdf('lhapdf-config')
         
    if not lhapdf:
            sys.exit('Can not run systematics since can not link python to lhapdf, specify --lhapdf-config=')
    call_systematics(sys_args)
    

   
   
        
        
        
    
    
        
