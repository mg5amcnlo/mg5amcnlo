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
    sys.path.append('../../')
import lhe_parser
import banner as banner_mod
import itertools
import misc
import math
import re
import time

class Systematics(object):
    
    def __init__(self, input_file, output_file,
                 start_event=0, stop_event=sys.maxint, write_banner=False,
                 mur=[0.5,1,2],
                 muf=[0.5,1,2],
                 alps=[1],
                 pdf='errorset', #[(id, subset)]
                 dyn=[-1],
                 together=[('mur', 'muf')],
                 lhapdf_config=misc.which('lhapdf-config')
                 ):
        
        # INPUT/OUTPUT FILE
        if isinstance(input_file, str):
            self.input = lhe_parser.EventFile(input_file)
        else:
            self.input = input_file
        if isinstance(output_file, str):
            self.output =  lhe_parser.EventFile(output_file, 'w')
        else:
            self.output = output_file
            
        #get some information from the run_card.
        self.banner = banner_mod.Banner(self.input.banner)  
        self.force_write_banner = bool(write_banner)
        self.orig_dyn = self.banner.get('run_card', 'dynamical_scale_choice')
        self.orig_pdf = self.banner.run_card.get_lhapdf_id()
    
        if isinstance(self.banner.run_card, banner_mod.RunCardLO):
            self.is_lo = True
        else:
            self.is_lo = False

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
 
        if isinstance(together, str):
            self.together = together.split(',')
        else:
            self.together = together
            
        # START/STOP EVENT                                   
        self.start_event=int(start_event)
        self.stop_event=int(stop_event)
        if start_event != 0:
            print "starting from event #%s" % start_event    
        if stop_event != sys.maxint:
            print "stoping at event #%s" % stop_event          
            
        # LHAPDF set 
        lhapdf = misc.import_python_lhapdf(lhapdf_config)
        lhapdf.setVerbosity(0)
        self.pdfsets = {}  
        if isinstance(pdf, str):
            pdf = pdf.split(',')
        if isinstance(pdf,list) and isinstance(pdf[0],(str,int)):
            self.pdf = []
            for data in pdf:
                if data == 'errorset':
                    data = '%s@*' % self.orig_pdf
                if data == 'central':
                    data = '%s' % self.orig_pdf
                if data.isdigit():
                    self.pdf.append(lhapdf.mkPDF(int(data)))
                elif '@' in data:
                    name, arg = data.rsplit('@',1)
                    if not(arg) or arg =='*':
                        if name.isdigit():
                            pdfset = lhapdf.mkPDF(int(name)).set()
                        else:
                            pdfset = lhapdf.getPDFSet(name)
                        self.pdfsets[pdfset.lhapdfID] = pdfset 
                        self.pdf += pdfset.mkPDFs()
                    elif name.isdigit():
                        raise Exception, 'invididual error set need to called with name not with lhapdfID'
                    else:
                        self.pdf.append(lhapdf.mkPDF(name, int(arg)))
                else:
                    
                    self.pdf.append(lhapdf.mkPDF(data))
        else:
            self.pdf = pdf
        for p in self.pdf:
            if p.lhapdfID == self.orig_pdf:
                self.orig_pdf = p
                break
        else:
            self.orig_pdf = lhapdf.mkPDF(self.orig_pdf)
        print "# events generated with PDF: %s (%s)" %(self.orig_pdf.set().name,self.orig_pdf.lhapdfID )             
                    
        # create all the function that need to be called
        self.get_all_fct() # define self.fcts and self.args

    def run(self):
        """ """
        start_time = time.time()
        if self.start_event == 0 or self.force_write_banner:
            lowest_id = self.write_banner(self.output)
        else:
            lowest_id = self.get_id()        

        ids = [lowest_id+i for i in range(len(self.args)-1)]
        all_cross = [0 for i in range(len(self.args))]
        
        for i,event in enumerate(self.input):
            if i < self.start_event:
                continue
            elif i >= self.stop_event:
                if self.force_write_banner:
                    self.output.write('</LesHouchesEvents>\n')
                break
            if self.is_lo:
                if (i-self.start_event)>=0 and (i-self.start_event) % 2500 ==0:
                    print '# currently at event', i, 'run in %.2g s' % (time.time()-start_time)
            else:
                if (i-self.start_event)>=0 and (i-self.start_event) % 1000 ==0:
                    print '# currently at event', i, 'run in %.2g s' % (time.time()-start_time)                
            
            self.new_event() #re-init the caching of alphas/pdf
            if self.is_lo:
                wgts = [self.get_lo_wgt(event, *arg) for arg in self.args]
            else:
                wgts = [self.get_nlo_wgt(event, *arg) for arg in self.args]
            
            if wgts[0] == 0:
                print wgts
                print event
                raise Exception
            
            wgt = [event.wgt*wgts[i]/wgts[0] for i in range(1,len(wgts))]
            all_cross = [(all_cross[i] + event.wgt*wgts[i]/wgts[0]) for i in range(len(wgts))]
            
            
            rwgt_data = event.parse_reweight()
            rwgt_data.update(zip(ids, wgt))
            self.output.write(str(event))
        else:
            self.output.write('</LesHouchesEvents>\n')
        
        self.print_cross_sections(all_cross, min(i,self.stop_event)-self.start_event)
        return all_cross
        
    def print_cross_sections(self, all_cross, nb_event):
        """print the cross-section."""
        
        norm = self.banner.get('run_card', 'event_norm', default='sum')
        #print "normalisation is ", norm
        #print "nb_event is ", nb_event
    
        max_scale, min_scale = 0,sys.maxint
        max_alps, min_alps = 0, sys.maxint
        max_other, min_other = 0,sys.maxint
        pdfs = {}
        
        if norm == 'sum':
            norm = 1
        elif norm == 'average':
            norm = 1./nb_event
        elif norm == 'unity':
            norm = 1
            
        all_cross = [c*norm for c in all_cross]
        print "# mur\t\tmuf\t\talpsfact\tdynamical_scale\tpdf\t\tcross-section"
        for i,arg in enumerate(self.args):
            
            to_print = list(arg)
            to_print[4] = to_print[4].lhapdfID
            to_print.append(all_cross[i])    
            print '%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s' % tuple(to_print) 
            
            mur, muf, alps, dyn, pdf = arg[:5]
            if pdf == self.orig_pdf and alps ==1 and (mur!=1 or muf!=1 or dyn!=-1):
                max_scale = max(max_scale,all_cross[i])
                min_scale = min(min_scale,all_cross[i])
            elif pdf == self.orig_pdf and mur==1 and muf==1 and dyn==-1 and alps!=1:
                max_alps = max(max_alps,all_cross[i])
                min_alps = min(min_alps,all_cross[i])                
            elif alps==1 and mur==1 and muf==1 and dyn==-1:
                pdfset = pdf.set()
                if pdfset.lhapdfID in self.pdfsets:
                    if pdfset.lhapdfID not in pdfs :
                        pdfs[pdfset.lhapdfID] = [0] * pdfset.size
                    pdfs[pdfset.lhapdfID][pdf.memberID] = all_cross[i]
                else:
                    max_other = max(max_other, all_cross[i])
                    min_other = min(min_other, all_cross[i])                     
            else:
                max_other = max(max_other, all_cross[i])
                min_other = min(min_other, all_cross[i])                 
        print 
        print '#*****************************************************************'
        print "#"
        print '# original cross-section:', all_cross[0]
        if max_scale:
            print '# scale variation: +%2.3g%% -%2.3g%%' % ((max_scale-all_cross[0])/all_cross[0]*100,(all_cross[0]-min_scale)/all_cross[0]*100) 
        if max_alps:
            print '# emission scale variation: +%2.3g%% -%2.3g%%' % ((max_alps-all_cross[0])/all_cross[0]*100,(max_alps-min_scale)/all_cross[0]*100) 
        for lhapdfid,values in pdfs.items():
            pdfset = self.pdfsets[lhapdfid]
            pdferr =  pdfset.uncertainty(values)
            if lhapdfid == self.orig_pdf.lhapdfID:
                print '# PDF variation: +%2.3g%% -%2.3g%%' % (pdferr.errplus*100/all_cross[0], pdferr.errminus*100/all_cross[0])
            else:
                print '#PDF %s: %s +%2.3g%% -%2.3g%%' % (pdfset.name, pdferr.central,pdferr.errplus*100/all_cross[0], pdferr.errminus*100/all_cross[0])
        if max_other:
            print '# Other variation: +%2.3g%% -%2.3g%%' % ((max_other-all_cross[0])/all_cross[0]*100,(all_cross[0]-min_other)/all_cross[0]*100) 
        print "#"            
        print '#*****************************************************************'
                
        




    def write_banner(self, fsock):
        """create the new banner with the information of the weight"""

        cid = self.get_id()
        lowest_id = cid
        
        in_scale = False
        in_pdf = False
        in_alps = False
        
        text = ''
        
        for arg in self.args[1:]:
            mur, muf, alps, dyn, pdf = arg[:5]
            if pdf == self.orig_pdf and alps ==1 and (mur!=1 or muf!=1 or dyn!=-1):
                if not in_scale:
                    text += "<weightgroup name=\"Central scale variation\" combine=\"envelope\">\n"
                    in_scale=True
            elif in_scale:
                if (pdf == self.orig_pdf and alps ==1):
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
            
            if pdf != self.orig_pdf and mur == muf == 1 and dyn==-1 and alps ==1:
                if pdf.lhapdfID in self.pdfsets:
                    if in_pdf:
                        text += "</weightgroup> # PDFSET -> PDFSET\n"
                    pdfset = self.pdfsets[pdf.lhapdfID]
                    text +="<weightgroup name=\"%s\" combine=\"%s\"> # %s: %s\n" %\
                            (pdfset.name, pdfset.errorType,pdfset.lhapdfID, pdfset.description)
                    in_pdf=pdf.lhapdfID 
                elif pdf.memberID == 1 and (pdf.lhapdfID - pdf.memberID) in self.pdfsets:
                    if in_pdf:
                        text += "</weightgroup> # PDFSET -> PDFSET\n"
                    pdfset = self.pdfsets[pdf.lhapdfID - 1]
                    text +="<weightgroup name=\"%s\" combine=\"%s\"> # %s: %s\n" %\
                            (pdfset.name, pdfset.errorType,pdfset.lhapdfID, pdfset.description)
                    in_pdf=pdfset.lhapdfID 
                elif in_pdf and pdf.lhapdfID - pdf.memberID != in_pdf:
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
                info += 'dyn_scale_choise=%s ' % {1:'sum pt', 2:'HT',3:'HT/2',4:'sqrts'}[dyn]
                                           
            if pdf != self.orig_pdf:
                tag += 'LHAPDF="%s" ' % pdf.lhapdfID
                info += 'LHAPDF=%s MemberID=%s' % (pdf.lhapdfID-pdf.memberID, pdf.memberID)
            else:
                tag += 'LHAPDF="%s" ' % pdf.lhapdfID
                
            text +='<weight id="%s" %s> %s </weight>\n' % (cid, tag, info)
            cid+=1
        
        if in_scale or in_alps or in_pdf:
             text += "</weightgroup>\n"
            
        if 'initrwgt' in self.banner:
            self.banner['initrwgt'] += text
        else:
            self.banner['initrwgt'] = text
            
        
        self.banner.write(self.output, close_tag=False)
        
        return lowest_id
        

    def get_id(self):
        
        if 'initrwgt' in self.banner:
            pattern = re.compile('<weight id=\'([_\w]+)\'', re.S+re.I+re.M)
            return  max([int(wid) for wid in  pattern.findall(self.banner['initrwgt'])])+1
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
        
        self.args = [default]+ [arg for arg in all_args if arg!= default]

        print "#Will Compute %s weights per event." % (len(self.args)-1)
        return
    
    def new_event(self):
        self.alphas = {}
        self.pdf = {}
        
            
    def get_pdfQ(self, pdf, pdg, x, scale):
        
        f = pdf.xfxQ(pdg, x, scale)
        if f == 0 and pdf.memberID ==0:
            pdfset = pdf.set()
            allnumber= [p.xfxQ(pdg, x, scale) for p in pdfset.mkPDFs()]
            f = pdfset.uncertainty(allnumber).central  
        return f

    def get_pdfQ2(self, pdf, pdg, x, scale):
        
        if (pdf, pdg,x,scale) in self.pdf:
            return self.pdf[(pdf, pdg,x,scale)]
        f = pdf.xfxQ2(pdg, x, scale)
        self.pdf[(pdf, pdg,x,scale)] = f
        return f        
        if f == 0 and pdf.memberID ==0:
            #print 'central pdf returns 0', pdg, x, scale
            print self.pdfsets
            pdfset = pdf.set()
            allnumber= [0] + [self.get_pdfQ2(p, pdg, x, scale) for p in pdfset.mkPDFs()[1:]]
            f = pdfset.uncertainty(allnumber).central
            f=1  
        self.pdf[(pdf, pdg,x,scale)] = f
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
                mur = event.get_pt_scale(1.)
            elif dyn == 2:
                mur = event.get_ht_scale(1.)
            elif dyn == 3:
                mur = event.get_ht_scale(0.5)
            elif dyn == 4:
                mur = event.get_sqrts_scale(1.)
            muf1 = mur
            muf2 = mur
            loinfo['pdf_q1'][-1] = mur
            loinfo['pdf_q2'][-1] = mur
            
        
            
        # MUR part
        wgt = pdf.alphasQ(Dmur*mur)**loinfo['n_qcd']
        # MUF/PDF part
        wgt *= self.get_pdfQ(pdf, loinfo['pdf_pdg_code1'][-1], loinfo['pdf_x1'][-1], Dmuf*muf1) 
        wgt *= self.get_pdfQ(pdf, loinfo['pdf_pdg_code2'][-1], loinfo['pdf_x2'][-1], Dmuf*muf2) 
        
        # ALS part
        for i in range(loinfo['n_pdfrw1']-1):
            scale = min(Dalps*loinfo['pdf_q1'][i], Dmuf*muf1)
            wgt *= self.get_pdfQ(pdf, loinfo['pdf_pdg_code1'][i], loinfo['pdf_x1'][i], scale)
            wgt /= self.get_pdfQ(pdf, loinfo['pdf_pdg_code1'][i], loinfo['pdf_x1'][i+1], scale)

        for i in range(loinfo['n_pdfrw2']-1):
            scale = min(Dalps*loinfo['pdf_q2'][i], Dmuf*muf1)
            wgt *= self.get_pdfQ(pdf, loinfo['pdf_pdg_code2'][i], loinfo['pdf_x2'][i], scale)
            wgt /= self.get_pdfQ(pdf, loinfo['pdf_pdg_code2'][i], loinfo['pdf_x2'][i+1], scale)            
            
        return wgt

    def get_nlo_wgt(self,event, Dmur, Dmuf, Dalps, dyn, pdf):
        """return the new weight for NLO event --with weight information-- """
        
        wgt = 0 
        nloinfo = event.parse_nlo_weight()
        for cevent in nloinfo.cevents:
            if dyn == 1: 
                mur2 = cevent.get_pt_scale(1.)**2
            elif dyn == 2:
                mur2 = cevent.get_ht_scale(1.)**2
            elif dyn == 3:
                mur2 = cevent.get_ht_scale(0.5)**2
            elif dyn == 4:
                mur2 = cevent.get_sqrts_scale(1.)**2
            else:
                mur2 = 0
            muf2 = mur2
            
            for onewgt in cevent.wgts:
                if dyn == -1:
                    mur2 = onewgt.scales2[1]
                    muf2 = onewgt.scales2[2]
                Q2 = onewgt.scales2[0] # Ellis-Sexton scale
                tmp = onewgt.pwgt[0]
                tmp += onewgt.pwgt[1] * math.log(Dmur**2 * mur2/ Q2)
                tmp += onewgt.pwgt[2] * math.log(Dmuf**2 * muf2/ Q2)
                
                tmp *= self.get_pdfQ2(pdf,onewgt.pdgs[0], onewgt.bjks[0],
                                      Dmuf**2 * muf2)                             
                tmp *= self.get_pdfQ2(pdf, onewgt.pdgs[1], onewgt.bjks[1],
                                      Dmuf**2 * muf2)
                tmp *= pdf.alphasQ2(Dmur**2*mur2)**onewgt.qcdpower
                
                wgt += tmp
                
        return wgt
                            
            
        

if __name__ == "__main__":
    
    input, output = sys.argv[1:3]
    opts = {}
    for arg in sys.argv[3:]:
        if '=' in arg:
            key,values= arg.split('=')
            key = key.replace('-','')
            values = values.split(',')
            if key == 'together':
                if key in opts:
                    opts[key].append(tuple(values))
                else:
                    opts[key]=[tuple(values)]
            elif key in ['start_event', 'stop_event']:
                opts[key] = int(values[0])
            elif key == 'write_banner':
                opts[key] = banner_mod.ConfigFile.format_variable(values[0], bool, 'write_banner')
            else:
                if key in opts:
                    opts[key] += values
                else:
                    opts[key] = values
        else:
            print "unknow argument", arg
            sys.exit(1)
    
    obj = Systematics(input, output, **opts)
    obj.run()
        
        
        
    
    
        
