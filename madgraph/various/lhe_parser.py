from __future__ import division
import collections
import random
import re
import operator
import numbers
import math
import time
import os
import shutil
import sys

pjoin = os.path.join

if '__main__' == __name__:
    import sys
    sys.path.append('../../')
import misc
import logging
import gzip
import banner as banner_mod
logger = logging.getLogger("madgraph.lhe_parser")

class Particle(object):
    """ """
    # regular expression not use anymore to speed up the computation
    #pattern=re.compile(r'''^\s*
    #    (?P<pid>-?\d+)\s+           #PID
    #    (?P<status>-?\d+)\s+            #status (1 for output particle)
    #    (?P<mother1>-?\d+)\s+       #mother
    #    (?P<mother2>-?\d+)\s+       #mother
    #    (?P<color1>[+-e.\d]*)\s+    #color1
    #    (?P<color2>[+-e.\d]*)\s+    #color2
    #    (?P<px>[+-e.\d]*)\s+        #px
    #    (?P<py>[+-e.\d]*)\s+        #py
    #    (?P<pz>[+-e.\d]*)\s+        #pz
    #    (?P<E>[+-e.\d]*)\s+         #E
    #    (?P<mass>[+-e.\d]*)\s+      #mass
    #    (?P<vtim>[+-e.\d]*)\s+      #displace vertex
    #    (?P<helicity>[+-e.\d]*)\s*      #helicity
    #    ($|(?P<comment>\#[\d|D]*))  #comment/end of string
    #    ''',66) #verbose+ignore case
    
    
    
    def __init__(self, line=None, event=None):
        """ """
        
        if isinstance(line, Particle):
            for key in line.__dict__:
                setattr(self, key, getattr(line, key))
            if event:
                self.event = event
            return
        else:
            try:
                import madgraph.various.hepmc_parser as hepmc_parser
            except Exception:
                pass
            else:
                if isinstance(line, hepmc_parser.HEPMC_Particle):
                    self.event = event
                    self.event_id = len(event) #not yet in the event
                    for key in ['pid', 'status', 'E','px','py','pz','mass']:
                        setattr(self, key, getattr(line, key))
                    self.mother1 = 1
                    self.mother2 = 1
                    self.color1 = 0
                    self.color2 = 0
                    self.vtim = 0
                    self.comment = ''
                    self.helicity = 9
                    self.rwgt = 0
                    return

                
        self.event = event
        self.event_id = len(event) #not yet in the event
        # LHE information
        self.pid = 0
        self.status = 0 # -1:initial. 1:final. 2: propagator
        self.mother1 = None
        self.mother2 = None
        self.color1 = 0
        self.color2 = None
        self.px = 0
        self.py = 0 
        self.pz = 0
        self.E = 0
        self.mass = 0
        self.vtim = 0
        self.helicity = 9
        self.rwgt = 0
        self.comment = ''

        if line:
            self.parse(line)
          
    @property
    def pdg(self):
        "convenient alias"
        return self.pid
            
    def parse(self, line):
        """parse the line"""
        
        args = line.split()
        keys = ['pid', 'status','mother1','mother2','color1', 'color2', 'px','py','pz','E',
                'mass','vtim','helicity']
        
        for key,value in zip(keys,args):
            setattr(self, key, float(value))
        self.pid = int(self.pid)
            
        self.comment = ' '.join(args[len(keys):])
        if self.comment.startswith(('|','#')):
            self.comment = self.comment[1:]

        # Note that mother1/mother2 will be modified by the Event parse function to replace the
        # integer by a pointer to the actual particle object.
    
    def __str__(self):
        """string representing the particles"""
        return " %8d %2d %4d %4d %4d %4d %+13.10e %+13.10e %+13.10e %14.10e %14.10e %10.4e %10.4e" \
            % (self.pid, 
               self.status,
               (self.mother1 if isinstance(self.mother1, numbers.Number) else self.mother1.event_id+1) if self.mother1 else 0,
               (self.mother2 if isinstance(self.mother2, numbers.Number) else self.mother2.event_id+1) if self.mother2 else 0,
               self.color1,
               self.color2,
               self.px,
               self.py,
               self.pz,
               self.E, 
               self.mass,
               self.vtim,
               self.helicity)
            
    def __eq__(self, other):

        if not isinstance(other, Particle):
            return False        
        if self.pid == other.pid and \
           self.status == other.status and \
           self.mother1 == other.mother1 and \
           self.mother2 == other.mother2 and \
           self.color1 == other.color1 and \
           self.color2 == other.color2 and \
           self.px == other.px and \
           self.py == other.py and \
           self.pz == other.pz and \
           self.E == other.E and \
           self.mass == other.mass and \
           self.vtim == other.vtim and \
           self.helicity == other.helicity:
            return True
        return False
        
    def set_momentum(self, momentum):
        
        self.E = momentum.E
        self.px = momentum.px 
        self.py = momentum.py
        self.pz = momentum.pz

    def add_decay(self, decay_event):
        """associate to this particle the decay in the associate event"""
        
        return self.event.add_decay_to_particle(self.event_id, decay_event)
            
    def __repr__(self):
        return 'Particle("%s", event=%s)' % (str(self), self.event)


class EventFile(object):
    """A class to allow to read both gzip and not gzip file"""
    
    allow_empty_event = False

    def __new__(self, path, mode='r', *args, **opt):
        
        if not path.endswith(".gz"):
            return file.__new__(EventFileNoGzip, path, mode, *args, **opt)
        elif mode == 'r' and not os.path.exists(path) and os.path.exists(path[:-3]):
            return EventFile.__new__(EventFileNoGzip, path[:-3], mode, *args, **opt)
        else:
            try:
                return gzip.GzipFile.__new__(EventFileGzip, path, mode, *args, **opt)
            except IOError, error:
                raise
            except Exception, error:
                if mode == 'r':
                    misc.gunzip(path)
                return file.__new__(EventFileNoGzip, path[:-3], mode, *args, **opt)


    def __init__(self, path, mode='r', *args, **opt):
        """open file and read the banner [if in read mode]"""

        self.to_zip = False
        if path.endswith('.gz') and mode == 'w' and\
                                              isinstance(self, EventFileNoGzip):
            path = path[:-3]
            self.to_zip = True # force to zip the file at close() with misc.gzip
        
        self.parsing = True # check if/when we need to parse the event.
        self.eventgroup  = False
        try:
            super(EventFile, self).__init__(path, mode, *args, **opt)
        except IOError:
            if '.gz' in path and isinstance(self, EventFileNoGzip) and\
                mode == 'r' and os.path.exists(path[:-3]):
                super(EventFile, self).__init__(path[:-3], mode, *args, **opt)
            else:
                raise
                
        self.banner = ''
        if mode == 'r':
            line = ''
            while '</init>' not in line.lower():
                try:
                    line  = super(EventFile, self).next()
                except StopIteration:
                    self.seek(0)
                    self.banner = ''
                    break 
                if "<event" in line.lower():
                    self.seek(0)
                    self.banner = ''
                    break                     

                self.banner += line

    def get_banner(self):
        """return a banner object"""
        import madgraph.various.banner as banner
        if isinstance(self.banner, banner.Banner):
            return self.banner
        
        output = banner.Banner()
        output.read_banner(self.banner)
        return output

    @property
    def cross(self):
        """return the cross-section of the file #from the banner"""
        try:
            return self._cross
        except Exception:
            pass

        onebanner = self.get_banner()
        self._cross = onebanner.get_cross()
        return self._cross
    
    def __len__(self):
        if self.closed:
            return 0
        if hasattr(self,"len"):
            return self.len
        self.seek(0)
        nb_event=0
        with misc.TMP_variable(self, 'parsing', False):
            for _ in self:
                nb_event +=1
        self.len = nb_event
        self.seek(0)
        return self.len

    def next(self):
        """get next event"""

        if not self.eventgroup:
            text = ''
            line = ''
            mode = 0
            while '</event>' not in line:
                line = super(EventFile, self).next()
                if '<event' in line:
                    mode = 1
                    text = ''
                if mode:
                    text += line
            if self.parsing:
                out = Event(text)
                if len(out) == 0  and not self.allow_empty_event:
                    raise Exception
                return out
            else:
                return text
        else:
            events = []
            text = ''
            line = ''
            mode = 0
            while '</eventgroup>' not in line:
                line = super(EventFile, self).next()
                if '<eventgroup' in line:
                    events=[]
                    text = ''
                elif '<event' in line:
                    text=''
                    mode=1
                elif '</event>' in line:
                    if self.parsing:
                        events.append(Event(text))
                    else:
                        events.append(text)
                    text = ''
                    mode = 0
                if mode:
                    text += line  
            if len(events) == 0:
                return self.next()
            return events
    

    def initialize_unweighting(self, get_wgt, trunc_error):
        """ scan once the file to return 
            - the list of the hightest weight (of size trunc_error*NB_EVENT
            - the cross-section by type of process
            - the total number of events in the file
            """
            
        # We need to loop over the event file to get some information about the 
        # new cross-section/ wgt of event.
        self.seek(0)
        all_wgt = []
        cross = collections.defaultdict(int)
        nb_event = 0
        for event in self:
            nb_event +=1
            wgt = get_wgt(event)
            cross['all'] += wgt
            cross['abs'] += abs(wgt)
            cross[event.ievent] += wgt
            all_wgt.append(abs(wgt))
            # avoid all_wgt to be too large
            if nb_event % 20000 == 0:
                all_wgt.sort()
                # drop the lowest weight
                nb_keep = max(20, int(nb_event*trunc_error*15))
                all_wgt = all_wgt[-nb_keep:]

        #final selection of the interesting weight to keep
        all_wgt.sort()
        # drop the lowest weight
        nb_keep = max(20, int(nb_event*trunc_error*10))
        all_wgt = all_wgt[-nb_keep:] 
        self.seek(0)
        return all_wgt, cross, nb_event
    
    def write_events(self, event):
        """ write a single events or a list of event
        if self.eventgroup is ON, then add <eventgroup> around the lists of events
        """
        if isinstance(event, Event):
            if self.eventgroup:
                self.write('<eventgroup>\n%s\n</eventgroup>\n' % event)
            else:
                self.write(str(event))
        elif isinstance(event, list):
            if self.eventgroup:
                self.write('<eventgroup>\n')
            for evt in event:
                self.write(str(evt))
            if self.eventgroup:
                self.write('</eventgroup>\n')
    
    def unweight(self, outputpath, get_wgt=None, max_wgt=0, trunc_error=0, 
                 event_target=0, log_level=logging.INFO, normalization='average'):
        """unweight the current file according to wgt information wgt.
        which can either be a fct of the event or a tag in the rwgt list.
        max_wgt allow to do partial unweighting. 
        trunc_error allow for dynamical partial unweighting
        event_target reweight for that many event with maximal trunc_error.
        (stop to write event when target is reached)
        """
        if not get_wgt:
            def weight(event):
                return event.wgt
            get_wgt  = weight
            unwgt_name = "central weight"
        elif isinstance(get_wgt, str):
            unwgt_name =get_wgt 
            def get_wgt(event):
                event.parse_reweight()
                return event.reweight_data[unwgt_name]
        else:
            unwgt_name = get_wgt.func_name

        # check which weight to write
        if hasattr(self, "written_weight"):
            written_weight = lambda x: math.copysign(self.written_weight,float(x))
        else: 
            written_weight = lambda x: x
                    
        all_wgt, cross, nb_event = self.initialize_unweighting(get_wgt, trunc_error)

        # function that need to be define on the flight
        def max_wgt_for_trunc(trunc):
            """find the weight with the maximal truncation."""
            
            xsum = 0
            i=1 
            while (xsum - all_wgt[-i] * (i-1) <= cross['abs'] * trunc):
                max_wgt = all_wgt[-i]
                xsum += all_wgt[-i]
                i +=1
                if i == len(all_wgt):
                    break

            return max_wgt
        # end of the function
                
        # choose the max_weight
        if not max_wgt:
            if trunc_error == 0 or len(all_wgt)<2 or event_target:
                max_wgt = all_wgt[-1]
            else:
                max_wgt = max_wgt_for_trunc(trunc_error)

        # need to modify the banner so load it to an object
        if self.banner:
            try:
                import internal
            except:
                import madgraph.various.banner as banner_module
            else:
                import internal.banner as banner_module
            if not isinstance(self.banner, banner_module.Banner):
                banner = self.get_banner()
                # 1. modify the cross-section
                banner.modify_init_cross(cross)
                # 3. add information about change in weight
                banner["unweight"] = "unweighted by %s" % unwgt_name
            else:
                banner = self.banner
            # modify the lha strategy
            curr_strategy = banner.get_lha_strategy()
            if normalization in ['unit', 'sum']:
                strategy = 3
            else:
                strategy = 4
            if curr_strategy >0: 
                banner.set_lha_strategy(abs(strategy))
            else:
                banner.set_lha_strategy(-1*abs(strategy))
                
        # Do the reweighting (up to 20 times if we have target_event)
        nb_try = 20
        nb_keep = 0
        for i in range(nb_try):
            self.seek(0)
            if event_target:
                if i==0:
                    max_wgt = max_wgt_for_trunc(0)
                else:
                    #guess the correct max_wgt based on last iteration
                    efficiency = nb_keep/nb_event
                    needed_efficiency = event_target/nb_event
                    last_max_wgt = max_wgt
                    needed_max_wgt = last_max_wgt * efficiency / needed_efficiency
                    
                    min_max_wgt = max_wgt_for_trunc(trunc_error)
                    max_wgt = max(min_max_wgt, needed_max_wgt)
                    max_wgt = min(max_wgt, all_wgt[-1])
                    if max_wgt == last_max_wgt:
                        if nb_keep < event_target and log_level>=10:
                            logger.log(log_level+10,"fail to reach target %s", event_target)
                            break   
                        else:
                            break

            #create output file (here since we are sure that we have to rewrite it)
            if outputpath:
                outfile = EventFile(outputpath, "w")
            # need to write banner information
            # need to see what to do with rwgt information!
            if self.banner and outputpath:
                banner.write(outfile, close_tag=False)

            # scan the file
            nb_keep = 0
            trunc_cross = 0
            for event in self:
                r = random.random()
                wgt = get_wgt(event)
                if abs(wgt) < r * max_wgt:
                    continue
                elif wgt > 0:
                    nb_keep += 1
                    event.wgt = written_weight(max(wgt, max_wgt))
                    if abs(wgt) > max_wgt:
                        trunc_cross += abs(wgt) - max_wgt 
                    if event_target ==0 or nb_keep <= event_target:
                        if outputpath:                         
                            outfile.write(str(event))

                elif wgt < 0:
                    nb_keep += 1
                    event.wgt =     -1* written_weight(max(abs(wgt), max_wgt))
                    if abs(wgt) > max_wgt:
                        trunc_cross += abs(wgt) - max_wgt
                    if outputpath and (event_target ==0 or nb_keep <= event_target):
                        outfile.write(str(event))
            
            if event_target and nb_keep > event_target:
                if not outputpath:
                    #no outputpath define -> wants only the nb of unweighted events
                    continue
                elif event_target and i != nb_try-1 and nb_keep >= event_target *1.05:
                    outfile.write("</LesHouchesEvents>\n")
                    outfile.close()
                    #logger.log(log_level, "Found Too much event %s. Try to reduce truncation" % nb_keep)
                    continue
                else:
                    outfile.write("</LesHouchesEvents>\n")
                    outfile.close()
                break
            elif event_target == 0:
                if outputpath:
                    outfile.write("</LesHouchesEvents>\n")
                    outfile.close()
                break                    
            elif outputpath:
                outfile.write("</LesHouchesEvents>\n")
                outfile.close()
#                logger.log(log_level, "Found only %s event. Reduce max_wgt" % nb_keep)
            
        else:
            # pass here if event_target > 0 and all the attempt fail.
            logger.log(log_level+10,"fail to reach target event %s (iteration=%s)", event_target,i)
        
#        logger.log(log_level, "Final maximum weight used for final "+\
#                    "unweighting is %s yielding %s events." % (max_wgt,nb_keep))
            
        if event_target:
            nb_events_unweighted = nb_keep
            nb_keep = min( event_target, nb_keep)
        else:
            nb_events_unweighted = nb_keep

        logger.log(log_level, "write %i event (efficiency %.2g %%, truncation %.2g %%) after %i iteration(s)", 
          nb_keep, nb_events_unweighted/nb_event*100, trunc_cross/cross['abs']*100, i)
     
        #correct the weight in the file if not the correct number of event
        if nb_keep != event_target and hasattr(self, "written_weight") and strategy !=4:
            written_weight = lambda x: math.copysign(self.written_weight*event_target/nb_keep, float(x))
            startfile = EventFile(outputpath)
            tmpname = pjoin(os.path.dirname(outputpath), "wgtcorrected_"+ os.path.basename(outputpath))
            outfile = EventFile(tmpname, "w")
            outfile.write(startfile.banner)
            for event in startfile:
                event.wgt = written_weight(event.wgt)
                outfile.write(str(event))
            outfile.write("</LesHouchesEvents>\n")
            startfile.close()
            outfile.close()
            shutil.move(tmpname, outputpath)
            
        
        
            
        self.max_wgt = max_wgt
        return nb_keep
    
    def apply_fct_on_event(self, *fcts, **opts):
        """ apply one or more fct on all event. """
        
        opt= {"print_step": 5000, "maxevent":float("inf"),'no_output':False}
        opt.update(opts)
        start = time.time()
        nb_fct = len(fcts)
        out = []
        for i in range(nb_fct):
            out.append([])
        self.seek(0)
        nb_event = 0
        for event in self:
            nb_event += 1
            if opt["print_step"] and (nb_event % opt["print_step"]) == 0:
                if hasattr(self,"len"):
                    print("currently at %s/%s event [%is]" % (nb_event, self.len, time.time()-start))
                else:
                    print("currently at %s event [%is]" % (nb_event, time.time()-start))
            for i in range(nb_fct):
                value = fcts[i](event)
                if not opt['no_output']:
                    out[i].append(value)
            if nb_event > opt['maxevent']:
                break
        if nb_fct == 1:
            return out[0]
        else:
            return out

    def split(self, nb_event=0, partition=None, cwd=os.path.curdir, zip=False):
        """split the file in multiple file. Do not change the weight!"""

        nb_file = -1
        for i, event in enumerate(self):
            if (not (partition is None) and i==sum(partition[:nb_file+1])) or \
                                   (partition is None and i % nb_event == 0):
                if i:
                    #close previous file
                    current.write('</LesHouchesEvent>\n')
                    current.close()
                # create the new file
                nb_file +=1
                # If end of partition then finish writing events here.
                if not partition is None and (nb_file+1>len(partition)):
                    return nb_file
                if zip:
                    current = EventFile(pjoin(cwd,'%s_%s.lhe.gz' % (self.name, nb_file)),'w')
                else:
                    current = open(pjoin(cwd,'%s_%s.lhe' % (self.name, nb_file)),'w')                    
                current.write(self.banner)
            current.write(str(event))
        if i!=0:
            current.write('</LesHouchesEvent>\n')
            current.close()
             
        return nb_file +1

    def update_HwU(self, hwu, fct, name='lhe', keep_wgt=False, maxevents=sys.maxint):
        """take a HwU and add this event file for the function fct"""
                
        if not isinstance(hwu, list):
            hwu = [hwu]

        class HwUUpdater(object):
            
            def __init__(self, fct, keep_wgt):
                
                self.fct = fct
                self.first = True
                self.keep_wgt = keep_wgt
                
            def add(self, event):

                value = self.fct(event)
                # initialise the curve for the first call
                if self.first:
                    for h in hwu:
                        # register the variables
                        if isinstance(value, dict):
                            h.add_line(value.keys())
                        else:
                        
                            h.add_line(name)
                            if self.keep_wgt is True:
                                event.parse_reweight()
                                h.add_line(['%s_%s' % (name, key)
                                                    for key in event.reweight_data])
                            elif self.keep_wgt:
                                h.add_line(self.keep_wgt.values())                            
                    self.first = False
                # Fill the histograms
                for h in hwu:
                    if isinstance(value, tuple):
                        h.addEvent(value[0], value[1])
                    else:
                        h.addEvent(value,{name:event.wgt})
                        if self.keep_wgt:
                            event.parse_reweight()
                            if self.keep_wgt is True:
                                data = dict(('%s_%s' % (name, key),event.reweight_data[key])
                                                    for key in event.reweight_data)
                                h.addEvent(value, data)
                            else:
                                data = dict(( value,event.reweight_data[key])
                                                    for key,value in self.keep_wgt.items())
                                h.addEvent(value, data)
                                
                                      
        
        self.apply_fct_on_event(HwUUpdater(fct,keep_wgt).add, no_output=True,maxevent=maxevents)
        return hwu
    
    def create_syscalc_data(self, out_path, pythia_input=None):
        """take the lhe file and add the matchscale from the pythia_input file"""
        
        if pythia_input:
            def next_data():
                for line in open(pythia_input):
                    if line.startswith('#'):
                        continue
                    data = line.split()
                    print (int(data[0]), data[-3], data[-2], data[-1])
                    yield (int(data[0]), data[-3], data[-2], data[-1])
        else:
            def next_data():
                i=0
                while 1:
                    yield [i,0,0,0]
                    i+=1
        sys_iterator = next_data()
        #ensure that we are at the beginning of the file
        self.seek(0)
        out = open(out_path,'w')
        
        pdf_pattern = re.compile(r'''<init>(.*)</init>''', re.M+re.S)
        init = pdf_pattern.findall(self.banner)[0].split('\n',2)[1]
        id1, id2, _, _, _, _, pdf1,pdf2,_,_ = init.split() 
        id = [int(id1), int(id2)]
        type = []
        for i in range(2):
            if abs(id[i]) == 2212:
                if i > 0:
                    type.append(1)
                else:
                    type.append(-1)
            else:
                type.append(0)           
        pdf = max(int(pdf1),int(pdf2))
        
        out.write("<header>\n" + \
                  "<orgpdf>%i</orgpdf>\n" % pdf + \
                  "<beams>  %s  %s</beams>\n" % tuple(type) + \
                  "</header>\n")
        
        
        nevt, smin, smax, scomp = sys_iterator.next()
        for i, orig_event in enumerate(self):
            if i < nevt:
                continue
            new_event = Event()
            sys = orig_event.parse_syscalc_info()
            new_event.syscalc_data = sys
            if smin:
                new_event.syscalc_data['matchscale'] = "%s %s %s" % (smin, scomp, smax)
            out.write(str(new_event), nevt)
            try:
                nevt, smin, smax, scomp = sys_iterator.next()
            except StopIteration:
                break
            
    def get_alphas(self, scale, lhapdf_config='lhapdf-config'):
        """return the alphas value associated to a given scale"""
        
        if hasattr(self, 'alpsrunner'):
            return self.alpsrunner(scale)
        
        #
        banner = banner_mod.Banner(self.banner)
        run_card = banner.charge_card('run_card')
        use_runner = False
        if abs(run_card['lpp1']) != 1 and abs(run_card['lpp2']) != 1:
            # no pdf use. -> use Runner
            use_runner = True
        else:
            # try to use lhapdf
            lhapdf = misc.import_python_lhapdf(lhapdf_config)
            if not lhapdf:
                logger.warning('fail to link to lhapdf for the alphas-running. Use Two loop computation')
                use_runner = True
            try:
                self.pdf = lhapdf.mkPDF(int(self.banner.run_card.get_lhapdf_id()))
            except Exception:
                logger.warning('fail to link to lhapdf for the alphas-running. Use Two loop computation')
                use_runner = True
                
        if not use_runner:
            self.alpsrunner = lambda scale: self.pdf.alphasQ(scale)
        else:
            try:
                from models.model_reader import Alphas_Runner
            except ImportError:
                root = os.path.dirname(__file__)
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
                
            if not hasattr(banner, 'param_card'):
                param_card = banner.charge_card('param_card')
            else:
                param_card = banner.param_card
            
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
            
            
            
        return self.alpsrunner(scale)
            
            
            
        
        
        
    
    
class EventFileGzip(EventFile, gzip.GzipFile):
    """A way to read/write a gzipped lhef event"""
    
    
    def tell(self):
        currpos = super(EventFileGzip, self).tell()
        if not currpos:
            currpos = self.size
        return currpos
        
class EventFileNoGzip(EventFile, file):
    """A way to read a standard event file"""
    
    def close(self,*args, **opts):
        
        out = super(EventFileNoGzip, self).close(*args, **opts)
        if self.to_zip:
            misc.gzip(self.name)
    
class MultiEventFile(EventFile):
    """a class to read simultaneously multiple file and read them in mixing them.
       Unweighting can be done at the same time. 
       The number of events in each file need to be provide in advance 
       (if not provide the file is first read to find that number"""
    
    def __new__(cls, start_list=[],parse=True):
        return object.__new__(MultiEventFile)
    
    def __init__(self, start_list=[], parse=True):
        """if trunc_error is define here then this allow
        to only read all the files twice and not three times."""
        self.eventgroup = False
        self.files = []
        self.parsefile = parse #if self.files is formatted or just the path
        self.banner = ''
        self.initial_nb_events = []
        self.total_event_in_files = 0
        self.curr_nb_events = []
        self.allcross = []
        self.error = []
        self.across = []
        self.scales = []
        if start_list:
            if parse:
                for p in start_list:
                    self.add(p)
            else:
                self.files = start_list
        self._configure = False
        
    def close(self,*args,**opts):
        for f in self.files:
            f.close(*args, **opts)
        
    def add(self, path, cross, error, across, nb_event=0, scale=1):
        """ add a file to the pool, across allow to reweight the sum of weight 
        in the file to the given cross-section 
        """
        
        if across == 0:
            # No event linked to this channel -> so no need to include it
            return 
        
        obj = EventFile(path)
        obj.eventgroup = self.eventgroup 
        if len(self.files) == 0 and not self.banner:
            self.banner = obj.banner
        self.curr_nb_events.append(0)
        self.initial_nb_events.append(0)
        self.allcross.append(cross)
        self.across.append(across)
        self.error.append(error)
        self.scales.append(scale)
        self.files.append(obj)
        if nb_event:
            obj.len = nb_event
        self._configure = False
        return obj
        
    def __iter__(self):
        return self
    
    def next(self):

        if not self._configure:
            self.configure()

        remaining_event = self.total_event_in_files - sum(self.curr_nb_events)
        if remaining_event == 0:
            raise StopIteration
        # determine which file need to be read
        nb_event = random.randint(1, remaining_event)
        sum_nb=0
        for i, obj in enumerate(self.files):
            sum_nb += self.initial_nb_events[i] - self.curr_nb_events[i]
            if nb_event <= sum_nb:
                self.curr_nb_events[i] += 1
                event = obj.next()
                if not self.eventgroup:
                    event.sample_scale = self.scales[i] # for file reweighting
                else:
                    for evt in event:
                        evt.sample_scale = self.scales[i]
                return event
        else:
            raise Exception
    

    def define_init_banner(self, wgt, lha_strategy, proc_charac=None):
        """define the part of the init_banner"""
        
        if not self.banner:
            return
        
        # compute the cross-section of each splitted channel
        grouped_cross = {}
        grouped_error = {}
        for i,ff in enumerate(self.files):
            filename = ff.name
            from_init = False
            Pdir = [P for P in filename.split(os.path.sep) if P.startswith('P')]
            if Pdir:
                Pdir = Pdir[-1]
                group = Pdir.split("_")[0][1:]
                if not group.isdigit():
                    from_init = True  
            else:
                from_init = True

            if not from_init:
                if group in grouped_cross:
                    grouped_cross[group] += self.allcross[i]
                    grouped_error[group] += self.error[i]**2 
                else:
                    grouped_cross[group] = self.allcross[i]
                    grouped_error[group] = self.error[i]**2
            else:
                ban = banner_mod.Banner(ff.banner)
                for line in  ban['init'].split('\n'):
                    splitline = line.split()
                    if len(splitline)==4:
                        cross, error, _, group = splitline
                        if int(group) in grouped_cross:
                            grouped_cross[group] += float(cross)
                            grouped_error[group] += float(error)**2                        
                        else:
                            grouped_cross[group] = float(cross)
                            grouped_error[group] = float(error)**2                             
        nb_group = len(grouped_cross)
        
        # compute the information for the first line 
        try:
            run_card = self.banner.run_card
        except:
            run_card = self.banner.charge_card("run_card")
            
        init_information = run_card.get_banner_init_information()
        #correct for special case
        if proc_charac and proc_charac['ninitial'] == 1:
            #special case for 1>N
            init_information = run_card.get_banner_init_information()
            event = self.next()
            init_information["idbmup1"] = event[0].pdg
            init_information["ebmup1"] = event[0].mass
            init_information["idbmup2"] = 0 
            init_information["ebmup2"] = 0
            self.seek(0)
        else:
            # check special case without PDF for one (or both) beam
            if init_information["idbmup1"] == 0:
                event = self.next()
                init_information["idbmup1"]= event[0].pdg
                if init_information["idbmup2"] == 0:
                    init_information["idbmup2"]= event[1].pdg
                self.seek(0)
            if init_information["idbmup2"] == 0:
                event = self.next()
                init_information["idbmup2"] = event[1].pdg
                self.seek(0)
        
        init_information["nprup"] = nb_group
        
        if run_card["lhe_version"] < 3:
            init_information["generator_info"] = ""
        else:
            init_information["generator_info"] = "<generator name='MadGraph5_aMC@NLO' version='%s'>please cite 1405.0301 </generator>\n" \
                % misc.get_pkg_info()['version']
        
        # cross_information:
        cross_info = "%(cross)e %(error)e %(wgt)e %(id)i"
        init_information["cross_info"] = []
        for id in grouped_cross:
            conv = {"id": int(id), "cross": grouped_cross[id], "error": math.sqrt(grouped_error[id]),
                    "wgt": wgt}
            init_information["cross_info"].append( cross_info % conv)
        init_information["cross_info"] = '\n'.join(init_information["cross_info"])
        init_information['lha_stra'] = -1 * abs(lha_strategy)
        
        template_init =\
        """    %(idbmup1)i %(idbmup2)i %(ebmup1)e %(ebmup2)e %(pdfgup1)i %(pdfgup2)i %(pdfsup1)i %(pdfsup2)i %(lha_stra)i %(nprup)i
%(cross_info)s
%(generator_info)s
"""
        
        self.banner["init"] = template_init % init_information
        
            
    
    def initialize_unweighting(self, getwgt, trunc_error):
        """ scan once the file to return 
            - the list of the hightest weight (of size trunc_error*NB_EVENT
            - the cross-section by type of process
            - the total number of events in the files
            In top of that it initialise the information for the next routine
            to determine how to choose which file to read 
            """
        self.seek(0)
        all_wgt = []
        total_event = 0
        sum_cross = collections.defaultdict(int)
        for i,f in enumerate(self.files):
            nb_event = 0 
            # We need to loop over the event file to get some information about the 
            # new cross-section/ wgt of event.
            cross = collections.defaultdict(int)
            new_wgt =[] 
            for event in f:
                nb_event += 1
                total_event += 1
                event.sample_scale = 1
                wgt = getwgt(event)
                cross['all'] += wgt
                cross['abs'] += abs(wgt)
                cross[event.ievent] += wgt
                new_wgt.append(abs(wgt))
                # avoid all_wgt to be too large
                if nb_event % 20000 == 0:
                    new_wgt.sort()
                    # drop the lowest weight
                    nb_keep = max(20, int(nb_event*trunc_error*15))
                    new_wgt = new_wgt[-nb_keep:]
            if nb_event == 0:
                raise Exception
            # store the information
            self.initial_nb_events[i] = nb_event
            self.scales[i] = self.across[i]/cross['abs'] if self.across[i] else 1
            #misc.sprint("sum of wgt in event %s is %s. Should be %s => scale %s (nb_event: %s)"
            #            % (i, cross['all'], self.allcross[i], self.scales[i], nb_event))
            for key in cross:
                sum_cross[key] += cross[key]* self.scales[i]
            all_wgt +=[self.scales[i] * w for w in new_wgt]
            all_wgt.sort()
            nb_keep = max(20, int(total_event*trunc_error*10))
            all_wgt = all_wgt[-nb_keep:] 
            
        self.total_event_in_files = total_event
        #final selection of the interesting weight to keep
        all_wgt.sort()
        # drop the lowest weight
        nb_keep = max(20, int(total_event*trunc_error*10))
        all_wgt = all_wgt[-nb_keep:]  
        self.seek(0)
        self._configure = True
        return all_wgt, sum_cross, total_event
    
    def configure(self):
        
        self._configure = True
        for i,f in enumerate(self.files):
            self.initial_nb_events[i] = len(f)
        self.total_event_in_files = sum(self.initial_nb_events)
    
    def __len__(self):
        
        return len(self.files)
    
    def seek(self, pos):
        """ """
        
        if pos !=0:
            raise Exception
        for i in range(len(self)):
            self.curr_nb_events[i] = 0         
        for f in self.files:
            f.seek(pos)
            
    def unweight(self, outputpath, get_wgt, **opts):
        """unweight the current file according to wgt information wgt.
        which can either be a fct of the event or a tag in the rwgt list.
        max_wgt allow to do partial unweighting. 
        trunc_error allow for dynamical partial unweighting
        event_target reweight for that many event with maximal trunc_error.
        (stop to write event when target is reached)
        """

        if isinstance(get_wgt, str):
            unwgt_name =get_wgt 
            def get_wgt_multi(event):
                event.parse_reweight()
                return event.reweight_data[unwgt_name] * event.sample_scale
        else:
            unwgt_name = get_wgt.func_name
            get_wgt_multi = lambda event: get_wgt(event) * event.sample_scale
        #define the weighting such that we have built-in the scaling

        if 'proc_charac' in opts:
            if opts['proc_charac']:
                proc_charac = opts['proc_charac']
            else:
                proc_charac=None
            del opts['proc_charac']
        else:
            proc_charac = None

        if 'event_target' in opts and opts['event_target']:
            if 'normalization' in opts:
                if opts['normalization'] == 'sum':
                    new_wgt = sum(self.across)/opts['event_target']
                    strategy = 3
                elif opts['normalization'] == 'average':
                    strategy = 4
                    new_wgt = sum(self.across)                    
                elif opts['normalization'] == 'unit':
                    strategy =3
                    new_wgt = 1.
            else:
                strategy = 4
                new_wgt = sum(self.across)
            self.define_init_banner(new_wgt, strategy, proc_charac=proc_charac)
            self.written_weight = new_wgt
        elif 'write_init' in opts and opts['write_init']:
            self.define_init_banner(0,0, proc_charac=proc_charac)
            del opts['write_init']
        return super(MultiEventFile, self).unweight(outputpath, get_wgt_multi, **opts)

    def write(self, path, random=False, banner=None, get_info=False):
        """ """
        
        if isinstance(path, str):
            out = EventFile(path, 'w')
            if self.parsefile and not banner:    
                banner = self.files[0].banner
            elif not banner:
                firstlhe = EventFile(self.files[0])
                banner = firstlhe.banner                
        else: 
            out = path
        if banner:
            out.write(banner)
        nb_event = 0
        info = collections.defaultdict(float)
        if random and self.open:
            for event in self:
                nb_event +=1
                out.write(event)
                if get_info:
                    event.parse_reweight()
                    for key, value in event.reweight_data.items():
                        info[key] += value
                    info['central'] += event.wgt
        elif not random:
            for i,f in enumerate(self.files):
                #check if we need to parse the file or not
                if not self.parsefile:
                    if i==0:
                        try:
                            lhe = firstlhe
                        except:
                            lhe = EventFile(f)
                    else:
                        lhe = EventFile(f)
                else:
                    lhe = f
                for event in lhe:
                    nb_event +=1
                    if get_info:
                        event.parse_reweight()
                        for key, value in event.reweight_data.items():
                            info[key] += value
                        info['central'] += event.wgt
                    out.write(str(event))
                lhe.close()
        out.write("</LesHouchesEvents>\n") 
        return nb_event, info
                            
    def remove(self):
        """ """
        if self.parsefile:
            for f in self.files:
                os.remove(f.name)
        else:
            for f in self.files:
                os.remove(f)
            
        
           
class Event(list):
    """Class storing a single event information (list of particles + global information)"""

    warning_order = True # raise a warning if the order of the particle are not in accordance of child/mother

    def __init__(self, text=None):
        """The initialization of an empty Event (or one associate to a text file)"""
        list.__init__(self)
        
        # First line information
        self.nexternal = 0
        self.ievent = 0
        self.wgt = 0
        self.aqcd = 0 
        self.scale = 0
        self.aqed = 0
        self.aqcd = 0
        # Weight information
        self.tag = ''
        self.eventflag = {} # for information in <event > 
        self.comment = ''
        self.reweight_data = {}
        self.matched_scale_data = None
        self.syscalc_data = {}
        if text:
            self.parse(text)


            
    def parse(self, text):
        """Take the input file and create the structured information"""
        #text = re.sub(r'</?event>', '', text) # remove pointless tag
        status = 'first' 
        for line in text.split('\n'):
            line = line.strip()
            if not line: 
                continue
            elif line[0] == '#':
                self.comment += '%s\n' % line
                continue
            elif line.startswith('<event'):
                if '=' in line:
                    found = re.findall(r"""(\w*)=(?:(?:['"])([^'"]*)(?=['"])|(\S*))""",line)
                    #for '<event line=4 value=\'3\' error="5" test=" 1 and 2">\n'
                    #return [('line', '', '4'), ('value', '3', ''), ('error', '5', ''), ('test', ' 1 and 2', '')]
                    self.eventflag = dict((n, a1) if a1 else (n,a2) for n,a1,a2 in found)
                    # return {'test': ' 1 and 2', 'line': '4', 'value': '3', 'error': '5'}
                continue
            
            elif 'first' == status:
                if '<rwgt>' in line:
                    status = 'tag'
                else:
                    self.assign_scale_line(line)
                    status = 'part' 
                    continue
            if '<' in line:
                status = 'tag'
                
            if 'part' == status:
                part = Particle(line, event=self)
                if part.E != 0:
                    self.append(part)
                elif self.nexternal:
                        self.nexternal-=1
            else:
                if '</event>' in line:
                    line = line.replace('</event>','',1)
                self.tag += '%s\n' % line
                
        self.assign_mother()
        
    def assign_mother(self):
        """convert the number in actual particle"""
        #Security if not incoming particle. Define a fake particle and set all particle as 
        # decaying from that fake particle
        if all(p.status != -1 for p in self):
            if self.warning_order:
                Event.warning_order = False
                logger.warning("Weird format for lhe format: no incoming particle... adding a fake one")
            
            mother = Particle(event=self)
            mother.status = -1
            mother.pid = 0
            self.insert(0,mother)
            mother.color2 = 0
            mother.event_id = 0
            self.nexternal += 1
            for p in self[1:]:
                p.mother1 = 1
                p.mother2 = 1
                p.event_id += 1
            
        
        # assign the mother:
        for i,particle in enumerate(self):
            if i < particle.mother1 or i < particle.mother2:
                if self.warning_order:
                    logger.warning("Order of particle in the event did not agree with parent/child order. This might be problematic for some code.")
                    Event.warning_order = False
                self.reorder_mother_child()
                return self.assign_mother()
                                   
            if particle.mother1:
                try:
                    particle.mother1 = self[int(particle.mother1) -1]
                except Exception:
                    logger.warning("WRONG MOTHER INFO %s", self)
                    particle.mother1 = 0
            if particle.mother2:
                try:
                    particle.mother2 = self[int(particle.mother2) -1]
                except Exception:
                    logger.warning("WRONG MOTHER INFO %s", self)
                    particle.mother2 = 0

    def rescale_weights(self, ratio):
        """change all the weights by a given ratio"""
        
        self.wgt *= ratio
        self.parse_reweight()
        for key in self.reweight_data:
            self.reweight_data[key] *= ratio
        return self.wgt
    
    def reorder_mother_child(self):
        """check and correct the mother/child position.
           only correct one order by call (but this is a recursive call)"""
    
        tomove, position = None, None
        for i,particle in enumerate(self):
            if i < particle.mother1:
                # move i after particle.mother1
                tomove, position = i, particle.mother1-1
                break
            if i < particle.mother2:
                tomove, position = i, particle.mother2-1
        
        # nothing to change -> we are done      
        if not tomove:
            return
   
        # move the particles:
        particle = self.pop(tomove)
        self.insert(int(position), particle)
        
        #change the mother id/ event_id in the event.
        for i, particle in enumerate(self):
            particle.event_id = i
            #misc.sprint( i, particle.event_id)
            m1, m2 = particle.mother1, particle.mother2
            if m1 == tomove +1:
                particle.mother1 = position+1
            elif tomove < m1 <= position +1:
                particle.mother1 -= 1
            if m2 == tomove +1:
                particle.mother2 = position+1
            elif tomove < m2 <= position +1:
                particle.mother2 -= 1  
        # re-call the function for the next potential change   
        return self.reorder_mother_child()
         
        
        
        
        
   
    def parse_reweight(self):
        """Parse the re-weight information in order to return a dictionary
           {key: value}. If no group is define group should be '' """
        if self.reweight_data:
            return self.reweight_data
        self.reweight_data = {}
        self.reweight_order = []
        start, stop = self.tag.find('<rwgt>'), self.tag.find('</rwgt>')
        if start != -1 != stop :
            pattern = re.compile(r'''<\s*wgt id=(?:\'|\")(?P<id>[^\'\"]+)(?:\'|\")\s*>\s*(?P<val>[\ded+-.]*)\s*</wgt>''',re.I)
            data = pattern.findall(self.tag[start:stop])
            try:
                self.reweight_data = dict([(pid, float(value)) for (pid, value) in data
                                           if not self.reweight_order.append(pid)])
                                      # the if is to create the order file on the flight
            except ValueError, error:
                raise Exception, 'Event File has unvalid weight. %s' % error
            self.tag = self.tag[:start] + self.tag[stop+7:]
        return self.reweight_data
    
    def parse_nlo_weight(self, real_type=(1,11), threshold=None):
        """ """
        if hasattr(self, 'nloweight'):
            return self.nloweight
        
        start, stop = self.tag.find('<mgrwgt>'), self.tag.find('</mgrwgt>')
        if start != -1 != stop :
        
            text = self.tag[start+8:stop]
            self.nloweight = NLO_PARTIALWEIGHT(text, self, real_type=real_type,
                                               threshold=threshold)
            return self.nloweight

    def rewrite_nlo_weight(self, wgt=None):
        """get the string associate to the weight"""
        
        text="""<mgrwgt>
        %(total_wgt).10e %(nb_wgt)i %(nb_event)i 0
        %(event)s
        %(wgt)s
        </mgrwgt>"""
        
        
        if not wgt:
            if not hasattr(self, 'nloweight'):
                return
            wgt = self.nloweight
            
        data = {'total_wgt': wgt.total_wgt, #need to check name and meaning,
                'nb_wgt': wgt.nb_wgt,
                'nb_event': wgt.nb_event,
                'event': '\n'.join(p.__str__(mode='fortran') for p in wgt.momenta),
                'wgt':'\n'.join(w.__str__(mode='formatted') 
                                         for e in wgt.cevents for w in e.wgts)}
         
        data['total_wgt'] = sum([w.ref_wgt for e in wgt.cevents for w in e.wgts])
        start, stop = self.tag.find('<mgrwgt>'), self.tag.find('</mgrwgt>')
        
        self.tag = self.tag[:start] + text % data + self.tag[stop+9:]
        
            
    def parse_lo_weight(self):
        """ """
        
        
        if hasattr(self, 'loweight'):
            return self.loweight
        
        if not hasattr(Event, 'loweight_pattern'):
            Event.loweight_pattern = re.compile('''<rscale>\s*(?P<nqcd>\d+)\s+(?P<ren_scale>[\d.e+-]+)\s*</rscale>\s*\n\s*
                                    <asrwt>\s*(?P<asrwt>[\s\d.+-e]+)\s*</asrwt>\s*\n\s*
                                    <pdfrwt\s+beam=["']?1["']?\>\s*(?P<beam1>[\s\d.e+-]*)\s*</pdfrwt>\s*\n\s*
                                    <pdfrwt\s+beam=["']?2["']?\>\s*(?P<beam2>[\s\d.e+-]*)\s*</pdfrwt>\s*\n\s*
                                    <totfact>\s*(?P<totfact>[\d.e+-]*)\s*</totfact>
            ''',re.X+re.I+re.M)
        
        start, stop = self.tag.find('<mgrwt>'), self.tag.find('</mgrwt>')
        
        if start != -1 != stop :
            text = self.tag[start+8:stop]
            
            info = Event.loweight_pattern.search(text)
            if not info:
                raise Exception, '%s not parsed'% text
            self.loweight={}
            self.loweight['n_qcd'] = int(info.group('nqcd'))
            self.loweight['ren_scale'] = float(info.group('ren_scale'))
            self.loweight['asrwt'] =[float(x) for x in info.group('asrwt').split()[1:]]
            self.loweight['tot_fact'] = float(info.group('totfact'))
            
            args = info.group('beam1').split()
            npdf = int(args[0])
            self.loweight['n_pdfrw1'] = npdf
            self.loweight['pdf_pdg_code1'] = [int(i) for i in args[1:1+npdf]]
            self.loweight['pdf_x1'] = [float(i) for i in args[1+npdf:1+2*npdf]]
            self.loweight['pdf_q1'] = [float(i) for i in args[1+2*npdf:1+3*npdf]]
            args = info.group('beam2').split()
            npdf = int(args[0])
            self.loweight['n_pdfrw2'] = npdf
            self.loweight['pdf_pdg_code2'] = [int(i) for i in args[1:1+npdf]]
            self.loweight['pdf_x2'] = [float(i) for i in args[1+npdf:1+2*npdf]]
            self.loweight['pdf_q2'] = [float(i) for i in args[1+2*npdf:1+3*npdf]]            
            
        else:
            return None
        return self.loweight            
    
            
    def parse_matching_scale(self):
        """Parse the line containing the starting scale for the shower"""
        
        if self.matched_scale_data is not None:
            return self.matched_scale_data
            
        self.matched_scale_data = []
        

        pattern  = re.compile("<scales\s|</scales>")
        data = re.split(pattern,self.tag)
        if len(data) == 1:
            return []
        else:
            tmp = {}
            start,content, end = data
            self.tag = "%s%s" % (start, end)
            pattern = re.compile("pt_clust_(\d*)=\"([\de+-.]*)\"")
            for id,value in pattern.findall(content):
                tmp[int(id)] = float(value)
            for i in range(1, len(self)+1):
                if i in tmp:
                    self.matched_scale_data.append(tmp[i])
                else:
                    self.matched_scale_data.append(-1)
        return self.matched_scale_data
            
    def parse_syscalc_info(self):
        """ parse the flag for syscalc between <mgrwt></mgrwt>
        <mgrwt>
<rscale>  3 0.26552898E+03</rscale>
<asrwt>0</asrwt>
<pdfrwt beam="1">  1       21 0.14527945E+00 0.26552898E+03</pdfrwt>
<pdfrwt beam="2">  1       21 0.15249110E-01 0.26552898E+03</pdfrwt>
<totfact> 0.10344054E+04</totfact>
</mgrwt>
        """
        if self.syscalc_data:
            return self.syscalc_data
        
        pattern  = re.compile("<mgrwt>|</mgrwt>")
        pattern2 = re.compile("<(?P<tag>[\w]*)(?:\s*(\w*)=[\"'](.*)[\"']\s*|\s*)>(.*)</(?P=tag)>")
        data = re.split(pattern,self.tag)
        if len(data) == 1:
            return []
        else:
            tmp = {}
            start,content, end = data
            self.tag = "%s%s" % (start, end)
            for tag, key, keyval, tagval in pattern2.findall(content):
                if key:
                    self.syscalc_data[(tag, key, keyval)] = tagval
                else:
                    self.syscalc_data[tag] = tagval
            return self.syscalc_data


    def add_decay_to_particle(self, position, decay_event):
        """define the decay of the particle id by the event pass in argument"""
        
        this_particle = self[position]
        #change the status to internal particle
        this_particle.status = 2
        this_particle.helicity = 0
        
        # some usefull information
        decay_particle = decay_event[0]
        this_4mom = FourMomentum(this_particle)
        nb_part = len(self) #original number of particle
        
        thres = decay_particle.E*1e-10
        assert max(decay_particle.px, decay_particle.py, decay_particle.pz) < thres,\
            "not on rest particle %s %s %s %s" % (decay_particle.E, decay_particle.px,decay_particle.py,decay_particle.pz) 
        
        self.nexternal += decay_event.nexternal -1
        old_scales = list(self.parse_matching_scale())
        if old_scales:
            jet_position = sum(1 for i in range(position) if self[i].status==1)
            initial_pos = sum(1 for i in range(position) if self[i].status==-1)
            self.matched_scale_data.pop(initial_pos+jet_position)
        # add the particle with only handling the 4-momenta/mother
        # color information will be corrected later.
        for particle in decay_event[1:]:
            # duplicate particle to avoid border effect
            new_particle = Particle(particle, self)
            new_particle.event_id = len(self)
            self.append(new_particle)
            if old_scales:
                self.matched_scale_data.append(old_scales[initial_pos+jet_position])
            # compute and assign the new four_momenta
            new_momentum = this_4mom.boost(FourMomentum(new_particle))
            new_particle.set_momentum(new_momentum)
            # compute the new mother
            for tag in ['mother1', 'mother2']:
                mother = getattr(particle, tag)
                if isinstance(mother, Particle):
                    mother_id = getattr(particle, tag).event_id
                    if mother_id == 0:
                        setattr(new_particle, tag, this_particle)
                    else:
                        try:
                            setattr(new_particle, tag, self[nb_part + mother_id -1])
                        except Exception, error:
                            print error
                            misc.sprint( self)
                            misc.sprint(nb_part + mother_id -1)
                            misc.sprint(tag)
                            misc.sprint(position, decay_event)
                            misc.sprint(particle)
                            misc.sprint(len(self), nb_part + mother_id -1)
                            raise
                elif tag == "mother2" and isinstance(particle.mother1, Particle):
                    new_particle.mother2 = this_particle
                else:
                    raise Exception, "Something weird happens. Please report it for investigation"
        # Need to correct the color information of the particle
        # first find the first available color index
        max_color=501
        for particle in self[:nb_part]:
            max_color=max(max_color, particle.color1, particle.color2)
        
        # define a color mapping and assign it:
        color_mapping = {}
        color_mapping[decay_particle.color1] = this_particle.color1
        color_mapping[decay_particle.color2] = this_particle.color2
        for particle in self[nb_part:]:
            if particle.color1:
                if particle.color1 not in color_mapping:
                    max_color +=1
                    color_mapping[particle.color1] = max_color
                    particle.color1 = max_color
                else:
                    particle.color1 = color_mapping[particle.color1]
            if particle.color2:
                if particle.color2 not in color_mapping:
                    max_color +=1
                    color_mapping[particle.color2] = max_color
                    particle.color2 = max_color
                else:
                    particle.color2 = color_mapping[particle.color2]                

    def add_decays(self, pdg_to_decay):
        """use auto-recursion"""

        pdg_to_decay = dict(pdg_to_decay)

        for i,particle in enumerate(self):
            if particle.status != 1:
                continue
            if particle.pdg in pdg_to_decay and pdg_to_decay[particle.pdg]:
                one_decay = pdg_to_decay[particle.pdg].pop()
                self.add_decay_to_particle(i, one_decay)
                return self.add_decays(pdg_to_decay)
        return self
                


    def remove_decay(self, pdg_code=0, event_id=None):
        
        to_remove = []
        if event_id is not None:
            to_remove.append(self[event_id])
    
        if pdg_code:
            for particle in self:
                if particle.pid == pdg_code:
                    to_remove.append(particle) 
                    
        new_event = Event()
        # copy first line information + ...
        for tag in ['nexternal', 'ievent', 'wgt', 'aqcd', 'scale', 'aqed','tag','comment']:
            setattr(new_event, tag, getattr(self, tag))
        
        for particle in self:
            if isinstance(particle.mother1, Particle) and particle.mother1 in to_remove:
                to_remove.append(particle)
                if particle.status == 1:
                    new_event.nexternal -= 1
                continue
            elif isinstance(particle.mother2, Particle) and particle.mother2 in to_remove:
                to_remove.append(particle)
                if particle.status == 1:
                    new_event.nexternal -= 1
                continue
            else:
                new_event.append(Particle(particle))
                
        #ensure that the event_id is correct for all_particle
        # and put the status to 1 for removed particle
        for pos, particle in enumerate(new_event):
            particle.event_id = pos
            if particle in to_remove:
                particle.status = 1
        return new_event

    def get_decay(self, pdg_code=0, event_id=None):
        
        to_start = []
        if event_id is not None:
            to_start.append(self[event_id])
    
        elif pdg_code:
            for particle in self:
                if particle.pid == pdg_code:
                    to_start.append(particle)
                    break 

        new_event = Event()
        # copy first line information + ...
        for tag in ['ievent', 'wgt', 'aqcd', 'scale', 'aqed','tag','comment']:
            setattr(new_event, tag, getattr(self, tag))
        
        # Add the decaying particle
        old2new = {}            
        new_decay_part = Particle(to_start[0])
        new_decay_part.mother1 = None
        new_decay_part.mother2 = None
        new_decay_part.status =  -1
        old2new[new_decay_part.event_id] = len(old2new) 
        new_event.append(new_decay_part)
        
        
        # add the other particle   
        for particle in self:
            if isinstance(particle.mother1, Particle) and particle.mother1.event_id in old2new\
            or isinstance(particle.mother2, Particle) and particle.mother2.event_id in old2new:
                old2new[particle.event_id] = len(old2new) 
                new_event.append(Particle(particle))

        #ensure that the event_id is correct for all_particle
        # and correct the mother1/mother2 by the new reference
        nexternal = 0
        for pos, particle in enumerate(new_event):
            particle.event_id = pos
            if particle.mother1:
                particle.mother1 = new_event[old2new[particle.mother1.event_id]]
            if particle.mother2:
                particle.mother2 = new_event[old2new[particle.mother2.event_id]]
            if particle.status in [-1,1]:
                nexternal +=1
        new_event.nexternal = nexternal
        
        return new_event

            
    def check(self):
        """check various property of the events"""
        
        #1. Check that the 4-momenta are conserved
        E, px, py, pz = 0,0,0,0
        absE, abspx, abspy, abspz = 0,0,0,0
        for particle in self:
            coeff = 1
            if particle.status == -1:
                coeff = -1
            elif particle.status != 1:
                continue
            E += coeff * particle.E
            absE += abs(particle.E)
            px += coeff * particle.px
            py += coeff * particle.py
            pz += coeff * particle.pz
            abspx += abs(particle.px)
            abspy += abs(particle.py)
            abspz += abs(particle.pz)
        # check that relative error is under control
        threshold = 5e-7
        if E/absE > threshold:
            logger.critical(self)
            raise Exception, "Do not conserve Energy %s, %s" % (E/absE, E)
        if px/abspx > threshold:
            logger.critical(self)
            raise Exception, "Do not conserve Px %s, %s" % (px/abspx, px)         
        if py/abspy > threshold:
            logger.critical(self)
            raise Exception, "Do not conserve Py %s, %s" % (py/abspy, py)
        if pz/abspz > threshold:
            logger.critical(self)
            raise Exception, "Do not conserve Pz %s, %s" % (pz/abspz, pz)
            
        #2. check the color of the event
        self.check_color_structure()            
         
    def assign_scale_line(self, line):
        """read the line corresponding to global event line
        format of the line is:
        Nexternal IEVENT WEIGHT SCALE AEW AS
        """
        inputs = line.split()
        assert len(inputs) == 6
        self.nexternal=int(inputs[0])
        self.ievent=int(inputs[1])
        self.wgt=float(inputs[2])
        self.scale=float(inputs[3])
        self.aqed=float(inputs[4])
        self.aqcd=float(inputs[5])
        
    def get_tag_and_order(self):
        """Return the unique tag identifying the SubProcesses for the generation.
        Usefull for program like MadSpin and Reweight module."""
        
        initial, final, order = [], [], [[], []]
        for particle in self:
            if particle.status == -1:
                initial.append(particle.pid)
                order[0].append(particle.pid)
            elif particle.status == 1: 
                final.append(particle.pid)
                order[1].append(particle.pid)
        initial.sort(), final.sort()
        tag = (tuple(initial), tuple(final))
        return tag, order
    
    @staticmethod
    def mass_shuffle(momenta, sqrts, new_mass, new_sqrts=None):
        """use the RAMBO method to shuffle the PS. initial sqrts is preserved."""
        
        if not new_sqrts:
            new_sqrts = sqrts
        
        oldm = [p.mass_sqr for p in momenta]
        newm = [m**2 for m in new_mass]
        tot_mom = sum(momenta, FourMomentum())
        if tot_mom.pt2 > 1e-5:
            boost_back = FourMomentum(tot_mom.mass,0,0,0).boost_to_restframe(tot_mom)
            for i,m in enumerate(momenta):
                momenta[i] = m.boost_to_restframe(tot_mom)
        
        # this is the equation 4.3 of RAMBO paper        
        f = lambda chi: new_sqrts - sum(math.sqrt(max(0, M + chi**2*(p.E**2-m))) 
                                    for M,p,m in zip(newm, momenta,oldm))
        # this is the derivation of the function
        df = lambda chi: -1* sum(chi*(p.E**2-m)/math.sqrt(max(0,(p.E**2-m)*chi**2+M))
            for M,p,m in zip(newm, momenta,oldm))
        
        if sum(new_mass) > new_sqrts:
            return momenta, 0
        try:
            chi = misc.newtonmethod(f, df, 1.0, error=1e-7,maxiter=1000)
        except:
            return momenta, 0 
        # create the new set of momenta # eq. (4.2)        
        new_momenta = []
        for i,p in enumerate(momenta):
            new_momenta.append(
                FourMomentum(math.sqrt(newm[i]+chi**2*(p.E**2-oldm[i])),
                              chi*p.px, chi*p.py, chi*p.pz))
        
        #if __debug__:
        #    for i,p in enumerate(new_momenta):
        #        misc.sprint(p.mass_sqr, new_mass[i]**2, i,p, momenta[i])
        #        assert p.mass_sqr == new_mass[i]**2
                
        # compute the jacobian factor (eq. 4.9)
        jac = chi**(3*len(momenta)-3)
        jac *= reduce(operator.mul,[p.E/k.E for p,k in zip(momenta, new_momenta)],1)
        jac *= sum(p.norm_sq/p.E for p in momenta)
        jac /= sum(k.norm_sq/k.E for k in new_momenta)
        
        # boost back the events in the lab-frame
        if tot_mom.pt2 > 1e-5:
            for i,m in enumerate(new_momenta):
                new_momenta[i] = m.boost_to_restframe(boost_back)
        return new_momenta, jac
        
        
    
    
    def change_ext_mass(self, new_param_card):
        """routine to rescale the mass via RAMBO method. no internal mass preserve.
           sqrts is preserve (RAMBO algo)
        """
        
        old_momenta = []
        new_masses = []
        change_mass = False # check if we need to change the mass
        for part in self:
            if part.status == 1:
                old_momenta.append(FourMomentum(part))
                new_masses.append(new_param_card.get_value('mass', abs(part.pid)))
                if not misc.equal(part.mass, new_masses[-1], 4, zero_limit=10):
                    change_mass = True
        
        if not change_mass:
            return 1
        
        sqrts = self.sqrts

        # apply the RAMBO algo
        new_mom, jac = self.mass_shuffle(old_momenta, sqrts, new_masses)
        
        #modify the momenta of the particles:
        ind =0
        for part in self:
            if part.status==1:
                part.E, part.px, part.py, part.pz, part.mass = \
                new_mom[ind].E, new_mom[ind].px, new_mom[ind].py, new_mom[ind].pz,new_mom[ind].mass
                ind+=1
        return jac
    
    def change_sqrts(self, new_sqrts):
        """routine to rescale the momenta to change the invariant mass"""
        
        old_momenta = []
        incoming = []
        masses = []        
        for part in self:
            if part.status == -1:
                incoming.append(FourMomentum(part))
            if part.status == 1:
                old_momenta.append(FourMomentum(part))
                masses.append(part.mass)
        
        p_init = FourMomentum()
        p_inits = []
        n_init = 0
        for p in incoming:
            n_init +=1
            p_init += p
            p_inits.append(p)
        old_sqrts = p_init.mass

        new_mom, jac = self.mass_shuffle(old_momenta, old_sqrts, masses, new_sqrts=new_sqrts)
        
        #modify the momenta of the particles:
        ind =0
        for part in self:
            if part.status==1:
                part.E, part.px, part.py, part.pz, part.mass = \
                new_mom[ind].E, new_mom[ind].px, new_mom[ind].py, new_mom[ind].pz,new_mom[ind].mass
                ind+=1
        
        #change the initial state
        p_init = FourMomentum()
        for part in self:
            if part.status==1:
                p_init += part
        if n_init == 1:
            for part in self:
                if part.status == -1:
                    part.E, part.px, part.py, part.pz = \
                                 p_init.E, p_init.px, p_init.py, p_init.pz
        elif n_init ==2:
            if not misc.equal(p_init.px, 0) or not  misc.equal(p_init.py, 0):
                raise Exception
            if not misc.equal(p_inits[0].px, 0) or not  misc.equal(p_inits[0].py, 0):
                raise Exception            
            #assume that initial energy is written as
            # p1 = (sqrts/2*exp(eta),   0, 0 , E1)
            # p2 = (sqrts/2*exp(-eta),   0, 0 , -E2)
            # keep eta fix
            eta = math.log(2*p_inits[0].E/old_sqrts)
            new_p = [[new_sqrts/2*math.exp(eta), 0., 0., new_sqrts/2*math.exp(eta)],
                     [new_sqrts/2*math.exp(-eta), 0., 0., -new_sqrts/2*math.exp(-eta)]] 
            
            ind=0
            for part in self:
                if part.status == -1:
                    part.E, part.px, part.py, part.pz = new_p[ind]
                    ind+=1
                    if ind ==2:
                        break
        else:
            raise Exception
                            
        return jac        
        
    
    def get_helicity(self, get_order, allow_reversed=True):
        """return a list with the helicities in the order asked for"""
        
        #avoid to modify the input
        order = [list(get_order[0]), list(get_order[1])] 
        out = [9] *(len(order[0])+len(order[1]))
        for i, part in enumerate(self):
            if part.status == 1: #final
                try:
                    ind = order[1].index(part.pid)
                except ValueError, error:
                    if not allow_reversed:
                        raise error
                    else:
                        order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                        try:
                            return self.get_helicity(order, False)
                        except ValueError:
                            raise error     
                position = len(order[0]) + ind
                order[1][ind] = 0   
            elif part.status == -1:
                try:
                    ind = order[0].index(part.pid)
                except ValueError, error:
                    if not allow_reversed:
                        raise error
                    else:
                        order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                        try:
                            return self.get_helicity(order, False)
                        except ValueError:
                            raise error
                 
                position =  ind
                order[0][ind] = 0
            else: #intermediate
                continue
            out[position] = int(part.helicity)
        return out  

    
    def check_color_structure(self):
        """check the validity of the color structure"""
        
        #1. check that each color is raised only once.
        color_index = collections.defaultdict(int)
        for particle in self:
            if particle.status in [-1,1]:
                if particle.color1:
                    color_index[particle.color1] +=1
                    if -7 < particle.pdg < 0:
                        raise Exception, "anti-quark with color tag"
                if particle.color2:
                    color_index[particle.color2] +=1     
                    if 7 > particle.pdg > 0:
                        raise Exception, "quark with anti-color tag"                
                
                
        for key,value in color_index.items():
            if value > 2:
                print self
                print key, value
                raise Exception, 'Wrong color_flow'           
        
        
        #2. check that each parent present have coherent color-structure
        check = []
        popup_index = [] #check that the popup index are created in a unique way
        for particle in self:
            mothers = []
            childs = []
            if particle.mother1:
                mothers.append(particle.mother1)
            if particle.mother2 and particle.mother2 is not particle.mother1:
                mothers.append(particle.mother2)                 
            if not mothers:
                continue
            if (particle.mother1.event_id, particle.mother2.event_id) in check:
                continue
            check.append((particle.mother1.event_id, particle.mother2.event_id))
            
            childs = [p for p in self if p.mother1 is particle.mother1 and \
                                         p.mother2 is particle.mother2]
            
            mcolors = []
            manticolors = []
            for m in mothers:
                if m.color1:
                    if m.color1 in manticolors:
                        manticolors.remove(m.color1)
                    else:
                        mcolors.append(m.color1)
                if m.color2:
                    if m.color2 in mcolors:
                        mcolors.remove(m.color2)
                    else:
                        manticolors.append(m.color2)
            ccolors = []
            canticolors = []
            for m in childs:
                if m.color1:
                    if m.color1 in canticolors:
                        canticolors.remove(m.color1)
                    else:
                        ccolors.append(m.color1)
                if m.color2:
                    if m.color2 in ccolors:
                        ccolors.remove(m.color2)
                    else:
                        canticolors.append(m.color2)
            for index in mcolors[:]:
                if index in ccolors:
                    mcolors.remove(index)
                    ccolors.remove(index)
            for index in manticolors[:]:
                if index in canticolors:
                    manticolors.remove(index)
                    canticolors.remove(index)             
                        
            if mcolors != []:
                #only case is a epsilon_ijk structure.
                if len(canticolors) + len(mcolors) != 3:
                    logger.critical(str(self))
                    raise Exception, "Wrong color flow for %s -> %s" ([m.pid for m in mothers], [c.pid for c in childs])              
                else:
                    popup_index += canticolors
            elif manticolors != []:
                #only case is a epsilon_ijk structure.
                if len(ccolors) + len(manticolors) != 3:
                    logger.critical(str(self))
                    raise Exception, "Wrong color flow for %s -> %s" ([m.pid for m in mothers], [c.pid for c in childs])              
                else:
                    popup_index += ccolors

            # Check that color popup (from epsilon_ijk) are raised only once
            if len(popup_index) != len(set(popup_index)):
                logger.critical(self)
                raise Exception, "Wrong color flow: identical poping-up index, %s" % (popup_index)
               
    def __eq__(self, other):
        """two event are the same if they have the same momentum. other info are ignored"""
        
        if other is None:
            return False
        
        for i,p in enumerate(self):
            if p.E != other[i].E:
                return False
            elif p.pz != other[i].pz:
                return False
            elif p.px != other[i].px:
                return False
            elif p.py != other[i].py:
                return False
        return True
        
               
    def __str__(self, event_id=''):
        """return a correctly formatted LHE event"""
        
        out="""<event%(event_flag)s>
%(scale)s
%(particles)s
%(comments)s
%(tag)s
%(reweight)s
</event>
""" 
        if event_id not in ['', None]:
            self.eventflag['event'] = str(event_id)

        if self.eventflag:
            event_flag = ' %s' % ' '.join('%s="%s"' % (k,v) for (k,v) in self.eventflag.items())
        else:
            event_flag = ''

        scale_str = "%2d %6d %+13.7e %14.8e %14.8e %14.8e" % \
            (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)

            
        if self.reweight_data:
            # check that all key have an order if not add them at the end
            if set(self.reweight_data.keys()) != set(self.reweight_order):
                self.reweight_order += [k for k in self.reweight_data.keys() \
                                                if k not in self.reweight_order]

            reweight_str = '<rwgt>\n%s\n</rwgt>' % '\n'.join(
                        '<wgt id=\'%s\'> %+13.7e </wgt>' % (i, float(self.reweight_data[i]))
                        for i in self.reweight_order if i in self.reweight_data)
        else:
            reweight_str = '' 
            
        tag_str = self.tag
        if hasattr(self, 'nloweight') and self.nloweight.modified:
            self.rewrite_nlo_weight()
            tag_str = self.tag
            
        if self.matched_scale_data:
            tmp_scale = ' '.join(['pt_clust_%i=\"%s\"' % (i+1,v)
                                   for i,v in enumerate(self.matched_scale_data)
                                              if v!=-1])
            if tmp_scale:
                tag_str = "<scales %s></scales>%s" % (tmp_scale, self.tag)
            
        if self.syscalc_data:
            keys= ['rscale', 'asrwt', ('pdfrwt', 'beam', '1'), ('pdfrwt', 'beam', '2'),
                   'matchscale', 'totfact']
            sys_str = "<mgrwt>\n"
            template = """<%(key)s%(opts)s>%(values)s</%(key)s>\n"""
            for k in keys:
                if k not in self.syscalc_data:
                    continue
                replace = {}
                replace['values'] = self.syscalc_data[k]
                if isinstance(k, str):
                    replace['key'] = k
                    replace['opts'] = ''
                else:
                    replace['key'] = k[0]
                    replace['opts'] = ' %s=\"%s\"' % (k[1],k[2])                    
                sys_str += template % replace
            sys_str += "</mgrwt>\n"
            reweight_str = sys_str + reweight_str
        
        out = out % {'event_flag': event_flag,
                     'scale': scale_str, 
                      'particles': '\n'.join([str(p) for p in self]),
                      'tag': tag_str,
                      'comments': self.comment,
                      'reweight': reweight_str}
        
        return re.sub('[\n]+', '\n', out)

    def get_momenta(self, get_order, allow_reversed=True):
        """return the momenta vector in the order asked for"""
        
        #avoid to modify the input
        order = [list(get_order[0]), list(get_order[1])] 
        out = [''] *(len(order[0])+len(order[1]))
        for i, part in enumerate(self):
            if part.status == 1: #final
                try:
                    ind = order[1].index(part.pid)
                except ValueError, error:
                    if not allow_reversed:
                        raise error
                    else:
                        order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                        try:
                            return self.get_momenta_str(order, False)
                        except ValueError:
                            raise error     
                position = len(order[0]) + ind
                order[1][ind] = 0   
            elif part.status == -1:
                try:
                    ind = order[0].index(part.pid)
                except ValueError, error:
                    if not allow_reversed:
                        raise error
                    else:
                        order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                        try:
                            return self.get_momenta_str(order, False)
                        except ValueError:
                            raise error
                 
                position =  ind
                order[0][ind] = 0
            else: #intermediate
                continue

            out[position] = (part.E, part.px, part.py, part.pz)
            
        return out

    
    def get_scale(self,type):
        
        if type == 1:
            return self.get_et_scale()
        elif type == 2:
            return self.get_ht_scale()
        elif type == 3:
            return self.get_ht_scale(prefactor=0.5)
        elif type == 4:
            return self.get_sqrts_scale()
        elif type == -1:
            return self.get_ht_scale(prefactor=0.5)
        
    
    def get_ht_scale(self, prefactor=1):
        
        scale = 0 
        for particle in self:
            if particle.status != 1:
                continue
            p=FourMomentum(particle)
            scale += math.sqrt(p.mass_sqr + p.pt**2)
    
        return prefactor * scale
    

    def get_et_scale(self, prefactor=1):
        
        scale = 0 
        for particle in self:
            if particle.status != 1:
                continue 
            p = FourMomentum(particle)
            pt = p.pt
            if (pt>0):
                scale += p.E*pt/math.sqrt(pt**2+p.pz**2)
    
        return prefactor * scale    
    
    @property
    def sqrts(self):
        return self.get_sqrts_scale(1)
    
    def get_sqrts_scale(self, prefactor=1):
        
        scale = 0 
        init = []
        for particle in self:
            if particle.status == -1:
                init.append(FourMomentum(particle))
        if len(init) == 1:
            return init[0].mass
        elif len(init)==2:
            return math.sqrt((init[0]+init[1])**2)
                   
    
    
    
    def get_momenta_str(self, get_order, allow_reversed=True):
        """return the momenta str in the order asked for"""
        
        out = self.get_momenta(get_order, allow_reversed)
        #format
        format = '%.12f'
        format_line = ' '.join([format]*4) + ' \n'
        out = [format_line % one for one in out]
        out = ''.join(out).replace('e','d')
        return out    

class WeightFile(EventFile):
    """A class to allow to read both gzip and not gzip file.
       containing only weight from pythia --generated by SysCalc"""

    def __new__(self, path, mode='r', *args, **opt):
        if  path.endswith(".gz"):
            try:
                return gzip.GzipFile.__new__(WeightFileGzip, path, mode, *args, **opt)
            except IOError, error:
                raise
            except Exception, error:
                if mode == 'r':
                    misc.gunzip(path)
                return file.__new__(WeightFileNoGzip, path[:-3], mode, *args, **opt)
        else:
            return file.__new__(WeightFileNoGzip, path, mode, *args, **opt)
    
    
    def __init__(self, path, mode='r', *args, **opt):
        """open file and read the banner [if in read mode]"""
        
        super(EventFile, self).__init__(path, mode, *args, **opt)
        self.banner = ''
        if mode == 'r':
            line = ''
            while '</header>' not in line.lower():
                try:
                    line  = super(EventFile, self).next()
                except StopIteration:
                    self.seek(0)
                    self.banner = ''
                    break 
                if "<event" in line.lower():
                    self.seek(0)
                    self.banner = ''
                    break                     

                self.banner += line


class WeightFileGzip(WeightFile, EventFileGzip):
    pass

class WeightFileNoGzip(WeightFile, EventFileNoGzip):
    pass


class FourMomentum(object):
    """a convenient object for 4-momenta operation"""
    
    def __init__(self, obj=0, px=0, py=0, pz=0, E=0):
        """initialize the four momenta"""

        if obj is 0 and E:
            obj = E
         
        if isinstance(obj, (FourMomentum, Particle)):
            px = obj.px
            py = obj.py
            pz = obj.pz
            E = obj.E
        elif isinstance(obj, (list, tuple)):
            assert len(obj) ==4
            E = obj[0]
            px = obj[1]
            py = obj[2] 
            pz = obj[3]
        elif  isinstance(obj, str):
            obj = [float(i) for i in obj.split()]
            assert len(obj) ==4
            E = obj[0]
            px = obj[1]
            py = obj[2] 
            pz = obj[3]            
        else:
            E =obj

            
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)

    @property
    def mass(self):
        """return the mass"""    
        return math.sqrt(max(self.E**2 - self.px**2 - self.py**2 - self.pz**2,0))

    @property
    def mass_sqr(self):
        """return the mass square"""    
        return self.E**2 - self.px**2 - self.py**2 - self.pz**2

    @property
    def pt(self):
        return math.sqrt(max(0, self.pt2))
    
    @property
    def pseudorapidity(self):
        norm = math.sqrt(self.px**2 + self.py**2+self.pz**2)
        return  0.5* math.log((norm - self.pz) / (norm + self.pz))
    
    @property
    def rapidity(self):
        return  0.5* math.log((self.E +self.pz) / (self.E - self.pz))
    
    
    @property
    def pt2(self):
        """ return the pt square """
        
        return  self.px**2 + self.py**2
    
    @property
    def norm(self):
        """ return |\vec p| """
        return math.sqrt(self.px**2 + self.py**2 + self.pz**2) 

    @property
    def norm_sq(self):
        """ return |\vec p|^2 """
        return self.px**2 + self.py**2 + self.pz**2
    
    @property
    def theta(self):                                                                                                  
        """return the mass square""" 
        import math  
        return math.atan(math.sqrt((self.px**2+self.py**2)/self.pz**2))
    
    
    def __add__(self, obj):
        
        assert isinstance(obj, FourMomentum)
        new = FourMomentum(self.E+obj.E,
                           self.px + obj.px,
                           self.py + obj.py,
                           self.pz + obj.pz)
        return new
    
    def __iadd__(self, obj):
        """update the object with the sum"""
        self.E += obj.E
        self.px += obj.px
        self.py += obj.py
        self.pz += obj.pz
        return self

    def __sub__(self, obj):
        
        assert isinstance(obj, FourMomentum)
        new = FourMomentum(self.E-obj.E,
                           self.px - obj.px,
                           self.py - obj.py,
                           self.pz - obj.pz)
        return new

    def __isub__(self, obj):
        """update the object with the sum"""
        self.E -= obj.E
        self.px -= obj.px
        self.py -= obj.py
        self.pz -= obj.pz
        return self
    
    def __mul__(self, obj):
        if isinstance(obj, FourMomentum):
            return self.E*obj.E - self.px *obj.px - self.py * obj.py - self.pz * obj.pz
        elif isinstance(obj, (float, int)):
            return FourMomentum(obj*self.E,obj*self.px,obj*self.py,obj*self.pz )
        else:
            raise NotImplemented
    __rmul__ = __mul__
    
    def __pow__(self, power):
        assert power in [1,2]
        
        if power == 1:
            return FourMomentum(self)
        elif power == 2:
            return self.mass_sqr
    
    def __repr__(self):
        return 'FourMomentum(%s,%s,%s,%s)' % (self.E, self.px, self.py,self.pz)
    
    def __str__(self, mode='python'):
        if mode == 'python':
            return self.__repr__()
        elif mode == 'fortran':
            return '%.10e %.10e %.10e %.10e' % self.get_tuple()
    
    def get_tuple(self):
        return (self.E, self.px, self.py,self.pz)
    
    def boost(self, mom):
        """mom 4-momenta is suppose to be given in the rest frame of this 4-momenta.
        the output is the 4-momenta in the frame of this 4-momenta
        function copied from HELAS routine."""

        
        pt = self.px**2 + self.py**2 + self.pz**2
        if pt:
            s3product = self.px * mom.px + self.py * mom.py + self.pz * mom.pz
            mass = self.mass
            lf = (mom.E + (self.E - mass) * s3product / pt ) / mass
            return FourMomentum(E=(self.E*mom.E+s3product)/mass,
                           px=mom.px + self.px * lf,
                           py=mom.py + self.py * lf,
                           pz=mom.pz + self.pz * lf)
        else:
            return FourMomentum(mom)

    def zboost(self, pboost=None, E=0, pz=0):
        """Both momenta should be in the same frame. 
           The boost perform correspond to the boost required to set pboost at 
           rest (only z boost applied).
        """
        if isinstance(pboost, FourMomentum):
            E = pboost.E
            pz = pboost.pz
        
        #beta = pz/E
        gamma = E / math.sqrt(E**2-pz**2)
        gammabeta = pz  / math.sqrt(E**2-pz**2)
        
        out =  FourMomentum([gamma*self.E - gammabeta*self.pz,
                            self.px,
                            self.py,
                            gamma*self.pz - gammabeta*self.E])
        
        if abs(out.pz) < 1e-6 * out.E:
            out.pz = 0
        return out
    
    def boost_to_restframe(self, pboost):
        """apply the boost transformation such that pboost is at rest in the new frame.
        First apply a rotation to allign the pboost to the z axis and then use
        zboost routine (see above)
        """
        
        if pboost.px == 0 == pboost.py:
            out = self.zboost(E=pboost.E,pz=pboost.pz)
            return out
        
        
        # write pboost as (E, p cosT sinF, p sinT sinF, p cosF)
        # rotation such that it become (E, 0 , 0 , p ) is
        #  cosT sinF  ,  -sinT  , cosT sinF
        #  sinT cosF  ,  cosT   , sinT sinF
        # -sinT       ,   0     , cosF
        p  =  math.sqrt( pboost.px**2 + pboost.py**2+ pboost.pz**2)
        cosF = pboost.pz / p
        sinF = math.sqrt(1-cosF**2)
        sinT = pboost.py/p/sinF
        cosT = pboost.px/p/sinF
        
        out=FourMomentum([self.E,
                          self.px*cosT*cosF + self.py*sinT*cosF-self.pz*sinF,
                          -self.px*sinT+      self.py*cosT,
                          self.px*cosT*sinF + self.py*sinT*sinF + self.pz*cosF
                          ])
        out = out.zboost(E=pboost.E,pz=p)
        return out
        
        
        

class OneNLOWeight(object):
        
    def __init__(self, input, real_type=(1,11)):
        """ """

        self.real_type = real_type
        if isinstance(input, str):
            self.parse(input)
        
    def __str__(self, mode='display'):
        
        if mode == 'display':
            out = """        pwgt: %(pwgt)s
            born, real : %(born)s %(real)s
            pdgs : %(pdgs)s
            bjks : %(bjks)s
            scales**2, gs: %(scales2)s %(gs)s
            born/real related : %(born_related)s %(real_related)s
            type / nfks : %(type)s  %(nfks)s
            to merge : %(to_merge_pdg)s in %(merge_new_pdg)s
            ref_wgt :  %(ref_wgt)s""" % self.__dict__
            return out
        elif mode == 'formatted':
            format_var = []
            variable = []
            
            def to_add_full(f, v, format_var, variable):
                """ function to add to the formatted output"""
                if isinstance(v, list):
                    format_var += [f]*len(v)
                    variable += v
                else:
                    format_var.append(f)
                    variable.append(v)
            to_add = lambda x,y: to_add_full(x,y, format_var, variable)
            #set the formatting
            to_add('%.10e', [p*self.bias_wgt for p in self.pwgt])
            to_add('%.10e', self.born)
            to_add('%.10e', self.real)
            to_add('%i', self.nexternal)
            to_add('%i', self.pdgs)
            to_add('%i', self.qcdpower)
            to_add('%.10e', self.bjks)
            to_add('%.10e', self.scales2)
            to_add('%.10e', self.gs)
            to_add('%i', [self.born_related, self.real_related])
            to_add('%i' , [self.type, self.nfks])
            to_add('%i' , self.to_merge_pdg)
            to_add('%i', self.merge_new_pdg)
            to_add('%.10e', self.ref_wgt*self.bias_wgt)
            to_add('%.10e', self.bias_wgt)
            return ' '.join(format_var) % tuple(variable)
            
        
    def parse(self, text, keep_bias=False):
        """parse the line and create the related object.
           keep bias allow to not systematically correct for the bias in the written information"""
        #0.546601845792D+00 0.000000000000D+00 0.000000000000D+00 0.119210435309D+02 0.000000000000D+00  5 -1 2 -11 12 21 0 0.24546101D-01 0.15706890D-02 0.12586055D+04 0.12586055D+04 0.12586055D+04  1  2  2  2  5  2  2 0.539995789976D+04
        #0.274922677249D+01 0.000000000000D+00 0.000000000000D+00 0.770516514633D+01 0.113763730192D+00  5 21 2 -11 12 1 2 0.52500539D-02 0.30205908D+00 0.45444066D+04 0.45444066D+04 0.45444066D+04 0.12520062D+01  1  2  1  3  5  1       -1 0.110944218997D+05
        # below comment are from Rik description email
        data = text.split()
        # 1. The first three doubles are, as before, the 'wgt', i.e., the overall event of this
        # contribution, and the ones multiplying the log[mu_R/QES] and the log[mu_F/QES]
        # stripped of alpha_s and the PDFs.
        # from example: 0.274922677249D+01 0.000000000000D+00 0.000000000000D+00
        self.pwgt = [float(f) for f in data[:3]]
        # 2. The next two doubles are the values of the (corresponding) Born and 
        #    real-emission matrix elements. You can either use these values to check 
        #    that the newly computed original matrix element weights are correct, 
        #    or directly use these so that you don't have to recompute the original weights. 
        #    For contributions for which the real-emission matrix elements were 
        #    not computed, the 2nd of these numbers is zero. The opposite is not true, 
        #    because each real-emission phase-space configuration has an underlying Born one 
        #    (this is not unique, but on our code we made a specific choice here). 
        #    This latter information is useful if the real-emission matrix elements 
        #    are unstable; you can then reweight with the Born instead. 
        #    (see also point 9 below, where the momentum configurations are assigned). 
        #    I don't think this instability is real problem when reweighting the real-emission 
        #    with tree-level matrix elements (as we generally would do), but is important 
        #    when reweighting with loop-squared contributions as we have been doing for gg->H. 
        #    (I'm not sure that reweighting tree-level with loop^2 is something that 
        #    we can do in general, because we don't really know what to do with the 
        #    virtual matrix elements because we cannot generate 2-loop diagrams.)
        #    from example: 0.770516514633D+01 0.113763730192D+00
        self.born = float(data[3])
        self.real = float(data[4])
        # 3. integer: number of external particles of the real-emission configuration  (as before)
        #    from example: 5
        self.nexternal = int(data[5])
        # 4. PDG codes corresponding to the real-emission configuration (as before)
        #    from example: 21 2 -11 12 1 2
        self.pdgs = [int(i) for i in data[6:6+self.nexternal]]
        flag = 6+self.nexternal # new starting point for the position
        # 5. next integer is the power of g_strong in the matrix elements (as before)
        #    from example: 2
        self.qcdpower = int(data[flag])
        # 6. 2 doubles: The bjorken x's used for this contribution (as before)
        #    from example: 0.52500539D-02 0.30205908D+00 
        self.bjks = [float(f) for f in data[flag+1:flag+3]]
        # 7. 3 doubles: The Ellis-sexton scale, the renormalisation scale and the factorisation scale, all squared, used for this contribution (as before)
        #    from example: 0.45444066D+04 0.45444066D+04 0.45444066D+04
        self.scales2 = [float(f) for f in data[flag+3:flag+6]]
        # 8.the value of g_strong
        #    from example:  0.12520062D+01 
        self.gs = float(data[flag+6])
        # 9. 2 integers: the corresponding Born and real-emission type kinematics. (in the list of momenta)
        #    Note that also the Born-kinematics has n+1 particles, with, in general, 
        #    one particle with zero momentum (this is not ALWAYS the case, 
        #    there could also be 2 particles with perfectly collinear momentum). 
        #    To convert this from n+1 to a n particles, you have to sum the momenta 
        #    of the two particles that 'merge', see point 12 below.
        #    from example:  1  2 
        self.born_related = int(data[flag+7])
        self.real_related = int(data[flag+8])
        # 10. 1 integer: the 'type'. This is the information you should use to determine 
        #     if to reweight with Born, virtual or real-emission matrix elements. 
        #     (Apart from the possible problems with complicated real-emission matrix elements
        #     that need to be computed very close to the soft/collinear limits, see point 2 above. 
        #     I guess that for tree-level this is always okay, but when reweighting 
        #     a tree-level contribution with a one-loop squared one, as we do 
        #     for gg->Higgs, this is important). 
        #     type=1 : real-emission:     
        #     type=2 : Born: 
        #     type=3 : integrated counter terms: 
        #     type=4 : soft counter-term: 
        #     type=5 : collinear counter-term: 
        #     type=6 : soft-collinear counter-term: 
        #     type=7 : O(alphaS) expansion of Sudakov factor for NNLL+NLO:  
        #     type=8 : soft counter-term (with n+1-body kin.):     
        #     type=9 : collinear counter-term (with n+1-body kin.): 
        #     type=10: soft-collinear counter-term (with n+1-body kin.): 
        #     type=11: real-emission (with n-body kin.): 
        #     type=12: MC subtraction with n-body kin.: 
        #     type=13: MC subtraction with n+1-body kin.: 
        #     type=14: virtual corrections minus approximate virtual
        #     type=15: approximate virtual corrections: 
        #     from example: 1 
        self.type = int(data[flag+9])
        # 11. 1 integer: The FKS configuration for this contribution (not really 
        #     relevant for anything, but is used in checking the reweighting to 
        #     get scale & PDF uncertainties).
        #     from example:  3  
        self.nfks = int(data[flag+10])
        # 12. 2 integers: the two particles that should be merged to form the 
        #     born contribution from the real-emission one. Remove these two particles
        #     from the (ordered) list of PDG codes, and insert a newly created particle
        #     at the location of the minimum of the two particles removed. 
        #     I.e., if you merge particles 2 and 4, you have to insert the new particle 
        #     as the 2nd particle. And particle 5 and above will be shifted down by one.
        #     from example: 5  1      
        self.to_merge_pdg = [int (f) for f in data[flag+11:flag+13]]
        # 13. 1 integer: the PDG code of the particle that is created after merging the two particles at point 12.
        #     from example  -1 
        self.merge_new_pdg = int(data[flag+13])
        # 14. 1 double: the reference number that one should be able to reconstruct 
        #     form the weights (point 1 above) and the rest of the information of this line. 
        #     This is really the contribution to this event as computed by the code 
        #     (and is passed to the integrator). It contains everything.
        #     from example: 0.110944218997D+05  
        self.ref_wgt = float(data[flag+14])
        # 15. The bias weight. This weight is included in the self.ref_wgt, as well as in
        #     the self.pwgt. However, it is already removed from the XWGTUP (and
        #     scale/pdf weights). That means that in practice this weight is not used.
        try:
            self.bias_wgt = float(data[flag+15])
        except IndexError:
            self.bias_wgt = 1.0
            
        if not keep_bias:
            self.ref_wgt /= self.bias_wgt
            self.pwgt = [p/self.bias_wgt for p in self.pwgt]

        #check the momenta configuration linked to the event
        if self.type in self.real_type:
            self.momenta_config = self.real_related
        else:
            self.momenta_config = self.born_related


class NLO_PARTIALWEIGHT(object):

    class BasicEvent(list):

        
        def __init__(self, momenta, wgts, event, real_type=(1,11)):
            
            list.__init__(self, momenta)
            assert self
            self.soft = False
            self.wgts = wgts
            self.pdgs = list(wgts[0].pdgs)
            self.event = event
            self.real_type = real_type
            
            if wgts[0].momenta_config == wgts[0].born_related:
                # need to remove one momenta.
                ind1, ind2 = [ind-1 for ind in wgts[0].to_merge_pdg] 
                if ind1> ind2: 
                    ind1, ind2 = ind2, ind1
                if ind1 >= sum(1 for p in event if p.status==-1):
                    new_p = self[ind1] + self[ind2]
                else:
                    new_p = self[ind1] - self[ind2]
                self.pop(ind1) 
                self.insert(ind1, new_p)
                self.pop(ind2)
                self.pdgs.pop(ind1) 
                self.pdgs.insert(ind1, wgts[0].merge_new_pdg )
                self.pdgs.pop(ind2)
                # DO NOT update the pdgs of the partial weight!

            elif any(w.type in self.real_type for w in wgts):
                if any(w.type not in self.real_type for w in wgts):
                    raise Exception
                # Do nothing !!!
                # previously (commented we were checking here if the particle 
                # were too soft this is done  later now
#                    The comment line below allow to convert this event 
#                    to a born one (old method)    
#                    self.pop(ind1) 
#                    self.insert(ind1, new_p)
#                    self.pop(ind2)
#                    self.pdgs.pop(ind1) 
#                    self.pdgs.insert(ind1, wgts[0].merge_new_pdg )
#                    self.pdgs.pop(ind2)                 
#                    # DO NOT update the pdgs of the partial weight!                    
            else:
                raise Exception

        def check_fks_singularity(self, ind1, ind2, nb_init=2, threshold=None):
            """check that the propagator associated to ij is not too light 
               [related to soft-collinear singularity]"""

            if threshold is None:
                threshold = 1e-8
                
            if ind1> ind2: 
                ind1, ind2 = ind2, ind1                
            if ind1 >= nb_init:
                new_p = self[ind1] + self[ind2]
            else:
                new_p = self[ind1] - self[ind2]
                
            inv_mass = new_p.mass_sqr
            if nb_init == 2:
                shat = (self[0]+self[1]).mass_sqr
            else:
                shat = self[0].mass_sqr
            
            
            if (abs(inv_mass)/shat < threshold):
                return True
            else:
                return False
 
 
        def get_pdg_code(self):
            return self.pdgs
            
        def get_tag_and_order(self):
            """ return the tag and order for this basic event""" 
            (initial, _), _ = self.event.get_tag_and_order()
            order = self.get_pdg_code()
            
            
            initial, out = order[:len(initial)], order[len(initial):]
            initial.sort()
            out.sort()
            return (tuple(initial), tuple(out)), order
        
        def get_momenta(self, get_order, allow_reversed=True):
            """return the momenta vector in the order asked for"""
             
            #avoid to modify the input
            order = [list(get_order[0]), list(get_order[1])] 
            out = [''] *(len(order[0])+len(order[1]))
            pdgs = self.get_pdg_code()
            for pos, part in enumerate(self):
                if pos < len(get_order[0]): #initial
                    try:
                        ind = order[0].index(pdgs[pos])
                    except ValueError, error:
                        if not allow_reversed:
                            raise error
                        else:
                            order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                            try:
                                return self.get_momenta(order, False)
                            except ValueError:
                                raise error   
                            
                                                 
                    position =  ind
                    order[0][ind] = 0             
                else: #final   
                    try:
                        ind = order[1].index(pdgs[pos])
                    except ValueError, error:
                        if not allow_reversed:
                            raise error
                        else:
                            order = [[-i for i in get_order[0]],[-i for i in get_order[1]]]
                            try:
                                return self.get_momenta(order, False)
                            except ValueError:
                                raise error     
                    position = len(order[0]) + ind
                    order[1][ind] = 0   
    
                out[position] = (part.E, part.px, part.py, part.pz)
                
            return out
            
            
        def get_helicity(self, *args):
            return [9] * len(self)
        
        @property
        def aqcd(self):
            return self.event.aqcd
        
        def get_ht_scale(self, prefactor=1):
        
            scale = 0 
            for particle in self:
                p = particle
                scale += math.sqrt(max(0, p.mass_sqr + p.pt**2))
        
            return prefactor * scale
        
        def get_et_scale(self, prefactor=1):
            
            scale = 0 
            for particle in self:
                p = particle
                pt = p.pt
                if (pt>0):
                    scale += p.E*pt/math.sqrt(pt**2+p.pz**2)
        
            return prefactor * scale    
        
        
        def get_sqrts_scale(self, event,prefactor=1):
            
            scale = 0 
            nb_init = 0
            for particle in event:
                if particle.status == -1:
                    nb_init+=1
            if nb_init == 1:
                return self[0].mass
            elif nb_init==2:
                return math.sqrt((self[0]+self[1])**2)
                   
    
        
            
    def __init__(self, input, event, real_type=(1,11), threshold=None):
        
        self.real_type = real_type
        self.event = event
        self.total_wgt = 0.
        self.nb_event = 0
        self.nb_wgts = 0
        self.threshold = threshold
        self.modified = False #set on True if we decide to change internal infor
                              # that need to be written in the event file.
                              #need to be set manually when this is the case
        if isinstance(input, str):
            self.parse(input)
        
            
        
    def parse(self, text):
        """create the object from the string information (see example below)"""
#0.2344688900d+00    8    2    0
#0.4676614699d+02 0.0000000000d+00 0.0000000000d+00 0.4676614699d+02
#0.4676614699d+02 0.0000000000d+00 0.0000000000d+00 -.4676614699d+02
#0.4676614699d+02 0.2256794794d+02 0.4332148227d+01 0.4073073437d+02
#0.4676614699d+02 -.2256794794d+02 -.4332148227d+01 -.4073073437d+02
#0.0000000000d+00 -.0000000000d+00 -.0000000000d+00 -.0000000000d+00
#0.4780341163d+02 0.0000000000d+00 0.0000000000d+00 0.4780341163d+02
#0.4822581633d+02 0.0000000000d+00 0.0000000000d+00 -.4822581633d+02
#0.4729127470d+02 0.2347155377d+02 0.5153455534d+01 0.4073073437d+02
#0.4627255267d+02 -.2167412893d+02 -.3519736379d+01 -.4073073437d+02
#0.2465400591d+01 -.1797424844d+01 -.1633719155d+01 -.4224046944d+00
#0.473706252575d-01 0.000000000000d+00 0.000000000000d+00  5 -3 3 -11 11 21 0 0.11849903d-02 0.43683926d-01 0.52807978d+03 0.52807978d+03 0.52807978d+03  1  2  1 0.106660059627d+03
#-.101626389492d-02 0.000000000000d+00 -.181915673961d-03  5 -3 3 -11 11 21 2 0.11849903d-02 0.43683926d-01 0.52807978d+03 0.52807978d+03 0.52807978d+03  1  3  1 -.433615206719d+01
#0.219583436285d-02 0.000000000000d+00 0.000000000000d+00  5 -3 3 -11 11 21 2 0.11849903d-02 0.43683926d-01 0.52807978d+03 0.52807978d+03 0.52807978d+03  1 15  1 0.936909375537d+01
#0.290043597283d-03 0.000000000000d+00 0.000000000000d+00  5 -3 3 -11 11 21 2 0.12292838d-02 0.43683926d-01 0.58606724d+03 0.58606724d+03 0.58606724d+03  1 12  1 0.118841547979d+01
#-.856330613460d-01 0.000000000000d+00 0.000000000000d+00  5 -3 3 -11 11 21 2 0.11849903d-02 0.43683926d-01 0.52807978d+03 0.52807978d+03 0.52807978d+03  1  4  1 -.365375546483d+03
#0.854918237609d-01 0.000000000000d+00 0.000000000000d+00  5 -3 3 -11 11 21 2 0.12112732d-02 0.45047393d-01 0.58606724d+03 0.58606724d+03 0.58606724d+03  2 11  1 0.337816057347d+03
#0.359257891118d-05 0.000000000000d+00 0.000000000000d+00  5 21 3 -11 11 3 2 0.12292838d-02 0.43683926d-01 0.58606724d+03 0.58606724d+03 0.58606724d+03  1 12  3 0.334254554762d+00
#0.929944817736d-03 0.000000000000d+00 0.000000000000d+00  5 21 3 -11 11 3 2 0.12112732d-02 0.45047393d-01 0.58606724d+03 0.58606724d+03 0.58606724d+03  2 11  3 0.835109616010d+02
        
        
        text = text.lower().replace('d','e')
        all_line = text.split('\n')
        #get global information
        first_line =''
        while not first_line.strip():
            first_line = all_line.pop(0)
            
        wgt, nb_wgt, nb_event, _ = first_line.split()
        self.total_wgt = float(wgt.replace('d','e'))
        nb_wgt, nb_event = int(nb_wgt), int(nb_event)
        self.nb_wgt, self.nb_event = nb_wgt, nb_event
        
        momenta = []
        self.momenta = momenta #keep the original list of momenta to be able to rewrite the events
        wgts = []
        for line in all_line:
            data = line.split()
            if len(data) == 4:
                p = FourMomentum(data)
                momenta.append(p)
            elif len(data)>0:
                wgt = OneNLOWeight(line, real_type=self.real_type)
                wgts.append(wgt)
        
        assert len(wgts) == int(nb_wgt)
        
        get_weights_for_momenta = dict( (i,[]) for i in range(1,nb_event+1)  )
        size_momenta = 0
        for wgt in wgts:
            if wgt.momenta_config in get_weights_for_momenta:
                get_weights_for_momenta[wgt.momenta_config].append(wgt)
            else: 
                if size_momenta == 0: size_momenta = wgt.nexternal
                assert size_momenta == wgt.nexternal
                get_weights_for_momenta[wgt.momenta_config] = [wgt]
    
        assert sum(len(c) for c in get_weights_for_momenta.values()) == int(nb_wgt), "%s != %s" % (sum(len(c) for c in get_weights_for_momenta.values()), nb_wgt)
    
        # check singular behavior
        for key in range(1, nb_event+1):
            wgts = get_weights_for_momenta[key]
            if not wgts:
                continue
            if size_momenta == 0: size_momenta = wgts[0].nexternal
            p = momenta[size_momenta*(key-1):key*size_momenta]
            evt = self.BasicEvent(p, wgts, self.event, self.real_type) 
            if len(evt) == size_momenta: #real type 
                for wgt in wgts:
                    if not wgt.type in self.real_type:
                        continue
                    if evt.check_fks_singularity(wgt.to_merge_pdg[0]-1,
                                                 wgt.to_merge_pdg[1]-1,
                                                 nb_init=sum(1 for p in self.event if p.status==-1),
                                                 threshold=self.threshold):
                        get_weights_for_momenta[wgt.momenta_config].remove(wgt)
                        get_weights_for_momenta[wgt.born_related].append(wgt)
                        wgt.momenta_config = wgt.born_related
         
        assert sum(len(c) for c in get_weights_for_momenta.values()) == int(nb_wgt), "%s != %s" % (sum(len(c) for c in get_weights_for_momenta.values()), nb_wgt)
           
        self.cevents = []   
        for key in range(1, nb_event+1): 
            if key in get_weights_for_momenta:
                wgt = get_weights_for_momenta[key]
                if not wgt:
                    continue
                pdg_to_event = {}
                for w in wgt:
                    pdgs = w.pdgs
                    if w.momenta_config == w.born_related:
                        pdgs = list(pdgs)
                        ind1, ind2 = [ind-1 for ind in w.to_merge_pdg] 
                        if ind1> ind2: 
                            ind1, ind2 = ind2, ind1
                        pdgs.pop(ind1) 
                        pdgs.insert(ind1, w.merge_new_pdg )
                        pdgs.pop(ind2)
                    pdgs = tuple(pdgs)
                    if pdgs not in pdg_to_event:
                        p = momenta[size_momenta*(key-1):key*size_momenta]
                        evt = self.BasicEvent(p, [w], self.event, self.real_type)                         
                        self.cevents.append(evt)
                        pdg_to_event[pdgs] = evt
                    else:
                        pdg_to_event[pdgs].wgts.append(w)
        
        if __debug__: 
            nb_wgt_check = 0 
            for cevt in self.cevents:
                nb_wgt_check += len(cevt.wgts)
            assert nb_wgt_check == int(nb_wgt)
            
            

if '__main__' == __name__:   
    
    if False:
        lhe = EventFile('unweighted_events.lhe.gz')
        #lhe.parsing = False
        start = time.time()
        for event in lhe:
            event.parse_lo_weight()
        print 'old method -> ', time.time()-start
        lhe = EventFile('unweighted_events.lhe.gz')
        #lhe.parsing = False
        start = time.time()
        for event in lhe:
            event.parse_lo_weight_test()
        print 'new method -> ', time.time()-start    
    

    # Example 1: adding some missing information to the event (here distance travelled)
    if False: 
        start = time
        lhe = EventFile('unweighted_events.lhe.gz')
        output = open('output_events.lhe', 'w')
        #write the banner to the output file
        output.write(lhe.banner)
        # Loop over all events
        for event in lhe:
            for particle in event:
                # modify particle attribute: here remove the mass
                particle.mass = 0
                particle.vtim = 2 # The one associate to distance travelled by the particle.
    
            #write this modify event
            output.write(str(event))
        output.write('</LesHouchesEvent>\n')
        
    # Example 3: Plotting some variable
    if False:
        lhe = EventFile('unweighted_events.lhe.gz')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        nbins = 100
        
        nb_pass = 0
        data = []
        for event in lhe:
            etaabs = 0 
            etafinal = 0
            for particle in event:
                if particle.status==1:
                    p = FourMomentum(particle)
                    eta = p.pseudorapidity
                    if abs(eta) > etaabs:
                        etafinal = eta
                        etaabs = abs(eta)
            if etaabs < 4:
                data.append(etafinal)
                nb_pass +=1     

                        
        print nb_pass
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[5,1])
        gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
        ax = plt.subplot(gs1[0])
        
        n, bins, patches = ax.hist(data, nbins, histtype='step', label='original')
        ax_c = ax.twinx()
        ax_c.set_ylabel('MadGraph5_aMC@NLO')
        ax_c.yaxis.set_label_coords(1.01, 0.25)
        ax_c.set_yticks(ax.get_yticks())
        ax_c.set_yticklabels([])
        ax.set_xlim([-4,4])
        print "bin value:", n
        print "start/end point of bins", bins
        plt.axis('on')
        plt.xlabel('weight ratio')
        plt.show()


    # Example 4: More complex plotting example (with ratio plot)
    if False:
        lhe = EventFile('unweighted_events.lhe')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        nbins = 100
        
        #mtau, wtau = 45, 5.1785e-06
        mtau, wtau = 1.777, 4.027000e-13
        nb_pass = 0
        data, data2, data3 = [], [], []
        for event in lhe:
            nb_pass +=1
            if nb_pass > 10000:
                break
            tau1 = FourMomentum()
            tau2 = FourMomentum()
            for part in event:
                if part.pid in [-12,11,16]:
                    momenta = FourMomentum(part)
                    tau1 += momenta
                elif part.pid == 15:
                    tau2 += FourMomentum(part)

            if abs((mtau-tau2.mass())/wtau)<1e6 and tau2.mass() >1:               
                data.append((tau1.mass()-mtau)/wtau)
                data2.append((tau2.mass()-mtau)/wtau)   
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[5,1])
        gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
        ax = plt.subplot(gs1[0])
        
        n, bins, patches = ax.hist(data2, nbins, histtype='step', label='original')
        n2, bins2, patches2 = ax.hist(data, bins=bins, histtype='step',label='reconstructed')
        import cmath
        
        breit = lambda m : math.sqrt(4*math.pi)*1/(((m)**2-mtau**2)**2+(mtau*wtau)**2)*wtau
        
        data3 = [breit(mtau + x*wtau)*wtau*16867622.6624*50 for x in bins]

        ax.plot(bins, data3,label='breit-wigner')
        # add the legend
        ax.legend()
        # add on the right program tag
        ax_c = ax.twinx()
        ax_c.set_ylabel('MadGraph5_aMC@NLO')
        ax_c.yaxis.set_label_coords(1.01, 0.25)
        ax_c.set_yticks(ax.get_yticks())
        ax_c.set_yticklabels([])
        
        plt.title('invariant mass of tau LHE/reconstructed')
        plt.axis('on')
        ax.set_xticklabels([])
        # ratio plot
        ax = plt.subplot(gs1[1])
        data4 = [n[i]/(data3[i]) for i in range(nbins)]
        ax.plot(bins, data4 + [0] , 'b')
        data4 = [n2[i]/(data3[i]) for i in range(nbins)]
        ax.plot(bins, data4 + [0] , 'g')
        ax.set_ylim([0,2])
        #remove last y tick to avoid overlap with above plot:
        tick = ax.get_yticks()
        ax.set_yticks(tick[:-1])
        
        
        plt.axis('on')
        plt.xlabel('(M - Mtau)/Wtau')                                                                                                                                 
        plt.show()

        

                            
                            
    
    
    

