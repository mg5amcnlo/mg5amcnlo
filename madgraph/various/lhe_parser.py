import collections
import re
import misc



class Particle(object):
    """ """
    pattern=re.compile(r'''^\s*
        (?P<pid>-?\d+)\s+           #PID
        (?P<status>-?\d+)\s+            #status (1 for output particle)
        (?P<mother1>-?\d+)\s+       #mother
        (?P<mother2>-?\d+)\s+       #mother
        (?P<color1>[+-e.\d]*)\s+    #color1
        (?P<color2>[+-e.\d]*)\s+    #color2
        (?P<px>[+-e.\d]*)\s+        #px
        (?P<py>[+-e.\d]*)\s+        #py
        (?P<pz>[+-e.\d]*)\s+        #pz
        (?P<E>[+-e.\d]*)\s+         #E
        (?P<mass>[+-e.\d]*)\s+      #mass
        (?P<vtim>[+-e.\d]*)\s+      #displace vertex
        (?P<helicity>[+-e.\d]*)\s*      #helicity
        ($|(?P<comment>\#[\d|D]*))  #comment/end of string
        ''',66) #verbose+ignore case
    
    
    
    def __init__(self, line=None, event=None):
        """ """
        
        self.event = event
        self.event_id = len(event) #not yet in the event
        # LHE information
        self.pid = 0
        self.status = 0
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
        self.comment = ''

        if line:
            self.parse(line)
            
    def parse(self, line):
        """parse the line"""
    
        obj = self.pattern.search(line)
        if not obj:
            raise Exception, 'the line\n%s\n is not a valid format for LHE particle' % line
        for key, value in obj.groupdict().items():
            if key not in  ['comment','pid']:
                setattr(self, key, float(value))
            elif key in ['pid']:
                setattr(self, key, int(value))
            else:
                self.comment = value
        # assign the mother:
        if self.mother1:
            try:
                self.mother1 = self.event[int(self.mother1) -1]
            except KeyError:
                raise Exception, 'Wrong Events format: a daughter appears before it\'s mother'
        if self.mother2:
            try:
                self.mother2 = self.event[int(self.mother2) -1]
            except KeyError:
                raise Exception, 'Wrong Events format: a daughter appears before it\'s mother'
    
    
    
    
    def __str__(self):
        """string representing the particles"""
        return " %8d %2d %4d %4d %4d %4d %+13.7e %+13.7e %+13.7e %14.8e %14.8e %10.4e %10.4e" \
            % (self.pid, 
               self.status,
               self.mother1.event_id+1 if self.mother1 else 0,
               self.mother2.event_id+1 if self.mother2 else 0,
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
        
        
        
            
    def __repr__(self):
        return 'Particle("%s", event=%s)' % (str(self), self.event)
        
class EventFile(file):
    """ """
    
    def __init__(self, path, mode='r', *args, **opt):
        """open file and read the banner [if in read mode]"""
        
        file.__init__(self, path, mode, *args, **opt)
        self.banner = ''
        if mode == 'r':
            line = ''
            while '</init>' not in line.lower():
                try:
                    line  = file.next(self)
                except StopIteration:
                    self.seek(0)
                    self.banner = ''
                    break 
                if "<event>" in line.lower():
                    self.seek(0)
                    self.banner = ''
                    break                     

                self.banner += line

    def get_banner(self):
        """return a banner object"""
        import madgraph.various.banner as banner
        output = banner.Banner()
        output.read_banner(self.banner.split('\n'))
        return output
    
    
    def next(self):
        """get next event"""
        text = ''
        line = ''
        mode = 0
        while '</event>' not in line:
            line = file.next(self).lower()
            if '<event>' in line:
                mode = 1
            if mode:
                text += line
        return Event(text)
        
           
class Event(list):
    """Class storing a single event information (list of particles + global information)"""

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
        self.comment = ''
        self.reweight_data ={}
        
        if text:
            self.parse(text)
            
    def parse(self, text):
        """Take the input file and create the structured information"""
        
        text = re.sub(r'</?event>', '', text) # remove pointless tag
        status = 'first' 
        for line in text.split('\n'):
            line = line.strip()
            if not line: 
                continue
            if line.startswith('#'):
                self.comment += '%s\n' % line
                continue
            if 'first' == status:
                self.assign_scale_line(line)
                status = 'part' 
                continue
            
            if '<' in line:
                status = 'tag'
                
            if 'part' == status:
                self.append(Particle(line, event=self))
            else:
                self.tag += '%s\n' % line
                
    def parse_reweight(self):
        """Parse the re-weight information in order to return a dictionary
           {key: value}. If no group is define group should be '' """
           
        self.reweight_data = {}
        self.reweight_order = []
        start, stop = self.tag.find('<rwgt>'), self.tag.find('</rwgt>')
        if start != -1 != stop :
            pattern = re.compile(r'''<\s*wgt id=\'(?P<id>[^\']+)\'\s*>\s*(?P<val>[\ded+-.]*)\s*</wgt>''')
            data = pattern.findall(self.tag)
            try:
                self.reweight_data = dict([(pid, float(value)) for (pid, value) in data
                                           if not self.reweight_order.append(pid)])
                      # the if is to create the order file on the flight
            except ValueError, error:
                raise Exception, 'Event File has unvalid weight. %s' % error
            self.tag = self.tag[:start] + self.tag[stop+7:]

        

        

        
        
        
        
           
           
           
           
            
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
    
    def check_color_structure(self):
        """check that each color structure is raised only once.
        This is a debug routine not used anywhere in the code"""
        color_index = collections.defaultdict(int)
        
        for particle in self:
            if particle.color1:
                color_index[particle.color1] +=1
            if particle.color2:
                color_index[particle.color2] +=1     
                
        for key,value in color_index.items():
            if value > 2:
                print self
                print key, value
                raise Exception, 'Wrong color_flow'           
        
        
     
    
        
    def __str__(self):
        """return a correctly formatted LHE event"""
                
        out="""<event>
%(scale)s
%(particles)s
%(comments)s
%(tag)s
%(reweight)s
</event>
""" 

        scale_str = "%2d %6d %+13.7e %14.8e %14.8e %14.8e" % \
            (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)
        if self.reweight_data:
            # check that all key have an order if not add them at the end
            if set(self.reweight_data.keys()) != set(self.reweight_order):
                self.reweight_order += [k for k in self.reweight_data.keys() \
                                                if k not in self.reweight_order]
                
                
            reweight_str = '<rwgt>\n%s\n</rwgt>' % '\n'.join(
                        '<wgt id=\'%s\'> %+13.7e </wgt>' % (i, float(self.reweight_data[i]))
                        for i in self.reweight_order)
        else:
            reweight_str = '' 
        out = out % {'scale': scale_str, 
                      'particles': '\n'.join([str(p) for p in self]),
                      'tag': self.tag,
                      'comments': self.comment,
                      'reweight': reweight_str}
        return re.sub('[\n]+', '\n', out)
    
    def get_momenta_str(self, get_order, allow_reversed=True):
        """return the momenta str in the order asked for"""
        
        
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
            out[position] = '%g %g %g %g \n'% (part.E, part.px, part.py, part.pz)
            
        out = ''.join(out).replace('e','d')
        return out    



if '__main__' == __name__:    
    lhe = EventFile('unweighted_events.lhe')
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

    
    
    

