import re

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
            if key != 'comment':
                setattr(self, key, float(value))
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
                line  = file.next(self)
                self.banner += line
                
    
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
        self.rwgt = ''
        self.comment = ''
        
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
                status = 'rwgt'
                
            if 'part' == status:
                self.append(Particle(line, event=self))
            else:
                self.rwgt += '%s\n' % line
            
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
     
    
        
    def __str__(self):
        """return a correctly formatted LHE event"""
                
        out="""<event>
%(scale)s
%(particles)s
%(comments)s%(reweight)s</event>
""" 

        scale_str = "%2d %6d %+13.7e %14.8e %14.8e %14.8e" % \
            (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)

        return out % {'scale': scale_str, 
                      'particles': '\n'.join([str(p) for p in self]),
                      'reweight': self.rwgt,
                      'comments': self.comment}

        



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

    
    
    

