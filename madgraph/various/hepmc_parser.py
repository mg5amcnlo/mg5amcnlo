from __future__ import division

import gzip

if '__main__' == __name__:
    import sys
    sys.path.append('../../')
import misc
import os
import logging

class HEPMC_Particle(object):
    
    def __init__(self, text=None, event=None):
        
        self.barcode = 0
        self.pdg = 0
        self.px =  0
        self.py = 0
        self.pz = 0
        self.E = 0
        self.mass = 0
        self.status = 0
        self.polarization_theta = 0
        self.polarization_phi = 0
        self.vertex_barcode = 0 #vertex on which this particle is incoming
        self.nb_flow_list = 0
        self.flows = []
        
        if text:
            self.parse(text, event)
    
    @property
    def pdg_code(self):
        return self.pdg
    
    pid = pdg_code

    @property
    def helicity(self):
        return 9
    
    def parse(self,line=None, event=None):
        """ P 3 -2 0 0 3.0332529367341937e+01 3.0332529367341937e+01 0 21 0 0 -3 1 2 501"""

        data = line.split()
        
        self.barcode = int(data[1]) # 3
        self.pdg = int(data[2])     #-2
        self.px =  float(data[3])   #0
        self.py = float(data[4])    #0
        self.pz = float(data[5])    #30.3
        self.E = float(data[6])     # 30.3
        self.mass = float(data[7])  # 0
        self.status = int(data[8])  # 21
        self.polarization_theta = float(data[9]) #0
        self.polarization_phi = float(data[10]) #0
        self.vertex_barcode = float(data[11]) #-3 vertex on which this particle is incoming
        self.nb_flow_list = int(data[12]) # 1
        self.flows = [(int(data[13+2*i]),int(data[13+2*i+1])) 
                                      for i in range(self.nb_flow_list)] # 2 501
        
        if event:
            event.curr_vertex.add_outcoming(self)
        
    def __str__(self):
        """P 3 -2 0 0 3.0332529367341937e+01 3.0332529367341937e+01 0 21 0 0 -3 1 2 501"""
        
        start = """P %i %i %17.16e %17.16e %17.16e %17.16e %17.16e %i %17.16e %17.16e %i %i %s\n""" %\
         (self.barcode, self.pdg, self.px, self.py, self.pz, self.E, self.mass,
          self.status, self.polarization_theta, self.polarization_phi, 
          self.vertex_barcode, self.nb_flow_list, ' '.join("%i %i" % f for f in self.flows))

        
        return start.replace("%17.16e" % 0, '0')
        

     
        
class HEPMC_Vertex(object):

    def __init__(self, text=None, event=None):

        self.barcode = 0
        self.id = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.ctau = 0
        self.nb_orphan = 0
        self.nb_outgoing = 0
        self.nb_weight = 0
        self.weights = [] 
        self.incoming = []
        self.outcoming = []
        

        if text:
            self.parse(text,event)

    def parse(self, line, event=None):
        """V -8 0 0 0 0 0 0 2 0"""
        
        data = line.split()
        self.barcode = int(data[1])
        self.id = float(data[2])
        self.x = float(data[3])
        self.y = float(data[4])
        self.z = float(data[5])
        self.ctau = float(data[6])
        self.nb_orphan = int(data[7])
        self.nb_outgoing = int(data[8])
        self.nb_weight = int(data[9])
        self.weights = [float(data[10+i]) for i in range(self.nb_weight)]  
        if event:
            event.vertex[self.barcode] = self      

    def add_incoming(self, particle):
        self.incoming.append(particle)
        
    def add_outcoming(self, particle):
        self.outcoming.append(particle)

class HEPMC_Event(object):

    def __init__(self, text=None):
        """The initialization of an empty Event (or one associate to a text file)"""
        #
        self.particles = {} #barcode to object
        self.vertex = {}    #barcode to object

        # First line information (E line)
        self.event_id = 0
        self.nb_interaction = 0
        self.scale = 0.
        self.alphas = 0.
        self.alphaew = 0.
        self.process_id = 0
        self.barcode_vertex =0
        self.nb_vertex = 0
        self.barcode_beam1 = 0
        self.barcode_beam2 = 0
        self.nb_random_state = 0
        self.randoms = []
        self.nb_weight = 0
        self.weights = []

        # not parse container (so far)
        self.N = ''
        self.U = ''
        self.C = ''
        self.H = ''
        self.F = ''

        if text:
            self.parse(text)

    @property
    def wgt(self):
        if self.weights:
            return self.weights[0]
        else:
            return 0.
    @wgt.setter
    def wgt(self, value):
        self.nb_weight = 1
        self.weights = [value]
    
    
    def parse(self, text):

        for line in text.split('\n'):
            if not line:
                continue
            if line[0] == 'P':
                P = HEPMC_Particle(line, self)
                self.add_particle(P)
            elif line[0] == 'V':
                V = HEPMC_Vertex(line, self)
                self.curr_vertex = V
                self.add_vertex(V)
            elif line[0] in ['E', 'N', 'U', 'H','F','C']:
                getattr(self, 'parse_%s' % line[0])(line)
            else:
                self.comment = '%s%s\n' % (self.comment,line) 
        
        # add the information about incoming particle
        for particle in self:
            try:
                self.vertex[particle.vertex_barcode].add_incoming(particle)
            except KeyError:
                if particle.vertex_barcode == 0:
                    continue
                raise

    def parse_E(self,line):
        """E 249 -1 -1.0000000000000000e+00 -1.0000000000000000e+00 -1.0000000000000000e+00 0 0 462 1 2 0 1 8.2247251000000005e-22"""

        data = line.split()
        self.event_id = int(data[1])
        self.nb_interaction = int(data[2])
        self.scale = float(data[3])
        self.alphas = float(data[4])
        self.alphaew = float(data[5])
        self.process_id = int(data[6])
        self.barcode_vertex= int(data[7])
        self.nb_vertex = int(data[8])
        self.barcode_beam1 = int(data[9])
        self.barcode_beam2 = int(data[10])
        self.nb_random_state = int(data[11])
        self.randoms = [float(data[12+i]) for i in range(self.nb_random_state)]
        self.nb_weight =  int(data[12+self.nb_random_state])
        self.weights = [float(data[13+self.nb_random_state+i]) 
                          for i in range(self.nb_weight)]      

    def parse_N(self,line):
        """just keep the information so far"""
        self.N = '%s\n' % line
    def parse_U(self,line):
        self.U = '%s\n' % line
    def parse_H(self,line):
        self.H = '%s\n' % line
    def parse_F(self,line):
        self.F = '%s\n' %  line
    def parse_C(self,line):
        self.C = '%s\n' % line

    def __iter__(self):
        return self.particles.values().__iter__()
    
    #def __next__(self):
    #    
    #    self.particles.__next__()
        
    def add_vertex(self, V):
        self.vertex[V.barcode] = V
        
    def add_particle(self, P):
        self.particles[P.barcode] = P

class HEPMC_EventFile(object):
    
    def __new__(self, path, mode='r', *args, **opt):

        if not path.endswith(".gz"):
            return file.__new__(HEPMC_EventFileNoGzip, path, mode, *args, **opt)
        elif mode == 'r' and not os.path.exists(path) and os.path.exists(path[:-3]):
            return HEPMC_EventFile.__new__(HEPMC_EventFileNoGzip, path[:-3], mode, *args, **opt)
        else:
            try:
                return gzip.GzipFile.__new__(HEPMC_EventFileGzip, path, mode, *args, **opt)
            except IOError, error:
                raise
            except Exception, error:
                if mode == 'r':
                    misc.gunzip(path)
                return file.__new__(HEPMC_EventFileNoGzip, path[:-3], mode, *args, **opt)  
  
    
    def __init__(self, path, mode='r', *args, **opt):
        """open file and read the banner [if in read mode]"""

        self.to_zip = False
        if path.endswith('.gz') and mode == 'w' and\
                                              isinstance(self, HEPMC_EventFileNoGzip):
            path = path[:-3]
            self.to_zip = True # force to zip the file at close() with misc.gzip
        
        self.parsing = True # check if/when we need to parse the event.
        self.eventgroup  = False
        try:
            super(HEPMC_EventFile, self).__init__(path, mode, *args, **opt)
        except IOError:
            if '.gz' in path and isinstance(self, HEPMC_EventFileNoGzip) and\
                mode == 'r' and os.path.exists(path[:-3]):
                super(HEPMC_EventFile, self).__init__(path[:-3], mode, *args, **opt)
            else:
                raise

        self.header = ''
        if mode == 'r':
            line = ''
            while 'HepMC::IO_GenEvent-START_EVENT_LISTING' not in line:

                try:
                    line  = super(HEPMC_EventFile, self).next()
                except StopIteration:
                    self.seek(0)
                    self.header = ''
                    break 
                self.header += line
        self.start_event = ''

    def seek(self, value, fromwhat=0):
        self.start_event = ""
        return super(HEPMC_EventFile, self).seek(value, fromwhat)
        
        
        
    def next(self):
        """get next event"""
        text = self.start_event
        line = ''
        while 1:
            line = super(HEPMC_EventFile, self).next()
            if line.startswith('E'):
                self.start_event = line
                if text:
                    return HEPMC_Event(text)
                else:
                    text += line
                    
            elif line.lstrip().startswith('HepMC::IO_GenEvent-END_EVENT_LISTING'):
                if text:
                    return HEPMC_Event(text)
            elif line.lstrip().startswith('HepMC::IO_GenEvent-START_EVENT_LISTING'):
                text = ''
            else:
                text += line


class HEPMC_EventFileGzip(HEPMC_EventFile, gzip.GzipFile):
    """A way to read/write a gzipped lhef event"""
    
    def tell(self):
        currpos = super(HEPMC_EventFileGzip, self).tell()
        if not currpos:
            currpos = self.size
        return currpos
    
    def getfilesize(self):
        fo = open(self.name, 'rb')
        fo.seek(-4, 2)
        r = fo.read()
        fo.close()
        import struct
        return struct.unpack('<I', r)[0]    
        
class HEPMC_EventFileNoGzip(HEPMC_EventFile, file):
    """A way to read a standard event file"""
    
    def close(self,*args, **opts):
        
        out = super(HEPMC_EventFileNoGzip, self).close(*args, **opts)
        if self.to_zip:
            misc.gzip(self.name)
            
    def getfilesize(self):
        self.seek(0,2)
        return self.tell()
    
    
if "__main__" == __name__:
    path = "/Users/omatt/Documents/eclipse/2.3.3_PY8_install_cmd/PROC_sm_24/Events/run_01/tag_1_pythia8_events.hepmc.gz"
    evts = HEPMC_EventFile(path)
    nb_event = 0
    nb_p = 0
    for event in evts:
        nb_event +=1
        for p in event:
             nb_p+=1
    print nb_event, nb_p
