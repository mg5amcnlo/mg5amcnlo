#!/usr/bin/env python

from __future__ import division

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
###############################################################################
"""
####################################################################
#
#    Routine to decay prodution events in a generic way, 
#    including spin correlation effects
#
#    Ref: S. Frixione, E. Laenen, P. Motylinski, B. R. Webber
#             JHEP 04 (2007) 081
#
#
#####################################################################
"""
import collections
import re
import os
import shutil
import logging
import time
import cmath
import copy
pjoin = os.path.join
from subprocess import Popen, PIPE, STDOUT

os.sys.path.append("../.")
import string
import itertools
#import madgraph.core.base_objects as base_objects
#import madgraph.core.helas_objects as helas_objects
#import madgraph.core.diagram_generation as diagram_generation

import models.import_ufo as import_ufo
#import madgraph.various.process_checks as process_checks
#from time import strftime
#from madgraph.interface.madgraph_interface import MadGraphCmd
import madgraph.interface.master_interface as Cmd
import madgraph.interface.madevent_interface as me_interface
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.files as files
import madgraph.fks.fks_common as fks_common
import aloha
logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr

import random 
import math
from madgraph import MG5DIR, MadGraph5Error
import madgraph.various.misc as misc
#import time
import tests.parallel_tests.test_aloha as test_aloha

class MadSpinError(MadGraph5Error):
    pass

class Event:
    """ class to read an event, record the information, write down the event in the lhe format.
            This class is used both for production and decayed events"""

    def __init__(self, inputfile=None, banner=None):
        """Store the name of the event file """
        self.inputfile=inputfile
        self.particle={}
        self.banner = banner

    def give_momenta(self, map_event=None):
        """ return the set of external momenta of the event, 
                in two different formats:
                p is list momenta, with each momentum a list [E,px,py,pz]
                string is a sting
        """
        
        if not map_event:
            map_event = {}
            for part in range(len(self.particle)):
                map_event[part] = part
                
        p=[]
        string=""
        for id in xrange(len(self.particle)):
            particle = self.particle[map_event[id] + 1]
            if particle["istup"] < 2:
                mom = particle["momentum"]
                p.append(mom)
                string+= '%s %s %s %s \n' % (mom.E, mom.px, mom.py, mom.pz)

        return p, string 
    
    def change_wgt(self, value=None, factor=None):
        
        if value:
            self.wgt = value
        elif factor:
            self.wgt *= factor
            # change the wgt associate to the additional weight
            start, stop = self.rwgt.find('<rwgt>'), self.rwgt.find('</rwgt>')
            if start != -1 != stop :
                pattern = re.compile(r'''<\s*wgt id=[\'\"](?P<id>[^\'\"]+)[\'\"]\s*>\s*(?P<val>[\ded+-.]*)\s*</wgt>''')
                data = pattern.findall(self.rwgt)
                if len(data)==0:
                    print self.rwgt
                try:
                    text = ''.join('   <wgt id=\'%s\'> %+15.7e </wgt>\n' % (pid, float(value) * factor)
                                     for (pid,value) in data) 
                except ValueError, error:
                    raise Exception, 'Event File has unvalid weight. %s' % error
                self.rwgt = self.rwgt[:start] + '<rwgt>\n'+ text + self.rwgt[stop:]          

    def string_event_compact(self):
        """ return a string with the momenta of the event written 
                in an easy readable way
        """
        line=""
        for part in range(1,len(self.particle)+1):
            line+=str(self.particle[part]["pid"])+" "
            line+=str(self.particle[part]["momentum"].px)+" "
            line+=str(self.particle[part]["momentum"].py)+" "
            line+=str(self.particle[part]["momentum"].pz)+" "
            line+=str(self.particle[part]["momentum"].E)+"    " 
            line+=str(self.particle[part]["momentum"].m)+"    " 
            line+="\n"
        return line
    
    def get_tag(self):
        
        initial = []
        final = []
        order = [[],[]]
        for part in self.particle.values():
            pid = part['pid']
            mother1 = part['mothup1']
            mother2 = part['mothup2']
            if 0 == mother1 == mother2:
                initial.append(pid)
                order[0].append(pid)
            else:
                final.append(pid)
                order[1].append(pid)
        initial.sort()
        final.sort()

        return (tuple(initial), tuple(final)), order
 
        
    

    def string_event(self):
        """ return a string with the information of the event written 
                in the lhe format.
        """
        line=self.event_init_line         # This is the <event> line
        line1=' %2d %6d %+13.7e %14.8e %14.8e %14.8e' % \
        (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)
        line+=line1+"\n"
        scales= []
        for item in range(1,len(self.event2mg.keys())+1):
            part=self.event2mg[item]
            if part>0:
                particle_line=self.get_particle_line(self.particle[part])
                if abs(self.particle[part]["istup"]) == 1:
                    if "pt_scale" in self.particle[part]:
                        scales.append(self.particle[part]["pt_scale"])
                    else:
                        scales.append(None)
            else:
                particle_line=self.get_particle_line(self.resonance[part])
            line+=particle_line        
        
        if any(scales):
            sqrts = self.particle[1]["pt_scale"]
            line += "<scales %s></scales>\n" % ' '.join(['pt_clust_%i=\"%s\"' 
                                                        %(i-1,s if s else sqrts)
                                                       for i,s in enumerate(scales)
                                                       if i>1])
        
        if self.diese:
            line += self.diese
        if self.rwgt:
            line += self.rwgt
        line+="</event> \n"
        return line


    def get_particle_line(self,leg):

        line=" %8d %2d %4d %4d %4d %4d %+18.11e %+18.11e %+18.11e %18.11e %18.11e %10.4e %10.4e" \
            % (leg["pid"], leg["istup"],leg["mothup1"],leg["mothup2"],\
               leg["colup1"],leg["colup2"],leg["momentum"].px,leg["momentum"].py,\
                leg["momentum"].pz,leg["momentum"].E, leg["mass"],\
                 0.0,float(leg["helicity"]) )
        line+="\n"
        return line

    def reshuffle_resonances(self,mother):
        """ reset the momentum of each resonance in the production event
                to the sum of momenta of the daughters
        """

        daughters=[]
        for part in self.event2mg.keys():
            index=self.event2mg[part]
            if index>0:
                if self.particle[index]["mothup1"]==mother:
                    daughters.append(index)
            if index<0:
                if self.resonance[index]["mothup1"]==mother:
                    daughters.append(index)

        if len(daughters)!=2:
            logger.info("Got more than 2 (%s) daughters for one particles" % len(daughters))
            logger.info("in one production event (before decay)")


        if daughters[0]>0:
            momentum_mother=self.particle[daughters[0]]["momentum"].copy()
        else:
            momentum_mother=self.resonance[daughters[0]]["momentum"].copy()

#       there might be more than 2 daughters, add all their momentum to get the momentum of the mother
        for index in range(1,len(daughters)):
            if daughters[index]>0:
                momentum_mother=momentum_mother.add(self.particle[daughters[index]]["momentum"])
            else:
                momentum_mother=momentum_mother.add(self.resonance[daughters[index]]["momentum"])

        res=self.event2mg[mother]
        del self.resonance[res]["momentum"]
        self.resonance[res]["momentum"]=momentum_mother.copy()

#     recurrence:
        if self.resonance[res]["mothup1"]>2:
            self.reshuffle_resonances(self.resonance[res]["mothup1"])             


    def reset_resonances(self):
        """ re-evaluate the momentum of each resonance, based on the momenta
                of the external particles 
        """

        mothers=[]
        for part in self.particle.keys():
            if self.particle[part]["mothup1"]>2 and \
                    self.particle[part]["mothup1"] not in mothers :
                mothers.append(self.particle[part]["mothup1"])
                self.reshuffle_resonances(self.particle[part]["mothup1"]) 

    def assign_scale_line(self, line):
        """read the line corresponding to global event line
        format of the line is:
        Nexternal IEVENT WEIGHT SCALE AEW AS
        """
        line = line.replace('d','e').replace('D','e')
        inputs = line.split()
        assert len(inputs) == 6
        self.nexternal=int(inputs[0])
        self.ievent=int(inputs[1])
        self.wgt=float(inputs[2])
        self.scale=float(inputs[3])
        self.aqed=float(inputs[4])
        self.aqcd=float(inputs[5])        
        
        

    def get_next_event(self):
        """ read next event in the lhe event file """
        line_type = 'none' # support type: init / event / rwgt
        self.diese = ''
        for line in self.inputfile:
            line = line.lower()
            if line=="":
                continue 
            # Find special tag in the line
            if line[0]=="#":
                self.diese+=line
                continue
            if '<event' in line:
                #start new_event
                #Get the right attributes (e.g. <event id='123' npNLO='-1'>)
                self.event_init_line=line.lstrip().replace('nplo','npLO').replace('npnlo','npNLO')
                line_type = 'init'
                continue
            elif '<rwgt>' in line:
                #re-weighting information block
                line_type = 'rwgt'
                #No Continue! need to keep track of this line
            elif '</event>' in line:
                if not self.particle:
                    continue
                self.shat=self.particle[1]["momentum"].dot(self.particle[2]["momentum"])
                return 1
            elif line_type == 'rwgt' and 'wgt' in line:
                # force to continue to be in rwgt line up to </rwgt>
                line_type = 'rwgt'
            elif "pt_clust_" in line:
                line_type="clusteringv3" 
            
            elif '<' in line:
                line_type = 'other_block'
            
            
            if line_type == 'none':
                continue
            elif line_type == 'other_block':
                self.diese += line
            # read the line and assign the date accordingly                
            elif line_type == 'init':
                line_type = 'event'
                self.assign_scale_line(line)         
                # initialize some local variable
                index_prod=0
                index_external=0
                index_resonance=0
                self.particle={}
                self.resonance={}
                self.max_col=500
                self.diese=""
                self.rwgt=""
                self.event2mg={} # dict. event-like label <-> "madgraph-like" label
                                                            # in "madgraph-like", resonances are labeled -1, -2, ...
            elif line_type == 'rwgt': #special aMC@NLO information
                self.rwgt += line
                if '</rwgt>' in line:
                    line_type = 'event'
            elif line_type == 'event':
                index_prod+=1
                line=line.replace("\n","")
                line = line.replace('d','e').replace('D','e')
                inputs=line.split()
                pid=int(inputs[0])
                istup=int(inputs[1])
                mothup1=int(inputs[2])
                mothup2=int(inputs[3])
                colup1=int(inputs[4])
                if colup1>self.max_col:
                    self.max_col=colup1 
                colup2=int(inputs[5])
                if colup2>self.max_col:
                    self.max_col=colup2 
                mom=momentum(float(inputs[9]),float(inputs[6]),float(inputs[7]),float(inputs[8]))
                mass=float(inputs[10])
                helicity=float(inputs[12])
                if abs(istup)==1:
                    index_external+=1
                    self.event2mg[index_prod]=index_external
                    self.particle[index_external]={"pid":pid,"istup":istup,"mothup1":mothup1,\
                    "mothup2":mothup2,"colup1":colup1,"colup2":colup2,"momentum":mom,"mass":mass,"helicity":helicity}
                elif istup==2:
                    index_resonance=index_resonance-1
                    self.event2mg[index_prod]=index_resonance
                    self.resonance[index_resonance]={"pid":pid,"istup":istup,"mothup1":mothup1,\
                    "mothup2":mothup2,"colup1":colup1,"colup2":colup2,"momentum":mom,"mass":mass,"helicity":helicity}
                else: 
                    logger.warning('unknown status in lhe file')
            elif line_type == "clusteringv3":
                scales = re.findall(r"""pt_clust_(\d+)=\"([e\+\-.\d]+)\"""", line)
                scales = sorted(scales, key= lambda x: -1*int(x[0]))
                for index in range(1,len(self.particle)+1):
                    if self.particle[index]["istup"] == 1:
                        self.particle[index]["pt_scale"] = scales.pop()[1]
                if not self.banner:
                    self.particle[1]["pt_scale"] = self.particle[1]["momentum"].E + self.particle[2]["momentum"].E
                else:
                    self.particle[1]["pt_scale"] = float(self.banner.get('run_card', 'ebeam1'))+float(self.banner.get('run_card', 'ebeam2'))
                
        return "no_event"

class pid2label(dict):
    """ dico pid:label for a given model"""

    def __init__(self,model):
        
        for particle in model["particles"]:
            self[particle["pdg_code"]]=particle["name"]
            self[-particle["pdg_code"]]=particle["antiname"]

class pid2color(dict):
    """ dico pid:color rep. for a given model (ex: 5:-3 )"""

    def __init__(self,model):

        for particle in model["particles"]:
            self[particle["pdg_code"]]=particle["color"]
            if particle["color"] not in [1,8]:
                self[-particle["pdg_code"]]=-particle["color"]
            else:
                self[-particle["pdg_code"]]=particle["color"]
class label2pid(dict):
    """ dico label:pid for a given model"""

    def __init__(self,model):
        
        for particle in model["particles"]:
            self[particle["name"]]=particle.get_pdg_code()
            self[particle["antiname"]]=-particle.get_pdg_code()
            if particle['self_antipart']:
                self[particle["name"]]=abs(self[particle["name"]])
                self[particle["antiname"]]=abs(self[particle["antiname"]])

class dc_branch_from_me(dict):
    """ A dictionary to record information necessary to decay particles 
            { -1 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },    
                -2 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },
                ....
            }
    """

    def __init__(self, process):
        """ """
        self.model = process.get('model')
        
        self["tree"]={}
        self.nexternal = 0
        self.nb_decays = 1
        
        #define a function to allow recursion.
        def add_decay(proc, propa_id=-1):
            # see what need to be decayed
            to_decay = {}
            for dec in proc.get('decay_chains'):
                pid =  dec.get('legs')[0].get('id')
                if pid  in to_decay:
                    to_decay[pid].append(dec)
                else:
                    to_decay[pid] = [dec]
            #done
            self['tree'][propa_id] = {'nbody': len(proc.get('legs'))-1,\
                                      'label':proc.get('legs')[0].get('id')}
            # loop over the child
            child_propa_id = propa_id
            for c_nb,leg in enumerate(proc.get('legs')):
                if c_nb == 0:
                    continue
                self["tree"][propa_id]["d%s" % c_nb] = {}
                c_pid = leg.get('id')
                self["tree"][propa_id]["d%s" % c_nb]["label"] = c_pid
                self["tree"][propa_id]["d%s" % c_nb]["labels"] = [c_pid]
                if c_pid in to_decay:
                    child_propa_id -= 1
                    self["tree"][propa_id]["d%s" % c_nb]["index"] = child_propa_id
                    self.nb_decays += 1
                    child_propa_id = add_decay(to_decay[c_pid].pop(), child_propa_id)
                else:
                    self.nexternal += 1
                    self["tree"][propa_id]["d%s" % c_nb]["index"] = self.nexternal
            return child_propa_id
        
        # launch the recursive loop
        add_decay(process)

    def generate_momenta(self,mom_init,ran, pid2width,pid2mass,BW_cut,E_collider, sol_nb=None):
        """Generate the momenta in each decay branch 
             If ran=1: the generation is random, with 
                                     a. p^2 of each resonance generated according to a BW distribution 
                                     b. cos(theta) and phi (angles in the rest frame of the decaying particle)
                                            are generated according to a flat distribution (no grid)
                                 the phase-space weight is also return (up to an overall normalization)
                                 since it is needed in the unweighting procedure

             If ran=0: evaluate the momenta based on the previously-generated p^2, cos(theta) 
                                 and phi in each splitting.    
                                 This is used in the reshuffling phase (e.g. when we give a mass to gluons 
                                 in the decay chain )
        """
        
        index2mom={}
#      pid2mom={}    # a dict { pid : {"status":status, "momentum":momentum}    }

        assert isinstance(mom_init, momentum)
        index2mom[-1] = {}
        index2mom[-1]["momentum"] = mom_init 
        
        if index2mom[-1]['momentum'].m < 1e-3:
            logger.warning('Decaying particle with m< 1e-3 GeV in generate_momenta')
        index2mom[-1]["pid"] = self['tree'][-1]["label"]
        index2mom[-1]["status"] = 2
        weight=1.0
        for res in range(-1,-self.nb_decays-1,-1):
            tree =  self["tree"][res]
            #     Here mA^2 has to be set to p^2:
            # 
            #     IF res=-1:
            #         p^2 has been either fixed to the value in the 
            #         production lhe event, or generated according to a    Breit-Wigner distr. 
            #         during the reshuffling phase of the production event
            #         -> we just need to read the value here
            #     IF res<-1:
            #         p^2 has been generated during the previous iteration of this loop 
            #         -> we just need to read the value here

            mA=index2mom[res]["momentum"].m
            if mA < 1.0e-3:
                logger.debug('Warning: decaying parting with m<1 MeV in generate_momenta ')

            mass_sum = mA
            all_mass = []
            for i in range(tree["nbody"]):
                tag =  "d%s" % (i+1)
                d = tree[tag]["index"]
                # For the daughters, the mass is either generate (intermediate leg + BW mode on)
                # or set to the pole mass (external leg or BW mode off)
                # If ran=0, just read the value from the previous generation of momenta 
                #(this is used for reshuffling purposes) 
                if d>0 or not BW_cut :
                    m = pid2mass(tree[tag]["label"])
                elif ran==0:    # reshuffling phase
                    m= tree[tag]["mass"]
                else:
                    pid=tree[tag]["label"]
                    # NOTE: here pole and width are normalized by 4.0*mB**2,
                    # Just a convention
                    pole=0.25         #pid2mass[pid]**2/mA**2
                    w=pid2width(pid)
                    mpole=pid2mass(pid)
                    width=pid2width(pid)*pid2mass(pid)/(4.0*pid2mass(pid)**2)     #/mA**2

                    m_min=max(mpole-BW_cut*w, 0.5)
                    m_max=mpole+BW_cut*w 
                    if E_collider>0: m_max=min(m_max,0.99*E_collider)

                    zmin=math.atan(m_min**2/w/mpole-mpole/w)/width
                    zmax=math.atan(m_max**2/w/mpole-mpole/w)/width

                    m, jac=self.transpole(pole,width, zmin,zmax)
                    m = math.sqrt(m * 4.0 * mpole**2)
                    # record the mass for the reshuffling phase, 
                    # in case the point passes the reweighting creteria
                    tree[tag]["mass"] = m
                    #update the weigth of the phase-space point
                    weight=weight*jac
                # for checking conservation of energy
                mass_sum -= m
                all_mass.append(m)
                if mass_sum < 0:
                    logger.debug('mA<mB+mC in generate_momenta')
                    logger.debug('mA = %s' % mA)
                    return 0, 0, 0 # If that happens, throw away the DC phase-space point ...
                    # I don't expect this to be inefficient, since there is a BW cut                                    

            if tree["nbody"] > 2:
                raise Exception, 'Phase Space generator not yet ready for 3 body decay'

            if ran==1:
                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, all_mass[0],all_mass[1])
#             record the angles for the reshuffling phase, 
#             in case the point passes the reweighting creteria
                tree["costh"]=decay_mom.costh
                tree["sinth"]=decay_mom.sinth
                tree["cosphi"]=decay_mom.cosphi
                tree["sinphi"]=decay_mom.sinphi
            else:
#             we are in the reshuffling phase, 
#             so we read the angles that have been stored from the 
#             previous phase-space point generation
                costh=self["tree"][res]["costh"]
                sinth=self["tree"][res]["sinth"]
                cosphi=self["tree"][res]["cosphi"]
                sinphi=self["tree"][res]["sinphi"]
                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, all_mass[0],all_mass[1],\
                                 costh_val=costh, sinth_val=sinth, cosphi_val=cosphi, \
                                 sinphi_val=sinphi)

            # record the momenta for later use
            index2mom[self["tree"][res]["d1"]["index"]]={}
            index2mom[self["tree"][res]["d1"]["index"]]["momentum"]=decay_mom.momd1
            if sol_nb is None:   
                sol_nb = random.randint(0,len(self["tree"][res]["d1"]["labels"])-1)
#                print self["tree"][res]["d1"]["labels"]
#                print '584 get sol_nb', sol_nb,'=>',self["tree"][res]["d1"]["labels"][sol_nb],self["tree"][res]["d2"]["labels"][sol_nb]
#            else:
#                print sol_nb, sol_nb is None, 
#                print 'take back', sol_nb,'=>',self["tree"][res]["d1"]["labels"][sol_nb],self["tree"][res]["d2"]["labels"][sol_nb]
            index2mom[self["tree"][res]["d1"]["index"]]["pid"]=self["tree"]\
                                                   [res]["d1"]["labels"][sol_nb]

            index2mom[self["tree"][res]["d2"]["index"]]={}
            index2mom[self["tree"][res]["d2"]["index"]]["momentum"]=decay_mom.momd2
            index2mom[self["tree"][res]["d2"]["index"]]["pid"]=self["tree"]\
                                                   [res]["d2"]["labels"][sol_nb]

            if (self["tree"][res]["d1"]["index"]>0):
                index2mom[self["tree"][res]["d1"]["index"]]["status"]=1
            else:
                index2mom[self["tree"][res]["d1"]["index"]]["status"]=2
            if (self["tree"][res]["d2"]["index"]>0):
                index2mom[self["tree"][res]["d2"]["index"]]["status"]=1
            else:
                index2mom[self["tree"][res]["d2"]["index"]]["status"]=2

        return index2mom, weight, sol_nb
        
    def transpole(self,pole,width, zmin, zmax):

        """ routine for the generation of a p^2 according to 
            a Breit Wigner distribution
            the generation window is 
            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
        """

        z=zmin+(zmax-zmin)*random.random()
        y = pole+width*math.tan(width*z)

        jac=(width/math.cos(width*z))**2*(zmax-zmin)
        return y, jac
    
    def add_decay_ids(self, proc_list):
        """ """
 
        #define a function to allow recursion.
        def add_decay(proc, propa_id=-1):
            # see what need to be decayed
            to_decay = {}
            for dec in proc.get('decay_chains'):
                pid =  dec.get('legs')[0].get('id')
                if pid  in to_decay:
                    to_decay[pid].append(dec)
                else:
                    to_decay[pid] = [dec]

            # loop over the child
            for c_nb,leg in enumerate(proc.get('legs')):
                if c_nb == 0:
                    continue
#                self["tree"][propa_id]["d%s" % c_nb] = {}
                c_pid = leg.get('id')
                self["tree"][propa_id]["d%s" % c_nb]["labels"].append(c_pid)
                if c_pid in to_decay:
                    add_decay(to_decay[c_pid].pop(), propa_id-1)
        
        # launch the recursive loop
        for proc in proc_list:
            add_decay(proc)        

class momentum:
    """A class to handel 4-vectors and the associated operations """
    def __init__(self,E,px,py,pz):
        self.px=px
        self.py=py
        self.pz=pz
        self.E=E
        self.mod2=px**2+py**2+pz**2
        self.sq=E**2-self.mod2
        control = E**2+self.mod2
        if not control:
            self.m = 0
        elif self.sq/control < 1e-8:
            self.m=0.0
        else:
            self.m=math.sqrt(self.sq)
        

    def dot3(self,q):
        """ return |p|^2 (spatial components only) """
        return self.px*q.px+self.py*q.py+self.pz*q.pz

    def dot(self,q):
        """ Minkowski inner product """
        return self.E*q.E-self.px*q.px-self.py*q.py-self.pz*q.pz

    def subtract(self,q):
        tot=momentum(self.E-q.E,self.px-q.px,self.py-q.py,self.pz-q.pz)
        return tot

    def add(self,q):
        tot=momentum(self.E+q.E,self.px+q.px,self.py+q.py,self.pz+q.pz)
        return tot
    
    __add__ = add

    def nice_string(self):
        return str(self.E)+" "+str(self.px)+" "+str(self.py)+" "+str(self.pz)

    __str__ = nice_string
    def boost(self, q):
        """ boost a vector from a frame where q is at rest to a frame where q is given 
                This routine has been taken from HELAS
        """
        qq = q.mod2

        if (qq > 1E-10*abs(q.E)):
            pq=self.dot3(q)
            m=q.m
            #if (abs(m-self.mA)>1e-6): print "warning: wrong mass"
            lf=((q.E-m)*pq/qq+self.E)/m
            pboost=momentum((self.E*q.E+pq)/m, self.px+q.px*lf,\
                            self.py+q.py*lf,self.pz+q.pz*lf)            
        else:
            pboost=momentum(self.E,self.px,self.py,self.pz)

        return pboost 

    def copy(self):
        copy_mom=momentum(self.E,self.px,self.py,self.pz)
        return copy_mom

    def invrot(self,q):
        # inverse of the "rot" operation 

        ppE=self.E
        qt2 = (q.px)**2 + (q.py)**2

        if(qt2==0.0):
            if ( q.pz>0 ):
                ppx = self.px
                ppy = self.py
                ppz = self.pz
            else:
                ppx = -self.px
                ppy = -self.py
                ppz = -self.pz
        else:
            qq = math.sqrt(qt2+q.pz**2)
            qt=math.sqrt(qt2)
            ppy=-q.py/qt*self.px+q.px/qt*self.py
            if (q.pz==0):
                ppx=-qq/qt*self.pz
                if (q.py!=0):
                    ppz=(self.py-q.py*q.pz/qq/qt-q.px/qt*ppy)*qq/q.py
                else:
                    ppz=(self.px-q.px*q.pz/qq/qt*ppx+q.py/qt*ppy)*qq/q.px
            else:
                if (q.py!=0):
                    ppz=(qt**2*self.py+q.py*q.pz*self.pz-q.px*qt*ppy)/qq/q.py
                else:
                    ppz=(q.px*self.px+q.pz*self.pz)/qq
                ppx=(-self.pz+q.pz/qq*ppz)*qq/qt
        pp=momentum(ppE,ppx,ppy,ppz)
        return pp 


    def rot(self, q):
        """ rotate the spatial components of the vector from a frame where q is 
                aligned with the z axis to a frame where the direction of q is specified 
                as an argument
                Taken from HELAS
        """
        protE =self.E
        qt2 = (q.px)**2 + (q.py)**2

        if(qt2==0.0):
            if ( q.pz>0 ):
                protx = self.px
                proty = self.py
                protz = self.pz
            else:
                protx = -self.px
                proty = -self.py
                protz = -self.pz
        else:
            qq = math.sqrt(qt2+q.pz**2)
            qt = math.sqrt(qt2)
            protx = q.px*q.pz/qq/qt*self.px -q.py/qt*self.py +q.px/qq*self.pz
            proty = q.py*q.pz/qq/qt*self.px +q.px/qt*self.py +q.py/qq*self.pz
            protz = -qt/qq*self.px + q.pz/qq*self.pz

        prot=momentum(protE,protx,proty,protz)
        return prot


class generate_2body_decay:
    """generate momenta for a generic A > B + C decay    """

    def __init__(self,p,mA,mB,mC, costh_val=None, sinth_val=None, cosphi_val=None, sinphi_val=None):
        """ Generate the momentum of B and C in the decay A -> B+C
                If the angles are given, use them to reconstruct the momenta of B, C
                in the rest fram of A. 
                If the routine is called without (costh_val, ...), then generate 
                cos(theta) and phi randomly (flat distr.) in the rest frame of A
                Finally, boost the momenta of B and C in the frame where A has 
                momentum p
        """        

        self.mA=mA
        self.mB=mB
        self.mC=mC

        pmod=self.lambda_fct()/(2.0 * self.mA)
        if not costh_val:
            costh=2.0*random.random()-1.0
            sinth=math.sqrt(1-costh**2)
        else:
            costh=costh_val
            sinth=sinth_val

        if not cosphi_val:
            phi=random.random()*2.0*math.pi
            sinphi=math.sin(phi)
            cosphi=math.cos(phi)
        else:
            sinphi=sinphi_val
            cosphi=cosphi_val

        energyB=math.sqrt(pmod**2+mB**2)
        energyC=math.sqrt(pmod**2+mC**2)
        pBrest=momentum(energyB, pmod*cosphi*sinth,pmod*sinphi*sinth, pmod*costh)
        pCrest=momentum(energyC,-pmod*cosphi*sinth,-pmod*sinphi*sinth, -pmod*costh)
        self.momd1=pBrest.boost(p)
        self.momd2=pCrest.boost(p)

#     record costh and phi for later use
        self.costh=costh
        self.sinth=sinth
        self.cosphi=cosphi
        self.sinphi=sinphi

    def lambda_fct(self):
        """ The usual lambda function involved in 2-body decay """
        lam=self.mA**4+self.mB**4+self.mC**4
        lam=lam-2.0*self.mA**2*self.mB**2-2.0*self.mA**2*self.mC**2\
                    -2.0*self.mC**2*self.mB**2
#        if lam<0:
            #print self.mA
            #print self.mB
            #print self.mC
        return math.sqrt(lam)



class production_topo(dict):
    """ A dictionnary to record information about a given topology of a production event 

                self["branchings"] is a list of the branchings defining the topology (see class branching)
                self["get_mass2"] is a dictionnary {index -> mass**2 of the corresponding particle} 
                self["get_momentum"] is a dictionnary {index -> momentum of the corresponding particle} 
                self["get_id"] is a dictionnary {index -> pid the corresponding particle} 

            Note: index= "madgraph-like" numerotation of the particles
    """

    def __init__(self, production, options):
        """ Initialise the dictionaries+list used later on to record the information
                about the topology of a production event.
                Note that self["branchings"] is a list, 
                as it makes it easier to scan the topology in the ascendent order
        """
        self["branchings"]=[]
        self["get_mass2"]={}
        self["get_momentum"]={}
        self["get_id"]={}
        self.production = production
        self.options = options

    def add_one_branching(self,index_propa, index_d1,index_d2,type_propa):
        """ add the information of one splitting in the topology """
        branch=branching(index_propa, index_d1,index_d2,type_propa)
        self["branchings"].append(branch)


    def print_topo(self):
        """Print the structure of the topology    """
        for branch in self["branchings"]:
            d1=branch["index_d1"]
            d2=branch["index_d2"]
            propa=branch["index_propa"]
            line=str(propa)+" > "
            line+=str(d1)+" + "
            line+=str(d2)+" ,    type="
            line+=branch["type"]
            print line

class AllMatrixElement(dict):
    """Object containing all the production topologies required for event to decay.
       This contains the routine to add a production topologies if needed.
    """
    
    def __init__(self, banner, options, decay_ids, model):
        
        dict.__init__(self)
        self.banner = banner
        self.options = options
        self.decay_ids = set([abs(id) for id in decay_ids])
        self.has_particles_ambiguity = False
        self.model = model

        
    def add(self, topologies, keys):
        """Adding one element to the list of production_topo"""
        
        for key in keys:
            self[key] = topologies

    def get_br(self, proc):
        # get the branching ratio associated to a process
       
        br = 1
        ids = collections.defaultdict(list) #check for identical decay
        for decay in proc.get('decay_chains'):
            init, final = decay.get_initial_final_ids()
            lhaid = tuple([len(final)] + [x for x in final])
            ids[init[0]].append(decay)
            if init[0] in self.banner.param_card['decay'].decay_table:
                br *= self.banner.param_card['decay'].decay_table[init[0]].get(lhaid).value
                br *= self.get_br(decay)
            elif -init[0] in self.banner.param_card['decay'].decay_table:
                init = -init[0]
                lhaid=[x if self.model.get_particle(x)['self_antipart'] else -x
                       for x in final]
                lhaid.sort()
                lhaid = tuple([len(final)] + lhaid)
                br *= self.banner.param_card['decay'].decay_table[init].get(lhaid).value
                br *= self.get_br(decay)
            elif init[0] not in self.decay_ids and -init[0] not in self.decay_ids:
                logger.warning("No Branching ratio applied for %s. Please check if this is expected" % init[0])
                br *= self.get_br(decay)
            else:
                raise MadGraph5Error,"No valid decay for %s. No 2 body decay for that particle. (three body are not supported by MadSpin)" % init[0]

                

        for decays in ids.values():
            if len(decays) == 1:
                continue
            br *= math.factorial(len(decays))
            while decays:
                nb=1
                curr = decays.pop()
                while 1:
                    try:
                        decays.remove(curr)
                        nb+=1
                    except ValueError:
                        break
                br /= math.factorial(nb)
                
        return br

            
    def add_decay(self, proc_list, me_path):
        """ adding a decay to the possibility
            me_path is the path of the fortran directory
            br is the associate branching ratio
            finals is the list of final states equivalent to this ME
            matrix_element is the MG5 matrix element
        """
        
        # Usefull debug code tell the status of the various imported decay
        text = ''
        data = []
        for key, prod in self.items():
            if prod in data:
                continue
            data.append(prod)
            br = prod['total_br']
            if br:
                text += '%s %s %s\n' %(key, os.path.basename(prod['path']), br)

        
        
        if  not isinstance(proc_list, list):
            proc_list = proc_list.get('processes')
            
        tag = proc_list[0].get_initial_final_ids()        
        # process with decay not compatible with tag [process under consideration]
        # or with process splitting different for process and decay.
        to_postpose = [proc for proc in proc_list 
                     if id(self[tag]) != id(self[proc.get_initial_final_ids()])]
                         
        finals = []
        for proc in proc_list:
            succeed = True # check if the decay is compatible with the process
                           #under consideration.
            tmp = []
            for dproc in proc.get('decay_chains'):
                pid = dproc.get('legs')[0].get('id')
                # check that the pid correspond -> if not postpone the process
                # to be added in a second step (happen due to symmetry)
                if pid not in tag[1]:
                    to_postpose.append(proc)
                    succeed= False
                    break
                tmp.append((pid,dproc.get_final_ids_after_decay()))
            if succeed and tmp not in finals:
                finals.append(tmp)
        
        # Treat Not compatible decay.        
        if to_postpose:
            self.add_decay(to_postpose, me_path)
        
        # 
        # Now that the various final states are computed, we can add the 
        # associated decay. First computing the branching ratio and then
        # decaying topology
        me = proc_list[0] # all the other are symmetric -> no need to keep those        
        decay_tags = [d.shell_string(pdg_order=True) for d in me['decay_chains']]
        
        #avoid duplicate
        if  any(tuple(decay_tags)==t['decay_tag'] for t in self[tag]['decays']):  
            return   
        
        # the decay:
        out = {'path': me_path, 
               'matrix_element': me, 
               'br': len(finals) * self.get_br(proc),
               'finals': finals, 
               'base_order':[l.get('id') for l in me.get_legs_with_decays()] ,
               'decay_struct':self.get_full_process_structure(proc_list),
               'decay_tag': tuple(decay_tags)}

        # adding it to the current object
        self[tag]['decays'].append(out)
        self[tag]['total_br'] += out['br']
        # update the particle decaying in the process
        decaying = [m.get('legs')[0].get('id') for m in me.get('decay_chains')]
        decaying.sort()
        self[tag]['decaying'] = tuple(decaying)
                
        # sanity check
        assert self[tag]['total_br'] <= 1.01, "wrong BR for %s: %s " % (tag,self[tag]['total_br'])


        
             
    
    def get_full_process_structure(self, me_list):
        """ return a string with the definition of the process fully decayed
                and also a list of dc_branch objects with all infomation about the topology 
                of each decay branch
        """

        me = me_list[0]
        
        decay_struct = {}
        to_decay = collections.defaultdict(list)
        
        for i, proc in enumerate(me.get('decay_chains')):
            pid =  proc.get('legs')[0].get('id')
            to_decay[pid].append((i,proc))
                  
                
        for leg in me.get('legs'):
            pid =  leg.get('id')
            nb = leg.get('number')
            if pid in to_decay:
                i, proc = to_decay[pid].pop()
                decay_struct[nb] = dc_branch_from_me(proc)
                identical = [me.get('decay_chains')[i] for me in me_list[1:]]
                decay_struct[nb].add_decay_ids(identical)
            
        return decay_struct

    def get_topologies(self, matrix_element):
        """Extraction of the phase-space self.topologies from mg5 matrix elements 
             This is used for the production matrix element only.

             the routine is essentially equivalent to    write_configs_file_from_diagrams
             except that I don't write the topology in a file, 
             I record it in an object production_topo (the class is defined above in this file)
        """

        # Extract number of external particles
        ( nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        del nexternal
        preconfigs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in preconfigs]
        configs=[[c[1]] for c in preconfigs]
        model = matrix_element.get('processes')[0].get('model')


        topologies ={}    # dictionnary {mapconfig number -> production_topology}
                                    # this is the object to be returned at the end of this routine

        s_and_t_channels = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
          for config in configs if [d for d in config if d][0].\
                                             get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0
    
        # Number of subprocesses
        #    nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any([vert > minvert for vert in
                    [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
                    # Only 3-vertices allowed in configs.inc
                    continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                    if h:
                            # get_s_and_t_channels gives vertices starting from
                            # final state external particles and working inwards
                            stchannels.append(h.get('amplitudes')[0].\
                                              get_s_and_t_channels(ninitial, model, new_pdg))
                    else:
                            stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                                             tchannels])



            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                    empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                    # Reorganize s-channel vertices to get a list of all
                    # subprocesses for each vertex
                    schannels = zip(*[s for s,t in stchannels])
            else:
                    schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                    # Write out tchannels only if there are any non-trivial ones
                    allchannels = schannels + tchannels

# Write out propagators for s-channel and t-channel vertices

#         use the AMP2 index to label the self.topologies
            tag_topo=mapconfigs[iconfig]
            topologies[tag_topo]=production_topo(topologies, self.options)

            for verts in allchannels:
                    if verts in schannels:
                            vert = [v for v in verts if v][0]
                    else:
                            vert = verts
                    daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                    last_leg = vert.get('legs')[-1]


                    if verts in schannels:
                            type_propa="s"
                    elif verts in tchannels[:-1]:
                            type_propa="t"


                    if (type_propa):
                        topologies[tag_topo].add_one_branching(last_leg.get('number'),\
                         daughters[0],daughters[1],type_propa)

        return topologies
   
    def get_decay_from_tag(self, production_tag, decay_tag):
        for decay in self[production_tag]['decays']:
            if decay['decay_tag']==decay_tag: return decay

        msg = 'Unable to retrieve decay from decay_tag\n%s\n%s' %(production_tag, decay_tag)
        raise Exception, msg
 
    def get_random_decay(self, production_tag,first=[]):
        """select randomly a decay channel"""
        
        a = random.random() * self[production_tag]['total_br']
        #print 'total', self[production_tag]['total_br']
        if __debug__:
            if not first:
                sum = 0
                for decay in self[production_tag]['decays']:
                    sum += decay['br']
                assert sum == self[production_tag]['total_br']
                first.append(1)
                
        sum = 0
        for decay in self[production_tag]['decays']:
            sum += decay['br']
            if a < sum:
                return decay

            
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        
        for me in matrix_elements:
            skip = [] # due to particles/anti-particles some me need to be add
                      # as a separate matrix element in the instance.
            topo = self.get_topologies(me)
            # get the orignal order:
            initial = []
            final = [l.get('id') for l in me.get('processes')[0].get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
            decaying_base = [id for id in final if abs(id) in self.decay_ids]
            decaying_base.sort()
            topo['base_order'] = (initial , final)
#            topo['matrix_element'] = me
            tags = []
            topo['tag2order'] = {}
             
            for proc in me.get('processes'): # set of processes accounted by the me
                initial = []
                final = [l.get('id') for l in proc.get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
                decaying = [id for id in final if abs(id) in self.decay_ids]
                decaying.sort()
                if decaying != decaying_base:
                    skip.append(proc)
                    continue
                topo['decaying'] = ()
                tags.append(proc.get_initial_final_ids())
                topo['tag2order'][tags[-1]] = (initial , final)
            
            if tags[0] not in self:
                self.add(topo, tags)  # mens self[tag]=topo for each tag in tags
            topo['path'] = pjoin(path, 'SubProcesses', 
                                  'P%s' % me.get('processes')[0].shell_string())
            topo['decays'] = []
            topo['total_br'] = 0 
        
            if skip:
                self.add_me_symmetric(skip, topo)
    
            
    def add_me_symmetric(self, process_list, topo):
        """ """
        self.has_particles_ambiguity = True
        skip = [] # due to particles/anti-particles some may need to be add
                  # as a separate matrix element in the instance.
        
        old_topo = topo
        topo = dict(topo) #change the main pointer
        topo['decays'] = []   # unlink information which need to be different.
        topo['total_br'] = 0  #
        topo['tag2order'] = {}
        topo['decaying'] = ()
        for key in topo.keys():
            if isinstance(key, int):
                topo[key] = copy.copy(topo[key])
        
        assert id(old_topo) != id(topo)
        assert id(topo['decays']) != id(old_topo['decays'])
        tags = []
        for i, proc in enumerate(process_list):
            initial = []
            final = [l.get('id') for l in proc.get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
            decaying = [pid for pid in final if abs(pid) in self.decay_ids]
            decaying.sort()
            if i == 0:
                decaying_base = decaying
            if decaying != decaying_base:
                skip.append(proc)
                continue
            
            tags.append(proc.get_initial_final_ids())
            topo['tag2order'][tags[-1]] = (initial , final)
            
            
        if tags[0] not in self:
            self.add(topo, tags)
            for key in topo.keys():
                if isinstance(key, int):     
                    topo[key].production = self[tags[0]]   
        if skip:
            self.add_me_symmetric(skip, topo)
        
                

class branching(dict):
    """ A dictionnary to record information about a given branching in a production event
            self["type"] is the type of the branching , either "s" for s-channel or "t" for t-channel
            self["invariant"] is a real number with the value of p^2 associated with the branching
            self["index_d1"] is the mg index of the first daughter
            self["index_d2"] is the mg index of the second daughter
            self["index_propa"] is the mg index of the propagator
    """

    def __init__(self, index_propa, index_d1,index_d2, s_or_t):
        self["index_d1"]=index_d1
        self["index_d2"]=index_d2
        self["index_propa"]=index_propa
        self["type"]=s_or_t


class width_estimate(object):
    """All methods used to calculate branching fractions"""

    def __init__(self,resonances,path_me, banner, model, pid2label):

        self.resonances=resonances
        self.path_me=path_me
        self.pid2label = pid2label
        self.label2pid = self.pid2label 
        self.model = model
        #print self.model
        self.banner = banner
        #self.model= 

    def update_branch(self,branches,to_add):
        """ complete the definition of the branch by appending each element of to_add"""
        newbranches={}

        for item1 in branches.keys():
            for item2 in to_add.keys():
                tag=item1+item2
                newbranches[tag]={}
                newbranches[tag]['config']=branches[item1]['config']+to_add[item2]['config']
                newbranches[tag]['br']=branches[item1]['br']*to_add[item2]['br']

        return newbranches

    def extract_br(self, decay_processes, mgcmd):
        """Find a way to get the branching ratio. (depending of the model/cards)"""

        # calculate which br are interesting to compute. 
        to_decay = set(decay_processes.keys())
        for decays in decay_processes.values():
            for decay in decays:
                if ',' in decay:
                    to_decay.update(set([l.split('>')[0].strip()
                                                    for l in decay.replace('(','').replace(')','').split(',')]))

        # Maybe the branching fractions are already given in the banner:
        self.extract_br_from_banner(self.banner)
        to_decay = list(to_decay)
        for part in to_decay[:]:
            if part in mgcmd._multiparticles:
                to_decay += [self.pid2label[id] for id in mgcmd._multiparticles[part]]
                to_decay.remove(part)
        to_decay = list(set([p for p in to_decay if not p in self.br]))
        
        if to_decay:
            logger.info('We need to recalculate the branching fractions for %s' % ','.join(to_decay))
            if hasattr(self.model.get('particles')[0], 'partial_widths'):
                logger.info('using the FeynRules formula present in the model (arXiv:1402.1178)')
            else:
                logger.info('Using MadWidth (arXiv:1402.1178)')
                #self.extract_br_from_width_evaluation(to_decay)
            self.launch_width_evaluation(to_decay, mgcmd) 

       
        return self.br

    def get_BR_for_each_decay(self, decay_processes, multiparticles):
        """ get the list for possible decays & the associated branching fraction  """
        
        model = self.model
        base_model = self.model
        pid2label = self.pid2label

        ponctuation=[',','>',')','(']
        new_decay_processes={}       

        for part in decay_processes.keys():
            pos_symbol=-1
            branch_list=decay_processes[part].split()
            new_decay_processes[part]={}
            new_decay_processes[part]['']={}
            new_decay_processes[part]['']['config']=""
            new_decay_processes[part]['']['br']=1.0

            initial=""
            final=[]
            for index, item in enumerate(branch_list):
                # First get the symbol at the next position
                if index<len(branch_list)-1:
                    next_symbol=branch_list[index+1]
                else:
                    next_symbol=''
                # Then handle the symbol item case by case 
                if next_symbol=='>':              # case1: we have a particle initiating a branching
                    initial=item
                    if item not in [ particle['name'] for particle in base_model['particles'] ] \
                        and item not in [ particle['antiname'] for particle in base_model['particles'] ]:
                        raise Exception, "No particle "+item+ " in the model "+model
                    continue
                elif item=='>': continue       # case 2: we have the > symbole
                elif item not in ponctuation : # case 3: we have a particle originating from a branching
                    final.append(item)
                    if next_symbol=='' or next_symbol in ponctuation:
                        #end of a splitting, verify that it exists
                        if initial not in self.br.keys():
                            logger.debug('Branching fractions of particle '+initial+' are unknown')
                            return 0
                        if len(final)>2:
                            raise Exception, 'splittings different from A > B +C are currently not implemented '

                        if final[0] in multiparticles.keys():
                            set_B=[pid2label[pid] for pid in multiparticles[final[0]]]
                        else:
                            if final[0] not in [ particle['name'] for particle in base_model['particles'] ] \
                               and final[0] not in [ particle['antiname'] for particle in base_model['particles'] ]:
                                raise Exception, "No particle "+item+ " in the model "
                            set_B=[final[0]]
                        if final[1] in multiparticles.keys():
                            set_C=[pid2label[pid] for pid in multiparticles[final[1]]]
                        else:
                            if final[1] not in [ particle['name'] for particle in base_model['particles'] ] \
                               and final[1] not in [ particle['antiname'] for particle in base_model['particles'] ]:
                                raise Exception, "No particle "+item+ " in the model "+model
                            set_C=[final[1]]

                        splittings={}
                        counter=0
                        for chan in range(len(self.br[initial])): # loop over all channels
                            got_it=0
                            for d1 in set_B: 
                                for d2 in set_C:
                                    if (d1==self.br[initial][chan]['daughters'][0] and \
                                                 d2==self.br[initial][chan]['daughters'][1]) or \
                                                  (d2==self.br[initial][chan]['daughters'][0] and \
                                                 d1==self.br[initial][chan]['daughters'][1]):
                                        split=" "+initial+" > "+d1+" "+d2+" "
                                        # For the tag we need to order d1 d2, so that equivalent tags can be correctly idetified
                                        list_daughters=sorted([d1,d2])
                                        tag_split="|"+initial+">"+list_daughters[0]+list_daughters[1]
                                        counter+=1
                                        splittings[tag_split]={}
                                        splittings[tag_split]['config']=split
                                        splittings[tag_split]['br']=self.br[initial][chan]['br']
                                        got_it=1
                                        break # to avoid double counting in cases such as w+ > j j 
                                if got_it: break                   
                
                        if len(splittings)==0:
                            logger.info('Branching '+initial+' > '+final[0]+' '+final[1])
                            logger.info('is currently unknown')
                            return 0
                        else:
                            new_decay_processes[part]=self.update_branch(new_decay_processes[part],splittings)
                    
                        inital=""
                        final=[]

                else: # case 4: ponctuation symbol outside a splitting
                      # just append it to all the current branches
                    fake_splitting={}
                    fake_splitting['']={}
                    fake_splitting['']['br']=1.0
                    fake_splitting['']['config']=item
                    new_decay_processes[part]=self.update_branch(new_decay_processes[part],fake_splitting)

        return new_decay_processes        







    def print_branching_fractions(self):
        """ print a list of all known branching fractions"""

        for res in self.br.keys():
            logger.info('  ')
            logger.info('decay channels for '+res+' : ( width = ' 
                        +str(self.width_value[res])+' GeV )')
            logger.info('       BR                 d1  d2' )
            for decay in self.br[res]:
                bran = decay['br']
                d1 = decay['daughters'][0]
                d2 = decay['daughters'][1]
                logger.info('   %e            %s  %s ' % (bran, d1, d2) )
            logger.info('  ')

    def print_partial_widths(self):
        """ print a list of all known partial widths"""

        for res in self.br.keys():
            logger.info('  ')
            logger.info('decay channels for '+res+' :')
            logger.info('       width                     d1  d2' )

            for chan, decay in enumerate(self.br[res]):
                width=self.br[res][chan]['width']
                d1=self.br[res][chan]['daughters'][0]
                d2=self.br[res][chan]['daughters'][1]
                logger.info('   %e            %s  %s ' % (width, d1, d2) )
            logger.info('  ')


    def extract_br_from_width_evaluation(self, to_decay):
        """ use MadGraph5_aMC@NLO to generate me's for res > all all  
        """
        raise DeprecationWarning

        if os.path.isdir(pjoin(self.path_me,"width_calculator")):
            shutil.rmtree(pjoin(self.path_me,"width_calculator"))
        
        assert not os.path.exists(pjoin(self.path_me, "width_calculator"))
        
        path_me = self.path_me 
        label2pid = self.label2pid
        # first build a set resonances with pid>0

        #particle_set= list(to_decay)
        pids = set([abs(label2pid[name]) for name in to_decay])
        particle_set = [label2pid[pid] for pid in pids]
    

        modelpath = self.model.get('modelpath')
        if os.path.basename(modelpath) != self.model['name']:
            name, restrict = self.model['name'].rsplit('-',1)
            if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                modelpath = pjoin(os.path.dirname(modelpath), self.model['name'])
    
        commandline="import model %s\n" % modelpath
        commandline+="generate %s > all all \n" % particle_set[0]
        commandline+= "set automatic_html_opening False --no_save\n"
        if len(particle_set)>1:
            for index in range(1,len(particle_set)):
                commandline+="add process %s > all all \n" % particle_set[index]

        commandline += "output %s/width_calculator -f \n" % path_me


        aloha.loop_mode = False
        aloha.unitary_gauge = False
        cmd = Cmd.MasterCmd()        
        for line in commandline.split('\n'):
            cmd.run_cmd(line)
        
        # WRONG Needs to takes the param_card from the input files.
        ff = open(pjoin(path_me, 'width_calculator', 'Cards', 'param_card.dat'),'w')
        ff.write(self.banner['slha'])
        ff.close()
        
        lhapdf = False
        if os.environ.has_key('lhapdf'):
            lhapdf = os.environ['lhapdf']
            del os.environ['lhapdf']
        
        # run but remove the pdf dependencies
        cmd.import_command_file(['launch',
                                 'set lpp1 0', 
                                 'set lpp2 0', 
                                 'done'])

        if lhapdf:
            os.environ['lhapdf'] = lhapdf
                
        #me_cmd = me_interface.MadEventCmd(pjoin(path_me,'width_calculator'))
        #me_cmd.exec_cmd('set automatic_html_opening False --no_save')

        filename=pjoin(path_me,'width_calculator','Events','run_01','param_card.dat')
        self.extract_br_from_card(filename)

    def extract_br_for_antiparticle(self):
        '''  
            for each channel with a specific br value, 
            set the branching fraction of the complex conjugated channel 
            to the same br value 
        '''
        
        label2pid = self.label2pid
        pid2label = self.label2pid
        for res in self.br.keys():
            particle=self.model.get_particle(label2pid[res])
            if particle['self_antipart']: 
                continue
            anti_res=pid2label[-label2pid[res]]
            self.br[anti_res] = []
            if res in self.width_value: 
                self.width_value[anti_res]=self.width_value[res]
            elif anti_res in self.width_value:
                self.width_value[res]=self.width_value[anti_res]
                res, anti_res = anti_res, res
            for chan, decay in enumerate(self.br[res]):
                self.br[anti_res].append({})
                bran=decay['br']
                d1=decay['daughters'][0]
                d2=decay['daughters'][1]
                d1bar=pid2label[-label2pid[d1]]
                d2bar=pid2label[-label2pid[d2]]
                self.br[anti_res][chan]['br']=bran
                self.br[anti_res][chan]['daughters']=[]
                self.br[anti_res][chan]['daughters'].append(d1bar)
                self.br[anti_res][chan]['daughters'].append(d2bar)
                if decay.has_key('width'):
                    self.br[anti_res][chan]['width']=decay['width']                  

    def launch_width_evaluation(self,resonances, mgcmd):
        """ launch the calculation of the partial widths """

        label2pid = self.label2pid
        pid2label = self.label2pid
        model = self.model
        # first build a set resonances with pid>0
        # since compute_width cannot be used for particle with pid<0
        
        particle_set = set()
        for part in resonances:
            if part in mgcmd._multiparticles:
                for pid in mgcmd._multiparticles[part]:
                    particle_set.add(abs(pid))
                continue
            pid_part = abs(label2pid[part]) 
            particle_set.add(abs(pid_part))  

        particle_set = list(particle_set)
        argument = {'particles': particle_set, 
                    'path': pjoin(self.path_me, 'param_card.dat'),
                    'output': pjoin(self.path_me, 'param_card.dat'),
                    'body_decay': 2}
        
        self.compute_widths(model, argument)
        self.extract_br_from_card(pjoin(self.path_me, 'param_card.dat'))
        self.banner['slha'] = open(pjoin(self.path_me, 'param_card.dat')).read()
        if hasattr(self.banner,'param_card'):
            del self.banner.param_card
        self.banner.charge_card('param_card')
        return      
          
    def compute_widths(self, model, opts):
        
        from madgraph.interface.master_interface import MasterCmd
        import madgraph.iolibs.helas_call_writers as helas_call_writers
        cmd = MasterCmd()
        #self.define_child_cmd_interface(cmd, interface=False)
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        if not opts['path']:
            opts['path'] = pjoin(self.me_dir, 'Cards', 'param_card.dat')
            if not opts['force'] :
                self.ask_edit_cards(['param_card'],[], plot=False)
        
        
        line = 'compute_widths %s %s' % \
                (' '.join([str(i) for i in opts['particles']]),
                 ' '.join('--%s=%s' % (key,value) for (key,value) in opts.items()
                        if key not in ['model', 'force', 'particles'] and value))
        cmd.exec_cmd('import model %s' % model.get('name'))
#        cmd._curr_model = model
#        cmd._curr_fortran_model = helas_call_writers.FortranUFOHelasCallWriter(model)
        cmd.exec_cmd(line)
        #self.child = None
        del cmd                                

    def extract_br_from_banner(self, banner):
        """get the branching ratio from the banner object:
           for each resonance with label 'res', and for each channel with index i,
           returns a dictionary branching_fractions[res][i]
           with keys
            'daughters' : label of the daughters (only 2 body)
            'br' : value of the branching fraction"""
        
        self.br = {}
        
        # read the param_card internally to the banner
        if not hasattr(banner, 'param_card'):
            banner.charge_card('param_card')
        param_card = banner.param_card
        return self.extract_br_from_card(param_card)

    def extract_br_from_card(self, param_card):
        """get the branching ratio from the banner object:
           for each resonance with label 'res', and for each channel with index i,
           returns a dictionary branching_fractions[res][i]
           with keys
            'daughters' : label of the daughters (only 2 body)
            'br' : value of the branching fraction"""        
        
        if isinstance(param_card, str):
            import models.check_param_card as check_param_card
            param_card = check_param_card.ParamCard(param_card)
        
        if 'decay' not in param_card or not hasattr(param_card['decay'], 'decay_table'):
            return self.br

        self.width_value={}
        for id, data in param_card['decay'].decay_table.items():
#           check is the new width is close to the one originally in the banner
            recalculated_width=param_card['decay'].param_dict[(id,)].value
            width_in_the_banner=self.banner.get('param_card', 'decay', abs(id)).value
            relative_diff=abs(recalculated_width-width_in_the_banner)/recalculated_width
            if (relative_diff > 0.05):
               logger.warning('The LO estimate for the width of particle %s ' % id)
               logger.warning('differs from the one in the banner by %d percent ' % (relative_diff*100))

            label = self.pid2label[id]
            current = [] # tmp name for  self.br[label]
            for parameter in data:
                if parameter.lhacode[0] == 2:
                    d = [self.pid2label[pid] for pid in  parameter.lhacode[1:]]
                    current.append({'daughters':d, 'br': parameter.value})
            self.br[label] = current
            self.width_value[label]=recalculated_width  

        #update the banner:
        self.banner['slha'] = param_card.write(None)
        self.banner.param_card = param_card
        
        self.extract_br_for_antiparticle()
        return self.br


class decay_misc:
    """class with various methods for the decay"""

    @staticmethod
    def get_all_resonances(banner, mgcmd, allowed):
        """ return a list of labels of each resonance involved in the decay chain """
        allowed = list(allowed)
        found = set()
        alias = {} # if an allowed particles is inside a multiparticle        
        
        # look at the content of the multiparticles in order to know which one
        # we need to track.
        multiparticles = mgcmd._multiparticles
        model = mgcmd._curr_model
        for name, content in multiparticles.items():
            curr = [model.get_particle(id).get('name') \
                              for id in content 
                              if model.get_particle(id).get('name') in allowed ]
            if found:
                alias[name] = set(curr)

        # Now look which of the possible decay that we need to look at are indeed
        # present in the final state of one process.
        for line in banner.proc_card:
            line.strip()
            if line.startswith('generate') or line.startswith('add process'):
                final_process = re.split(r'>.*>|>|[\$/,@\[]', line)[1]
                for search in allowed:
                    if search in final_process:
                        found.add(search)
                        allowed.remove(search)
                for search, data in alias.items():
                    if search in final_process:
                        found.update(data)
                        del alias[search]
                        
        # treat multiparticles
        finalfound = set()
        for name in found:
            if name in mgcmd._multiparticles:
                finalfound.update([model.get_particle(id).get('name') 
                                   for id in mgcmd._multiparticles[name]])
                finalfound.discard(name)
            else:
                finalfound.add(name)

        return finalfound
   
        
    def reorder_branch(self,branch):
        """ branch is a string with the definition of a decay chain
                If branch contains " A > B C , B > ... " 
                reorder into             " A > C B , B > ... "

        """
        branch=branch.replace(',', ' , ')
        branch=branch.replace(')', ' ) ')
        branch=branch.replace('(', ' ( ')
        list_branch=branch.split(" ")
        for index in range(len(list_branch)-1,-1,-1):
            if list_branch[index]==' ' or list_branch[index]=='': del list_branch[index]
        #print list_branch
        for index, item in enumerate(list_branch):

            if item[-1] =="," and list_branch[index+1]!="(":
                # search pos of B and C 
                counter=1
                while 1:
                  if list_branch[index-counter].find("=")<0:
                     break
                  counter+=1
                if list_branch[index-counter-1]==list_branch[index+1]:
                    # swap the two particles before the comma:
                    temp=list_branch[index-counter-1]
                    list_branch[index-counter-1]=list_branch[index-counter]
                    list_branch[index-counter]=temp
            if item[-1] =="," and list_branch[index+1]=="(":
                # search pos of B and C 
                counter=1
                while 1:
                  if list_branch[index-counter].find("=")<0:
                     break
                  counter+=1
                if list_branch[index-counter -1]==list_branch[index+2]:
                    # swap the two particles before the comma:
                    temp=list_branch[index-counter-1]
                    list_branch[index-counter-1]=list_branch[index-counter]
                    list_branch[index-counter]=temp

        new_branch=""
        for item in list_branch:
            new_branch+=item+" "

        return new_branch, list_branch[0]

    def set_light_parton_massless(self,topo):
        """ masses of light partons are set to zero for 
            the evaluation of the matrix elements
        """

        light_partons=[21,1,2,3]
        for part in topo["get_id"].keys():
            if abs(topo["get_id"][part]) in light_partons :
                topo["get_mass2"][part]=0.0

#    need to check if last branch is a t-branching. If it is, 
#    we need to update the value of branch["m2"]
#    since this will be used in the reshuffling procedure
        if len(topo["branchings"])>0:  # Exclude 2>1 self.topologies
            if topo["branchings"][-1]["type"]=="t":
                if topo["branchings"][-2]["type"]!="t":
                    logger.info('last branching is t-channel')
                    logger.info('but last-but-one branching is not t-channel')
                else:
                    part=topo["branchings"][-1]["index_d2"] 
                    if part >0: # reset the mass only if "part" is an external particle
                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])

    @staticmethod
    def modify_param_card(pid2widths, path_me):
        """Modify the param_card w/r to what is read from the banner:
             if the value of a width is set to zero in the banner, 
             it is automatically set to its default value in this code
        """

        param_card=open(pjoin(path_me,'param_card.dat'), 'r')
        new_param_card=""
        while 1:
            line=param_card.readline()
            if line =="": break
            list_line=line.split()
            if len(list_line)>2:
                if list_line[0]=="DECAY" and int(list_line[1]) in pid2widths.keys():
                    list_line[2]=str(pid2widths[int(list_line[1])]) 
                    line=""
                    for item in list_line:
                        line+=item+ "    "
                    line+="\n"
            new_param_card+=line

        param_card.close()
        param_card=open(pjoin(path_me, 'param_card.dat'), 'w')
        param_card.write(new_param_card) 
        param_card.close()


    def select_one_topo(self,prod_values):
#
#    prod_values[0] is the value of |M_prod|^2
#    prod_values[1], prod_values[2], ... are the values of individual diagrams
#                                                                            (called AMP2 in mg) 
#

# first evaluate the sum of all single diagram values
        total=0.0
        cumul=[0.0]
        for i in range(1,len(prod_values)):
            cumul.append(cumul[i-1]+float(prod_values[i]))
            total+=float(prod_values[i])

        for i in range(len(cumul)): cumul[i]=cumul[i]/total

        #print "Cumulative AMP2 values: "
        #print cumul
        select_topo=random.random()

        for i in range(1,len(cumul)):
            if select_topo>cumul[i-1] and select_topo<cumul[i]: 
                good_topo=i
                break

        #print "Selected topology"
        #print good_topo
        return good_topo, cumul
    
    def get_final_state_compact(self,final_state_full):

        dc_pos=final_state_full.find(",")

        if dc_pos>0:
            branch_list=final_state_full.split(",")
            del branch_list[0]
            list_obj=final_state_full.split()
            final_state_compact=""
            to_be_deleted=[]
            for index, obj in enumerate(list_obj):
                if obj==">":
                    to_be_deleted.append(list_obj[index-1])

            for obj in list_obj:
                if obj!=">" and obj!="," and obj not in to_be_deleted:
                    final_state_compact+=obj+"    "

            branches={}
            for branch in branch_list:
                list_part=branch.split()
                branches[list_part[0]]={"finalstate":list_part[2:]}
                branches[list_part[0]]["branch"], dummy= self.reorder_branch(branch)
        else:
            final_state_compact=final_state_full
            branches={}

        return final_state_compact, branches
                  
    def get_mean_sd(self,list_obj):
        """ return the mean value and the standard deviation of a list of reals """
        sum=0.0
        N=float(len(list_obj))
        for item in list_obj:
            sum+=item
        mean=sum/N
        sd=0.0
        for item in list_obj:
            sd+=(item-mean)**2
        sd=sd/(N-1.0)
        
        return mean, sd
     

class decay_all_events(object):
    
    def __init__(self, ms_interface, banner, inputfile, options):
        """Store all the component and organize special variable"""
    
        # input
        self.options = options
        #max_weight_arg = options['max_weight']  
        self.path_me = os.path.realpath(options['curr_dir']) 
        if options['ms_dir']:
            self.path_me = os.path.realpath(options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        self.mgcmd = ms_interface.mg5cmd
        self.mscmd = ms_interface
        self.model = ms_interface.model
        self.banner = banner
        self.evtfile = inputfile
        self.curr_event = Event(self.evtfile, banner) 
        self.inverted_decay_mapping={}
        self.width_estimator = None
        self.curr_dir = os.getcwd()
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        # need to unbuffer all I/O in fortran, otherwise
        # the values of matrix elements are not passed to the Python script
        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'  
    
        # Remove old stuff from previous runs
        # so that the current run is not confused
        # Don't have to do that for gridpack / or if asked.
        if not (options["ms_dir"] or options["use_old_dir"]):
            if os.path.isdir(pjoin(self.path_me,"production_me")):
                shutil.rmtree(pjoin(self.path_me,"production_me"))
            if os.path.isdir(pjoin(self.path_me,"full_me")):
                shutil.rmtree(pjoin(self.path_me,"full_me"))    
            if os.path.isdir(pjoin(self.path_me,"decay_me")):
                shutil.rmtree(pjoin(self.path_me,"decay_me"))     
    
        # Prepare some dict usefull for optimize model imformation
        # pid -> label and label -> pid
        self.pid2label=pid2label(self.model)
        self.banner.check_pid(self.pid2label)
        self.pid2label.update(label2pid(self.model))
        self.pid2massvar={}
        self.pid2widthvar={}
        for part in self.model['particles']:
            self.pid2massvar[int(part['pdg_code'])]=part['mass']    
            self.pid2widthvar[int(part['pdg_code'])]=part['width']    

        # load the Monte Carlo masses
        self.MC_masses=self.get_MC_masses()

#        logger.info('Value of the Monte Carlo masses: ')
#        logger.info(self.MC_masses)


        # dictionary pid > color_rep
        self.pid2color = pid2color(self.model)

        # energy of the collider
        self.Ecollider=float(self.banner.get('run_card', 'ebeam1'))\
                       +float(self.banner.get('run_card', 'ebeam2'))


        # write down the seed:
        seedfile=open(pjoin(self.path_me, 'seeds.dat'),'w')
        seedfile.write('  %s \n' % self.options['seed'])
        seedfile.close()       
 
        # width and mass information will be filled up later
        self.pid2width = lambda pid: self.banner.get('param_card', 'decay', abs(pid)).value
        self.pid2mass = lambda pid: self.banner.get('param_card', 'mass', abs(pid)).value
        
        if os.path.isfile(pjoin(self.path_me,"param_card.dat")):
            os.remove(pjoin(self.path_me,"param_card.dat"))        

        # now overwrite the param_card.dat in Cards:
        param_card=self.banner['slha']
        #param_card=decay_tools.check_param_card( param_card)

        # now we can write the param_card.dat:
        # Note that the width of each resonance in the    
        # decay chain should be >0 , we will check that later on
        model_name = os.path.basename(self.model.get('name'))
        param=open(pjoin(self.path_me,'param_card.dat'),"w")
        param.write(param_card)
        param.close()   
        if model_name == 'mssm' or model_name.startswith('mssm-'):
            #need to convert to SLHA2 format
            import models.check_param_card as check_param_card
            check_param_card.convert_to_mg5card(pjoin(self.path_me,'param_card.dat'))
  
        
        self.list_branches = ms_interface.list_branches
        decay_ids = [self.pid2label[key] for key in self.list_branches \
                                                       if key in self.pid2label]
        for multi in self.mgcmd._multiparticles:
            if multi in self.list_branches:
                decay_ids += self.mgcmd._multiparticles[multi]
        self.all_ME = AllMatrixElement(banner, self.options, decay_ids, self.model)
        self.all_decay = {}


 
        # generate BR and all the square matrix element based on the banner.
        pickle_info = pjoin(self.path_me,"production_me", "all_ME.pkl")
        if not options["use_old_dir"] or not os.path.exists(pickle_info):
            self.generate_all_matrix_element()
            save_load_object.save_to_file(pickle_info,
                                          (self.all_ME,self.all_decay,self.width_estimator))
        else:
            try:
                self.all_ME, self.all_decay,self.width_estimator = save_load_object.load_from_file(pjoin(self.path_me,"production_me", "all_ME.pkl"))
            except Exception,error:
                logger.debug(str(error))
                self.generate_all_matrix_element()
                save_load_object.save_to_file(pickle_info,
                                          (self.all_ME,self.all_decay,self.width_estimator))                
        
        if not self.options["onlyhelicity"]:
            resonances = self.width_estimator.resonances
            logger.debug('List of resonances: %s' % resonances)
            self.extract_resonances_mass_width(resonances) 

        self.compile()
        
    def get_MC_masses(self):
        
        MC_masses={}
        pid_heavyquarks=[4,5]
        if 'montecarlomasses' in self.banner:
            
            MC_masses_lines=self.banner['montecarlomasses'].split('\n')
            for line in MC_masses_lines:
                pidvalue=line.split()
                if len(pidvalue)<2: continue # skip blank lines
                pid=abs(int(pidvalue[0]))
                value=float(pidvalue[1])
                MC_masses[pid]=value
                if pid in pid_heavyquarks:
                    value_ME=self.banner.get('param_card','mass', pid).value
                    if value_ME>1E-10:
                        if pid==5:
                            logger.warning('set the mass of the b-quark to its value in the param_card.dat: %s GeV ' % value_ME)
                        if pid==4:
                            logger.warning('set the mass of the c-quark to its value in the param_card.dat: %s GeV ' % value_ME)
                        MC_masses[pid]=value_ME
            
        return MC_masses 

        
    def run(self):
        """Running the full code""" 
    
        max_weight_arg = self.options['max_weight']  
        decay_tools=decay_misc()
        
        #Next step: we need to determine which matrix elements are really necessary
        #==========================================================================
        decay_mapping = self.get_identical_decay()

        # also compute the inverted map, which will be used in the decay procedure       
        for tag in decay_mapping:
            for equiv_decay in decay_mapping[tag]:
                self.inverted_decay_mapping[equiv_decay[0]]=tag
 
        self.mscmd.update_status('MadSpin: Estimate the maximum weight')
        # Estimation of the maximum weight
        #=================================
        if max_weight_arg>0:
            for key in self.all_ME:
                for mode in self.all_ME['decays']:
                    mode['max_weight'] = max_weight_arg
        elif self.options["onlyhelicity"]:
            logger.info("not needed in helicity mode")
        else:
            #self.get_max_weight_from_1toN()
            self.get_max_weight_from_event(decay_mapping)  
        
        # add probability of not writting events (for multi production with 
        # different decay
        self.add_loose_decay()
        # Store this object with all the associate number for gridpack:
        if self.options['ms_dir']:
            self.save_status_to_pickle()
    
        self.ending_run()
        
    def ending_run(self):
        """launch the unweighting and deal with final information"""    
        # launch the decay and reweighting
        self.mscmd.update_status('MadSpin: Decaying Events')
        efficiency = self.decaying_events(self.inverted_decay_mapping)
        if  efficiency != 1:
            # need to change the banner information [nb_event/cross section]
            files.cp(self.outputfile.name, '%s_tmp' % self.outputfile.name)
            self.outputfile = open(self.outputfile.name, 'w')
            self.write_banner_information(efficiency)
            pos = self.outputfile.tell()
            old = open('%s_tmp' % self.outputfile.name)
            line=''
            while '</init>' not in line:
                line = old.readline()
            
            self.outputfile.write(old.read())
            files.rm('%s_tmp' % self.outputfile.name)
            
        # Closing all run
        self.terminate_fortran_executables()
        if not self.options['ms_dir']:
            shutil.rmtree(pjoin(self.path_me,'production_me'))
            shutil.rmtree(pjoin(self.path_me,'full_me'))
            if not self.options["onlyhelicity"]:
                shutil.rmtree(pjoin(self.path_me,'decay_me'))
        # set the environment variable GFORTRAN_UNBUFFERED_ALL 
        # to its original value
        #os.environ['GFORTRAN_UNBUFFERED_ALL']='n'

    def save_status_to_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        #don't store the event file in the pkl
        evt_file, self.evtfile = self.evtfile, None
        curr_event, self.curr_event = self.curr_event , None
        mgcmd, self.mgcmd = self.mgcmd, None
        mscmd, self.mscmd = self.mscmd , None
        pid2mass, self.pid2mass = self.pid2mass, None
        pid2width, self.pid2width = self.pid2width, None
        #banner, self.banner = self.banner, None
        #self.all_ME.banner = None
        
        name = pjoin(self.options['ms_dir'], 'madspin.pkl')
        save_load_object.save_to_file(name, self)
        
        #restore the event file
        self.evtfile = evt_file
        self.curr_event = curr_event
        self.mgcmd = mgcmd
        self.mscmd = mscmd 
        self.pid2mass = pid2mass
        self.pid2width = pid2width
        #self.banner = banner
        #self.all_ME.banner = banner
        
    def decaying_events(self,inverted_decay_mapping):
        """perform the decay of each events"""

        decay_tools = decay_misc()
        # tools for checking if max_weight is too often broken.
        report = collections.defaultdict(int,{'over_weight': 0}) 


        logger.info(' ' )
        logger.info('Decaying the events... ')
        self.outputfile = open(pjoin(self.path_me,'decayed_events.lhe'), 'w')
        self.write_banner_information()
        
        
        event_nb, fail_nb = 0, 0
        nb_skip = 0 
        trial_nb_all_events=0
        starttime = time.time()
        nb_fail_mc_mass=0
        while 1: # loop on event file
            production_tag, event_map = self.load_event()
            if production_tag == 0 == event_map: #end of file
                break

            if  event_nb and \
                (event_nb % max(int(10**int(math.log10(float(event_nb)))),1000)==0): 
                running_time = misc.format_timer(time.time()-starttime)
                logger.info('Event nb %s %s' % (event_nb, running_time))
                self.mscmd.update_status(('$events',1,event_nb, 'decaying events'), 
                                         force=False, print_log=False)
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')


            if self.options["onlyhelicity"]:
                trial_nb, fail = self.adding_only_helicity(event_map, production_tag)
                trial_nb_all_events+=trial_nb
                fail_nb += fail
                event_nb += 1
                continue

            # Here we need to select a decay configuration on a random basis:
            decay = self.all_ME.get_random_decay(production_tag)
            if not decay['decay_tag']:
                #Not writting events due to global reweighting
                nb_skip +=1
                continue 
            else:
                #  for the matrix element, identify the master decay channel to which 'decay' is equivalent:
                decay_tag_me=inverted_decay_mapping[decay['decay_tag']]
                try:
                    decay_me=self.all_ME.get_decay_from_tag(production_tag, decay_tag_me)
                except Exception:
                    #if the master didn't exsit try the original one.
                    decay_me=self.all_ME.get_decay_from_tag(production_tag, decay['decay_tag'])
                    
                event_nb+=1
                report[decay['decay_tag']] += 1 

            indices_for_mc_masses, values_for_mc_masses=self.get_montecarlo_masses_from_event(decay['decay_struct'], event_map, decay['prod2full'])
            nb_mc_masses=len(indices_for_mc_masses)

            p, p_str=self.curr_event.give_momenta(event_map)
            stdin_text=' %s %s %s %s \n' % ('2', self.options['BW_cut'], self.Ecollider, decay_me['max_weight'])
            stdin_text+=p_str
            # here I also need to specify the Monte Carlo Masses
            stdin_text+=" %s \n" % nb_mc_masses
            if  nb_mc_masses>0:
                stdin_text+='%s  \n' % str(indices_for_mc_masses).strip('[]').replace(',', ' ')
                stdin_text+='%s  \n' % str(values_for_mc_masses).strip('[]').replace(',', ' ')
            
#            here apply the reweighting procedure in fortran
            output = self.loadfortran( 'unweighting', decay_me['path'], stdin_text)
            if not output:
                fail_nb +=1
                continue
            trial_nb, BWvalue, weight, momenta, failed, use_mc_masses, helicities = output 
            
            # next: need to fill all intermediate momenta
            if nb_mc_masses>0 and use_mc_masses==0:nb_fail_mc_mass+=1
            
            ext_mom=self.get_mom(momenta)
            # fill all momenta in the decay branches
            momenta_in_decay=self.get_int_mom_in_decay(decay['decay_struct'],ext_mom)
            # reset extrenal momenta in the production event
            self.reset_mom_in_prod_event(decay['decay_struct'],decay['prod2full'],\
                                         event_map,momenta_in_decay,ext_mom, use_mc_masses, helicities)
            # reset intermediate momenta in prod event
            self.curr_event.reset_resonances()
            
            #
            decayed_event = self.decay_one_event_new(self.curr_event,decay['decay_struct'],\
                                                      event_map, momenta_in_decay,use_mc_masses, helicities)
            
            
            # Treat the case we get too many failures for the PS generation.
            if failed > 500 :
                logger.debug('Got a production event with %s failures for the phase-space generation generation ' % failed)

            # Treat the case that we ge too many overweight.
            if weight > decay_me['max_weight']:
                report['over_weight'] += 1
                report['%s_f' % (decay['decay_tag'],)] +=1
                if __debug__:               
                    misc.sprint('''over_weight: %s %s, occurence: %s%%, occurence_channel: %s%%
                    production_tag:%s [%s], decay:%s [%s], BW_cut: %1g\n
                    ''' %\
                    (weight/decay['max_weight'], decay['decay_tag'], 
                    100 * report['over_weight']/event_nb,
                    100 * report['%s_f' % (decay['decay_tag'],)] / report[decay['decay_tag']],
                    os.path.basename(self.all_ME[production_tag]['path']),
                    production_tag,
                    os.path.basename(decay['path']),
                    decay['decay_tag'],BWvalue))
                        
                
                if weight > 10.0 * decay['max_weight']:
                    error = """Found a weight MUCH larger than the computed max_weight (ratio: %s). 
    This usually means that the Narrow width approximation reaches it's limit on part of the Phase-Space.
    Do not trust too much the tale of the distribution and/or relaunch the code with smaller BW_cut.
    This is for channel %s with current BW_value at : %g'""" \
                    % (weight/decay['max_weight'], decay['decay_tag'], BWvalue)  
                    logger.error(error)
                elif report['over_weight'] > max(0.005*event_nb,3):
                    error = """Found too many weight larger than the computed max_weight (%s/%s = %s%%). 
    Please relaunch MS with more events/PS point by event in the
    computation of the maximum_weight.
                    """ % (report['over_weight'], event_nb, 100 * report['over_weight']/event_nb )  
                    raise MadSpinError, error
                        
                    error = True
                elif report['%s_f' % (decay['decay_tag'],)] > max(0.01*report[decay['decay_tag']],3):
                    error = """Found too many weight larger than the computed max_weight (%s/%s = %s%%),
    for channel %s. Please relaunch MS with more events/PS point by event in the
    computation of the maximum_weight.
                    """ % (report['%s_f' % (decay['decay_tag'],)],\
                            report['%s' % (decay['decay_tag'],)],\
                            100 * report['%s_f' % (decay['decay_tag'],)] / report[ decay['decay_tag']] ,\
                            decay['decay_tag'])  
                    raise MadSpinError, error
                    
             
            decayed_event.change_wgt(factor= self.branching_ratio) 
            #decayed_event.wgt = decayed_event.wgt * self.branching_ratio
                    
            self.outputfile.write(decayed_event.string_event())
                #print "number of trials: "+str(trial_nb)
            trial_nb_all_events+=trial_nb
            
             
                    
 
        self.outputfile.write('</LesHouchesEvents>\n')
        self.evtfile.close()
        self.outputfile.close()

        if report['over_weight'] > max(0.15*math.sqrt(event_nb),1):
            error = """Found many weight larger than the computed max_weight (%s/%s = %s%%). 
            """ % (report['over_weight'], event_nb, 100 * report['over_weight']/event_nb )  
            logger.warning(error)
        for decay_tag in self.all_decay.keys():
            if report['%s_f' % (decay_tag,)] > max(0.2*report[decay_tag],1):
                error = """Found many weight larger than the computed max_weight (%s/%s = %s%%),
    for channel %s.""" % (report['%s_f' % (decay_tag,)],\
                               report['%s' % (decay_tag,)],\
                               100 * report['%s_f' % (decay_tag,)] / report[decay_tag] ,\
                               decay_tag)
                logger.warning(error)  
        
        

        logger.info('Total number of events written: %s/%s ' % (event_nb, event_nb+nb_skip))
        logger.info('Average number of trial points per production event: '\
            +str(float(trial_nb_all_events)/float(event_nb)))
        logger.info('Branching ratio to allowed decays: %g' % self.branching_ratio)
        logger.info('Number of events with weights larger than max_weight: %s' % report['over_weight'])
        logger.info('Number of subprocesses '+str(len(self.calculator)))
        logger.info('Number of failures when restoring the Monte Carlo masses: %s ' % nb_fail_mc_mass)
        if fail_nb:
            logger.info('Number of failures in reshuffling (event skipped): %s ' % fail_nb)
        
        return  event_nb/(event_nb+nb_skip)       


    def adding_only_helicity(self, event_map, production_tag):
        """no decays for this production mode, run in passthrough mode,
           only adding the helicities to the events """

        #no decays for this production mode, run in passthrough mode, only adding the helicities to the events
        nb_mc_masses=0
        p, p_str=self.curr_event.give_momenta(event_map)
        stdin_text=' %s %s %s %s \n' % ('2', self.options['BW_cut'], self.Ecollider, 1.0)
        stdin_text+=p_str
        # here I also need to specify the Monte Carlo Masses
        stdin_text+=" %s \n" % nb_mc_masses
        
        mepath = self.all_ME[production_tag]['path']
        decay = self.all_ME[production_tag]['decays'][0]
        decay_me=self.all_ME.get_decay_from_tag(production_tag, decay['decay_tag'])
        mepath = decay_me['path']
                        
        output = self.loadfortran( 'unweighting', mepath, stdin_text)
        if not output:
            # Event fail
            return 0, 1
        trial_nb, BWvalue, weight, momenta, failed, use_mc_masses, helicities = output                
        self.reset_helicityonly_in_prod_event(event_map, helicities)

        decayed_event = self.curr_event
        self.outputfile.write(decayed_event.string_event())
        #print "number of trials: "+str(trial_nb)
        
        return trial_nb, 0


    def get_int_mom_in_decay(self,decay_struct,ext_mom):
        """  fill  """
        momenta_in_decay={}
        for part in decay_struct.keys():
            branch=decay_struct[part]['mg_tree']
            nb_splitting=len(branch)
            for split in range(nb_splitting-1,-1,-1): 
                mother=branch[split][0]
                d1=branch[split][1]
                d2=branch[split][2]
                if d1>0:
                    momenta_in_decay[d1]=ext_mom[d1-1]  # list_momenta is ordered according to ME
                if d2>0:
                    momenta_in_decay[d2]=ext_mom[d2-1]  # list_momenta is ordered according to ME
                momenta_in_decay[mother]=momenta_in_decay[d1].add(momenta_in_decay[d2])
                
        return momenta_in_decay
    
    def reset_mom_in_prod_event(self, decay_struct,prod2full, event_map, momenta_in_decay,ext_mom,use_mc_masses,helicities):

        """ Reset the external momenta in the production event, since
            the virtuality of decaying particles has slightly changed the kinematics
        """
       
        for index in self.curr_event.event2mg.keys():
            if self.curr_event.event2mg[index]>0:
                part=self.curr_event.event2mg[index]       # index for production ME
                part_for_curr_evt=event_map[part-1]+1 # index for curr event
                pid=self.curr_event.particle[part_for_curr_evt]['pid']
                if part in decay_struct:
                    id_res=decay_struct[part]['mg_tree'][0][0]
                    self.curr_event.particle[part_for_curr_evt]['momentum']=momenta_in_decay[id_res].copy()
                    self.curr_event.particle[part_for_curr_evt]['mass']=self.curr_event.particle[part_for_curr_evt]['momentum'].m
                else:
                    self.curr_event.particle[part_for_curr_evt]['momentum']=ext_mom[prod2full[part-1]-1]
                    self.curr_event.particle[part_for_curr_evt]['helicity']=helicities[prod2full[part-1]-1]
                    if not use_mc_masses or abs(pid) not in self.MC_masses:
                        try:
                            self.curr_event.particle[part_for_curr_evt]['mass']=self.banner.get('param_card','mass', abs(pid)).value
                        except KeyError:
                            if self.model.get_particle(abs(pid)).get('mass').lower() == 'zero':
                                self.curr_event.particle[part_for_curr_evt]['mass'] = 0
                            else:
                                raise
                    else:
                        self.curr_event.particle[part_for_curr_evt]['mass']=self.MC_masses[abs(pid)]

 
    def reset_helicityonly_in_prod_event(self, event_map, helicities):

        """ Reset the external momenta in the production event, since
            the virtuality of decaying particles has slightly changed the kinematics
        """
       
        for index in self.curr_event.event2mg.keys():
            if self.curr_event.event2mg[index]>0:
                part=self.curr_event.event2mg[index]       # index for production ME
                part_for_curr_evt=event_map[part-1]+1 # index for curr event
                pid=self.curr_event.particle[part_for_curr_evt]['pid']
                self.curr_event.particle[part_for_curr_evt]['helicity']=helicities[part-1]

    def get_mom(self,momenta):
        """ input: list of momenta in a string format 
            output: list of momenta in a 'momentum' format
        """
        output=[]
        for item in momenta:
            comps=item.split()
            mom=momentum(float(comps[0]),float(comps[1]),float(comps[2]),float(comps[3]))
            output.append(mom)
        return output
        
    def get_identical_decay(self):
        """identify the various decay which are identical to each other"""
        
        logger.info('detect independant decays')
        start = time.time()
        # Possbilitiy to Bypass this step 
        if len(self.all_decay) == 1:
            relation = {}
            base_tag = None
            for prod in self.all_ME.values():
                for decay in prod['decays']:
                    tags = decay['decay_tag']
                    for tag in tags:
                        if not base_tag:
                            relation[tag] = (tag, 1)
                            base_tag = tag
                        elif (tag,1) not in relation[base_tag]:
                            relation[tag] = (base_tag,1)
            decay_mapping = self.get_process_identical_ratio(relation)
            return decay_mapping
        
        BW_cut = self.options['BW_cut']       
        
        #class the decay by class (nbody/pid)
        nbody_to_decay = collections.defaultdict(list)
        for decay in self.all_decay.values():
            id = decay['dc_branch']['tree'][-1]['label']
            id_final = decay['processes'][0].get_final_ids_after_decay()
            cut = 0.0 
            mass_final = tuple([m if m> cut else 0 for m in map(self.pid2mass, id_final)])
            
            nbody_to_decay[(decay['nbody'], abs(id), mass_final)].append(decay)
        
        relation = {} # {tag: {(tag2, ratio)}}
        # Loop over the class and create the relation information about the 1     
        for ((nbody, pid, finals),decays) in nbody_to_decay.items():
            if len(decays) == 1:
                continue  
            mom_init = momentum(self.pid2mass(pid), 0, 0, 0)
            
            # create an object for the validation, keeping the ratio between
            # MEM i and MEM j. this is set at zero when the ratio is not found
            #constant
            valid = dict([ ((i, j), True) for j in range(len(decays)) 
                                          for i in range(len(decays)) 
                                          if i != j])
                
            for nb in range(125):
                tree, jac, nb_sol = decays[0]['dc_branch'].generate_momenta(mom_init,\
                                        True, self.pid2width, self.pid2mass, BW_cut,self.Ecollider)
                if not tree:
                    continue
                p_str = '%s\n%s\n'% (tree[-1]['momentum'],
                    '\n'.join(str(tree[i]['momentum']) for i in range(1, len(tree))
                                                                  if i in tree))
                
                
                values = {}                
                for i in range(len(decays)):
                    if any([valid[(i,j)] for j in range(len(decays)) if i !=j]):
                        values[i] = self.calculate_matrix_element('decay', 
                                                       decays[i]['path'], p_str)
                    else:
                        #skip computation if all possibility are ruled out.
                        values[i] = 0
                              
                #check if the ratio is constant for all possibilities
                for i in range(len(decays)):
                    for j in range(i+1, len(decays)):
                        if values[i] == 0 or values[j] == 0 or valid[(i,j)] == 0:
                            continue # already not valid
                        elif valid[(i,j)] is True:
                            valid[(i,j)] = values[j]/values[i]
                            valid[(j,i)] = valid[(i,j)]
                        elif (valid[(i,j)] - values[j]/values[i]) < 1e-6 * (valid[(i,j)] + values[j]/values[i]):
                            pass
                        else:
                            valid[(i, j)] = 0
                            valid[(j, i)] = 0
                
            if __debug__: 
                for i in range(len(decays)):
                    comment= "| "
                    for j in range(len(decays)):
                        if i == j:
                            comment+= "%4e " % 1
                            continue
                        comment+=  "%4e " % valid[(i,j)]
                    comment+= "|"+ os.path.basename(decays[i]['path'])                     
                    logger.debug(comment)
            
            # store the result in the relation object. (using tag as key)
            for i in range(len(decays)):
                tag_i = decays[i]['tag'][2:]
                for j in range(i+1, len(decays)): 
                    tag_j = decays[j]['tag'][2:]     
                    if valid[(i,j)] and tag_j not in relation:
                        relation[tag_j] = (tag_i, valid[(i,j)])
                        
        # fullfill the object with the already identify to one decay.
        #and add those who doesn't have any relations.
        for decay in self.all_decay.values():
            tags = [m.shell_string(pdg_order=True)[2:] for m in decay['processes']]
            init_tag = tags[0]
            if init_tag not in relation:
                out = (init_tag, 1)
            else:
                out = relation[init_tag]
            for tag in tags[1:]:
                relation[tag] = out

        decay_mapping = self.get_process_identical_ratio(relation)
        
        logger.info('Done in %ss' % (time.time()-start))
        return decay_mapping


    def get_process_identical_ratio(self, relation):
        # Now that we have ratio relation between each tag, we need to say 
        #what is the relation between the decay of the production process.
        #This is not only the product since some decay can be equivalent.
        
        decay_mapping = {} # final output: {first_process: [(equiv_proc, ratio), ...]
        tag2real = {}    # basic tag [the one related via relation] -> first process
        # basic tag ratio doesn't have any identical factor (this simplify calculation)
        nb=0
        for prod in self.all_ME.values():
            for decay in prod['decays']:
                tag = decay['decay_tag']
                nb+=1
                # build the basic tag (all equiv process are related to this tag)
                basic_tag = []
                ratio = 1
                for t in tag:
                    if  t in relation:
                        basic_tag.append(relation[t][0])
                        ratio *= relation[t][1]
                    else:
                        basic_tag.append(t)
                basic_tag = tuple(basic_tag)
                
                # compute identical factor ratio compare to a fully diffent decay 
                #that we have assume for the basic tag
                if len(set(tag)) != len(tag):
                    for t in set(tag):
                        ratio /= math.factorial(tag.count(t))
                
                # Now build the output
                if basic_tag not in tag2real:
                    tag2real[basic_tag] = (tag, ratio)
                    decay_mapping[tag] = set([(tag, 1)])
                    ratio2=1
                else:
                    real_tag, ratio2 = tag2real[basic_tag]
                    if real_tag != tag:
                        decay_mapping[real_tag].add((tag, ratio/ratio2))


        return decay_mapping
    

    @misc.mute_logger()
    @test_aloha.set_global()
    def generate_all_matrix_element(self):
        """generate the full series of matrix element needed by Madspin.
        i.e. the undecayed and the decay one. And associate those to the 
        madspin production_topo object"""

        # 1. compute the partial width        
        # 2. compute the production matrix element
        # 3. create the all_topology object 
        # 4. compute the full matrix element (use the partial to throw away 
        #     pointless decay.
        # 5. add the decay information to the all_topology object (with branching
        #     ratio)  
        
        
        # 0. clean previous run ------------------------------------------------
        path_me = self.path_me
        try:
            shutil.rmtree(pjoin(path_me,'full_me'))
        except Exception: 
            pass
        try:
            shutil.rmtree(pjoin(path_me,'production_me'))
        except Exception, error:
            pass
        path_me = self.path_me        
        
        # 1. compute the partial width------------------------------------------
        if not self.options["onlyhelicity"]:
            self.get_branching_ratio()
        
        # 2. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search('^\s*add\s+process', line)]
        
        mgcmd = self.mgcmd
        modelpath = self.model.get('modelpath')
        if os.path.basename(modelpath) != mgcmd._curr_model['name']:
            name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
            if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])
            
        commandline="import model %s " % modelpath
        mgcmd.exec_cmd(commandline)
        # Handle the multiparticle of the banner        
        #for name, definition in self.mscmd.multiparticles:
        if hasattr(self.mscmd, 'multiparticles_ms'):
            for name, pdgs in  self.mscmd.multiparticles_ms.items():
                if name == 'all':
                    continue
                #self.banner.get('proc_card').get('multiparticles'):
                mgcmd.do_define("%s = %s" % (name, ' '.join(`i` for i in pdgs)))
            

        mgcmd.exec_cmd("set group_subprocesses False")

        logger.info('generating the production square matrix element')
        start = time.time()
        commandline=''
        for proc in processes:
            # deal with @ syntax need to move it after the decay specification
            if '@' in proc:
                proc, proc_nb = proc.split('@')
                try:
                    proc_nb = int(proc_nb)
                except ValueError:
                    raise MadSpinError, 'MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb
                proc_nb = '@ %i' % proc_nb 
            else:
                proc_nb = ''
            
            if ',' in proc:
                raise MadSpinError, 'MadSpin can not decay event which comes from a decay chain.'+\
                        '\n  The full decay chain should either be handle by MadGraph or by Masdspin.'
            
            if '[' not in proc:
                commandline+="add process %s  --no_warning=duplicate;" % proc
            else:
                process, order, final = re.split('\[\s*(.*)\s*\]', proc)
                commandline+="add process %s %s --no_warning=duplicate;" % (process, proc_nb)
                if not order:
                    continue
                elif not order.startswith('virt='):
                    if '=' in order:
                        order = order.split('=',1)[1]
                    # define the list of particles that are needed for the radiateion
                    pert = fks_common.find_pert_particles_interactions(
                         mgcmd._curr_model,pert_order = order)['soft_particles']
                    commandline += "define pert_%s = %s;" % (order, ' '.join(map(str,pert)) )
                    
                    # check if we have to increase by one the born order
                    if '%s=' % order in process:
                        result=re.split(' ',process)
                        process=''
                        for r in result:
                            if '%s=' % order in r:
                                ior=re.split('=',r)
                                r='QCD=%i' % (int(ior[1])+1)
                            process=process+r+' '
                    #handle special tag $ | / @
                    result = re.split('([/$@]|\w+=\w+)', process, 1)                    
                    if len(result) ==3:
                        process, split, rest = result
                        commandline+="add process %s pert_%s %s%s %s --no_warning=duplicate;" % (process, order ,split, rest, proc_nb)
                    else:
                        commandline +='add process %s pert_%s  %s --no_warning=duplicate;' % (process,order, proc_nb)                                       
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output standalone_msP %s %s' % \
        (pjoin(path_me,'production_me'), ' '.join(self.list_branches.keys()))        
        mgcmd.exec_cmd(commandline, precmd=True)        
        logger.info('Done %.4g' % (time.time()-start))

        # 3. Create all_ME + topology objects ----------------------------------
        
        matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
        
        self.all_ME.adding_me(matrix_elements, pjoin(path_me,'production_me'))
        
        # 3b. simplify list_branches -------------------------------------------
        # remove decay which are not present in any production ME.
        final_states = set()        
        for me in matrix_elements:
            for leg in me.get('base_amplitude').get('process').get('legs'):
                if not leg.get('state'):
                    continue
                label = self.model.get_particle(leg.get('id')).get_name()
                if self.all_ME.has_particles_ambiguity:
                    final_states.add(self.pid2label[-1*self.pid2label[label]])
                final_states.add(label)
        for key in self.list_branches.keys():
            if key not in final_states and key not in self.mgcmd._multiparticles:
                if (len(self.list_branches)>1):
                    del self.list_branches[key]
                elif not self.options["onlyhelicity"]:
                    raise Exception, " No decay define for process."
                    logger.info('keeping dummy decay for passthrough mode')

        # 4. compute the full matrix element -----------------------------------
        if not self.options["onlyhelicity"]:
            logger.info('generating the full square matrix element (with decay)')
            start = time.time()
            to_decay = self.mscmd.list_branches.keys()
            decay_text = []
            for decays in self.mscmd.list_branches.values():
                for decay in  decays:
                    if '=' not in decay:
                        decay += ' QCD=99'
                    if ',' in decay:
                        decay_text.append('(%s)' % decay)
                    else:
                        decay_text.append(decay)
            decay_text = ', '.join(decay_text)
            commandline = ''
            
            
            for proc in processes:
                # deal with @ syntax need to move it after the decay specification
                if '@' in proc:
                    proc, proc_nb = proc.split('@')
                    try:
                        proc_nb = int(proc_nb)
                    except ValueError:
                        raise MadSpinError, 'MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb
                    proc_nb = '@ %i' % proc_nb 
                else:
                    proc_nb = '' 
                
                if '[' not in proc:
                    nb_comma = proc.count(',')
                    if nb_comma == 0:
                        commandline+="add process %s, %s %s  --no_warning=duplicate;" % (proc, decay_text, proc_nb)
                    elif nb_comma == 1:
                        before, after = proc.split(',')
                        commandline+="add process %s, %s, (%s, %s) %s  --no_warning=duplicate;" % (before, decay_text, after, decay_text, proc_nb)
                    else:
                        raise Exception, 'too much decay at MG level. this can not be done for the moment)'
                else:
                    process, order, final = re.split('\[\s*(.*)\s*\]', proc)
                    commandline+="add process %s, %s %s  --no_warning=duplicate;" % (process, decay_text, proc_nb)
                    if not order:
                        continue
                    elif not order.startswith('virt='):
                        if '=' in order:
                            order = order.split('=',1)[1]
                        # define the list of particles that are needed for the radiateion
                        pert = fks_common.find_pert_particles_interactions(
                             mgcmd._curr_model,pert_order = order)['soft_particles']
                        commandline += "define pert_%s = %s;" % (order, ' '.join(map(str,pert)) )
                        
                        # check if we have to increase by one the born order
                        if '%s=' % order in process:
                            result=re.split(' ',process)
                            process=''
                            for r in result:
                                if '%s=' % order in r:
                                    ior=re.split('=',r)
                                    r='QCD=%i' % (int(ior[1])+1)
                                process=process+r+' '
                        #handle special tag $ | / @
                        result = re.split('([/$@]|\w+=\w+)', process, 1)                    
                        if len(result) ==3:
                            process, split, rest = result
                            commandline+="add process %s pert_%s %s%s , %s %s --no_warning=duplicate;" % \
                                  (process, order, split, rest, decay_text, proc_nb)
                        else:
                            commandline +='add process %s pert_%s, %s %s  --no_warning=duplicate;' % \
                                               (process, order, decay_text, proc_nb)
            commandline = commandline.replace('add process', 'generate',1)
            logger.info(commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            # remove decay with 0 branching ratio.
            mgcmd.remove_pointless_decay(self.banner.param_card)
            commandline = 'output standalone_msF %s %s' % (pjoin(path_me,'full_me'),
                                                          ' '.join(self.list_branches.keys()))
            mgcmd.exec_cmd(commandline, precmd=True)
            logger.info('Done %.4g' % (time.time()-start))
        elif self.options["onlyhelicity"]:
            logger.info("Helicity Matrix-Element")      
            commandline = 'output standalone_msF %s %s' % \
            (pjoin(path_me,'full_me'), ' '.join(self.list_branches.keys()))        
            mgcmd.exec_cmd(commandline, precmd=True)        
            logger.info('Done %.4g' % (time.time()-start))                    



        # 5. add the decay information to the all_topology object --------------                        
        for matrix_element in mgcmd._curr_matrix_elements.get_matrix_elements():
            me_path = pjoin(path_me,'full_me', 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())
            self.all_ME.add_decay(matrix_element, me_path)

        # 5.b import production matrix elements (+ related info) in the full process directory
        list_prodfiles=['matrix_prod.f','configs_production.inc','props_production.inc','nexternal_prod.inc']
        for tag in self.all_ME:
            prod_path=self.all_ME[tag]['path']
            nfinal=len(self.all_ME[tag]['base_order'][1])
            for dico in self.all_ME[tag]['decays']:
                full_path=dico['path']
                #print prod_path
                #print full_path
                #print ' '
                for item in list_prodfiles:
                     #print full_path
                     prodfile=pjoin(prod_path,item)
                     destination=pjoin(full_path,item)
                     shutil.copyfile(prodfile, destination)  
                # we need to write the file config_decays.inc
                self.generate_configs_file(nfinal,dico,full_path)
                

        if self.options["onlyhelicity"]:
            return

        # 6. generate decay only part ------------------------------------------
        logger.info('generate matrix element for decay only (1 - > N).')
        start = time.time()
        commandline = ''
        i=0
        for processes in self.list_branches.values():
            for proc in processes:
                commandline+="add process %s @%i --no_warning=duplicate;" % (proc,i)
                i+=1        
        commandline = commandline.replace('add process', 'generate',1)
        mgcmd.exec_cmd(commandline, precmd=True)
        # remove decay with 0 branching ratio.
        mgcmd.remove_pointless_decay(self.banner.param_card)
        #
        commandline = 'output standalone_msF %s' % pjoin(path_me,'decay_me')
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        logger.info('Done %.4g' % (time.time()-start))        
        #
        self.all_decay = {}
        for matrix_element in mgcmd._curr_matrix_elements.get_matrix_elements():
            me = matrix_element.get('processes')[0]
            me_string = me.shell_string()
            dirpath = pjoin(path_me,'decay_me', 'SubProcesses', "P%s" % me_string)
        #    
            self.all_decay[me_string] = {'path': dirpath, 
                                         'dc_branch':dc_branch_from_me(me),
                                         'nbody': len(me.get_final_ids_after_decay()),
                                         'processes': matrix_element.get('processes'),
                                         'tag': me.shell_string(pdg_order=True)}
        #
#        if __debug__:
#            #check that all decay matrix element correspond to a decay only
#            for prod in self.all_ME.values():
#                for decay in prod['matrix_element']['base_amplitude']['process']['decay_chains']:
#                    assert decay.shell_string() in self.all_decay
            
        
    def get_branching_ratio(self):
        """compute the branching ratio of all the decaying particles"""
    
        # Compute the width branching ratio. Doing this at this point allows
        #to remove potential pointless decay in the diagram generation.
        resonances = decay_misc.get_all_resonances(self.banner, 
                         self.mgcmd, self.mscmd.list_branches.keys())

        logger.debug('List of resonances:%s' % resonances)
        path_me = os.path.realpath(self.path_me) 
        width = width_estimate(resonances, path_me, self.banner, self.model,
                                self.pid2label)
        width.extract_br(self.list_branches, self.mgcmd)
        width.print_branching_fractions()
        #self.channel_br = width.get_BR_for_each_decay(self.decay_processes, 
        #                                    self.mgcmd._multiparticles)
        self.width_estimator = width
        self.banner.param_card = width.banner.param_card
        return width    


    def compile(self):
        logger.info('Compiling code')
        self.compile_fortran(self.path_me, mode="full_me")
        if not self.options["onlyhelicity"]:
            self.compile_fortran(self.path_me, mode="production_me")
            self.compile_fortran(self.path_me, mode="decay_me")

    def compile_fortran(self, path_me, mode='production_me'):
        """ Compile the fortran executables associated with the evalutation of the 
                matrix elements (production process)
                Returns the path to the fortran executable
        """

        base_dir = pjoin(path_me, mode,"SubProcesses")
        list_prod=os.listdir(base_dir)
        logger.debug("""Finalizing %s's """% mode)

        # COMPILATION OF LIBRARY
        misc.compile( cwd=pjoin(path_me, mode,"Source","DHELAS"), mode='fortran')
        file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
        shutil.copyfile(file_madspin, pjoin(path_me, mode,"Source","MODEL","lha_read.f" ))
        if not self.options["use_old_dir"]: 
            misc.compile(arg=['clean'], cwd=pjoin(path_me, mode,"Source","MODEL"), mode='fortran')
        misc.compile( cwd=pjoin(path_me, mode,"Source","MODEL"), mode='fortran')     

        file=pjoin(path_me, 'param_card.dat')
        shutil.copyfile(file,pjoin(path_me,mode,"Cards","param_card.dat")) 

#       get all paths to matix elements
        list_prod=[]
        if mode == 'full_me':
            for tag in self.all_ME:    
                for dico in self.all_ME[tag]['decays']:
                    full_path=dico['path']
                    if full_path not in list_prod: list_prod.append(full_path)
        elif mode == 'production_me':
            for tag in self.all_ME:    
                prod_path=self.all_ME[tag]['path']
                if prod_path not in list_prod: list_prod.append(prod_path)
        elif mode == 'decay_me':
                for dir in os.listdir(base_dir):
                    if dir[0] == 'P': list_prod.append(pjoin(base_dir, dir))

        for i,me_path in enumerate(list_prod):
#            if direc[0] == "P" and os.path.isdir(pjoin(base_dir, direc)):
#                new_path = pjoin(base_dir, direc)
                new_path = me_path

                if mode == 'full_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver.f')
                    shutil.copyfile(file_madspin, pjoin(new_path,"driver.f")) 
                elif mode == 'production_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_prod.f')
                    shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f")) 
                else:
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_decay.f')
                    shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f")) 
                     
                
                if mode=='full_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'ranmar.f')
                    shutil.copyfile(file_madspin, pjoin(new_path,"ranmar.f"))
                    file_madspin=pjoin(path_me, 'seeds.dat')  
                    files.ln(file_madspin, new_path)
                    file_madspin=pjoin(new_path, 'offset.dat')
                    open(file_madspin,'w').write('%i\n' % i)
                    
                    
                      
                    

                
                
                if mode == 'full_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_full')
                elif mode == 'production_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_prod')
                else:
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_decay')
                    
                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )

                # files to produce the parameters:
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
                    
                shutil.copyfile(pjoin(path_me, mode,'Source','MODEL','input.inc'),
                                pjoin(new_path,'input.inc'))
                if not os.path.exists(pjoin(new_path,os.path.pardir, 'parameters.inc')):
                    if not self.options["use_old_dir"]:
                        misc.compile(arg=['clean'], cwd=new_path, mode='fortran')
                    misc.compile(arg=['init'],cwd=new_path,mode='fortran')
                    misc.call('./init', cwd=new_path)
                    shutil.copyfile(pjoin(new_path,'parameters.inc'), 
                               pjoin(new_path,os.path.pardir, 'parameters.inc'))
                if mode == 'production_me':
                    misc.compile(cwd=new_path, mode='fortran')
                else:
                    misc.compile(cwd=new_path, mode='fortran')
                    misc.compile(arg=['check'], cwd=new_path, mode='fortran')

                if __debug__:
                    if(os.path.getsize(pjoin(path_me, mode,'SubProcesses', 'parameters.inc'))<10):
                        print pjoin(path_me, mode,'SubProcesses', 'parameters.inc')
                        raise Exception, "Parameters of the model were not written correctly ! %s " %\
                            os.path.getsize(pjoin(path_me, mode,'SubProcesses', 'parameters.inc'))


    def extract_resonances_mass_width(self, resonances):
        """ """

        label2width = {}
        label2mass = {}
        pid2width = {}
        #pid2mass = self.pid2mass
        need_param_card_modif = False
        
        # now extract the width of the resonances:
        for particle_label in resonances:
            try:
                part=abs(self.pid2label[particle_label])
                #mass = self.banner.get('param_card','mass', abs(part))
                width = self.banner.get('param_card','decay', abs(part))
            except ValueError, error:
                continue
            else:
                if (width.value > 0.001):  
                    label2width[particle_label]=float(width.value)
                else: # the width is less than 1 MeV, need to use an effective width !!
                      # this is useful to handle cases like tau decays
                    label2width[particle_label]=0.001
                    need_param_card_modif = True
                    logger.warning('ATTENTION')
                    logger.warning('Found a very small width in the param_card for particle '\
                                   +str(particle_label))
                    logger.warning('Use instead an effective width of 1 MeV ' )
                #label2mass[particle_label]=float(mass.value)
                #pid2mass[part]=label2mass[particle_label]
                pid2width[abs(part)]=label2width[particle_label]
                if label2width[particle_label]==0.0:
                    need_param_card_modif = True
                    for param in self.model["parameters"][('external',)]:
                        if param.lhablock=="DECAY" and param.lhacode==[abs(part)]:
                            label2width[particle_label]=max(param.value,0.001)
                            pid2width[abs(part)]=label2width[particle_label]
                    logger.warning('ATTENTION')
                    logger.warning('Found a zero width in the param_card for particle '\
                                   +str(particle_label))
                    logger.warning('Use instead the default/effective value '\
                                   +str(label2width[particle_label]))
               
        # now we need to modify the values of the width
        # in param_card.dat, since this is where the input 
        # parameters will be read when evaluating matrix elements
        if need_param_card_modif:
            decay_misc.modify_param_card(pid2width, self.path_me)            
            
    def get_max_weight_from_event(self, decay_mapping):
        """ """

        decay_tools = decay_misc()
        
        # check all set of decay that need to be done:
        decay_set = set()
        for production in self.all_ME.values():
            decay_set.add(production['decaying'])

        numberev = self.options['Nevents_for_max_weigth'] # number of events
        numberps = self.options['max_weight_ps_point'] # number of phase pace points per event
        
        logger.info('  ')
        logger.info('   Estimating the maximum weight    ')
        logger.info('   *****************************    ')
        logger.info('     Probing the first '+str(numberev)+' events')
        logger.info('     with '+str(numberps)+' phase space points')
        if len(decay_set) > 1: 
            logger.info('     For %s decaying particle type in the final states' % len(decay_set))
        logger.info('  ')

        
        probe_weight = []
       

        starttime = time.time()
        ev = -1
        nb_decay = dict( (key,0) for key in decay_set)
        probe_weight = dict( (key,[]) for key in decay_set)
        while ev+1 < len(decay_set) * numberev: 
            production_tag, event_map = self.load_event()

            if production_tag == 0 == event_map: #end of file
                logger.info('Not enough events for at least one production mode.')
                logger.info('This is ok as long as you don\'t reuse the max weight for other generations.')
                break
            
            #check if this event is usefull or not
            decaying = self.all_ME[production_tag]['decaying']
            if nb_decay[decaying] >=  numberev:
                continue 
            ev += 1
            nb_decay[decaying] += 1 

#            mg5_me_prod, prod_values = self.evaluate_me_production(production_tag, event_map)   

   
            logger.debug('Event %s/%s: ' % (ev+1, len(decay_set)*numberev))
            if (len(decay_set)*numberev -(ev+2)) >0:
                self.mscmd.update_status((len(decay_set)*numberev -(ev+2),1,ev+1, 
                                          'MadSpin: Maximum weight'), 
                                         force=False, print_log=False)
            #logger.debug('Selected topology               : '+str(tag_topo))

            max_decay = {}
            mean_decay= {}
            std_decay = {}

            atleastonedecay=False
            for decay in self.all_ME[production_tag]['decays']:
                tag = decay['decay_tag']

                if decay_mapping and not tag in decay_mapping:
                    continue
                if not tag:
                    continue # No decay for this process
                atleastonedecay = True
                weight = self.get_max_weight_from_fortran(decay['path'], event_map,numberps,self.options['BW_cut'])
                
                    #weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod
                if tag in max_decay:
                    max_decay[tag] = max([max_decay[tag], weight])
                else:
                    max_decay[tag] = weight
                    #print weight, max_decay[name]
                    #raise Exception 
                      
            if not atleastonedecay:
                # NO decay [one possibility is all decay are identical to their particle]
                logger.info('No independent decay for one type of final states -> skip those events for the maximum weight computation')
                nb_decay[decaying] = numberev
                ev += numberev -1
                continue
            probe_weight[decaying].append(max_decay)

            self.terminate_fortran_executables()
            self.calculator = {}
            self.calculator_nbcall = {}
            if ev % 5 == 0:
                running_time = misc.format_timer(time.time()-starttime)
                info_text = 'Event %s/%s : %s \n' % (ev + 1, len(decay_set)*numberev, running_time) 
                #for  index,tag_decay in enumerate(max_decay):
                #    info_text += '            decay_config %s [%s] : %s\n' % \
                #       (index+1, ','.join(tag_decay), probe_weight[decaying][nb_decay[decaying]-1][tag_decay])
                logger.info(info_text[:-1])
        
        
        
        # Computation of the maximum weight used in the unweighting procedure
        for decaying in probe_weight:
            if not probe_weight[decaying]:
                continue
            #me_linked = [me for me in self.all_ME.values() if me['decaying'] == decaying]
            for decay_tag in probe_weight[decaying][0].keys():
                weights=[]
                for ev in range(numberev):
                    try:
                        weights.append(probe_weight[decaying][ev][decay_tag])
                    except:
                        continue
                if not weights:
                    logger.warning( 'no events for %s' % decay_tag)
                    continue
                weights.sort(reverse=True)
                assert weights[0] >= weights[1]
                ave_weight, std_weight = decay_tools.get_mean_sd(weights)
                std_weight=math.sqrt(std_weight)
                base_max_weight = 1.05 * (ave_weight+self.options['nb_sigma']*std_weight)

                for i in [20, 30, 40, 50]:
                    if len(weights) < i:
                        break
                    ave_weight, std_weight = decay_tools.get_mean_sd(weights[:i])
                    std_weight=math.sqrt(std_weight)
                    base_max_weight = max(base_max_weight, 1.05 * (ave_weight+self.options['nb_sigma']*std_weight))
                    
                if weights[0] > base_max_weight:
                    base_max_weight = 1.05 * weights[0]
              
                for associated_decay, ratio in decay_mapping[decay_tag]:
                    max_weight= ratio * base_max_weight
                    if ratio != 1:
                        max_weight *= 1.1 #security

                    br = 0                   
                    #assign the value to the associated decays
                    for k,m in self.all_ME.items():
                        for mi in m['decays']:

                            if mi['decay_tag'] == associated_decay:
                                mi['max_weight'] = max_weight
                                br = mi['br']
                                nb_finals = len(mi['finals'])

                    if decay_tag == associated_decay:                
                        logger.debug('Decay channel %s :Using maximum weight %s [%s] (BR: %s)' % \
                               (','.join(decay_tag), base_max_weight, max(weights), br/nb_finals))
                    else:  
                        logger.debug('Decay channel %s :Using maximum weight %s (BR: %s)' % \
                                    (','.join(associated_decay), max_weight, br/nb_finals)) 

#        if __debug__: 
        # check that all decay have a max_weight and fix it if not the case.
        for prod in self.all_ME.values():
            for dec in prod['decays']:
                if dec['decay_tag'] and not 'max_weight' in dec:
                    dec['max_weight'] = 0. 

#                        assert 'max_weight' in dec and dec['max_weight'] ,\
#                                  'fail for %s (%s)' % (str(dec['decay_tag']), \
#                                                  os.path.basename(prod['path']))
        self.evtfile.seek(0)
        return

    def load_event(self):
        """Load the next event and ensure that the ME is define"""

        #decay_tools = decay_misc()
        if self.curr_event.get_next_event() == 'no_event':
            return 0, 0
        production_tag, order = self.curr_event.get_tag()
        P_order = self.all_ME[production_tag]['tag2order'][production_tag]

        event_map = {}
        evt_order = list(order[0])+list(order[1])
        for i, id in enumerate(P_order[0] + P_order[1]):
            in_event = [pos for pos, label in enumerate(evt_order) \
                               if label == id]
            if i < len(order[0]):
                in_event = [pos for pos in in_event if pos < len(order[0])]
            else:
                in_event = [pos for pos in in_event if pos >= len(order[0])]
                
            if len(in_event) == 1:
                in_event = in_event[0]
            else:
                config = random.randint(0, len(in_event)-1)
                in_event = in_event[config]            
            evt_order[in_event] = 0
            event_map[i] = in_event
        
        if __debug__ and len(order[0]) == 2:   
            assert event_map[0] in [0,1], 'wrong event mapping %s' % event_map
            assert event_map[1] in [0,1], 'wrong event mapping %s' % event_map
        assert production_tag in self.all_ME
        
        return production_tag, event_map
    
    def get_max_weight_from_fortran(self, path, event_map,nbpoints,BWcut):
        """return the max. weight associated with me decay['path']"""

        p, p_str=self.curr_event.give_momenta(event_map)

        std_in=" %s  %s %s %s  \n" % ("1",BWcut, self.Ecollider, nbpoints)
        std_in+=p_str
        max_weight = self.loadfortran('maxweight',
                               path, std_in)

        return max_weight
        
    def loadfortran(self, mode, path, stdin_text, first=True):
        """ call the fortran executable """

        tmpdir = ''
        if ('full',path) in self.calculator:
            external = self.calculator[('full',path)]
            self.calculator_nbcall[('full',path)] += 1
        else:
            logger.debug('we have %s calculator ready' % len(self.calculator))
            tmpdir = path

            executable_prod="./check"
            external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, 
                                                      stderr=STDOUT, cwd=tmpdir)
            self.calculator[('full',path,)] = external 
            self.calculator_nbcall[('full',path)] = 1 

        try:
            external.stdin.write(stdin_text)
        except IOError:
            if not first:
                raise
            try:
                external.terminate()
            except:
                pass
            del self.calculator[('full',path,)]
            return self.loadfortran(mode, path, stdin_text, first=False)

        if mode == 'maxweight':
            maxweight=float(external.stdout.readline())
            output = maxweight
        elif mode == 'full_me':
            me_value=float(external.stdout.readline())
            output = me_value
        elif mode == 'unweighting':
            firstline=external.stdout.readline().split()
            try:
                nexternal=int(firstline[0])
                trials= int(firstline[1])
                BWvalue= float(firstline[2])
                weight= float(firstline[3])
                failed= float(firstline[4])
                use_mc_masses=int(firstline[5])
            except ValueError:
                logger.debug(firstline)
                return
            momenta=[external.stdout.readline() for i in range(nexternal)]
            lastline=external.stdout.readline().split()
            helicities=[lastline[i] for i in range(len(lastline))]
            output = trials, BWvalue, weight, momenta, failed, use_mc_masses, helicities

        if len(self.calculator) > self.options['max_running_process']:
            logger.debug('more than %s calculators. Perform cleaning' % self.options['max_running_process'])
            nb_calls = self.calculator_nbcall.values()
            nb_calls.sort()
            cut = max([nb_calls[len(nb_calls)//2], 0.001 * nb_calls[-1]])
            for key, external in list(self.calculator.items()):
                nb = self.calculator_nbcall[key]
                if nb < cut:
                    if key[0]=='full':
                        path=key[1]
                        end_signal="5 0 0 0 \n"  # before closing, write down the seed 
                        external.stdin.write(end_signal)
                        ranmar_state=external.stdout.readline()
                        ranmar_file=pjoin(path,'ranmar_state.dat')
                        ranmar=open(ranmar_file, 'w')
                        ranmar.write(ranmar_state)
                        ranmar.close()
                    external.stdin.close()
                    external.stdout.close()
                    external.terminate()
                    del self.calculator[key]
                    del self.calculator_nbcall[key]
                else:
                    self.calculator_nbcall[key] = self.calculator_nbcall[key] //10
                    
        return output
    
    def calculate_matrix_element(self, mode, production, stdin_text):
        """routine to return the matrix element"""

        if mode != "decay":
            raise Exception, "This function is only secure in mode decay."

        tmpdir = ''
        if (mode, production) in self.calculator:
            external = self.calculator[(mode, production)]
            self.calculator_nbcall[(mode, production)] += 1
        else:
            logger.debug('we have %s calculator ready' % len(self.calculator))
            if mode == 'prod':
                tmpdir = pjoin(self.path_me,'production_me', 'SubProcesses',
                           production)
            elif mode in ['full','decay']:
                tmpdir = pjoin(self.path_me,'%s_me' % mode, 'SubProcesses',
                           production)
            executable_prod="./check"
            external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, 
                                                      stderr=STDOUT, cwd=tmpdir)
            self.calculator[(mode, production)] = external 
            self.calculator_nbcall[(mode, production)] = 1       

        external.stdin.write(stdin_text)
        if mode == 'prod':
            info = int(external.stdout.readline())
            nb_output = abs(info)+1
        else:
            info = 1
            nb_output = 1
         
        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])
        if info < 0:
            print 'ZERO DETECTED'
            print prod_values
            print stdin_text
            os.system('lsof -p %s' % external.pid)
            return ' '.join(prod_values.split()[-1*(nb_output-1):])
        
        if len(self.calculator) > self.options['max_running_process']:
            logger.debug('more than 100 calculator. Perform cleaning')
            nb_calls = self.calculator_nbcall.values()
            nb_calls.sort()
            cut = max([nb_calls[len(nb_calls)//2], 0.001 * nb_calls[-1]])
            for key, external in list(self.calculator.items()):
                nb = self.calculator_nbcall[key]
                if nb < cut:
                    external.stdin.close()
                    external.stdout.close()
                    external.terminate()
                    del self.calculator[key]
                    del self.calculator_nbcall[key]
                else:
                    self.calculator_nbcall[key] = self.calculator_nbcall[key] //10
        
        if mode == 'prod':
            return prod_values
        else:
            return float(prod_values)
              
    def generate_configs_file(self,nfinal,decay, path):
        """ write the file configs_decay.inc
            also record the itree information in a python variable, 
            this will be needed to write down the event
            
            decay_struct['mg_tree'] = [(d1,d2, mother), (d1,d2,mother), ...]
                with - BACKWARD ORDER, 
                     - me indices
        """
   
        decay_struct=decay['decay_struct'] 
        me_index=2 # should match the particle index in the full matrix element 
        count_res=0 # count number of resonances
        iforest=[] 
        pmasswidth=[]
        
#              data (map_external2res(i), i=1,4)/1,2,-2,-4/
 
        decay['prod2full']=[1,2]
        map_external='      data (map_external2res(i), i=1,%s)/1,2,' %(nfinal+2)    
        for part in range(3,nfinal+3):
            if part in decay_struct:  # particle in the prod. event to be decayed
                #print part
                decay_struct[part]['mg_tree']=[]

                nb_res=len(decay_struct[part]["tree"].keys())
                for res in range(-1,-nb_res-1,-1):
                    label=abs(decay_struct[part]["tree"][res]['label'])
                    mass=self.pid2massvar[label]
                    width=self.pid2widthvar[label] 
                    me_res=-nb_res-res-count_res-1
                    indexd1=decay_struct[part]["tree"][res]["d1"]["index"]
                    if indexd1>0:
                        me_index+=1
                        me_d1=me_index
                    else: 
                        # need to label resonances backward
                        me_d1 = -nb_res-indexd1-count_res-1
                    indexd2=decay_struct[part]["tree"][res]["d2"]["index"]
                    if indexd2>0:
                        me_index+=1
                        me_d2=me_index
                    else: 
                        # need to label resonances backward
                        me_d2 = -nb_res-indexd2-count_res-1
                    iforest.append("      DATA (IDECAY(I, %s ),I=1,2)/  %s ,  %s / \n" % (me_res, me_d1, me_d2))
                    decay_struct[part]['mg_tree'].append((me_res,me_d1,me_d2))
                    pmasswidth.append("      PRMASS(%s)=%s \n" %(me_res,mass) )
                    pmasswidth.append("      PRWIDTH(%s)=%s \n" %(me_res,width) )

                count_res=count_res+nb_res
                map_external+='%s ,' % (-count_res)
                decay['prod2full'].append(-count_res)
            else:
                me_index+=1
                map_external+='%s ,' % me_index
                decay['prod2full'].append(me_index)
   
        map_external=map_external[:-1]+'/ \n'
        
        trappe=open(pjoin(path,'configs_decay.inc'),'w')
        trappe.write(map_external)
        for item in iforest:
            trappe.write(item)
        trappe.write('      ns_channel_decay= %s \n' % count_res)
        for item in pmasswidth:
            trappe.write(item)
        trappe.close()

    def get_montecarlo_masses_from_event(self,decay_struct, event_map, map_prod2full):
        """
            from the production event curr_event and from the decay channel 'decay_struct'
            (which has just been selected randomly), get the MonteCarlo masses
        """
        
        # in order to preserve the natural order in lhe file,
        # we need the inverse of the dico event_map
        inv_event_map={}
        for i in event_map.keys():
            inv_event_map[event_map[i]]=i
        
        indices_for_mc_masses=[]
        values_for_mc_masses=[]
        
        for index in self.curr_event.event2mg.keys():
            if self.curr_event.event2mg[index]>0: # no need to consider resonances in the production event file
                part=inv_event_map[self.curr_event.event2mg[index]-1]+1 # index for prod. matrix element
                part_for_curr_evt=self.curr_event.event2mg[index]       # index for event file
                
                if part not in decay_struct:
                    # get the pid
                    curr_pid=abs(self.curr_event.particle[part_for_curr_evt]['pid'])
                    if curr_pid in self.MC_masses:
                        #print part
                        #print map_prod2full
                        indices_for_mc_masses.append(map_prod2full[part-1])
                        values_for_mc_masses.append(self.MC_masses[curr_pid])
                    
                else:
                    # now we need to write the decay products in the event
                    # follow the decay chain order, so that we can easily keep track of the mother index                                           
                    for res in range(-1,-len(decay_struct[part]["tree"].keys())-1,-1):
                        index_d1=decay_struct[part]['mg_tree'][-res-1][1]
                        index_d2=decay_struct[part]['mg_tree'][-res-1][2]
                        
                        pid_d1=abs(decay_struct[part]\
                                    ["tree"][res]["d1"]["label"])
                        pid_d2=abs(decay_struct[part]\
                                    ["tree"][res]["d2"]["label"])
                        if index_d1 >0 and pid_d1 in self.MC_masses:
                            indices_for_mc_masses.append(index_d1)
                            values_for_mc_masses.append(self.MC_masses[pid_d1])

                        if index_d2 >0 and pid_d2 in self.MC_masses:
                            indices_for_mc_masses.append(index_d2)
                            values_for_mc_masses.append(self.MC_masses[pid_d2])

        return indices_for_mc_masses,values_for_mc_masses

    def decay_one_event_new(self,curr_event,decay_struct, event_map, momenta_in_decay, use_mc_masses, helicities):
        """Write down the event 
           momenta is the list of momenta ordered according to the productin ME
        """

        pid2color = self.pid2color
        decayed_event=Event()
        decayed_event.event2mg={}

        decayed_event.ievent=curr_event.ievent
        decayed_event.wgt=curr_event.wgt
        decayed_event.scale=curr_event.scale
        decayed_event.aqed=curr_event.aqed
        decayed_event.aqcd=curr_event.aqcd
        decayed_event.diese=curr_event.diese
        decayed_event.rwgt=curr_event.rwgt
        decayed_event.event_init_line=curr_event.event_init_line

        part_number=0
        external=0
        maxcol=curr_event.max_col

        # in order to preserve the natural order in lhe file,
        # we need the inverse of the dico event_map
        inv_event_map={}
        for i in event_map.keys():
            inv_event_map[event_map[i]]=i
        sol_nb = None
        
        for index in curr_event.event2mg.keys():
            if curr_event.event2mg[index]>0:
                part=inv_event_map[curr_event.event2mg[index]-1]+1 # index for prod. matrix element
                part_for_curr_evt=curr_event.event2mg[index]       # index for event file
                
                if part not in decay_struct:
                    external+=1 
                    part_number+=1
                    decayed_event.particle[part_number]=curr_event.particle[part_for_curr_evt]
                    decayed_event.event2mg[part_number]=part_number
                
                else:
                    # now we need to write the decay products in the event
                    # follow the decay chain order, so that we can easily keep track of the mother index
                       
                    map_to_part_number={}
                    for res in range(-1,-len(decay_struct[part]["tree"].keys())-1,-1):
                        index_res_for_mom=decay_struct[part]['mg_tree'][-res-1][0]
                        if (res==-1):
                            part_number+=1
                            mom=momenta_in_decay[index_res_for_mom].copy()
                            pid=decay_struct[part]["tree"][res]['label'] 
                            istup=2
                            mothup1=curr_event.particle[part_for_curr_evt]["mothup1"]
                            mothup2=curr_event.particle[part_for_curr_evt]["mothup2"]
                            colup1=curr_event.particle[part_for_curr_evt]["colup1"]
                            colup2=curr_event.particle[part_for_curr_evt]["colup2"]
                            decay_struct[part]["tree"][res]["colup1"]=colup1
                            decay_struct[part]["tree"][res]["colup2"]=colup2
                            mass=mom.m
                            helicity=0.
                            decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":colup1,"colup2":colup2,"momentum":mom,\
                                "mass":mass,"helicity":helicity}
                            decayed_event.event2mg[part_number]=part_number

                            map_to_part_number[res]=part_number
   
#
#             Extract color information so that we can write the color flow
#
                        colormother=pid2color[decay_struct[part]["tree"][res]["label"]]
                        colord1=pid2color[decay_struct[part]\
                                            ["tree"][res]["d1"]["label"]]
                        colord2=pid2color[decay_struct[part]\
                                            ["tree"][res]["d2"]["label"]]
                
                        colup1=decay_struct[part]["tree"][res]["colup1"]
                        colup2=decay_struct[part]["tree"][res]["colup2"]

#            now figure out what is the correct color flow informatio
#            Only consider 1,3, 3-bar and 8 color rep.
#            Normally, the color flow needs to be determined only
#            during the reshuffling phase, but it is currenlty assigned 
#            for each "trial event"
                        if abs(colord1)==1:
                            d2colup1=colup1
                            d2colup2=colup2
                            d1colup1=0
                            d1colup2=0
                        elif abs(colord2)==1:
                            d1colup1=colup1
                            d1colup2=colup2
                            d2colup1=0
                            d2colup2=0
                        elif colord1==3 and colord2==-3 and colormother ==1:
                            maxcol+=1
                            d1colup1=maxcol
                            d1colup2=0
                            d2colup1=0
                            d2colup2=maxcol
                     
                        elif colord1==3 and colord2==-3 and colormother ==8:
                            d1colup1=colup1
                            d1colup2=0
                            d2colup1=0
                            d2colup2=colup2
                        elif colord1==-3 and colord2==3 and colormother ==8:
                            d1colup1=0
                            d1colup2=colup2
                            d2colup1=colup1
                            d2colup2=0
                        elif colord1==-3 and colord2==3 and colormother ==1:
                            maxcol+=1
                            d1colup1=0
                            d1colup2=maxcol
                            d2colup1=maxcol
                            d2colup2=0
                        elif colord1==3 and colord2==8 and colormother ==3:
                            maxcol+=1
                            d2colup1=colup1
                            d2colup2=maxcol
                            d1colup1=maxcol
                            d1colup2=0

                        elif colord2==3 and colord1==8 and colormother ==3:
                            maxcol+=1
                            d1colup1=colup1
                            d1colup2=maxcol
                            d2colup1=maxcol
                            d2colup2=0

                        elif colord1==-3 and colord2==8 and colormother ==-3:
                            maxcol+=1
                            d2colup2=colup2
                            d2colup1=maxcol
                            d1colup2=maxcol
                            d1colup1=0

                        elif colord2==-3 and colord1==8 and colormother ==-3:
                            maxcol+=1
                            d1colup2=colup2
                            d1colup1=maxcol
                            d2colup2=maxcol
                            d2colup1=0
                        elif colord1==-3 and colord2==-3 and colormother == 3:
                            maxcol+=2
                            d1colup1=0
                            d1colup2=maxcol
                            d2colup1=0
                            d2colup2=maxcol-1
                        elif (colord1==-3 and colord2==3 and colormother == 3) or\
                             (colord1==-3 and colord2==3 and colormother == -3):
                            maxcol+=2
                            d1colup1 = 0
                            d1colup2 = maxcol
                            d2colup1 = maxcol-1
                            d2colup2 = 0
                        elif (colord1==3 and colord2==-3 and colormother == 3) or\
                            (colord1==3 and colord2==-3 and colormother == -3):
                            maxcol+=2
                            d1colup1=maxcol
                            d1colup2=0
                            d2colup1=0
                            d2colup2=maxcol-1
                        elif colord1==3 and colord2==3 and colormother == -3:
                            maxcol+=2
                            d1colup1=maxcol
                            d1colup2=0
                            d2colup1=maxcol-1
                            d2colup2=0    
                        elif colord2==8 and colord1==8 and colormother ==8:
                            maxcol+=1
                            ran = random.random()
                            if ran> 0.5:
                                d1colup2=colup2
                                d1colup1=maxcol
                                d2colup2=maxcol
                                d2colup1=colup1
                            else:                            
                                d1colup2=maxcol
                                d1colup1=colup1
                                d2colup2=colup2
                                d2colup1=maxcol                        
                        else:
                            raise Exception, 'color combination not treated by MadSpin (yet). (%s,%s,%s)' \
                                % (colord1,colord2,colormother)
                        part_number+=1
                        index_d1_for_mom=decay_struct[part]['mg_tree'][-res-1][1]
                        mom=momenta_in_decay[index_d1_for_mom].copy()
                        #mom=decay_products[decay_struct[part]\
                        #            ["tree"][res]["d1"]["index"]]["momentum"]
                        pid=decay_struct[part]\
                                    ["tree"][res]["d1"]["label"]


                        indexd1=decay_struct[part]["tree"][res]["d1"]["index"]
                        if ( indexd1>0):
                            hel=helicities[index_d1_for_mom-1]
                            istup=1
                            external+=1
                            if not use_mc_masses or abs(pid) not in self.MC_masses:
                                mass=self.banner.get('param_card','mass', abs(pid)).value
                            else:
                                mass=self.MC_masses[abs(pid)]
                        else:
                            hel=0.
                            decay_struct[part]["tree"][indexd1]["colup1"]=d1colup1
                            decay_struct[part]["tree"][indexd1]["colup2"]=d1colup2
                            istup=2                    
                            mass=mom.m
                            map_to_part_number[indexd1]=part_number 
 
                        mothup1=map_to_part_number[res]
                        mothup2=map_to_part_number[res]
                        decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":d1colup1,"colup2":d1colup2,"momentum":mom,\
                                "mass":mass,"helicity":hel}
                        decayed_event.event2mg[part_number]=part_number

                        part_number+=1
                        index_d2_for_mom=decay_struct[part]['mg_tree'][-res-1][2]
                        mom=momenta_in_decay[index_d2_for_mom].copy()

                        #mom=decay_products[decay_struct[part]["tree"][res]["d2"]\
                        #                   ["index"]]["momentum"]
                        pid=decay_struct[part]["tree"][res]["d2"]\
                                           ["label"]

                        indexd2=decay_struct[part]["tree"][res]["d2"]["index"]
                        if ( indexd2>0):
                            hel=helicities[index_d2_for_mom-1]
                            istup=1
                            external+=1
                            if not use_mc_masses or abs(pid) not in self.MC_masses:
                                mass=self.banner.get('param_card','mass', abs(pid)).value
                            else:
                                mass=self.MC_masses[abs(pid)]
                        else:
                            hel=0.
                            istup=2
                            decay_struct[part]["tree"][indexd2]["colup1"]=d2colup1
                            decay_struct[part]["tree"][indexd2]["colup2"]=d2colup2
                            mass=mom.m
                            map_to_part_number[indexd2]=part_number

                        mothup1=map_to_part_number[res]
                        mothup2=map_to_part_number[res]
                        decayed_event.particle[part_number]={"pid":pid,"istup":istup,\
                           "mothup1":mothup1,"mothup2":mothup2,"colup1":d2colup1,\
                           "colup2":d2colup2,\
                           "momentum":mom,"mass":mass,"helicity":hel}

                        decayed_event.event2mg[part_number]=part_number

                
            
            else: # resonance in the production event
                part=curr_event.event2mg[index]
                part_number+=1
                decayed_event.particle[part_number]=curr_event.resonance[part]
                decayed_event.event2mg[part_number]=part_number
#        Here I need to check that the daughters still have the correct mothup1 and mothup2
                for part in curr_event.resonance.keys():
                    mothup1=curr_event.resonance[part]["mothup1"]         
                    mothup2=curr_event.resonance[part]["mothup2"] 
                    if mothup1==index:
                        if mothup2!=index: print "Warning: mothup1!=mothup2"
                        curr_event.resonance[part]["mothup1"]=part_number
                        curr_event.resonance[part]["mothup2"]=part_number
                for part in curr_event.particle.keys():
                    mothup1=curr_event.particle[part]["mothup1"]         
                    mothup2=curr_event.particle[part]["mothup2"] 
                    if mothup1==index:
                        if mothup2!=index: print "Warning: mothup1!=mothup2"
                        curr_event.particle[part]["mothup1"]=part_number
                        curr_event.particle[part]["mothup2"]=part_number

        decayed_event.nexternal=part_number        
        return decayed_event       


    def add_loose_decay(self):
        """ in presence of multiprocess with multiple decay options all the 
        BR might not be identical. In such case, the total number of events should
        drop such that the events file is still a unweighted physical events sample.
        This routines add null decay (=> not written events) if appropriate."""
        
        first = True
        max_br = max([m['total_br'] for m in self.all_ME.values()])
        if max_br >= 1:
            if max_br > 1.0001:
                raise MadSpinError, 'BR is larger than one.'
            max_br = 1
        for production in self.all_ME.values():
            if production['total_br'] < max_br:
                if production['total_br'] > 0.9999:
                    continue
                if first:
                    first = False
                    min_br = min([m['total_br'] for m in self.all_ME.values()])
                    logger.info('''All production process does not have the same total Branching Ratio.
                    Therefore the total number of events after decay will be lower than the original file.
                    [max_br = %s, min_br = %s]''' % (max_br, min_br),'$MG:color:BLACK')
                fake_decay = {'br': max_br - production['total_br'], 
                              'path': None, 'matrix_element': None, 
                              'finals': None, 'base_order': None,
                              'decay_struct':None, 'decay_tag': None}
                production['decays'].append(fake_decay)
                production['total_br'] = max_br




    def write_banner_information(self, eff=1):
        
        ms_banner = ""
        cross_section = True # tell if possible to write the cross-section in advance
        total_br = []
        for production in self.all_ME.values():
            one_br = 0
            for decay in production['decays']:
                if not decay['decay_tag']:
                    cross_section = False
                    one_br += decay['br']
                    continue
                ms_banner += "# %s\n" % ','.join(decay['decay_tag']).replace('\n',' ')
                ms_banner += "# BR: %s\n# max_weight: %s\n" % (decay['br'], decay['max_weight'])
                one_br += decay['br']
            total_br.append(one_br)
        
        if __debug__:
            for production in self.all_ME.values():
                assert production['total_br'] - min(total_br) < 1e-4
        
        self.branching_ratio = max(total_br) * eff

        #self.banner['madspin'] += ms_banner
        # Update cross-section in the banner
        if 'mggenerationinfo' in self.banner:
            mg_info = self.banner['mggenerationinfo'].split('\n')
            for i,line in enumerate(mg_info):
                if 'Events' in line:
                    if eff == 1:
                        self.err_branching_ratio = 0
                        continue
                    initial_event = int(mg_info[i].split()[-1])
                    nb_event =  int(initial_event * eff) 
                    mg_info[i] = '#  Number of Events        :       %i' % nb_event
                    if eff >0.5:
                        self.err_branching_ratio = max(total_br) * math.sqrt(initial_event - eff * initial_event)/initial_event
                    else:
                        self.err_branching_ratio = max(total_br) * math.sqrt(eff * initial_event)/initial_event
                    continue
                if ':' not in line:
                    continue
                info, value = line.rsplit(':',1)
                try:
                    value = float(value)
                except:
                    continue
                if cross_section:
                    mg_info[i] = '%s : %s' % (info, value * self.branching_ratio)            
                else:
                    mg_info[i] = '%s : %s' % (info, value * self.branching_ratio)
                self.banner['mggenerationinfo'] = '\n'.join(mg_info)
                
            
        
        if 'init' in self.banner:
            new_init =''
            for line in self.banner['init'].split('\n'):
                if len(line.split()) != 4:
                    new_init += '%s\n' % line
                else:
                    data = [float(nb) for nb in line.split()]
                    data[:3] = [ data[i] * self.branching_ratio for i  in range(3)]
                    new_init += ' %.12E %.12E %.12E %i\n' % tuple(data)
            self.banner['init'] = new_init
        self.banner.write(self.outputfile, close_tag=False)        
        
    def terminate_fortran_executables(self, path_to_decay=0 ):
        """routine to terminate all fortran executables"""

        if not path_to_decay:
            for (mode, path) in self.calculator:
                if mode=='decay':
                    external = self.calculator[(mode, path)]
                    external.terminate()
                    del external
                elif mode=='full':
                    stdin_text="5 0 0 0 \n"  # before closing, write down the seed 
                    external = self.calculator[('full',path)]
                    external.stdin.write(stdin_text)
                    ranmar_state=external.stdout.readline()
                    ranmar_file=pjoin(path,'ranmar_state.dat')
                    ranmar=open(ranmar_file, 'w')
                    ranmar.write(ranmar_state)
                    ranmar.close()
                    external.stdin.close()
                    external.stdout.close()
                    external.terminate()
                    del external
        else:
            try:
                external = self.calculator[('full', path_to_decay)]
            except Exception:
                pass
            else:
                stdin_text="5 0 0 0"
                external.stdin.write(stdin_text)
                external.stdin.close()
                external.stdout.close()
                external.terminate()       
                del external

        self.calculator = {}
