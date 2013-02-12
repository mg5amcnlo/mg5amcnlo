#! /usr/bin/env python

from __future__ import division

################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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
import madgraph.iolibs.files as files
import aloha
logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr

import random 
import math
from madgraph import MG5DIR
import madgraph.various.misc as misc
#import time

class Event:
    """ class to read an event, record the information, write down the event in the lhe format.
            This class is used both for production and decayed events"""

    def __init__(self, inputfile=None):
        """Store the name of the event file """
        self.inputfile=inputfile
        self.particle={}

    def give_procdef(self, pid2label):
        """ Return a string with the process in the format i j > k l ... """
        proc_line=""
        initial=0
        for part in range(1,len(self.particle)+1):
            if self.particle[part]["istup"]==-1:
                initial+=1
                proc_line+=pid2label[self.particle[part]["pid"]]+" "
                if initial==2 :
                    proc_line+="> "

            if self.particle[part]["istup"]==1:
                proc_line+=pid2label[self.particle[part]["pid"]]+" "
        return proc_line
 

    def get_map_resonances(self, branchindex2label,pid2label):
        """
           decay_processes is a dictionary 1 : "A1"
                                           2 : "A2" 
                                           ...
          returns a map  {mg index : index of the branch}
          for each mg index associated with a resonance to be decayed
        """
        # first build a dictionary {}

        dico_map={}
        dico_label={}

#        dico_symmetry_fac={}  # dico {label : symmetry factor}
        branch_not_yet_found=branchindex2label.keys()
        branch_found=[]

         
        for index in self.event2mg.keys():
           if self.event2mg[index]>0:
                part=self.event2mg[index]
                pid=self.particle[part]['pid']
                found=0
                for  index, id in enumerate(branch_not_yet_found):
                    if pid2label[pid]==branchindex2label[id]:
                        found=1
                        dico_map[part]=id
                        label=pid2label[pid]
                        dico_label[part]=label 
#                        if not dico_symmetry_fac.has_key(label):
#                            dico_symmetry_fac[label]=1
#                        else:
#                            dico_symmetry_fac[label]+=1

                        branch_found.append(id)
                        to_delete=index
                        break
                if found: del branch_not_yet_found[to_delete]

                if not found:
                    for index in branch_found:  
                        if pid2label[pid]==branchindex2label[index]:
                            dico_map[part]=index
                            dico_label[part]=pid2label[pid]

#       now get the symmetry factor:
#        symm_fac=1.0
        #print dico_symmetry_fac
#        for label in dico_symmetry_fac.keys():
#            for index in range(2,dico_symmetry_fac[label]+1):
#                symm_fac=symm_fac*(index)

        return dico_map, dico_label #, symm_fac


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
        for id in range(len(self.particle)):
            part = map_event[id] + 1
            if self.particle[part]["istup"]<2:
                mom=[self.particle[part]["momentum"].E, \
                     self.particle[part]["momentum"].px,\
                     self.particle[part]["momentum"].py,\
                     self.particle[part]["momentum"].pz]
                mom = self.particle[part]["momentum"]
                p.append(mom)
                string+=str(self.particle[part]["momentum"].E)+" "+\
                        str(self.particle[part]["momentum"].px)\
                         +" "+str(self.particle[part]["momentum"].py)+\
                         " "+str(self.particle[part]["momentum"].pz)+"\n"
        return p, string 

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
        line="<event> \n"
        line1=' %2d %6d %+13.7e %14.8e %14.8e %14.8e' % \
        (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)
        line+=line1+"\n"
        for item in range(1,len(self.event2mg.keys())+1):
            part=self.event2mg[item]
            if part>0:
                particle_line=self.get_particle_line(self.particle[part])
            else:
                particle_line=self.get_particle_line(self.resonance[part])
            line+=particle_line        

        if self.diese:
            line += self.diese
        if self.rwgt:
            line += self.rwgt
        line+="</event> \n"
        return line


    def get_particle_line(self,leg):

        line=" %8d %2d %4d %4d %4d %4d %+13.7e %+13.7e %+13.7e %14.8e %14.8e %10.4e %10.4e" \
            % (leg["pid"], leg["istup"],leg["mothup1"],leg["mothup2"],\
               leg["colup1"],leg["colup2"],leg["momentum"].px,leg["momentum"].py,\
                leg["momentum"].pz,leg["momentum"].E, leg["momentum"].m,\
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
            logger.info("Got more than 2 daughters for one particles")
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
            if '<event>' in line:
                #start new_event
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
            
            
            if line_type == 'none':
                continue
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
                 

#class mass_n_width(dict):
#    """ dictionary to extract easily the mass ad the width of the particle in the model 
#            {name : {"mass": mass_value, "width": widht_value} }
#            I assume that masses and widths are real
#    """
#
#    def __init__(self, model, banner):
#        """ Fill the dictionary based on the information in the banner """
#
#        raise Exception, 'still in use'
#        self.model = model
#        self.banner = banner
#        for particle in model["particles"]:
#            self[particle["name"]]={"mass":0.0, "width":0.0}
#            if particle["mass"]!="ZERO":
#                self[particle["name"]]["mass"]=self.get_mass(particle["pdg_code"])
#            if particle["width"]!="ZERO":
#                self[particle["name"]]["width"]=self.get_width(particle["pdg_code"])
#
#    def get_mass(self,pid):
#        """ extract the mass of particle with PDG code=pid from the banner"""
#        try: 
#            mass=self.banner.get('param_card', 'mass', pid)
#        except Exception, error:
#            mass=0.0
#        return mass
#    
#
#    def get_width(self,pid):
#        """ extract the width of particle with PDG code=pid from the banner"""
#        try:
#            width=self.banner.get('param_card', 'decay', pid)
#        except Exception, error:
#            width=0.0
#        return width


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
            for c_nb,leg in enumerate(proc.get('legs')):
                if c_nb == 0:
                    continue
                self["tree"][propa_id]["d%s" % c_nb] = {}
                c_pid = leg.get('id')
                self["tree"][propa_id]["d%s" % c_nb]["label"] = c_pid
                self["tree"][propa_id]["d%s" % c_nb]["labels"] = [c_pid]
                if c_pid in to_decay:
                    self["tree"][propa_id]["d%s" % c_nb]["index"] = propa_id-1
                    self.nb_decays += 1
                    add_decay(to_decay[c_pid].pop(), propa_id-1)
                else:
                    self.nexternal += 1
                    self["tree"][propa_id]["d%s" % c_nb]["index"] = self.nexternal
        
        # launch the recursive loop
        add_decay(process)

    def generate_momenta(self,mom_init,ran, pid2width,pid2mass,BW_cut):
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
        sol_nb = 0
        
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
                    width=pid2width(pid)*pid2mass(pid)/(4.0*pid2mass(pid)**2)     #/mA**2
                    m, jac=self.transpole(pole,width, BW_cut)
                    m = math.sqrt(m * 4.0 * pid2mass(pid)**2)
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
                    return 0, 0 # If that happens, throw away the DC phase-space point ...
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
            if not sol_nb:
                sol_nb = random.randint(0,len(self["tree"][res]["d1"]["labels"])-1)
            
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

        return index2mom, weight
        
    def transpole(self,pole,width, BW_cut):

        """ routine for the generation of a p^2 according to 
            a Breit Wigner distribution
            the generation window is 
            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
        """

        zmin = math.atan(-BW_cut)/width
        zmax = math.atan(BW_cut)/width

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
    

#
#class dc_branch(dict):
#    """ A dictionary to record information necessary to decay particles 
#            { -1 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },    
#                -2 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },
#                ....
#            }
#    """
#
#    def __init__(self, process_line,model,banner,check):
#
#        self.banner=banner
#        self.label2pid=label2pid(model)
#        list_decays=process_line.split(",")
#        self.nb_decays=len(list_decays)
#        self["m_label2index"]={}
#        self["m_index2label"]={}
#        
#        for dc_nb, dc in enumerate(list_decays):
#            if dc.find(">")<0: logger.warning('warning: invalid decay chain syntax')
#            mother=dc[:dc.find(">")].replace(" ","")
#            self["m_label2index"][mother]=-dc_nb-1
#            self["m_index2label"][-dc_nb-1]=mother
#            
#        self["nexternal"]=self.find_tree(list_decays)
#        if (check): self.check_parameters() 
#        # here I check that the relevant masses and widths can 
#        # be extracted from the banner    
#
#    def check_parameters(self):
#        for res in range(-1,-self.nb_decays-1,-1): 
#            d1=self["tree"][res]["d1"]["index"]
#            d2=self["tree"][res]["d2"]["index"]
#            if (d1>0): 
#                try: 
#                    logger.info('Mass of particle with id '\
#                                +str(self.label2pid[self["tree"][res]["d1"]["label"]]))
#                    logger.info(self.banner.param["Block mass"]\
#                         [abs(self.label2pid[self["tree"][res]["d1"]["label"]])])
#                except Exception, error:
#                    logger.info('The mass of particle with id '\
#                                +str(self.label2pid[self["tree"][res]["d1"]["label"]]))
#                    logger.info('was not defined in the param_card.dat') 
#                    mass=raw_input("Please enter the mass: ")
#                    self.banner.param["Block mass"][abs(self.label2pid[self["tree"][res]["d1"]["label"]])]=mass
#
#            if (d2>0):
#                try:
#                    logger.info('Mass of particle with id '\
#                                +str(self.label2pid[self["tree"][res]["d2"]["label"]]))
#                    logger.info(     self.banner.param["Block mass"]\
#                         [abs(self.label2pid[self["tree"][res]["d2"]["label"]])])
#                except Exception, error:
#                    logger.info('The mass of particle with id '\
#                                +str(self.label2pid[self["tree"][res]["d2"]["label"]]))
#                    logger.info('was not defined in the param_card.dat')
#                    mass=raw_input("Please enter the mass: ")
#                    self.banner.param["Block mass"][abs(self.label2pid[self["tree"][res]["d2"]["label"]])]=mass
#
#
#    def transpole(self,pole,width, BW_cut):
#
#        """ routine for the generation of a p^2 according to 
#            a Breit Wigner distribution
#            the generation window is 
#            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
#        """
#
#        zmin = math.atan(-BW_cut)/width
#        zmax = math.atan(BW_cut)/width
#
#        z=zmin+(zmax-zmin)*random.random()
#        y = pole+width*math.tan(width*z)
#
#        jac=(width/math.cos(width*z))**2*(zmax-zmin)
#        return y, jac
#
#    def generate_momenta(self,mom_init,ran, pid2width,pid2mass,resonnances,BW_cut):
#        """Generate the momenta in each decay branch 
#             If ran=1: the generation is random, with 
#                                     a. p^2 of each resonance generated according to a BW distribution 
#                                     b. cos(theta) and phi (angles in the rest frame of the decaying particle)
#                                            are generated according to a flat distribution (no grid)
#                                 the phase-space weight is also return (up to an overall normalization)
#                                 since it is needed in the unweighting procedure
#
#             If ran=0: evaluate the momenta based on the previously-generated p^2, cos(theta) 
#                                 and phi in each splitting.    
#                                 This is used in the reshuffling phase (e.g. when we give a mass to gluons 
#                                 in the decay chain )
#        """
#        index2mom={}
##      pid2mom={}    # a dict { pid : {"status":status, "momentum":momentum}    }
#
#        index2mom[-1]={}
#        index2mom[-1]["momentum"]=mom_init
#        if index2mom[-1]['momentum'].m < 1e-3:
#            logger.debug('Decaying particle with m> 1e-3 GeV in generate_momenta')
#        index2mom[-1]["pid"]=self.label2pid[self["m_index2label"][-1]]
#        index2mom[-1]["status"]=2
#        weight=1.0
#        for res in range(-1,-self.nb_decays-1,-1): 
##     Here mA^2 has to be set to p^2:
## 
##     IF res=-1:
##         p^2 has been either fixed to the value in the 
##         production lhe event, or generated according to a    Breit-Wigner distr. 
##         during the reshuffling phase of the production event
##         -> we just need to read the value here
##     IF res<-1:
##         p^2 has been generated during the previous iteration of this loop 
##         -> we just need to read the value here
#
#            mA=index2mom[res]["momentum"].m
#            if mA < 1.0e-3:
#                logger.debug('Warning: decaying parting with m<1 MeV in generate_momenta ')
#            #print index2mom[res]["momentum"].px
#            #print index2mom[res]["momentum"].py
#            #print index2mom[res]["momentum"].pz
#            #print index2mom[res]["momentum"].E
#            #print mA
#            #print " "
#
#            d1=self["tree"][res]["d1"]["index"]
#            d2=self["tree"][res]["d2"]["index"]
#
##         For the daughters, the mass is either generate (intermediate leg + BW mode on)
##         or set to the pole mass (external leg or BW mode off)
##         If ran=0, just read the value from the previous generation of momenta 
##                             (this is used for reshuffling purposes) 
#            if d1>0 or not BW_cut :
#                mB=float(self.banner.get('param_card', 'mass', 
#                         abs(self.label2pid[self["tree"][res]["d1"]["label"]])).value)
#            elif ran==0:    # reshuffling phase
#                mB=self["tree"][res]["d1"]["mass"]
#            else:
#                pid=self.label2pid[self["tree"][res]["d1"]["label"]]
##             NOTE: here pole and width are normalized by 4.0*mB**2,
##             Just a convention
#                pole=0.25         #pid2mass[pid]**2/mA**2
#                width=pid2width[pid]*pid2mass[pid]/(4.0*pid2mass[pid]**2)     #/mA**2
#                mB, jac=self.transpole(pole,width, BW_cut)
#                mB=math.sqrt(mB*4.0*pid2mass[pid]**2)
##             record the mass for the reshuffling phase, 
##             in case the point passes the reweighting creteria
#                self["tree"][res]["d1"]["mass"]=mB
##             update the weigth of the phase-space point
#                weight=weight*jac
#
#            if d2>0 or not BW_cut:
#                mC=float(self.banner.get('param_card', 'mass', 
#                    abs(self.label2pid[self["tree"][res]["d2"]["label"]])).value)
#            elif ran==0:
#                mC=self["tree"][res]["d2"]["mass"]
#            else:
#                pid=self.label2pid[self["tree"][res]["d2"]["label"]]
##             NOTE: here pole and width are normalized by 4.0*mC**2,
##             Just a convention
#                pole=0.25    #pid2mass[pid]**2/mA**2
#                width=pid2width[pid]*pid2mass[pid]/(4.0*pid2mass[pid]**2) #mA**2
#                mC, jac=self.transpole(pole,width, BW_cut)
#                mC=math.sqrt(mC*4.0*pid2mass[pid]**2)
##             record the mass for the reshuffling phase, 
##             in case the point passes the reweighting creteria
#                self["tree"][res]["d2"]["mass"]=mC
##             update the weigth of the phase-space point
#                weight=weight*jac
#
#
#                if (mA<mB+mC):
#                    logger.debug('mA<mB+mC in generate_momenta')
#                    logger.debug('mA = %s' % mA)
#                    return 0, 0 # If that happens, throw away the DC phase-space point ...
#                        # I don't expect this to be inefficient, since there is a BW cut
#
#            if ran==1:
#                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, mB,mC)
##             record the angles for the reshuffling phase, 
##             in case the point passes the reweighting creteria
#                self["tree"][res]["costh"]=decay_mom.costh
#                self["tree"][res]["sinth"]=decay_mom.sinth
#                self["tree"][res]["cosphi"]=decay_mom.cosphi
#                self["tree"][res]["sinphi"]=decay_mom.sinphi
#            else:
##             we are in the reshuffling phase, 
##             so we read the angles that have been stored from the 
##             previous phase-space point generation
#                costh=self["tree"][res]["costh"]
#                sinth=self["tree"][res]["sinth"]
#                cosphi=self["tree"][res]["cosphi"]
#                sinphi=self["tree"][res]["sinphi"]
#                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, mB,mC,\
#                                 costh_val=costh, sinth_val=sinth, cosphi_val=cosphi, \
#                                 sinphi_val=sinphi)
#
##         record the momenta for later use
#            index2mom[self["tree"][res]["d1"]["index"]]={}
#            index2mom[self["tree"][res]["d1"]["index"]]["momentum"]=decay_mom.momd1
#            index2mom[self["tree"][res]["d1"]["index"]]["pid"]=self.label2pid[self["tree"]\
#                                                                    [res]["d1"]["label"]]
#
#            index2mom[self["tree"][res]["d2"]["index"]]={}
#            index2mom[self["tree"][res]["d2"]["index"]]["momentum"]=decay_mom.momd2
#            index2mom[self["tree"][res]["d2"]["index"]]["pid"]=self.label2pid[self["tree"]\
#                                                                    [res]["d2"]["label"]]
#
#            if (self["tree"][res]["d1"]["index"]>0):
#                index2mom[self["tree"][res]["d1"]["index"]]["status"]=1
#            else:
#                index2mom[self["tree"][res]["d1"]["index"]]["status"]=2
#            if (self["tree"][res]["d2"]["index"]>0):
#                index2mom[self["tree"][res]["d2"]["index"]]["status"]=1
#            else:
#                index2mom[self["tree"][res]["d2"]["index"]]["status"]=2
#
#        return index2mom, weight
#
#    def find_tree(self,list_decays):
#        """ 
#            record the topology of the decay chain in suitable variables
#            This is roughly the equivalent of the configs.inc file in madevent
#        """
#        self["tree"]={}
#        nexternal=0
#        for mother in range(-1, -len(list_decays)-1, -1):
#            self["tree"][mother]={}
#            self["tree"][mother]["d1"]={}
#            self["tree"][mother]["d2"]={}
# 
#            #print "filling the tree"
#            #print "res "+str(mother)
#            dc_nb=-(mother)-1
#            daughters=list_decays[dc_nb][list_decays[dc_nb].find(">")+1:].split()
#            self["tree"][mother]["d1"]["label"]=daughters[0]
#            self["tree"][mother]["d2"]["label"]=daughters[1]
#
#            if self["m_label2index"].has_key(daughters[0]):
#                self["tree"][mother]["d1"]["index"]=self["m_label2index"][daughters[0]]
#            else:
#                nexternal=nexternal+1
#                self["tree"][mother]["d1"]["index"]=nexternal
#            if self["m_label2index"].has_key(daughters[1]):    
#                self["tree"][mother]["d2"]["index"]=self["m_label2index"][daughters[1]]
#            else:
#                nexternal=nexternal+1
#                self["tree"][mother]["d2"]["index"]=nexternal        
#        return nexternal
#
#    def print_branch(self):
#        """Print the decay chain structure (for debugging purposes)"""
#        length=len(self["tree"])
#        for res in range(-1,-length-1, -1):
#            logger.info('Decay '+str(res))
#            #print "Mother: "+self["tree"][res]
#            logger.info( 'd1: '+str(self["tree"][res]["d1"]["label"])+\
#                         '    '+str(self["tree"][res]["d1"]["index"]))
#            logger.info('d2: '+str(self["tree"][res]["d2"]["label"])+\
#                        '     '+str(self["tree"][res]["d2"]["index"]))
#    
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




#class production_topo_old(dict):
#    """ A dictionnary to record information about a given topology of a production event 
#
#                self["branchings"] is a list of the branchings defining the topology (see class branching)
#                self["get_mass2"] is a dictionnary {index -> mass**2 of the corresponding particle} 
#                self["get_momentum"] is a dictionnary {index -> momentum of the corresponding particle} 
#                self["get_id"] is a dictionnary {index -> pid the corresponding particle} 
#
#            Note: index= "madgraph-like" numerotation of the particles
#    """
#
#    def __init__(self):
#        """ Initialise the dictionaries+list used later on to record the information
#                about the topology of a production event.
#                Note that self["branchings"] is a list, 
#                as it makes it easier to scan the topology in the ascendent order
#        """
#        self["branchings"]=[]
#        self["get_mass2"]={}
#        self["get_momentum"]={}
#        self["get_id"]={}
#
#    def add_one_branching(self,index_propa, index_d1,index_d2,type_propa):
#        """ add the information of one splitting in the topology """
#        branch=branching(index_propa, index_d1,index_d2,type_propa)
#        self["branchings"].append(branch)
#
#
#    def topo2event(self,event,to_decay):
#        """This routine is typically called and the end of the reshuffling phase.
#             The momenta in the topology were reshuffled in a previous step, and now they are copied 
#             back to the production event in this routine 
#        """
##     start with external legs
#        for part in range(1,len(event.particle)+1):
#            event.particle[part]["momentum"]=self["get_momentum"][part].copy()
#            if part in to_decay:
#                if event.particle[part]["momentum"].m < 1.0e-3:
#                    logger.debug('Decaying particle with a mass of less than 1 MeV in topo2event')
#            #print part 
#            #print self["get_momentum"][part].nice_string()
#            #print event.particle[part]["momentum"].nice_string()
#
#    def print_topo(self):
#        """Print the structure of the topology    """
#        for branch in self["branchings"]:
#            d1=branch["index_d1"]
#            d2=branch["index_d2"]
#            propa=branch["index_propa"]
#            line=str(propa)+" > "
#            line+=str(d1)+" + "
#            line+=str(d2)+" ,    type="
#            line+=branch["type"]
##            try:
#                #print "momentum propa"
#                #print self["get_momentum"][propa].nice_string()
#                #print "P^2:    "+str(self["get_mass2"][propa])
#                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][propa])))
#                #print "momentum d1"
#                #print self["get_momentum"][d1].nice_string()
#                #print "P^2:    "+str(self["get_mass2"][d1])
#                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][d1])))
#                #print "momentum d2"
#                #print self["get_momentum"][d2].nice_string()
#                #print "P^2:    "+str(self["get_mass2"][d2])
#                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][d2])))
##            except Exception, error:
#                #print "topology not yet dressed" 
#
#    def dress_topo_from_event(self,event,to_decay):
#        """ event has been read from the production events file,
#                use these momenta to dress the topology
#        """
#
##     start with external legs
#        for part in range(1,len(event.particle)+1):
#            self["get_momentum"][part]=event.particle[part]["momentum"].copy()
#            self["get_mass2"][part]=event.particle[part]["mass"]**2
#            self["get_id"][part]=event.particle[part]["pid"]
#            if part in to_decay :
#                if self["get_momentum"][part].m<1e-3:
#                    logger.debug\
#                    ('decaying particle with m < 1MeV in dress_topo_from_event (1)')
#                if self["get_mass2"][part]<1e-3:
#                    logger.debug\
#                    ('decaying particle with m < 1MeV in dress_topo_from_event (2)')
#
##    now fill also intermediate legs
##    Don't care about the pid of intermediate legs
#        for branch in self["branchings"]:
#            part=branch["index_propa"]
#            if branch["type"]=="s":
#                mom_propa=self["get_momentum"][branch["index_d1"]].add(self["get_momentum"][branch["index_d2"]])
#            elif branch["type"]=="t":
#                mom_propa=self["get_momentum"][branch["index_d1"]].subtract(self["get_momentum"][branch["index_d2"]])
#            self["get_momentum"][part]=mom_propa
#            self["get_mass2"][part]=mom_propa.sq
#
## Also record shat and rapidity, since the initial momenta will also be reshuffled
##
#        p1=self["get_momentum"][1].copy()
#        p2=self["get_momentum"][2].copy()
#        ptot=p1.add(p2)
#        self["shat"]=ptot.sq
#        self["rapidity"]=0.5*math.log((ptot.E+ptot.pz)/(ptot.E-ptot.pz))
#
#    def reshuffle_momenta(self):
#        """
#            At this stage, 
#            - the topo should be dressed with momenta.
#            - the angles should be already extracted.
#            - the masses should be corrected.
#            This routine scan all branchings:
#                 - first consider the t-branchings, go to the appropriate frame, 
#                     modify the three-momenta to account for the corrected masses,
#                 - then consider the s-branchings, go to the approriate frame,
#                     rescale the three-momenta to account for the corrected masses.
#        """
# 
##     step number one: need to check if p^2 of each propa
##     is ok with the new set of masses ...
##     we first check this for all the s-branchings
#
##     Close to threshold, problems may occur: 
##         e.g. in A>B+C, one may have mA>mB+mC
##     Currently, if this happens, I throw away the point
##     but I am not sure what is the best prescription here ... 
#
#        for  branch in self["branchings"]:
#             
##            no need to consider t-branching here:
#            if branch["type"]!="s":
#                    continue
#            d1=branch["index_d1"]
#            d2=branch["index_d2"]
#            propa=branch["index_propa"]
#
#            MA=math.sqrt(self["get_mass2"][propa])
#            MB=math.sqrt(self["get_mass2"][d1])
#             #print self["get_mass2"][d2]
#            MC=math.sqrt(self["get_mass2"][d2])
#            if MA < MB + MC:
#                 #print "WARNING: s-channel propagator with too low p^2 "
#                 #print "in branch "+str(iter)
#                 #print "MA = "+str(MA)
#                 #print "MB = "+str(MB)
#                 #print "MC = "+str(MC)
#                 #print "throw away the point"
#
#                 #print "increase the value of p^2"
##                 self["get_mass2"][propa]=(MC+MB+iota)**2 
##                 if iter==len(self["branchings"])-1:             # if last branching, needs to 
##                        self["shat"]=self["get_mass2"][propa]    # set shat to the new value of MA**2 
#                return 0
#
#
#     
##     then loop over all t-channels
##     and re-genenate the "d2" daughter in each of these branchings
#        got_a_t_branching=0
#        for nu, branch in enumerate(self["branchings"]):
#
##            no need to consider the last branching in this loop:
#            if(nu==len(self["branchings"])-1): break
#
##            no need to scan the s-branching now
#            if branch["type"]!="t":
#                    continue
#
#            got_a_t_branching=1
#            # t-channel sequence: A+B > 1 + 2,     r= pa-p1
#            ida=branch["index_d1"]
#            idb=2
#            id1=branch["index_d2"]
#            res=branch["index_propa"]
#            # go to the rest frame of    A+B
#            # set momenta A, B, 1
#            pa = self["get_momentum"][ida]
#            pb = self["get_momentum"][idb]
#            p1 = self["get_momentum"][id1]
##            set masses
#            ma2=self["get_mass2"][ida]
#            if (self["get_mass2"][id1]>=0):
#                #print "m1^2, t-branch "+str(iter)
#                #print self["get_mass2"][id1]
#                m1=math.sqrt(self["get_mass2"][id1])
#            else:
#                 #print "WARNING: m1^2 is negative for t-branching "+str(iter)
#                 #print self["get_mass2"][id1]
#                 #print "throw away the point"
#                return 0
#            m2=branch["m2"]
#            t=self["get_mass2"][res]
#
#            # express momenta p1 and pa in A+B CMS system
#            pboost=self["get_momentum"][2].add(self["get_momentum"][branch["index_d1"]])
#            pboost.px=-pboost.px
#            pboost.py=-pboost.py
#            pboost.pz=-pboost.pz
##            p1_cms=p1.boost(pboost)
#            pa_cms=pa.boost(pboost)
#
##             determine the magnitude of p1 in the cms frame
#            Esum=pboost.sq
#
#            if Esum>0 :
#                Esum=math.sqrt(Esum)
#            else:
#                 #print "WARNING: (pa+pb)^2 is negative for t-branching "
#                return 0
#            md2=(m1+m2)*(m1-m2)
#            ed=md2/Esum
#            if (m1*m2==0) :
#                pp=(Esum-abs(ed))*0.5
#            else:
#                pp=(md2/Esum)**2-2.0*(m1**2+m2**2)+Esum**2
#                if pp>0 :
#                    pp=0.5*math.sqrt(pp)
#                else:
#                        #print "WARNING: cannot get the momentum of p1 in t-branching    "+str(iter)
#                        #print "pp is negative : "+ str(pp)
#                        #print "m1: "+ str(m1)
#                        #print "m2: "+ str(m2)
#                        #print "id1: "+ str(id1)
#                        #print "throw away the point"
#                    return 0
#
##                Now evaluate p1
#            E_acms=pa_cms.E
#            p_acms=math.sqrt(pa_cms.mod2)
#
#            p1E=(Esum+ed)*0.5
#            if p1E < m1: 
##                logger.warning('E1 is smaller than m1 in t-branching')
##                logger.warning('Try to reshuffle the momenta once more')
#                return 0
#            p1z=-(m1*m1+ma2-t-2.0*p1E*E_acms)/(2.0*p_acms)
#            ptsq=pp*pp-p1z*p1z
#
#            if (ptsq<0 ): 
##                if pT=0 to begin with, one can get p1z slightly larger 
##                                than pp due to numerical uncertainties.
##                     In that case, just change slightly the invariant t
#                if (-ptsq/(pp*pp) <1e-6 ):
#                    oldt=t
#                    if (p1z>0):
#                        p1z=pp
#                    else:
#                        p1z=-pp
#                    pt=0.0
#                    t=m1*m1+ma2-2.0*p1E*E_acms+2.0*p_acms*p1z
#                    diff_t=abs((t-oldt)/t)*100
#                    if (diff_t>2.0): 
#                        logger.warning('t invariant was changed by '+str(diff_t)+' percents')
#                else:
#                     #print "WARNING: |p|^2 is smaller than p1z^2 in t-branching "+str(iter)
#                     #print "|p| : "+str(pp) 
#                     #print "pz : "+str(p1z) 
#                 #print "throw away the point"
#                 #print "Set pz^2=|p|^2 and recompute t"
#                 #print "previous t:"+str(-math.sqrt(abs(t)))+"^2"
##                 p1z=pp
##                 pt=0.0
##                 t=m1*m1+ma2-2.0*p1E*E_acms+2.0*p_acms*p1z
#                 #print "new t:"+str(-math.sqrt(abs(t)))+"^2"
#                    return 0
#            else:
#                pt=math.sqrt(pp*pp-p1z*p1z)
#
#            p1x=pt*branch["cosphi"]
#            p1y=pt*branch["sinphi"]
#
#            p1=momentum(p1E,p1x,p1y,p1z)
#
#            p1=p1.rot(pa_cms)
#            pboost.px=-pboost.px
#            pboost.py=-pboost.py
#            pboost.pz=-pboost.pz
#            p1=p1.boost(pboost)
#            pr=pa.subtract(p1)
#                #print " p1 is "
#                #print p1.nice_string()
#                #print " pr is "
#                #print pr.nice_string()
#            p2=(pa.add(pb)).subtract(p1)
#                #print " p2 is "
#                #print p2.nice_string()
##            now update momentum 
#            self["get_momentum"][id1]=p1.copy()
#            self["get_momentum"][res]=pr.copy()
#
#        # after we have looped over all t-branchings,
#        # p2 can be identified with the momentum of the second daughter of the last branching
#
#        if got_a_t_branching==1:
#            pid=self["branchings"][-1]["index_d2"]
#            self["get_momentum"][pid]=p2.copy()
#        #else: it means that there were no t-channel at all
#        #            last branching should be associated with shat 
#        #            note that the initial momenta will be reshuffled 
#        #            at the end of this routine 
#
#
##    Now we can    loop over all the s-channel branchings.
##    Need to start at the end of the list of branching
#        for branch in reversed(self["branchings"]):
#            if branch["type"]!="s":
#                    continue
#            d1=branch["index_d1"]
#            d2=branch["index_d2"]
#            propa=branch["index_propa"]
#            del self["get_momentum"][d1]
#            del self["get_momentum"][d2]
#            mA=math.sqrt(self["get_mass2"][propa])
#            mB=math.sqrt(self["get_mass2"][d1])
#            mC=math.sqrt(self["get_mass2"][d2])
#            mom=self["get_momentum"][propa]
#            costh=branch["costheta"]
#            sinth=branch["sintheta"]
#            cosphi=branch["cosphi"]
#            sinphi=branch["sinphi"]
#            decay2body=generate_2body_decay(mom, mA,mB,mC, \
#                                    costh_val=costh, sinth_val=sinth, \
#                                    cosphi_val=cosphi, sinphi_val=sinphi)
#            self["get_momentum"][d1]=decay2body.momd1.copy()
#            self["get_momentum"][d2]=decay2body.momd2.copy()
#
##    Need a special treatment for the 2 > 1 processes:
#        if len(self["get_mass2"])==3:
#            self["shat"]=self["get_mass2"][3]
#
##    Finally, compute the initial momenta
##    First generate initial momenta in the CMS frame
##    Then boost
#        mB=self["get_mass2"][1]
#        mC=self["get_mass2"][2]
#        if mB>0:
#            mB=math.sqrt(mB)
#        else:
#            mB=0.0
#        if mC>0:
#            mC=math.sqrt(mC)
#        else:
#            mC=0.0
#        mA=math.sqrt(self["shat"])
#        Etot=mA*math.cosh(self["rapidity"])
#        pztot=mA*math.sinh(self["rapidity"])
#        ptot=momentum(Etot,0.0,0.0,pztot)
#
#        decay2body=generate_2body_decay(ptot,mA,mB,mC, \
#                             costh_val=1.0, sinth_val=0.0, \
#                             cosphi_val=0.0, sinphi_val=1.0)
#
#        self["get_momentum"][1]=decay2body.momd1
#        self["get_momentum"][2]=decay2body.momd2
#
##    Need a special treatment for the 2 > 1 processes:
#        if len(self["get_momentum"])==3:
#            self["get_momentum"][3]=momentum(Etot,0.0,0.0,pztot)
#        
#        return 1
#
#    def extract_angles(self):
#        """ the topo should be dressed with momenta at this stage.
#                    Now: extract the angles characterizing each branching.
#                    For t-channel, the equivalent of cos(theta) is the mass of p2
#        """    
#
#        for nu, branch in enumerate(self["branchings"]):
#            if branch["type"]=="s":
#            # we have the decay A > B + C
#            # go to the rest frame of the decaying particle A, and extract cos theta and phi
#                propa=branch["index_propa"]
##                d1=branch["index_d1"]
#                d2=branch["index_d1"]
#                pboost=self["get_momentum"][propa].copy()
#                pboost.px=-pboost.px
#                pboost.py=-pboost.py
#                pboost.pz=-pboost.pz
##               pb_cms=self["get_momentum"][d1].boost(pboost)
#                pc_cms=self["get_momentum"][d2].boost(pboost)
#                mod_pc_cms=math.sqrt(pc_cms.mod2)
#                branch["costheta"]=pc_cms.pz/mod_pc_cms
#                branch["sintheta"]=math.sqrt(1.0-branch["costheta"]**2)
#                branch["cosphi"]=pc_cms.px/mod_pc_cms/branch["sintheta"]
#                branch["sinphi"]=pc_cms.py/mod_pc_cms/branch["sintheta"]
#
#            if branch["type"]=="t":
#            # we have a t-channel decay A + B > 1 + 2
#            # go to the rest frame of    A+B, and extract phi
#                    
#            # set momenta A, B, 1
#                pa = self["get_momentum"][branch["index_d1"]]
#                pb = self["get_momentum"][2]
#                p1 = self["get_momentum"][branch["index_d2"]]
#                p2=(pa.add(pb)).subtract(p1)
#
#                if (nu==len(self["branchings"])-1):
#                    # last t-channel branching, p2 should be zero
#                    check=p2.E**2+p2.mod2
#                    if check>1e-3: 
#                        logger.warning('p2 in the last t-branching is not zero')
#                        # If last t-channel branching, there is no angles to extract
#                    continue
#                elif (nu==len(self["branchings"])-2):
#                    # here m2 corresponds to the mass of the second daughter in the last splitting 
#                    part=self["branchings"][-1]["index_d2"]
#                    if self["get_mass2"][part]<0.0 :
#                        logger.warning('negative mass for particle '+str(part))
#                        logger.warning( self["get_mass2"][part])
#                    branch["m2"]=math.sqrt(self["get_mass2"][part])
#                elif p2.sq <0 and abs(p2.E)>1e-2: 
#                    logger.warning('GET A NEGATIVE M2 MASS IN THE T-BRANCHING '+str(iter))
#                    logger.warning( '(ROUTINE EXTRACT_ANGLES)')
#                    logger.warning( p2.nice_string() )
#                    logger.warning( p2.sq )
#                else:
#                    branch["m2"]=p2.m
#                 
#                # express momenta p1 and pa in A+B CMS system
#                pboost=self["get_momentum"][2].add(self["get_momentum"][branch["index_d1"]])
#                pboost.px=-pboost.px
#                pboost.py=-pboost.py
#                pboost.pz=-pboost.pz
#
#                if pboost.m<1e-6: 
#                    logger.warning('Warning: m=0 in T-BRANCHING '+str(iter))
#                    logger.warning(' pboost.nice_string()')
#
#                p1_cms=p1.boost(pboost)
#                pa_cms=pa.boost(pboost)
#
##               E_acms=pa.E
##               mod_acms=math.sqrt(pa_cms.mod2)
##
##               now need to go to the frame where pa is aligned along with the z-axis
#                p_rot=pa_cms.copy()
# 
#                p1_cmsrot=p1_cms.invrot(p_rot)
##               pa_cmsrot=pa_cms.invrot(p_rot)
##               now extract cosphi, sinphi
#                pt=math.sqrt(p1_cmsrot.px**2+p1_cmsrot.py**2)
#                if (pt>p1_cms.E*10e-10):
#                    branch["cosphi"]=p1_cmsrot.px/pt                 
#                    branch["sinphi"]=p1_cmsrot.py/pt
#                else:                 
#                    branch["cosphi"]=1                 
#                    branch["sinphi"]=0.0

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


    def topo2event(self,event):
        """This routine is typically called and the end of the reshuffling phase.
             The momenta in the topology were reshuffled in a previous step, and now they are copied 
             back to the production event in this routine 
        """
        to_decay = self.production["decaying"]
        #     start with external legs
        for part in range(1,len(event.particle)+1):
            event.particle[part]["momentum"]=self["get_momentum"][part].copy()
            if part in to_decay:
                if event.particle[part]["momentum"].m < 1.0e-3:
                    logger.debug('Decaying particle with a mass of less than 1 MeV in topo2event')


    def generate_BW_masses (self, pid2width, pid2mass,s):
        """Generate the BW masses of the particles to be decayed in the production event    """    
        # topo, to_decay, pid2name

        weight=1.0
        for part in self["get_id"].keys():
            pid=self["get_id"][part]
            if pid in self.production['decaying']:
                mass=0.25
                width=pid2width(pid)*pid2mass(pid)/(2.0*pid2mass(pid))**2
                virtualmass2, jac=self.transpole(mass,width)
                virtualmass2=virtualmass2*(2.0*pid2mass(pid))**2
                weight=weight*jac
                #print "need to generate BW mass of "+str(part)
                ##print "mass: "+str(pid2mass[pid])
                #print "width: "+str(pid2width[pid])
                #print "virtual mass: "+str(math.sqrt(virtualmass2))
                #print "jac: "+str(jac)
                old_mass=self["get_mass2"][part]
                self["get_mass2"][part]=virtualmass2
                # sanity check
                if pid2mass(pid)<1e-6:
                    logger.debug('A decaying particle has a mass of less than 1e-6 GeV')
# for debugg purposes:
                if abs((pid2mass(pid)-math.sqrt(self["get_mass2"][part]))/pid2mass(pid))>1.0 :
                    logger.debug('Mass after BW smearing affected by more than 100 % (1)') 
                    logger.debug('Pole mass: '+str(pid2mass(pid)))
                    logger.debug('Virtual mass: '+str(math.sqrt(self["get_mass2"][part])))
                                       
                #print self["get_mass2"]         

#    need to check if last branch is a t-branching. If it is, 
#    we need to update the value of branch["m2"]

        if len(self["branchings"])>0:  # Exclude 2>1 self.topologies
            if self["branchings"][-1]["type"]=="t":
                if self["branchings"][-2]["type"]!="t":
                    logger.debug('last branching is t-channel')
                    logger.debug('but last-but-one branching is not t-channel')
                else:
                    part=self["branchings"][-1]["index_d2"] 
                    if part >0: # reset the mass only if "part" refers to an external particle 
                        old_mass=self["branchings"][-2]["m2"]
                        self["branchings"][-2]["m2"]=math.sqrt(self["get_mass2"][part])
                        #sanity check

                        if abs(old_mass-self["branchings"][-2]["m2"])>1e-10:
                            if abs((old_mass-self["branchings"][-2]["m2"])/old_mass)>1.0 :
                                logger.debug('Mass after BW smearing affected by more than 100 % (2)')
                                logger.debug('Previous value: '+ str(old_mass))
                                logger.debug('New mass: '+ str((self["branchings"][-2]["m2"])))
                                try:
                                    pid=self["get_id"][part]
                                    logger.debug('pole mass: %s' % pid2mass[pid])
                                except Exception:
                                    pass
        return weight

    def transpole(self,pole,width):

        """ routine for the generation of a p^2 according to 
            a Breit Wigner distribution
            the generation window is 
            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
        """

        gap = float(self.options['BW_cut'])

        zmin = math.atan(-gap)/width
        zmax = math.atan(gap)/width

        z=zmin+(zmax-zmin)*random.random()
        y = pole+width*math.tan(width*z)

        jac=(width/math.cos(width*z))**2*(zmax-zmin)
        return y, jac



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
#            try:
                #print "momentum propa"
                #print self["get_momentum"][propa].nice_string()
                #print "P^2:    "+str(self["get_mass2"][propa])
                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][propa])))
                #print "momentum d1"
                #print self["get_momentum"][d1].nice_string()
                #print "P^2:    "+str(self["get_mass2"][d1])
                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][d1])))
                #print "momentum d2"
                #print self["get_momentum"][d2].nice_string()
                #print "P^2:    "+str(self["get_mass2"][d2])
                #print "root:    "+str(math.sqrt(abs(self["get_mass2"][d2])))
#            except Exception, error:
                #print "topology not yet dressed" 

    def dress_topo_from_event(self,event):
        """ event has been read from the production events file,
                use these momenta to dress the topology
        """

#     start with external legs
        for id in range(len(event.particle)):
            part = id + 1
            self["get_momentum"][id+1]=event.particle[part]["momentum"].copy()
            self["get_mass2"][id+1]=event.particle[part]["mass"]**2
            self["get_id"][id+1]=event.particle[part]["pid"]


#    now fill also intermediate legs
#    Don't care about the pid of intermediate legs
        for branch in self["branchings"]:
            part=branch["index_propa"]
            if branch["type"]=="s":
                mom_propa=self["get_momentum"][branch["index_d1"]].add(self["get_momentum"][branch["index_d2"]])
            elif branch["type"]=="t":
                mom_propa=self["get_momentum"][branch["index_d1"]].subtract(self["get_momentum"][branch["index_d2"]])
            self["get_momentum"][part]=mom_propa
            self["get_mass2"][part]=mom_propa.sq

# Also record shat and rapidity, since the initial momenta will also be reshuffled
#
        p1=self["get_momentum"][1].copy()
        p2=self["get_momentum"][2].copy()
        ptot=p1.add(p2)
        self["shat"]=ptot.sq
        self["rapidity"]=0.5*math.log((ptot.E+ptot.pz)/(ptot.E-ptot.pz))

    def reshuffle_momenta(self):
        """
            At this stage, 
            - the topo should be dressed with momenta.
            - the angles should be already extracted.
            - the masses should be corrected.
            This routine scan all branchings:
                 - first consider the t-branchings, go to the appropriate frame, 
                     modify the three-momenta to account for the corrected masses,
                 - then consider the s-branchings, go to the approriate frame,
                     rescale the three-momenta to account for the corrected masses.
        """
 
#     step number one: need to check if p^2 of each propa
#     is ok with the new set of masses ...
#     we first check this for all the s-branchings

#     Close to threshold, problems may occur: 
#         e.g. in A>B+C, one may have mA>mB+mC
#     Currently, if this happens, I throw away the point
#     but I am not sure what is the best prescription here ... 

        for  branch in self["branchings"]:
             
#            no need to consider t-branching here:
            if branch["type"]!="s":
                    continue
            d1=branch["index_d1"]
            d2=branch["index_d2"]
            propa=branch["index_propa"]

            MA=math.sqrt(self["get_mass2"][propa])
            MB=math.sqrt(self["get_mass2"][d1])
             #print self["get_mass2"][d2]
            MC=math.sqrt(self["get_mass2"][d2])
            if MA < MB + MC:
                 #print "WARNING: s-channel propagator with too low p^2 "
                 #print "in branch "+str(iter)
                 #print "MA = "+str(MA)
                 #print "MB = "+str(MB)
                 #print "MC = "+str(MC)
                 #print "throw away the point"

                 #print "increase the value of p^2"
#                 self["get_mass2"][propa]=(MC+MB+iota)**2 
#                 if iter==len(self["branchings"])-1:             # if last branching, needs to 
#                        self["shat"]=self["get_mass2"][propa]    # set shat to the new value of MA**2 
                return 0


     
#     then loop over all t-channels
#     and re-genenate the "d2" daughter in each of these branchings
        got_a_t_branching=0
        for nu, branch in enumerate(self["branchings"]):

#            no need to consider the last branching in this loop:
            if(nu==len(self["branchings"])-1): break

#            no need to scan the s-branching now
            if branch["type"]!="t":
                    continue

            got_a_t_branching=1
            # t-channel sequence: A+B > 1 + 2,     r= pa-p1
            ida=branch["index_d1"]
            idb=2
            id1=branch["index_d2"]
            res=branch["index_propa"]
            # go to the rest frame of    A+B
            # set momenta A, B, 1
            pa = self["get_momentum"][ida]
            pb = self["get_momentum"][idb]
            p1 = self["get_momentum"][id1]
#            set masses
            ma2=self["get_mass2"][ida]
            if (self["get_mass2"][id1]>=0):
                #print "m1^2, t-branch "+str(iter)
                #print self["get_mass2"][id1]
                m1=math.sqrt(self["get_mass2"][id1])
            else:
                return 0
            m2=branch["m2"]
            t=self["get_mass2"][res]

            # express momenta p1 and pa in A+B CMS system
            pboost=self["get_momentum"][2].add(self["get_momentum"][branch["index_d1"]])
            pboost.px=-pboost.px
            pboost.py=-pboost.py
            pboost.pz=-pboost.pz
#            p1_cms=p1.boost(pboost)
            pa_cms=pa.boost(pboost)

#             determine the magnitude of p1 in the cms frame
            Esum=pboost.sq

            if Esum>0 :
                Esum=math.sqrt(Esum)
            else:
                return 0
            md2=(m1+m2)*(m1-m2)
            ed=md2/Esum
            if (m1*m2==0) :
                pp=(Esum-abs(ed))*0.5
            else:
                pp=(md2/Esum)**2-2.0*(m1**2+m2**2)+Esum**2
                if pp>0 :
                    pp=0.5*math.sqrt(pp)
                else:
                    return 0

#                Now evaluate p1
            E_acms=pa_cms.E
            p_acms=math.sqrt(pa_cms.mod2)

            p1E=(Esum+ed)*0.5
            if p1E < m1: 
#                logger.warning('E1 is smaller than m1 in t-branching')
#                logger.warning('Try to reshuffle the momenta once more')
                return 0
            p1z=-(m1*m1+ma2-t-2.0*p1E*E_acms)/(2.0*p_acms)
            ptsq=pp*pp-p1z*p1z

            if (ptsq<0 ): 
#                if pT=0 to begin with, one can get p1z slightly larger 
#                                than pp due to numerical uncertainties.
#                     In that case, just change slightly the invariant t
                if (-ptsq/(pp*pp) <1e-6 ):
                    oldt=t
                    if (p1z>0):
                        p1z=pp
                    else:
                        p1z=-pp
                    pt=0.0
                    t=m1*m1+ma2-2.0*p1E*E_acms+2.0*p_acms*p1z
                    diff_t=abs((t-oldt)/t)*100
                    if (diff_t>2.0): 
                        logger.warning('t invariant was changed by '+str(diff_t)+' percents')
                else:
                    return 0
            else:
                pt=math.sqrt(pp*pp-p1z*p1z)

            p1x=pt*branch["cosphi"]
            p1y=pt*branch["sinphi"]

            p1=momentum(p1E,p1x,p1y,p1z)

            p1=p1.rot(pa_cms)
            pboost.px=-pboost.px
            pboost.py=-pboost.py
            pboost.pz=-pboost.pz
            p1=p1.boost(pboost)
            pr=pa.subtract(p1)
                #print " p1 is "
                #print p1.nice_string()
                #print " pr is "
                #print pr.nice_string()
            p2=(pa.add(pb)).subtract(p1)
                #print " p2 is "
                #print p2.nice_string()
#            now update momentum 
            self["get_momentum"][id1]=p1.copy()
            self["get_momentum"][res]=pr.copy()

        # after we have looped over all t-branchings,
        # p2 can be identified with the momentum of the second daughter of the last branching

        if got_a_t_branching==1:
            pid=self["branchings"][-1]["index_d2"]
            self["get_momentum"][pid]=p2.copy()
        #else: it means that there were no t-channel at all
        #            last branching should be associated with shat 
        #            note that the initial momenta will be reshuffled 
        #            at the end of this routine 


#    Now we can    loop over all the s-channel branchings.
#    Need to start at the end of the list of branching
        for branch in reversed(self["branchings"]):
            if branch["type"]!="s":
                    continue
            d1=branch["index_d1"]
            d2=branch["index_d2"]
            propa=branch["index_propa"]
            del self["get_momentum"][d1]
            del self["get_momentum"][d2]
            mA=math.sqrt(self["get_mass2"][propa])
            mB=math.sqrt(self["get_mass2"][d1])
            mC=math.sqrt(self["get_mass2"][d2])
            mom=self["get_momentum"][propa]
            costh=branch["costheta"]
            sinth=branch["sintheta"]
            cosphi=branch["cosphi"]
            sinphi=branch["sinphi"]
            decay2body=generate_2body_decay(mom, mA,mB,mC, \
                                    costh_val=costh, sinth_val=sinth, \
                                    cosphi_val=cosphi, sinphi_val=sinphi)
            self["get_momentum"][d1]=decay2body.momd1.copy()
            self["get_momentum"][d2]=decay2body.momd2.copy()

#    Need a special treatment for the 2 > 1 processes:
        if len(self["get_mass2"])==3:
            self["shat"]=self["get_mass2"][3]

#    Finally, compute the initial momenta
#    First generate initial momenta in the CMS frame
#    Then boost
        mB=self["get_mass2"][1]
        mC=self["get_mass2"][2]
        if mB>0:
            mB=math.sqrt(mB)
        else:
            mB=0.0
        if mC>0:
            mC=math.sqrt(mC)
        else:
            mC=0.0
        mA=math.sqrt(self["shat"])
        Etot=mA*math.cosh(self["rapidity"])
        pztot=mA*math.sinh(self["rapidity"])
        ptot=momentum(Etot,0.0,0.0,pztot)

        decay2body=generate_2body_decay(ptot,mA,mB,mC, \
                             costh_val=1.0, sinth_val=0.0, \
                             cosphi_val=0.0, sinphi_val=1.0)

        self["get_momentum"][1]=decay2body.momd1
        self["get_momentum"][2]=decay2body.momd2

#    Need a special treatment for the 2 > 1 processes:
        if len(self["get_momentum"])==3:
            self["get_momentum"][3]=momentum(Etot,0.0,0.0,pztot)
        
        return 1

    def extract_angles(self):
        """ the topo should be dressed with momenta at this stage.
                    Now: extract the angles characterizing each branching.
                    For t-channel, the equivalent of cos(theta) is the mass of p2
        """    

        for nu, branch in enumerate(self["branchings"]):
            if branch["type"]=="s":
            # we have the decay A > B + C
            # go to the rest frame of the decaying particle A, and extract cos theta and phi
                propa=branch["index_propa"]
#                d1=branch["index_d1"]
                d2=branch["index_d1"]
                pboost=self["get_momentum"][propa].copy()
                pboost.px=-pboost.px
                pboost.py=-pboost.py
                pboost.pz=-pboost.pz
#               pb_cms=self["get_momentum"][d1].boost(pboost)
                pc_cms=self["get_momentum"][d2].boost(pboost)
                mod_pc_cms=math.sqrt(pc_cms.mod2)
                branch["costheta"]=pc_cms.pz/mod_pc_cms
                branch["sintheta"]=math.sqrt(1.0-branch["costheta"]**2)
                branch["cosphi"]=pc_cms.px/mod_pc_cms/branch["sintheta"]
                branch["sinphi"]=pc_cms.py/mod_pc_cms/branch["sintheta"]

            if branch["type"]=="t":
            # we have a t-channel decay A + B > 1 + 2
            # go to the rest frame of    A+B, and extract phi
                    
            # set momenta A, B, 1
                pa = self["get_momentum"][branch["index_d1"]]
                pb = self["get_momentum"][2]
                p1 = self["get_momentum"][branch["index_d2"]]
                p2=(pa.add(pb)).subtract(p1)

                if (nu==len(self["branchings"])-1):
                    # last t-channel branching, p2 should be zero
                    check=p2.E**2+p2.mod2
                    if check>1e-3: 
                        logger.warning('p2 in the last t-branching is not zero')
                        # If last t-channel branching, there is no angles to extract
                    continue
                elif (nu==len(self["branchings"])-2):
                    # here m2 corresponds to the mass of the second daughter in the last splitting 
                    part=self["branchings"][-1]["index_d2"]
                    if self["get_mass2"][part]<0.0 :
                        logger.warning('negative mass for particle '+str(part))
                        logger.warning( self["get_mass2"][part])
                    branch["m2"]=math.sqrt(self["get_mass2"][part])
                elif p2.sq <0 and abs(p2.E)>1e-2: 
                    logger.warning('GET A NEGATIVE M2 MASS IN THE T-BRANCHING '+str(iter))
                    logger.warning( '(ROUTINE EXTRACT_ANGLES)')
                    logger.warning( p2.nice_string() )
                    logger.warning( p2.sq )
                else:
                    branch["m2"]=p2.m
                 
                # express momenta p1 and pa in A+B CMS system
                pboost=self["get_momentum"][2].add(self["get_momentum"][branch["index_d1"]])
                pboost.px=-pboost.px
                pboost.py=-pboost.py
                pboost.pz=-pboost.pz

                if pboost.m<1e-6: 
                    logger.warning('Warning: m=0 in T-BRANCHING '+str(iter))
                    logger.warning(' pboost.nice_string()')

                p1_cms=p1.boost(pboost)
                pa_cms=pa.boost(pboost)

#               E_acms=pa.E
#               mod_acms=math.sqrt(pa_cms.mod2)
#
#               now need to go to the frame where pa is aligned along with the z-axis
                p_rot=pa_cms.copy()
 
                p1_cmsrot=p1_cms.invrot(p_rot)
#               pa_cmsrot=pa_cms.invrot(p_rot)
#               now extract cosphi, sinphi
                pt=math.sqrt(p1_cmsrot.px**2+p1_cmsrot.py**2)
                if (pt>p1_cms.E*10e-10):
                    branch["cosphi"]=p1_cmsrot.px/pt                 
                    branch["sinphi"]=p1_cmsrot.py/pt
                else:                 
                    branch["cosphi"]=1                 
                    branch["sinphi"]=0.0


class AllMatrixElement(dict):
    """Object containing all the production topologies required for event to decay.
       This contains the routine to add a production topologies if needed.
    """
    
    def __init__(self, banner, options):
        
        dict.__init__(self)
        self.banner = banner
        self.options = options
        
    def add(self, topologies, keys):
        """Adding one element to the list of production_topo"""

        for key in keys:
            self[key] = topologies

    def get_br(self, proc):
        br = 1
        ids = collections.defaultdict(list) #check for identical decay
        for decay in proc.get('decay_chains'):
            init, final = decay.get_initial_final_ids()
            lhaid = tuple([len(final)] + [x for x in final])
            ids[init[0]].append(decay)
            if init[0] in self.banner.param_card['decay'].decay_table:
                br *= self.banner.param_card['decay'].decay_table[init[0]].get(lhaid).value
                br *= self.get_br(decay)
            else:
                init = -init[0]
                lhaid = [-x for x in final]
                lhaid.sort()
                lhaid = tuple([len(final)] + lhaid)
                br *= self.banner.param_card['decay'].decay_table[init].get(lhaid).value
                br *= self.get_br(decay)
        for decays in ids.values():
            len_decay = len(decays)
            if len(decays) == 1:
                continue
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

            
    def add_decay(self, matrix_element, me_path):
        """ adding a decay to the possibility
            me_path is the path of the fortran directory
            br is the associate branching ratio
            finals is the list of final states equivalent to this ME
            matrix_element is the MG5 matrix element
        """

        proc =  matrix_element.get('base_amplitude').get('process')
        tag = proc.get_initial_final_ids()
            
        finals = []
        for proc in matrix_element.get('processes'):
            tmp = []
            for dproc in proc.get('decay_chains'):
                id = dproc.get('legs')[0].get('id')
                tmp.append((id,dproc.get_final_ids_after_decay()))
            if tmp not in finals:
                finals.append(tmp)           
        # get BR
        br = len(finals) * self.get_br(proc) #update intermediate_id as well        
        
        
        
        
        
        me = matrix_element.get('processes')[0]
        # all the other are symmetric -> no need to keep those
            
        all_legs = me.get_legs_with_decays()
        ids = [l.get('id') for l in all_legs]
        decay_tags = [d.shell_string() for d in me['decay_chains']]
        out = {'path': me_path, 'matrix_element': me, 'br': br,
               'finals': finals, 'base_order': ids,
               'decay_struct':self.get_full_process_structure(matrix_element.get('processes')),
               'decay_tag': tuple(decay_tags)}
                
        self[tag]['decays'].append(out)
        self[tag]['total_br'] += br
        assert self[tag]['total_br'] <= 1.01, self[tag]['total_br']
        # decaying
        decaying = [m.get('legs')[0].get('id') for m in matrix_element.get('processes')[0].get('decay_chains')]
        self[tag]['decaying'].update(decaying)
        
             
    
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

        minvert = min([max([d for d in config if d][0].get_vertex_leg_numbers()) \
                                             for config in configs])

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
                                              get_s_and_t_channels(ninitial, new_pdg))
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
    
    def get_random_decay(self, production_tag):
        """select randomly a decay channel"""
        
        a = random.random() * self[production_tag]['total_br']
        sum = 0
        for decay in self[production_tag]['decays']:
            sum += decay['br']
            if a < sum:
                return decay

            


        
    
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        
        for me in matrix_elements:
            topo = self.get_topologies(me)
            # get the orignal order:
            initial = []
            final = [l.get('id') for l in me.get('processes')[0].get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
            topo['base_order'] = (initial , final)
            topo['matrix_element'] = me
            tags = []
            topo['tag2order'] = {}
            for proc in me.get('processes'):
                initial = []
                final = [l.get('id') for l in proc.get('legs')\
                      if l.get('state') or initial.append(l.get('id'))]
                tags.append(proc.get_initial_final_ids())
                topo['tag2order'][tags[-1]] = (initial , final)
            
            if tags[0] not in self:
                self.add(topo, tags)
            topo['path'] = pjoin(path, 'Subprocesses', 
                                  'P%s' % me.get('processes')[0].shell_string())
            topo['decays'] = []
            topo['total_br'] = 0
            topo['decaying'] = set()    

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
                print decay, to_decay
                if ',' in decay:
                    to_decay.update(set([l.split('>')[0].strip() 
                                                    for l in decay.split(',')]))

        # Maybe the branching fractions are already given in the banner:
        self.extract_br_from_banner(self.banner)
        misc.sprint(to_decay)
        to_decay = [p for p in to_decay if not p in self.br]

        misc.sprint(to_decay)
        if to_decay:
            logger.info('We need to recalculate the branching fractions')
            if hasattr(self.model.get('particles')[0], 'partial_widths'):
                logger.debug('using the compute_width module of madevent')
                # use FR to get all partial widths                                      
                # set the br to partial_width/total_width
                self.launch_width_evaluation(to_decay, mgcmd) 
            else:
                logger.info('compute_width module not available, use numerical estimates instead ')
                self.extract_br_from_width_evaluation(to_decay)
       
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
            logger.info('decay channels for '+res+' :')
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
        """ use madgraph to generate me's for res > all all  
        """
        if os.path.isdir(pjoin(self.path_me,"width_calculator")):
            shutil.rmtree(pjoin(self.path_me,"width_calculator"))
            
        assert not os.path.exists(pjoin(self.path_me, "width_calculator"))
        
        path_me = self.path_me 
        label2pid = self.label2pid
        # first build a set resonances with pid>0

        #particle_set= list(to_decay)
        pids = set([abs(label2pid[name]) for name in to_decay])
        particle_set = [label2pid[pid] for pid in pids]
    
        commandline="import model %s\n" % self.model.get('modelpath')
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
        
        cmd.run_cmd('launch -f')
                
        #me_cmd = me_interface.MadEventCmd(pjoin(path_me,'width_calculator'))
        #me_cmd.exec_cmd('set automatic_html_opening False --no_save')

        filename=pjoin(path_me,'width_calculator','Events','run_01','param_card.dat')
#        misc.sprint(pjoin(path_me,'width_calculator','Events','run_01','param_card.dat'))
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
        
        particle_set=[]
        for part in resonances:
            pid_part = abs(label2pid[part]) 
            if pid_part not in particle_set:
                particle_set.append(pid_part)  

        particle_set = list(set(particle_set))
        argument = {'particles': particle_set, 
                    'input': pjoin(self.path_me, 'param_card.dat'),
                    'output': pjoin(self.path_me, 'param_card.dat')}
        
        me_interface.MadEventCmd.compute_widths(model, argument)
        self.extract_br_from_card(pjoin(self.path_me, 'param_card.dat'))
        return      
                                         

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

        for id, data in param_card['decay'].decay_table.items():
            label = self.pid2label[id]
            current = [] # tmp name for  self.br[label]
            for parameter in data:
                if parameter.lhacode[0] == 2:
                    d = [self.pid2label[pid] for pid in  parameter.lhacode[1:]]
                    current.append({'daughters':d, 'br': parameter.value})
            self.br[label] = current
        
        #update the banner:
        self.banner['slha'] = param_card.write(None)
        self.banner.param_card = param_card
        
        self.extract_br_for_antiparticle()
        return self.br



#class width_estimate_old:
#    """All methods used to calculate branching fractions"""
#
#    def __init__(self,resonances,path_me,pid2label_dic,banner,base_model):
#
#        self.resonances=resonances
#        self.path_me=path_me
#        self.pid2label=pid2label_dic
#        self.label2pid = self.pid2label 
#        self.model=banner.get('model')
#        self.banner = banner
#        self.base_model=base_model
#
#    def update_branch(self,branches,to_add):
#        """ complete the definition of the branch by appending each element of to_add"""
#        newbranches={}
#
#        for item1 in branches.keys():
#            for item2 in to_add.keys():
#                tag=item1+item2
#                newbranches[tag]={}
#                newbranches[tag]['config']=branches[item1]['config']+to_add[item2]['config']
#                newbranches[tag]['br']=branches[item1]['br']*to_add[item2]['br']
#
#        return newbranches
#
#    def get_BR_for_each_decay(self, decay_processes, multiparticles):
#        """ get the list for possible decays & the associated branching fraction  """
#        
#        model = self.model
#        base_model = self.base_model
#        pid2label = self.pid2label
#
#        ponctuation=[',','>',')','(']
#        new_decay_processes={}       
#
#        for part in decay_processes.keys():
#            pos_symbol=-1
#            branch_list=decay_processes[part].split()
#            new_decay_processes[part]={}
#            new_decay_processes[part]['']={}
#            new_decay_processes[part]['']['config']=""
#            new_decay_processes[part]['']['br']=1.0
#
#            initial=""
#            final=[]
#            for index, item in enumerate(branch_list):
#                # First get the symbol at the next position
#                if index<len(branch_list)-1:
#                    next_symbol=branch_list[index+1]
#                else:
#                    next_symbol=''
#                # Then handle the symbol item case by case 
#                if next_symbol=='>':              # case1: we have a particle initiating a branching
#                    initial=item
#                    if item not in [ particle['name'] for particle in base_model['particles'] ] \
#                        and item not in [ particle['antiname'] for particle in base_model['particles'] ]:
#                        raise Exception, "No particle "+item+ " in the model "+model
#                    continue
#                elif item=='>': continue       # case 2: we have the > symbole
#                elif item not in ponctuation : # case 3: we have a particle originating from a branching
#                    final.append(item)
#                    if next_symbol=='' or next_symbol in ponctuation:
#                        #end of a splitting, verify that it exists
#                        if initial not in self.br.keys():
#                            logger.debug('Branching fractions of particle '+initial+' are unknown')
#	        	    return 0
#                        if len(final)>2:
#                            raise Exception, 'splittings different from A > B +C are currently not implemented '
#
#                        if final[0] in multiparticles.keys():
#                            set_B=[pid2label[pid] for pid in multiparticles[final[0]]]
# 		        else:
#                            if final[0] not in [ particle['name'] for particle in base_model['particles'] ] \
#                               and final[0] not in [ particle['antiname'] for particle in base_model['particles'] ]:
#                               raise Exception, "No particle "+item+ " in the model "
#                            set_B=[final[0]]
#                        if final[1] in multiparticles.keys():
#                            set_C=[pid2label[pid] for pid in multiparticles[final[1]]]
#		        else:
#                            if final[1] not in [ particle['name'] for particle in base_model['particles'] ] \
#                               and final[1] not in [ particle['antiname'] for particle in base_model['particles'] ]:
#                               raise Exception, "No particle "+item+ " in the model "+model
#                            set_C=[final[1]]
#
#                        splittings={}
#                        counter=0
#			for chan in range(len(self.br[initial])): # loop over all channels
#			    got_it=0
#                            for d1 in set_B: 
#				for d2 in set_C:
#                                  if (d1==self.br[initial][chan]['daughters'][0] and \
#                                     d2==self.br[initial][chan]['daughters'][1]) or \
#				     (d2==self.br[initial][chan]['daughters'][0] and \
#                                     d1==self.br[initial][chan]['daughters'][1]):
#                                      split=" "+initial+" > "+d1+" "+d2+" "
#                                      # For the tag we need to order d1 d2, so that equivalent tags can be correctly idetified
#                                      list_daughters=sorted([d1,d2])
#                                      tag_split="|"+initial+">"+list_daughters[0]+list_daughters[1]
#                                      counter+=1
#                                      splittings[tag_split]={}
#                                      splittings[tag_split]['config']=split
#                                      splittings[tag_split]['br']=self.br[initial][chan]['br']
#				      got_it=1
#                                      break # to avoid double counting in cases such as w+ > j j 
#                                if got_it: break                   
#
#                        if len(splittings)==0:
#			    logger.info('Branching '+initial+' > '+final[0]+' '+final[1])
#  			    logger.info('is currently unknown')
#                  
#                            return 0
#		        else:
#                            new_decay_processes[part]=self.update_branch(new_decay_processes[part],splittings)
#                        
#                        inital=""
#                        final=[]
#
#                else:                             # case 4: ponctuation symbol outside a splitting
#                                                  # just append it to all the current branches
#		    fake_splitting={}
#                    fake_splitting['']={}
#                    fake_splitting['']['br']=1.0
#                    fake_splitting['']['config']=item
#                    new_decay_processes[part]=self.update_branch(new_decay_processes[part],fake_splitting)
#
#        return new_decay_processes
#
#    def print_branching_fractions(self):
#        """ print a list of all known branching fractions"""
#
#        for res in self.br.keys():
#            logger.info('  ')
#            logger.info('decay channels for '+res+' :')
#            logger.info('       BR                 d1  d2' )
#            for decay in self.br[res]:
#                bran = decay['br']
#                d1 = decay['daughters'][0]
#                d2 = decay['daughters'][1]
#                logger.info('   %e            %s  %s ' % (bran, d1, d2) )
#            logger.info('  ')
#
#    def print_partial_widths(self):
#        """ print a list of all known partial widths"""
#
#        for res in self.br.keys():
#            logger.info('  ')
#            logger.info('decay channels for '+res+' :')
#            logger.info('       width                     d1  d2' )
#
#            for chan, decay in enumerate(self.br[res]):
#                width=self.br[res][chan]['width']
#                d1=self.br[res][chan]['daughters'][0]
#                d2=self.br[res][chan]['daughters'][1]
#                logger.info('   %e            %s  %s ' % (width, d1, d2) )
#            logger.info('  ')
#
#
#    def extract_br_from_width_evaluation(self):
#        """ use madgraph to generate me's for res > all all  
#        """
#        if os.path.isdir(pjoin(self.path_me,"width_calculator")):
#            shutil.rmtree(pjoin(self.path_me,"width_calculator"))
#            
#        assert not os.path.exists(pjoin(self.path_me, "width_calculator"))
#        
#        path_me = self.path_me 
#        label2pid = self.label2pid
#        # first build a set resonances with pid>0
#
#        particle_set=[]
#        for part in self.resonances:
#            if label2pid[part]>0: particle_set.append(part)
#        for part in self.resonances:
#            if label2pid[part]<0:
#                pid_part=-label2pid[part]
#                if self.pid2label[pid_part] not in particle_set:
#                    particle_set.append(self.pid2label[pid_part])
#        particle_set = list(set(particle_set))
#    
#        commandline="import model %s\n" % self.model
#        commandline+="generate %s > all all \n" % particle_set[0]
#        commandline+= "set automatic_html_opening False --no_save\n"
#        if len(particle_set)>1:
#            for index in range(1,len(particle_set)):
#                commandline+="add process %s > all all \n" % particle_set[index]
#
#        commandline += "output %s/width_calculator -f \n" % path_me
#
#
#        aloha.loop_mode = False
#        aloha.unitary_gauge = False
#        cmd = Cmd.MasterCmd()        
#        for line in commandline.split('\n'):
#            cmd.run_cmd(line)
#        files.cp(pjoin(path_me, 'Cards', 'param_card.dat'), 
#                 pjoin(path_me, 'width_calculator', 'Cards'))
#        
#        cmd.run_cmd('launch -f')
#                
#        #me_cmd = me_interface.MadEventCmd(pjoin(path_me,'width_calculator'))
#        #me_cmd.exec_cmd('set automatic_html_opening False --no_save')
#
#        filename=pjoin(path_me,'width_calculator','Events','run_01','param_card.dat')
##        misc.sprint(pjoin(path_me,'width_calculator','Events','run_01','param_card.dat'))
#        self.extract_br_from_card(filename)
#
#    def extract_br_for_antiparticle(self):
#        '''  
#            for each channel with a specific br value, 
#            set the branching fraction of the complex conjugated channel 
#            to the same br value 
#        '''
#        
#        label2pid = self.label2pid
#        pid2label = self.label2pid
#        
#        for res in self.br.keys():
#            particle=self.base_model.get_particle(label2pid[res])
#            if particle['self_antipart']: 
#                continue
#            anti_res=pid2label[-label2pid[res]]
#            self.br[anti_res] = []
#            for chan, decay in enumerate(self.br[res]):
#                self.br[anti_res].append({})
#                bran=decay['br']
#                d1=decay['daughters'][0]
#                d2=decay['daughters'][1]
#                d1bar=pid2label[-label2pid[d1]]
#                d2bar=pid2label[-label2pid[d2]]
#                self.br[anti_res][chan]['br']=bran
#                self.br[anti_res][chan]['daughters']=[]
#                self.br[anti_res][chan]['daughters'].append(d1bar)
#                self.br[anti_res][chan]['daughters'].append(d2bar)
#                if decay.has_key('width'):
#                    self.br[anti_res][chan]['width']=decay['width']                  
#
#    def launch_width_evaluation(self,resonances, model, mgcmd):
#        """ launch the calculation of the partial widths """
#
#        label2pid = self.label2pid
#        pid2label = self.label2pid
#        # first build a set resonances with pid>0
#        # since compute_width cannot be used for particle with pid<0
#        
#        particle_set=[]
#        for part in resonances:
#            pid_part = abs(label2pid[part]) 
#            if pid_part not in particle_set:
#                particle_set.append(pid_part)  
#        # erase old info
#        del self.br
#        self.br={}
#        particle_set = list(set(particle_set))
#        argument = {'particles': particle_set, 
#                    'input': pjoin(self.path_me, 'param_card.dat'),
#                    'output': pjoin(self.path_me, 'param_card.dat')}
#        
#        me_interface.MadEventCmd.compute_widths(model, argument)
#        self.extract_br_from_card(pjoin(self.path_me, 'param_card.dat'))
#        return      
#                                         
#
#    def extract_br_from_banner(self, banner):
#        """get the branching ratio from the banner object:
#           for each resonance with label 'res', and for each channel with index i,
#           returns a dictionary branching_fractions[res][i]
#           with keys
#            'daughters' : label of the daughters (only 2 body)
#            'br' : value of the branching fraction"""
#        
#        self.br = {}
#        
#        # read the param_card internally to the banner
#        if not hasattr(banner, 'param_card'):
#            banner.charge_card('param_card')
#        param_card = banner.param_card
#        return self.extract_br_from_card(param_card)
#
#    def extract_br_from_card(self, param_card):
#        """get the branching ratio from the banner object:
#           for each resonance with label 'res', and for each channel with index i,
#           returns a dictionary branching_fractions[res][i]
#           with keys
#            'daughters' : label of the daughters (only 2 body)
#            'br' : value of the branching fraction"""        
#        
#        if isinstance(param_card, str):
#            import models.check_param_card as check_param_card
#            param_card = check_param_card.ParamCard(param_card)
#        
#        if 'decay' not in param_card or not hasattr(param_card['decay'], 'decay_table'):
#            return self.br
#
#        for id, data in param_card['decay'].decay_table.items():
#            label = self.pid2label[id]
#            current = [] # tmp name for  self.br[label]
#            for parameter in data:
#                if parameter.lhacode[0] == 2:
#                    d = [self.pid2label[pid] for pid in  parameter.lhacode[1:]]
#                    current.append({'daughters':d, 'br': parameter.value})
#            self.br[label] = current
#        
#        self.extract_br_for_antiparticle()
#        return self.br


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

        return found

    def decay_one_event_old(self,curr_event,decay_struct,pid2color_dico,\
                        to_decay, pid2width,pid2mass,resonnances,BW_cut,ran=1):

# Consider the production event recorded in "curr_event", and decay it
# according to the structure recoreded in "decay_struct".
# If ran=1: random decay, phi and cos theta generated according to 
#                     a uniform distribution in the rest frame of the decaying particle
# Ir ran=0 : use the previsously-generated angles and masses to get the momenta


        decayed_event=Event()
        decayed_event.event2mg={}

        if ran==0: # reshuffling phase: the decayed event is about to be written, so we need 
                    # to record some information 
            decayed_event.ievent=curr_event.ievent
            decayed_event.wgt=curr_event.wgt
            decayed_event.scale=curr_event.scale
            decayed_event.aqed=curr_event.aqed
            decayed_event.aqcd=curr_event.aqcd
            decayed_event.diese=curr_event.diese
            decayed_event.rwgt=curr_event.rwgt

        part_number=0
        external=0
        maxcol=curr_event.max_col
        weight=1.0

#event2mg

        for index    in curr_event.event2mg.keys():
            if curr_event.event2mg[index]>0:
                part=curr_event.event2mg[index]
                if part in to_decay.keys():
                    mom_init=curr_event.particle[part]["momentum"].copy()
                    branch_id=to_decay[part]
                    # sanity check
                    if mom_init.m<1e-6:
                        logger.debug('Decaying particle with mass less than 1e-6 GeV in decay_one_event_old')
                    
                    decay_products, jac=decay_struct[branch_id].generate_momenta(mom_init,\
                                        ran, pid2width,pid2mass,resonnances,BW_cut)

                    if ran==1:
                        if decay_products==0: return 0, 0
                        weight=weight*jac

                    # now we need to write the decay products in the event
                    # follow the decay chain order, so that we can easily keep track of the mother index
                    for res in range(-1,-len(decay_struct[branch_id]["tree"].keys())-1,-1):
                        if (res==-1):
                            part_number+=1
                            mom=decay_products[res]["momentum"]
                            pid=decay_products[res]["pid"]
                            istup=2
                            mothup1=1
                            mothup2=2
                            colup1=curr_event.particle[part]["colup1"]
                            colup2=curr_event.particle[part]["colup2"]
                            decay_products[res]["colup1"]=colup1
                            decay_products[res]["colup2"]=colup2
                            mass=mom.m
                            helicity=0.
                            decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":colup1,"colup2":colup2,"momentum":mom,\
                                "mass":mass,"helicity":helicity}
                            decayed_event.event2mg[part_number]=part_number
                    #print part_number
                    #print pid
                    #print " "
                        mothup1=part_number
                        mothup2=part_number
#
#             Extract color information so that we can write the color flow
#
                        colormother=pid2color_dico[decay_products[res]["pid"]]
                        colord1=pid2color_dico[decay_products[decay_struct[branch_id]\
                                            ["tree"][res]["d1"]["index"]]["pid"]]
                        colord2=pid2color_dico[decay_products[decay_struct[branch_id]\
                                            ["tree"][res]["d2"]["index"]]["pid"]]
                
                        colup1=decay_products[res]["colup1"]
                        colup2=decay_products[res]["colup2"]

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
                        else:
                            raise Exception, 'color combination not treated by MadSpin (yet). (%s,%s,%s)' \
                                % (colord1,colord2,colormother)
                        part_number+=1
                        mom=decay_products[decay_struct[branch_id]\
                                    ["tree"][res]["d1"]["index"]]["momentum"]
                        pid=decay_products[decay_struct[branch_id]\
                                    ["tree"][res]["d1"]["index"]]["pid"]


                        indexd1=decay_struct[branch_id]["tree"][res]["d1"]["index"]
                        if ( indexd1>0):
                            istup=1
                            external+=1
                        else:
                            decay_products[indexd1]["colup1"]=d1colup1
                            decay_products[indexd1]["colup2"]=d1colup2
                            istup=2
                    
                        mass=mom.m
                        helicity=0.
                        decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":d1colup1,"colup2":d1colup2,"momentum":mom,\
                                "mass":mass,"helicity":helicity}
                        decayed_event.event2mg[part_number]=part_number

                        part_number+=1
                        mom=decay_products[decay_struct[branch_id]["tree"][res]["d2"]\
                                           ["index"]]["momentum"]
                        pid=decay_products[decay_struct[branch_id]["tree"][res]["d2"]\
                                           ["index"]]["pid"]

                        indexd2=decay_struct[branch_id]["tree"][res]["d2"]["index"]
                        if ( indexd2>0):
                            istup=1
                            external+=1
                        else:
                            istup=2
                            decay_products[indexd2]["colup1"]=d2colup1
                            decay_products[indexd2]["colup2"]=d2colup2

                        mothup1=part_number-2
                        mothup2=part_number-2
                        mass=mom.m
                        helicity=0.
                        decayed_event.particle[part_number]={"pid":pid,"istup":istup,\
                           "mothup1":mothup1,"mothup2":mothup2,"colup1":d2colup1,\
                           "colup2":d2colup2,\
                           "momentum":mom,"mass":mass,"helicity":helicity}

                        decayed_event.event2mg[part_number]=part_number

                else:
                    external+=1 
                    part_number+=1
                    decayed_event.particle[part_number]=curr_event.particle[part]
                    decayed_event.event2mg[part_number]=part_number
           
            else: # resonance in the production event
                if (ran==0): # write resonances in the prod. event ONLY if the 
                    # decayed event is ready to be written down    
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

        return decayed_event, weight

#    @staticmethod
#    def get_topologies(matrix_element):
#        """Extraction of the phase-space self.topologies from mg5 matrix elements 
#             This is used for the production matrix element only.
#
#             the routine is essentially equivalent to    write_configs_file_from_diagrams
#             except that I don't write the topology in a file, 
#             I record it in an object production_topo (the class is defined above in this file)
#        """
#        misc.sprint('DEPRECIATION WARNING')
#        # Extract number of external particles
#        ( nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
#
#        del nexternal
#        preconfigs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
#        mapconfigs = [c[0] for c in preconfigs]
#        configs=[[c[1]] for c in preconfigs]
#        model = matrix_element.get('processes')[0].get('model')
#
#
#        topologies ={}    # dictionnary {mapconfig number -> production_topology}
#                                    # this is the object to be returned at the end of this routine
#
#        s_and_t_channels = []
#
#        minvert = min([max([d for d in config if d][0].get_vertex_leg_numbers()) \
#                                             for config in configs])
#
#    # Number of subprocesses
##    nsubprocs = len(configs[0])
#
#        nconfigs = 0
#
#        new_pdg = model.get_first_non_pdg()
#
#        for iconfig, helas_diags in enumerate(configs):
#            if any([vert > minvert for vert in
#                            [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
#                    # Only 3-vertices allowed in configs.inc
#                    continue
#            nconfigs += 1
#
#            # Need s- and t-channels for all subprocesses, including
#            # those that don't contribute to this config
#            empty_verts = []
#            stchannels = []
#            for h in helas_diags:
#                    if h:
#                            # get_s_and_t_channels gives vertices starting from
#                            # final state external particles and working inwards
#                            stchannels.append(h.get('amplitudes')[0].\
#                                              get_s_and_t_channels(ninitial, new_pdg))
#                    else:
#                            stchannels.append((empty_verts, None))
#
#            # For t-channels, just need the first non-empty one
#            tchannels = [t for s,t in stchannels if t != None][0]
#
#            # For s_and_t_channels (to be used later) use only first config
#            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
#                                                             tchannels])
#
#
#
#            # Make sure empty_verts is same length as real vertices
#            if any([s for s,t in stchannels]):
#                    empty_verts[:] = [None]*max([len(s) for s,t in stchannels])
#
#                    # Reorganize s-channel vertices to get a list of all
#                    # subprocesses for each vertex
#                    schannels = zip(*[s for s,t in stchannels])
#            else:
#                    schannels = []
#
#            allchannels = schannels
#            if len(tchannels) > 1:
#                    # Write out tchannels only if there are any non-trivial ones
#                    allchannels = schannels + tchannels
#
## Write out propagators for s-channel and t-channel vertices
#
##         use the AMP2 index to label the self.topologies
#            tag_topo=mapconfigs[iconfig]
#            topologies[tag_topo]=production_topo_old()
#
#            for verts in allchannels:
#                    if verts in schannels:
#                            vert = [v for v in verts if v][0]
#                    else:
#                            vert = verts
#                    daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
#                    last_leg = vert.get('legs')[-1]
#
#
#                    if verts in schannels:
#                            type_propa="s"
#                    elif verts in tchannels[:-1]:
#                            type_propa="t"
#
#
#                    if (type_propa):
#                        topologies[tag_topo].add_one_branching(last_leg.get('number'),\
#                         daughters[0],daughters[1],type_propa)
#
#        return topologies
#
#    @misc.mute_logger()
#    def generate_fortran_me(self,processes,base_model,mode, mgcmd,path_me):
#        """Given a process and a model, use the standanlone module of mg5
#         to generate a fortran executable for the evaluation of the 
#         corresponding matrix element
#                mode=0 : production part 
#                mode=1 : process fully decayed
#        """
#        commandline="import model "+base_model
#        mgcmd.exec_cmd(commandline)
#
#        mgcmd.exec_cmd("set group_subprocesses False")
#
#        commandline="generate "+processes[0]
#        mgcmd.exec_cmd(commandline)
#
#        # output the result in Fortran format:
#        if mode==0: # production process
#            misc.sprint("output standalone_ms %s -f" % pjoin(path_me,'production_me'))
#            mgcmd.exec_cmd("output standalone_ms %s -f" % pjoin(path_me,'production_me') )
#        
#        elif mode==1: # full process
#            mgcmd.exec_cmd("output standalone_ms %s -f" % pjoin(path_me,'full_me'))
#          
#
#        
#              
#        # now extract the information about the self.topologies
#        if mode==0:
#            me_list=mgcmd._curr_matrix_elements.get_matrix_elements()
#            if(len(me_list)!=1): 
#                logger.warning('WARNING: unexpected number of matrix elements')
#            topo=self.get_topologies(me_list[0])
#            return topo
        

    def get_resonances(self,decay_processes):
        """ return a list of    labels of each resonance involved in the decay chain """

        resonances=[]
        for line in decay_processes:
            line=line.replace(">", " > ")
            line=line.replace("(", " ( ")
            line=line.replace(")", " ) ")
            list_proc=line.split()
            for i , item in enumerate(list_proc): 
                if item ==">":
                    resonances.append(list_proc[i-1])
        return resonances
         

#    def get_full_process_structure(self,decay_processes,line_prod_proc, base_model, banner, proc_option, check=0):
#        """ return a string with the definition of the process fully decayed
#                and also a list of dc_branch objects with all infomation about the topology 
#                of each decay branch
#        """
#        decay_struct={}    
#        self.full_proc_line=line_prod_proc+" "+proc_option+" , "
#        for proc_index in decay_processes.keys():
#            if ',' in decay_processes[proc_index]:
#                current_branch=decay_processes[proc_index]
##                list_branch=current_branch.split(",")
##                current_nb_decays=len(list_branch)
##                for nu in range(current_nb_decays):
##                    if nu >0 and nu < current_nb_decays-1: list_branch[nu]=" ( "+list_branch[nu]
##                for nu in range(current_nb_decays-2):
##                    list_branch[current_nb_decays-1]=list_branch[current_nb_decays-1]+" ) "
##                current_branch=""
##                for nu in range(current_nb_decays-1):
##                    current_branch+=list_branch[nu]+ "    , "
##                current_branch+=list_branch[current_nb_decays-1]
#                self.full_proc_line=self.full_proc_line+" ( "+current_branch+ " )    , "
#            else:
#                self.full_proc_line=self.full_proc_line+"    "+\
#                    decay_processes[proc_index]+ "     , "
#            decay_proc=decay_processes[proc_index]
#            decay_proc=decay_proc.replace("\n","")
#            decay_proc=decay_proc.replace("("," ")
#            decay_proc=decay_proc.replace(")"," ")
#
#            decay_struct[proc_index]=dc_branch(decay_proc,base_model,banner,check)
##            decay_struct[proc_index].print_branch()
##        decay_struct[proc_index].print_branch()
#        self.full_proc_line=self.full_proc_line[:-3]
#        #print self.full_proc_line
#
#        return self.full_proc_line, decay_struct
#
#    def compile_fortran_me_production(self, path_me):
#        """ Compile the fortran executables associated with the evalutation of the 
#                matrix elements (production process)
#                Returns the path to the fortran executable
#        """
#
#        list_prod=os.listdir(pjoin(path_me,"production_me/SubProcesses"))
#        counter=0
#        logger.debug("""Finalizing production me's """)
#
#         
#        for direc in list_prod:
#            if direc[0]=="P":
#                counter+=1
#                prod_name=direc[string.find(direc,"_")+1:]
#                
#                old_path=pjoin(path_me,'production_me','SubProcesses',direc)
#                new_path=pjoin(path_me,'production_me','SubProcesses',prod_name)
#                if os.path.isdir(new_path): shutil.rmtree(new_path)
#                os.rename(old_path, new_path)
#
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_prod.f')
#                shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f"))  
#                
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
#                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )
#
#                file=pjoin(path_me, 'param_card.dat')
#                shutil.copyfile(file,pjoin(path_me,"production_me","Cards","param_card.dat"))                
#
#		# files to produce the parameters:
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
#                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
#                    
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
#                shutil.copyfile(file_madspin, pjoin(path_me,"production_me","Source","MODEL","lha_read.f" )) 
#                shutil.copyfile(pjoin(path_me,'production_me','Source','MODEL','input.inc'),pjoin(new_path,'input.inc')) 
#
#
#                # COMPILATION
#                # in case there are new DHELAS routines, we need to recompile 
#                misc.compile(arg=['clean'], cwd=pjoin(path_me,"production_me","Source", "DHELAS"), mode='fortran')
#                misc.compile( cwd=pjoin(path_me,"production_me","Source","DHELAS"), mode='fortran')
#
#                misc.compile(arg=['clean'], cwd=pjoin(path_me,"production_me","Source", "MODEL"), mode='fortran')
#                misc.compile( cwd=pjoin(path_me,"production_me","Source","MODEL"), mode='fortran')                   
#
#                #os.chdir(new_path)
#                misc.compile(arg=['clean'], cwd=new_path, mode='fortran')
#                misc.compile(arg=['init'],cwd=new_path,mode='fortran')
#                misc.call('./init', cwd=new_path)
#
#                shutil.copyfile(pjoin(new_path,'parameters.inc'), 
#                               pjoin(new_path,os.path.pardir, 'parameters.inc'))
#                os.chdir(path_me)
#                    
#                misc.compile(cwd=new_path, mode='fortran')
#
#                if(os.path.getsize(pjoin(path_me,'production_me','SubProcesses', 'parameters.inc'))<10):
#                    raise Exception, "Parameters of the model were not written correctly ! " 
#                return prod_name
#
#
#    def compile_fortran_me_full(self,path_me):
#        """ Compile the fortran executables associated with the evalutation of the 
#                matrix elements (full process)
#                Returns the path to the fortran executable
#        """
#
#        list_full=os.listdir(pjoin(path_me,"full_me","SubProcesses"))
#
#        logger.debug("""Finalizing decay chain me's """)
#        for direc in list_full:
#            if direc[0]=="P":
#                
#                decay_name=direc[string.find(direc,"_")+1:]
#                
#                old_path=pjoin(path_me,'full_me','SubProcesses',direc)
#                new_path=pjoin(path_me, 'full_me','SubProcesses',decay_name)
#
#
#                if os.path.isdir(new_path): shutil.rmtree(new_path)
#                os.rename(old_path, new_path)               
#                
#                
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_full.f')
#                shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f")  )
#
#
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
#                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )
#                                
#                shutil.copyfile(pjoin(path_me,'full_me','Source','MODEL','input.inc'),pjoin(new_path,'input.inc'))
#
#                # write all the parameters:
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
#                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
#                         
#                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
#                shutil.copyfile(file_madspin, pjoin(path_me,"full_me","Source","MODEL","lha_read.f" ))  
#
#                file=pjoin(path_me, 'param_card.dat')
#                shutil.copyfile(file,pjoin(path_me,"full_me","Cards","param_card.dat")) 
#
#                # BEGIN COMPILATION
#                # in case there are new DHELAS routines, we need to recompile                
#                misc.compile(arg=['clean'], cwd=pjoin(path_me,"full_me","Source","DHELAS"), mode='fortran')
#                misc.compile( cwd=pjoin(path_me,"full_me","Source","DHELAS"), mode='fortran')
#
#                misc.compile(arg=['clean'], cwd=pjoin(path_me,"full_me","Source","MODEL"), mode='fortran')
#                misc.compile( cwd=pjoin(path_me,"full_me","Source","MODEL"), mode='fortran')   
#
#                os.chdir(new_path)
#                misc.compile(arg=['clean'], cwd=new_path, mode='fortran')
#                misc.compile(arg=['init'],cwd=new_path,mode='fortran')
#                misc.call('./init')
#                shutil.copyfile('parameters.inc', '../parameters.inc')
#                os.chdir(path_me)
#                
#                # now we can compile check
#                misc.compile(arg=['check'], cwd=new_path, mode='fortran')
#                # END COMPILATION
#
#            
#                if(os.path.getsize(pjoin(path_me,'full_me','SubProcesses', 'parameters.inc'))<10):
#                    raise Exception, "Parameters of the model were not written correctly ! " 
#
#                #decay_pattern=direc[string.find(direc,"_")+1:]
#                #decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]
#                #decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]
#
#        os.chdir(path_me)
#        return decay_name

    def restore_light_parton_masses(self,topo,event):
        """ masses of light partons were set to zero for 
                the evaluation of the matrix elements
                now we need to restore the initial masses before
                writting the decayed event in the lhe file
        """

        light_partons=[21,1,2,3]
        for part in topo["get_id"].keys():
            if abs(topo["get_id"][part]) in light_partons:
                topo["get_mass2"][part]=event.particle[part]["mass"]**2

#    need to check if last branch is a t-branching. If it is, 
#    we need to update the value of branch["m2"]
#    since this will be used in the reshuffling procedure
        if len(topo["branchings"])>0:  # Exclude 2>1 self.topologies
            if topo["branchings"][-1]["type"]=="t":
                if topo["branchings"][-2]["type"]!="t":
                    logger.warning('last branching is t-channel')
                    logger.warning('but last-but-one branching is not t-channel')
                else:
                    part=topo["branchings"][-1]["index_d2"]
                    if part >0: # reset the mass only if "part" is an external particle
                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])

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
            if item =="," and list_branch[index+1]!="(": 
                if list_branch[index-2]==list_branch[index+1]:
                    # swap the two particles before the comma:
                    temp=list_branch[index-2]
                    list_branch[index-2]=list_branch[index-1]
                    list_branch[index-1]=temp
            if item =="," and list_branch[index+1]=="(":
                if list_branch[index-2]==list_branch[index+2]:
                    # swap the two particles before the comma:
                    temp=list_branch[index-2]
                    list_branch[index-2]=list_branch[index-1]
                    list_branch[index-1]=temp


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

#    def find_resonances_old(self,proc, branches):
#        """ restore the resonances in a production process 
#            the expected decay chains (variable branches)
#            were extracted from the banner
#        """
#
#        pos1=proc.find(">")
#        list_finalstate=proc[pos1+1:].split()
#
#        for res in branches.keys():
#            resFS=[ item for item in branches[res]["finalstate"]]
#            for index in range(len(list_finalstate)-1,-1,-1):
#                if list_finalstate[index] in resFS: 
#                    pos2 = resFS.index(list_finalstate[index])
#                    del resFS[pos2]
#                    del list_finalstate[index]
#            if resFS!=[]:
#                logger.warning('CANNOT RECOGNIZE THE EXPECTED DECAY \
#                CHAIN STRUCTURE IN PRODUCTION EVENT')
#                return proc
#            else:
#                list_finalstate.append(res)
#
#        initstate=proc[:pos1]
#        finalstate=""
#        for part in list_finalstate:
#            finalstate+=" "+part
#        for res in branches.keys():
#            finalstate+=" , "+branches[res]["branch"]
#        newproc=initstate+" > "+finalstate+" "
#        return newproc



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

#    def get_banner_old(self):
#        pass


#    def set_cumul_proba_for_tag_decay_old(self, multi_decay_processes):
#        """
#        """
#        sum_br=0.0
#        for tag_decay in multi_decay_processes:
#            sum_br+=multi_decay_processes[tag_decay]['br']
#
#        cumul=0.0
#        for tag_decay in multi_decay_processes:
#            cumul += multi_decay_processes[tag_decay]['br']/sum_br
#            multi_decay_processes[tag_decay]['cumul_br']=cumul
#        return sum_br

#    @staticmethod
#    def get_identical_old(decay_processes):
#        """
#          return a dictionary  label : (indices)
#          with   'label' the label of a particle to be decayed in the production process
#                 '[indices]' the list of indices identifying the decay branches associated with 'label' 
#        """
#
#        ident_part={}
#        for item in decay_processes.keys():
#            line=decay_processes[item]
#            curr_list=line.split()
#            init_part=curr_list[0]
#            if init_part not in ident_part.keys():
#                ident_part[init_part]=[ item ]
#            else:
#                ident_part[init_part].append(item)
#
#        return ident_part

#    def generate_tag_decay_old(self,multi_decay_processes,decay_tags ):
#        """ generate randomly the tag for the decay config. using a probability law 
#            based on branching fractions
#        """
#
#        r=random.random()
#
#        for index, tag_decay in enumerate(decay_tags):
#            if r < multi_decay_processes[tag_decay]['cumul_br']:
#               return tag_decay
#        if r==1.0:
#            return decay_tags[-1]
#
#
#    @staticmethod
#    def update_tag_decays_old(decay_tags, branch_tags):
#        """ update the set of tags to identify the final state content of a decay channel """
#
#        new_decay_tags=[]
#        for item1 in branch_tags:
#            if len(decay_tags)==0:
#                new_decay_tags.append( (item1,) )
#            else:
#                for item2 in decay_tags:
#                    new_decay_tags.append(item2+(item1,))
#        return new_decay_tags
                  
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
     

#    def check_decay_tag_old(self, tag,tag_list,check_weights):
#        """ check is check_weights[tag] is proportional to 
#            check_weights[x] for one x in tag_list   """
#
#        for x in tag_list:
##           build a list of [el1/el2]
#            ratio=[ check_weights[x][index]/item for index, item in enumerate(check_weights[tag])]
#            mean, sd = self.get_mean_sd(ratio)
# 
#            if sd<mean*1e-6: return x
#        return 0
#
#    def item_belong_to_list_old(self,item, thelist):
#        """check if an item belongs to a list of items, but after converting all items to string 
#           NOT USED
#        """
#
#        for it in thelist:
#            if str(it)==str(item): return True
#        return False

#    @staticmethod
#    def symmetrize_tags_old(decay_tags, identical_part):
#        """ add new tags that are obtained by symmetrizing identical particles 
#            Also need a mapping to retrieve the information in 'branching per channel'
#        """ 
#
#        new_decay_tags=[]
#        mapping_tag2branch={}
#        for item in decay_tags: 
#            new_decay_tags.append(item)
#            # the mapping in that case is just the identity
#            mapping_tag2branch[item]=range(len(item))
#            
#
#        for list_ident in identical_part.values():
#            # 1. get the complete list of permutation of indices in list_ident
#            # 2. create the corresponding tags and verify if they should be added
#            # 3. update the mapping 'mapping_tag2branch' so that we can retrieve information 
#            #    from  'branching_per_channel' 
#            nb_sym=len(list_ident)
#            all_per=itertools.permutations(list_ident)
#            for i in range(math.factorial(nb_sym)):
#                this_per=all_per.next()
#                #print this_per
#                for tag in decay_tags:
#                    # bluid the new potential tag:
#                    new_tag=list(tag)
#                    for ind, part in enumerate(list_ident):
#                        new_tag[this_per[ind]]=tag[part]
#                    new_tag=tuple(new_tag)
#                    if new_tag not in  new_decay_tags:
#                        new_decay_tags.append(new_tag)
#                        mapping_tag2branch[new_tag]=range(len(new_tag))
#                        # also need to update extended branching
#                        for ind, part in enumerate(list_ident):
#                            mapping_tag2branch[new_tag][this_per[ind]]=ind
#                        #print new_tag
#                        #print mapping_tag2branch[new_tag]
# 
#        return new_decay_tags, mapping_tag2branch


#    def check_param_card_old(self, param_card):
#        raise Exception
#        list_line=param_card.split('\n')
#        output=""
#        loop_block=0
#        for line in list_line:
#            if line=="": continue
#            current_line=line.split()
#
#            if loop_block==1 and line[0]!='#': continue
#            if loop_block==1 and line[0]=='#': 
#                loop_block=0
#                continue
#            
#            if len(current_line)<2:
#                output+=line+"\n" 
#            elif current_line[0]!='Block' or current_line[1]!='loop':
#                output+=line+"\n"
#            else:
#                loop_block=1
#
#        return output

    def get_dico_branchindex2label(self, decay_processes):
        """ return a dictionary {index branch : label of decaying particle}"""

        dico={}
        for part in decay_processes:
            line_proc=decay_processes[part].split()
            dico[part]=line_proc[0]

        return dico

#    def process_decay_syntax_old(self,decay_processes):
#        """ add spaces to avoid any confusion in the decay chain syntax """
#
#        for part in decay_processes:
#            decay_processes[part]=decay_processes[part].replace(',',' , ')
#            decay_processes[part]=decay_processes[part].replace('>',' > ')
#            decay_processes[part]=decay_processes[part].replace(')',' ) ')
#            decay_processes[part]=decay_processes[part].replace('(',' ( ')

class decay_all_events:
    
    def __init__(self, ms_interface, banner, inputfile, options):
        """Store all the component and organize special variable"""
    
        # input
        self.options = options
        #max_weight_arg = options['max_weight']  
        #BW_effects = options['BW_effect']
        self.path_me = os.path.realpath(options['curr_dir']) 
        self.mgcmd = ms_interface.mg5cmd
        self.mscmd = ms_interface
        misc.sprint(self.mscmd.list_branches)
        self.model = ms_interface.model
        self.banner = banner
        self.evtfile = inputfile
        self.curr_event = Event(self.evtfile) 
               
        self.curr_dir = os.getcwd()
    
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        # need to unbuffer all I/O in fortran, otherwise
        # the values of matrix elements are not passed to the Python script
        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'  
    
        # Remove old stuff from previous runs
        # so that the current run is not confused

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
        # dictionary pid > color_rep
        self.pid2color = pid2color(self.model)
        
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
        param=open(pjoin(self.path_me,'param_card.dat'),"w")
        param.write(param_card)
        param.close()     
        
        self.list_branches = ms_interface.list_branches
        self.all_ME = AllMatrixElement(banner, self.options)
        self.all_decay = {}
        
        # generate BR and all the square matrix element baes on the banner.
        self.generate_all_matrix_element()
        self.compile()
        
    def run(self):
        """Running the full code""" 
    
        max_weight_arg = self.options['max_weight']  
        BW_effects = self.options['BW_effect']
        decay_tools=decay_misc()
        resonances = self.width_estimator.resonances
        logger.debug('List of resonances: %s' % resonances)
        self.extract_resonances_mass_width(resonances) 
        
        #Next step: we need to determine which matrix elements are really necessary
        #==========================================================================
        # Need to change the way this is done !
        decay_mapping = self.get_identical_decay()
        
        # Estimation of the maximum weight
        #=================================
        if max_weight_arg>0:
            for key in self.all_ME:
                for mode in self.all_ME['decays']:
                    mode['max_weight'] = max_weight_arg
        else:
            #self.get_max_weight_from_1toN()
            self.get_max_weight_from_event(decay_mapping)  
        
        # add probability of not writting events (for multi production with 
        # different decay
        self.add_loose_decay()
        
        # launch the decay and reweighting
        efficiency = self.decaying_events()
        misc.sprint(efficiency)
        if  efficiency != 1:
            # need to change the banner information [nb_event/cross section]
            print self.outputfile
            print self.outputfile.name
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
        shutil.rmtree(pjoin(self.path_me,'production_me'))
        shutil.rmtree(pjoin(self.path_me,'full_me'))

        # set the environment variable GFORTRAN_UNBUFFERED_ALL 
        # to its original value
        #os.environ['GFORTRAN_UNBUFFERED_ALL']='n'
 
        
    
    def decaying_events(self):
        """perform the decay of each events"""

        decay_tools = decay_misc()

        logger.info(' ' )
        logger.info('Decaying the events... ')
        
        self.outputfile = open(pjoin(self.path_me,'decayed_events.lhe'), 'w')
        self.write_banner_information()
        
        
        event_nb = 0
        nb_skip = 0 
        trial_nb_all_events=0
        starttime = time.time()
        while 1: # loop on event file
            production_tag, event_map = self.load_event()
            if production_tag == 0 == event_map: #end of file
                break

            # Here we need to select a decay configuration on a random basis:
            decay = self.all_ME.get_random_decay(production_tag)
            if not decay['decay_tag']:
                #Not writting events due to global reweighting
                nb_skip +=1
                continue 
            else:
                event_nb+=1
                if (event_nb % max(int(10**int(math.log10(float(event_nb)))),10)==0): 
                    running_time = misc.format_timer(time.time()-starttime)
                    logger.info('Event nb %s %s' % (event_nb, running_time))

            mg5_me_prod, prod_values = self.evaluate_me_production(production_tag, event_map)   
            tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)
            topology = self.all_ME[production_tag][tag_topo]            

            topology.dress_topo_from_event(self.curr_event)
            topology.extract_angles()
            if self.options['BW_effect']:
                decay_tools.set_light_parton_massless(self.all_ME[production_tag][tag_topo])            
            
            trial_nb=0
            while 1: # loop untill find a valid decay
                trial_nb += 1
                if self.options['BW_effect']:
                    BW_weight_prod, topology = self.reshuffle_event(production_tag, tag_topo, 
                                                        event_map, prod_values)
                else:
                    BW_weight_prod = 1
                #BW_weight_prod = 1
                topology.topo2event(self.curr_event)
                name =  ','.join([m.nice_string() for m  in decay['matrix_element'].get('decay_chains')])
                BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0
                
                decayed_event, BW_weight_decay = self.decay_one_event(\
                                    self.curr_event, 
                                    decay['decay_struct'],
                                    event_map,
                                    BW_cut)
                                    
                if decayed_event==0:
                    logger.warning('failed to decay event properly')
                    continue
                mg5_me_prod, prod_values = self.evaluate_me_production(production_tag, event_map)
                #     then decayed weight:
                p_full, p_full_str=decayed_event.give_momenta()
                #print p_full_str
                mg5_me_full = self.calculate_matrix_element('full',
                                                  decay['path'], p_full_str)
                #print dec, mg5_me_full, mg5_me_prod, BW_weight_prod,BW_weight_decay,
                weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod

                if weight > decay['max_weight']: 
                    logger.info('warning: got a larger weight than max_weight estimate')
                    logger.info('the ratio with the max_weight estimate is %s' % (weight/decay['max_weight']))
                    logger.info('decay channel ')
                    logger.info(decay['decay_tag'])
                    
                if (weight/decay['max_weight'] > random.random()):
                    # Here we need to restore the masses of the light partons 
                    # initially found in the lhe production event
                    if self.options['BW_effect']:
                        decay_tools.restore_light_parton_masses(topology, self.curr_event)
                        succeed = topology.reshuffle_momenta()
                        if not succeed:
                            logger.info('Warning: unable to restore masses of light partons')
                        else:
                            topology.topo2event(self.curr_event)
                            self.curr_event.reset_resonances() # re-evaluate the momentum of each resonance in prod. event
                    BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0
                    decayed_event, BW_weight_decay = self.decay_one_event(\
                                    self.curr_event, 
                                    decay['decay_struct'],
                                    event_map,
                                    BW_cut,
                                    ran=0)
                    
                    decayed_event.wgt = decayed_event.wgt * self.branching_ratio
                     
                    self.outputfile.write(decayed_event.string_event())
                #print "number of trials: "+str(trial_nb)
                    trial_nb_all_events+=trial_nb
                    break # pass to next event.
 
        self.outputfile.write('</LesHouchesEvents>\n')
        self.evtfile.close()
        self.outputfile.close()

        logger.info('Total number of events written: %s/%s ' % (event_nb, event_nb+nb_skip))
        logger.info('Average number of trial points per production event: '\
            +str(float(trial_nb_all_events)/float(event_nb)))
        logger.info('Number of subprocesses '+str(len(self.calculator)))
        
        return  event_nb/(event_nb+nb_skip)       
        
    def get_identical_decay(self):
        """identify the various decay which are identical to each other"""
        
        logger.info('detect independant decays')
        start = time.time()
        if len(self.all_decay) == 1:
            decay_mapping = {}
            for prod in self.all_ME.values():
                for decay in prod['decays']:
                    tag = decay['decay_tag']
                    decay_mapping[tag] = set([(tag, 1)]) 
            return decay_mapping
        
        BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0        
        #class the decay by class (nbody/pid)
        nbody_to_decay = collections.defaultdict(list)
        for decay in self.all_decay.values():
            id = decay['dc_branch']['tree'][-1]['label']
            id_final = decay['processes'][0].get_final_ids_after_decay()
            cut = self.options['zeromass_for_max_weight']
            mass_final = tuple([m if m> cut else 0 for m in map(self.pid2mass, id_final)])
            
            nbody_to_decay[(decay['nbody'], abs(id), mass_final)].append(decay)
        
        relation = {}     
        for ((nbody, pid, finals),decays) in nbody_to_decay.items():  
            if len(decays) == 1:
                continue  
            mom_init = momentum(self.pid2mass(pid), 0, 0, 0)
            
            # create an object for the validation
            valid = dict([ ((i, j), True) for j in range(len(decays)) 
                                          for i in range(len(decays)) 
                                          if i != j])
                
            for nb in range(25):
                tree, jac = decays[0]['dc_branch'].generate_momenta(mom_init,\
                                        True, self.pid2width, self.pid2mass, BW_cut)

                p_str = '%s\n%s\n'% (tree[-1]['momentum'],
                    '\n'.join(str(tree[i]['momentum']) for i in range(1, len(tree))
                                                                  if i in tree))
                
                
                values = {}                
                for i in range(len(decays)):
                    if any([valid[(i,j)] for j in range(len(decays)) if i !=j]):
                        values[i] = self.calculate_matrix_element('decay', 
                                                       decays[i]['path'], p_str)
                    else:
                        values[i] = 0
                
                for i in range(len(decays)):
                    for j in range(i+1, len(decays)):
                        if values[i] == 0 or values[j] == 0 or valid[(i,j)] == 0:
                            continue # already not valid
                        elif valid[(i,j)] is True:
                            valid[(i,j)] = values[j]/values[i]
                            valid[(j,i)] = valid[(i,j)]
                        elif (valid[(i,j)] - values[j]/values[i]) < 1e-5 * (valid[(i,j)] + values[j]/values[i]):
                            pass
                        else:
                            valid[(i, j)] = 0
                            valid[(j, i)] = 0
    
            for i in range(len(decays)):
                tag_i = decays[i]['tag'][2:]
                for j in range(i+1, len(decays)): 
                    tag_j = decays[j]['tag'][2:]     
                    if valid[(i,j)] and tag_j not in relation:
                        relation[tag_j] = (tag_i, valid[(i,j)])

        for decay in self.all_decay.values():
            tags = [m.shell_string()[2:] for m in decay['processes']]
            init_tag = tags[0]
            if init_tag not in relation:
                out = (init_tag, 1)
            else:
                out = relation[init_tag]
            for tag in tags[1:]:
                relation[tag] = out
        
        print relation

        decay_mapping = {} # final output
        tag2real = {}    # return basic representation
        for prod in self.all_ME.values():
            for decay in prod['decays']:
                tag = decay['decay_tag']
                basic_tag = []
                br = 1
                for t in tag:
                    if  t in relation:
                        basic_tag.append(relation[t][0])
                        br *= relation[t][1]
                    else:
                        basic_tag.append(t)
                basic_tag = tuple(basic_tag)
                if len(set(tag)) != len(tag):
                    for t in set(tag):
                        br /= math.factorial(tag.count(t))
                 
                if basic_tag not in tag2real:
                    tag2real[basic_tag] = (tag, br)
                    decay_mapping[tag] = set([(tag, 1)])
                else:
                    real_tag, br2 = tag2real[basic_tag]
                    
                    decay_mapping[real_tag].add((tag,br/br2))

        print decay_mapping
        

        logger.info('Done in %ss' % (time.time()-start))
        return decay_mapping
    

    @misc.mute_logger()
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
        self.get_branching_ratio()

        
        # 2. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [line[12:].strip() for line in self.banner.proc_card
                     if line.startswith('add process')]        
        
        mgcmd = self.mgcmd
        commandline="import model %s " % self.model.get('modelpath')
        mgcmd.exec_cmd(commandline)
        mgcmd.exec_cmd("set group_subprocesses False")

        logger.info('generating the production square matrix element')
        start = time.time()
        commandline=''
        for proc in processes:
            if '[' not in proc:
                commandline+="add process %s ;" % proc
            else:
                process, order, final = re.split('\[\s*(.*)\s*\]', proc)
                commandline+="add process %s;" % (process)
                if not order.startswith('virt='):
                    if 'QCD' in order:
                        commandline+="add process %s j ;" % (process)
                    else:
                        raise Exception('Madspin not implemented NLO corrections.')

        commandline = commandline.replace('add process', 'generate',1)
        misc.sprint(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
                
        commandline = 'output standalone_ms %s' % pjoin(path_me,'production_me')
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
                final_states.add(label)
        for key in self.list_branches.keys():
            if key not in final_states:
                del self.list_branches[key]

        # 4. compute the full matrix element -----------------------------------
        logger.info('generating the full square matrix element (with decay)')
        start = time.time()
        to_decay = self.mscmd.list_branches.keys()
        decay_text = []
        misc.sprint( self.mscmd.list_branches)
        for decays in self.mscmd.list_branches.values():
            for decay in  decays:
                if ',' in decay:
                    decay_text.append('(%s)' % decay)
                else:
                    decay_text.append(decay)
        decay_text = ', '.join(decay_text)
        commandline = ''
        for proc in processes:
            if '[' not in proc:
                if ',' in proc:
                    raise Exception, 'all decay need to be done by MS (for the moment)'
                else:
                    commandline+="add process %s, %s;" % (proc, decay_text)
            else:
                process, order, final = re.split('\[\s*(.*)\s*\]', proc)
                commandline+="add process %s, %s;" % (process, decay_text)
                if not order.startswith('virt='):
                    if 'QCD' in order:
                        commandline+="add process %s j, %s ;" % (process, decay_text)
                    else:
                        raise Exception('Madspin not implemented NLO corrections.')        
        
        
        commandline = commandline.replace('add process', 'generate',1)
        misc.sprint(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        # remove decay with 0 branching ratio.
        mgcmd.remove_pointless_decay(self.banner.param_card)

        commandline = 'output standalone_ms %s' % pjoin(path_me,'full_me')
        mgcmd.exec_cmd(commandline, precmd=True)
        logger.info('Done %.4g' % (time.time()-start))

        # 5. add the decay information to the all_topology object --------------                        
        for matrix_element in mgcmd._curr_matrix_elements.get_matrix_elements():
            me_path = pjoin(path_me,'full_me', 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())
            self.all_ME.add_decay(matrix_element, me_path)

        # 6. generate decay only part ------------------------------------------
        logger.info('generate matrix element for decay only (1 - > N).')
        start = time.time()
        commandline = ''
        for processes in self.list_branches.values():
            for proc in processes:
                commandline+="add process %s;" % (proc)        
        commandline = commandline.replace('add process', 'generate',1)
        misc.sprint(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        # remove decay with 0 branching ratio.
        mgcmd.remove_pointless_decay(self.banner.param_card)
        #
        commandline = 'output standalone_ms %s' % pjoin(path_me,'decay_me')
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
                                         'tag': me_string}
        #
        if __debug__:
            #check that all decay matrix element correspond to a decay only
            for prod in self.all_ME.values():
                for decay in prod['matrix_element']['base_amplitude']['process']['decay_chains']:
                    assert decay.shell_string() in self.all_decay
            
        
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
        
        return width    


    def compile(self):
        logger.info('Compiling code')
        self.compile_fortran(self.path_me, mode="production_me")
        self.compile_fortran(self.path_me, mode="full_me")
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
        misc.compile(arg=['clean'], cwd=pjoin(path_me, mode,"Source","MODEL"), mode='fortran')
        misc.compile( cwd=pjoin(path_me, mode,"Source","MODEL"), mode='fortran')                   

        file=pjoin(path_me, 'param_card.dat')
        shutil.copyfile(file,pjoin(path_me,mode,"Cards","param_card.dat"))    

        for direc in list_prod:
            if direc[0] == "P" and os.path.isdir(pjoin(base_dir, direc)):
                new_path = pjoin(base_dir, direc)

                if mode == 'production_me':
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_prod.f')
                else:
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_full.f')
                shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f"))  
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )

                # files to produce the parameters:
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
                    
                shutil.copyfile(pjoin(path_me, mode,'Source','MODEL','input.inc'),
                                pjoin(new_path,'input.inc'))
                if not os.path.exists(pjoin(new_path,os.path.pardir, 'parameters.inc')):
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
                label2width[particle_label]=float(width.value)
                #label2mass[particle_label]=float(mass.value)
                #pid2mass[part]=label2mass[particle_label]
                pid2width[abs(part)]=label2width[particle_label]
                if label2width[particle_label]==0.0:
                    need_param_card_modif = True
                    for param in self.model["parameters"][('external',)]:
                        if param.lhablock=="DECAY" and param.lhacode==[abs(part)]:
                            label2width[particle_label]=param.value
                            pid2width[abs(part)]=label2width[particle_label]
                    logger.warning('ATTENTION')
                    logger.warning('Found a zero width in the param_card for particle '\
                                   +str(particle_label))
                    logger.warning('Use instead the default value '\
                                   +str(label2width[particle_label]))
        # now we need to modify the values of the width
        # in param_card.dat, since this is where the input 
        # parameters will be read when evaluating matrix elements
        if need_param_card_modif:
            decay_misc.modify_param_card(self.pid2width, self.path_me)

    def get_max_weight_from_1toN(self):
        """Get the maximum weight value from the 1->N formula"""
        
        numberps = self.options['max_weight_ps_point'] # number of phase pace points per event
        
        starttime = time.time()
        for decay in self.all_decay.values():
            max_weight = 0
            dc_branch = decay['dc_branch']
            index = dc_branch['tree'][-1]['label']
            mass = self.pid2mass(index)
            width = self.pid2width[abs(index)]
            p  = momentum(mass, 0, 0, 0) 
            BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0
            
            for i in range(10*numberps):
                decay_products, jac = dc_branch.generate_momenta(p, 1, self.pid2width,
                                                          self.pid2mass, BW_cut)
                
                p_str = {0:p.nice_string()}
                for pos in decay_products.keys():
                    if pos < 1:
                        continue
                    mom = decay_products[pos]["momentum"]
                    p_str[pos] = mom.nice_string()
                p_str = '\n'.join([p_str[pos] for pos in range(len(p_str))]+[''])
                weight =  self.calculate_matrix_element('decay', decay['path'], p_str)
                #print p_str, weight
                #print weight, jac
                max_weight = max([max_weight, weight * jac / mass**2 / width**2])
            decay['max_weight']= max_weight
            if self.options['BW_effect']:
                decay['max_weight']= max_weight * math.atan(2*self.options['BW_cut'])*width/mass/2
            print decay['path'], decay['max_weight'], misc.format_timer(time.time()-starttime)
            
            
            
                
            
            


    def get_max_weight_from_event(self, decay_mapping):
        """ """

        misc.sprint(decay_mapping)
        decay_tools = decay_misc()
        
        # check all set of decay that need to be done:
        decay_set = set()
        for production in self.all_ME.values():
            decay_set.add(frozenset(production['decaying']))
        
        numberev = self.options['Nevents_for_max_weigth'] # number of events
        numberps = self.options['max_weight_ps_point'] # number of phase pace points per event
        
        logger.info('  ')
        logger.info('   Estimating the maximum weight    ')
        logger.info('   *****************************    ')
        logger.info('     Probing the first '+str(numberev)+' events')
        logger.info('     at '+str(numberps)+' phase space points')
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
            decaying = frozenset(self.all_ME[production_tag]['decaying'])
            if nb_decay[decaying] >=  numberev:
                continue 
            ev += 1
            nb_decay[decaying] += 1 
            mg5_me_prod, prod_values = self.evaluate_me_production(production_tag, event_map)   
            tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)
            topology = self.all_ME[production_tag][tag_topo]
    
            logger.debug('Event %s/%s: ' % (ev+1, len(decay_set)*numberev))
            logger.debug('Selected topology               : '+str(tag_topo))

            topology.dress_topo_from_event(self.curr_event)
            topology.extract_angles()

            if self.options['BW_effect']:
                decay_tools.set_light_parton_massless(self.all_ME[production_tag][tag_topo])
            
            max_decay = {}
            for dec in range(numberps):
                if self.options['BW_effect']:
                    BW_weight_prod, topology = self.reshuffle_event(production_tag, tag_topo, 
                                                        event_map, prod_values)
                else:
                    BW_weight_prod = 1
                #BW_weight_prod = 1
                topology.topo2event(self.curr_event)
                atleastonedecay = False
                for decay in topology.production['decays']:
                    tag = decay['decay_tag']
                    if decay_mapping and not tag in decay_mapping:
                        continue
                    atleastonedecay = True
                    BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0

                    decayed_event, BW_weight_decay = self.decay_one_event(\
                                        self.curr_event, 
                                        decay['decay_struct'],
                                        event_map,
                                        BW_cut)
                                        
                    if decayed_event==0:
                        logger.warning('failed to decay event properly')
                        continue
                    mg5_me_prod, prod_values = self.evaluate_me_production(production_tag, event_map)
                    #     then decayed weight:
                    p_full, p_full_str=decayed_event.give_momenta()
                    #print p_full_str
                    mg5_me_full = self.calculate_matrix_element('full',
                                                      decay['path'], p_full_str)
                    #print dec, mg5_me_full, mg5_me_prod, BW_weight_prod,BW_weight_decay,
                    weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod
                    if tag in max_decay:
                        max_decay[tag] = max([max_decay[tag], weight])
                    else:
                        max_decay[tag] = weight
                    #print weight, max_decay[name]
                    #raise Exception   
                if not atleastonedecay:
                    # NO decay [one possibility is all decay are identical to their particle]
                    logger.info('No independant decay for one type of final states -> skip those events')
                    nb_decay[decaying] = numberev
                    ev += numberev -1
                    break
            probe_weight[decaying].append(max_decay)

                    

            running_time = misc.format_timer(time.time()-starttime)
            info_text = 'Event %s/%s : %s \n' % (ev + 1, len(decay_set)*numberev, running_time) 
            for  index,tag_decay in enumerate(max_decay):
                info_text += '            decay_config %s [%s] : %s\n' % \
                   (index+1, ','.join(tag_decay), probe_weight[decaying][nb_decay[decaying]-1][tag_decay])
            logger.info(info_text[:-1])
        
        
        
        # Computation of the maximum weight used in the unweighting procedure
        for decaying in probe_weight:
            me_linked = [me for me in self.all_ME.values() if me['decaying'] == decaying]
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
                weights.sort()
                weights[len(weights)//2:]
                ave_weight, std_weight=decay_tools.get_mean_sd(weights)
                std_weight=math.sqrt(std_weight)
                base_max_weight = 1.05 * (ave_weight+4.5*std_weight)
                logger.info('Decay channel %s :Using maximum weight %s [%s]' % \
                           (','.join(decay_tag), base_max_weight, max(weights)))                
                for associated_decay, ratio in decay_mapping[decay_tag]:
                    max_weight= ratio * base_max_weight
                    if ratio != 1:
                        max_weight *= 1.1 #security
                        logger.info('Decay channel %s :Using maximum weight %s' % \
                                (','.join(associated_decay), max_weight))
                                                
                    #assign the value to the associated decays
                    for m in self.all_ME.values():
                        for mi in m['decays']:
                            if mi['decay_tag'] == associated_decay:
                                mi['max_weight'] = max_weight
 
        # sanity check that all decay have a max_weight
        if __debug__:
            for prod in self.all_ME.values():
                for dec in prod['decays']:
                    assert 'max_weight' in dec and dec['max_weight'], 'fail for %s' % (dec['decay_tag'])

            
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
            if len(in_event) == 1:
                in_event = in_event[0]
            else:
                in_event = in_event[random.randint(0, len(in_event)-1)]            
            evt_order[in_event] = 0
            event_map[i] = in_event
                    
        assert production_tag in self.all_ME
        
        
        
        
        #to_decay_map, to_decay_label = self.curr_event.get_map_resonances(\
        #                            self.dico_branchindex2label, self.pid2label)
    
        return production_tag, event_map
    
        
    def evaluate_me_production(self, production_tag, map_event):
        """return the numerical value of the matrix element"""
        p, p_str=self.curr_event.give_momenta(map_event)

        prod_values = self.calculate_matrix_element('prod',
                               self.all_ME[production_tag]['path'], p_str)
        prod_values=prod_values.replace("\n", "")
        prod_values=prod_values.split()
        mg5_me_prod = float(prod_values[0])
        return mg5_me_prod, prod_values    
    
    def calculate_matrix_element(self, mode, production, stdin_text):
        """routine to return the matrix element"""

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
        
        if len(self.calculator) > 100:
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
        
    def reshuffle_event(self, production_tag, tag_topo, event_map, prod_values):
        """reshuffle event"""
        #to_decay_label, to_decay_amp
        decay_tools = decay_misc()
        
        try_reshuffle=0
        while 1:
            try_reshuffle+=1
            if try_reshuffle > 10:
                logger.debug('current %s' % try_reshuffle)
            topology = self.all_ME[production_tag][tag_topo]
            
            BW_weight_prod = topology.generate_BW_masses(self.pid2width,self.pid2mass, \
                                                           self.curr_event.shat)

            # end sanity check
            if topology.reshuffle_momenta():
                if try_reshuffle > 10:
                    logger.debug('pass at %s' % try_reshuffle)
                break
            
            if try_reshuffle % 10 == 0:
                logger.debug( 'tried %ix to reshuffle the momenta, failed'% try_reshuffle)
                logger.debug( ' So let us try with another topology')
                tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)
                topology = self.all_ME[production_tag][tag_topo]
                topology.dress_topo_from_event(self.curr_event)
                topology.extract_angles()
                # sometimes not possible to set the masses of the external partons to zero,
                # keep the original masses
#                    decay_tools.set_light_parton_massless(self.topologies\
#                                              [tag_production][tag_topo])
                continue
                #try_reshuffle=0
                if try_reshuffle >100:
                    misc.sprint('fail 100 times')
                    break
 
        return BW_weight_prod, topology
        
    def decay_one_event(self,curr_event,decay_struct, event_map,BW_cut,ran=1):
        """Consider the production event recorded in "curr_event", and decay it
            according to the structure recoreded in "decay_struct".
            If ran=1: random decay, phi and cos theta generated according to 
               a uniform distribution in the rest frame of the decaying particle
            If ran=0 : use the previsously-generated angles and masses to get 
               the momenta."""

        pid2width = self.pid2width
        pid2mass = self.pid2mass
        pid2color = self.pid2color


        decayed_event=Event()
        decayed_event.event2mg={}

        if ran==0: # reshuffling phase: the decayed event is about to be written, so we need 
                    # to record some information 
            decayed_event.ievent=curr_event.ievent
            decayed_event.wgt=curr_event.wgt
            decayed_event.scale=curr_event.scale
            decayed_event.aqed=curr_event.aqed
            decayed_event.aqcd=curr_event.aqcd
            decayed_event.diese=curr_event.diese
            decayed_event.rwgt=curr_event.rwgt

        part_number=0
        external=0
        maxcol=curr_event.max_col
        weight=1.0

        #event2mg
        for index in curr_event.event2mg.keys():
            if curr_event.event2mg[index]>0:
                part=curr_event.event2mg[index]
                if part in decay_struct:
                    mom_init = curr_event.particle[event_map[part-1]+1]["momentum"].copy()
                    # sanity check
                    if mom_init.m<1e-6:
                        logger.debug('Decaying particle with mass less than 1e-6 GeV in decay_one_event_old')
                    
                    decay_products, jac=decay_struct[part].generate_momenta(mom_init,\
                                        ran, pid2width,pid2mass,BW_cut)

                    if ran==1:
                        if decay_products==0: return 0, 0
                        weight=weight*jac

                    # now we need to write the decay products in the event
                    # follow the decay chain order, so that we can easily keep track of the mother index
                    for res in range(-1,-len(decay_struct[part]["tree"].keys())-1,-1):
                        if (res==-1):
                            part_number+=1
                            mom=decay_products[res]["momentum"]
                            pid=decay_products[res]["pid"]
                            istup=2
                            mothup1=1
                            mothup2=2
                            colup1=curr_event.particle[part]["colup1"]
                            colup2=curr_event.particle[part]["colup2"]
                            decay_products[res]["colup1"]=colup1
                            decay_products[res]["colup2"]=colup2
                            mass=mom.m
                            helicity=0.
                            decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":colup1,"colup2":colup2,"momentum":mom,\
                                "mass":mass,"helicity":helicity}
                            decayed_event.event2mg[part_number]=part_number
                    #print part_number
                    #print pid
                    #print " "
                        mothup1=part_number
                        mothup2=part_number
#
#             Extract color information so that we can write the color flow
#
                        colormother=pid2color[decay_products[res]["pid"]]
                        colord1=pid2color[decay_products[decay_struct[part]\
                                            ["tree"][res]["d1"]["index"]]["pid"]]
                        colord2=pid2color[decay_products[decay_struct[part]\
                                            ["tree"][res]["d2"]["index"]]["pid"]]
                
                        colup1=decay_products[res]["colup1"]
                        colup2=decay_products[res]["colup2"]

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
                        else:
                            raise Exception, 'color combination not treated by MadSpin (yet). (%s,%s,%s)' \
                                % (colord1,colord2,colormother)
                        part_number+=1
                        mom=decay_products[decay_struct[part]\
                                    ["tree"][res]["d1"]["index"]]["momentum"]
                        pid=decay_products[decay_struct[part]\
                                    ["tree"][res]["d1"]["index"]]["pid"]


                        indexd1=decay_struct[part]["tree"][res]["d1"]["index"]
                        if ( indexd1>0):
                            istup=1
                            external+=1
                        else:
                            decay_products[indexd1]["colup1"]=d1colup1
                            decay_products[indexd1]["colup2"]=d1colup2
                            istup=2
                    
                        mass=mom.m
                        helicity=0.
                        decayed_event.particle[part_number]={"pid":pid,\
                                "istup":istup,"mothup1":mothup1,"mothup2":mothup2,\
                                "colup1":d1colup1,"colup2":d1colup2,"momentum":mom,\
                                "mass":mass,"helicity":helicity}
                        decayed_event.event2mg[part_number]=part_number

                        part_number+=1
                        mom=decay_products[decay_struct[part]["tree"][res]["d2"]\
                                           ["index"]]["momentum"]
                        pid=decay_products[decay_struct[part]["tree"][res]["d2"]\
                                           ["index"]]["pid"]

                        indexd2=decay_struct[part]["tree"][res]["d2"]["index"]
                        if ( indexd2>0):
                            istup=1
                            external+=1
                        else:
                            istup=2
                            decay_products[indexd2]["colup1"]=d2colup1
                            decay_products[indexd2]["colup2"]=d2colup2

                        mothup1=part_number-2
                        mothup2=part_number-2
                        mass=mom.m
                        helicity=0.
                        decayed_event.particle[part_number]={"pid":pid,"istup":istup,\
                           "mothup1":mothup1,"mothup2":mothup2,"colup1":d2colup1,\
                           "colup2":d2colup2,\
                           "momentum":mom,"mass":mass,"helicity":helicity}

                        decayed_event.event2mg[part_number]=part_number

                else:
                    external+=1 
                    part_number+=1
                    decayed_event.particle[part_number]=curr_event.particle[event_map[part-1]+1]
                    decayed_event.event2mg[part_number]=part_number
           
            else: # resonance in the production event
                if (ran==0): # write resonances in the prod. event ONLY if the 
                    # decayed event is ready to be written down    
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

        return decayed_event, weight        

    def add_loose_decay(self):
        """ in presence of multiprocess with multiple decay options all the 
        BR might not be identical. In such case, the total number of events should
        drop such that the events file is still a unweighted physical events sample.
        This routines add null decay (=> not written events) if appropriate."""
        
        first = True
        max_br = max([m['total_br'] for m in self.all_ME.values()])
        for production in self.all_ME.values():
            if production['total_br'] < max_br:
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
                assert production['total_br'] == min(total_br)
        
        self.branching_ratio = min(total_br) * eff
        
        
        self.banner['madspin'] += ms_banner
        # Update cross-section in the banner
        if 'mggenerationinfo' in self.banner:
            mg_info = self.banner['mggenerationinfo'].split('\n')
            for i,line in enumerate(mg_info):
                if 'Events' in line:
                    if eff == 1:
                        continue
                    nb_event =  int(int(mg_info[i].split()[-1]) * eff) 
                    mg_info[i] = '#  Number of Events        :       %i' % nb_event
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
            for (mode, production) in self.calculator:
                external = self.calculator[(mode, production)]
                external.terminate()
        else:
            try:
                external = self.calculator[('full', path_to_decay)]
            except Exception:
                pass
            else:
                external.terminate()       
    
#class decay_all_events_old:
#    
#    def __init__(self, ms_interface, banner, inputfile, options):
#        """Store all the component and organize special variable"""
#    
#        # input
#        self.options = options
#        #max_weight_arg = options['max_weight']  
#        #BW_effects = options['BW_effect']
#        self.path_me = os.path.realpath(options['curr_dir']) 
#        self.mgcmd = ms_interface.mg5cmd
#        self.mscmd = ms_interface
#        self.model = ms_interface.model
#        self.banner = banner
#        self.evtfile = inputfile
#        self.curr_event = Event(self.evtfile) 
#               
#        self.curr_dir = os.getcwd()
#    
#        # dictionary to fortan evaluator
#        self.calculator = {}
#        self.calculator_nbcall = {}
#        # need to unbuffer all I/O in fortran, otherwise
#        # the values of matrix elements are not passed to the Python script
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'  
#    
#        # Remove old stuff from previous runs
#        # so that the current run is not confused
#
#        if os.path.isdir(pjoin(self.path_me,"production_me")):
#            shutil.rmtree(pjoin(self.path_me,"production_me"))
#
#        if os.path.isdir(pjoin(self.path_me,"full_me")):
#            shutil.rmtree(pjoin(self.path_me,"full_me"))    
#    
#    
#        # Prepare some dict usefull for optimize model imformation
#        # pid -> label and label -> pid
#        self.pid2label=pid2label(self.model)
#        self.banner.check_pid(self.pid2label)
#        self.pid2label.update(label2pid(self.model))
#        # dictionary pid > color_rep
#        self.pid2color = pid2color(self.model)
#        
#        # width and mass information will be filled up later
#        self.pid2width={}
#        self.pid2mass={}
#        
#        if os.path.isfile(pjoin(self.path_me,"param_card.dat")):
#            os.remove(pjoin(self.path_me,"param_card.dat"))        
#
#        # now overwrite the param_card.dat in Cards:
#        param_card=self.banner['slha']
#        #param_card=decay_tools.check_param_card( param_card)
#
#        # now we can write the param_card.dat:
#        # Note that the width of each resonance in the    
#        # decay chain should be >0 , we will check that later on
#        param=open(pjoin(self.path_me,'param_card.dat'),"w")
#        param.write(param_card)
#        param.close()     
#        
#        self.list_branches = ms_interface.list_branches
#        
#             
#        
#    def run(self, decay_processes, prod_branches, proc_option):
#        """Running the full code"""
#        
#        self.prod_branches = prod_branches
#        self.proc_option = proc_option
#        self.decay_processes = decay_processes
#        
#        max_weight_arg = self.options['max_weight']  
#        BW_effects = self.options['BW_effect']
#        decay_tools=decay_misc()
#
#             
#        # process a bit the decay chain strings, so that the code is not confused by the syntax
#        decay_tools.process_decay_syntax(decay_processes)        
#        # extract all resonances in the decay:
#        resonances=decay_tools.get_resonances(decay_processes.values())
#        logger.debug('List of resonances: %s' % resonances)
#
#        self.extract_resonances_mass_width(resonances)
#
#        # now we need to evaluate the branching fractions:
#        # =================================================
#        branching_per_channel = self.get_branching_ratio_old(resonances, decay_processes)
#        # now we need to sort all the different decay configurations, and get
#        #the br for each of them
#        #       first key = a tag in decay_tags
#        #       second key :  ['br'] = float with the branching fraction for the 
#        #                              FS associated with 'tag'   
#        #                     ['config'] = a list of strings, each of them giving 
#        #                                  the definition of a branch    
#        # ======================================================================
#        self.multi_decay_processes = self.create_multi_decay_processes(decay_processes,
#                                                          branching_per_channel)
#        decay_tags = self.multi_decay_processes.keys()
#        # Compute the cumulative probabilities associated with the branching fractions
#        #keep track of sum of br over the decay channels at work, since we need to rescale
#        #the event weight by this number [-> modify self.multi_decay_processes]
#        sum_br=decay_tools.set_cumul_proba_for_tag_decay(self.multi_decay_processes)
#
#
#
#
#        # A few initialisations:
#        #========================
#        #    consider the possibility of several production process
#        self.set_of_processes=[]
#        self.decay_struct={}
#        self.full_proc_line={}
#        self.decay_path={}      # dictionary to record the name of the directory with decay fortran me
#        self.production_path={} # dictionary to record the name of the directory with production fortran me
#                                #    also for production matrix elements, 
#        #    we need to keep track of the self.topologies
#        self.topologies={}
#        self.dico_branchindex2label=decay_tools.get_dico_branchindex2label(decay_processes)
#        #print 'dico_branchindex2label'
#        #print dico_branchindex2label
#        symm_fac=1.0    #decay_tools.get_symm_fac(decay_processes)
#        # the symmetry factor is not used anymore, since we explicitely generate ALL possible decay channels
#
#
#        #Next step: we need to determine which matrix elements are really necessary
#        #==========================================================================
#        decay_me_tags, map_decay_me = self.get_identical_decay(resonances)
#        
#        # Estimation of the maximum weight
#        #=================================
#        if max_weight_arg>0:
#            max_weight={}
#            for tag_decay in decay_me_tags: 
#                max_weight[tag_decay]=max_weight_arg
#        else:
#            max_weight = self.get_max_weight(resonances, decay_me_tags)
#        
#        # Performing the decay for all events
#        #====================================
#        logger.info(' ' )
#        logger.info('Decaying the events... ')
#        
#        self.outputfile = open(pjoin(self.path_me,'decayed_events.lhe'), 'w')
#        self.write_banner_information(decay_tags, max_weight, map_decay_me)
#        
#        
#        event_nb=0
#        trial_nb_all_events=0
#        starttime = time.time()
#        while 1:
#            tag_production, to_decay_label, to_decay_map = self.load_event(decay_me_tags)
#            if not tag_production:
#                break # No more event
#            event_nb+=1
#            if (event_nb % max(int(10**int(math.log10(float(event_nb)))),10)==0): 
#                running_time = misc.format_timer(time.time()-starttime)
#                logger.info('Event nb %s %s' % (event_nb, running_time))
#            trial_nb=0
#            
#            # First evaluate production matrix element            
#            mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)
#            #     select topology based on sigle-diagram weights
#            tag_topo, cumul_proba=decay_tools.select_one_topo(prod_values)
#            #    dress the topology with momenta and extract the canonical numbers 
#            self.topologies[tag_production][tag_topo].dress_topo_from_event(self.curr_event,to_decay_label)
#            self.topologies[tag_production][tag_topo].extract_angles()
#            if BW_effects:
#                decay_tools.set_light_parton_massless(self.topologies[tag_production][tag_topo])
#
#            while 1:
#                trial_nb += 1
#                BW_weight_prod, tag_topo = self.reshuffle_event(tag_production, tag_topo, 
#                                      to_decay_label, to_decay_map, prod_values)
#
#                # Here we need to select a decay configuration on a random basis:
#                tag_decay=decay_tools.generate_tag_decay(self.multi_decay_processes,decay_tags)
#
#                self.topologies[tag_production][tag_topo].topo2event(self.curr_event,to_decay_label)
#                BW_cut = self.options['BW_cut'] if BW_effects else 0
#                
#                decayed_event, BW_weight_decay = decay_tools.decay_one_event_old(
#                   self.curr_event,self.decay_struct[tag_production][tag_decay], \
#                   self.pid2color, to_decay_map, self.pid2width, self.pid2mass, \
#                   resonances,BW_cut)
#
#                if decayed_event==0: 
#                    logger.info('failed to decay one event properly')
#                    continue # means we had mA<mB+mC in one splitting A->B+C
#                
#                
#                mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)
#
#                # then decayed weight:
#                p_full, p_full_str = decayed_event.give_momenta()
#                mg5_me_full = self.calculate_matrix_element('full', 
#                       self.decay_path[tag_production][map_decay_me[tag_decay]], 
#                       p_full_str)
#
#                weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod
#                if weight > max_weight[map_decay_me[tag_decay]]: 
#                    logger.info('warning: got a larger weight than max_weight estimate')
#                    logger.info('the ratio with the max_weight estimate is '+str(weight/max_weight[map_decay_me[tag_decay]]))
#                    logger.info('decay channel ')
#                    logger.info(self.multi_decay_processes[tag_decay])
#                    
#                if (weight/max_weight[map_decay_me[tag_decay]]> random.random()):
#
#                    # Here we need to restore the masses of the light partons 
#                    # initially found in the lhe production event
#                    decay_tools.restore_light_parton_masses(self.topologies[tag_production][tag_topo],self.curr_event)
#                    succeed = self.topologies[tag_production][tag_topo].reshuffle_momenta()
#                    if not succeed:
#                        logger.info('Warning: unable to restore masses of light partons')
#                    else:
#                        self.topologies[tag_production][tag_topo].topo2event(self.curr_event,to_decay_label)
#                    self.curr_event.reset_resonances() # re-evaluate the momentum of each resonance in prod. event
#                    BW_cut = self.options['BW_cut'] if BW_effects else 0
#                    decayed_event, BW_weight_decay=decay_tools.decay_one_event_old(self.curr_event,self.decay_struct[tag_production][tag_decay], \
#                                            self.pid2color, to_decay_map, self.pid2width, self.pid2mass, resonances,BW_cut,ran=0)
#                    decayed_event.wgt=decayed_event.wgt*sum_br*float(symm_fac)
#                    self.outputfile.write(decayed_event.string_event())
#                #print "number of trials: "+str(trial_nb)
#                    trial_nb_all_events+=trial_nb
#                    break # pass to next event.
# 
#        self.outputfile.write('</LesHouchesEvents>\n')
#        self.evtfile.close()
#        self.outputfile.close()
#
#        logger.info('Total number of events: '+str(event_nb))
#        logger.info('Average number of trial points per production event: '\
#            +str(float(trial_nb_all_events)/float(event_nb)))
#        logger.info('Number of subprocesses '+str(len(self.decay_path)))
#        
#        # Closing all run
#        self.terminate_fortran_executables()
#        shutil.rmtree(pjoin(self.path_me,'production_me'))
#        shutil.rmtree(pjoin(self.path_me,'full_me'))
#
#        # set the environment variable GFORTRAN_UNBUFFERED_ALL 
#        # to its original value
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'
#
#
#    def write_banner_information(self, decay_tags, max_weight, map_decay_me):
#        
#        ms_banner = ""
#        total_br = 0
#        for index,tag_decay in enumerate(decay_tags):
#            ms_banner+="# Decay channel "+str(index+1)+"\n"
#            for part in self.multi_decay_processes[tag_decay]['config'].keys():
#                ms_banner+="# "+self.multi_decay_processes[tag_decay]['config'][part]+"\n"
#            ms_banner+="# branching fraction: "+str(self.multi_decay_processes[tag_decay]['br']) + "\n"
#            ms_banner+="# estimate of the maximum weight: "+str(max_weight[map_decay_me[tag_decay]]) + "\n"
#            total_br += self.multi_decay_processes[tag_decay]['br']
#        self.branching_ratio = total_br
#        self.banner['madspin'] += ms_banner
#        # Update cross-section in the banner
#        if 'mggenerationinfo' in self.banner:
#            mg_info = self.banner['mggenerationinfo'].split('\n')
#            for i,line in enumerate(mg_info):
#                if 'Events' in line:
#                    continue
#                if ':' not in line:
#                    continue
#                info, value = line.rsplit(':',1)
#                try:
#                    value = float(value)
#                except:
#                    continue
#                mg_info[i] = '%s : %s' % (info, value * total_br)
#            self.banner['mggenerationinfo'] = '\n'.join(mg_info)
#        if 'init' in self.banner:
#            new_init =''
#            for line in self.banner['init'].split('\n'):
#                if len(line.split()) != 4:
#                    new_init += '%s\n' % line
#                else:
#                    data = [float(nb) for nb in line.split()]
#                    data[:3] = [ data[i] *total_br for i  in range(3)]
#                    new_init += ' %.12E %.12E %.12E %i\n' % tuple(data)
#            self.banner['init'] = new_init
#        self.banner.write(self.outputfile, close_tag=False)
#        
#        
#        
#
#    def extract_resonances_mass_width(self, resonances):
#        """ """
#
#        label2width = {}
#        label2mass = {}
#        pid2width = self.pid2width 
#        pid2mass = self.pid2mass
#        need_param_card_modif = False
#        
#        # now extract the width of the resonances:
#        for particle_label in resonances:
#            try:
#                part=self.pid2label[particle_label]
#                mass = self.banner.get('param_card','mass', abs(part))
#                width = self.banner.get('param_card','decay', abs(part))
#            except ValueError, error:
#                continue
#            else:
#                label2width[particle_label]=float(width.value)
#                label2mass[particle_label]=float(mass.value)
#                pid2mass[part]=label2mass[particle_label]
#                pid2width[part]=label2width[particle_label]
#                if label2width[particle_label]==0.0:
#                    need_param_card_modif = True
#                    for param in self.model["parameters"][('external',)]:
#                        if param.lhablock=="DECAY" and param.lhacode==[abs(part)]:
#                            label2width[particle_label]=param.value
#                            pid2width[part]=label2width[particle_label]
#                    logger.warning('ATTENTION')
#                    logger.warning('Found a zero width in the param_card for particle '\
#                                   +str(particle_label))
#                    logger.warning('Use instead the default value '\
#                                   +str(label2width[particle_label]))
#        # now we need to modify the values of the width
#        # in param_card.dat, since this is where the input 
#        # parameters will be read when evaluating matrix elements
#        if need_param_card_modif:
#            decay_misc.modify_param_card(self.pid2width, self.path_me)
#
#            
#
#
#
#    def get_branching_ratio_old(self, resonances, decay_processes):
#        """determine the branching ratio of the model and of the various 
#        decay channel."""
#        
#        logger.info('Determining partial decay widths...')
#
#        calculate_br = width_estimate_old(resonances, self.path_me, self.pid2label,
#                                                           self.banner, self.model)
##       Maybe the branching fractions are already given in the banner:
#        calculate_br.extract_br_from_banner(self.banner)
#        calculate_br.print_branching_fractions()
##
##        now we check that we have all needed pieces of info regarding the branching fraction:
#        multiparticles = self.mgcmd._multiparticles
#        branching_per_channel=calculate_br.get_BR_for_each_decay(decay_processes,
#                                                                 multiparticles)
#        
#
#        # check that we get branching fractions for all resonances to be decayed:
#        
#        if branching_per_channel == 0:
#            logger.debug('We need to recalculate the branching fractions')
#            if hasattr(self.model.get('particles')[0], 'partial_widths'):
#                logger.debug('using the compute_width module of madevent')
#                calculate_br.launch_width_evaluation(resonances,self.model, self.mgcmd) # use FR to get all partial widths                                      # set the br to partial_width/total_width
#            else:
#                logger.debug('compute_width module not available, use numerical estimates instead ')
#                calculate_br.extract_br_from_width_evaluation()
#            calculate_br.print_branching_fractions()
#            branching_per_channel=calculate_br.get_BR_for_each_decay(decay_processes,multiparticles)       
#
#        if branching_per_channel==0:
#            raise Exception, 'Failed to extract the branching fraction associated with each decay channel'
#
#        return branching_per_channel
#
#
#
#
#    def create_multi_decay_processes(self, decay_processes, branching_per_channel):
#        
#        # 1. first create a list of branches that ordered according to the 
#        #canonical numeratation 1, 2, ...:
#        list_particle_to_decay=decay_processes.keys()
#        list_particle_to_decay.sort()
#
#        # also determine which particles are identical
#        identical_part=decay_misc.get_identical(decay_processes)
#
#        #       2. then use a tuple to identify a decay channel
#        #       (tag1 , tag2 , tag3, ...)  
#        #       where the number of entries is the number of branches, 
#        #       tag1 is the tag that identifies the final state of branch 1, 
#        #       tag2 is the tag that identifies the final state of branch 2, 
#        #       etc ...
#        
#        #       2.a. first get decay_tags = the list of all the tuples
#
#        decay_tags=[]
#        for part in list_particle_to_decay:  # loop over particle to decay in the production process
#            branch_tags=[ fs for fs in branching_per_channel[part]] # list of tags in a given branch
#            decay_tags=decay_misc.update_tag_decays(decay_tags, branch_tags)
#        #      Now here is how we account for symmetry if identical particles:
#        #      EXAMPLE:  (Z > mu+ mu- ) (Z> e+ e-)
#        #      currently in decay_tags, the first Z is mapped onto the channel mu+ mu- 
#        #                               the second Z is mapped onto the channel e+ e-
#        #                              => we need to add the symmetric channel obtained by 
#        #                                 swapping the role of the first Z and the role of the snd Z 
#        decay_tags, map_tag2branch =\
#            decay_misc.symmetrize_tags(decay_tags, identical_part)
#            
#            #       2.b. then build the dictionary self.multi_decay_processes = multi dico
##       first key = a tag in decay_tags
##       second key :  ['br'] = float with the branching fraction for the FS associated with 'tag'   
##                     ['config'] = a list of strings, each of them giving the definition of a branch    
#
#        multi_decay_processes={}
#        for tag in decay_tags:
##           compute br + get the config
#            br=1.0
#            list_branches={}
#            for index, part in enumerate(list_particle_to_decay):
#                br=br*branching_per_channel[map_tag2branch[tag][part]][tag[index]]['br']
#                list_branches[part]=branching_per_channel[map_tag2branch[tag][part]][tag[index]]['config']
#            multi_decay_processes[tag]={}
#            multi_decay_processes[tag]['br']=br
#            multi_decay_processes[tag]['config']=list_branches
#            
#        return multi_decay_processes
#            
#
#    def terminate_fortran_executables(self, path_to_decay=0 ):
#	"""routine to terminate all fortran executables"""
#
#        if not path_to_decay:
#            for (mode, production) in self.calculator:
#                external = self.calculator[(mode, production)]
#                external.terminate()
#        else:
#            try:
#                external = self.calculator[('full', path_to_decay)]
#            except Exception:
#                pass
#            else:
#                external.terminate()
#
#    def calculate_matrix_element(self, mode, production, stdin_text):
#        """routine to return the matrix element"""
#
#        tmpdir = ''
#        if (mode, production) in self.calculator:
#            external = self.calculator[(mode, production)]
#            self.calculator_nbcall[(mode, production)] += 1
#        else:
#            logger.debug('we have %s calculator ready' % len(self.calculator))
#            if mode == 'prod':
#                tmpdir = pjoin(self.path_me,'production_me', 'SubProcesses',
#                           production)
#            else:
#                tmpdir = pjoin(self.path_me,'full_me', 'SubProcesses',
#                           production)
#            executable_prod="./check"
#            external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, 
#                                                      stderr=STDOUT, cwd=tmpdir)
#            self.calculator[(mode, production)] = external 
#            self.calculator_nbcall[(mode, production)] = 1       
#
#                    
#        external.stdin.write(stdin_text)
#        if mode == 'prod':
#            info = int(external.stdout.readline())
#            nb_output = abs(info)+1
#        else:
#            info = 1
#            nb_output = 1
#         
#
#   
#        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])
#        if info < 0:
#            print 'ZERO DETECTED'
#            print prod_values
#            print stdin_text
#            os.system('lsof -p %s' % external.pid)
#            return ' '.join(prod_values.split()[-1*(nb_output-1):])
#        
#        if len(self.calculator) > 100:
#            logger.debug('more than 100 calculator. Perform cleaning')
#            nb_calls = self.calculator_nbcall.values()
#            nb_calls.sort()
#            cut = max([nb_calls[len(nb_calls)//2], 0.001 * nb_calls[-1]])
#            for key, external in list(self.calculator.items()):
#                nb = self.calculator_nbcall[key]
#                if nb < cut:
#                    external.stdin.close()
#                    external.stdout.close()
#                    external.terminate()
#                    del self.calculator[key]
#                    del self.calculator_nbcall[key]
#                else:
#                    self.calculator_nbcall[key] = self.calculator_nbcall[key] //10
#        
#        if mode == 'prod':
#            return prod_values
#        else:
#            return float(prod_values)
#    
#    
#    def load_event(self, decay_me_tags):
#        """Load the next event and ensure that the ME is define"""
#
#        decay_tools = decay_misc()
#        if self.curr_event.get_next_event() == 'no_event':
#            return 0, 0, 0
#        to_decay_map, to_decay_label = self.curr_event.get_map_resonances(\
#                                    self.dico_branchindex2label, self.pid2label)
#    
#        prod_process = self.curr_event.give_procdef(self.pid2label)
#        extended_prod_process=decay_tools.find_resonances(prod_process, self.prod_branches)
#    
#        tag_production=prod_process.replace(">", "_")
#        tag_production=tag_production.replace(" ","")
#        tag_production=tag_production.replace("~","x")
#        
#        # CHECK if we need to generate new matrix elements
#        if tag_production not in self.set_of_processes: 
#            logger.info('Found a new process: '+extended_prod_process+self.proc_option)
##                    logger.info(prod_process)
##                    logger.info('Re-interpreted as    ')
##                    logger.info(extended_prod_process+proc_option)
##                    logger.info( tag_production)
#            logger.debug( ' -> need to generate the corresponding fortran matrix element ... ')
#            self.set_of_processes.append(tag_production)
#
#            # generate fortran me for production only
#            self.topologies[tag_production]=\
#                decay_tools.generate_fortran_me([extended_prod_process+self.proc_option],\
#                    self.banner.get("model"), 0,self.mgcmd,self.path_me)
#
#            prod_name=decay_tools.compile_fortran_me_production(self.path_me)
#            self.production_path[tag_production]=prod_name
#
#            # for the decay, we need to keep track of all possibilities for the decay final state:
#            self.decay_struct[tag_production]={}
#            self.decay_path[tag_production]={}
#            for tag_decay in self.multi_decay_processes:
#                new_full_proc_line, new_decay_struct=\
#                    decay_tools.get_full_process_structure(self.multi_decay_processes[tag_decay]['config'],\
#                    extended_prod_process, self.model, self.banner,self.proc_option)
#                self.decay_struct[tag_production][tag_decay]=new_decay_struct
#                if tag_decay in decay_me_tags:
#                    self.full_proc_line[tag_decay]=new_full_proc_line
##
#            for tag_decay in decay_me_tags:
#                new_full_proc_line=self.full_proc_line[tag_decay]
#                decay_tools.generate_fortran_me([new_full_proc_line+self.proc_option],\
#                                            self.banner.get("model"), 1,self.mgcmd,self.path_me)
#
#                decay_name=decay_tools.compile_fortran_me_full(self.path_me)
#                self.decay_path[tag_production][tag_decay]=decay_name
#            
#            logger.debug('Done.')
#        return tag_production, to_decay_label, to_decay_map
#        
#    def reshuffle_event(self, tag_production, tag_topo, to_decay_label, to_decay_map, prod_values):
#        """reshuffle event"""
#        
#        decay_tools = decay_misc()
#        
#        try_reshuffle=0
#        while 1:
#            try_reshuffle+=1
#            if try_reshuffle > 10:
#                logger.debug('current %s' % try_reshuffle)                
#            BW_weight_prod = self.generate_BW_masses(\
#                   self.topologies[tag_production][tag_topo], \
#                   to_decay_label.values(),self.pid2label, self.pid2width,self.pid2mass, \
#                   self.curr_event.shat)
#
#            succeed=self.topologies[tag_production][tag_topo].reshuffle_momenta()
#            # sanlity check
#            for part in self.topologies[tag_production][tag_topo]['get_momentum'].keys():
#                if part in to_decay_map and \
#                       self.topologies[tag_production][tag_topo]['get_momentum'][part].m<1.0:
#                    logger.debug('Mass of a particle to decay is less than 1 GeV')
#                    logger.debug('in reshuffling loop')
#            # end sanity check
#            if succeed:
#                if try_reshuffle > 10:
#                    logger.debug('pass at %s' % try_reshuffle)
#                break
#            if try_reshuffle % 10 == 0:
#                logger.debug( 'tried %ix to reshuffle the momenta, failed'% try_reshuffle)
#                logger.debug( ' So let us try with another topology')
#                tag_topo, cumul_proba=decay_tools.select_one_topo(prod_values)
#                self.topologies[tag_production][tag_topo].dress_topo_from_event(\
#                                                         self.curr_event,to_decay_label)
#                self.topologies[tag_production][tag_topo].extract_angles()
#
#                # sometimes not possible to set the masses of the external partons to zero,
#                # keep the original masses
##                    decay_tools.set_light_parton_massless(self.topologies\
##                                              [tag_production][tag_topo])
#                continue
#                #try_reshuffle=0
#                if try_reshuffle >100:
#                    misc.sprint('fail 100 times')
#                    break
# 
#        return BW_weight_prod, tag_topo
# 
#        
#    def evaluate_me_production(self, tag_production):
#        """return the numerical value of the matrix element"""
#        p, p_str=self.curr_event.give_momenta()
#        prod_values = self.calculate_matrix_element('prod',
#                               self.production_path[tag_production], p_str)
#        prod_values=prod_values.replace("\n", "")
#        prod_values=prod_values.split()
#        mg5_me_prod = float(prod_values[0])
#        return mg5_me_prod, prod_values
#        
#    def get_identical_decay(self, resonances):
#        """identify the various decay which are identical to each other"""
#        
#        decay_tools = decay_misc()
#        decay_tags = self.multi_decay_processes.keys()
#        
#        if len(decay_tags) == 1:
#            return decay_tags, dict([(t,t) for t in decay_tags])
#        
#        
#        
#        BW_effects = self.options['BW_effect']
#        
#        # create a dictionnary that maps a 'tag_decay' onto 'tag_me_decay' = tag for the matrix elements
#        # call this map 'map_decay_me' 
#        logger.info('Checking the equality of matrix elements for different decay channels ... ')
#        check_weights=dict([(tag,[]) for tag in decay_tags])
#        curr_event=self.curr_event
#        # load the first event and create all production/decay channel
#        tag_production, to_decay_label, to_decay_map = self.load_event(decay_tags)
#
#        mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)
#
#        # select a topology for the reshuffling
#        tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)
#
#        # extract the canonical phase-space variable based on that topology       
#        self.topologies[tag_production][tag_topo].dress_topo_from_event(curr_event,to_decay_label)
#        self.topologies[tag_production][tag_topo].extract_angles()
#
#        # Breit-Wigner effects + reshuffling      
#        decay_tools.set_light_parton_massless(self.topologies[tag_production][tag_topo])
#        for dec in range(100):
#            BW_weight_prod, tag_topo = self.reshuffle_event(tag_production, tag_topo, 
#                                      to_decay_label, to_decay_map, prod_values)
#            BW_cut = self.options['BW_cut'] if BW_effects else 0
#            decayed_event, BW_weight_decay = decay_tools.decay_one_event_old(\
#                     curr_event,self.decay_struct[tag_production][decay_tags[0]], \
#                     self.pid2color, to_decay_map, self.pid2width, \
#                     self.pid2mass, resonances,BW_cut)
#
#            if decayed_event==0:
#                logger.warning('failed to decay event properly')
#                continue
#            for tag_decay in decay_tags:
#                #     set the momenta for the production event and the decayed event:
#                mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)
#                p_full, p_full_str=decayed_event.give_momenta()
#                #     then decayed weight:
#                mg5_me_full =self.calculate_matrix_element('full',
#                            self.decay_path[tag_production][tag_decay], p_full_str)
#                os.chdir(self.curr_dir)
#                weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod
#                check_weights[tag_decay].append(weight)
#
##       verify if some matrix elements are identical up to an overal factor                
#        map_decay_me={}
#        decay_me_tags=[]
#        for tag_decay in decay_tags:
#            tag=decay_tools.check_decay_tag(tag_decay,map_decay_me.values(),check_weights)
#            if tag==0:
#                map_decay_me[tag_decay]=tag_decay
#                decay_me_tags.append(tag_decay)
#            else:
#                map_decay_me[tag_decay]=tag
#                self.terminate_fortran_executables(self.decay_path[tag_production][tag_decay])
#                shutil.rmtree(pjoin(self.path_me,'full_me', 'SubProcesses',self.decay_path[tag_production][tag_decay]))
#
#        logger.info('Out of %d decay channels, %s matrix elements are independent ' % (len(decay_tags), len(decay_me_tags)))
##        for tag in decay_me_tags:
##            logger.info(self.multi_decay_processes[tag])
#        self.evtfile.seek(0)
#        return decay_me_tags, map_decay_me
#    
#    def get_max_weight(self, resonances, decay_me_tags):
#        """ """
#        decay_tools = decay_misc()
#        
#        numberev = self.options['Nevents_for_max_weigth'] # number of events
#        numberps = self.options['max_weight_ps_point'] # number of phase pace points per event
#        
#        logger.info('  ')
#        logger.info('   Estimating the maximum weight    ')
#        logger.info('   *****************************    ')
#        logger.info('     Probing the first '+str(numberev)+' events')
#        logger.info('     at '+str(numberps)+' phase space points')
#        logger.info('  ')
#        
#        curr_event = self.curr_event
#        probe_weight=[]
#        
#        starttime = time.time()
#        for ev in range(numberev):
#            probe_weight.append({})
#            for tag_decay in decay_me_tags:
#                probe_weight[ev][tag_decay]=0.0
#            tag_production, to_decay_label, to_decay_map = self.load_event(decay_me_tags)
#        
#            mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)   
#            tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)
#    
#            logger.debug('Event %s: ' % (ev+1))
#            logger.debug('Number of production self.topologies : '\
#                        +str(len(self.topologies[tag_production].keys())))
#            logger.debug('Selected topology               : '+str(tag_topo))
#        
#            self.topologies[tag_production][tag_topo].dress_topo_from_event(curr_event,to_decay_label)
#            self.topologies[tag_production][tag_topo].extract_angles()
#
#            if self.options['BW_effect']:
#                decay_tools.set_light_parton_massless(self.topologies[tag_production][tag_topo])
#        
#            for dec in range(numberps):
#                if self.options['BW_effect']:
#                    BW_weight_prod, tag_topo = self.reshuffle_event(tag_production, tag_topo, 
#                                  to_decay_label, to_decay_map, prod_values)
#                else:
#                    BW_weight_prod = 1 
#                self.topologies[tag_production][tag_topo].topo2event(curr_event,to_decay_label)
#                
#                for tag_decay in decay_me_tags:
#                    BW_cut = self.options['BW_cut'] if self.options['BW_effect'] else 0
#                    decayed_event, BW_weight_decay=decay_tools.decay_one_event_old(\
#                                        curr_event,self.decay_struct[tag_production][tag_decay], \
#                                        self.pid2color, to_decay_map, self.pid2width, \
#                                        self.pid2mass, resonances,BW_cut)
#        
#                    if decayed_event==0:
#                        logger.warning('failed to decay event properly')
#                        continue
#                    
#                    mg5_me_prod, prod_values = self.evaluate_me_production(tag_production)
#                    #     then decayed weight:
#                    p_full, p_full_str=decayed_event.give_momenta() 
#                    mg5_me_full =self.calculate_matrix_element('full',
#                                     self.decay_path[tag_production][tag_decay], p_full_str)
#                    weight=mg5_me_full*BW_weight_prod*BW_weight_decay/mg5_me_prod
#                    if (weight>probe_weight[ev][tag_decay]): 
#                        probe_weight[ev][tag_decay] = weight
#            running_time = misc.format_timer(time.time()-starttime)
#            info_text = 'Event %s/%s : %s \n' % (ev + 1, numberev, running_time) 
#            for  index,tag_decay in enumerate(decay_me_tags):
#                info_text += '            decay_config %s : %s\n' % (index+1, probe_weight[ev][tag_decay])
#            logger.info(info_text[:-1])
#        # Computation of the maximum weight used in the unweighting procedure
#        max_weight={}
#        
#        for index, tag_decay in enumerate(decay_me_tags):
#            weights=[]
#            for ev in range(numberev):
#                weights.append(probe_weight[ev][tag_decay])
#            ave_weight, std_weight=decay_tools.get_mean_sd(weights)
#            std_weight=math.sqrt(std_weight)
#            logger.info(' ')
#            logger.info(' Decay channel '+str(index+1))
#            for part in self.multi_decay_processes[tag_decay]['config'].keys():
#                logger.info(self.multi_decay_processes[tag_decay]['config'][part])
#        #                logger.info('     average maximum weight that we got is '+str(ave_weight))
#        #                logger.info('     with a standard deviation of          '+str(std_weight))
#        #                logger.info('     -> W_max = average + 4 * standard deviation')
#            max_weight[tag_decay]=ave_weight+4.0*std_weight
#            logger.info('      Using maximum weight '+str(max_weight[tag_decay]))
#        
#        self.evtfile.seek(0)
#        return max_weight
#        
#    def transpole(self,pole,width):
#
#        """ routine for the generation of a p^2 according to 
#            a Breit Wigner distribution
#            the generation window is 
#            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
#        """
#
#        gap = float(self.options['BW_cut'])
#
#        zmin = math.atan(-gap)/width
#        zmax = math.atan(gap)/width
#
#        z=zmin+(zmax-zmin)*random.random()
#        y = pole+width*math.tan(width*z)
#
#        jac=(width/math.cos(width*z))**2*(zmax-zmin)
#        return y, jac
#
#
#    def generate_BW_masses (self,topo, to_decay, pid2name, pid2width, pid2mass,s):
#        """Generate the BW masses of the particles to be decayed in the production event    """    
#
#        weight=1.0
#        for part in topo["get_id"].keys():
#            pid=topo["get_id"][part]
#            if pid2name[pid] in to_decay:
#                mass=0.25
#                width=pid2width[pid]*pid2mass[pid]/(2.0*pid2mass[pid])**2
#                virtualmass2, jac=self.transpole(mass,width)
#                virtualmass2=virtualmass2*(2.0*pid2mass[pid])**2
#                weight=weight*jac
#                #print "need to generate BW mass of "+str(part)
#                ##print "mass: "+str(pid2mass[pid])
#                #print "width: "+str(pid2width[pid])
#                #print "virtual mass: "+str(math.sqrt(virtualmass2))
#                #print "jac: "+str(jac)
#                old_mass=topo["get_mass2"][part]
#                topo["get_mass2"][part]=virtualmass2
#                # sanity check
#                if pid2mass[pid]<1e-6:
#                    logger.debug('A decaying particle has a mass of less than 1e-6 GeV')
## for debugg purposes:
#                if abs((pid2mass[pid]-math.sqrt(topo["get_mass2"][part]))/pid2mass[pid])>1.0 :
#                    logger.debug('Mass after BW smearing affected by more than 100 % (1)') 
#                    logger.debug('Pole mass: '+str(pid2mass[pid]))
#                    logger.debug('Virtual mass: '+str(math.sqrt(topo["get_mass2"][part])))
#                                       
#                #print topo["get_mass2"]         
#
##    need to check if last branch is a t-branching. If it is, 
##    we need to update the value of branch["m2"]
#
#        if len(topo["branchings"])>0:  # Exclude 2>1 self.topologies
#            if topo["branchings"][-1]["type"]=="t":
#                if topo["branchings"][-2]["type"]!="t":
#                    logger.debug('last branching is t-channel')
#                    logger.debug('but last-but-one branching is not t-channel')
#                else:
#                    part=topo["branchings"][-1]["index_d2"] 
#                    if part >0: # reset the mass only if "part" refers to an external particle 
#                        old_mass=topo["branchings"][-2]["m2"]
#                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])
#                        #sanity check
#
#                        if abs(old_mass-topo["branchings"][-2]["m2"])>1e-10:
#                            if abs((old_mass-topo["branchings"][-2]["m2"])/old_mass)>1.0 :
#                                logger.debug('Mass after BW smearing affected by more than 100 % (2)')
#                                logger.debug('Previous value: '+ str(old_mass))
#                                logger.debug('New mass: '+ str((topo["branchings"][-2]["m2"])))
#                                try:
#                                    pid=topo["get_id"][part]
#                                    logger.debug('pole mass: %s' % pid2mass[pid])
#                                except Exception:
#                                    pass
#        return weight
#

