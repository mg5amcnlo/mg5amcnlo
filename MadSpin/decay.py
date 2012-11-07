#! /usr/bin/env python


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
################################################################################
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
#import madgraph.core.base_objects as base_objects
#import madgraph.core.helas_objects as helas_objects
#import madgraph.core.diagram_generation as diagram_generation

import models.import_ufo as import_ufo
#import madgraph.various.process_checks as process_checks
#from time import strftime
#from madgraph.interface.madgraph_interface import MadGraphCmd
import madgraph.interface.master_interface as Cmd
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

    def give_momenta(self):
        """ return the set of external momenta of the event, 
                in two different formats:
                p is list momenta, with each momentum a list [E,px,py,pz]
                string is a sting
        """
        p=[]
        string=""
        for part in range(1,len(self.particle)+1):
            if self.particle[part]["istup"]<2:
                mom=[self.particle[part]["momentum"].E, \
                     self.particle[part]["momentum"].px,\
                     self.particle[part]["momentum"].py,\
                     self.particle[part]["momentum"].pz]
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

    def string_event(self):
        """ return a string with the information of the event written 
                in the lhe format.
        """
        line="<event> \n"
        line1=' %2d %6d %14.8e %14.8e %14.8e %14.8e' % \
        (self.nexternal,self.ievent,self.wgt,self.scale,self.aqed,self.aqcd)
        line1=line1.replace("e","d")
        line+=line1+"\n"
        for item in range(1,len(self.event2mg.keys())+1):
            part=self.event2mg[item]
            if part>0:
                particle_line=self.get_particle_line(self.particle[part])
            else:
                particle_line=self.get_particle_line(self.resonance[part])
            line+=particle_line        

        if self.diese!="0":
            line+=self.diese
        if self.rwgt!="0":
            line+=self.rwgt
        line+="</event> \n"
        return line


    def get_particle_line(self,leg):

        line=" %8d %2d %4d %4d %4d %4d %+13.7e %+13.7e %+13.7e %14.8e %14.8e %10.4e %10.4e" \
            % (leg["pid"], leg["istup"],leg["mothup1"],leg["mothup2"],\
               leg["colup1"],leg["colup2"],leg["momentum"].px,leg["momentum"].py,\
                leg["momentum"].pz,leg["momentum"].E, leg["momentum"].m,\
                 0.0,float(leg["helicity"]) )
        line=line.replace("e","d")
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

        if len(daughters)!=2: logger.warning("Got more than 2 daughters for one particles")

        if daughters[0]>0:
            momentum_mother=self.particle[daughters[0]]["momentum"].copy()
        else:    
            momentum_mother=self.resonance[daughters[0]]["momentum"].copy()

        if daughters[1]>0:
            momentum_mother=momentum_mother.add(self.particle[daughters[1]]["momentum"])
        else:
            momentum_mother=momentum_mother.add(self.resonance[daughters[1]]["momentum"])

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

    def get_next_event(self):
        """ read next event in the lhe event file """
        while 1:
            line=self.inputfile.readline()
            if line=="": return "no_event"
            if line.find("<event>")>-1:
                line=self.inputfile.readline()
                line=line.replace("\n","")
                inputs=line.split()
                self.nexternal=int(inputs[0])
                self.ievent=int(inputs[1])
                self.wgt=float(inputs[2].replace("D","E"))
                self.scale=float(inputs[3].replace("D","E"))
                self.aqed=float(inputs[4].replace("D","E"))
                self.aqcd=float(inputs[5].replace("D","E"))
                index_prod=0
                index_external=0
                index_resonance=0
                self.particle={}
                self.resonance={}
                self.max_col=500
                self.diese="0"
                self.rwgt="0"
                self.event2mg={} # dict. event-like label <-> "madgraph-like" label
                                                            # in "madgraph-like", resonances are labeled -1, -2, ...                                        
                while 1:
                    line=self.inputfile.readline()
#                 aMC@NLO events have an extra line starting with #
                    if line[0]=="#":
                        self.diese=line
                        continue
#                 aMC@NLO events have extra lines for PDF and scale uncertainties #
                    if line.find("<rwgt>")>-1:
                        self.rwgt=line
                        while 1:
                            line=self.inputfile.readline()
                            self.rwgt=self.rwgt+line
                            if line.find("</rwgt>")>-1: break
                        continue
                    if line=="": return "no_event"
                    if line.find("</event>")>-1:break
                    index_prod+=1
                    line=line.replace("\n","")
                    line=line.replace("D","E")
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
                self.shat=self.particle[1]["momentum"].dot(self.particle[2]["momentum"])
                return 1
                break


class Banner:
    """ A banner object contains all the information about the process,
    the run and the parameters of the model"""
 
    def __init__(self,inputfile):
        """Initialize self.file=input file """
        self.inputfile =inputfile
        self.proc={}
        self.run={}
        self.param={}
#
        self.proc_keys=["model","generate"]
        self.param_keys=["Block mass","Block sminputs","Block yukawa", "DECAY"]
        self.whole_banner=""

    def ReadBannerFromFile(self):
        """Read the whole banner"""    

        while 1:
            line=self.inputfile.readline()
            if line=="": break
            self.whole_banner+=line
            if line.find("</init>")>-1: break
            if string.find(line,"MG5ProcCard")>-1:self.ReadProc()
            if string.find(line,"slha")>-1:self.ReadParam()
            if string.find(line,"MGRunCard")>-1:self.ReadRun()

        if "model" not in self.proc.keys():
            logger.warning('The model was not found in the banner, assuming the Standard Model')
#        if len(self.proc)!=len(self.proc_keys):     
#            logger.warning('Some parameters have not been read in the proc card')
#        if len(self.param)!=len(self.param_keys): 
#            logger.warning('Some parameters have not been read in the param card')

    def GetProcValue(self,line,keyword):
        """Extract a parameter from a line in the Proc Card"""

        line = line.split("#", 2)[0]
        index=string.find(line,keyword)+len(keyword)
        line=line[index:-1]
        while 1:
            if line[0]==" ":line=line[1:]
            elif line[-1]==" ":line=line[:-1]
            else: break
        return line

    def ReadProc(self): 
        """ Read the parameters associated with the Proc Card """
        while 1:
            line=self.inputfile.readline()
            self.whole_banner+=line
            if line=="" or string.find(line,"MG5ProcCard")>-1: break
            for key in self.proc_keys:
                if string.find(line,key)>-1: self.proc[key]=self.GetProcValue(line,key)
        # if self.proc["model"].find("loop_sm")>-1:self.proc["model"]="sm"

    def GetParamValue(self,line,keyword):
        """Extract the tag and the value from a line in the Param Card"""

        if line[0]=="#": return 
        line = line.split("#", 2)[0]
        line=line.split()
        if len(line)==2: return int(line[0]), line[1]
        elif line[0]==keyword: return int(line[1]), line[2]
#       Here I need to check if 

    def ReadParam(self): 
        """ Read the paramters associated with the Param Card """
        while 1:
            line=self.inputfile.readline()
            self.whole_banner+=line
            if line=="" or string.find(line,"slha")>-1: break
            for key in self.param_keys:
                if string.find(line,key)>-1:
                    currentkey=key
                    if currentkey not in self.param: self.param[currentkey]={}     
                    
            try:
                tag, value=self.GetParamValue(line,currentkey)
                self.param[currentkey][tag]=value
            except: continue

    def ReadRun(self):
        """ Read the paramters associated with the Run Card """
        pass

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
            self[-particle["pdg_code"]]=-particle["color"]

class label2pid(dict):
    """ dico label:pid for a given model"""

    def __init__(self,model):
        
        for particle in model["particles"]:
            self[particle["name"]]=particle.get_pdg_code()
            self[particle["antiname"]]=-particle.get_pdg_code()
            if particle['self_antipart']:
                self[particle["name"]]=abs(self[particle["name"]])
                self[particle["antiname"]]=abs(self[particle["antiname"]])
                 

class mass_n_width(dict):
    """ dictionary to extract easily the mass ad the width of the particle in the model 
            {name : {"mass": mass_value, "width": widht_value} }
            I assume that masses and widths are real
    """

    def __init__(self, model,banner):
        """ Fill the dictionary based on the information in the banner """

        self.model=model
        self.banner=banner
        for particle in model["particles"]:
            self[particle["name"]]={"mass":0.0, "width":0.0}
            if particle["mass"]!="ZERO":
                self[particle["name"]]["mass"]=self.get_mass(particle["pdg_code"])
            if particle["width"]!="ZERO":
                self[particle["name"]]["width"]=self.get_width(particle["pdg_code"])

    def get_mass(self,pid):
        """ extract the mass of particle with PDG code=pid from the banner"""
        try: 
            mass=self.banner.param["Block mass"][pid]
        except:
            mass=0.0

        return mass
    

    def get_width(self,pid):
        """ extract the width of particle with PDG code=pid from the banner"""
        try:
            width=self.banner.param["DECAY"][pid]
        except:
            width=0.0
        return width


class dc_branch(dict):
    """ A dictionary to record information necessary to decay particles 
            { -1 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },    
                -2 : {"d1": { "label": XX , "nb": YY },    "d2": { "label": XX , "nb": YY }    },
                ....
            }
    """

    def __init__(self, process_line,model,banner,check):

        self.banner=banner
        self.label2pid=label2pid(model)
        list_decays=process_line.split(",")
        self.nb_decays=len(list_decays)
        self["m_label2index"]={}
        self["m_index2label"]={}
        
        for dc_nb, dc in enumerate(list_decays):
            if dc.find(">")<0: logger.warning('warning: invalid decay chain syntax')
            mother=dc[:dc.find(">")].replace(" ","")
            self["m_label2index"][mother]=-dc_nb-1
            self["m_index2label"][-dc_nb-1]=mother
            
        self["nexternal"]=self.find_tree(list_decays)
        if (check): self.check_parameters() 
        # here I check that the relevant masses and widths can 
        # be extracted from the banner    

    def check_parameters(self):
        for res in range(-1,-self.nb_decays-1,-1): 
            d1=self["tree"][res]["d1"]["index"]
            d2=self["tree"][res]["d2"]["index"]
            if (d1>0): 
                try: 
                    logger.info('Mass of particle with id '\
                                +str(self.label2pid[self["tree"][res]["d1"]["label"]]))
                    logger.info(self.banner.param["Block mass"]\
                         [abs(self.label2pid[self["tree"][res]["d1"]["label"]])])
                except:
                    logger.info('The mass of particle with id '\
                                +str(self.label2pid[self["tree"][res]["d1"]["label"]]))
                    logger.info('was not defined in the param_card.dat') 
                    mass=raw_input("Please enter the mass: ")
                    self.banner.param["Block mass"][abs(self.label2pid[self["tree"][res]["d1"]["label"]])]=mass

            if (d2>0):
                try:
                    logger.info('Mass of particle with id '\
                                +str(self.label2pid[self["tree"][res]["d2"]["label"]]))
                    logger.info(     self.banner.param["Block mass"]\
                         [abs(self.label2pid[self["tree"][res]["d2"]["label"]])])
                except:
                    logger.info('The mass of particle with id '\
                                +str(self.label2pid[self["tree"][res]["d2"]["label"]]))
                    logger.info('was not defined in the param_card.dat')
                    mass=raw_input("Please enter the mass: ")
                    self.banner.param["Block mass"][abs(self.label2pid[self["tree"][res]["d2"]["label"]])]=mass


    def transpole(self,pole,width):

        """ routine for the generation of a p^2 according to 
            a Breit Wigner distribution
            the generation window is 
            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
        """

        zmin = math.atan(-30.0)/width
        zmax = math.atan(30.0)/width

        z=zmin+(zmax-zmin)*random.random()
        y = pole+width*math.tan(width*z)

        jac=(width/math.cos(width*z))**2*(zmax-zmin)
        return y, jac

    def generate_momenta(self,mom_init,ran, pid2width,pid2mass,resonnances,BW_effects):
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

        index2mom[-1]={}
        index2mom[-1]["momentum"]=mom_init
        if index2mom[-1]['momentum'].m < 1e-3:
            logger.debug('Decaying particle with m> 1e-3 GeV in generate_momenta')
        index2mom[-1]["pid"]=self.label2pid[self["m_index2label"][-1]]
        index2mom[-1]["status"]=2
        weight=1.0
        for res in range(-1,-self.nb_decays-1,-1): 
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
            #print index2mom[res]["momentum"].px
            #print index2mom[res]["momentum"].py
            #print index2mom[res]["momentum"].pz
            #print index2mom[res]["momentum"].E
            #print mA
            #print " "

            d1=self["tree"][res]["d1"]["index"]
            d2=self["tree"][res]["d2"]["index"]

#         For the daughters, the mass is either generate (intermediate leg + BW mode on)
#         or set to the pole mass (external leg or BW mode off)
#         If ran=0, just read the value from the previous generation of momenta 
#                             (this is used for reshuffling purposes) 
            if d1>0 or not BW_effects :
                mB=float(self.banner.param["Block mass"][abs(self.label2pid[self["tree"]\
                                                                [res]["d1"]["label"]])])
            elif ran==0:    # reshuffling phase
                mB=self["tree"][res]["d1"]["mass"]
            else:
                pid=self.label2pid[self["tree"][res]["d1"]["label"]]
#             NOTE: here pole and width are normalized by 4.0*mB**2,
#             Just a convention
                pole=0.25         #pid2mass[pid]**2/mA**2
                width=pid2width[pid]*pid2mass[pid]/(4.0*pid2mass[pid]**2)     #/mA**2
                mB, jac=self.transpole(pole,width)
                mB=math.sqrt(mB*4.0*pid2mass[pid]**2)
#             record the mass for the reshuffling phase, 
#             in case the point passes the reweighting creteria
                self["tree"][res]["d1"]["mass"]=mB
#             update the weigth of the phase-space point
                weight=weight*jac

            if d2>0 or not BW_effects:
                mC=float(self.banner.param["Block mass"][abs(self.label2pid[self["tree"]\
                                                                [res]["d2"]["label"]])])
            elif ran==0:
                mC=self["tree"][res]["d2"]["mass"]
            else:
                pid=self.label2pid[self["tree"][res]["d2"]["label"]]
#             NOTE: here pole and width are normalized by 4.0*mC**2,
#             Just a convention
                pole=0.25    #pid2mass[pid]**2/mA**2
                width=pid2width[pid]*pid2mass[pid]/(4.0*pid2mass[pid]**2) #mA**2
                mC, jac=self.transpole(pole,width)
                mC=math.sqrt(mC*4.0*pid2mass[pid]**2)
#             record the mass for the reshuffling phase, 
#             in case the point passes the reweighting creteria
                self["tree"][res]["d2"]["mass"]=mC
#             update the weigth of the phase-space point
                weight=weight*jac


                if (mA<mB+mC):
                    logger.debug('mA<mB+mC in generate_momenta')
                    logger.debug('mA = %s' % mA)
                    return 0, 0 # If that happens, throw away the DC phase-space point ...
                        # I don't expect this to be inefficient, since there is a BW cut

            if ran==1:
                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, mB,mC)
#             record the angles for the reshuffling phase, 
#             in case the point passes the reweighting creteria
                self["tree"][res]["costh"]=decay_mom.costh
                self["tree"][res]["sinth"]=decay_mom.sinth
                self["tree"][res]["cosphi"]=decay_mom.cosphi
                self["tree"][res]["sinphi"]=decay_mom.sinphi
            else:
#             we are in the reshuffling phase, 
#             so we read the angles that have been stored from the 
#             previous phase-space point generation
                costh=self["tree"][res]["costh"]
                sinth=self["tree"][res]["sinth"]
                cosphi=self["tree"][res]["cosphi"]
                sinphi=self["tree"][res]["sinphi"]
                decay_mom=generate_2body_decay(index2mom[res]["momentum"],mA, mB,mC,\
                                 costh_val=costh, sinth_val=sinth, cosphi_val=cosphi, \
                                 sinphi_val=sinphi)

#         record the momenta for later use
            index2mom[self["tree"][res]["d1"]["index"]]={}
            index2mom[self["tree"][res]["d1"]["index"]]["momentum"]=decay_mom.momd1
            index2mom[self["tree"][res]["d1"]["index"]]["pid"]=self.label2pid[self["tree"]\
                                                                    [res]["d1"]["label"]]

            index2mom[self["tree"][res]["d2"]["index"]]={}
            index2mom[self["tree"][res]["d2"]["index"]]["momentum"]=decay_mom.momd2
            index2mom[self["tree"][res]["d2"]["index"]]["pid"]=self.label2pid[self["tree"]\
                                                                    [res]["d2"]["label"]]

            if (self["tree"][res]["d1"]["index"]>0):
                index2mom[self["tree"][res]["d1"]["index"]]["status"]=1
            else:
                index2mom[self["tree"][res]["d1"]["index"]]["status"]=2
            if (self["tree"][res]["d2"]["index"]>0):
                index2mom[self["tree"][res]["d2"]["index"]]["status"]=1
            else:
                index2mom[self["tree"][res]["d2"]["index"]]["status"]=2

        return index2mom, weight

    def find_tree(self,list_decays):
        """ 
            record the topology of the decay chain in suitable variables
            This is roughly the equivalent of the configs.inc file in madevent
        """
        self["tree"]={}
        nexternal=0
        for mother in range(-1, -len(list_decays)-1, -1):
            self["tree"][mother]={}
            self["tree"][mother]["d1"]={}
            self["tree"][mother]["d2"]={}
 
#            print "filling the tree"
#            print "res "+str(mother)
            dc_nb=-(mother)-1
            daughters=list_decays[dc_nb][list_decays[dc_nb].find(">")+1:].split()
            self["tree"][mother]["d1"]["label"]=daughters[0]
            self["tree"][mother]["d2"]["label"]=daughters[1]

            if self["m_label2index"].has_key(daughters[0]):
                self["tree"][mother]["d1"]["index"]=self["m_label2index"][daughters[0]]
            else:
                nexternal=nexternal+1
                self["tree"][mother]["d1"]["index"]=nexternal
            if self["m_label2index"].has_key(daughters[1]):    
                self["tree"][mother]["d2"]["index"]=self["m_label2index"][daughters[1]]
            else:
                nexternal=nexternal+1
                self["tree"][mother]["d2"]["index"]=nexternal        
        return nexternal

    def print_branch(self):
        """Print the decay chain structure (for debugging purposes)"""
        length=len(self["tree"])
        for res in range(-1,-length-1, -1):
            logger.info('Decay '+str(res))
            #print "Mother: "+self["tree"][res]
            logger.info( 'd1: '+str(self["tree"][res]["d1"]["label"])+\
                         '    '+str(self["tree"][res]["d1"]["index"]))
            logger.info('d2: '+str(self["tree"][res]["d2"]["label"])+\
                        '     '+str(self["tree"][res]["d2"]["index"]))
    
class momentum:
    """A class to handel 4-vectors and the associated operations """
    def __init__(self,E,px,py,pz):
        self.px=px
        self.py=py
        self.pz=pz
        self.E=E
        self.mod2=px**2+py**2+pz**2
        self.sq=E**2-self.mod2
        if (self.sq) > self.mod2*1e-10:
            self.m=math.sqrt(self.sq)
#     in case we get a very small negative value, set the mass to zero
        elif (self.sq) > -self.mod2*1e-10: self.m=0.0

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

    def nice_string(self):
        return str(self.E)+" "+str(self.px)+" "+str(self.py)+" "+str(self.pz)

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
                ppz=(qt**2*self.py+q.py*q.pz*self.pz-q.px*qt*ppy)/qq/q.py
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
#            print self.mA
#            print self.mB
#            print self.mC
        return math.sqrt(lam)




class production_topo(dict):
    """ A dictionnary to record information about a given topology of a production event 

                self["branchings"] is a list of the branchings defining the topology (see class branching)
                self["get_mass2"] is a dictionnary {index -> mass**2 of the corresponding particle} 
                self["get_momentum"] is a dictionnary {index -> momentum of the corresponding particle} 
                self["get_id"] is a dictionnary {index -> pid the corresponding particle} 

            Note: index= "madgraph-like" numerotation of the particles
    """

    def __init__(self):
        """ Initialise the dictionaries+list used later on to record the information
                about the topology of a production event.
                Note that self["branchings"] is a list, 
                as it makes it easier to scan the topology in the ascendent order
        """
        self["branchings"]=[]
        self["get_mass2"]={}
        self["get_momentum"]={}
        self["get_id"]={}

    def add_one_branching(self,index_propa, index_d1,index_d2,type_propa):
        """ add the information of one splitting in the topology """
        branch=branching(index_propa, index_d1,index_d2,type_propa)
        self["branchings"].append(branch)


    def topo2event(self,event,to_decay):
        """This routine is typically called and the end of the reshuffling phase.
             The momenta in the topology were reshuffled in a previous step, and now they are copied 
             back to the production event in this routine 
        """
#     start with external legs
        for part in range(1,len(event.particle)+1):
            event.particle[part]["momentum"]=self["get_momentum"][part].copy()
            if part in to_decay:
                if event.particle[part]["momentum"].m < 1.0e-3:
                    logger.debug('Decaying particle with a mass of less than 1 MeV in topo2event')
#            print part 
#            print self["get_momentum"][part].nice_string()
#            print event.particle[part]["momentum"].nice_string()

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
#                print "momentum propa"
#                print self["get_momentum"][propa].nice_string()
#                print "P^2:    "+str(self["get_mass2"][propa])
#                print "root:    "+str(math.sqrt(abs(self["get_mass2"][propa])))
#                print "momentum d1"
#                print self["get_momentum"][d1].nice_string()
#                print "P^2:    "+str(self["get_mass2"][d1])
#                print "root:    "+str(math.sqrt(abs(self["get_mass2"][d1])))
#                print "momentum d2"
#                print self["get_momentum"][d2].nice_string()
#                print "P^2:    "+str(self["get_mass2"][d2])
#                print "root:    "+str(math.sqrt(abs(self["get_mass2"][d2])))
#            except:
#                print "topology not yet dressed" 

    def dress_topo_from_event(self,event,to_decay):
        """ event has been read from the production events file,
                use these momenta to dress the topology
        """

#     start with external legs
        for part in range(1,len(event.particle)+1):
            self["get_momentum"][part]=event.particle[part]["momentum"].copy()
            self["get_mass2"][part]=event.particle[part]["mass"]**2
            self["get_id"][part]=event.particle[part]["pid"]
            if part in to_decay :
                if self["get_momentum"][part].m<1e-3:
                    logger.debug\
                    ('decaying particle with m < 1MeV in dress_topo_from_event (1)')
                if self["get_mass2"][part]<1e-3:
                    logger.debug\
                    ('decaying particle with m < 1MeV in dress_topo_from_event (2)')

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
#             print self["get_mass2"][d2]
            MC=math.sqrt(self["get_mass2"][d2])
            if MA < MB + MC:
#                 print "WARNING: s-channel propagator with too low p^2 "
#                 print "in branch "+str(iter)
#                 print "MA = "+str(MA)
#                 print "MB = "+str(MB)
#                 print "MC = "+str(MC)
#                 print "throw away the point"

#                 print "increase the value of p^2"
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
#                 print "WARNING: m1^2 is negative for t-branching "+str(iter)
#                 print self["get_mass2"][id1]
#                 print "throw away the point"
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
#                 print "WARNING: (pa+pb)^2 is negative for t-branching "
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
#                        print "WARNING: cannot get the momentum of p1 in t-branching    "+str(iter)
#                        print "pp is negative : "+ str(pp)
#                        print "m1: "+ str(m1)
#                        print "m2: "+ str(m2)
#                        print "id1: "+ str(id1)
#                        print "throw away the point"
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
                    p1z=pp
                    pt=0.0
                    t=m1*m1+ma2-2.0*p1E*E_acms+2.0*p_acms*p1z
                    diff_t=abs((t-oldt)/t)*100
                    if (diff_t>2.0): 
                        logger.warning('t invariant was changed by '+str(diff_t)+' percents')
                else:
#                     print "WARNING: |p|^2 is smaller than p1z^2 in t-branching "+str(iter)
#                     print "|p| : "+str(pp) 
#                     print "pz : "+str(p1z) 
#                 print "throw away the point"
#                 print "Set pz^2=|p|^2 and recompute t"
#                 print "previous t:"+str(-math.sqrt(abs(t)))+"^2"
#                 p1z=pp
#                 pt=0.0
#                 t=m1*m1+ma2-2.0*p1E*E_acms+2.0*p_acms*p1z
#                 print "new t:"+str(-math.sqrt(abs(t)))+"^2"
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
#                print " p1 is "
#                print p1.nice_string()
                #print " pr is "
                #print pr.nice_string()
                p2=(pa.add(pb)).subtract(p1)
#                print " p2 is "
#                print p2.nice_string()
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

class width_estimate:
    """All methods used to calculate branching fractions"""

    def __init__(self,resonances,path_me,pid2label_dic,model):

	self.resonances=resonances
	self.path_me=path_me
	self.pid2label=pid2label_dic
	self.model=model

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

    def get_BR_for_each_decay(self,decay_processes,multiparticles,model,base_model,pid2label):
	""" get the list for possible decays & the associated branching fraction  """
        

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
                            logger.info('Branching fractions of particle '+initial+' are unknown')
	        	    return 0
                        if len(final)>2:
                            raise Exception, 'splittings different from A > B +C are currently not implemented '

                        if final[0] in multiparticles.keys():
                            set_B=[pid2label[pid] for pid in multiparticles[final[0]]]
 		        else:
                            if final[0] not in [ particle['name'] for particle in base_model['particles'] ] \
                               and final[0] not in [ particle['antiname'] for particle in base_model['particles'] ]:
                               raise Exception, "No particle "+item+ " in the model "+mybanner.proc["model"]
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
			for chan in self.br[initial].keys(): # loop over all channels
			    got_it=0
                            for d1 in set_B: 
				for d2 in set_C:
                                  if (d1==self.br[initial][chan]['daughters'][0] and \
                                     d2==self.br[initial][chan]['daughters'][1]) or \
				     (d2==self.br[initial][chan]['daughters'][0] and \
                                     d1==self.br[initial][chan]['daughters'][1]):
                                      split=" "+initial+" > "+d1+" "+d2+" "
                                      counter+=1
                                      splittings['s'+str(counter)]={}
                                      splittings['s'+str(counter)]['config']=split
                                      splittings['s'+str(counter)]['br']=self.br[initial][chan]['br']
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

                else:                             # case 4: ponctuation symbol outside a splitting
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

 	    for chan in self.br[res].keys():
		bran=self.br[res][chan]['br']
		d1=self.br[res][chan]['daughters'][0]
		d2=self.br[res][chan]['daughters'][1]
	        logger.info('   %e            %s  %s ' % (bran, d1, d2) )
	logger.info('  ')

    def print_partial_widths(self):
	""" print a list of all known partial widths"""

	for res in self.br.keys():
	    logger.info('  ')
 	    logger.info('decay channels for '+res+' :')
	    logger.info('       width                     d1  d2' )

 	    for chan in self.br[res].keys():
		width=self.br[res][chan]['width']
		d1=self.br[res][chan]['daughters'][0]
		d2=self.br[res][chan]['daughters'][1]
	        logger.info('   %e            %s  %s ' % (width, d1, d2) )
	logger.info('  ')

    def extract_br_from_card(self,filename):
        """read the file width_calculator/Events/run_width/param_card.dat
	   and extrach the branching fractions

           for each resonance with label 'res', and for each channel with index i,
	   returns a dictionary 
		branching_fractions[res][i]
           with keys
		'd1' : label of the first daughter
		'd2' : label of the second daughter
		'br' : value of the branching fraction 
	"""
	os.chdir(self.path_me)
	trappe=open(filename)
        branching_fractions={}
        while 1:
 	    line=trappe.readline()
	    if line=="": break
	    list1=line.split()
	    if len(list1)==0: continue
	    if list1[0]=='DECAY':
		label_mother=self.pid2label[int(list1[1])]
                pos=trappe.tell()
                line=trappe.readline()  
		if line=="": break
     	        list2=line.split()
	        if len(list2)<2:
		    trappe.seek(pos) 
                    continue
		if list2[1]=='BR':
		    channel_index=0
		    branching_fractions[label_mother]={}	
		    while 1:
		        line=trappe.readline()  
		        if line=="" or line[0]=='#': break
			channel_index+=1
			#logger.info('Found a new channel in the param_card.dat')
			branching_fractions[label_mother][channel_index]={}
			list3=line.split()
			if list3[1]=='2':
			    branching_fractions[label_mother][channel_index]['daughters']=[]
			    branching_fractions[label_mother][channel_index]['br']=\
							float(list3[0]) 
			    branching_fractions[label_mother][channel_index]\
                                   ['daughters'].append(self.pid2label[int(list3[2])])
			    branching_fractions[label_mother][channel_index]\
                                   ['daughters'].append(self.pid2label[int(list3[3])])
                else:
                    trappe.seek(pos)

	self.br=branching_fractions


    def generate_code_for_width_evaluation(self,mgcmd):
        """ use madgraph to generate me's for res > all all  
            OBSELETE: NOW WE USE THE FR IMPLEMENTATION
        """
   
        commandline="import model "+self.model
        mgcmd.exec_cmd(commandline)#

	logger.info("generate "+self.resonances[0]+" > all all")
        commandline="generate "+self.resonances[0]+" > all all"
	mgcmd.exec_cmd(commandline)

        if len(self.resonances)>1:
	    for index in range(1,len(self.resonances)):
		logger.info("add process "+self.resonances[index]+" > all all")		
	        commandline="add process "+self.resonances[index]+" > all all "
		mgcmd.exec_cmd(commandline)

	commandline="output width_calculator -f"
	mgcmd.exec_cmd(commandline)
	
    def extract_br_for_antiparticle(self,label2pid,pid2label):
        '''  
            for each channel with a specific br value, 
            set the branching fraction of the complex conjugated channel 
            to the same br value 
        '''

	for res in self.br.keys():
            anti_res=pid2label[-label2pid[res]]
            self.br[anti_res]={}
 	    for chan in self.br[res].keys():
                self.br[anti_res][chan]={}
		bran=self.br[res][chan]['br']
		d1=self.br[res][chan]['daughters'][0]
		d2=self.br[res][chan]['daughters'][1]
                d1bar=pid2label[-label2pid[d1]]
                d2bar=pid2label[-label2pid[d2]]
                self.br[anti_res][chan]['br']=bran
                self.br[anti_res][chan]['daughters']=[]
                self.br[anti_res][chan]['daughters'].append(d1bar)
                self.br[anti_res][chan]['daughters'].append(d2bar)
                if self.br[res][chan].has_key('width'):
		    self.br[anti_res][chan]['width']=self.br[res][chan]['width']
		    
    def extract_br_from_width(self):
        """
          If compute_width is used to get the partial width values from FR formulae,
          we still need to evaluate the br's from the partial widths
        """
	for res in self.br.keys():
            total_width=0.0
 	    for chan in self.br[res].keys():
 		total_width+=self.br[res][chan]['width']
 	    for chan in self.br[res].keys():
 		self.br[res][chan]['br']=self.br[res][chan]['width']/total_width                   


    def launch_width_evaluation(self,resonances,label2pid,pid2label,model):
        """ launch the calculation of the partial widths """

        # first build a set resonances with pid>0
	# since compute_width cannot be used for particle with pid<0
        
        particle_set=[]
        for part in resonances:
            if label2pid[part]>0: particle_set.append(part)
        for part in resonances:
	    if label2pid[part]<0:
                pid_part=-label2pid[part]
                if pid2label[pid_part] not in particle_set:
		    particle_set.append(pid2label[pid_part])  

        # erase old info
        del self.br
        self.br={}

        data = model.set_parameters_and_couplings(pjoin(self.path_me,
                                                              'param_card.dat'))

        # find UFO particles linked to the require names. COPIED FROM madevent_interface.py
        for part in particle_set:
            self.br[part]={}
            chan=0
            pid=label2pid[part] 
            particle = model.get_particle(pid)

            mass = abs(eval(str(particle.get('mass')), data).real)
            data = model.set_parameters_and_couplings(pjoin(self.path_me,
                                            'param_card.dat'), scale= mass)
            for mode, expr in particle.partial_widths.items():
                tmp_mass = mass
                for p in mode:
                    tmp_mass -= abs(eval(str(p.mass), data))
                if tmp_mass <=0:
                    continue

                decay_to = [p.get('pdg_code') for p in mode]
                value = eval(expr,{'cmath':cmath},data).real
                if -1e-10 < value < 0:
                    value = 0
                if -1e-5 < value < 0:
                    logger.warning('Partial width for %s > %s negative: %s automatically set to zero' %
                                   (particle.get('name'), ' '.join([p.get('name') for p in mode]), value))
                    value = 0
                elif value < 0:
                    raise Exception, 'Partial width for %s > %s negative: %s' % \
                                   (particle.get('name'), ' '.join([p.get('name') for p in mode]), value)

                if value>0:
                    chan+=1
                    self.br[part][chan]={}
                    self.br[part][chan]['width']=value
                    self.br[part][chan]['daughters']=[]
                    for pid in decay_to:
                        self.br[part][chan]['daughters'].append(pid2label[pid])




class decay_misc:
    """class with various methods for the decay"""


   

    def decay_one_event(self,curr_event,decay_struct,pid2color_dico,\
                        to_decay, pid2width,pid2mass,resonnances,BW_effects,ran=1):

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
                    # sanity check
                    if mom_init.m<1e-6:
                        logger.debug('Decaying particle with mass less than 1e-6 GeV in decay_one_event')
                    decay_products, jac=decay_struct[part].generate_momenta(mom_init,\
                                        ran, pid2width,pid2mass,resonnances,BW_effects)

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
#                    print part_number
#                    print pid
#                    print " "
                            mothup1=part_number
                            mothup2=part_number
#
#             Extract color information so that we can write the color flow
#
                        colormother=pid2color_dico[decay_products[res]["pid"]]
                        colord1=pid2color_dico[decay_products[decay_struct[part]\
                                            ["tree"][res]["d1"]["index"]]["pid"]]
                        colord2=pid2color_dico[decay_products[decay_struct[part]\
                                            ["tree"][res]["d2"]["index"]]["pid"]]
                
                        colup1=decay_products[res]["colup1"]
                        colup2=decay_products[res]["colup2"]

#            now figure out what is the correct color flow informatio
#            Only consider 1,3, 3-bar and 8 color rep.
#            Normally, the color flow needs to be determined only
#            during the reshuffling phase, but it is currenlty assigned 
#            for each "trial event"
                
                        if abs(colord2)==1:
                            d1colup1=colup1
                            d1colup2=colup2
                            d2colup1=0
                            d2colup2=0

                        if abs(colord1)==1:
                            d2colup1=colup1
                            d2colup2=colup2
                            d1colup1=0
                            d1colup2=0

                        if colord1==3 and colord2==-3 and colormother ==1:
                            maxcol+=1
                            d1colup1=maxcol
                            d1colup2=0
                            d2colup1=0
                            d2colup2=maxcol
                     
                        if colord1==-3 and colord2==3 and colormother ==1:
                            maxcol+=1
                            d1colup1=0
                            d1colup2=maxcol
                            d2colup1=maxcol
                            d2colup2=0

                        if colord1==3 and colord2==8 and colormother ==3:
                            maxcol+=1
                            d2colup1=colup1
                            d2colup2=maxcol
                            d1colup1=maxcol
                            d1colup2=0

                        if colord2==3 and colord1==8 and colormother ==3:
                            maxcol+=1
                            d1colup1=colup1
                            d1colup2=maxcol
                            d2colup1=maxcol
                            d2colup2=0

                        if colord1==-3 and colord2==8 and colormother ==-3:
                            maxcol+=1
                            d2colup2=colup2
                            d2colup1=maxcol
                            d1colup2=maxcol
                            d1colup1=0

                        if colord2==-3 and colord1==8 and colormother ==-3:
                            maxcol+=1
                            d1colup2=colup2
                            d1colup1=maxcol
                            d2colup2=maxcol
                            d2colup1=0

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


    def get_topologies(self,matrix_element):
        """Extraction of the phase-space topologies from mg5 matrix elements 
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

#         use the AMP2 index to label the topologies
            tag_topo=mapconfigs[iconfig]
            topologies[tag_topo]=production_topo()

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

    @misc.mute_logger()
    def generate_fortran_me(self,processes,base_model,mode, mgcmd,path_me):
        """Given a process and a model, use the standanlone module of mg5
         to generate a fortran executable for the evaluation of the 
         corresponding matrix element
                mode=0 : production part 
                mode=1 : process fully decayed
        """


        commandline="import model "+base_model
        mgcmd.exec_cmd(commandline)

        mgcmd.exec_cmd("set group_subprocesses False")

        commandline="generate "+processes[0]
        mgcmd.exec_cmd(commandline)

        # output the result in Fortran format:
        if mode==0: # production process
            mgcmd.exec_cmd("output standalone_ms %s -f" % pjoin(path_me,'production_me') )
        
        elif mode==1: # full process
            mgcmd.exec_cmd("output standalone_ms %s -f" % pjoin(path_me,'full_me'))
          

        
              
        # now extract the information about the topologies
        if mode==0:
            me_list=mgcmd._curr_matrix_elements.get_matrix_elements()
            if(len(me_list)!=1): 
                logger.warning('WARNING: unexpected number of matrix elements')
            topo=self.get_topologies(me_list[0])
            return topo
        

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
         

    def get_full_process_structure(self,decay_processes,line_prod_proc, base_model, banner,check=0):
        """ return a string with the definition of the process fully decayed
                and also a list of dc_branch objects with all infomation about the topology 
                of each decay branch
        """

        decay_struct={}    
        full_proc_line=line_prod_proc+" , "
        for proc_index in decay_processes.keys():
            if ',' in decay_processes[proc_index]:
                current_branch=decay_processes[proc_index]
#                list_branch=current_branch.split(",")
#                current_nb_decays=len(list_branch)
#                for nu in range(current_nb_decays):
#                    if nu >0 and nu < current_nb_decays-1: list_branch[nu]=" ( "+list_branch[nu]
#                for nu in range(current_nb_decays-2):
#                    list_branch[current_nb_decays-1]=list_branch[current_nb_decays-1]+" ) "
#                current_branch=""
#                for nu in range(current_nb_decays-1):
#                    current_branch+=list_branch[nu]+ "    , "
#                current_branch+=list_branch[current_nb_decays-1]
                full_proc_line=full_proc_line+" ( "+current_branch+ " )    , "
            else:
                full_proc_line=full_proc_line+"    "+\
                    decay_processes[proc_index]+ "     , "
            decay_proc=decay_processes[proc_index]
            decay_proc=decay_proc.replace("\n","")
            decay_proc=decay_proc.replace("("," ")
            decay_proc=decay_proc.replace(")"," ")

            decay_struct[proc_index]=dc_branch(decay_proc,base_model,banner,check)
#            decay_struct[proc_index].print_branch()
#        decay_struct[proc_index].print_branch()
        full_proc_line=full_proc_line[:-3]
        #print full_proc_line

        return full_proc_line, decay_struct

    def compile_fortran_me_production(self,path_me):
        """ Compile the fortran executables associated with the evalutation of the 
                matrix elements (production process)
                Returns the path to the fortran executable
        """

        list_prod=os.listdir(pjoin(path_me,"production_me/SubProcesses"))
        counter=0
        logger.info("""Finalizing production me's """)

         
        for direc in list_prod:
            if direc[0]=="P":
                counter+=1
                prod_name=direc[string.find(direc,"_")+1:]
                
                old_path=pjoin(path_me,'production_me','SubProcesses',direc)
                new_path=pjoin(path_me,'production_me','SubProcesses',prod_name)
                if os.path.isdir(new_path): shutil.rmtree(new_path)
                os.rename(old_path, new_path)

                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_prod.f')
                shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f"))  
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )

                file=pjoin(path_me, 'param_card.dat')
                shutil.copyfile(file,pjoin(path_me,"production_me","Cards","param_card.dat"))                

		# files to produce the parameters:
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
                    
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
                shutil.copyfile(file_madspin, pjoin(path_me,"production_me","Source","MODEL","lha_read.f" )) 
                shutil.copyfile(pjoin(path_me,'production_me','Source','MODEL','input.inc'),pjoin(new_path,'input.inc')) 


                # COMPILATION
                # in case there are new DHELAS routines, we need to recompile 
                misc.compile(arg=['clean'], cwd=pjoin(path_me,"production_me","Source", "DHELAS"), mode='fortran')
                misc.compile( cwd=pjoin(path_me,"production_me","Source","DHELAS"), mode='fortran')

                misc.compile(arg=['clean'], cwd=pjoin(path_me,"production_me","Source", "MODEL"), mode='fortran')
                misc.compile( cwd=pjoin(path_me,"production_me","Source","MODEL"), mode='fortran')                   

                os.chdir(new_path)
                misc.compile(arg=['clean'], cwd=new_path, mode='fortran')
                misc.compile(arg=['init'],cwd=new_path,mode='fortran')
                misc.call('./init')

                shutil.copyfile('parameters.inc', '../parameters.inc')
                os.chdir(path_me)
                    

                misc.compile(cwd=new_path, mode='fortran')

                if(os.path.getsize(pjoin(path_me,'production_me','SubProcesses', 'parameters.inc'))<10):
		    raise Exception, "Parameters of the model were not written correctly ! " 
                return prod_name


    def compile_fortran_me_full(self,path_me):
        """ Compile the fortran executables associated with the evalutation of the 
                matrix elements (full process)
                Returns the path to the fortran executable
        """

 
        list_full=os.listdir(pjoin(path_me,"full_me","SubProcesses"))

        logger.info("""Finalizing decay chain me's """)
        for direc in list_full:
            if direc[0]=="P":
                
                decay_name=direc[string.find(direc,"_")+1:]
                
                old_path=pjoin(path_me,'full_me','SubProcesses',direc)
                new_path=pjoin(path_me, 'full_me','SubProcesses',decay_name)


                if os.path.isdir(new_path): shutil.rmtree(new_path)
                os.rename(old_path, new_path)               
                
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_full.f')
                shutil.copyfile(file_madspin, pjoin(new_path,"check_sa.f")  )


                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
                shutil.copyfile(file_madspin, pjoin(new_path,"makefile") )
                                
                shutil.copyfile(pjoin(path_me,'full_me','Source','MODEL','input.inc'),pjoin(new_path,'input.inc'))

                # write all the parameters:
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
                shutil.copyfile(file_madspin,pjoin(new_path,"initialize.f"))
                         
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
                shutil.copyfile(file_madspin, pjoin(path_me,"full_me","Source","MODEL","lha_read.f" ))  

                file=pjoin(path_me, 'param_card.dat')
                shutil.copyfile(file,pjoin(path_me,"full_me","Cards","param_card.dat")) 

                # BEGIN COMPILATION
                # in case there are new DHELAS routines, we need to recompile                
                misc.compile(arg=['clean'], cwd=pjoin(path_me,"full_me","Source","DHELAS"), mode='fortran')
                misc.compile( cwd=pjoin(path_me,"full_me","Source","DHELAS"), mode='fortran')

                misc.compile(arg=['clean'], cwd=pjoin(path_me,"full_me","Source","MODEL"), mode='fortran')
                misc.compile( cwd=pjoin(path_me,"full_me","Source","MODEL"), mode='fortran')   

                os.chdir(new_path)
                misc.compile(arg=['clean'], cwd=new_path, mode='fortran')
                misc.compile(arg=['init'],cwd=new_path,mode='fortran')
                misc.call('./init')
                shutil.copyfile('parameters.inc', '../parameters.inc')
                os.chdir(path_me)
                
                # now we can compile check
                misc.compile(arg=['check'], cwd=new_path, mode='fortran')
                # END COMPILATION

            
                if(os.path.getsize(pjoin(path_me,'full_me','SubProcesses', 'parameters.inc'))<10):
		    raise Exception, "Parameters of the model were not written correctly ! " 

                #decay_pattern=direc[string.find(direc,"_")+1:]
                #decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]
                #decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]

        os.chdir(path_me)
        return decay_name

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
        if len(topo["branchings"])>0:  # Exclude 2>1 topologies
            if topo["branchings"][-1]["type"]=="t":
                if topo["branchings"][-2]["type"]!="t":
                    logger.warning('last branching is t-channel')
                    logger.warning('but last-but-one branching is not t-channel')
                else:
                    part=topo["branchings"][-1]["index_d2"]
                    if part >0: # reset the mass only if "part" is an external particle
                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])

    def reorder_branch(self,branch):
        """ branch is a sting with the definition of a decay chain
                If branch contains " A > B C , B > ... " 
                reorder into             " A > C B , B > ... "

        """
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
        if len(topo["branchings"])>0:  # Exclude 2>1 topologies
            if topo["branchings"][-1]["type"]=="t":
                if topo["branchings"][-2]["type"]!="t":
                    logger.info('last branching is t-channel')
                    logger.info('but last-but-one branching is not t-channel')
                else:
                    part=topo["branchings"][-1]["index_d2"] 
                    if part >0: # reset the mass only if "part" is an external particle
                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])


    def transpole(self,pole,width):

        """ routine for the generation of a p^2 according to 
            a Breit Wigner distribution
            the generation window is 
            [ M_pole^2 - 30*M_pole*Gamma , M_pole^2 + 30*M_pole*Gamma ] 
        """

        zmin = math.atan(-30.0)/width
        zmax = math.atan(30.0)/width

        z=zmin+(zmax-zmin)*random.random()
        y = pole+width*math.tan(width*z)

        jac=(width/math.cos(width*z))**2*(zmax-zmin)
        return y, jac


    def generate_BW_masses (self,topo, to_decay, pid2name, pid2width, pid2mass,s):
        """Generate the BW masses of the particles to be decayed in the production event    """    

        weight=1.0
        for part in topo["get_id"].keys():
            pid=topo["get_id"][part]
            if pid2name[pid] in to_decay:
                mass=0.25
                width=pid2width[pid]*pid2mass[pid]/(2.0*pid2mass[pid])**2
                virtualmass2, jac=self.transpole(mass,width)
                virtualmass2=virtualmass2*(2.0*pid2mass[pid])**2
                weight=weight*jac
                #print "need to generate BW mass of "+str(part)
                ##print "mass: "+str(pid2mass[pid])
                #print "width: "+str(pid2width[pid])
                #print "virtual mass: "+str(math.sqrt(virtualmass2))
                #print "jac: "+str(jac)
                old_mass=topo["get_mass2"][part]
                topo["get_mass2"][part]=virtualmass2
                # sanity check
                if pid2mass[pid]<1e-6:
                    logger.debug('A decaying particle has a mass of less than 1e-6 GeV')
# for debugg purposes:
                if abs((pid2mass[pid]-math.sqrt(topo["get_mass2"][part]))/pid2mass[pid])>1.0 :
                    logger.debug('Mass after BW smearing affected by more than 100 % (1)') 
                    logger.debug('Pole mass: '+str(pid2mass[pid]))
                    logger.debug('Virtual mass: '+str(math.sqrt(topo["get_mass2"][part])))
                                       
                #print topo["get_mass2"]         

#    need to check if last branch is a t-branching. If it is, 
#    we need to update the value of branch["m2"]

        if len(topo["branchings"])>0:  # Exclude 2>1 topologies
            if topo["branchings"][-1]["type"]=="t":
                if topo["branchings"][-2]["type"]!="t":
                    logger.debug('last branching is t-channel')
                    logger.debug('but last-but-one branching is not t-channel')
                else:
                    part=topo["branchings"][-1]["index_d2"] 
                    if part >0: # reset the mass only if "part" refers to an external particle 
                        old_mass=topo["branchings"][-2]["m2"]
                        topo["branchings"][-2]["m2"]=math.sqrt(topo["get_mass2"][part])
                        #sanity check

                        if abs(old_mass-topo["branchings"][-2]["m2"])>1e-10:
                            if abs((old_mass-topo["branchings"][-2]["m2"])/old_mass)>1.0 :
                                logger.debug('Mass after BW smearing affected by more than 100 % (2)')
                                logger.debug('Previous value: '+ str(old_mass))
                                logger.debug('New mass: '+ str((topo["branchings"][-2]["m2"])))
                                try:
                                    pid=topo["get_id"][part]
                                    logger.debug('pole mass: %s' % pid2mass[pid])
                                except Exception:
                                    pass
        return weight

    def modify_param_card(self,pid2widths,path_me):
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

    def find_resonances(self,proc, branches):
        """ restore the resonances in a production process 
            the expected decay chains (variable branches)
            were extracted from the banner
        """

        pos1=proc.find(">")
        list_finalstate=proc[pos1+1:].split()

        for res in branches.keys():
            resFS=[ item for item in branches[res]["finalstate"]]
            for index in range(len(list_finalstate)-1,-1,-1):
                if list_finalstate[index] in resFS: 
                    pos2 = resFS.index(list_finalstate[index])
                    del resFS[pos2]
                    del list_finalstate[index]
            if resFS!=[]:
                logger.warning('CANNOT RECOGNIZE THE EXPECTED DECAY \
                CHAIN STRUCTURE IN PRODUCTION EVENT')
                return proc
            else:
                list_finalstate.append(res)

        initstate=proc[:pos1]
        finalstate=""
        for part in list_finalstate:
            finalstate+=" "+part
        for res in branches.keys():
            finalstate+=" , "+branches[res]["branch"]
        newproc=initstate+" > "+finalstate+" "
        return newproc



    def get_final_state_compact(self,final_state_full):

#    prod_resonances={}
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
                branches[list_part[0]]["branch"], dummy=decay_tools.reorder_branch(branch)
        else: 
            final_state_compact=final_state_full
            branches={}

        return final_state_compact, branches

    def get_banner(self):
        pass


    def set_cumul_proba_for_tag_decay(self,multi_decay_processes,decay_tags):
        """
        """

        sum_br=0.0
        for index, tag_decay in enumerate(decay_tags):
           sum_br+=multi_decay_processes[tag_decay]['br']

        cumul=0.0
        for index, tag_decay in enumerate(decay_tags):
           cumul+=multi_decay_processes[tag_decay]['br']/sum_br
           multi_decay_processes[tag_decay]['cumul_br']=cumul
        return sum_br




    def generate_tag_decay(self,multi_decay_processes,decay_tags ):
        """ generate randomly the tag for the decay config. using a probability law 
            based on branching fractions
        """

        r=random.random()

        for index, tag_decay in enumerate(decay_tags):
            if r < multi_decay_processes[tag_decay]['cumul_br']:
               return tag_decay
        if r==1.0:
            return decay_tags[-1]



    def update_tag_decays(self, decay_tags, branch_tags):
        """ update the set of tags to identify the final state content of a decay channel """

        new_decay_tags=[]
        for item1 in branch_tags:
            if len(decay_tags)==0:
               new_decay_tags.append( (item1,) )
            else:
               for item2 in decay_tags:
                   new_decay_tags.append(item2+(item1,))
        return new_decay_tags
                  
    
    def format_proc_line(self,procline):
        
# remove the tag "[*]": this tag is used in aMC@LNO , 
# but it is not a valid syntax for mg5
        line=procline
        pos1=line.find("[")
        pos2=line.find("]")
        if pos1>0 and pos2 >pos1:
            line=line[:pos1]+line[pos2+1:]
#
# Extract the options:
#
# A. Remove process number (identified by "@")
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(line)
        if proc_number_re:
            line = proc_number_re.group(1) + \
                         proc_number_re.group(3)

# B. search for the beginning of the option string
        pos=1000
# start with order
        order_pattern = re.compile("^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(line)
        if (order_re):
            pos_order=line.find(order_re.group(2))
            if pos_order>0 and pos_order < pos : pos=pos_order

# then look for slash or dollar
        slash = line.find("/")
        if slash > 0 and slash < pos: pos=slash
        dollar = line.find("$")
        if dollar > 0 and dollar < pos: pos=dollar

        if pos<1000:
            proc_option=line[pos:]
            line=line[:pos]
        else:
            proc_option=""

        return line, proc_option

class decay_all_events:
    
    @misc.mute_logger()
    def __init__(self,inputfile,mybanner,to_decay,decay_processes,\
                 prod_branches,proc_option, max_weight_arg, BW_effects,\
                path_me):
        
        self.calculator = {}
        self.calculator_nbcall = {}
        self.path_me = path_me


        # need to unbuffer all I/O in fortran, otherwise
	# the values of matrix elements are not passed to the Python script
        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'  
#        os.system(" export GFORTRAN_UNBUFFERED_ALL ")


        mgcmd=Cmd.MasterCmd()

        curr_dir=os.getcwd()
        
# Remove old stuff from previous runs
# so that the current run is not confused
        if os.path.isfile(pjoin(path_me,"param_card.dat")):
            os.remove(pjoin(path_me,"param_card.dat"))

        if os.path.isdir(pjoin(path_me,"production_me")):
            shutil.rmtree(pjoin(path_me,"production_me"))

        if os.path.isdir(pjoin(path_me,"full_me")):
            shutil.rmtree(pjoin(path_me,"full_me"))
        decay_tools=decay_misc()

        logger.info('model:     '+ mybanner.proc["model"])
        model_path=mybanner.proc["model"]
        base_model = import_ufo.import_model(model_path)

        # Import model
        base_model = import_ufo.import_model(model_path,
                                        decay=True)
        if not hasattr(base_model.get('particles')[0], 'partial_widths'):
            logger.warning('The UFO model does not include widths information. Impossible to compute widths automatically')

        base_model.pass_particles_name_in_mg_default() 
 	mgcmd.exec_cmd('import model '+mybanner.proc["model"])
        
#        for part in decay_processes:
#            list_symbol=decay_processes[part].split()
#            for item in list_symbol:
#                if item =="," or item==")" or item=="(" or item==">": 
#                    continue
#                elif item not in [ particle['name'] for particle in base_model['particles'] ] \
#                    and item not in [ particle['antiname'] for particle in base_model['particles'] ]:
#                    raise Exception, "No particle "+item+ " in the model "+mybanner.proc["model"]


# we will need a dictionary pid > label
        pid2label_dict=pid2label(base_model)
        label2pid_dict=label2pid(base_model)
# we will also need a dictionary pid > color_rep
        pid2color_dict=pid2color(base_model)


# now overwrite the param_card.dat in Cards:
        init=mybanner.whole_banner.find("<slha>")
        fin=mybanner.whole_banner.find("</slha>")

        if init<0 or fin <0:
            logger.warning('EXTRACTION OF THE PARAM_CARD FROM THE BANNER FAILED!!')

        param_card=mybanner.whole_banner[init+7:fin]

# now we can write the param_card.dat:
# Note that the width of each resonance in the    
# decay chain should be >0 , we will check that later on
        param=open(pjoin(path_me,'param_card.dat'),"w")
        param.write(param_card)
        param.close()

# extract all resonances in the decay:
	resonances=decay_tools.get_resonances(decay_processes.values())
	logger.info('List of resonances:')
	logger.info(resonances)

        label2width={}
        label2mass={}
        pid2width={}
        pid2mass={}
# now extract the width of the resonances:
        for particle_label in resonances:
            try:
                part=label2pid_dict[particle_label]
                label2width[particle_label]=float(mybanner.param["DECAY"][abs(part)])
                label2mass[particle_label]=float(mybanner.param["Block mass"][abs(part)])
                pid2width[part]=label2width[particle_label]
                pid2mass[part]=label2mass[particle_label]
                if label2width[particle_label]==0.0:
                    for param in base_model["parameters"][('external',)]:
                        if param.lhablock=="DECAY" and param.lhacode==[abs(part)]:
                            label2width[particle_label]=param.value
                            pid2width[part]=label2width[particle_label]
                    logger.warning('ATTENTION')
                    logger.warning('Found a zero width in the param_card for particle '\
                                   +str(particle_label))
                    logger.warning('Use instead the default value '\
                                   +str(label2width[particle_label]))

            except: 
                continue
# now we need to modify the values of the width
# in param_card.dat, since this is where the input 
# parameters will be read when evaluating matrix elements
        decay_tools.modify_param_card(pid2width,path_me)


# now we need to evaluate the branching fractions:
# =================================================
	logger.info('We need information on the partial widhts ')
	logger.info('First look inside the banner of the event file ')
	logger.info('and check whether this information is available ')

        calculate_br=width_estimate(resonances,path_me,pid2label_dict,mybanner.proc["model"])#
#       Maybe the branching fractions are already given in the banner:
	filename=pjoin(path_me,'param_card.dat')
        calculate_br.extract_br_from_card(filename)
        calculate_br.extract_br_for_antiparticle(label2pid_dict,pid2label_dict)
	calculate_br.print_branching_fractions()
#
#        now we check that we have all needed pieces of info regarding the branching fraction:
        multiparticles=mgcmd._multiparticles
        branching_per_channel=calculate_br.get_BR_for_each_decay(decay_processes,multiparticles,\
						mybanner.proc["model"],base_model,pid2label_dict)       

        # check that we get branching fractions for all resonances to be decayed:
        
	if(branching_per_channel==0):
	    #calculate_br.generate_code_for_width_evaluation(mgcmd)#NOT USED ANYMORE, USE THE FR IMPLEMENTATION INSTEAD
	    logger.info('We need to recalculate the branching franctions')
	    logger.info('using the compute_width module of madevent')
	    calculate_br.launch_width_evaluation(resonances,label2pid_dict,pid2label_dict,base_model) # use FR to get all partial widths
            calculate_br.extract_br_from_width()                                       # set the br to partial_width/total_width
            calculate_br.extract_br_for_antiparticle(label2pid_dict,pid2label_dict)    # set the partial widths of antiparticles equal to the one of the CC channel  
    	    calculate_br.print_branching_fractions()
            branching_per_channel=calculate_br.get_BR_for_each_decay(decay_processes,multiparticles,\
						mybanner.proc["model"],base_model,pid2label_dict)       

        if branching_per_channel==0:
	    raise Exception, 'Failed to extract the branching franction associated with each decay channel'


# now we need to sort all the different decay configurations, and get the br for each of them
# ===========================================================================================

#       1. first create a list of branches that ordered according to the mg numeratation:
        list_particle_to_decay=to_decay.keys()
        list_particle_to_decay.sort()

#       2. then use a tuple to identify a decay channel
#
#       (tag1 , tag2 , tag3, ...)  
#
#       where the number of entries is the number of branches, 
#       tag1 is the tag that identifies the final state of branch 1, 
#       tag2 is the tag that identifies the final state of branch 2, 
#       etc ...

#       2.a. first get decay_tags = the list of all the tuples

        decay_tags=[]
        for part in list_particle_to_decay:  # loop over particle to decay in the production process
            branch_tags=[ fs for fs in branching_per_channel[part]] # list of tags in a given branch
            decay_tags=decay_tools.update_tag_decays(decay_tags, branch_tags)

#       2.b. then build the dictionary multi_decay_processes = multi dico
#       first key = a tag in decay_tags
#       second key :  ['br'] = float with the branching fraction for the FS associated with 'tag'   
#                     ['config'] = a list of strings, each of then giving the definition of a branch    

        multi_decay_processes={}
        for tag in decay_tags:
#           compute br + get the congis
            br=1.0
            list_branches={}
            for index, part in enumerate(list_particle_to_decay):
                br=br*branching_per_channel[part][tag[index]]['br']
                list_branches[part]=branching_per_channel[part][tag[index]]['config']
            multi_decay_processes[tag]={}
            multi_decay_processes[tag]['br']=br
            multi_decay_processes[tag]['config']=list_branches

#       compute the cumulative probabilities associated with the branching fractions
        sum_br=decay_tools.set_cumul_proba_for_tag_decay(multi_decay_processes,decay_tags)



#      Now we have a dictionary (multi_decay_processes) of which values 
#      identifies all the decay channels to be considered, and the associated branching fractions.

#      The only difference with the one-channel implementation resides in the fact that 
#      decay_processes is replaced by multi_decay_processes

#Next step: we need to determine which matrix elements are really necessary
#==========================================================================







# Estimation of the maximum weight
#=================================

#    consider the possibility of several production process
        set_of_processes=[]
#    me_full_mg5format=[]
#    me_prod_mg5format=[]
        decay_struct={}
        decay_path={}     # dictionary to record the name of the directory with decay fortran me
        production_path={} # dictionary to record the name of the directory with production fortran me

#    also for production matrix elements, 
#    we need to keep track of the topologies
        topologies={}

 
        os.system("date")
# Now we are ready to start the evaluation of the maximum weight 
        if max_weight_arg>0:
            max_weight={}
            for tag_decay in decay_tags: max_weight[tag_decay]=max_weight_arg
        else:
            logger.info('  ')
            logger.info('   Estimating the maximum weight    ')
            logger.info('   *****************************    ')
            logger.info('  => find the maximum value of |M_full|^2/|M_prod|^2')
            logger.info('     for the 5 first events  ')
            logger.info('  ')

            curr_event=Event(inputfile)
            probe_weight=[]

            starttime = time.time()
            for ev in range(5):
                probe_weight.append({})
                for tag_decay in decay_tags:
                    probe_weight[ev][tag_decay]=0.0
                curr_event.get_next_event()

#    check if we have a new production process, in which case 
#    we need to generate the corresponding matrix elements
                prod_process=curr_event.give_procdef(pid2label_dict)
                extended_prod_process=decay_tools.find_resonances(prod_process, prod_branches)

                tag_production=prod_process.replace(">", "_")
                tag_production=tag_production.replace(" ","")
                tag_production=tag_production.replace("~","x")
                if tag_production not in set_of_processes: # we need to generate new matrix elements
                    logger.info('  ')
                    logger.info('Found a new production process: ')
                    logger.info(prod_process)
                    logger.info('Re-interpreted as    ')
                    logger.info(extended_prod_process+proc_option)
                    logger.info( tag_production)
                    logger.info( ' -> need to generate the corresponding fortran matrix element ... ')
                    set_of_processes.append(tag_production)

                    # generate fortran me for production only
                    topologies[tag_production]=\
                        decay_tools.generate_fortran_me([extended_prod_process+proc_option],\
                            mybanner.proc["model"], 0,mgcmd,path_me)

                    prod_name=decay_tools.compile_fortran_me_production(path_me)
                    production_path[tag_production]=prod_name

#                   for the decay, we need to keep track of all possibilities for the decay final state:
                    decay_struct[tag_production]={}
                    decay_path[tag_production]={}
                    for tag_decay in decay_tags:
                        new_full_proc_line, new_decay_struct=\
                            decay_tools.get_full_process_structure(multi_decay_processes[tag_decay]['config'],\
                            extended_prod_process, base_model, mybanner)
                        decay_struct[tag_production][tag_decay]=new_decay_struct
#
                        decay_tools.generate_fortran_me([new_full_proc_line+proc_option],\
                                                    mybanner.proc["model"], 1,mgcmd,path_me)

                        decay_name=decay_tools.compile_fortran_me_full(path_me)
                        decay_path[tag_production][tag_decay]=decay_name
                    
                    logger.info('Done.')

#    Now the relevant matrix elements for the current event are there.
#    But we still need to select a production topology 
#    So first we evaluate the production me's only, 
#    and select one production topolgy randomly, 
#    with each topology    weighted by the corresponding diagram squared. 


                p, p_str=curr_event.give_momenta()   
#            Note here that no momentum reshuffling is done, 
#            since we don't know yet which topology should be used.    
#            so light quarks and gluons in the final state may have a small mass

                
                prod_values = self.calculate_matrix_element('prod', 
                                         production_path[tag_production], p_str)
                prod_values=prod_values.replace("\n", "")
                prod_values=prod_values.split()
                mg5_me_prod = float(prod_values[0])

                tag_topo, cumul_proba = decay_tools.select_one_topo(prod_values)

                logger.info(' ')
                running_time = misc.format_timer(time.time()-starttime)
                logger.info('Event %s %s: ' % (ev+1, running_time))
#     print "Shat: "+str(curr_event.shat)
                logger.info('Number of production topologies : '\
                            +str(len(topologies[tag_production].keys())))
#         print "Cumulative probabilities                : "+str(cumul_proba)
                logger.info('Selected topology               : '\
                            +str(tag_topo))
#         topologies[tag_production][tag_topo].print_topo()
 
#         print "Event before reshuffling:"
#         print curr_event.string_event_compact()

#     We have our topology now. So we can 
#        1. dress the topology with the momenta 
#             in the production event, 
#        2. set the masses    the light partons to zero,
#        3. generate the BW masses, 
#        4. reshuffle the momenta in the production event
#        5. pass the info to curr_event

                topologies[tag_production][tag_topo].dress_topo_from_event(curr_event,to_decay)
                topologies[tag_production][tag_topo].extract_angles()
 
                if BW_effects:
                    decay_tools.set_light_parton_massless(topologies[tag_production][tag_topo])

                for dec in range(5000):

#        try to reshuffle the momenta 
                    if  BW_effects:
                        try_reshuffle=0
                        while 1:
                            try_reshuffle+=1
                            BW_weight_prod=decay_tools.generate_BW_masses(\
                                            topologies[tag_production][tag_topo], \
                                            to_decay.values(),pid2label_dict, pid2width,pid2mass, \
                                            curr_event.shat)
                            
                            succeed=topologies[tag_production][tag_topo].reshuffle_momenta()
                            # sanlity check
                            for part in topologies[tag_production][tag_topo]['get_momentum'].keys():
                                if part in to_decay and \
                                topologies[tag_production][tag_topo]['get_momentum'][part].m<1.0:
                                    logger.debug('Mass of a particle to decay is less than 1 GeV')
                                    logger.debug('in reshuffling loop')
                            # end sanity check
                            if succeed: break
                            if try_reshuffle==10:
                                logger.warning( 'tried 10x to reshuffle the momenta, failed')
                                logger.warning( ' So let us try with another topology')
                                tag_topo, cumul_proba=decay_tools.select_one_topo(prod_values)
                                topologies[tag_production][tag_topo].dress_topo_from_event(\
                                                                                    curr_event,to_decay)
                                topologies[tag_production][tag_topo].extract_angles()
                                decay_tools.set_light_parton_massless(topologies\
                                                            [tag_production][tag_topo])
                                try_reshuffle=0
                    else: 
                        BW_weight_prod=1.0

                    topologies[tag_production][tag_topo].topo2event(curr_event,to_decay)

#             if dec==0:
#                 print "Event after reshuffling:"
#                 print curr_event.string_event_compact()

#                   Here the production event has been reshuffled, 
#                   now we need to decay it.
#                   There might be several decay channels -> loop over them

                    for tag_decay in decay_tags:

                        decayed_event, BW_weight_decay=decay_tools.decay_one_event(\
                                            curr_event,decay_struct[tag_production][tag_decay], \
                                            pid2color_dict, to_decay, pid2width, \
                                            pid2mass, resonances,BW_effects)

                        if decayed_event==0:
                            logger.warning('failed to decay event properly')
                            continue
#     set the momenta for the production event and the decayed event:
                        p, p_str=curr_event.give_momenta()    
                        p_full, p_full_str=decayed_event.give_momenta()    

#     start with production weight:
                        prod_values =self.calculate_matrix_element('prod',
                                         production_path[tag_production], p_str)

                        prod_values=prod_values.replace("\n", "")
                        prod_values=prod_values.split()
                        mg5_me_prod = float(prod_values[0])
#     then decayed weight:
                        full_values =self.calculate_matrix_element('full',
                                         decay_path[tag_production][tag_decay], p_full_str)

                        mg5_me_full = float(full_values)
                    #mg5_me_full = float(external.communicate(input=p_full_str)[0])
                        mg5_me_full=mg5_me_full*BW_weight_prod*BW_weight_decay
                        os.chdir(curr_dir)

                        if(not mg5_me_full>0 or not mg5_me_prod >0 ):
                            logger.warning('WARNING: NEGATIVE MATRIX ELEMENT !!')

                        weight=mg5_me_full/mg5_me_prod
                        if (weight>probe_weight[ev][tag_decay]): probe_weight[ev][tag_decay]=weight

                for  index,tag_decay in enumerate(decay_tags):
                  logger.info('Max weight,  event '+str(ev+1)+\
                            ' , dk config '+str(index+1)+' : '+str(probe_weight[ev][tag_decay]))

            max_weight={}
            min_weight={}

            for index,tag_decay in enumerate(decay_tags):
                max_weight[tag_decay]=0.0
                min_weight[tag_decay]=1e10
                for ev in range(5):
                    if(max_weight[tag_decay]<probe_weight[ev][tag_decay]): max_weight[tag_decay]=probe_weight[ev][tag_decay]
                    if(min_weight[tag_decay]>probe_weight[ev][tag_decay]): min_weight[tag_decay]=probe_weight[ev][tag_decay]
                logger.info(' ')
                logger.info(' Decay channel '+str(index+1))
                for part in multi_decay_processes[tag_decay]['config'].keys():
                     logger.info(multi_decay_processes[tag_decay]['config'][part])
                logger.info('     maximum weight that we got is '+str(max_weight[tag_decay]))
                logger.info('     with a fluctuation of '+str(max_weight[tag_decay]-min_weight[tag_decay]))
                logger.info('     -> add 2x this fluctuation to the max. weight ')
                max_weight[tag_decay]=3.0*max_weight[tag_decay]-2.0*min_weight[tag_decay]
            del curr_event



#
        os.system("date")
        logger.info(' ' )
        logger.info('Decaying the events... ')
        inputfile.seek(0)
        outputfile = open(pjoin(path_me,'decayed_events.lhe'), 'w')
        curr_event=Event(inputfile)
        old_banner=mybanner.whole_banner
        pos=old_banner.find("</header>")
        if (pos>0):
            new_banner=old_banner[:pos]
            new_banner+="<DECAY>\n"
            for index,tag_decay in enumerate(decay_tags):
                new_banner+="# Decay channel "+str(index+1)+"\n"
                for part in multi_decay_processes[tag_decay]['config'].keys():
                    new_banner+="# "+multi_decay_processes[tag_decay]['config'][part]+"\n"
                new_banner+="# branching fraction: "+str(multi_decay_processes[tag_decay]['br']) + "\n"
                new_banner+="# estimate of the maximum weight: "+str(max_weight[tag_decay]) + "\n"
            new_banner+="</DECAY>\n"
            new_banner+=old_banner[pos:]
            outputfile.write(new_banner)


        event_nb=0
        trial_nb_all_events=0
        starttime = time.time()
        while 1:
            if (curr_event.get_next_event()=="no_event"): break
            event_nb+=1
            if (event_nb % 100==0): 
                running_time = misc.format_timer(time.time()-starttime)
                logger.info('Event nb %s %s' % (event_nb, running_time))
            trial_nb=0

#        if event_nb>10: break
#    check if we have a new production process, in which case we need to generate the correspomding matrix elements
            prod_process=curr_event.give_procdef(pid2label_dict)
            extended_prod_process=decay_tools.find_resonances(prod_process, prod_branches)

            tag_production=prod_process.replace(">", "_")
            tag_production=tag_production.replace(" ","")
            tag_production=tag_production.replace("~","x")
            if tag_production not in set_of_processes: # we need to generate new matrix elements
                logger.info(' ')
                logger.info('Found a new production process: ')
                logger.info(prod_process)
                logger.info('Re-interpreted as    ')
                logger.info(extended_prod_process+proc_option)
                logger.info( tag_production)
                logger.info( ' -> need to generate the corresponding fortran matrix element ... ')
                set_of_processes.append(tag_production)

                # generate fortran me for production only
                topologies[tag_production]=\
                    decay_tools.generate_fortran_me([extended_prod_process+proc_option],\
                        mybanner.proc["model"], 0,mgcmd,path_me)

                prod_name=decay_tools.compile_fortran_me_production(path_me)
                production_path[tag_production]=prod_name

#               for the decay, we need to keep track of all possibilities for the decay final state:
                decay_struct[tag_production]={}
                decay_path[tag_production]={}
                for tag_decay in decay_tags:
                    new_full_proc_line, new_decay_struct=\
                        decay_tools.get_full_process_structure(multi_decay_processes[tag_decay]['config'],\
                        extended_prod_process, base_model, mybanner)
                    decay_struct[tag_production][tag_decay]=new_decay_struct
#
                    decay_tools.generate_fortran_me([new_full_proc_line+proc_option],\
                                                mybanner.proc["model"], 1,mgcmd,path_me)

                    decay_name=decay_tools.compile_fortran_me_full(path_me)
                    decay_path[tag_production][tag_decay]=decay_name
                    
                logger.info('Done.')


# First evaluate production matrix element
            p, p_str=curr_event.give_momenta()
            prod_values = self.calculate_matrix_element('prod', 
                                         production_path[tag_production], p_str)
            prod_values=prod_values.replace("\n", "")
            prod_values=prod_values.split()
            mg5_me_prod = float(prod_values[0])

#     select topology based on sigle-diagram weights
            tag_topo, cumul_proba=decay_tools.select_one_topo(prod_values)

#    dress the topology with momenta and extract the canonical numbers 
            topologies[tag_production][tag_topo].dress_topo_from_event(curr_event,to_decay)
            topologies[tag_production][tag_topo].extract_angles()

            if BW_effects:
                decay_tools.set_light_parton_massless(topologies[tag_production][tag_topo])

            while 1:
                trial_nb+=1

#         try to reshuffle the event:
                if BW_effects:
                    try_reshuffle=0
                    while 1:
                        try_reshuffle+=1
                        BW_weight_prod=decay_tools.generate_BW_masses(topologies[tag_production][tag_topo], \
                                                        to_decay.values(),pid2label_dict, pid2width,pid2mass, \
                                                        curr_event.shat)
                        succeed=topologies[tag_production][tag_topo].reshuffle_momenta()
                        if succeed: break
                        if try_reshuffle==10:
                            logger.info('WARNING: tried 10x to reshuffle the momenta, failed')
                            logger.info(' So let us try with another topology')
#                        print "Event: "+str(event_nb)
#                        topologies[tag_production][tag_topo].print_topo()
                            tag_topo, cumul_proba=decay_tools.select_one_topo(prod_values)
                            topologies[tag_production][tag_topo].dress_topo_from_event(curr_event,to_decay)
                            topologies[tag_production][tag_topo].extract_angles()
                            decay_tools.set_light_parton_massless(topologies[tag_production][tag_topo])
                            try_reshuffle=0

                else:
                    BW_weight_prod=1.0

#               Here we need to select a decay configuration on a random basis:
                tag_decay=decay_tools.generate_tag_decay(multi_decay_processes,decay_tags)

                topologies[tag_production][tag_topo].topo2event(curr_event,to_decay)
                decayed_event, BW_weight_decay=decay_tools.decay_one_event(curr_event,decay_struct[tag_production][tag_decay], \
                                            pid2color_dict, to_decay, pid2width, pid2mass, resonances,BW_effects)

                if decayed_event==0: 
                    logger.info('failed to decay one event properly')
                    continue # means we had mA<mB+mC in one splitting A->B+C
                p, p_str=curr_event.give_momenta()    
                p_full, p_full_str=decayed_event.give_momenta()    
 
#        Now evaluate the matrix elements ...
#            start with production weight: 
                prod_values = self.calculate_matrix_element('prod', 
                                         production_path[tag_production], p_str)
                prod_values=prod_values.replace("\n", "")
                prod_values=prod_values.split()
                mg5_me_prod = float(prod_values[0])
#            then decayed weight:
                full_value = self.calculate_matrix_element('full', 
                                         decay_path[tag_production][tag_decay], p_full_str)
                mg5_me_full = float(full_value)
                mg5_me_full=mg5_me_full*BW_weight_prod*BW_weight_decay

                if(not mg5_me_full>0 or not mg5_me_prod >0 ):
                    logger.warning('NEGATIVE MATRIX ELEMENT !!')
                
#            mg5_me_prod, amp2_prod = evaluator.evaluate_matrix_element(me_prod[tag_production],p)
#            mg5_me_full, amp_full = evaluator.evaluate_matrix_element(me_full[tag_production],p_full)

                weight=mg5_me_full/mg5_me_prod
                if weight>max_weight[tag_decay]: 
                    logger.info('warning: got a larger weight than max_weight estimate')
                    logger.info('the ratio with the max_weight estimate is '+str(weight/max_weight[tag_decay]))
                if (weight/max_weight[tag_decay]> random.random()):

#             Here we need to restore the masses of the light partons 
#             initially found in the lhe production event
                    decay_tools.restore_light_parton_masses(topologies[tag_production][tag_topo],curr_event)
                    succeed=topologies[tag_production][tag_topo].reshuffle_momenta()
                    if not succeed:
                        logger.info('Warning: unable to restore masses of light partons')
                        break
                    topologies[tag_production][tag_topo].topo2event(curr_event,to_decay)
                    curr_event.reset_resonances() # re-evaluate the momentum of each resonance in prod. event
                    decayed_event, BW_weight_decay=decay_tools.decay_one_event(curr_event,decay_struct[tag_production][tag_decay], \
                                            pid2color_dict, to_decay, pid2width, pid2mass, resonances,BW_effects,ran=0)
      #              print decayed_event.string_event_compact()
      #              print "    "
      #              print p_full_str
      #              print "    "
                    decayed_event.wgt=decayed_event.wgt*sum_br
                    outputfile.write(decayed_event.string_event())
#                print "number of trials: "+str(trial_nb)
                    trial_nb_all_events+=trial_nb
                    break
 
        os.system("date")
        outputfile.write('</LesHouchesEvents>\n')
        inputfile.close()
        outputfile.close()

        logger.info('Total number of events: '+str(event_nb))
        logger.info('Average number of trial points per production event: '\
            +str(float(trial_nb_all_events)/float(event_nb)))
        logger.info('Number of subprocesses '+str(len(decay_path)))
	self.terminate_fortran_executables()
        shutil.rmtree(pjoin(path_me,'production_me'))
        shutil.rmtree(pjoin(path_me,'full_me'))

	# set the environment variable GFORTRAN_UNBUFFERED_ALL 
	# to its original value
        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'



    def terminate_fortran_executables(self):
	"""routine to terminate all fortran executables"""

        for (mode, production) in self.calculator:
            external = self.calculator[(mode, production)]
            external.terminate()

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
            else:
                tmpdir = pjoin(self.path_me,'full_me', 'SubProcesses',
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
        
        
        return prod_values
        
    



if __name__=="__main__":

# This is the main program

    import sys
    curr_dir=os.getcwd() 
    root_path = os.path.dirname(os.path.realpath( __file__ ))
    sys.path.insert(0, root_path)

    import logging.config
    import coloring_logging

    logging.config.fileConfig(os.path.join(curr_dir,  'me5_logging.conf'))
    logging.root.setLevel(eval('logging.' + 'DEBUG'))
    logging.getLogger('madgraph').setLevel(eval('logging.' + 'DEBUG'))

#    load the tools
    decay_tools=decay_misc()
#
#    Ask the user where is the event file
#
    filename=raw_input("Enter the name of the input lhe file \n")
    inputfile = open(filename, 'r')
#
# Read the banner from the input event file
#
    print "    "
    print "Extracting the banner ..."
    mybanner=Banner(inputfile)
    mybanner.ReadBannerFromFile()
   
    mybanner.proc["generate"], proc_option=decay_tools.format_proc_line(\
                                        mybanner.proc["generate"])
    print "process: "+mybanner.proc["generate"]
    print "options: "+proc_option
    print "model: "+mybanner.proc["model"]


# Read the final state of the production process:
#     "_full" means with the complete decay chain syntax 
#     "_compact" means without the decay chain syntax 
    final_state_full=mybanner.proc["generate"][mybanner.proc["generate"].find(">")+1:]
    final_state_compact, prod_branches=decay_tools.get_final_state_compact(final_state_full)

    print "branches: "
    print prod_branches
    print "final state: "+ final_state_compact
    decay_processes={}

# Ask the user which particle should be decayed
    particle_index=2
    to_decay={}
    for particle in final_state_compact.split():
        particle_index+=1
        do_decay=raw_input("decay the "+str(particle)+" ? (yes/no)\n")
        if do_decay=="yes":
            to_decay[particle_index]=particle
            decay_processes[particle_index], dummy=\
            decay_tools.reorder_branch(raw_input("enter the decay process  \n"))
            #print decay_processes[particle_index]

    print "particles to decay:    "
    print to_decay.keys()

    print "By any chance, do you know the max. weight already ? \n"
    answer=raw_input("Type the max weight if yes, otherwise type a negative number \n")
    try:
        answer=answer.replace("\n","") 
        if float(answer)>0:
            max_weight=float(answer)
        else:
            max_weight=-1
            print "ok, let us evaluate the max weight then ..."
    except: 
        print "ok, let us evaluate the max weight then ..."
        max_weight=-1

    answer=raw_input( "Include Breit Wigner effects ? (yes/no) \n")
    if (answer=="yes"): 
        BW_effects=1
    else:
        BW_effects=0


    generate_all=decay_all_events(inputfile,mybanner,to_decay,decay_processes,\
                prod_branches,proc_option, max_weight, BW_effects, curr_dir)

