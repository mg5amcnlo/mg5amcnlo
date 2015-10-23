#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##       email:  olivier.mattelaer@uclouvain.be                         ##
##   author: Artoisenet Pierre (CP3)                                    ##
##       email:  pierre.artoisenet@uclouvain.be                         ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:12/01/09                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##      +verif_event                                                    ##
##      +Lhco_filter                                                    ##
##      |    + init                                                     ##
##      |    + load_particule_number                                    ##
##      |    + extract_file_info                                        ##
##      |    |   +    define_particule_number                           ##
##      |    + init_hlt_cut                                             ##
##      |    + update_hlt_cut                                           ##
##      |    + verif_event                                              ##
##      |    |   +    check_data                                        ##
##                                                                      ##
##########################################################################
#Extension
import os
import popen2
import re
import sys
import time
import math

try: 
    import madgraph.madweight.diagram_class as diagram_class
    import madgraph.madweight.substructure_class as substructure_class
    import madgraph.madweight.MW_info as MW_param
    
except ImportError:
    import internal.madweight.diagram_class as diagram_class
    import internal.madweight.substructure_class as substructure_class
    import internal.madweight.MW_info as MW_param

go_to_main_dir = MW_param.go_to_main_dir
pjoin = os.path.join

#1 ##############################################
def verif_event(MWparam):

    # 0 ##############
    ####  go to main directory and copy file
    go_to_main_dir()
    try:
        os.mkdir('./Events/'+MWparam.name)
    except:
        pass
    os.system('cp ./Events/input.lhco ./Events/'+MWparam.name+'/')
    # 1 ##############
    ####   take run information
    for MW_dir in MWparam.MW_listdir:
        start=time.time()
        select=Lhco_filter(MW_dir,'input.lhco',MWparam)
        print 'time Lhco_filter',time.time()-start

def restrict_event_passing_cut(MWparam):
    """ return the number of events in a (previous) run which pass the 'new' cut """
    
    # 0 ##############
    ####  go to main directory and copy file
    go_to_main_dir()
    try:
        os.mkdir('./Events/'+MWparam.name)
    except:
        pass
    os.system('cp ./SubProcesses/'+MWparam.MW_listdir[0]+'/'+MWparam.old_name+'/verif.lhco ./Events/'+MWparam.name+'/')
    # 1 ##############
    ####   take run information
    MW_dir=MWparam.MW_listdir[0]
    filter=Lhco_filter(MW_dir,MWparam=MWparam,auto=0)
    return filter.verif_event(MWparam.name+'/verif.lhco',output=1) #output 1:return which event pass the cut
    
    

#1 ###############################################################################################################
class Lhco_filter:

    class Lepton_Without_Charge_Error(Exception): pass
    
    #2 ###############################################################################################################
    def __init__(self,directory,lhco_file='',MWparam='',auto=1,write_mode=1):
        """ input is either a file containing particule number info or a SubProcesses directory """

        start=time.time()        
        if MWparam:
            self.MWparam=MWparam
        else:
            import MW_param
            self.MWparam=MW_param.MW_info('MadWeight_card.dat')
        self.write_mode=write_mode

        #treat directory info
        if directory.count('SubProcesses'):
            self.directory=directory
        else:
            self.directory='./SubProcesses/'+directory

        self.lhco_file=lhco_file
        self.partdef=self.find_particle_number()        # find number of particle of each type
        #define internal variable
        self.write_events=0
        if lhco_file and auto:
            self.verif_event(lhco_file,self.partdef)

    #2 ###############################################################################################################
    def find_particle_number(self):

        # find number of particle of each type
        #self.nb_part['photon']=0
        self.load_particle_number(self.directory)
        #define each type of particle
        partdef=lhco_all_particles_def()
        if not self.MWparam.info['mw_perm']['bjet_is_jet_for_selection']:
            partdef.use_bjet()
        if self.MWparam.info.has_key('eventselection'):
            partdef.update_hlt_cut(self.MWparam.info['eventselection'])	

        return partdef
    
    #2 ###############################################################################################################        
    def load_particle_number(self,directory):
        """ extract the number of particule from the iconfigs """

        diag=diagram_class.MG_diagram(directory,1)

        list=['jet','bjet','electron','positron','muon','amuon','tau','atau', 'miss','photon']#,'miss']
        content=diag.output_type_info()
        
        total=0
        data={}
        for i in range(0,len(list)):
            data[list[i]]=content[i]
            total+=content[i]
        #data['n_out']=total
        
        #check status of the b-jet
        if not self.MWparam.info['mw_perm']['bjet_is_jet_for_selection']:
            self.use_bjet=1
        else:
            self.use_bjet=0
            data['jet']+=data['bjet']
            data['bjet']=0

        data['begin']=[0,1]
        data['miss']=[0,1]
        data['unknow']=range(0,10)
        self.nb_part=data
        return data

    #2 ###############################################################################################################
    def extract_file_info(self,dir=''):

        if dir:
            os.chdir('./SubProcesses/'+dir)

        ff=open('./info_part.dat','r')

        text=ff.readline()
        ff.close()
        list=['jet','bjet','electron','positron','muon','amuon','tau','atau','miss']
        info=text.split()[1:]
        #split.split(text)[1:]
        data={}
        total=0
        for i in range(0,len(list)):
            data[list[i]]=int(info[i])
            total+=int(info[i])
        data['n_out']=total
        os.chdir('../..')

        #check status of the b-jet
        if not self.MWparam.info['mw_perm']['bjet_is_jet_for_selection']:
            self.use_bjet=1
        else:
            self.use_bjet=0
            data['jet']+=data['bjet']
            data['bjet']=0



        self.nb_part=data
        return data

    #2 ###############################################################################################################
    def define_particle_number(self,particle,number):
        
        list=['jet','bjet','electron','positron','muon','amuon','tau','atau','miss']
        if particle not in list:
            print 'unknown type of particle'
            return
        else:
            self.nb_part[particle]=int(number)
        return

    #2 ###############################################################################################################
    def verif_event(self,file='',part_def='',output=0):
        """ use the cuts to select event in file
            output defines what returns the routine
                 0: write the file + returns how many events pass
                 1: returns the list of events passing cuts
        """ 
        start=time.time()

        #control input
        if not part_def:
            print 'use default part_def'
            part_def=self.partdef
        if not file:
            file='./Events/'+self.lhco_file

        if   os.path.isfile(file):  f_in=open(file,'r')
        elif os.path.isfile('./Events/'+file):  f_in=open('./Events/'+file,'r')
        else: sys.exit('FATAL ERROR: No experimental file \"'+file+'\" in Events directory.')

            
        #supress first X valid events:
        if self.MWparam.info['mw_run'].has_key('21'):
            self.start=int(self.MWparam.info['mw_run']['21'])
            print 'start', self.start
        else:
            self.start=0
            
        #define the output file
        if output==0:
            os.system('mkdir '+self.directory+'/'+self.MWparam.name+' &>/dev/null')
            self.f_out=open(self.directory+'/'+self.MWparam.name+'/verif_0.lhco','w')
        elif output==1:
            self.accepted_list=[]

        #print 'time  begin verif event Lhco_filter',time.time()-start
        #end init
        
        #initialize variable for the loop on events
        list_part=[] #store the different particle of the events
        nb_part={}   #dictionary saying with type of particles are expected
        self.event_position=0
        lhco_id_tag = set()
        nb_accepted = 0
        #start to reading the file
        for line in f_in:
            if nb_accepted >= self.MWparam.info['mw_run']['nb_exp_events']:
                break
            if line[0]=='#':
                continue
            try:
                start2=time.time()
                part=lhco_part(line)
                identity=part_def.identify_particle(part)
                part.def_identity(identity)
                if identity=='begin':
                    lhco_id_tag.add(line.split()[1])
                    if self.check_valid(nb_part):
                        nb_accepted += 1
                        self.write(list_part)
#                    elif self.write_events:
#                        print 'not valid'
                    #reinit with new block
                    list_part=[part]
                    nb_part={'begin':1,'unknow':0}
                    self.event_position+=1                    
                else:
                    list_part.append(part)
                    if nb_part.has_key(identity):
                        nb_part[identity]+=1
                    else:
                        nb_part[identity]=1
            except lhco_part.ErrorNotLHCOformat:
                #print 'error case'
                #if self.check_valid(nb_part):
                #    self.write(list_part)
               #reinit for next step
                list_part=[]
                nb_part={}			

        #check last data to be sure that we don't forget the last event
        if self.check_valid(nb_part):
            if nb_accepted < self.MWparam.info['mw_run']['nb_exp_events']:
                self.write(list_part)	
        print 'time  verif event Lhco_filter',time.time()-start
        print self.write_events-self.start,'selected  events for ',self.directory,' subprocess'
        # Comment this for multi-output run
        if self.write_events-self.start<self.MWparam.nb_event:
            name = self.directory.split('/')[-1]
            self.MWparam.nb_event_MW[name] = self.write_events-self.start
        if output==0:    
            return self.write_events
        elif output==1:
            return self.accepted_list

    #2 ###############################################################################################################
    def check_valid(self,nb_part):
        """ check if we have the correct number of input for each type of particle """

        list_key=self.nb_part.keys()+[key for key in nb_part if key not in self.nb_part.keys()]
        try:
            for key in list_key:
                if self.nb_part[key]==0:
                    if not nb_part.has_key(key):
                        continue
                    elif nb_part[key]==0:
                        continue
                    else:
                        return 0
                    
                if not nb_part.has_key(key):
                    return 0
                
                if type(self.nb_part[key])==list:
                    if nb_part[key] not in self.nb_part[key] :
                        return 0
                elif nb_part[key]!=self.nb_part[key]:
                    return 0
            return 1
        except KeyError:
            print nb_part
            print self.nb_part
            print key
            if self.write_events:   print 'key error'
            return 0

    write_order=['begin','jet','bjet','electron','positron','muon','amuon','tau','atau', 'photon','miss','init']
    #2 ###############################################################################################################
    def write(self,list_part):
        """ write the output file """

        if hasattr(self, 'f_out') and self.write_events and \
        self.write_events % (self.MWparam['mw_run']['nb_event_by_node'] * self.MWparam['mw_run']['event_packing']) == 0:
            
            i = self.write_events // (self.MWparam['mw_run']['nb_event_by_node'] * self.MWparam['mw_run']['event_packing'])
            name = self.f_out.name
            base, name = os.path.split(name)
            name = os.path.join(base, name.replace('_%i' % (i-1), '_%i' % i ))
            self.f_out = open(name,'w')    
        
        
        self.write_events+=1
        if self.write_mode==0:
            return
        if self.write_events<=self.start:
            return

        #check wich output to update
        if hasattr(self,'f_out'): #output file is defined:
            
            write_order = self.write_order

            for i in range(0,len(write_order)):
                for j in range(0,len(list_part)):
                    if list_part[j].name==write_order[i]:
                        self.f_out.write(list_part[j].line)
                        
        if hasattr(self,'accepted_list'): #output which event passing cut
            self.accepted_list.append(self.event_position-1) #the first should be zero and this number is already updated

#1 ########################################################################	
class lhco_all_particles_def(dict):
        """
                a class containing all the particles definition
        """
        
        #2 ########################################################################	
        class lhco_id:
                """ a class containing the LHCO definition-restriction of on each lhco particles
                        this defines rules to know of wich type a particle is	
                """
                eta_max=1e2
                pt_max=1e6
                ntrk_max=1e3
                
                #3 ########################################################################	
                class bound_limit:
                        """ set a minimum and a maximum value for a parameter """
                        #4 ########################################################################	
                        def __init__(self,vmin,vmax):
                                self.vmin=vmin
                                self.vmax=vmax
                                
                        #4 ########################################################################	
                        def redefine(self,min,max):
                                self.__init__(min,max)		
                                
                        #4 ########################################################################	
                        def inlimit(self,value):
                                """ check if value is between min and max """
                                value=float(value)
                                if value>=self.vmin and value<=self.vmax:
                                    return 1
                                else:
#                                    print 'failed check',value,self.vmin ,self.vmax
                                    return 0
                                
                #3 ########################################################################	
                def __init__(self,name,type,pid):
                        """ initialize the object. 
                                        name is the name of the particle described
                                        type is the type value in the lhco file
                        """
                                        
                        self.name=name
                        self.lhcoid=str(type)
                        self.init_default()
                        self.pid=pid
                
                #3 ########################################################################		
                def init_default(self):
                    """ put the zero cut on the particle """
                        
                    self.eta=self.bound_limit(-self.eta_max,self.eta_max)
                    self.phi=self.bound_limit(-math.pi,2*math.pi)
                    self.pt=self.bound_limit(0,self.pt_max)		
                    self.jmass=self.bound_limit(-1e-5,self.pt_max)
                    self.ntrk=self.bound_limit(-self.ntrk_max,self.ntrk_max)
                    self.btag=self.bound_limit(-100,100)
                    self.hadem=self.bound_limit(-1e99,1e99)
                    self.dum1=self.bound_limit(-1e99,1e99)
                    self.dum2=self.bound_limit(-1e99,1e99)
                    #special variable (not line of the lhco file)
                    self.E=self.bound_limit(0,1e99)
                    
                #3 ########################################################################		
                def restrict(self,tag,min_val,max_val):
                        """ add a restriction on a parameter """
                        
                        eval('self.'+tag+'.redefine('+str(min_val)+','+str(max_val)+')')
                        
                #3 ########################################################################		
                def check(self,particle):
                        """ check if a particle is of this type or not """

                        if (particle.lhcoid!=self.lhcoid):
                            self.failed_reason='lhcoid'
                            return 0
                        elif(not self.ntrk.inlimit(particle.ntrk)):
                            self.failed_reason='ntrk'
                            return 0
                        elif(not self.btag.inlimit(particle.btag)):
                            self.failed_reason='btag'
                            return 0
                        elif(not self.eta.inlimit(particle.eta)):
                            self.failed_reason='eta'
                            return 0
                        elif(not self.phi.inlimit(particle.phi)):
                            self.failed_reason='phi'
                            return 0
                        elif(not self.pt.inlimit(particle.pt)):
                            self.failed_reason='pt'
                            return 0
                        elif(not self.jmass.inlimit(particle.jmass)):
                            self.failed_reason='jmass'
                            return 0
                        elif(not self.hadem.inlimit(particle.hadem)):
                            self.failed_reason='hadem'
                            return 0
                        elif(not self.E.inlimit(particle.E)):
                            self.failed_reason='E'
                            return 0
                        elif(not self.dum1.inlimit(particle.dum1)):
                            self.failed_reason='dum1'
                            return 0
                        elif(not self.dum2.inlimit(particle.dum2)):
                            self.failed_reason='dum2'
                            return 0
                        else:
                            self.failed_reason='no'
                            return 1




                        if (particle.lhcoid==self.lhcoid and
                           self.ntrk.inlimit(particle.ntrk) and
                           self.btag.inlimit(particle.btag) and
                           self.hadem.inlimit(particle.hadem) and
                           self.eta.inlimit(particle.eta) and
                           self.phi.inlimit(particle.phi) and
                           self.pt.inlimit(particle.pt) and
                           self.jmass.inlimit(particle.jmass) and
                           self.dum1.inlimit(particle.dum1) and
                           self.dum2.inlimit(particle.dum2) and
                           self.E.inlimit(particle.E)):
                                return 1
                        else: 
                                return 0
                                
                
                
        #2 ########################################################################		
        def __init__(self):
        
                #lepton definition
                self['electron']=self.lhco_id('electron',1,11)
                self.electron=self['electron']
                self.electron.restrict('ntrk',-1,-1)
                self['positron']=self.lhco_id('positron',1,-11)
                self.positron=self['positron']
                self.positron.restrict('ntrk',1,1)
                self['muon']=self.lhco_id('muon',2,13)
                self.muon=self['muon']
                self.muon.restrict('ntrk',-1,-1)
                self['amuon']=self.lhco_id('amuon',2,-13)
                self.amuon=self['amuon']
                self.amuon.restrict('ntrk',1,1)
                self['tau']=self.lhco_id('tau',3,15)
                self.tau=self['tau']
                self.tau.restrict('ntrk',-3,-1)
                self['atau']=self.lhco_id('atau',3,15)
                self.atau=self['atau']
                self.atau.restrict('ntrk',1,3)

                #photon definition
                self['photon']=self.lhco_id('photon',0,22)
                self.photon=self['photon']
                
                #hadronic definition
                #default no distinction between jet and b jet
                self['jet']=self.lhco_id('light_jet',4,1)
                self.jet=self['jet']
                #missing-et
                self['miss']=self.lhco_id('miss',6,0)
                self.miss=self['miss']
                #parton
                self['init']=self.lhco_id('init',7,0)
                self.init=self['init']
                #begin
                self['begin']=self.lhco_id('begin',99,0)
                self.begin=self['begin']

                #avoid to many warning
                self.nb_warning = 0
        
        #2 ########################################################################		
        def use_bjet(self):
                """ separate the class jet between jet and bjet """
                self['jet'].restrict('btag',0,0)
                self['bjet']=self.lhco_id('bjet',4,5)
                self['bjet'].restrict('btag',1,4)
                self.bjet=self['bjet']
                
        #2 ########################################################################
        def update_hlt_cut(self,hltcut):
            """ take the hlt cut from the Madweight card """

            print 'update cut :',hltcut
            for key in hltcut:
                name,param=key.split('_')	
                if(type(hltcut[key])==list):			
                    self[name].restrict(param,hltcut[key][0],hltcut[key][1])
                else:				
                    self[name].restrict(param,hltcut[key],9e99)

#2 ########################################################################				
        def identify_particle(self,particle):
            """ find in wich category the particles belongs """

            #print particle.line[:-1],
            for name in ['begin','jet','miss','electron','photon']: #to fastenize the test in most of the case
                if self[name].check(particle):
                    #print name
                    return name
                
            for name in [name for name in self.keys() if name not in ['begin','jet','miss','electron','photon']]:
                if self[name].check(particle):
                    #print name
                    return name
                
            if not self.nb_warning:
                print 'Some particles are not identified to any types. This could occur if you specify some cuts.'
                print 'Following lines shows  a sample of those unidentified lines:'
                
            if self.nb_warning<10:
                self.nb_warning+=1
                print particle.line[:-1]
            return 'unknow'




#define here for optimization purpose
pat_lhco_line=re.compile(r'''^\s*(?P<card>\d*)\s+               #cardinal
                               (?P<type>\d*)                # type: 0 = photon,1 = elec,2 = muon,3 = hadronic tau,4 = jet,6 =MTE
                               \s+(?P<eta>[+-\.\de]+)             # pseudorapidity
                               \s+(?P<phi>[+-\.\de]+)\s+       # phi
                               (?P<pt>[+-\.\de]*)\s+       #pt
                               (?P<jmass>[+-\.\de]+)\s+    #invariant mass of the object
                               (?P<ntrk>[+-\.\de]+)\s+     #number of tracks associated( muliplied by charge for lepton)
                               (?P<btag>[+-\.\de]+)\s+     #jet: !=0 taggued b//muon: closest jet
                               (?P<hadem>[+-\.\de]+)\s+  #hadronic versus electromagnetic energy deposited
                               (?P<dum1>[+-\.\de]+)\s+   # user free at this stage
                               (?P<dum2>[+-\.\de]+)\s+$  # user free at this stage
                        ''',re.I+re.VERBOSE) # 80= VERBOSE +ignore case
                        
                        
pat_new_block=re.compile(r'''^\s*0\s+               #cardinal
                               (?P<type>\S*)                # type: 0 = photon,1 = elec,2 = muon,3 = hadronic tau,4 = jet,6 =MTE
                               \s+(?P<eta>[+-\.\de]+)\s+
                               ''',re.I+re.VERBOSE)
    


                                
#1 ########################################################################		
class lhco_part(dict):
    """ a class for a particle from the lhco line """
        
    class ErrorNotLHCOformat(Exception): pass

        
    #2 ########################################################################
    def __init__(self,line):
        """ charge a particle """
        self.line=line
        reobj=pat_lhco_line.search(line)
        if not reobj:
            if not self.beginblok(line):
                raise self.ErrorNotLHCOformat	
            else:
                return
        self.card=reobj.group('card')
        self.lhcoid=reobj.group('type')
        self.eta=reobj.group('eta')				
        self.phi=reobj.group('phi')		
        self.pt=reobj.group('pt')	
        self.jmass=reobj.group('jmass')	
        self.ntrk=reobj.group('ntrk')	
        self.btag=reobj.group('btag')
        self.hadem=reobj.group('hadem')
        self.dum1=reobj.group('dum1')
        self.dum2=reobj.group('dum2')
        #special variable:
        self.px=float(self.pt)*math.cos(float(self.phi))
        self.py=float(self.pt)*math.sin(float(self.phi))
        self.pz=float(self.pt)*math.sinh(float(self.eta))
        self.E =math.sqrt(self.px**2+self.py**2+self.pz**2+float(self.jmass)**2)

    #2 ########################################################################
    def beginblok(self,line):
        """ charge a particle """
                
        reobj=pat_new_block.search(line)	
        if not reobj:
            return 0
                        
        self.card=0
        self.lhcoid='99'
        self.eta='0'
        self.phi='0'		
        self.pt='0'
        self.jmass='0'
        self.ntrk='0'
        self.btag='0'
        self.hadem='0'
        self.dum1='0'
        self.dum2='0'
        self.E='0'
        return 1

    #2  #########################################################################
    def def_identity(self,name):
        """ def name of the type of line """

        self.name=name

class Test_one_file(Lhco_filter):

    def __init__(self):
        self.lhco_file=raw_input('enter the name of the file to test : ').split()[0]
        Card_pos=raw_input('give position to a MadWeight_card.dat: ').split()[0]
        import MW_param
        self.MWparam=MW_param.MW_info(Card_pos)
        val=''
        while val not in ['0','1']: val=raw_input('use the file info_part.dat (0/1)?')
        if val=='1':
            self.partdef=self.extract_file_info()
        else:
            self.nb_part={}
            for element in ['jet','bjet','electron','positron','muon','amuon','tau','atau','miss']:
                value=raw_input('enter authorize value for the nb of particule of type '+element+' : ')
                value.split()
                if len(value)==1: self.nb_part[element]=int(value[0])
                else: self.nb_part[element]=value
            self.nb_part['photon']=0

        partdef=lhco_all_particles_def()
        if not self.MWparam.info['mw_perm']['bjet_is_jet_for_selection']:
            partdef.use_bjet()
        if self.MWparam.info.has_key('eventselection'):
            partdef.update_hlt_cut(self.MWparam.info['eventselection'])
        self.partdef=partdef
        #define internal variable
        self.write_events=0    
        self.verif_event(self.lhco_file,self.partdef,output=1)

                
#########################################################################
#########################################################################
if(__name__=="__main__"):
    #import MW_param
    #info=MW_param.MW_info('MadWeight_card.dat')
    #verif_event(info)
    from MW_param import go_to_main_dir
    opt=sys.argv
    if len(opt)==1:
            go_to_main_dir()
            Lhco_filter('proc_card.dat')
    else:
        pos='/'.join(opt[0].split('/')[:-1])
        print pos
        sys.path.append(pos)
        print sys.path
        Test_one_file()
