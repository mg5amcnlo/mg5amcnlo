
invisible_list = [12,14,16,18]
invisible_list += [1000012,1000014,1000016,1000022,1000023,1000024,1000025,1000035]

class Particle:

    def __init__(self,MG_num,pid,fuse=0):
        if fuse:
            self.MG=str(MG_num)
        else:
            self.MG=int(MG_num)
        self.pid=int(pid)
        self.mass=0
        self.mother=0
        self.level=0
        self.twin=0
        self.neutrino=0
        self.external=0

               
    def add_mother(self,obj):
        self.mother=obj

    def def_twin(self):  #this routine is normaly not used any more#
        """ not use anymore: define the twin for the particle if not defined """
        print 'WARNING: Particle.def_twin() is an old routine. This routine hasn\'t been updated since 1/2/08'
        #but perhaps no update are needed
        
        if self.twin!=0:
            return self.twin
        else:
            if self.mother:
                if self.mother.des[0].MG==self.MG:
                    self.twin=self.mother.des[1]
                    self.mother.des[1].twin=self
                else:
                    self.twin=self.mother.des[0]
                    self.mother.des[0].twin=self
            else:
                self.twin=0
        return self.twin

    def def_mass(self,info):
        """definition of the mass of the particle"""
        
        try:       #this is for unfixed data coming from param_card
            self.mass=float(info['mass'][str(abs(self.pid))])
        except:
            self.mass=0 #O if not defined in param card
        

    def def_all_mother(self):
        "define list containing all mother of the particle"
        try:
            if self.all_mother_:
                return self.all_mother_
        except:
            pass

        mother=self.mother
        if mother==0:
            self.all_mother_=[]
            
        else:
            self.all_mother_=[mother]+mother.def_all_mother()
        
        return self.all_mother_ 

            
    def all_mother(self):
        """convinient alias"""
        return self.def_all_mother()
    
    def detect_neut_in_decay(self):
        """ detect the nearest neutrino in the decay branch
            return this neutrino and its level compare to this particle """

        analyse_list=list(self.des)
        level_list=[1,1]
        while analyse_list:
            particle=analyse_list.pop(0)
            level=level_list.pop(0)
            
            if particle.neutrino:
                return particle,level
            elif particle.external==0:
                analyse_list.append(particle.des)
                level_list.append(level+1)
                level_list.append(level+1)

        return 0,0 #no neutrino detected
                        

    


    def __str__(self):
        """ print routine """
        
        text='particle: '+str(self.MG)+'\tpid : '
        if self.pid<10 and self.pid>0:
            text+='  '
        elif self.pid>9:
            text+=' '
        text+=str(self.pid)+'\tlevel: '+str(self.level)
        try:
            text+='\tmother: '+str(self.mother.MG)
        except:
            pass
        try:
            text+='\ttwin: '+str(self.twin.MG)
        except:
            pass
        return text


class external_part(Particle):

    def __init__(self,MG_num,pid):
        Particle.__init__(self,MG_num,pid)
        self.external=1
        self.neutrino=self.is_invisible()
        self.width=0
        #self.tf_width={}
        #self.tf_width['p']=-1
        #self.tf_width['eta']=-1
        #self.tf_width['phi']=-1        

    def is_invisible(self):
        "detect if the external particle is visible in the detector or not"

        if (abs(self.pid) in invisible_list):
            self.neutrino=1
        else:
            self.neutrino=0

        return self.neutrino

    def define_tf_width(self,W_p,W_eta,W_phi):
        tf_width={}
        tf_width['p']=float(W_p)
        tf_width['eta']=float(W_eta)
        tf_width['phi']=float(W_phi)

        self.tf_width=tf_width

    def unaligned_propa(self,part,total=1):
        """ compute the number of propa in common
            return the total number of propagator before the two particle (default)
            in addtion it can give the number of uncorelated propagator for the two particle (put total=0) 
        """
        not_defined_mother=[]
        if part.mother in self.all_mother():
            if total:
                return self.level
            else:
                return self.level, self.level-part.level,0

        if self.mother in part.all_mother():
            if total:
                return part.level        
            else:
                print part.level,self.level
                return part.level, 0, part.level-self.level
        #=======================================================================
        # try:
        #    if part.mother in self.all_mother():
        #        return self.level
        # except: #self.all_mother not defined
        #    not_defined_mother.append(self)
        #    
        # try:
        #    if self.mother in part.all_mother():
        #        return part.level
        # except: #part.all_mother not defined
        #    not_defined_mother.append(part)
        #    
        # #define all mother if needed
        # for ext_part in not_defined_mother:
        #    ext_part.def_all_mother()
        #=======================================================================

        common=0
        for propa in self.all_mother():
            if propa in part.all_mother() and propa.channel[0]=='S':
                common+=1
#        print 'propagator in front',self.MG,'-',part.MG,':',self.level,'+',part.level,'-',common,'=',self.level+part.level-common
        
        if total:
            return self.level+part.level-common
        else:
            return self.level+part.level-common,self.level-common,part.level-common
        
class propagator(Particle):

    def __init__(self,MG_num,pid,channel):
        Particle.__init__(self,MG_num,pid)
        self.external=0
        self.channel=channel
        self.mass=0
        self.width=0

    def def_desintegation(self,obj):
        "define mother/child/twin relation: the two entry are the child object"

        try:
            self.des.append(obj)
            #create twin relation
            obj.twin=self.des[0]
            self.des[0].twin=obj
        except:
            self.des=[obj]

        obj.mother=self

    
    def def_mass(self,info):
        "define mass and width of the particle"

        Particle.def_mass(self,info) #definition of the mass
        
        #definition of the width
        try:
            self.width=float(info['decay'][str(abs(self.pid))])
        except:
            self.width=0 #O if not defined in param card
        







    def __str__(self):
        """ print lot of information """
        
        text='particle: '+str(self.MG)+'\tpid : '
        if self.pid<10 and self.pid>0:
            text+='  '
        elif self.pid>9:
            text+=' '
        text+=str(self.pid)+'\tlevel: '+str(self.level)
        text+='\tchannel: '+str(self.channel)
        try:
            text+='\tdes: '+str(self.des[0].MG)+' '+str(self.des[1].MG)
        except:
            try:
                text+='\tdes: '+str(self.des[0].MG)
            except:
                pass
        try:
            text+='\tmother: '+str(self.mother.MG)
        except:
            text+='\t\t'
            
        text+='\tmass/width: '+str(self.mass)+'/'+str(self.width)
        return text
