#!/usr/bin/env python


try:
    import madgraph.madweight.blob_solution as blob_solution
except ImportError:
    import internal.madweight.blob_solution as blob_solution
    
Level_generation = blob_solution.Level_generation
Blob_solution = blob_solution.Blob_solution  
Block_ECS =blob_solution.Block_ECS
Block_sector =blob_solution.Block_sector
Block_B =blob_solution.Block_B

import sys

class diagram:

    def __init__(self):

        self.content={}
        self.prop_content=[]
        self.ext_content=[]
        self.neut_content=[]
        self.num_propa=0
        self.ext_part=0
        self.num_neut=0
        self.config=0



    def add_content(self,MG_id,object):
        "add a element in the diagram"
        self.content[int(MG_id)]=object
        
        if object.external and int(MG_id) > 2:
            self.ext_content.append(object)
            self.ext_part+=1
        elif not object.external:
            self.prop_content.append(object)
            self.num_propa+=1
        

    def define_neut_content(self):
        "define neutrino content"
        self.neut_content=[]
        for ExtPart in self.ext_content :
           if(ExtPart.neutrino):
               self.neut_content.append(ExtPart)
        self.num_neut=len(self.neut_content)


    def define_level(self):
        """ complete self.content[i].level:number a propa before a external particle, and the number of propa before
        (with himself for a propa)"""

        #level of a propagator
        for i in range(1,self.num_propa+1):
            propa=self.prop_content[-i]
            
            if(propa.mother in self.prop_content and propa.channel=='S'):
                propa.level=propa.mother.level+1
            elif(propa.channel=='T'):
                propa.level=0   
            else:
                propa.level=1
            #print propa.MG, propa.level
        #level of initial particle
        self.content[1].level=0
        self.content[2].level=0
        #level of an external particle
        for ExtPart in self.ext_content:
            if ExtPart.mother:
                ExtPart.level=ExtPart.mother.level
            else:
                ExtPart.level=0


    def order_in_level(self):
        "return list with external particle ordering following level"

        #organise in level in a dictionary
        dict_level_to_part={}
        for particle in self.ext_content:
            if particle.level in dict_level_to_part.keys():
                dict_level_to_part[particle.level].append(particle)
            else:
                dict_level_to_part[particle.level]=[particle]

        #re-organise like a list
        self.ext_content=[]
        level=-1
        while 1:
            if len(dict_level_to_part)==0:
                break
            level+=1
            self.ext_content+=dict_level_to_part.pop(level,[])

    def contains_particle(self,particle):
        """ check if tha particle is in the diagram """
        #note that content is a dict or list depending of the object
        if type(self.content)== dict: 
            if self.content.has_key(particle.MG):
                return 1
            else:
                return 0
        elif type(self.content)== list:
            if particle in self.content:
                return 1
            else:
                return 0



class ECS_sector_no_multi_channel(diagram,Level_generation):
    """ The ECS sector.
        The parenty to Blob_solution is slightly dangerous, lot of variables are different, but some part are equivalent
        lot of time ECS play the two role: Blob and Blob solution
    """

    def __init__(self,diag,chgt_var,neut_in_ecs,unaligned):
        """ create the ECS SECTOR
            neut_in_ecs are the fondamental particle of the class (and so not always neutrino)
        """
        neut_in_class={'a':0,'b':1,'c':1,'d':2,'e':2,'f':2,'g':2}
        intrinsec_in_class={'a':2,'b':1,'c':1,'d':2,'e':2,'f':2,'g':2}#number of particles used to restore E-p conservation
        
        #initialization
        diagram.__init__(self)
        Level_generation.__init__(self,'ecs')
        self.fuse=0
        #store initial input
        self.MG_sec=diag
        self.chgt_var=chgt_var  
        if type(neut_in_ecs)==list:
            self.main_content=neut_in_ecs
        else:
            self.main_content=[neut_in_ecs]
        self.unaligned=unaligned
        #construct more elaborate things
        self.num_neut=neut_in_class[chgt_var]
        self.intrinsec=intrinsec_in_class[chgt_var] #number of particles used to restore E-p conservation
        self.opt=diag.opt

    def define_blob(self,diag,propa_in_ecs=""):
        """ define the blob associated to this Enlarged constraint sector """

        if not propa_in_ecs:
            propa_in_ecs=self.main_content

        self.blob_content=[]
        
        #define blob from the black box
        

        #define all the mother of the one/two central particle of the ecs
        for i in range(0,self.intrinsec):

            #define a list of combine all mother
            if i==0:
               combine_all_mother=[] 
               combine_all_mother+=propa_in_ecs[i].all_mother() #not in one line (we doesn't want to have a pointer)
               continue
            #i>0
            for mother in  propa_in_ecs[i].all_mother():
                if mother not in combine_all_mother:
                    combine_all_mother.append(mother)
                    
        #define blob
        for particle in combine_all_mother:
            #print 'mother',particle.MG, len(combine_all_mother)
            for desint in particle.des:
                if desint not in combine_all_mother and desint not in propa_in_ecs:
                    blob_sector(desint,diag,ECS_sec=self)

        #This technique is not complete for 'T' channel. In this case some branch are simply forget.
        #add those branch
        for i in range(-1*diag.num_propa,0):
            propa=diag.content[i]
            #first look at T Channel
            if propa.channel=='T':
                for desint in propa.des:
                    if desint.external:
                        if desint not in propa_in_ecs:
                            blob_sector(desint,diag,ECS_sec=self)
                    #at this point particle is propagator
                    elif(desint not in combine_all_mother):
                        blob_sector(desint,diag,ECS_sec=self)
        #This technique fails for the 'free' branch => look at particle without mother
        for particle in diag.prop_content+diag.ext_content:
            if particle.mother==0:
                if particle not in combine_all_mother and particle not in propa_in_ecs:
                    blob_sector(particle,diag,ECS_sec=self)

    

    def equivalent_ECS(self):
        """ 1) define completely the change of variable for the enlarged ECS
                -find which propagator aligned,...
            2) define equivalent solution if any (B->C)
        """
      
        if self.num_neut==0:
            #option: authorize a!
            if self.opt.use_ecs_a:
                solution=Block_ECS(self,'a',self.main_content)
                return [self]
            else:
                return []
        elif self.num_neut==1:
            #option: authorize b or c!
            if self.opt.use_ecs_b or self.opt.use_ecs_c:
                output=self.equivalent_ECS_1neut()
                return output
            else:
                return []
        else:
            #two neutrino case
            #option: authorize d or e or f!
            if self.opt.use_ecs_d or self.opt.use_ecs_e or self.opt.use_ecs_f :
                output=self.equivalent_ECS_2neut()
                return output
            else:
               return [] 



    def equivalent_ECS_1neut(self,use_noresonant=0):
        """ 1) define completly the change of variable for the enlarged ECS
                -find which propagator aligned,...
            2) define equivalent solution if any (B->C)
        """
        # 1.1) ->search for the two thiner propagator+ check if we can pass in C change of variable
        thiner=self.main_content[0].mother.width
        part_thin=self.main_content[0].mother
        thiner_bef_m0=500     #thiner propagator in ascendance of mass nul (for Case C)
        thiner_m0=500         #thiner mother of a mass null particle (for Case C)
        madeC=0
        motherX=self.main_content[0]

        while 1:
            motherX=motherX.mother
            if motherX==0:
                break
            
            if motherX.width<thiner and motherX.channel=='S':
                thiner=motherX.width
                part_thin=motherX

            #check for C splitting:
            try:
                if motherX.twin.external and motherX.twin.mass==0:
                    if motherX.twin.tf_level>1 and motherX.channel=='S':
                        madeC=1        
                        if motherX.mother.width<thiner_m0:
                           thiner_m0=motherX.mother.width
                           if thiner<thiner_bef_m0:
                               thiner_bef_m0=thiner    
            except:
                    pass    

        #1.2) ->create block and new ECS first for C change of variable secondly for B
        if madeC:
            New_sol=ECS_sector(self.MG_sec,'c',self.main_content,self.unaligned)
            fuse_list=[]
            motherX=self.main_content[0]
            while motherX.width!=thiner_bef_m0:
                fuse_list.append(motherX.twin)
                motherX=motherX.mother
            fuse_particle1=New_sol.define_fuse_region(fuse_list)
            fuse_list=[]
            while motherX.width!=thiner_m0:
                fuse_list.append(motherX.twin)
                motherX=motherX.mother
            fuse_particle2=New_sol.define_fuse_region(fuse_list)   
            Block_ECS(New_sol,'c',self.main_content+[fuse_particle1,fuse_particle2])                
            New_sol.order_block(New_sol)
            
            
        # define B change of variable
        fuse_list=[]
        motherX=self.main_content[0]
        while part_thin!=motherX:
            fuse_list.append(motherX.twin)
            motherX=motherX.mother
        fuse_particle=self.define_fuse_region(fuse_list)
        Block_ECS(self,'b',self.main_content+[fuse_particle])
        self.order_block(self)
        
        #2) Check definition from option
        sol=[]
        if self.opt.use_ecs_b:
            sol.append(self)
        if madeC and self.opt.use_ecs_c:
            sol.append(New_sol)
        
        return sol
            


    def equivalent_ECS_2neut(self):
        """ 1) define completly the change of variable for the enlarged ECS
                -find which propagator aligned,...
            Each change of variable are factorized!!!
        """
        total_propa,lim1,lim2=self.main_content[0].unaligned_propa(self.main_content[1],0)
        lim=[lim1,lim2]
        fuse_particle=[]
        #
        #     Enlarged constraint Sector D
        #
        if self.chgt_var=='d':
            #check validity from option
            if not self.opt.use_ecs_d:
                return []
            #find the two thiner propagator for each neutrino
            for i in [0,1]:
                thiner=500
                thiner2=600               
                #search thiner propa if option authorized
                if self.opt.ecs_fuse:
                    motherX=self.main_content[i]
                    for j in range(0,lim[i]):
                        motherX=motherX.mother
                        if motherX.width<thiner and motherX.channel=='S':
                            thiner2=thiner
                            thiner=motherX.width
                        elif motherX.width<thiner2 and motherX.channel=='S':
                            thiner2=motherX.width
                #check number of find solution and reassign
                if thiner==500:     #no propagator in S channel or don't use fuse
                    thiner=self.main_content[i].mother.width         #order between thiner
                    thiner2=self.main_content[i].mother.mother.width #is irrelevant at this stage
                if thiner2==500:    #only one propagator in S channel
                    if thiner!=self.main_content[i].mother.width:
                        thiner2=self.main_content[i].mother.mother.width
                    elif self.main_content[i].mother.channel=='S_flat':
                        thiner2=self.main_content[i].mother.mother.width
                    else:
                        thiner2=self.main_content[i].mother.mother.width
                
                #define fuse particle
                motherX=self.main_content[i]
                fuse_list=[]
                fuse_with_this_neut=0
                while fuse_with_this_neut<2:
                    fuse_list.append(motherX.twin)
                    motherX=motherX.mother
                    if motherX==0:
                        break
                    if motherX.width in [thiner,thiner2]:
                        fuse=self.define_fuse_region(fuse_list)
                        fuse_particle.append(fuse)
                        fuse_list=[]
                        fuse_with_this_neut+=1
            Block_ECS(self,'d',self.main_content+fuse_particle)
            self.order_block(self)
            return [self]
        #
        #     Enlarged constraint Sector E
        #
        elif self.chgt_var=='e':
            #check validity from option
            if not self.opt.use_ecs_e:
                return []
            #find the two thiner propagator
            for i in [0,1]:
                #search thiner propagator if option authorized
                if self.opt.ecs_fuse:
                    motherX=self.main_content[i].mother
                    thiner=motherX.width
                    for j in range(0,lim[i]-1):
                        motherX=motherX.mother
                        if motherX.width<thiner and motherX.channel=='S':
                            thiner=motherX.width
                else:
                    thiner=self.main_content[i].mother.width
                #define fuse particle
                motherX=self.main_content[i]
                fuse_list=[]
                while 1:
                    fuse_list.append(motherX.twin)
                    motherX=motherX.mother
                    if motherX.width==thiner:
                        fuse=self.define_fuse_region(fuse_list)
                        fuse_particle.append(fuse)
                        break               
            a=Block_ECS(self,'e',self.main_content+fuse_particle)
            self.order_block(self)
            #print 'ECS E content: ',
            #for particle in self.main_content+fuse_particle:#a.order_content:
            #    print particle.MG,
            #print
            return [self]           
        #
        #     Enlarged constraint Sector F
        #
        elif self.chgt_var=='f' or self.chgt_var=='g':
            #check validity from option
            if not self.opt.use_ecs_f:
                return []
            #F change of variable can't have freedom to move those propagator
            twin_part=[]
            twin_part.append(self.main_content[0].twin)
            twin_part.append(self.main_content[1].twin)                       
            Block_ECS(self,self.chgt_var,self.main_content+twin_part)
            self.order_block(self)
            return [self]


    def update_unaligned_with_blob(self):
        """ take care of the position of the other neutrino in order to have a more
            serious definition of the number of un-aligned variable

            this routine is designed for update the 1 neutrino case
        """

        control=0 #control of flux
        for neutrino in self.main_content:
            motherX=neutrino
            #treat first zero neutrino case: ->particle following
            if motherX.neutrino==0:
                try:
                    neutrino2,step=motherX.twin.detect_neut_in_decay()
                    if step<3 and neutrino2:
                        self.unaligned+=3-step
                except:
                    pass
            #treat all case ->particle before main
            while 1:
                motherX=motherX.mother
                if motherX==0:
                    break
                
                try:
                    neutrino2,step=motherX.twin.detect_neut_in_decay()
                except:
                    continue
                if neutrino2==0:
                    continue
                if neutrino2 in self.main_content:
                    control+=1
                    if control==2:
                        break
                    else:
                        continue
                if step<3:
                    self.unaligned+=3-step
    

    def order_block(self,main_sec=''):

        new_order=[]
        step=1
        while 1:
            step+=1
            if len(self.step)==1:
                break
            if step>20:
                sys.exit('''ERROR: infinite loop detected in ECS_sector.order_block ''')
            
            block=self.step.pop(0)
            #print block
            if block.chgt_var not in ['2','3']:
                self.step.append(block)
            else:
                new_order.append(block)
        new_order.append(self.step[0])
        self.step=new_order


      

    
    def info(self):
        """ return some information about the ECS and the associated blob """
        num_in_part={'a':2,'b':2,'c':3,'d':6,'e':4,'f':4,'g':4}
        
        text='\t** Enlarged Contraint Sector global information **\n\n'
        text+= 'Class: '+self.chgt_var.upper()+'\n'
        text+='particle in ECS : '
        for i in range(0,num_in_part[self.chgt_var]):
            part=self.step[-1].order_content[i]
            if i%3==0 and i!=0:
                text+='\n\t\t\t\t  '
            if part.external:
                if part.neutrino:
                    text+=str(part.MG)+'(missing)\t'
                else:
                    text+=str(part.MG)+'(visible)\t'
            elif type(part.MG)==str:
                text+=str(part.MG)+'(fuse)\t'
            else:
                text+=str(part.MG)+'(propagator)\t'
                
        text+='\nblob linked are generated by :'
        for blob in self.blob_content:
            if blob.main.MG<0:
                text+=str(blob.main.MG)+'\t'
        text+='\n'
        return text

    def __str__(self):
        text= 'ECS info:'
        text+= 'Class: '+str(self.chgt_var)+' particles in ECS : '
        for part in self.ext_content:
            text+=str(part.MG)+'\t'
        text+='|| linked blob(\'s): '
        try:
            for blob in self.blob_content:
                text+=str(blob.main.MG)+'\t'
        except:
            text+='not yet defined'
        return text








class blob_sector(diagram):
    """ blob structure """
    
    def __init__(self,particle,diag,ECS_sec=''):
        """create completly a blob if he is not defined already """

        #Check if already defined + validity of the definition of a blob
        if diag.blob_content.has_key(particle.MG): # check if the blob is already defined
            self=diag.blob_content[particle.MG]
            self.put_in_ecs(ECS_sec)
            return #stop creation
        elif particle.MG  in range(0,diag.num_init):#remove initial particle from creating blob
            return None
        elif not particle.external:
            if not particle.channel.startswith('S'): #remove T channel from creating a blob
                    return None    

        #creation of a new blob
        diagram.__init__(self)
        #put the main_diag up-to-date
        diag.blob_content[particle.MG]=self
        #put the ECS up-to-date
        self.put_in_ecs(ECS_sec)
        #put information in the blob
        self.main=particle
        self.generate_content()
        self.solution=[]
        #option
        self.opt=diag.opt


    def put_in_ecs(self,ECS_sec):
        """ put this blob in the blob content of ECS """
        #option put in ECS-blob content ->remark the blob is independant of the ECS
        if ECS_sec:
            if self not in ECS_sec.blob_content:
                ECS_sec.blob_content.append(self)

    def generate_content(self):
        """ import all the information for the blog """
        self.content=[self.main]
        for particle in self.content:
            if particle.external:
                self.ext_content.append(particle)
                self.ext_part+=1
                if particle.neutrino:
                    self.neut_content.append(particle)
                    self.num_neut+=1
            else:
                self.prop_content.append(particle)
                self.num_propa+=1
                self.content+=particle.des
        self.order_in_level()
        #order in propa is not good ->change it?

    def find_solutions(self):
        """ find a first solution to resolve the blob.
            The idea is to take the more local possibility in all case
            this is perhaps not the best solution but it's only the beginning of the resolution"""
        sol1=Blob_solution(self)
        sol1.find_all_solutions(self)
        self.supress_identical_solution()


    def supress_identical_solution(self):
        """ supress identical solution """
        
        del_sol=[]
        for i in range(0,len(self.solution)):
            for j in range(i+1,len(self.solution)):
                if str(self.solution[i])==str(self.solution[j]):
                    del_sol.append(i)
                    break
        for i in range(0,len(del_sol)):
             del self.solution[del_sol[i]-i]
             

    def __str__(self):
        text= 'Blob details: main '+str(self.main.MG)+'\n'
        for solution in self.solution:
            text+=str(solution)+'\n'
        return text


class ECS_sector(ECS_sector_no_multi_channel):
    """ modify version of ECS sector, returning the different 
        possibility of unalignment in the black box 
        (usefull for multichannel mode) 
    """
    
    def equivalent_ECS_1neut(self,use_noresonant=0):
        """
            define completely the change of variable for the enlarged ECS
            - return all the possible ECS changing the particles entering in the B case
            - define equivalent soltution if any (B->C)
        """
        
        sol=[]    
        fuse_list=[self.main_content[0].twin]       #define the blob fuse 
        for propagator in self.main_content[0].all_mother():
            if propagator.channel.startswith('T'):
                break
            # define a B change of variable
            if self.opt.use_ecs_b:
                New_sol=ECS_sector(self.MG_sec,'b',self.main_content,self.unaligned)
                fuse_particle=New_sol.define_fuse_region(fuse_list)
                Block_ECS(New_sol,'b',self.main_content+[fuse_particle])
                New_sol.order_block(New_sol)
                sol.append(New_sol)
                
            #look for class C
            if self.opt.use_ecs_c:
                sol+=self.equivalent_ECS_passinC(propagator,fuse_list)
                
            #update fuse for next level
            fuse_list.append(propagator.twin)
            
            
        return sol
        
    
    def equivalent_ECS_passinC(self,propa1,fuse1):
        """ check if those information can create a C block and define it
            propa1: first propagator than should enter in the C block
            fuse1: list of particles following this propa and should be fuse 
        """
        
        particle2=propa1.twin
        if not particle2 or particle2.mass:
            return [] #the mass should be 0
        if particle2.MG<3:
            return []
        propa2=propa1.mother
        if propa2 == 0 or propa2.channel.startswith('T'):
            return [] #not enough propa remaining
        
        New_sol=ECS_sector(self.MG_sec,'c',self.main_content,self.unaligned)
        fuse=self.define_fuse_region(fuse1)
        Block_ECS(New_sol,'c',self.main_content+[fuse,particle2])                
        New_sol.order_block(New_sol)
        
        return [New_sol]
        
    def equivalent_ECS_2neut(self):
        """ 1) define completely the change of variable for the enlarged ECS
                -find which propagator aligned,...
            Each change of variable are factorized!!!
        """       

        if self.chgt_var!='d':
             return ECS_sector_no_multi_channel.equivalent_ECS_2neut(self)
        
        if not self.opt.use_ecs_d:
            return []
        
        sol=[]


        
        possible_propa_1=[propa for propa in self.main_content[0].all_mother() \
                          if propa not in self.main_content[1].all_mother() ]
        possible_propa_2=[propa for propa in self.main_content[1].all_mother() \
                          if propa not in self.main_content[0].all_mother() ]

        fuse_list1_1=[self.main_content[0].twin]       #define the blob fuse
        for propagator1_1 in possible_propa_1:
            fuse_list1_2=[propagator1_1.twin]
            for propagator1_2 in propagator1_1.all_mother():
                if propagator1_2 not in possible_propa_1 or propagator1_2.channel.startswith('T'):
                    break
                fuse_list2_1= [self.main_content[1].twin]       #define the blob fuse
                for propagator2_1 in possible_propa_2:
                    fuse_list2_2=[propagator2_1.twin]
                    for propagator2_2 in propagator2_1.all_mother():
                        if propagator2_2 not in possible_propa_2 or propagator2_2.channel.startswith('T'):
                            break
                        propagator=[propagator1_1,propagator1_2,propagator2_1,propagator2_2]
                        fuse=[fuse_list1_1,fuse_list1_2,fuse_list2_1,fuse_list2_2]          
                        sol+=self.define_new_ecs_d(propagator,fuse)

                        fuse_list2_2.append(propagator2_2.twin)
                    fuse_list2_1.append(propagator2_1.twin)
                fuse_list1_2.append(propagator1_2.twin)
            fuse_list1_1.append(propagator1_1.twin)
            
        return sol
               
    def define_new_ecs_d(self,propagator,fuse):
        """ return a valid object for this change of variable """
        
        for propa in propagator:
            if propa.channel.startswith('T'):
                return []

        fuse_particle=[]
        for data in fuse:
            fuse_particle.append(self.define_fuse_region(data))
        for particule in fuse_particle:
            if particule.MG in [1,2]:
                return []
        
            
        New_sol=ECS_sector(self.MG_sec,'d',self.main_content,self.unaligned)
        Block_ECS(New_sol,'d',self.main_content+fuse_particle)
        New_sol.order_block(New_sol)
        return [New_sol]
        
        
        
        

