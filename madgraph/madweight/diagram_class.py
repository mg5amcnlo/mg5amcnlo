#!/usr/bin/env python

import sys
import logging
import os
import stat
import re

logger = logging.getLogger('madweight.diagram_class')

try: 
    import madgraph.madweight.Cards as Cards
    import madgraph.madweight.particle_class as particle_class
    import madgraph.madweight.substructure_class as substructure_class
    import madgraph.madweight.blob_solution as blob_solution
    import madgraph.madweight.proc_info as proc_info
    import madgraph.madweight.MW_fct as MW_fct
    import madgraph.madweight.MW_info as MW_param
    import madgraph.various.misc as misc
except ImportError:
    import internal.madweight.Cards as Cards
    import internal.madweight.particle_class as particle_class
    import internal.madweight.substructure_class as substructure_class
    import internal.madweight.blob_solution as blob_solution
    import internal.madweight.proc_info as proc_info
    import internal.madweight.MW_fct as MW_fct   
    import internal.madweight.MW_info as MW_param 
    import internal.misc as misc
Particle = particle_class.Particle
propagator = particle_class.propagator
external_part = particle_class.external_part
diagram = substructure_class.diagram
ECS_sector = substructure_class.ECS_sector
blob_sector = substructure_class.blob_sector
Block_sector = blob_solution.Block_sector
Decay_info = proc_info.Decay_info
Multi_list = MW_fct.Multi_list

           
            
class MG_diagram(diagram):

    def __init__(self,dir_file,config,opt='default'):

        #init variable
        diagram.__init__(self)
        self.ECS_sol=[]
        self.blob_content={}
        self.num_init=2 #number of initial particle
        self.directory=dir_file

        #put information
        self.import_ext_part(dir_file)
        self.set_option(opt)
        self.config=config
        
    def organize_particle_content(self,param_card,tf_file):
        """ define production area and organize all the needed information """
                
        #Charge Proc_card information+ update information for production part.
        self.import_process(self.directory,self.config)
        process_tag=re.compile(r'''P(?P<tag>\d*)''')
        cond=process_tag.search(self.directory).group('tag')
        ParticlesFile=''
        if hasattr(self, 'ParticlesFile'):
            ParticlesFile=self.ParticlesFile
        proc_decay=Decay_info(self.directory,cond,ParticlesFile=ParticlesFile)
        self.identify_production_area(proc_decay)

        #structure information
        self.define_parameter(param_card)
        self.define_neutrino_stuff()
        self.define_tf_width(tf_file)
        self.define_level()
        self.order_in_level()


        #self.define_twin() -> put in the definition of the desintegration

    #####################################################################
    #                  FUNCTION LINKED TO INIT                          #
    #####################################################################

    def import_ext_part(self,dir_file):
        """ read leshouches.inc information """


        pid_list = Cards.read_leshouches_file(dir_file+'/'+'leshouche.inc')

        #create the particle object and put them in the content of the diag
        #add the mass corresponding to the card
        for i in range(1,len(pid_list)+1):#MG start at one
            a=external_part(i,pid_list[i-1]) 
            self.add_content(i,a)

    def import_process(self,dir_file,config):
        "definis le diagramme soit via un fichier, soit via un autre diagram"
        self.config=config
      
        #
        #    import propagator content
        #

        #pattern to recognise
        new_config=re.compile(r'''^\s*\*\s*(?P<iconfig>\d+)\s+(?P<igraph>\d+)''')
        propa_des=re.compile(r'''^\s+                      #begin line
                            (?P<propa>-?\d+)\s+         #desintegrated propa
                            (?P<des1>-?\d+)\s+          #product of desintegration
                            (?P<des2>-?\d+)\s+          #product of desintegration 2
                            (?P<mass>\w*)\s*          #text for the mass ot the part
                            (?P<width>\w*)\s*         #text for the width ot the part
                            (?P<channel>[ST]*)\s*         # S or T propagator
                            (?P<pid>-?\d*)              # pid of the propa                                
                            ''',re.VERBOSE)

        ff=open(dir_file+'/configs.inc','r')

        #lecture du fichier
        read=0   #tag is 1 if we are at the correct config.
        while 1:
            line=ff.readline()
            if line=='':
                break

            #check the configuration
            if new_config.search(line):
                if(new_config.search(line).group('iconfig')!=str(config)):
                    if read:
                        break
                    else:
                        continue
                else:
                    read=1
            elif(not read):
                continue
            

            pattern=propa_des.search(line)
            if not pattern:
                continue
            #
            #put the information in diag
            #
            i=int(pattern.group('propa'))
            propa=propagator(i,pattern.group('pid'),pattern.group('channel'))
            self.add_content(i,propa)
            self.content[i].def_desintegation(self.content[int(pattern.group('des1'))])
            self.content[i].def_desintegation(self.content[int(pattern.group('des2'))])


    def define_parameter(self,param_card):
        """define mass+width of all particle"""

        #load the information from the param card
        info=MW_param.read_card(param_card)

        for obj in self.content.values():
            obj.def_mass(info) #this routines defines also width if the particle is a propagator

    def define_tf_width(self,file):
        """ assign width of the TF function for all external particle """

        if not hasattr(self,'ParticlesFile'):
            self.ParticlesFile=Cards.Particles_file('./Source/MODEL/particles.dat')
        dico_pid_to_label=self.ParticlesFile.give_pid_to_label()           

        dico_type_to_tflevel={}
        has_theta_tf, has_phi_tf  = [] , []
        ff=open(file,'r')
        #pattern
        pattern=re.compile(r'''^\s*(?P<width>\w+)\s+(?P<type>[\w,01]*)''',re.VERBOSE)
        #read file
        while 1:
            line=ff.readline()
            if line=='':
                break

            if pattern.search(line):
                re_obj=pattern.search(line)
                if re_obj.group('width').lower()=='delta':
                    tf_level=0
                elif(re_obj.group('width').lower()=='thin'):
                    tf_level=1
                elif(re_obj.group('width').lower()=='x'):
                    self.x_constrained=int(re_obj.group('type'))
                    continue
                elif(re_obj.group('width').lower()=='theta'):
                    list=re_obj.group('type').split(',')
                    for tag in list:
                        has_theta_tf.append(tag)
                    continue
                elif(re_obj.group('width').lower()=='phi'):
                    list=re_obj.group('type').split(',')
                    for tag in list:
                        has_phi_tf.append(tag)
                    continue                
                else:
                    tf_level=2
                list=re_obj.group('type').split(',')
                for tag in list:
                    dico_type_to_tflevel[tag]=tf_level

        #put information in external particle
        for part in self.ext_content:
            try:
                label=dico_pid_to_label[abs(part.pid)]
            except KeyError:
                logger.info('particle with pdg %s has no transfer function define: Treated as missing energy' % part.pid)
                label = None
            if not part.neutrino:
                if dico_type_to_tflevel.has_key(label):
                    part.tf_level=dico_type_to_tflevel[label]
                else:
                    part.tf_level=0  #delta is the default            
            else:
                part.tf_level=3
            if label in has_theta_tf:
                part.has_theta_tf = True
            if label in has_phi_tf:
                part.has_phi_tf = True            
                
    def define_neutrino_stuff(self):
        """ put all neutrino dependent variable """

        #create neutrino_list
        diagram.define_neut_content(self)

        #"find the number of neutrino in the decay chain for each propa"
        self.find_num_neut_decay()

        #check for invisible propagator(propagator decaying only in invisible particle)
        self.detect_invisible_propa()
        

    def find_num_neut_decay(self):
        "find the number of neutrino in the decay chain for each propa"

        #put step by step using MG order classification
        for Propa in self.prop_content:
            num_neut=0
            for j in range(0,2):
                if Propa.des[j].external:
                    if Propa.des[j].neutrino:
                        num_neut+=1
                else:
                    num_neut+=Propa.des[j].NeutInDecay
            Propa.NeutInDecay=num_neut

    def detect_invisible_propa(self):
        "detect propagator decaying in fully invisible particle and treat this case"

        for neut in  self.neut_content:
            if neut.twin in self.neut_content:
                #update the  neutrino list
                self.neut_content.remove(neut)
                self.neut_content.remove(neut.twin)
                mother = external_part(neut.mother.MG, neut.mother.pid)
                self.neut_content.append(mother)
                #update the external list
                self.ext_content.remove(neut)
                self.ext_content.remove(neut.twin)
                self.ext_content.append(mother)
                #update the propagator list
                self.prop_content.remove(neut.mother)
                self.num_propa-=1

                #update on the mother to have 'correct' ext_particle definition
                mother.neutrino=1
                mother.level = neut.mother.level -1 #pass from propa to external definition
                #put standard info of external part:
                mother.tf_level=3


                #update num_neut_decay (remove 1 for all preceding propagator)
                obj=neut
                while 1:
                    try:
                        obj=obj.mother
                        obj.NeutInDecay+=-1
                    except:
                        break

                #to be sure that there is no missing cascade recall the function and stop
                self.detect_invisible_propa()
                break

    def define_twin(self): #This routine is not used any more#
        """ not use anymore: define the twin for all particle in the diagram """
        print 'WARNING: MG_diagram.define_twin() is an old routine. This routine has not been updated since 1/2/08'
        #but perhaps no update are needed
        for particle in self.content.values():
            particle.def_twin()
            #print particle.twin.MG, particle.MG


    #####################################################################################
    #                      INIT: find production area                                   #
    #####################################################################################


    def identify_production_area(self,proc_decay):
        """ identify (and tag) production part from the information decay of the proc_card

            Step:  1) -> find initial particle in this process (no mother or S channel with T channel mother)
                   2) -> launch (self-iterative) check for the decay
                   3) -> tag particle
        """
        # Step 1
        init_list=[]
        for particle in self.ext_content:
            if particle.mother==0 or particle.mother.channel=='T':
                init_list.append(particle)
        for particle in self.prop_content:
                if particle.mother==0 and particle.channel=='S':
                    init_list.append(particle)
                elif particle.channel=='S':
                    if  particle.mother.channel=='T':
                        init_list.append(particle)

# Pierre
        #Step 2
        init_propa=self.check_decay(init_list,proc_decay.decay_diag)
        #step 3:
        if not init_propa:
            return
            raise Exception('No matching between process information and feynman diagram information')
        
        #print 'firt available propa (MG:PID) : ',
        for particle in init_propa:
            #print '(',particle.MG,':',particle.pid,') ',
            motherX=particle
            while 1:
                motherX=motherX.mother
                if motherX==0:
                    break
                motherX.channel='T'
        #print
        

    def check_decay(self,particle_list,decay_list):
        """ check if the Particle_list is equivalent to the Decay_list if not return the list of correct Particle

            Step 1) check if we have correct number of particle ->if not decay one or the other and restart
                 2) check if we have correct pid
                 3) check for decay in each case
        """

        #step 1:
        if len(particle_list)<len(decay_list):
            init_list=list(particle_list)
            for particle in init_list:
                particle_list=list(init_list)
                if  particle.external:
                    continue
                particle_list.remove(particle)
                particle_list+=particle.des
                out=self.check_decay(particle_list,decay_list)
                if out:
                    return out
            #no config found
            print "no config found"
            return 0
        
        #step 2:

        particle_list_pid=[]
        decay_Mlist_pid=Multi_list()
        decay_list_pid=[]
        for particle in particle_list:
            particle_list_pid.append(particle.pid)
        for particle in decay_list:
            decay_Mlist_pid.append(particle.pid) #particle.pid is a list of possible pid
            decay_list_pid+=particle.pid


        for pid in particle_list_pid:
            if pid not in decay_list_pid:
                return 0
            
        #print 'pass in step 3 '
        #step 3
        all_order_particle=decay_Mlist_pid.give_list_possiblity(particle_list,'pid')
        for order_particle in all_order_particle:
            for i in range(0,len(order_particle)):
                if not order_particle[i].external:
                    out=self.check_decay(order_particle[i].des,decay_list[i].des)
                    if not out:
                        break
                if i==len(order_particle)-1: #all suceed
                    return order_particle    #return succeed info

        print 'error 392'
        return 0


    def detect_non_resonant_area(self):
        """ detect the non resonant part of the diagram """

        #detection
        nb_sigma=3
        detect=[]
        for particle in self.prop_content:
            if particle.channel!='S':
                continue

            value=particle.mass+nb_sigma*particle.width
            for daughter in particle.des:
                value-=daughter.mass-nb_sigma*daughter.width

            if value<0:
                detect.append([particle]+particle.des)

                
        #verify possible choice, and put in output unaligned particle
        nb_sigma=3
        nb2_sigma=5
        unaligned=Multi_list()
        for ambiguous_area in detect:
            deviation={}
            for i in range(0,len(ambiguous_area)):
                value=0
                if ambiguous_area[i].external:
                    continue
                

                for j in range(0,len(ambiguous_area)):
                    if i!=j and j!=0:
                        value+=ambiguous_area[j].mass-nb_sigma*ambiguous_area[j].width
                    elif i!=j:
                        value-=ambiguous_area[j].mass-nb_sigma*ambiguous_area[j].width
                    else:
                        value-=ambiguous_area[j].mass
                if ambiguous_area[i].width:
                    value=abs(value/ambiguous_area[i].width)
                else:
                    value=1e500
                if not deviation.has_key(value):
                    deviation[value]=[ambiguous_area[i]]
                else:
                    deviation[value].append(ambiguous_area[i])

            unaligned_here=[]
            minimal_sigma=1e500
            for sigma in deviation.keys():       #search for the minimal sigma
                if sigma<minimal_sigma:
                    minimal_sigma=sigma
            
            if minimal_sigma<nb2_sigma:
                gap=2
            else:
                gap=1

            for sigma in deviation.keys():
                if sigma<minimal_sigma+gap:
                    unaligned_here+=deviation[sigma]

            unaligned.append(unaligned_here)
                
        return unaligned


    def tag_unaligned(self,unaligned):
        """ store information that this propagator cann't be generated following BW
            restore the other propagator
        """

        for particle in self.prop_content:
            if particle in unaligned:
                particle.channel='S_flat'
            elif particle.channel=='S_flat':
                particle.channel='S'


    def clear_solution(self):
        """
           supress all data from the solution in order to restart a new solution possibility
        """
        self.ECS_sol=[]
        self.blob_content={}


    #####################################################################################
    #                      FUNCTION POUR LA DEFINITION DES ECS                          #
    #####################################################################################

    
    def define_Constraint_sector(self):
        "define the constraint sector and the different blob"
        
        #Case 0 neutrino
        if len(self.neut_content)==0:
            ECS=self.find_ECS_0neut(force=1) #return is a list!
            self.select_ECS(ECS)
        elif len(self.neut_content)==1:
            ECS0=self.find_ECS_0neut()
            ECS1=self.find_ECS_1neut()
            self.select_ECS(ECS0+ECS1)
        else:
            ECS0=self.find_ECS_0neut()
            ECS1=self.find_ECS_1neut()
            ECS2=self.find_ECS_2neut()
            self.select_ECS(ECS0+ECS1+ECS2)

    def define_ECS_as_solution(self,ECS_list):
        """define ECS for a new (or a list) of new solution(s) for the ECS change of variable"""

        if type(ECS_list)!=list:
            ECS_list=[ECS_list]
        self.ECS_sol+=ECS_list


        #update the blob needed for this diagram
        for ECS in ECS_list:
            ECS.define_blob(self)
        

    def select_ECS(self,ECS_list,define_solution=1):
        """ select the best(s) ECS (minimizing unfactorized propagator \
            define solution if define==1"""
        #idea
        #first find the lowest value of non-aligned intrinsic variable
        #secondly select those one and return(def=0) or put in solution(def=1)

        #create list with autorized change of variable:
        #at this point they are no C~B
        authorize_ecs=list(self.opt.ecs_on)
        if authorize_ecs.count('b') ^  authorize_ecs.count('c'):
            if authorize_ecs.count('b'):
                authorize_ecs.append('c')
            else:
                authorize_ecs.append('b')
        if authorize_ecs.count('f'):
            authorize_ecs.append('g')


        #first find the lowest value of non-aligned intrinsic variable
        lowest_value=100
        for ECS in ECS_list:
            if ECS.chgt_var not in authorize_ecs:
                continue
            ECS.update_unaligned_with_blob()
            if ECS.unaligned<lowest_value:
                lowest_value=ECS.unaligned

        #secondly select those one and return(def=0) or put in solution(def=1)
        solution=[]
        at_least_one_sol=0
        num_sol=0
        for ECS in ECS_list:
            if num_sol>=self.opt.max_sol:
                break
            if ECS.unaligned==lowest_value:
                ECS_equi_list=ECS.equivalent_ECS() #move the unfactorized propagator and finish to define ECS
                for ECS in ECS_equi_list:
                    at_least_one_sol=1
                if define_solution:
                    num_sol+=1
                    self.define_ECS_as_solution(ECS_equi_list)                   
                else:
                    num_sol+=1
                    solution+=ECS_equi_list

        if not at_least_one_sol:
            print self
            sys.exit('FATAL ERROR: NO CHANGE OF VARIABLE\n\
                      you probably put bad option in your card')

        #option on the number of condition
        if len(solution)>self.opt.max_sol:
            solution=solution[:self.opt.max_sol]

        return solution



    def find_ECS_0neut(self,force=0):
        """  find ECS containing no neutrino """


        #find 2 particle at lower level with sufficient width
        #but we have to minimise non aligned propa and not level!!
        min_tf_level=2
        if force==2:
            min_tf_level=1
        elif force==3:
            min_tf_level=0
        
        select=[]
        for part in self.ext_content:
            # part.tf_level,min_tf_level,force,part.tf_level>=min_tf_level
            if part.tf_level>=min_tf_level:
                select.append(part)
                level=part.level
                
        if len(select)<2 and force:
            force+=force
            if force>3:
                sys.exit('too tiny transfer function on all particle')
            ECS=self.find_ECS_0neut(force) #relaunch but be less restrictive
            return ECS

        ##check all possibility from select part in order to minimise non aligned propa
        sol=[]
        limit=min(6+self.ext_content[0].level,10)
        for i in range(0,len(select)):
            part1=select[i]
            if part1.level>limit/2+1:
                break
            for j in range(i+1,len(select)):           
                part2=select[j]
                if part2.level>limit:
                    break
                # 'couple', part1.MG,part2.MG,':',part1.unaligned_propa(part2),'<?',limit
                if part1.unaligned_propa(part2)<limit:
                    limit=part1.unaligned_propa(part2)
                    sol=[[part1,part2]]
                elif part1.unaligned_propa(part2)==limit:
                    sol.append([part1,part2])
                    
        ECS_list=[]
        for soluce in sol:
            ECS_sol=ECS_sector(self,'a',soluce,limit+2)
            ECS_list.append(ECS_sol)
        return ECS_list
        
        

    def find_ECS_1neut(self):
        """ find the lowest(s) neutrino and define ECS """

        lowest_neut=self.find_lowest_particle(lowest_level=1,out_num=1)
        ECS_list=[]
        for neut in lowest_neut:
            ECS_sol=ECS_sector(self,'b',neut,neut.level-1)
            ECS_list.append(ECS_sol)
        return ECS_list       
        

    def find_ECS_2neut(self):
        """ return best 2 neutrino ECS
            The way to solve this problem is globally the following:
            - A: find the two lowest neutrino
            - B: if both are at level 1 -> return F/G depending on TF on x1,x2
            - C: check if both have two propagator free before them -> return D
            - D: check if they are linked by a S-channel -> return E
            - E: else return nothing (always best in one neutrino)
        """
        ECS_list=[]
        #step A: find lowest neutrino
        lowest_neutrino=self.find_lowest_particle(lowest_level=1)
        if len(lowest_neutrino) < 2:
            return ECS_list

        #step B: check F case
        if lowest_neutrino[0].level==1 and lowest_neutrino[1].level==1:
            #typical F case return all combinatory if ambiguity
            for i in range(0,len(lowest_neutrino)):
                for j in range(i+1,len(lowest_neutrino)):
                    if self.x_constrained:
                        ECS_sol=ECS_sector(self,'f',[lowest_neutrino[i],lowest_neutrino[j]],2)
                    else:
                        ECS_sol=ECS_sector(self,'g',[lowest_neutrino[i],lowest_neutrino[j]],2)
                    ECS_list.append(ECS_sol)
            return ECS_list

        #step C: check for D case
        for i in range(0,len(lowest_neutrino)):
            for j in range(i+1,len(lowest_neutrino)):
                total_propa,free1,free2=lowest_neutrino[i].unaligned_propa(lowest_neutrino[j],0)
                if free1>1 and free2>1:                   
                    ECS_sol=ECS_sector(self,'d',[lowest_neutrino[i],lowest_neutrino[j]],total_propa-4)
                    ECS_list.append(ECS_sol)
                #step D: check for E case
                if free1!=lowest_neutrino[i].level and free2!=lowest_neutrino[j].level:
                    #this condition implies that they are (at least) one common propagator
                    if free1 != 0 and free2 != 0:
                        #this condition check that the two neutrino are independant
                        ECS_sol=ECS_sector(self,'e',[lowest_neutrino[i],lowest_neutrino[j]],total_propa-3)
                        ECS_list.append(ECS_sol)                   
        
        
        return ECS_list

         
    
    def find_lowest_particle(self,neutrino=1,lowest_level=0,out_num=2):
        "find the one/two lowest (lower level) neutrino/particle, if the are ambiguity return more than two"

        if neutrino:
            list=self.neut_content
        else:
            list=self.ext_content

        #find the value of the corresponding level and after find the neutrino
        min_level1=1000
        min_level2=1000
        for neut in list:
            if(neut.level>=lowest_level and neut.level<=min_level1):
                min_level2=min_level1
                min_level1=neut.level
                if out_num==1:
                    min_level2=min_level1
            elif(neut.level>=lowest_level and neut.level<min_level2):
                min_level2=neut.level

        #return the corresponding neutrino
        lowest=[]
        for neut in list:
            if(neut.level>=lowest_level and neut.level<=min_level2):
                lowest.append(neut)

        return lowest
            
        


    #####################################################################################
    #                                  OTHER FUNCTION                                   #
    #####################################################################################


    def solve_blob_sector(self):
        """ resolve the change of variable for blob """

        for blob in self.blob_content.values():
            blob.find_solutions()

    def set_option(self,info='default'):
        """ store the different option linked to the generation of this MG_diagram """
        
        self.opt=Option(info)

        
    def __str__(self):
       try:
        text='structure of the configuration '+str(self.config)+':\n'
        for i in self.ext_content:
            text+=str(i)+'\n'
        for i in self.prop_content:
            text+=str(i)+'\n'
        text+=str(len(self.ECS_sol))+' ECS(\'s) '+str(self.num_propa)+' propagator(s) '+str(self.num_neut)+' missing particles(s)\n'
        
        text+='detail :\n'
        for ECS in self.ECS_sol:
            text+=str(ECS.chgt_var)+'('+str(len(ECS.blob_content))+')'+'\t'
        text+='\n'+str(len(self.blob_content))+' blob(s) associated\n'
        for blob in self.blob_content.values():
            text+=str(blob)

        return text
       except:
               text='not supportted string in not full mode: organize_particle_content is needed fisrt'
               return text
       
       
       
    def output_type_info(self):
        """ return output containing the number of muon/electron/jet/bjet/invisible_part """

        mu_m_list=[13]
        el_m_list=[11]
        ta_m_list=[15]
        mu_p_list=[-13]
        el_p_list=[-11]
        ta_p_list=[-15]
        photon_list = [22]

        jet_list=[1,2,3,4,21]
        bjet_list=[5]
        inv_list=[12,14,16,18] #sm 
        inv_list+=[1000012,1000014,100016,1000022,1000023,1000024,1000025,1000035,1000037] #susy
        
        content=[0    ,0        ,0        ,0        ,0        ,0        ,0        ,0        ,0       ,0]
        list=[jet_list,bjet_list,el_m_list,el_p_list,mu_m_list,mu_p_list,ta_m_list,ta_p_list,inv_list,photon_list]
        auth_sign=[1  ,1        ,0        ,0        ,0        ,0        ,0        ,0        ,1       ,0] 

        for i in range(0,len(list)):
            for particle in self.ext_content:
                if auth_sign[i] and abs(particle.pid) in list[i]:
                    content[i]+=1
                elif particle.pid in list[i]:
                    content[i]+=1  
        return content

                
                

class Option:

    def __init__(self,info='default'):
       "initialize option"
       if isinstance(info, basestring) and info!='default':
            info=MW_param.read_card(info)
       # DEFAULT VALUE:
       self.ecs_fuse=1
       self.blob_fuse=1
       self.use_sol_type_1=1
       self.use_sol_type_2=1
       self.use_sol_type_3=1
       self.max_sol=10
       self.use_ecs_a=1
       self.use_ecs_b=1
       self.use_ecs_c=1
       self.use_ecs_d=1
       self.use_ecs_e=1
       self.use_ecs_f=1
       self.foce_nwa = 1e-9

       if info=='default':
           return
       
       tag_to_genvar={'1':'self.ecs_fuse',
                      '2':'self.blob_fuse',
                      '3':'self.max_sol',
                      '4':'self.use_sol_type_1',
                      '5':'self.use_sol_type_2',
                      '6':'self.use_sol_type_3',
                      '10':'self.use_ecs_a',
                      '11':'self.use_ecs_b',
                      '12':'self.use_ecs_c',
                      '13':'self.use_ecs_d',
                      '14':'self.use_ecs_e',
                      '15':'self.use_ecs_f',
                      'force_nwa':'self.force_nwa'}

       #replace defined value:
       for key, value in info['mw_gen'].items():
           if key in tag_to_genvar:
               exec('%s = %s' % (tag_to_genvar[key],value))

       self.ecs_on=[]
       for letter in 'abcdef':
           cond='self.use_ecs_'+letter
           if eval(cond):
               self.ecs_on.append(letter)
               
       #
       self.force_nwa = max(self.force_nwa, float(info['mw_run']['nwa']))




    def __str__(self):
        text='    *** OPTION  ***\n\n'
        text='self.ecs_fuse='+str(self.ecs_fuse)+'\n'
        text+='self.blob_fuse='+str(self.blob_fuse)+'\n'
        text+='self.use_sol_type_1='+str(self.use_sol_type_1)+'\n'
        text+='self.use_sol_type_2='+str(self.use_sol_type_2)+'\n'
        text+='self.use_sol_type_3='+str(self.use_sol_type_3)+'\n'
        text+='self.max_sol='+str(self.max_sol)+'\n'
        text+='self.use_ecs_a='+str(self.use_ecs_a)+'\n'
        text+='self.use_ecs_b='+str(self.use_ecs_b)+'\n'
        text+='self.use_ecs_c='+str(self.use_ecs_c)+'\n'
        text+='self.use_ecs_d='+str(self.use_ecs_d)+'\n'
        text+='self.use_ecs_e='+str(self.use_ecs_e)+'\n'
        text+='self.use_ecs_f='+str(self.use_ecs_f)+'\n'

        return text

                

if(__name__=="__main__"):
    """test"""
    os.chdir('..')
    for i in range(1,2):
        diag=MG_diagram('./SubProcesses/MW_P_mu+mu-_mu+mu-e+e-/','param_card.dat','./Source/transfer_function/ordering_file.inc',i)
        diag.define_Constraint_sector()
        diag.solve_blob_sector()
        print diag

            


    
