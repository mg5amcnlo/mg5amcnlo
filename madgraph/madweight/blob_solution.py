#!/usr/bin/env python

try:
    import madgraph.madweight.particle_class as particle_class
    import madgraph.various.misc as misc
except ImportError:
    import internal.madweight.particle_class as particle_class
    import internal.misc as misc
Particle = particle_class.Particle

import sys

class Level_generation:
    """ define generic tool for maintained how to generate a sequential change of variable
        this class in only defined in order to have common routine for ECS_sector and Blob-solution
    """

    def __init__(self,tag,def_step=0):
        """init  generic variable """
        self.num_fuse=0
        self.sol_tag=tag
        
        if def_step:
            if type(def_step)==list:
                self.step=def_step
            else:
                self.step=[def_step]
        else:
            self.step=[]




    def define_fuse_region(self,fuse_list,output_mode=1):
        """ define a fuse part(Block '2' but with new particle) for the particle in the list
            output_mode=1: return the particle
            output_mode=2: return the particle and the unaligned propagator
        """
        lowercase='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        fuse_list_=list(fuse_list)

        unaligned=[]
        while 1:
            if len(fuse_list_)==1:
                break
                if output_mode==1:     return fuse_list_[0]
                else: return fuse_list_[0],unaligned
            else:
                part1=fuse_list_.pop(0)
                part2=fuse_list_.pop(0)
                fuse_particle=Particle(lowercase[self.num_fuse],0,fuse=1)
                fuse_particle.level=part1.level
                if part1.level<part2.level:
                    fuse_particle.mother=part1.mother
                    unaligned.append(part2.mother)
                else:
                    fuse_particle.mother=part2.mother
                    unaligned.append(part1.mother)
                self.num_fuse+=1
                fuse_list_.append(fuse_particle)
                
                if(self.sol_tag=='blob'):
                    Block_B(self,'3',[part1,part2],fuse_particle)                   
                else:
                    Block_sector(self,'3',[part1,part2],fuse_particle)

        if output_mode==1:     
            return fuse_list_[0]   #if we are out of the loop this means that the fuse is complete and fuse has only one element
        else: 
            return fuse_list_[0],unaligned            

    def order_block(self,main_sec):
        """ (re)ordrer so that the new order follow the generation order """
        
        new_order=[]
        #ready: particle who can enter in change of variable
        ready=[]
        ready+=main_sec.ext_content
        control=0 #to avoid infinite loop
        #print 'enter in ordering block routine'
        while 1:
            if len(self.step)==0:#  or self.step==[blob_sec.main]: 
                #print 'reordering complete'
                break
            control+=1
            if control>len(self.step)+1: #check if we have looked at all possibility without finding solution
                sys.exit('ERROR: infinite loop detected in Level_generation.order_block()')   
            #print 'ready status:',
            #for particle in ready:
            #    print particle.MG,
            #print '\ncontinue: block still to order ',len(self.step),':',
            #for block in self.step:
            #    print block
            #print
            block=self.step.pop(0)
            for particle in block.in_part:
                if particle not in ready:
                    self.step.append(block)
                    suceed=0
                    #print 'failed', particle.MG
                    break
                else:
                    suceed=1

            if not suceed:
                continue
            #print 'suceed'
            control=0
            #pass here only if the block can be generated at this step
            new_order.append(block)
            for particle in block.out_part:
                ready.append(particle)
            for particle in block.in_part:
                ready.remove(particle) #to speed the search in ready
        
        self.step=new_order


        

class Blob_solution(Level_generation):
    """ store the information of how to generate a blob """

    def __init__(self,blob_sec,solution=''):
        """ initialize how_gen: dict: Particle-> related block
                       step: list of all needed block in the correct order
            update link with the related blob
        """
        #standard-simple tool
        Level_generation.__init__(self,'blob')
        #link with the blob
        self.blob_sec=blob_sec
        blob_sec.solution.append(self)
        #global option
        self.opt=blob_sec.opt
        
        self.unaligned=[]

        #look if we have an original copy
        if solution:
            self.copy_solution(solution)
        else:
            self.how_gen={}
            for part in blob_sec.content:
                self.how_gen[part]='' #normaly block
                


                
            
    def copy_solution(self,solution):
        """ duplicate solution (no alias use) in order to modify it later """
        
        
        self.num_fuse=solution.num_fuse
        self.how_gen={}
        self.ready_to_enter=list(solution.ready_to_enter)
        
        self.num_fuse=solution.num_fuse        
        self.step=list(solution.step)
        self.how_gen=dict(solution.how_gen)
        self.unaligned=list(solution.unaligned)
        
        a=self.step.pop(0)
        if self.step==solution.step:
            sys.exit('stop same object in step')
        self.step.insert(0,a)
        self.how_gen[1]=1
        if self.how_gen==solution.how_gen:
            sys.exit('stop same object in how_gen')
        del self.how_gen[1]


        
        

    def find_all_solutions(self,blob_sec):
        """ (blob)->None : complete blob_sec.sol

            * find tree solutions:
                1) this solutions maximize the number of propagator generated following BW
                2) this solutions maximize the number of external particle generated following tf (i.e. all execpt neutrino)
                3) this solutions try to be an intermediate solution between the two
            * In practise, we start to construct a 'beta' solution, which one will identify globaly the structure of the solution
                The beta try to made the more local change, and don't consider E case (but have E* tag for an D which can pass in E)
            * Secondly we will apply modification to this beta, to pass in each solution case
                The first solution will be the first one with improvment in 'A' case and will restore 'E'(and extend E) case
                the second one will change all 'D','E' in '1', with little modification in 'A' case
                The third solution will be essentially the beta without 'E' (only D for visible case)"""
        
        #start with beta solution
        self.find_beta_solution(blob_sec)
        self.order_block(blob_sec)
        #copy solution
        if self.opt.use_sol_type_1:
            sol1=Blob_solution(blob_sec,self)
        if self.opt.use_sol_type_2:
            sol2=Blob_solution(blob_sec,self)
        if self.opt.use_sol_type_3:
            sol3=self #no need to define new thing we can simply upgrate this solution
                      #be careful that a sol3=Blob_solution(blol_sec,self) (in place of sol3=self) is wrong
                      #it will add a new solution for the blob=> 4 solutions for the blob and not 3
                      #  (in this case beta -self- stay a solution)
        if self.opt.use_sol_type_1:
            sol1.pass_in_solution(1)
        if self.opt.use_sol_type_2:            
            sol2.pass_in_solution(2)
        if self.opt.use_sol_type_3:
            sol3.pass_in_solution(3)      # update 'self' solution after the other solution,
        else:                             # if we don't want to have this solution->remove from up level
            blob_sec.solution.remove(self)# don't remove completely this solution it's the 'generic' one for the other

        if not (self.opt.use_sol_type_1 or self.opt.use_sol_type_2 or self.opt.use_sol_type_3):
            sys.exit('FATAL ERROR: At least one solution for Blob generation must be authorized')
           
        ##########################################################################################
        ##                              HOW FIND BETA SOLUTION                                  ##
        ##########################################################################################        

    def find_beta_solution(self,blob_sec):
        """ find a first solution to resolve the blob.
            The idea is to take the more local possibility in all case
            this is perhaps not the best solution but it's only the beginning of the resolution
            Secondly we will not defined 'final' block but some more generic one (like E*: restriction at one level of E)
            """
        
        self.ready_to_enter=list(blob_sec.ext_content) 
        self.ready_to_enter.reverse()
        step=0 #security to supress infinite program
        while 1:
            step+=1
            if step>40:
                sys.exit('ERROR: infinite loop detected in Blob_solution.find_beta_solution()')
            if len(self.ready_to_enter)<2:
                break
            #DEBUG
            #print 'continue loop content:',
            #for part in self.ready_to_enter:
            #    print part.MG,
            #print
            part1=self.ready_to_enter.pop(0)
            #print 'test on :',part1.MG, '(value:',self.how_gen[part1],')',
            part2=part1.twin
            #print 'with :',part2.MG, 
            if part2 not in self.ready_to_enter:
                #print 'failed'                
                self.ready_to_enter.append(part1)
                continue
            #print 'suceed',
            solution=self.resolve_piece(blob_sec,[part1,part2])
            #print solution.chgt_var
            
        #treat one particle case
        if self.how_gen[blob_sec.main]=='':
            #print 'main not defined-> pass in 0'
            Block_B(self,'0',[blob_sec.main],[])
        
        
    def resolve_piece(self,blob_sec,in_part):
        "find the 'first' solution for this piece"
        
        #usefull particle:
        mother1=in_part[0].mother
        mother_twin=mother1.twin
        
        #print  'resolve piece',in_part[0].MG,in_part[1].MG
        #check case 2
        if  not( in_part[0].external or in_part[1].external):
            #two propagator already generated =>sum them=>block '2'
            result=Block_B(self,'2',in_part,mother1) #take care of all dependencies
            return result
        
        #print 'pass step 1'
        self.sol_type=''
        #check neutrino case (A,B,C)
        for particle in in_part:
            if particle.neutrino:
                #print 'neutrino detect'
                #first generate the other particle following tf function
                twin=particle.twin
                #now look how to generate this neutrino
                motherX=particle
                mother_list=[]
                for X in range(1,4): #check the tree propa before the neutrino
                    #print 'motherXbut1',motherX.MG
                    motherX=motherX.mother
                    if motherX==0:
                        #print 'Not_exist'
                        stop='Not_exist'
                        break
                    #print 'motherX',motherX.MG
                    if blob_sec.contains_particle(motherX):
                        if self.how_gen[motherX]=='':
                            mother_list.append(motherX)
                            #print mother_list[-1].pid,mother_list[-1].mother
                            if mother_list[-1].twin!=0 and mother_list[-1].twin.neutrino:
                                #check for following neutrino configuration
                                stop='neut'
                                break
                        else:
                            #print self.how_gen[motherX]
                            #print 'already defined'
                            stop=self.how_gen[motherX]
                            break
                    else:
                        #print 'not exist'
                        stop='Not_exist'
                        break
                #SO we have information to create the block
                if len(mother_list)==0:
                    Block_B(self,'0',[particle],[])
                elif len(mother_list)==1:
                    block=Block_B(self,'C',[particle,twin],[mother_list[0]])
                    block.give_stop_reason(stop)
                    self.treat_conflicting_block(block) #try to pass block in 'A' (but pass the 'A' block in 'B')
                elif len(mother_list)==2:
                    block=Block_B(self,'B',[particle,twin,mother_list[0].twin],[mother_list[-1]])
                    block.give_stop_reason(stop)
                elif len(mother_list)==3:
                    block=Block_B(self,'A',[particle,twin,mother_list[0].twin,mother_list[1].twin],[mother_list[-1]])
                return block #end neutrino case
            
        #print 'pass step 2'
        #check for one visible particle and a propagator case (1,D)
        if (not (in_part[0].external and in_part[1].external)):
            if in_part[0].external:
                particle=in_part[0]
                propa=in_part[1]
            else:
                particle=in_part[1]
                propa=in_part[0]
            if particle.tf_level<2:
                #must use TF function
                result=Block_B(self,'1',in_part,[mother1])
            else:
                #transfert are authorized
                result=Block_B(self,'D',in_part,[mother1])
            return result#end visible+propa case
        
        #check for  double visible case (1,D,E)
        #check if the width of the two visible particle autorize E/D
        #print 'pass double visible case'
        if in_part[0].tf_level<2 and in_part[1].tf_level<2:
            #no transfer are authorized => 1
            result=Block_B(self,'1',in_part,[mother1])
        elif in_part[0].tf_level<2 or in_part[1].tf_level<2:
            #no transfer authorized for 0(1) but for 1(0) => D
            #print 'no transfer authorized for 0(1) but for 1(0) => D'
            result=Block_B(self,'D',in_part,[mother1])
        #Now transfer authorized for both =>check for E
        elif not(in_part[0].mass and in_part[1].mass):
            #mass condition for E fullfilled. store an E* information
            #We don't create E block at this level, the extension of this block can cause inefficiency in some case
            result=Block_B(self,'E*',in_part,[mother1])#special case of E
        else:
            #mass condition for E failed -> D
            #print 'mass condition for E failed -> D'
            result=Block_B(self,'D',in_part,[in_part[0].mother])
        return result #end of double visible case


    def treat_conflicting_block(self,block):
        """ if a 'C' block is block by a 'A' block it's sometimes more powerful to degrade the 'A' in 'B'
            if the 'C' can be upgrated to 'A'"""
        
        #CHECK IF WE CAN DO SOMETHING
        if block.stop!='A': 
            return
        #block was stop by another change of variable ->try to improve that

        #check if the third level if free
        try:
            mother1=block.out_part[0].mother
            mother2=mother1.mother
            if self.how_gen[mother2].chgt_var not in ['','0','1','2','D']:
                return
        except:
            return

        #PASS ALL CONDITION
        #so pass block A->B
        blockA=self.how_gen[mother1]
        #change block data
        new_in=blockA.in_part   #be careful this is only a alias!
        new_in.remove(block.out_part[0])
        blockA.redefine_block(self,'B',new_in,out_part[0].twin)
        #pass this block -> A
        new_in=block.in_part+[out_part[0].twin,mother1.twin]
        block.redefine_block(self,'A',new_in,mother2)

        ##########################################################################################
        ##                          BETA UPGRATING FUNCTION                                     ##
        ##########################################################################################

        
    def pass_in_solution(self,tag):
        """ tag -> upgrate solution in order to fullfill taged criteria 
                upgrate 'beta' solution following some criteria(tagged with tag)"""

        #print 'input', self
        if tag==1:
            self.sol_type=1
            self.extend_block('E*')
            if self.opt.blob_fuse:
                self.extend_block('D')
                self.extend_block('A')
        elif tag==2:
            self.sol_type=2
            self.convert_DE_in_1()
            if self.opt.blob_fuse:
                self.extend_block('A')
            
        elif tag==3:
            self.sol_type=3
            self.convert_E_in_D()
        #print 'ouput',self
        #print 'pass in solution ',tag    
        #be sure that the order is still uptodate
        self.order_block(self.blob_sec)
        for block in self.step:
            block.def_order_content()

    def convert_DE_in_1(self):
        """ convert all 'E*' and all 'D' in '1' block """
        
        #find  blok 'E*,D-> pass to '1'
        all_block=list(self.step)
        for block in all_block:
            if self.blob_sec.content[0].width < self.opt.force_nwa:
                continue
                    
            if block.chgt_var in ['E*','D']:
                block.change_chgt_var(self,'1')
                

    def convert_E_in_D(self):
        """ convert all 'E*' in 'D' """
        
        # find  blok 'E*-> pass to 'D'
        for block in self.step:
            if block.chgt_var in ['E*']:
                block.change_chgt_var(self,'D')    


    def extend_block(self,chgt_var):
        """ put the A/D/E propagator in place to minize problem with thin transfer function"""

        num_propa={'A':3,'D':1,'E*':2} #maximum number of aligned propa
        num_propa=num_propa[chgt_var]
        new_chgt_var={'A':'A','D':'D','E*':'E'}
        new_chgt_var=new_chgt_var[chgt_var]
        
        #print '*** check how gen',self.sol_type,'for change',chgt_var,' *** number of block in step',len(self.step)
        
        #step 1: find a blok 'chgt_var'
        #step 2: find the area where we have freedom
        #           * define a list of delta-thin width associated to external particle
        #           * define which one to take (first condition thiner and second nearest of the neutrino)
        #step 3: define the solution:
        #           * We will limit (for the moment) to move up to one propagator.
        #           * We move only propagator before a large tf (of course) and the one (if many) with largest width
        #               for the propagator
        
        #step 1: find  blok 
        step2=0
        all_block=list(self.step)
        for block in all_block:
            step2+=1
            if block.chgt_var!=chgt_var:
                continue
            if block not in self.step:
                continue
            #print 'find one'
            #step 1 succeed
        #step2:find the area where we have freedom
            #initialisation of vector for width category
            thin=[] #for the three particle associated to the thiner width
            
            #look for area before 'A'/'D'/'E*-E':
            try:
                motherX=block.neut_content[0] #for 'A' like case
            except:
                motherX=block.in_part[0] #for 'E*' like case
            while 1:
                motherX=motherX.mother
                if self.blob_sec.contains_particle(motherX)==0:
                    break
                if not( self.how_gen[motherX].chgt_var in ['0','1','2','D'] or self.how_gen[motherX]==block):
                    break
                
                #print 'create thin'
                for i in range(0,len(thin)+1):
                    #print 'thin content: [',
                    #for particle in thin:
                    #    print particle.MG,',',
                    #print '] step:',i, 'num_enter',num_propa 
                    
                    if i==len(thin) and i<num_propa:
                        thin.append(motherX)
                    elif(motherX.width<thin[i].width and motherX.channel=='S'):
                        thin.insert(i,motherX)
                        if len(thin)>num_propa:
                            thin=thin[:num_propa]
                        break

                #print 'particle selected in step2',
                #for particle in thin:
                #    print [particle.MG],
                #print '|'                     
                        
            #change order of thin to be in the correct order (anti-level) for the generation
            propa_in=[thin[0]]
            for i in range(1,len(thin)):#lent(thin) can be lower than num_propa in some E* case
                for j in range(0,len(propa_in)+1):
                    if j==len(propa_in):
                        propa_in.append(thin[i])
                    elif(thin[i].level<propa_in[j].level):
                        propa_in.insert(j,thin[i])
                        break
            #end of step 2
            #OPTION -> return to first propa if asked
            if (block.chgt_var=='E*' and (not self.opt.blob_fuse) and len(propa_in)==2):
                #CASE A/D with not blob_fuse cann't pass in the full routine if blob_fuse=0
                propa_in=[block.in_part[0].mother,block.in_part[0].mother.mother]
            
            #print 'propa selected in step2.2',
            #for particle in propa_in:
            #    print particle.MG,
            #print '|'
        #step 3:define the solution
            #'update' the generation
            fuse_list=[]
            step=0
            try:
                motherX=block.neut_content[0] #for 'A' like case
            except:
                motherX=block.in_part[0] #for 'E*' like case

            part_in=[motherX]               
            while 1:
                motherXbut1=motherX
                motherX=motherX.mother
                       
                if step==len(propa_in):
                    break
                if self.blob_sec.contains_particle(motherX)==0:
                        sys.exit('ERROR: unusual error in Blob_solution.expand_block: debug this routine (error type 1)')
                if not( self.how_gen[motherX].chgt_var in ['0','1','2','D'] or self.how_gen[motherX]==block):                  
                        sys.exit('ERROR: unusual error in Blob_solution.expand_block: debug this routine (error type 2)')
         
                if motherX in propa_in:
                    step+=1
                    if self.how_gen[motherX]!=block:
                        self.how_gen[motherX].del_from_sol(self)
                    if fuse_list:
                        fuse_part=self.define_fuse_region(fuse_list)
                        part_in.append(fuse_part)
                        fuse_list=[]
                    else:
                        part_in.append(motherXbut1.twin)
                elif fuse_list:
                    self.how_gen[motherX].del_from_sol(self)
                    fuse_list.append(motherX.twin)
                else:
                    self.how_gen[motherX].del_from_sol(self)
                    fuse_list=[motherXbut1.twin,motherX.twin]
                    
            #redifine this block
            #print 'redefine_block', num_propa, len(part_in)-1           
            if num_propa==len(part_in)-1:
                block.redefine_block(self,new_chgt_var,part_in,propa_in[0])
            elif chgt_var=='E*':
                #failed to expand in E
                block.change_chgt_var(self,'D')
            else:
                sys.exit('ERROR: unusual error in Blob_solution.expand_block: debug this routine (error type 3)')



    def del_ext1_in(self,particle_list):    
        """ delete blok  '1' but only if the particle is an external one!"""

        for particle in particle_list:
            if particle.external and self.how_gen[particle].chgt_var=='1':
                 self.how_gen[particle].del_from_sol[self]



            
        ##########################################################################################
        ##                                    OTHER FUNCTION                                    ##
        ##########################################################################################

    def debug(self):
        """ function for debugging: print how_gen"""
        print "how_gen"
        list=self.how_gen.items()
        for i in range(0,len(list)):
            try:
                print '{'+str(list[i][0].MG),':'+str(list[i][1].chgt_var)+'}',
            except:
                print '}',
        print


    def __str__(self):
        text='blob generation: '
        for block in self.step:
            text+='['+block.chgt_var+'|'
            for particle in block.in_part:
                text+=str(particle.MG)+' '
            text+=':'
            for particle in block.out_part:
                text+=str(particle.MG)+' '           
            text+='] '
            #if block.stop:
            #    print block.stop,
        return text


class Block_sector:
    """define:store information on a Block.
       This is a simple version of block dedicated for ECS
       This is the basis for the blob dedicated
    """
    
    def __init__(self,sol_sec,chgt_var,input_propa,output_propa):

        self.sol_sec=sol_sec    
        self.chgt_var=chgt_var
        self.in_part=input_propa                   
        if type(output_propa)==list: #output_propa must be a list, but sometimes it's forgotten
            self.out_part=output_propa
        else:
            self.out_part=[output_propa]
        self.opt=sol_sec.opt
        self.unaligned=[]
            
        #neutrino stuff
        self.neut_content=[]
        self.num_neut=0
        for particle in input_propa:
            if particle.neutrino:
                self.neut_content.append(particle)
                self.num_neut+=1
                
        #stop reason
        self.stop=''

        #link to the solution_sector
        sol_sec.step.append(self)
        self.def_order_content()

    def give_stop_reason(self,message):
        """ store the stop information (why not a more general change of variable) """
        self.stop=message


    def def_order_content(self):
        """ define (in the corect order) the particle used in this ECS
            only for fuse sector (routine overwirtted in other case)
        """

        #normaly self.in_part are ordering in decreasing order (respected to level)
        
        if self.chgt_var=='3': #define a fuse particle
            self.order_content=self.in_part+self.out_part
            self.chgt_var='2'
            


    def __str__(self):
        """ print routine """

        text="["+str(self.chgt_var)+ ' |'
        for particle in self.in_part:
            text+=str(particle.MG)+','
        text=text[:-1]+' :'
        for particle in self.out_part:
            text+=str(particle.MG)+','
        
        return text[:-1]+']'

        
class Block_ECS(Block_sector):
    """ Define a Block dedicated to be an ECS central part"""

    def __init__(self,sol_sec,chgt_var,input_part):
        """ create a block for the dedicated ECS """
        Block_sector.__init__(self,sol_sec,chgt_var,input_part,[])
        self.def_order_content()
        self.def_unaligned()


    def def_order_content(self):
        """ define (in the correct order) the particle used in this ECS
            This routine is overwrited for block linked to blob
        """
        #first treat zero neutrino case
        if self.chgt_var=='a':
            self.order_content=self.in_part
            return

        #consider other case
        ext_content=[]
        propa_content=[]
        neut_content=[]
        self.order_content=[]

        for particle in self.in_part:
            if particle.neutrino:
                neut_content.append(particle)
                continue
        
            propa_content.append(particle.mother)       
            ext_content.append(particle)
                
        self.order_content=neut_content+ext_content+propa_content
        
        #special case: E class -> add the fisrt propa in S channel
        if self.chgt_var=='e':
            neut_in_ecs=neut_content[0]
            self.order_content.append(neut_in_ecs.all_mother()[-1])

        #print 'order_content',self.chgt_var
        #for particle in self.order_content:
        #    print particle.MG,
        #print
     
    def def_unaligned(self):
        
        if self.chgt_var=='a':
            unaligned=self.in_part[0].all_mother()
            for part in self.in_part[1].all_mother():
                if part not in unaligned:
                    unaligned.append(part)
            self.unaligned=unaligned+self.in_part
            return
        
        
        aligned=[part.mother for part in self.in_part if part.neutrino==0]
        if self.chgt_var=='a':
            aligned.append(self.order_content[-1]) # add the first propagator
        
        unaligned=[]    
        for i in range(0,self.num_neut):
            for part in self.order_content[i].all_mother():
                if part not in aligned+unaligned:
                    unaligned.append(part)    

        self.unaligned=unaligned
        
class Block_B(Block_sector):
    """ Define a Block dedicated to be in a blob"""
    
    def __init__(self,sol_sec,chgt_var,input_propa,output_propa,unaligned=[]):

        Block_sector.__init__(self,sol_sec,chgt_var,input_propa,output_propa)

        self.put_B_sol_uptodate(sol_sec) #maintain tool for beta solution and future update
        self.def_unaligned()

            
    def put_B_sol_uptodate(self,sol_sec):
        """ put the solution status uptodate with this block definition """
        
        for particle in self.in_part:
            try: #we can generate some 'not ready' variable with block A,B,E
                sol_sec.ready_to_enter.remove(particle)
            except:
                pass
            if particle.external:
                sol_sec.how_gen[particle]=self
            if particle.mother and particle.mother not in self.out_part and self.chgt_var!='0':
                if sol_sec.blob_sec.contains_particle(particle.mother):
                    sol_sec.how_gen[particle.mother]=self    
        for particle in self.out_part:
            #check that the output particle isn't already consider like an input particle somewhere else
            
            if sol_sec.blob_sec.contains_particle(particle.mother): #consider fuse case
                if sol_sec.how_gen[particle.mother]=='':
                    sol_sec.ready_to_enter.append(particle)
            sol_sec.how_gen[particle]=self


    def def_order_content(self):
        """ define (in the corect order) the particle used in this blob """

        propa_content=[]
        #normaly self.in_part are ordering in decreasing order (respected to level)
        
        if self.chgt_var=='3': #define a fuse particle
            self.order_content=self.in_part+self.out_part
            self.chgt_var='2'
        elif self.chgt_var=='0': #external particle
            self.order_content=self.in_part
        elif self.chgt_var=='E':
            propa_content=[self.in_part[0].mother,self.in_part[2].mother]
            if self.in_part[0].mass:
                self.order_content=self.in_part+propa_content
            else:
                self.order_content=[self.in_part[1],self.in_part[0],self.in_part[2]]+propa_content
        elif isinstance(self.in_part[0].MG, basestring): #check if first particle are fuse particle
            self.order_content=self.in_part+[self.in_part[0].mother]
            try:
                self.order_content.append(self.in_part[2].mother)
            except:
                pass
        elif isinstance(self.in_part[1].MG, basestring): #check if first particle are fuse particle
            self.order_content=self.in_part+[self.in_part[1].mother]
            try:
                self.order_content.append(self.in_part[2].mother)
            except:
                pass
        else:
            for particle in self.in_part :
                if particle.mother not in propa_content:
                    propa_content.append(particle.mother)
            self.order_content=self.in_part+propa_content
            

    def change_chgt_var(self,sol_sec,new_chgt_var):
        """ change the changement of variable associated """
        equivalent_class=[['E*','E','D','1','2'] #2 particle in entry one in output
                          #add a ,[] for a second type of equivalent class
                          ]#end of equivalent class

        if self.sol_sec==sol_sec:
            obj=self
        else:
            #supress old block from step
            try:
                sol_sec.step.remove(self)
            except:
                pass
            obj=Block_B(sol_sec,self.chgt_var,self.in_part,self.out_part)
            obj.def_unaligned()
                   
        for i in range(0,len(equivalent_class)):
            if obj.chgt_var in equivalent_class[i]:
                if new_chgt_var in equivalent_class[i]:
                        obj.chgt_var=new_chgt_var
                        obj.def_unaligned()
        if obj.chgt_var!=new_chgt_var:
            print 'WARNING: unexpected modification:'
            print '         pass from',[obj.chgt_var],'to',[new_chgt_var]
            print '         there are strictly non equivalent: but we go on anyway'
            obj.chgt_var=new_chgt_var               

    def def_unaligned(self):
        """ associate in self.unaligned the particle with unaligned peaks """
        
        if self.chgt_var in ['1','2']:
            self.unaligned=self.out_part[0]
        elif self.chgt_var=='E':
            self.unaligned=[self.in_part[0],self.in_part[1]]
        elif self.chgt_var=='D':
            tag1=self.in_part[0].MG
            tag2=self.in_part[1].MG
            if tag1<0:
                self.unaligned=self.in_part[1]
            elif tag2<0:
                self.unaligned=self.in_part[0]
            else:
                if tag1>tag2:
                    tag1,tag2=tag2,tag1
                self.unaligned='first_d_'+str(tag1)+'_'+str(tag2)

    def redefine_block(self,sol_sec,chgt_var,input_propa,output_propa):
        """redifine block """
        if self.sol_sec==sol_sec:
            sol_sec.step.remove(self)
            stop=self.stop
            self.__init__(sol_sec,chgt_var,input_propa,output_propa)
            self.give_stop_reason(stop)
        else:
            stop=self.stop
            #supress old block from step
            try:
                sol_sec.step.remove(self)
            except:
                pass
            new_block=Block_B(sol_sec,chgt_var,input_propa,output_propa)
            new_block.give_stop_reason(stop)
            pass


    def del_from_sol(self,sol_sec):
        """ supress corectly the block of the solutions """

        sol_sec.step.remove(self)
                
        #update the solution-link
        for particle in self.in_part:
            if particle.external:
                sol_sec.how_gen[particle]=''
            sol_sec.ready_to_enter.append(particle) #normaly useless but if we are in a new type of generation in the future
            
        for particle in self.out_part:
            sol_sec.how_gen[particle]=''
            try: 
                sol_sec.ready_to_enter.remove(particle) #idem,but normaly particle is not there.
            except:
                pass
        
        if self.sol_sec==sol_sec:
            #in this case supress the object:
            del self
        


