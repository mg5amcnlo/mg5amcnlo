

import re
import sys
import string


try: 
    import Cards
    import madgraph.various.misc as misc
except ImportError:
    import internal.madweight.Cards as Cards
    import internal.misc as misc
    
class Decay_info:
    """ all routine linked to the reconaissance of the topology from the proc card
        
        The proc-card has two information:
            1) the process decay pp>(t>blvl~)(t~>(W->l~vl)b)
            2) the MultiParticle content

        The diagram is treated as follow:
            1) We don't consider production part (i.e. how the first on-shell particle are produced)
            2) We will check mass shell possibility for all 1->2 decay
            3) We will not consider possibility of mass shell for 1->3,4,...
                  even if they are only one possibility or one possibility is dominant
    """


    def __init__(self,current_dir,cond='',ParticlesFile=''):
        """we need to read information from leshouche.inc and from configs.inc"""

        mglabel2pid_list=Cards.read_leshouches_file(current_dir+'/leshouche.inc')
        #      create a dict:

        mglabel2pid_dic={}
        for index, pid in enumerate(mglabel2pid_list):
            mglabel2pid_dic[index+1]=pid

       
        topology=self.read_config(current_dir+'/configs.inc', mglabel2pid_dic)

        #process_line,multi=self.read_proc_card(proc_card,cond)
        #self.decay_diag=self.pass_in_pid(process_line,multi)
        self.decay_diag=self.decay_structure(topology,mglabel2pid_dic)

        if ParticlesFile is not None:
            self.ParticlesFile=ParticlesFile #avoid multiple load of this file

    def read_config(self, file_name, mglabel2pid_dic):
        print file_name
        trappe=open(file_name, 'r')
        buff=trappe.readline()
        res_patern=re.compile(r'''^\s*(?P<mg_id>[\-,\d*]*)\s*(?P<pid_d1>[\-,\d*]*)\s*(?P<pid_d2>[\-,\d*]*)\s*(?P<mass>[\_a-zA-Z0-9]*)\s*(?P<width>[\_a-zA-Z0-9]*)\s*(?P<SorT>[a-zA-Z]*)\s*(?P<pid_m>[\-,\d*]*)''',re.I)

        topo={}
        while 1:
            buff=trappe.readline()
            if buff.find('*')>-1: 
                return topo
            elif buff!="":        
                if res_patern.search(buff): 
                    mg_id=res_patern.search(buff).group('mg_id')
                    pid_d1=int(res_patern.search(buff).group('pid_d1'))
#             if pid_d1>0: pid_d1=mglabel2pid_dic[pid_d1]
                    pid_d2=int(res_patern.search(buff).group('pid_d2'))
#             if pid_d2>0: pid_d2=mglabel2pid_dic[pid_d2]
                    mass=res_patern.search(buff).group('mass')
                    width= res_patern.search(buff).group('width')
                    SorT= res_patern.search(buff).group('SorT')
                    pid_m= res_patern.search(buff).group('pid_m')
                    topo[int(mg_id)]={}
                    topo[int(mg_id)]['daughters']=[int(pid_d1),int(pid_d2)]
                    topo[int(mg_id)]['mass']=mass
                    topo[int(mg_id)]['width']=width
                    topo[int(mg_id)]['channel']=SorT
                    topo[int(mg_id)]['pid']=int(pid_m)
                else: 
                    print "error: unexpected format in configs.inc "


    def decay_structure(self,topo,mglabel2pid_dic):
        """translate topological info in 'topo' and in 'mglabel2pid' into the object decay_diag """

        decay_diag=[]
        res_nb=len(topo)

        decay_item={}
        for res_label in range(-1,-len(topo)-1,-1):
           decay_item[res_label]=Proc_decay([topo[res_label]['pid']])
           for daughter_id in topo[res_label]['daughters']:
             if daughter_id>0 : decay_item[res_label].des.append(Proc_decay([int(mglabel2pid_dic[daughter_id])],res_label))
             else: 
                decay_item[daughter_id].mother=res_label
                decay_item[res_label].des.append(decay_item[daughter_id])

#       now decay item is a dictionnary with 
#                   each key = one of the legs in the diagrams
#           associated value = decay chain initiated by this leg

#       the list "decay_diag" should contains the values associated with the legs
#       originating from the production part         

#       here I use the criteria written by Olivier, so that "proc_list"="decay_list" later on in the code

        particles_from_HI=[]
        list_external=[]
        for leg in range(-len(decay_item.keys()),0):
            #print "  "
            #print "leg "
            #print leg
            #print "pid "
            #print decay_item[leg].pid
            #print "mother"
            #print decay_item[leg].mother
            #print "channel"
            #print topo[leg]['mass']

            if topo[leg]['daughters'][0]>2 and topo[leg]['daughters'][0] not in list_external:
                list_external.append(topo[leg]['daughters'][0])
            if topo[leg]['daughters'][1]>2 and topo[leg]['daughters'][1] not in list_external:
                list_external.append(topo[leg]['daughters'][1])

            if  topo[leg]['mass']=='ZERO':
                decay_item[leg].mother=0  # off-shell gluons/photons/quarks should not be part of the decay chain
                decay_item[leg].des[0].mother=0
                decay_item[leg].des[1].mother=0 
                if topo[leg]['daughters'][0]>2 and topo[leg]['daughters'][0] not in particles_from_HI :particles_from_HI.append(topo[leg]['daughters'][0]) 
                if topo[leg]['daughters'][1]>2 and topo[leg]['daughters'][1] not in particles_from_HI :particles_from_HI.append(topo[leg]['daughters'][1]) 
                continue  

            # consider T-channel here
            if topo[leg]['channel']=='T':
                if topo[leg]['daughters'][0]>2 and topo[leg]['daughters'][0] not in particles_from_HI  :particles_from_HI.append(topo[leg]['daughters'][0])
                if topo[leg]['daughters'][1]>2 and topo[leg]['daughters'][1] not in particles_from_HI :particles_from_HI.append(topo[leg]['daughters'][1])
            # I also need to check if we have A(virtual) > A(real) + B 
            if decay_item[leg].pid[0]==decay_item[leg].des[0].pid[0] or decay_item[leg].pid[0]==decay_item[leg].des[1].pid[0]:
                decay_item[leg].des[0].mother=0
                decay_item[leg].des[1].mother=0
                if topo[leg]['daughters'][0]>2 and topo[leg]['daughters'][0] not in particles_from_HI  :particles_from_HI.append(topo[leg]['daughters'][0]) 
                if topo[leg]['daughters'][1]>2 and topo[leg]['daughters'][1] not in particles_from_HI  :particles_from_HI.append(topo[leg]['daughters'][1]) 
                continue
            if decay_item[leg].mother==0 and topo[leg]['channel']=='S':
                particles_from_HI.append(leg)
            elif topo[leg]['channel']=='S':
                if topo[decay_item[leg].mother]['channel']=='T' or decay_item[decay_item[leg].mother].pid[0]==21:
                   particles_from_HI.append(leg)

        # now check if all external particles have been scanned:
        for index in  mglabel2pid_dic.keys():
          if index >2 and index not in list_external:
              particles_from_HI.append(index) 
        

        for leg in particles_from_HI:
          if leg<0:
            decay_diag.append(decay_item[leg])
          else:
            temp=Proc_decay(pid=[int(mglabel2pid_dic[leg])], mother=0)
            decay_diag.append(temp)
        return decay_diag

            
    def pass_in_pid(self,process_line,multi):
        """ convert information in pid information """
        
        if hasattr(self,'ParticlesFile'):
            ParticlesFile=self.ParticlesFile
        else:
            ParticlesFile=Cards.Particles_file('./Source/MODEL/particles.dat')
        pid=ParticlesFile.give_pid_dict()
        
        #
        # Update information with multi_particle tag
        #
        for couple in multi.items():
            text=couple[1]
            tag=couple[0]
            pid_list=[]
            len_max=3
            key_list=pid.keys()
            while text:
                text,add=self.first_part_pid(text,pid)
                pid_list+=add

            pid.update({tag:pid_list})



        #
        #   pid list is now complete
        #   convert line in decay pid information
        decay_rule=[]
        #1) take only the decay part:
        for letter in ['$','/','\\','@','#','\n']:
            if letter in process_line:
                process_line=process_line[:process_line.index(letter)]
                #break # only one such symbol to signify the end of the decay part
        process_line=process_line[process_line.index('>')+1:]

        
        decay_diag=[]
        level_decay=0
        while process_line:
            if process_line[0] in [' ', '\t']:
                process_line=process_line[1:]
                continue
            if process_line[0]=='>':
                process_line=process_line[1:]
                continue            

            if process_line[0]=='(':
                process_line=process_line[1:]
                level_decay+=1
                new_decay=1
                continue
            
            if process_line[0]==')':
                level_decay-=1
                current_part=current_part.mother
                process_line=process_line[1:]
                continue


            process_line,pid_content=self.first_part_pid(process_line,pid)

            if level_decay==0 or (level_decay==1 and new_decay):
                new_decay=0
                part=Proc_decay(pid_content)
                decay_diag.append(part)
                current_part=part
            elif new_decay:
                new_decay=0
                part=current_part.add_desint(pid_content) #return new part
                current_part=part
            else:
                current_part.add_desint(pid_content)


        return decay_diag


    def first_part_pid(self,text,pid):
        """find the pid(s) of the fist tag in text.
           return the text without this tag and the pid.
           pid is a dictonary
        """

        len_max=4
        key_list=pid.keys()
        while 1:
            num=min(len_max,len(text))
            if len_max==0:
                sys.exit('error pid dico not complete or invalid input :'+str([text[:min(3,len(text))]])+'\
                          \n Complete proc_info.py')
                
            if text[:num].lower() in key_list:
                tag=text[:num].lower()
                text=text[num:]
                return text, pid[tag]
            else:
                len_max+=-1


    def __str__(self):
        """ print information """

        text=""
        for particle in self.decay_diag:
            text+=str(particle)
            text+='\n'


        return text       




class Proc_decay:
    """ little class to store information of decay from the proc card """


    def __init__(self,pid,mother=0):
        """ init particle saying from which proc_decay the particle is coming"""
        self.mother=mother
        self.pid=pid
        self.des=[]

    def add_desint(self,pid):
        new=Proc_decay(pid,self)
        self.des.append(new)
        return new

    def __str__(self):
        """ print """

        text='('+str(self.pid)+',['
        for particle in self.des:
            text+=str(particle)
        text+='])'


        return text


if __name__=='__main__':
    "test"
    Decay_info('../Cards/proc_card_mg5.dat')
    
