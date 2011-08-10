

import re
import sys
import Cards

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


    def __init__(self,proc_card,cond='',ParticlesFile=''):
       process_line,multi=self.read_proc_card(proc_card,cond)
       self.decay_diag=self.pass_in_pid(process_line,multi)
       if ParticlesFile is not None:
           self.ParticlesFile=ParticlesFile #avoid multiple load of this file

    def read_proc_card(self,proc_card,cond=''):
        """ read the information of the proc_card """

# Define Process

        ff=open(proc_card,'r')
        #read step by step 
        #   1) find the begin of the definition of process
        #   2) read all process
        #   3) find multiparticle
        
        #   1) find the begin of the definition of process

        tag_proc=re.compile(r'''#\s*Begin\s+PROCESS\s*#''',re.I)
        while 1:
            line=ff.readline()
            if line=='':
                sys.exit('bad proc_card information: missing BEGIN PROCESS tag')

            if tag_proc.search(line):
                break #(pass to step 2)

        #   2) read all process

        done=re.compile(r'''done''')
        end_coup=re.compile(r'''end_coup''')
        if cond:
            process=re.compile(r'''>.*@\s*'''+cond,re.DOTALL)
        else:
            process=re.compile(r'''>''')

        while 1:
            line=ff.readline()
            if line=='':
                sys.exit('bad proc_card information: missing request process tag (end of file)')
            if '#' in line:
                line = line.split('#')[0]

            if done.search(line):
                if cond=='0':
                    ff.close()
                    return self.read_proc_card(proc_card)
                sys.exit('bad proc_card information: missing request process tag')
            
            if process.search(line):
                process_line=line
                break #(pass to step 3)


        #   3) find multiparticle
        begin_multi=re.compile(r'''#\s*Begin\s+MULTIPARTICLES\s*#''',re.I)
        end_multi  =re.compile(r'''#\s*End\s+MULTIPARTICLES\s*#''',re.I)
        info       =re.compile(r'''^(?P<tag>[\S]+)\s+(?P<multi>[\S]*)''')
        multi={}
        in_multi_area=0
        while 1:
            line=ff.readline()
            if line=='':
                sys.exit('bad proc_card information: missing multiparticle tag')       
            if  end_multi.search(line):
                break

            if begin_multi.search(line):
                in_multi_area=1
                continue
            
            if in_multi_area:
                if info.search(line):
                    info2= info.search(line)
                    multi[info2.group('tag').lower()]=info2.group('multi')


        

        return process_line,multi

            
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
    Decay_info('../Cards/proc_card.dat')
    
