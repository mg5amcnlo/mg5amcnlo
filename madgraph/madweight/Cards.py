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
##   last-modif:04/08/08                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##                                                                      ##
##      + create_include_file                                           ##
##      + Card                                                          ##
##      |    + init                                                     ##
##      |    |    +   detect_type                                       ##
##      |    + read_card                                                ##
##      |    + read_ident                                               ##
##      |    + read_particles                                           ##
##      |    |    +   give_pid_dict                                     ##
##      |    + create_include_file                                      ##
##      |    |    +   pass_in_type                                      ##
##                                                                      ##
##########################################################################
import re
import os
import math
import sys
import logging

logger = logging.getLogger('madgraph.madweight')
pjoin = os.path.join

class FileInputException(Exception): pass

#1 #########################################################################
def create_include_file(MWparam):
    """ create all the include file and make all the link """

    #create include file corresponding to MW_card and transfer_card
    #charge card
    ident=Card('./Cards/ident_mw_card.dat')
    if MWparam:    madweight=MWparam.mw_card
    else: madweight=Card('./Cards/MadWeight_card.dat')
    transfer=Card('./Cards/transfer_card.dat')

    #create output
    madweight.create_include_file(ident,'./Source/madweight_card.inc')
    transfer.create_include_file_tf(ident,'./Source/MadWeight/transfer_function/')

    #make all the symbolic link
    for dir in MWparam.MW_listdir:
        os.system('ln -sf ../../Source/madweight_card.inc ./SubProcesses/'+dir+'/madweight_card.inc')
        os.system('ln -sf ../../Source/MadWeight/transfer_function/transfer_card.inc ./SubProcesses/'+dir+'/transfer_card.inc')



#1 #########################################################################
class Card(dict):
    """ The routine to read and treat all the card (but the run_card.dat)"""
    
    #2 #########################################################################
    def __init__(self,file,type=''):
        
        dict.__init__(self)
        self.info = self #retro compatibility
        
        self.file=file
        self.charged = 0 # most of the file are automaticaly read, but not all
        
        if not type:
            self.type=self.detect_type()
        else:
            self.type=type.lower()

        if self.type in ['transfer','param','madweight']:
            self.read(self.file)
        elif(self.type in ['ident']):
            self.read_ident()
        
        

    #3 #########################################################################           
    def detect_type(self):
        """detect the type of the card """
        
        return self.file.split('/')[-1].split('_')[0].lower()
        
        
    #2 #########################################################################
    def read_card(self,name_card):
        """ DEPRECIATED """
        print "warning depreciated funtion read_card in use"
        raise Exception
        self.read(name_card)
        
    #2 #########################################################################
    def read(self,name_card):
        """put all card information in a dictionary"""
        
        self.charged=1

        p_block=re.compile(r'''^block\s+(?P<block>\w*)\s*(?P<comment>.*)''',re.I)
        
        get_comment = False
        if 'madweight' in name_card.lower():
            get_comment = True

        try:
            card=open("./Cards/"+name_card,'r')
        except:
            try:
                card=open(name_card,'r')
            except:
                card=open("./Events/"+name_card,'r')  #->read banner

        #init output
        info = self
        self['comment'] = {}
        name_block=''

        #read the card
        while 1:
            line=card.readline()
            if line=='':
                break
            prov=[name_block]

            if p_block.search(line):
                name_block=p_block.search(line).group('block')
                name_block=name_block.lower()
                info[name_block]={}
                info['comment'][name_block]=p_block.search(line).group('comment')
                continue

            if '#' in line:
                line_content, comment= line.split('#',1)
            else: 
                line_content = line
                comment = ''
            line_content = line_content.strip()
            comment = comment.strip()
            if not line_content:
                continue
            
            #deal with line as   queue '-o test -d u'
            if "'" in line_content:
                begin = line_content.find("'")
                end = line_content.rfind("'")
                line_content = line_content[:begin].split() + [line_content[begin+1:end]] + line_content[end+1:].split()
            else:
                line_content = line_content.split()
            #treat decay anomaly
            line_content[0] = line_content[0].lower()
            if line_content[0].lower()=='decay':
                name_block='decay'
                line_content=line_content[1:]
                if 'decay' not in info.keys():
                    info['decay']={}
                decay_tag=line_content[0]
            elif name_block in ['decay','br']:
                name_block='br'
                line_content=[decay_tag]+line_content[2:]+[line_content[0]]
                if 'br' not in info.keys():
                    info['br']={}
                    
            #create list of dictionary
            obj=line_content[-1]
            for i in range(-2,-len(line_content)-1,-1):
                obj={line_content[i]:obj}
            #put in final data
            dico=info[name_block]
            if 'comment' not in dico and get_comment:
                dico['comment'] = {}
            if get_comment:
                comments = dico['comment']
            
            if len(line_content)==1:
                dico[' ']=obj
            for i in range(0,len(line_content)-1):
                if line_content[0] == 'comment':
                    continue
                if line_content[i] not in dico.keys():
                    dico[line_content[i]]=obj[line_content[i]]
                    if get_comment and line_content[i] != 'comment':
                        comments[line_content[i]] = comment
                    break
                elif i!=len(line_content)-2:
                    dico=dico[line_content[i]]
                    if get_comment and line_content[i] != 'comment':
                        comments[line_content[i]] = comment
                    obj=obj[line_content[i]]
                elif(type(dico[line_content[i]])==list):                         #multiple definition of the same input
                    dico[line_content[i]].append(obj[line_content[i]])
                    if get_comment and line_content[i] != 'comment':
                        comments[line_content[i]] += comment
                else:
                    dico[line_content[i]]=[dico[line_content[i]],line_content[i+1]]
                    if get_comment and line_content[i] != 'comment':
                        comments[line_content[i]] += comment
        return info

    #2 #########################################################################
    def read_ident(self):
        """ read ident file only four and five column format are supported yet """
        
        self.charged=1
        try:
            ff=open(self.file,'r')
        except:
            sys.exit('FATAL ERROR: no file ./Cards/ident_mw_card.dat\n This File is created during the definition of the transfer function\n\
            either newprocess was not perform after ./bin/PassToMadWeight, either your transfer function is not correctly set up. launch\n\
            $>./bin/change_tf.py to solve that problem')
        line_format=re.compile(r'''^\s*(?P<block>\w+)\s+(?P<tag>\w+)\s+(?P<name>\w+)\s+(?P<type>\w+)\s*(?P<default>[\w.+-]*)\s*$''')
        ident = self
        while 1:
            line=ff.readline()
            if line=='':
                break
            words=line_format.search(line)
            if not words:
                continue
            block=words.group('block').lower()
            tag=words.group('tag').lower()
            name=words.group('name').lower()
            type=words.group('type').lower()
            value=words.group('default').lower()

            if not ident.has_key(block):
                ident[block]={tag:[name,type,value]}
            else:
                ident[block][tag]=[name,type,value]
                
        return ident
    


    
   


    #2 #########################################################################
    def create_include_file(self,card,output):
        """ create an output of type name=value from part common in both card
            be careful of the restriction in the ident file -> 4 column (only one tag) or 5 (one tag+default value)

            Default value are used only if the block is defined in the card but not the entry
        """
        
        
        logger.debug('create file %s' % output)
        out=file(output,'w')
        out.writelines('C automatic include file for card '+self.file+' and '+card.file+'\n\n')

        if card.type=='ident':
            ident=card.info
            info=self.info
        elif self.type=='ident':
            info=card.info
            ident=self.info
        
        for block in info.keys(): 
            if ident.has_key(block):
                for tag in ident[block].keys():
                    if info[block].has_key(tag):
                        value=self.pass_in_type(info[block][tag],ident[block][tag][1])
                        out.writelines('        '+ident[block][tag][0]+' = '+str(value)+'\n')
                    elif ident[block][tag][2]: #check if default value is defined
                        value=self.pass_in_type(ident[block][tag][2],ident[block][tag][1])
                        out.writelines('        '+ident[block][tag][0]+' = '+str(value)+'\n')

    #2 #########################################################################
    def create_include_file_tf(self,card,output):
        """ create an output of type name(I)=value from part common in both card
            be careful of the restriction in the ident file -> 4 column (only one tag) or 5 (one tag+default value)

            Default value are used only if the block is defined in the card but not the entry
        """
        
        
        logger.debug('create file %s' % output)
        out=file(output+'transfer_card.inc','w')
        out.writelines('C automatic include file for card '+self.file+' and '+card.file+'\n\n')

        if card.type=='ident':
            ident=card.info
            info=self.info
        elif self.type=='ident':
            info=card.info
            ident=self.info
        
        nb_element = 0    
        
        #template = "        DATA ( %s(I), I=1,nb_tf) / %s /\n"
        template = "        %s(%s) = %s \n"    
        for block in info.keys(): 
            if ident.has_key(block):
                for tag in ident[block].keys():
                    type_format = ident[block][tag][1]
                    if info[block].has_key(tag):
                        values = info[block][tag]
                    else:
                        values = [ident[block][tag][2]] * max(1, nb_element)

                    # convert the input in the fortran format
                    if isinstance(values, list):
                        values = [str(self.pass_in_type(value, type_format)) for value in values]
                    else:
                        values = [str(self.pass_in_type(values, type_format))]
                    # check that all element have the same number of input
                    if not nb_element:
                        nb_element = len(values)
                    elif nb_element != len(values):    
                        print nb_element, len(values)
                        raise Exception, 'All input in tranfer_card.dat should have the same number of element'

                    # add the line
                    out.writelines(''.join(template % (ident[block][tag][0], i+1,val)
                                           for i,val in enumerate(values)))
                    out.writelines('\n')
                    
                    
                    
                    #if info[block].has_key(tag):
                    #    value=self.pass_in_type(info[block][tag],ident[block][tag][1])
                    #    out.writelines('        '+ident[block][tag][0]+' = '+str(value)+'\n')
                    #elif ident[block][tag][2]: #check if default value is defined
                    #    value=self.pass_in_type(ident[block][tag][2],ident[block][tag][1])
                    #   out.writelines('        '+ident[block][tag][0]+' = '+str(value)+'\n')
        if nb_element ==0:
            nb_element = 1          
        fsock = open(pjoin(output,'nb_tf.inc'),'w')
        fsock.write("""
        integer nb_tf
        parameter (nb_tf=%i)
        """ % nb_element)
        


    #3 #########################################################################
    def pass_in_type(self,value,type):
        """ value convertisor """
        if type=='logical':
            if value in ['F','f','0', 0, False]:
                return '.false.'
            elif value in ['T','t','1',1, True]:
                return '.true.'
            else:
                return value
        elif type in ['integer','string']:
            return value
        elif type in ['double','real']:
            value = str(value)
            value=value.replace('d','e')
            value=value.replace('D','e')
            value=float(value)
            value='%(test)e' % {'test':value}
            value=value.replace('e','d')
            return value
        else:
            print 'error in type for',value,type
            
    #3 #########################################################################
    def write(self, output):

        if isinstance(output, str):
            output = open(output, 'w')
                
        header = """##########################################################################
##                                                                      ##
##                             MadWeigth                                ##
##                             =============                            ##
##                                                                      ##
##                              Run control                             ##
##                              -----------                             ##
##                                                                      ##
##                                                                      ##
##    Author: Mattelaer Olivier (UIUC)                                  ##
##            Artoisenet Pierre (NIKHEF)                                ##
##                                                                      ##
##    Version:     5.0.0                                                ##
##    Last change: 27/03/13                                             ##
##                                                                      ##
##########################################################################
##                                                                      ##
##  This Card defines all specific parameters of Madweight              ##
##                                                                      ##
##########################################################################
"""
        output.write(header)
        
        for key in self:
            if key == 'comment':
                continue

            output.write('\n\nBlock %s\n' % key)
            for key2, value in self[key].items():
                if key2 == 'comment':
                    continue
                if isinstance(key2,tuple):
                    key2 = '%s'.join(map(str, key2))
                comment = ''
                try:
                    comment = self[key]['comment'][key2] 
                except Exception:
                    pass
                    
                if not isinstance(value, list):
                    value = [value]
                for value in value:
                    output.write('    %s   %s    #%s\n' % (key2,value,comment))
        
        
        
        
        




class Particles_file(Card):
    """ all routine linked to particles.dat file """
    
    #2 #########################################################################
    def __init__(self,file='./Source/MODEL/particles.dat',type=''):
        
        self.charged = False
        if not os.path.exists(file):
            self.v4model = False
            #UFO model
        else:
            self.v4model = True
            Card.__init__(self, file, type)

    #2 #########################################################################
    def read(self):
        """read a particles.dat file (don't read multiple info now)"""

        self.charged=1
        if self.v4model:
            particle_pattern=re.compile(r'''^\s*
                                            (?P<part>[\w+-~]{1,4})\s+
                                            (?P<partx>[\w+-~]{1,5})\s+
                                            (?P<spin>[SFVT])\s+
                                            (?P<LineType>[WSDC])\s+
                                            (?P<mass>\w+)\s+
                                            (?P<width>\w+)\s+
                                            (?P<color>[STO])\s+
                                            (?P<label>[\w+-~]{1,5})\s+
                                            (?P<pid>[\d-]*)\s*$''',re.VERBOSE)
            ff=open(self.file,'r')
    
            particle=[]
            while 1:
                line=ff.readline()
                if line=='':
                    break
                pat_particle=particle_pattern.search(line)
                if pat_particle:
                    particle.append(list(pat_particle.groups()))
                
            self.info={'particles':particle} 
        else:
            # UFO MODEL
            try:
                import internal.ufomodel.particles as particles
            except Exception:
                sys.path.append(pjoin(os.getcwd(), 'bin'))
                import internal.ufomodel.particles as particles
            
            v4_part_info = []
            done = []
            for p in particles.all_particles:
                if p.pdg_code in done:
                    continue
                done += [p.pdg_code,-1*p.pdg_code]
                info = [p.name, 
                        p.antiname,
                        p.spin,
                        p.line,
                        p.mass,
                        p.width,
                        p.color,
                        p.name,
                        p.pdg_code]
                v4_part_info.append([str(i) for i in info])
            self.info={'particles':v4_part_info}
        
        
    #3 #########################################################################
    def give_pid_dict(self):
        """ return a list of pid for each tag -d'ont treat multiple tag-"""
 
        if not self.charged:
            self.read() #not automaticly read for the moment
        pid={}
        for data in self.info['particles']:
            
            if data[0]==data[1]:
                pid.update({data[0]:[int(data[-1])]})
            else:
                pid.update({data[0]:[int(data[-1])],data[1]:[-1*int(data[-1])]})

        return pid
        #return {'e-': [11], 'a': [22], 've': [12], 'e+': [-11], 'w-': [-24], 'w+': [24], 'vm': [14], 'd~': [-1], 'b': [5], 'vt': [16], 'c~': [-4], 'ta-': [15], 'b~': [-5], 'ta+': [-15], 't~': [-6], 'vm~': [-14], 'c': [4], 'u~': [-2], 'd': [1], 'g': [21], 've~': [-12], 'h': [25], 's~': [-3], 's': [3], 'u': [2], 't': [6], 'mu-': [13], 'z': [23], 'mu+': [-13], 'vt~': [-16]}

    #3 #########################################################################
    def give_mass_width_dict(self):
        """ return two dict {pid:fortran_mass_code} and {pid:fortran_width_code}
            pid value are always positive
        """
        
        
        
        if not self.charged:
            self.read() #not automaticly read for the moment
        dict_mass={}
        dict_width={}
        for data in self.info['particles']:
            pid=abs(int(data[-1]))
            mass=data[-5]
            width=data[-4]
           
            # Adding model prefixing
            if mass.lower() != 'zero':
                dict_mass[pid]='mdl_%s' % mass
            else: 
                dict_mass[pid] = str(mass)
            if width.lower() != 'zero':
                dict_width[pid] = 'mdl_%s' % width
            else:
                dict_width[pid] =  str(width)
             
#        return {1: 'ZERO', 2: 'ZERO', 3: 'ZERO', 4: 'ZERO', 5: 'MB', 6: 'MT', 11: 'ZERO', 12: 'ZERO', 13: 'ZERO', 14: 'ZERO', 15: 'MTA', 16: 'ZERO', 21: 'ZERO', 22: 'ZERO', 23: 'MZ', 24: 'MW', 25: 'MH'}, {1: 'ZERO', 2: 'ZERO', 3: 'ZERO', 4: 'ZERO', 5: 'ZERO', 6: 'WT', 11: 'ZERO', 12: 'ZERO', 13: 'ZERO', 14: 'ZERO', 15: 'ZERO', 16: 'ZERO', 21: 'ZERO', 22: 'WA', 23: 'WZ', 24: 'WW', 25: 'WH'}

        return dict_mass,dict_width 

    #3 #########################################################################
    def give_pid_to_label(self):
        """ return a dict {pid:label} 
        """


        return {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 11: 'e', 12: 've', 13: 'mu', 14: 'vm', 15: 'ta', 16: 'vt', 21: 'g', 22: 'A', 23: 'Z', 24: 'W', 25: 'h'}
#        if not self.charged:
#            self.read() #not automaticly read for the moment
#        out={}
#        for data in self.info['particles']:
#            pid=abs(int(data[-1]))
#            label=data[-2]
            
#            out[pid]=label
            
            

        return out 

def read_leshouches_file(filepos):
    """ read a leshouches.inc file and returns [list of pid] in the MG order """


#    pid_pat=re.compile(r'''^\s*DATA\s+\(IDUP\(I,\s*\d+\s*\),I=\d,\s*\d+\s*\)/(?P<pid>[\-,\s\d]*)/''',re.I)
    pid_pat=re.compile(r'''\s*DATA\s*\(IDUP\(I,\s*\d+\s*,\s*\d+\s*\),I=\d,\s*\d+\s*\)/(?P<pid>[\-,\s\d]*)/''',re.I)


#    mass_pat=re.compile(r'''^\s*pmass\(\s*(?P<MG_id>\d+)\s*\)\s*=\s*(?P<mass>\w+)''',re.I)   
    
    dict_pid={}
    ff=open(filepos,'r')
    while 1:
        line=ff.readline()
        #print line, [line[4],line[5], line[6]]

        if line=='':
            raise FileInputException, 'incorrect leshouche.inc at position '+filepos
        if line[5] != ' ':
            line = old_line.rstrip() + line[6:]
        if pid_pat.search(line):
            pid_list=pid_pat.search(line).group('pid')
            pid_list=pid_list.replace(',',' ').split() #pass to a list of pid
            ff.close()
            break
        old_line = line
    
    return pid_list

