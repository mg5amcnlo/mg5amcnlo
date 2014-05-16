#!/usr/bin/env python 
####################################################################################################
####################################################################################################
##                                                                                                ##
##                                    MOD FILE MODULE                                             ##
##                                                                                                ##
####################################################################################################
####################################################################################################
##                                                                                                ##
##    Author: Mattelaer Olivier                                                                   ##
##    Institution: UCL-CP3                                                                        ##
##    contact: omattelaer@uclouvain.be                                                            ##
##                                                                                                ##
##    last modification: 01/06/10                                                                 ##
##    tag release:       1.4                                                                      ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
##                                         MANUAL                                                 ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
##  This module is dedicated to modified file in planified way. The file must have tag in order   ##
##     to perform the needed modifications.  All tag are in the following syntax:                 ##
##      $B$ NAME $B$ -> begin of a block to modify with RULE=NAME                                 ##
##      $E$ NAME $E$ -> end of the block to modify with RULE=NAME                                 ##
##      $B$ NAME $E$ -> replace tag by RULE=NAME                                                  ##
##  Some character are reserved:                                                                  ##
##   'S-','_' and '+'                                                                             ##
##   'S-': indicates a special tag (see later). Don't start name with this two caracter           ##
##   '_' : indicates begin of option in special tag, use freely if you are not in this case       ##
##   '+' : for splitting different option value (use freely elsewhere)                            ##
##                                                                                                ##
##  This is the special expression authorized in NAME (all start with S-)                         ##                                            
##                                                                                                ##
##  S-END:(only in rule) add the text in end (no tag needed)                                      ##
##  S-DEl:(in mod_file) supress this part of the file                                             ##
##  S-COMMENT_?: start commenting the defined region.                                             ##
##      The comment tag will be the value in place of the '?'. the '?' Can not contain '+' or '_' ##
##  S-DECOMMENT_?:(only in rule) inverse the comment procedure apply with S-COMMENT (go back to   ##
##                original line)                                                                  ##
##  S-REGEX_exp+new[+opt]: regular expression to move an expression to another.                   ##
##        "exp": a Python regular expression for the text to modify                               ##
##        "new": the new expression to write                                                      ##
##        "opt": python option for regular option (the one in re.compile)                         ##
##      if this rule appear in rule file, the modification appear in the complete file.           ##
##      no '_' or '+' are autorized in "exp" or "new". those one can be echap with '\'            ##
##                                                                                                ##
##  Priority order                                                                                ##
##     if a step define a tag that will in a future step the modification on the tag will be done ##
##     if you want to prevent this create the tag with $B-DIFF$ and with $E-DIFF$                 ##
##                                                                                                ##
##     1) we first apply the global REGEXP                                                        ##
##     2) we apply decomment module                                                               ##
##     3) we apply the modification for $B$ NAME $E$                                              ##
##     4) we apply the modification for $B$ NAME $B$ TEXT $E$ NAME $E$                            ##
##     5) we convert the $ ?-DIFF$ in normal tag                                                  ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
## Exemple of use:                                                                                ##
## 1) with a master file                                                                          ##
##                                                                                                ##
##    import mod_file                                                                             ##
##                                                                                                ##
##    mod_file.mod_file(./master_file.txt)                                                        ##
##                                                                                                ##
## 2) without master file                                                                         ##
##                                                                                                ##
##    import mod_file                                                                             ##
##                                                                                                ##
##    mod_file.mod_file(file_to_change,rule,[write=''])                                           ##
##       file_to_change: can be string or list of file                                            ##
##       rule: position of the rule-file. You can also use a list of rule files (this must have   ##
##              the same length than file_to_change list                                          ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
##  master file                                                                                   ##
##      is a file with tree column corresponding to mod_file/rule_file/write the write is         ##
##      not optional in file.  Comment start with # in this file                                  ##
##                                                                                                ##
##  rule file                                                                                     ##
##      in this file, you can defined, what are the new text to for each tag. Syntax is:          ##
##      $B$ NAME $B$                                                                              ##
##             CONTENT                                                                            ##
##      $E$ NAME $E$                                                                              ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
##   modification list:                                                                           ##
##                                                                                                ##
##     01/06/10: - make the modification inside a unknow blok                                     ##
##               - add a test suite for MadWeight case                                            ##
##                                                                                                ##
##     29/09/09: - differentiate $b$...$b$ from $B$...$B$ (gestion of end of line)                ## 
##                                                                                                ##
##     22/05/09: - add decomment option                                                           ##
##               - ensure that all end of line use tag \n                                         ##
##                                                                                                ##
##     11/11/08: - modify documentation                                                           ##
##               - authorize differate affectation with B-DIFF                                    ## 
##                                                                                                ##
##     31/01/08: - pass in object                                                                 ##
##               - add S-comment super tag                                                        ##
##               - add S-del super tag                                                            ##
##               - pass in isolated module                                                        ##
##               - create documentation                                                           ##
##               - replace tag can be inserted in a line                                          ##
##                                                                                                ##
##    23/06/08:  - add S-REGEXP super tag                                                         ##
##               - add function list                                                              ##
##               - differentiate $B$...$E$ with $b$...$e$                                         ##
##                    the first supress the full tag line                                         ##
##                                                                                                ##
####################################################################################################
##                                                                                                ##
##                                      Function                                                  ##
##                                      --------                                                  ##
##                                                                                                ##
##    mod_file                                                                                    ##
##    mod_text                                                                                    ##
##    Mod_file                                                                                    ##
##    |    + init                                                                                 ##
##    |    + mod_all_file                                                                         ##
##    |    |    +    mod_one_file                                                                 ##
##    |    |    +    mod_one_text                                                                 ##
##    |    |    +     +  treat_begin_end_line                                                     ## 
##    |    + extract_modif                                                                        ##
##    |    + return_mod_text                                                                      ##
##    |    |    +    comment_text                                                                 ##
##    |    |    +    del_text                                                                     ##
##    |    |    +    regexp_text                                                                  ##
##    |    + back_to_init_dir                                                                     ##
##    |    + go_to_main_dir                                                                       ##
##                                                                                                ##
####################################################################################################

# Module
import os
import sys
import re
import shutil
import string
from time import time

# 1 ###############################################################
def mod_file(mod_file,rule_file='',write='',opt={}):

    if rule_file: #not a master asking
        if type(mod_file)!=list:
            mod_obj=Mod_file(opt=opt)
            mod_obj.mod_one_file(mod_file,rule_file,write)
        elif type(mod_file)==list:
            #supress problem of output
            if write=='':
                write=['']*len(mod_file)
            #supress problem if only one rule_file
            if type(rule_file)!=list:
                rule_file=[rule_file]*len(mod_file)
            if type(rule_file)==str:
                mod_obj=Mod_file(rule_file=rule_file,opt=opt)
                for i in range(0,len(mod_file)):
                    mod_obj.mod_one_file(mod_file[i],mod_obj.dico,write[i])
            else:
               mod_obj=Mod_file(opt=opt)
               for i in range(0,len(mod_file)):
                   mod_obj.mod_one_file(mod_file[i],rule_file[i],write[i])        
    else:
        mod_obj=Mod_file(mod_file,opt=opt)

# 1 ###############################################################
def mod_text(text,rule_file='',write=''):

    
    mod_obj=Mod_file()
    mod_obj.file='input text'
    text=mod_obj.mod_one_text(text,rule_file,write)
    return text



# 1 ###############################################################
class Mod_file:


    nowarning=[]

    # 2 ###############################################################
    def __init__(self,main_file='',rule_file='',opt={}):
        """ start the instruction of modification present in the main file if present """

        self.d_init=os.getcwd()
        self.d_rule=os.getcwd()
        self.d_main=os.getcwd()

        self.failed=0 #tag to know if a tag is not modify

        if opt:
            for key,value in opt.items():
                exec('self.'+key+'='+str(value)+'')
        
        if main_file:
            #main_dir is the directory of the main_file
            self.d_rule=os.path.dirname(os.path.realpath(main_file))
            self.d_main=self.d_rule
            self.mod_all_file(os.path.basename(main_file))
        if rule_file:
            self.extract_modif(rule_file)            


            
    # 2 ###############################################################
    def   mod_all_file(self,rule_pos):
        """ apply modification following manager main_file """

        self.go_to_main_dir()

        #pattern for tree column in a line
        opt_pattern=re.compile(r'''^\s*(?P<opt>\S+)\s*=\s*(?P<value>\S+)''')
        Pattern=re.compile(r'''^\s*(?P<file>\S+)\s+(?P<rule>\S+)\s+(?P<write>\S*)\s*$''')
        ff=open(rule_pos,'r')

        while 1:
            line=ff.readline()
            if line=='':
                break
            if line[0]=='#':
                continue
            obj_opt=opt_pattern.search(line)
            if obj_opt:
                if obj_opt.group('opt')=='main_dir':
                    self.d_main=os.path.join(self.d_main,obj_opt.group('value'))
                    self.go_to_main_dir()
                                         
            obj_pat=Pattern.search(line)
            if obj_pat:
                self.mod_one_file(obj_pat.group('file'),obj_pat.group('rule'),obj_pat.group('write'))

        self.back_to_init_dir()

    # 3 ###############################################################
    def  mod_one_file(self,mod_file,rule_file,write=''):
        """ modify the file mod_file with rule_file instruction output will ba place in write (same file by default)"""
        start=time()
        self.go_to_main_dir()
        self.file=mod_file

        ff=open(mod_file,'r')
        text=ff.read()
        ff.close()
        if write:
            self.mod_one_text(text,rule_file,write)
        else:
            self.mod_one_text(text,rule_file,mod_file)
        self.back_to_init_dir()
        stop=time()
        #print 'time used to modifiy file:',stop-start,'s'
        return

    # 3 ###############################################################
    def  mod_one_text(self,text,rule_file='',write=''):
        """ modify the text with rule_file instruction output will be place in write (same file by default)"""
    
        self.go_to_main_dir()

        #print "modify ",mod_file.split('/')[-1]
        if rule_file=='':
            dico=self.dico
        elif type(rule_file)!=dict:
            self.extract_modif(rule_file)
        else:
            dico=rule_file
            self.dico=dico
#        print "dico", self.dico.keys()

        begin_end=re.compile(r'''\$(?P<maj>B)\$\s?(?P<tag>\S+)\s?\$B\$(?P<text>.*)\$E\$\s?(?P=tag)\s?\$E\$''',re.S+re.I)    # $B$ TAG $B$ TEXT $E$ TAG $E$
        end_file=re.compile(r'''\$\$\s*END\s+FILE\s*\$\$''')     # $$ END FILE $$
        replace=re.compile(r'''\$(?P<maj>B)\$[ \t]*(?P<tag>\S+)[ \t]*\$E\$''',re.I)  # $B$ TAG $E$
        end_begin=re.compile(r'''\$(?P<maj>E)\$\s?(?P<tag>\S+)\s?\$E\$(?P<text>.*)\$B\$\s?(?P=tag)\s?\$B\$''',re.S+re.I)    # $E$ TAG $E$ TEXT $B$ TAG $B$ -> in case of multiple tag

        ##treat global regexp
        for key in self.dico.keys():
            if key.startswith('S-REGEXP'):
                text=self.return_mod_text(key,text)

        ##treat decomment module
        for key in self.dico.keys():
            if key.startswith('S-DECOMMENT_'):
                text=self.return_mod_text(key,text)
                
        ##treat replacment
        text_list=replace.split(text)
        text_to_write=text_list.pop(0)
        while len(text_list)>1:
            maj=text_list.pop(0).isupper()
            tag=text_list.pop(0)
            text_to_write2=self.return_mod_text(tag,'')
            text_to_write3=text_list.pop(0)
            text_to_write=self.treat_begin_end_line(maj,text_to_write,text_to_write2,text_to_write3)
            
        ##treat block part
        text_list=begin_end.split(text_to_write)
        text_to_write=text_list.pop(0)
        multiple=0 #control
        while len(text_list)>2:
            maj=text_list.pop(0).isupper()
            tag=text_list.pop(0)
            text=text_list.pop(0)
            if end_begin.search(text) and end_begin.search(text).group('tag')==tag:
                mod_text=self.treat_multiple_tag(text,maj,tag)
            else:
                mod_text=self.return_mod_text(tag,text)
            text_next=text_list.pop(0)
            text_to_write=self.treat_begin_end_line(maj,text_to_write,mod_text,text_next)
            

        ## treat end file:
        if self.dico.has_key("S-END"):
            if not end_file.search(text_to_write):
                text_to_write+=self.dico["S-END"]

        ## restore diff affectation
        text_to_write=text_to_write.replace('$B-DIFF$','$B$')
        text_to_write=text_to_write.replace('$E-DIFF$','$E$')

        ##check that only one type of end of line is in use
        text_to_write=text_to_write.replace('\r\n','\n')

        ##write output
        if write:
            ff=open(write,'w')
            ff.writelines(text_to_write)
            ff.close()

        self.back_to_init_dir()
        return text_to_write

    # 4 #########################################################################
    def treat_begin_end_line(self,clearline,text_before,text,text_after):


        if clearline and not self.failed:
            output=text_to_write=text_before[:text_before.rfind('\n')]+'\n'
            output+=text
            output+='\n'+text_after[text_after.find('\n'):]
        else:
            output=text_before+text+text_after

        self.failed=0
        return output

    # 4 #########################################################################
    def treat_multiple_tag(self,text,maj,tag):
        end_begin=re.compile(r'''\$E\$\s?(?P<tag>\S+)\s?\$E\$(?P<text>.*)\$(?P<maj>B)\$\s?(?P=tag)\s?\$B\$''',re.S+re.I)    # $E$ TAG $E$ TEXT $B$ TAG $B$ -> in case of multiple tag

        split_text=end_begin.split(text)
        text1=split_text.pop(0)
        tag=split_text.pop(0)
        mod_text=self.return_mod_text(tag,text1)
        text_next=split_text.pop(0)
        text_next=self.mod_one_text(text_next)
        text_to_write=self.treat_begin_end_line(maj,'',mod_text,text_next)
        maj=split_text.pop(0)
        text2=split_text.pop(0)
        mod_text=self.return_mod_text(tag,text2)
        text_to_write=self.treat_begin_end_line(maj,text_to_write,mod_text,'')

        return text_to_write

        

        
        
            #############################################################################
            #                          Extract rule information                         #
            #############################################################################
            
    # 2 ###############################################################
    def extract_modif(self,rule_file):
        """put the information in a dictionary"""
        try:
            ff=open(rule_file,'r')
        except:
            ff=open(os.path.join(self.d_rule,rule_file),'r')
        begin=re.compile(r'''^\$B\$\s?(?P<tag>\S+)\s?\$B\$''')
        end=re.compile(r'''^\$E\$\s?(?P<tag>\S+)\s?\$E\$''')
        comment=re.compile(r'''^##\**\s*$''')
        special_begin=re.compile(r'''^\$(?P<tag>S-\S+)-B\$''')
        special_end=re.compile(r'''^\$(?P<tag>S-\S+)-E\$''')
        special=re.compile(r'''^\$(?P<tag>S-\S+)\$''')
        self.dico={}
        tag=""
        replace_text=""
        rec_mode=0

        while 1:
            line=ff.readline()
            if line=='':
                break
            if comment.search(line):
                continue
            if special.search(line):
                tag=special.search(line).group('tag')
                self.dico[tag]=''
            if begin.search(line) or special_begin.search(line):
                try:
                    tag=begin.search(line).group('tag')
                except:
                    tag=special_begin.search(line).group('tag')                
                if rec_mode:
                    print 'error in ',rule_file,' wrong termination for ',tag,' rule'
                    sys.exit()
                rec_mode=1
                continue        
            if end.search(line) or special_end.search(line):
                try:
                    tag=end.search(line).group('tag')
                except:
                    tag=special_end.search(line).group('tag')             
                if rec_mode==0:
                    print 'error in ',rule_file,'no initial tag:', tag
                    sys.exit()
                #detect one-line replacment => supress blank and '\n'
                if replace_text.count('\n')==1:
                    replace_text=replace_text[:-1]
                    while replace_text.endswith(' '):
                        replace_text=replace_text[:-1]
                self.dico[tag]=replace_text
                tag=""
                replace_text=""
                rec_mode=0
                continue

            if rec_mode:
                replace_text+=line

        if rec_mode:
            print 'error in ',rule_file,' wrong end-file termination '
            sys.exit()
        return self.dico


                                  
            #############################################################################
            #                               tag treatment                               #
            #############################################################################

    # 2 ###############################################################            
    def return_mod_text(self,tag,text):
            """ by default return the text linked to tag
                special tag are S-TAG_OPT: OPT=OPT1+OPT2+OPT3+..."""
            
            special_tag=re.compile(r'''S-(?P<tag>[^ \t\n\r\f\v_]+)_?(?P<opt>[^\t\n\r\f\v]*)''') # S-TAG_OPT

            if not special_tag.search(tag):
                try:
                    return self.dico[tag]
                except:
                    if tag not in self.nowarning and self.nowarning != 'all':
                        print 'WARNING: tag:',tag,' not defined in file ',self.file
                        print 'no modification done for this tag'
                    if text:
                        output = '$B$ '+tag+' $B$'+ self.mod_one_text(text)+' $E$ '+tag+' $E$'
                    else:
                        output = '$B$ '+tag+' $E$'
                    self.failed=1
                    return output
                    
            #SPECIAL TAG CASE
            short_tag=special_tag.search(tag).group('tag')
            opt=special_tag.search(tag).group('opt').split('+')
            #be sure that some split are not with a echap tag
            old=''
            opt2=[]
            for part in opt:
                if len(part) and part[-1]=='\\' :
                    old=part[:-1]+'+'
                else:
                    opt2.append(old+part)
                    old=''
            opt=opt2
            
            tag=short_tag.lower()
            if tag=='comment':
                text=self.comment_text(text,opt[0])
            elif tag=='del':
                text=self.del_text(text)
            elif tag=='regexp':
                if len(opt)==2:
                    text=self.regexp_text(text,opt[0],opt[1])
                elif len(opt)==3:
                    text=self.regexp_text(text,opt[0],opt[1],opt[2])
            elif tag=='decomment':
                text=self.decomment_text(text,opt[0])
                    
            return text
            
    # 3 ###############################################################
    def comment_text(self,text,comment_tag):
        """ add comment_tag before each line """
        end_line=re.compile(r''' ''')
        #print [text]
        #print [text.split('\n')]
        #print text
        #print text.replace('\n','\n'+comment_tag+'\t')
        
        text=comment_tag+'|\t'+text.replace('\n','\n'+comment_tag+'|\t')
        if text[-3:]=="|\t\n":
            text=text[-3:]
        text=text.replace('\t','    ')      
        text2=''
        for line in text.split('\n'):
            if line=='':
                continue
            if len(line)<74:
                if line[-1]=='\n':
                    line=line[:-1]
                for i in range(len(line),73):
                    line+=' '
                line+='|\n'
            else:
                line+='\n'
            text2+=line
            
        line=comment_tag+'+'+71*'-'+'+\n'

        return line+text2+line+'\n'

    # 3 ###############################################################
    def decomment_text(self,text,comment_tag):
        """ remove comment inserted by comment_text """

        carac_line=re.compile(comment_tag+'\+'+71*'-'+'\+')

        def decomment_line(line,comment_tag):
            if line[:6]==comment_tag+'|    ':
                line=line[6:]
            else:
                print [line[:6]]
                print 'failed decomment'

            if line[-1]=='|':
                line=line[:-1]
            return line

        decomment=0
        init_text=text.split('\n')
        end_text=''
        for line in init_text:
            if carac_line.search(line):
                decomment=not decomment
                if decomment:
                    end_text+=comment_tag+' $B-DIFF$ S-COMMENT_'+comment_tag+' $B-DIFF$\n'
                    continue
                else:
                    end_text+=comment_tag+' $E-DIFF$ S-COMMENT_'+comment_tag+' $E-DIFF$\n'
                    continue
            if decomment:
#               end_text+=line+'\n'
                end_text+=decomment_line(line,comment_tag)+'\n'
            else:
                end_text+=line+'\n'

        return end_text
        return end_text
    
    # 3 ###############################################################
    def del_text(self,text):
        return ''

    # 3 ###############################################################
    def regexp_text(self,text,exp,new,opt=''):
        """ replace the text exp (python regular expression) with new"""

        ### step A) remove escape '_' and '+'
        ### step B) apply the modification
        ### step C) return new text

        ## Step A: remove escape '_' and '+'
        exp=exp.replace('\\\\','@888@') #stupid security against string like (\\_)
        exp=exp.replace('\_','_').replace('\+','+')
        exp=exp.replace('@888@','\\\\') #end of the trick


        # Step B: apply the modification
        pattern=re.compile(exp,eval(opt))
        text=pattern.sub(new, text)

        # Step C: return
        return text

            ############################################################################
            #                          positioning routine                             #
            ############################################################################
            
    # 2 ###############################################################            
    def back_to_init_dir(self):
        os.chdir(self.d_init)
        
    # 2 ###############################################################        
    def go_to_main_dir(self):
        os.chdir(self.d_main)

  
#########################################################################################################
#  TEST #################################################################################################
#########################################################################################################
if '__main__' == __name__:

    import sys
    sys.path.append('./Source/MadWeight/Python')
    import create_run
    import unittest
    import os, shutil
    

    class TestMod_file(unittest.TestCase):
        """ Test the the mod routines works correctly on MadWeight """

        def setUp(self):
            """ create a copy of the original file """
            shutil.copyfile('../Template/SubProcesses/cuts.f', './SubProcesses/cuts.bk')

        def tearDown(self):
            os.system('rm -f ./SubProcesses/cuts.mod')
            os.system('rm -f ./SubProcesses/cuts.bk')
            os.system('rm -f ./SubProcesses/cuts.o')


        def test_cuts(self):
            """ test if we can activate/desactivate the cuts """

            self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)

            file_to_mod='./SubProcesses/cuts.bk'
            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            #modify file                                            
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            file_to_mod='./SubProcesses/cuts.mod'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        def test_mw_cuts(self):

            file_to_mod ='./SubProcesses/cuts.bk'
            rule= './Source/MadWeight/mod_file/mod_cuts'
            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)            
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            file_to_mod='./SubProcesses/cuts.mod'
            rule = './Source/MadWeight/mod_file/suppress_cuts_MW'

            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        def test_P_BW_cuts(self):

            self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)

            file_to_mod='./SubProcesses/cuts.bk'
            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            #modify file                                                                                                                                                                   
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            file_to_mod='./SubProcesses/cuts.mod'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())
 
            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            # Next one will Fail but is not supose to be called whitout check of the second 
            #rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            #mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            #self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            #self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)


        def test_MW_BW_cuts(self):

            self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)

            file_to_mod='./SubProcesses/cuts.bk'
            rule= './Source/MadWeight/mod_file/mod_cuts'
            mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            file_to_mod='./SubProcesses/cuts.mod'
            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            #modify file                                                                                                                                                                   
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())
            
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

            rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
            mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
            self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
            self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
            self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())



    unittest.main()


