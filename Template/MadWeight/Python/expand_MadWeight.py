#!/usr/bin/env python

# Module
import string
import os
import sys
import re
import shutil
import mod_file
from MW_param import go_to_main_dir

def expand_all(echap=[]):
    go_to_main_dir()
    print "copying files"
    copy_file(echap)
    del_file()
    print "modifying files"
    opt={}
    opt['nowarning']="""['DESACTIVATE_CUT','DESACTIVATE_BW_CUT','get_user_params','main_make','obj_for_MW']"""
    mod_file.mod_file('./Source/MadWeight/mod_file/MW_pos',opt=opt)
    #mod_all_file()



def copy_file(echap=[]):

    #pattern for two column in a line

    Pattern=re.compile(r'''^\s*(?P<file>\S+)\s+(?P<pos>\S+)\s*$''')
    init_pos=os.path.abspath('./Source/MadWeight/MWP_template/')+'/'
    ff=open(init_pos+'/MW_pos','r')

    while 1:
        line=ff.readline()
        if line=='':
            break
        if line[0]=='#':
            continue
        obj_pat=Pattern.search(line)
        if obj_pat:
            if obj_pat.group('file') in echap:
                continue

#            print "copy ",obj_pat.group('file')
            if '_card.dat' not in obj_pat.group('file'):
                os.system('ln -sf '+init_pos+obj_pat.group('file')+' '+obj_pat.group('pos'))
            else:
                shutil.copy(init_pos+obj_pat.group('file'),obj_pat.group('pos'))


def del_file():
    del_list=['./bin/split_banner.pl']
    for file in del_list:
        if file!='' and '*' not in file:
            os.system('rm '+ file)
                
if(__name__=="__main__"):
    "expand the MadWeight program"
    
    expand_all()
