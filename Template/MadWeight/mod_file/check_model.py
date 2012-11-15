#!/usr/bin/env python

#test

#Extension
import string
import os
import sys
import time
import re
import stat
from mod_file import *


#############################################################################
##                          function
############################################################################

def go_to_main_dir():
    """ move to main position """
    
    list_dir=os.listdir('./')
    if 'bin' in list_dir:
        return
    else:        
        os.chdir(os.pardir)  
    
    list_dir=os.listdir('./')
    if 'bin' in list_dir:
        return
    else:
        print 'script must be launched from main or bin/Source directory'
        sys.exit()
            

#############################################################################
##                          Main
############################################################################
if(__name__=="__main__"):
    go_to_main_dir()
    #supress open_file subroutine  from couplings.f if defined
    rem_open_file("Source/MODEL/couplings.f","Source/MODEL/couplings_new.f")
    os.system(  "mv Source/MODEL/couplings_new.f   Source/MODEL/couplings.f")
    #same for couplingsValues.f 
    if os.path.exists("Source/MODEL/couplingsValues.f") :
        rem_open_file("Source/MODEL/couplingsValues.f","Source/MODEL/couplingsValues_new.f")
        os.system(  "mv Source/MODEL/couplingsValues_new.f   Source/MODEL/couplingsValues.f")

    #treat special case of lha_reading.f
    if not os.path.exists("Source/MODEL/lha_reading.f"):
        os.system("cp Source/MadWeight_File/copy_file/lha_reading.f Source/MODEL/")
    else:
        fuse_f77_files(["Source/MODEL/lha_reading.f","Source/MadWeight_File/copy_file/lha_reading.f"],"Source/MODEL/lha_reading_new.f")
        os.system("mv Source/MODEL/lha_reading_new.f   Source/MODEL/lha_reading.f")
    mod_model_make("Source/MODEL/makefile")
