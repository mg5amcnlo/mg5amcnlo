#!/usr/bin/env python
import re,sys
import avh_pc_olo

sysargv = sys.argv
srcdir = re.sub(r'avh_olo\.py','',sysargv[0])
del sysargv[0]

kindmod = ''
dpkind = ''
qpkind = ''
ddtype = ''
qdtype = ''
mptype = ''
tlevel = ''
cppintf = ''
for option in sysargv:
    if   re.match(r'kindmod',option): kindmod = re.sub(r'kindmod=','',option)
    elif re.match(r'dpkind' ,option): dpkind  = re.sub(r'dpkind=' ,'',option) 
    elif re.match(r'qpkind' ,option): qpkind  = re.sub(r'qpkind=' ,'',option) 
    elif re.match(r'ddtype' ,option): ddtype  = re.sub(r'ddtype=' ,'',option) 
    elif re.match(r'qdtype' ,option): qdtype  = re.sub(r'qdtype=' ,'',option) 
    elif re.match(r'mptype' ,option): mptype  = re.sub(r'mptype=' ,'',option) 
    elif re.match(r'tlevel' ,option): tlevel  = re.sub(r'tlevel=' ,'',option) 
    elif re.match(r'cppintf',option): cppintf = re.sub(r'cppintf=','',option) 
    else:
        avh_pc.prnt('')
        avh_pc.prnt('usage: python avh_pc_olo.py [dpkind="kind"]')
        avh_pc.prnt('                            [qpkind="kind"]')
        avh_pc.prnt('                            [kindmod="module"]')
        avh_pc.prnt('                            [ddtype="type"]')
        avh_pc.prnt('                            [qdtype="type"]')
        avh_pc.prnt('                            [mptype="type"]')
        avh_pc.prnt('                            [tlevel="yes"]')
        avh_pc.prnt('                            [cppintf="yes"]')
        avh_pc.prnt('       If no option is provided, dpkind="kind(1d0)" is set')
        avh_pc.prnt('       by default. However, dpkind must be provided expli-')
        avh_pc.prnt('       citly if to be combined with qpkind and/or mptype.')
        avh_pc.prnt('       Setting any option to "" is equivalent to omitting.')
        avh_pc.prnt('')
        sys.exit()

filename = avh_pc_olo.full( srcdir ,dpkind,qpkind,kindmod
                                   ,ddtype,qdtype,mptype
                                   ,tlevel,cppintf )
