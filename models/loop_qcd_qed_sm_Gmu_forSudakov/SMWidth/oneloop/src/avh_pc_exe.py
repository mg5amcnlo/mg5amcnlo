#!/usr/bin/env python

import sys
import avh_pc

ninput = len(sys.argv)-1
lines = open(sys.argv[ninput]).readlines()

if   sys.argv[1] == 'case': avh_pc.case(sys.argv[2],sys.argv[3],lines)
elif sys.argv[1] == 'dich': avh_pc.dich(sys.argv[2],sys.argv[3],lines)
elif sys.argv[1] == 'incl': avh_pc.incl(sys.argv[2],lines)
elif sys.argv[1] == 'subs': avh_pc.subs(sys.argv[2],sys.argv[3],lines)
elif sys.argv[1] == 'clean': avh_pc.clean(lines)
else:
    print 'ERROR in avh_pc_exe.py: option '+sys.argv[1]+' not defined.'
    sys.exit

iofile = open(sys.argv[ninput],'w')
iofile.writelines(lines)
iofile.close()
