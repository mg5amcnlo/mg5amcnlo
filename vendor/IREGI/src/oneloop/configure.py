#!/usr/bin/env python
import re,os

def replace(match,old,new,filename):
    lines = open(filename).readlines()
    nn = 0
    for line in lines:
        if re.match(match,line):
            lines[nn] = re.sub(old,new,line)
            break
        nn = nn+1
    iofile = open(filename,'w')
    iofile.writelines(lines)
    iofile.close()

srcdir = os.path.abspath('./src')

replace('^srcdir *='
       ,'^srcdir *=.*' ,'srcdir = \''+srcdir+'\''
       ,'create.py')
