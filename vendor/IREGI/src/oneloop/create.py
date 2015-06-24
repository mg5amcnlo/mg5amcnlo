#!/usr/bin/env python
import re,string,os,sys,subprocess
locdir,thisFile = os.path.split(os.path.abspath(sys.argv[0]))
srcdir = os.path.join(locdir,'src')
sys.path.append(srcdir)
import avh_pc,avh_pc_olo

dpkind  = ''
qpkind  = ''
kindmod = ''
ddtype  = ''
qdtype  = ''
mptype  = ''
tlevel = 'no'
cppintf = 'no'
ddf90mod = ''
mpf90mod = ''

lines = open('makefile_config','r').readlines()
for line in lines:
    line = line.rstrip('\n')
    if   re.match(r'^ *FC'      ,line): 
        fc       = re.sub(r'^ *FC *= *'      ,'',line)
    elif re.match(r'^ *#FC'     ,line):
        # it will be used in compile.py of madgraph
        fc       = os.environ['FC']
        
    elif re.match(r'^ *FFLAGS'  ,line): fflags   = re.sub(r'^ *FFLAGS *= *'  ,'',line)
    elif re.match(r'^ *DPKIND'  ,line): dpkind   = re.sub(r'^ *DPKIND *= *'  ,'',line)
    elif re.match(r'^ *QPKIND'  ,line): qpkind   = re.sub(r'^ *QPKIND *= *'  ,'',line)
    elif re.match(r'^ *KINDMOD' ,line): kindmod  = re.sub(r'^ *KINDMOD *= *' ,'',line)
    elif re.match(r'^ *DDTYPE'  ,line): ddtype   = re.sub(r'^ *DDTYPE *= *'  ,'',line)
    elif re.match(r'^ *QDTYPE'  ,line): qdtype   = re.sub(r'^ *QDTYPE *= *'  ,'',line)
    elif re.match(r'^ *MPTYPE'  ,line): mptype   = re.sub(r'^ *MPTYPE *= *'  ,'',line)
    elif re.match(r'^ *TLEVEL'  ,line): tlevel   = re.sub(r'^ *TLEVEL *= *'  ,'',line)
    elif re.match(r'^ *CPPINTF' ,line): cppintf  = re.sub(r'^ *CPPINTF *= *' ,'',line)
    elif re.match(r'^ *DDF90MOD',line): ddf90mod = re.sub(r'^ *DDF90MOD *= *','',line)
    elif re.match(r'^ *MPF90MOD',line): mpf90mod = re.sub(r'^ *MPF90MOD *= *','',line)

ddf90mod = re.sub(r'^~',os.getenv("HOME"),re.sub(r'\/*$','/',ddf90mod))
mpf90mod = re.sub(r'^~',os.getenv("HOME"),re.sub(r'\/*$','/',mpf90mod))

avh_pc.prnt(sys.argv[0]+': creating source file...')
sourcef = avh_pc_olo.full( srcdir ,dpkind,qpkind,kindmod 
                                  ,ddtype,qdtype,mptype
                                  ,tlevel,cppintf )

if len(sys.argv) > 1:
    if sys.argv[1] == 'source':
        sys.exit()
    elif sys.argv[1] == 'dynamic':
        avh_pc.prnt(sys.argv[0]+': compiling source file...')
        subprocess.call([fc,'-O','-fPIC','-c','avh_olo.f90','-o','avh_olo.o'])
        avh_pc.prnt(sys.argv[0]+': building dynamic library...')
        subprocess.call([fc,'-shared','-o','libavh_olo.so','avh_olo.o'])
        sys.exit()
    elif sys.argv[1] != 'static':
        avh_pc.prnt('ERROR in '+sys.argv[0]+': don\'t know how to create '+sys.argv[1])
        sys.exit()

objects = re.sub(r'\.f90','.o',sourcef)
library = 'lib'+re.sub(r'\.f90','.a',sourcef)

head = [fc]+string.split(fflags)+['-c']
tail = [sourcef,'-o',objects]

if ddtype+qdtype+mptype == '':
    avh_pc.prnt(sys.argv[0]+': compiling source file...')
    subprocess.call(head+tail)
    avh_pc.prnt(sys.argv[0]+': building library...')
    subprocess.call(['ar','cru',library,objects])
    subprocess.call(['ranlib',library])
#    subprocess.call(['gfortran','-shared','-o','libavh_olo.so','avh_olo.o'])

elif ddtype+qdtype+mptype == 'ddfun90':
    avh_pc.prnt(sys.argv[0]+': compiling source file...')
    subprocess.call(head+['-I'+ddf90mod]+tail)
    avh_pc.prnt(sys.argv[0]+': building library...')
    subprocess.call(['ar','cru',library,objects])
    subprocess.call(['ranlib',library])

elif ddtype+qdtype+mptype == 'mpfun90':
    avh_pc.prnt(sys.argv[0]+': compiling source file...')
    subprocess.call(head+['-I'+mpf90mod]+tail)
    avh_pc.prnt(sys.argv[0]+': building library...')
    subprocess.call(['ar','cru',library,objects])
    subprocess.call(['ranlib',library])

elif ddtype+qdtype+mptype == 'ddfun90'+'mpfun90':
    avh_pc.prnt(sys.argv[0]+': compiling source file...')
    subprocess.call(head+['-I'+ddf90mod,'-I'+mpf90mod]+tail)
    avh_pc.prnt(sys.argv[0]+': building library...')
    subprocess.call(['ar','cru',library,objects])
    subprocess.call(['ranlib',library])
