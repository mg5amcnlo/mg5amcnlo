#!/usr/bin/env python

#script to combine reults and tell the number of events that need
#  to be generated in each channel. 
#  Replaces the sumres.f and sumres2.f files
#  MZ, 2011-10-22

from __future__ import division
import math
import sys
import random
import os

nexpected=int(sys.argv[1])
nevents=int(sys.argv[2])
req_acc=float(sys.argv[3])
# if nevents is >=0 the script will also determine the 
# number of events required for each process


def Mirrorprocs(p1, p2):
    """determine if the folder names p1, p2 (with the _N already taken out)
    correspond to the same process with
    mirrror initial state. Returns true/false"""
    return False

file=open("res.txt")
content = file.read()
file.close()
lines = content.split("\n")
processes=[]
tot=0
err=0
totABS=0
errABS=0

# open the file containing the list of directories
file=open("dirs.txt")
dirs = file.read().split("\n")
file.close()
dirs.remove('')

# The syntax of lines should be first the ABS cross section for the
# channel and the line after that the cross section for the same
# channel.
for line in range(0,len(lines),2):
    list = lines[line].split()
    if list:
        proc={}
        proc['folder'] = list[0].split('/')[0]
        proc['subproc'] = proc['folder'][0:proc['folder'].rfind('_')]
        proc['channel'] = list[0].split('/')[1]
        dirs.remove(os.path.join(proc['folder'], proc['channel']))
        proc['resultABS'] = float(list[4])
        proc['errorABS'] = float(list[6])
        proc['err_percABS'] = proc['errorABS']/proc['resultABS']*100.
        processes.append(proc)
        totABS+= proc['resultABS']
        errABS+= math.pow(proc['errorABS'],2)
        list = lines[line+1].split()
        if list:
            proc['result'] = float(list[3])
            proc['error'] = float(list[5])
            proc['err_perc'] = proc['error']/proc['result']*100.
            tot+= proc['result']
            err+= math.pow(proc['error'],2)
if dirs:
    print "%d jobs did not terminate correctly: " % len(dirs)
    print '\n'.join(dirs)
    print "The results are probably not correct. Please check the relevant log files corresponding to the above jobs for more information."

processes.sort(key = lambda proc: -proc['errorABS'])

correct = len(processes) == nexpected
print "Found %d correctly terminated jobs " %len(processes) 
if not len(processes)==nexpected:
    print len(processes), nexpected

subprocs_string=[]
for proc in processes:
    subprocs_string.append(proc['subproc'])
subprocs_string=set(subprocs_string)

content+='\n\nCross-section per integration channel:\n'
for proc in processes:
    content+='%(folder)20s  %(channel)15s   %(result)10.8e    %(error)6.4e       %(err_perc)6.4f%%  \n' %  proc

content+='\n\nABS cross-section per integration channel:\n'
for proc in processes:
    content+='%(folder)20s  %(channel)15s   %(resultABS)10.8e    %(errorABS)6.4e       %(err_percABS)6.4f%%  \n' %  proc

content+='\n\nCross-section per subprocess:\n'
#for subpr in sorted(set(subprocs)):
subprocesses=[]
for sub in subprocs_string:
    subpr={}
    subpr['subproc']=sub
    subpr['xsect']=0.
    subpr['err']=0.
    for proc in processes:
        if proc['subproc'] == sub:
            subpr['xsect'] += proc['result']
            subpr['err'] += math.pow(proc['error'],2)
    subpr['err']=math.sqrt(subpr['err'])
    subprocesses.append(subpr)


#find and combine mirror configurations (if in v4)
for i1, s1 in enumerate(subprocesses):
    for i2, s2 in enumerate(subprocesses):
        if Mirrorprocs(s1['subproc'], s2['subproc']) and i1 >= i2:
            s1['xsect'] += s2['xsect']
            s1['err'] = math.sqrt(math.pow(s1['err'],2)+ math.pow(s2['err'],2))
            s2['toremove'] = True

new = []
for s in subprocesses:
    try:
        a= s['toremove']
    except KeyError:
        new.append(s)
subprocesses= new


subprocesses.sort(key = lambda proc: -proc['xsect'])
for subpr in subprocesses:
    content+=  '%(subproc)20s    %(xsect)10.8e   %(err)6.4e\n' % subpr
 

content+='\nTotal ABS and \nTotal: \n                      %10.8e +- %6.4e  (%6.4e%%)\n                      %10.8e +- %6.4e  (%6.4e%%)\n' %\
        (totABS, math.sqrt(errABS), math.sqrt(errABS)/totABS *100.,tot, math.sqrt(err), math.sqrt(err)/tot *100.)

if not correct:
    sys.exit('ERROR: not all jobs terminated correctly\n')

file=open("res.txt", 'w')

file.write(content)
file.close()

#determine the events for each process:
if nevents>=0:
    if req_acc<0: 
        req_acc2_inv=nevents
    else:
        req_acc2_inv=1/(req_acc*req_acc)
    #get the random number seed from the randinit file
    file=open("randinit")
    exec file
    file.close
    print "random seed found in 'randinit' is", r
    random.seed(r)
    totevts=nevents
    for proc in processes:
        proc['lhefile'] = os.path.join(proc['folder'], proc['channel'], 'events.lhe')
        proc['nevents'] = 0
    while totevts :
        target = random.random() * totABS
        crosssum = 0.
        i = 0
        while i<len(processes) and crosssum < target:
            proc = processes[i]
            crosssum += proc['resultABS']
            i += 1            
        totevts -= 1
        i -= 1
        processes[i]['nevents'] += 1
        
#check that we now have all the events in the channels
    totevents = sum(proc['nevents'] for proc in processes)
    if totevents != nevents:
        sys.exit('failed to obtain the correct number of events. Required: %d,   Obtained: %d' \
                 % (nevents, totevents))

    content_evts = ''
    for proc in processes:
        content_evts+= ' '+proc['lhefile']+'              %(nevents)10d            %(resultABS)10.8e        1.0 \n' %  proc
        nevts_file = open(os.path.join(proc['folder'], proc['channel'], 'nevts'),'w')
        nevts_file.write('%10d\n' % proc['nevents'])
        nevts_file.close()
        if proc['channel'][1] == 'B':
            fileinputs = open("madinMMC_B.2")
        elif proc['channel'][1] == 'F':
            fileinputs = open("madinMMC_F.2")
        elif proc['channel'][1] == 'V':
            fileinputs = open("madinMMC_V.2")
        else:
            sys.exit("ERROR, DONT KNOW WHICH INPUTS TO USE")
        fileinputschannel = open(os.path.join(proc['folder'], proc['channel'], 'madinM1'),'w')
        i=0
        for line in fileinputs:
            i += 1
            if i == 2:
                accuracy=min(math.sqrt(totABS/(req_acc2_inv*proc['resultABS'])),0.2)
                fileinputschannel.write('%10.8e\n' % accuracy)
            elif i == 8:
                fileinputschannel.write('1        ! MINT mode\n')
            else:
                fileinputschannel.write(line)
        fileinputschannel.close()
        fileinputs.close()

    evts_file = open('nevents_unweighted', 'w')
    evts_file.write(content_evts)
    evts_file.close()

# if nevents = -1 and req_acc >= 0, we need to determine the required
# accuracy in each of the channels: this is for fixed order running!
elif req_acc>=0 and nevents==-1:
    req_accABS=req_acc*abs(tot)/totABS
    content_evts = ''
    for proc in processes:
        if proc['channel'][0:3] == 'all':
            fileinputs = open("madin.all")
        elif proc['channel'][0:4] == 'novB':
            fileinputs = open("madin.novB")
        elif proc['channel'][0:4] == 'born':
            fileinputs = open("madin.born")
        elif proc['channel'][0:4] == 'grid':
            fileinputs = open("madin.grid")
        elif proc['channel'][0:4] == 'viSB':
            fileinputs = open("madin.viSB")
        elif proc['channel'][0:4] == 'virt':
            fileinputs = open("madin.virt")
        elif proc['channel'][0:4] == 'novi':
            fileinputs = open("madin.novi")
        else:
            sys.exit("ERROR, DONT KNOW WHICH INPUTS TO USE")
        fileinputschannel = open(os.path.join(proc['folder'], proc['channel'], 'madinM1'),'w')
        i=0
        for line in fileinputs:
            i += 1
            if i == 2:
                accuracy=req_accABS*math.sqrt(totABS*proc['resultABS'])
                fileinputschannel.write('%10.8e\n' % accuracy)
            elif i == 9:
                fileinputschannel.write('-1        ! restart from existing grids\n')
            else:
                fileinputschannel.write(line)
        fileinputschannel.close()
        fileinputs.close()
