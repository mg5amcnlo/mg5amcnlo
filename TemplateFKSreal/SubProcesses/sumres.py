#!/usr/bin/env python

#script to combine reults and tell the number of events that need
#  to be generated in each channel. 
#  Replaces the sumres.f and sumres2.f files
#  MZ, 2011-10-22

import math
import sys
import random
import os

nevents=int(sys.argv[1])
# if nevents is >0 the script will also determine the 
# number of events required for each process


def Mirrorprocs(p1, p2):
    """determine if the folder names p1, p2 (with the _N already taken out)
    correspond to the same process with
    mirrror initial state. Returns true/false"""
    if p1 ==p2:
        return False
    p1s = p1.split('_')
    p2s = p2.split('_')
    if len(p1s) != len(p2s):
        return False
    for a1, a2 in zip(p1s, p2s):
        if len(a1) != len(a2):
            return False

   # check that final state is the same
    if p1s[2] != p2s[2]:
        return False
    else:
        i1 = p1s[1]
        i2 = p2s[1]
        if i1[len(i1)/2:] + i1[:len(i1)/2] == i2 or \
            i1[len(i1)/2+1:] + i1[:len(i1)/2+1] == i2 :
            return True

file=open("res.txt")
content = file.read()
file.close()
lines = content.split("\n")
processes=[]
tot=0
err=0
for line in lines:
    list = line.split()
    if list:
        proc={}
        proc['folder'] = list[0].split('/')[0]
        proc['subproc'] = proc['folder'][0:proc['folder'].rfind('_')]
        proc['channel'] = list[0].split('/')[1]
        if list[3] != '[ABS]:':
            proc['result'] = float(list[3])
            proc['error'] = float(list[5])
        else:
            proc['result'] = float(list[4])
            proc['error'] = float(list[6])
        proc['err_perc'] = proc['error']/proc['result']*100.
        processes.append(proc)
        tot+= proc['result']
        err+= math.pow(proc['error'],2)

processes.sort(key = lambda proc: -proc['result'])

subprocs_string=[]
for proc in processes:
    subprocs_string.append(proc['subproc'])
subprocs_string=set(subprocs_string)

content+='\n\nCross-section per integration channel:\n'
for proc in processes:
    content+='%(folder)20s %(channel)15s   %(result)10.8e    %(error)6.4e       %(err_perc)6.4e%%  \n' %  proc

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
 

content+='\nTotal cross-section: %10.8e +- %6.4e  (%6.4e%%)\n' %\
        (tot, math.sqrt(err), math.sqrt(err)/tot *100.)

file=open("res.txt", 'w')

file.write(content)
file.close()

#determine the events for each process:
if nevents>0:
    totevts=0
    totrest=0.
    for proc in processes:
        proc['lhefile'] = os.path.join(proc['folder'], proc['channel'], 'events.lhe')
        proc['nevents'] = math.trunc(proc['result']/tot * nevents)
        totevts += proc['nevents']
        proc['rest'] = proc['result']/tot * nevents - math.trunc(proc['result']/tot * nevents)
        totrest += totrest + math.sqrt(proc['rest'] + proc['nevents'])


    restevts = nevents - totevts

    if restevts>len(processes) or restevts < 0:
        sys.exit("ERROR, restevts is not valid: %d" % restevts)

# Determine what to do with the few remaining 'rest events'.
# Put them to channels at random given by the sqrt(Nevents) 
# already in the channel (including the rest)

    while restevts:
        target = random.random() * totrest
        restsum=0.
        i = 0
        while i<len(processes) and restsum < target:
            proc = processes[i]
            restsum += math.sqrt(proc['rest']+proc['nevents'])
            i += 1
        i -= 1
        totrest += - math.sqrt(proc['rest']+proc['nevents']) \
                   + math.sqrt(1.+proc['nevents'])
        processes[i]['nevents'] += 1
        processes[i]['rest'] = 0.
        restevts -=1
        
#check that we now have all the events in the channels
    totevents = sum(proc['nevents'] for proc in processes)
    if totevents != nevents:
        sys.exit('failed to obtain the correct number of events. Required: %d,   Obtained: %d' \
                 % (nevents, totevents))

    content_evts = ''
    for proc in processes:
        content_evts+= '%(lhefile)50s %(nevents)10d   %(result)10.8e \n' %  proc
        nevts_file = open(os.path.join(proc['folder'], proc['channel'], 'nevts'),'w')
        nevts_file.write('%10d\n' % proc['nevents'])
        nevts_file.close()



    evts_file = open('nevents_unweighted', 'w')
    evts_file.write(content_evts)
    evts_file.close()
