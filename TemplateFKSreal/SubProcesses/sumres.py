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
content+='\n\n'
for proc in processes:
    content+='%(folder)20s %(channel)15s   %(result)10.8e    %(error)6.4e       %(err_perc)6.4e%%  \n' %  proc

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
