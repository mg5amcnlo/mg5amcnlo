#!/usr/bin/env python

import math

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
        proc['result'] = float(list[3])
        proc['error'] = float(list[5])
        proc['err_perc'] = proc['error']/proc['result']*100.
        processes.append(proc)
        tot+= proc['result']
        err+= math.pow(proc['error'],2)

processes.sort(key = lambda proc: -proc['result'])
content+='\n\n'
for proc in processes:
    content+='%(folder)20s %(channel)15s    %(result)10.8e    %(error)6.4e       %(err_perc)6.4e%%  \n' %  proc

content+='\nTotal cross-section: %10.8e +- %6.4e  (%6.4e%%)\n' %\
        (tot, math.sqrt(err), math.sqrt(err)/tot *100.)

file=open("res.txt", 'w')

file.write(content)
file.close()
