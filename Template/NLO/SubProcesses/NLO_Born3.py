#! /usr/bin/env python

inp = open('./MADatNLO.top', 'r')
out = open('./MADatNLO_combined.top', 'w')

whole = inp.readlines()

a=0
b=0
startNLO=[]
startBorn=[]
stop=[]

for i, v in enumerate(whole):
    vs=v.strip()
    if (vs[:1] == '(' and vs[-3:] == 'NLO'):
        whole[i]=vs[0:-3]
        a=a+1
        startNLO.append(i+2)
    if (vs[:1] == '(' and vs[-4:] == 'Born'):
        b=b+1
        startBorn.append(i+2)
    if (vs[:4] == 'HIST'):
        stop.append(i)

stopNLO=stop[0:len(startNLO)]
stopBorn=stop[len(startNLO):]

stopNLO.insert(0,0)

print startNLO
print stopNLO
print startBorn
print stopBorn


for j in range(len(startNLO)):
    for i in range(stopNLO[j]+2,startNLO[j]):
        out.write(whole[i].strip()+'\n')
    out.write('( NLO'+'\n')
    for i in range(startNLO[j],stopNLO[j+1]):
        out.write(whole[i].strip()+'\n')
    out.write('HISTO red'+'\n')
    out.write('PLOT'+'\n')
    out.write('( Born'+'\n')
    for i in range(startBorn[j],stopBorn[j]):
        out.write(whole[i].strip()+'\n')
    out.write('HISTO blue dashed'+'\n')
    out.write('\n')
