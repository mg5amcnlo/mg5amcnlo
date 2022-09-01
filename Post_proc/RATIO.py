#!/usr/bin/env python

import os.path
import sys
import math

data1={}
data2={}

### QCD+EW or QCD
infile1=sys.argv[1]
### QCD or LO
infile2=sys.argv[2]

with open(infile1,'r') as f:
     data1=f.readlines()
with open(infile2,'r') as f:
     data2=f.readlines()

outfile = 'ratio.HwU'
title={}
N_his=0
his_start={}
his_end={}

for i,line in enumerate(data2):
    if i == 0:
       header=line
    if '<histogram>' in line:
     N_his=N_his+1
     his_start[N_his]=i
     title[N_his]=line
    if '<\histogram>' in line:
     his_end[N_his]=i
     ending=line

out=[]

sums0=[]
sums_dy=[]
sums_cen=[]
sums_min=[]
sums_max=[]
sums_var1=[]
sums_var2=[]
sums_var3=[]
sums_var4=[]
sums_var5=[]
sums_var6=[]
sums_var7=[]
sums_var8=[]
sums_var9=[]

sum=0
sum_dy=0
sum_cen=0
sum_min=0
sum_max=0


for j in range(1,N_his+1):
  for k in range(his_start[j]+1,his_end[j]):
   if ('+0.0' not in data2[k].split()[2]) and ('-0.0' not in data2[k].split()[2]) and\
           ('0.0' not in data2[k].split()[2]):
    sum=float(data1[k].split()[2])/float(data2[k].split()[2])
    sum_dy=math.sqrt((float(data1[k].split()[3])/(float(data2[k].split()[2])))**2+(float(data1[k].split()[2])*(float(data2[k].split()[3]))/((float(data2[k].split()[2])**2)))**2)
    sum_cen=float(data1[k].split()[4])/float(data2[k].split()[4])
    sum_min=float(data1[k].split()[5])/float(data2[k].split()[5])
    sum_max=float(data1[k].split()[6])/float(data2[k].split()[6])
   else:
    sum=0.0
    sum_dy=0.0
    sum_cen=0.0
    sum_min=0.0
    sum_max=0.0

   sums0.append(sum)
   sums_dy.append(sum_dy)
   sums_cen.append(sum_cen)
   sums_min.append(sum_min)
   sums_max.append(sum_max)

nn=0
for j in range(1,N_his+1):
  for k in range(his_start[j]+1,his_end[j]):

      out.append(data1[k].split()[0])
      out.append(data1[k].split()[1])
      out.append(sums0[nn])
      out.append(sums_dy[nn])
      out.append(sums_cen[nn])
      out.append(sums_min[nn])
      out.append(sums_max[nn])
      nn=nn+1
      out.append(' ')
  out.append(ending) 
  out.append('stop')
    


u=1
with open(outfile,'w') as f:
 f.write(header)
 f.write('\n')
 f.write(title[1])
 for i in range(0,len(out)):
   if (out[i] == 'stop'):
     u=u+1
     f.write('\n')
     f.write('\n')
     if u<N_his+1:
      f.write(title[u])
   if (out[i] == ' '):
     f.write('\n')
   if (out[i] != 'stop' and out[i] != ' '):
     f.write(str(out[i]))
     f.write(' ')

print '\n','-----------> Output in ratio.HwU'




