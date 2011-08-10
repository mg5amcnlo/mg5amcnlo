#!/usr/bin/env python
import os
import math

###########################################################################
final_precision=0.001
###########################################################################

def launch_ajob():
    update_seed()
    jobdir=os.listdir('.')
    scriptlist=[script for script in jobdir if len(script)>4 and script[:4]=='ajob']
    for script in scriptlist:
        os.system('chmod +x '+script)
        os.system('./'+script)    

def update_seed(step=1):
    """add 'step[1] to the seed """
    #read seed value
    file='./seedvalue.inc'
    try:
        ff=open(file,'r')
        line=ff.readline()
        seed=int(line)
        ff.close()
    except:
        seed=0
    #update seed
    seed+=step
    #write new seed
    ff=open(file,'w')
    line=ff.writelines(str(seed)+'\n')
    ff.close()

def main():
    os.system('cd ../..;make;cd -')
    scriptlist=[script for script in os.listdir('../..') if len(script)>4 and script[:4]=='ajob']
    for script in scriptlist:
        os.system('ln ../../'+script)
    ff=open('start','w')
    ff.writelines('1')
    ff.close()
    launch_ajob()
    os.system('../../../../bin/sum_html &> /dev/null')
    ff=open('stop','w')
    ff.writelines('1')
    ff.close()


def refine(nbmax):
    #enter in refine mode
    #print 'delete ajob'
    #os.system('rm ajob? ajob?? ajob???')
    #os.system('echo '+str(final_precision)+' 5 | ../../../../bin/gen_ximprove &>/dev/null')
    #launch_ajob()
    #os.system('../../../../bin/sum_html &> /dev/null')

    #enter in refine 2 mode
    print 'delete ajob'
    os.system('rm ajob? ajob?? ajob???')
    os.system('echo '+str(final_precision)+' '+str(nbmax)+' | ../../../../bin/gen_ximprove &>/dev/null')
    launch_ajob()
    os.system('../../../../bin/sum_html &> /dev/null')


def average(data):

    return sum(data)/len(data)

def quad(lerror):

    prov=[val**2 for val in lerror]
    return math.sqrt(sum(prov))/len(prov)




def add_iter(value,err):
    ff=open('all_iter.dat','a')
    ff.writelines(str(value)+'\t'+str(err)+'\n')
    ff.close()

def value_int():
    try:
        ff=open('results.dat','r')
    except:
        return 0,0
    line=ff.read()
    out=float(line.split()[0])
    error=math.sqrt(float(line.split()[1])**2+float(line.split()[2])**2)
    return out,error


def store_value():
    out,error=value_int()
    add_iter(out,error)
                

def write_final_result():

    lvalue=[]
    lerror=[]
    for line in file('all_iter.dat'):
        value,err=line.split()
        lvalue.append(float(value))
        lerror.append(float(err))
    cross=sum([0]+[lvalue[i]/lerror[i]**2 for i in range(0,len(lvalue)) if lerror[i]])
    error_sq=sum([0]+[1.0/error**2 for error in lerror if error])
    if error_sq:
        cross=cross/error_sq
        error=math.sqrt(1.0/error_sq)
    else:
        cross=0
        error=0

    ff=open('results.dat','w')
    ff.writelines(str(cross)+'\t'+str(error)+'\t0.00000E+00        0        3    0        0 0.000E+00\n')
    ff.close()

if __name__=='__main__':
    main()
    refine(5)
    refine(1)
    store_value()
    write_final_result()
