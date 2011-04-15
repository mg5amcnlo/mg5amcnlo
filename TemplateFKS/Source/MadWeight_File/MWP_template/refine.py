#!/usr/bin/env python
import os
import math
import sys
#
sys.path+=['../'*i+'/Source/MadWeight_File/MWP_template/' for i in range(1,6)]
from madevent import *
print 'starting refine'
import madevent
############################################################################################
if __name__=='__main__':
    opt=sys.argv
    if len(opt)==2:
        final_precision=float(opt[1])
    else:
        final_precision=madevent.final_precision

    value,err=value_int()
    if not value or err/value>final_precision:
        main()
        refine(5)
        refine(1)
        store_value()
        write_final_result()
    else:
        print 'actual precision',err/value,' find the goal of ',final_precision
    ff=open('stop','w')
    ff.writelines('1')
    ff.close()

