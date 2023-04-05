import madnis_dev as madnis
import random
import time
import sys 
multi_channel_in = True
helicity_sum = True
dconfig = 1.0


madnis.configure_code(multi_channel_in,helicity_sum,dconfig)
R= [random.random() for _ in range(20)]
wgt = madnis.madnis_api(R,True)
p = madnis.get_momenta()
print(p)
print(wgt)
print(madnis.get_number_of_random_used())

i=0
wgt =0
start = time.time()
last =  1
work = 0
index= []
index_fail = []
try:
    for j in range(50):
        R= [random.random() for _ in range(20)]
        wgt = madnis.madnis_api(R,False)
        p = madnis.get_momenta()
#        print(p)
#        print("value is ", wgt)
        i+=1
        current = time.time() - start
        if wgt == 0:
            last = current +1
#            print(i, current)
#            if len(index_fail) > 100:
            index_fail.append(j)
#                break
#            else:
#                index_fail.append(j)
        else:
            work+=1
            index.append(madnis.get_number_of_random_used())
except:
    print(i, current)
    
print("working for", work, index)
print('failing for', len(index_fail))
