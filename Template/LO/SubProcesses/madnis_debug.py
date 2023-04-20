import madnis_dev as madnis
import random
import time
import sys 
import numpy as np
multi_channel_in = True
helicity_sum = True
channel = 1
dconfig = channel


madnis.configure_code(multi_channel_in,helicity_sum,dconfig)


nbatch = 1000
wgts = np.zeros(nbatch)
wgts_corr = np.zeros(nbatch)
#random.seed(9001)
chans = np.random.randint(3,5,size=nbatch)
a = 0
n = 0
for j in range(nbatch):
    R = np.array([random.random() for _ in range(20)])
    #R[7:] = 0.6
    #print(R)
    #print(f"Randon numbers: {R}")
    try:
        w = madnis.madnis_api(R, channel, True)
        alpha, used_channel = madnis.get_multichannel()
        if w > 0:
            n += 1
            a += used_channel
            print(used_channel)
        n_rand = madnis.get_number_of_random_used()
        r_ut = madnis.get_random_used_utility()
        wgts[j] = w
        wgts_corr[j] = np.nan_to_num(w/alpha[used_channel-1])
        #wgts[j] = w
        #print(n_rand, wgts[j], r_ut, alpha)
        #print(n_rand, wgts[j], r_ut)
        #print(f"bare weight: {wgts[j]}, correct weight: {wgts_corr[j]}, alpha: {alpha[channel-1]}")
        # print(f"alphas: {alpha}")
    except:
        pass

wgt2 = wgts**2
mean = np.mean(wgts)
error = np.sqrt((np.mean(wgt2) - mean**2)/(nbatch-1))
print(mean, error)

wgtc2 = wgts_corr**2
meanc = np.mean(wgts_corr)
errorc = np.sqrt((np.mean(wgtc2) - meanc**2)/(nbatch-1))
print(meanc, errorc)
