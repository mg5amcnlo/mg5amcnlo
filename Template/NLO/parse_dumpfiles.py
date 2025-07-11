import glob
import os
import math
import sys


def parse_ps(ps):
    xsect = 0.
    #axsect = 0.
    for l in ps.split('\n'):
        values = l.split()
        if values and values[0] == 'WGT':
            xsect+= float(values[2])
    return xsect, abs(xsect)


def parse_dumpfile(dump):
    dumpfile = open(dump)
    this_ps = ""
    xsect = 0.
    axsect = 0.
    xsect2 = 0.
    nps = 0
    while True:
        line = dumpfile.readline()
        if line.strip() == 'ENDPS': 
            this_xsect, this_axsect = parse_ps(this_ps)
            #print('PS', this_xsect, this_axsect)
            xsect+= this_xsect 
            axsect+= this_axsect 
            xsect2+= this_xsect**2
            this_ps = ""
            nps += 1
        if not line or 'NEWITER' in line:
            # here we just return
            print('nps', nps)
            return nps, xsect/nps, axsect/nps, xsect2/nps

        this_ps += line



if len(sys.argv) == 2:
    run_dir = sys.argv[1]
    dumpfiles = glob.glob(os.path.join('Events', run_dir, 'dump*.dat'))

else:
    dumpfiles = glob.glob('SubProcesses/P*/GF*/dump.dat')
print(dumpfiles)

rates = []
for dump in dumpfiles:
    print('Parsing %s' % dump)
    nps, xs, absxs, xs2 = parse_dumpfile(dump)
    err = math.sqrt((xs2-xs**2)/nps)
    abserr = math.sqrt((xs2-absxs**2)/nps)
    rates.append((xs, absxs)) 
    print('channel xsect: %5e +- %5e (rel error %4.1f pc) ' % (xs,err,err/xs*100))
    print('channel absxsect: %5e +- %5e (rel error %4.1f pc) ' % (absxs,abserr,abserr/absxs*100))

xsect = sum(r[0] for r in rates)
axsect = sum(r[1] for r in rates)

print('total xsect:', xsect)
print('total absxsect:', axsect)
