#!/usr/bin/python

import glob
import sys
import os
import copy

def_path = sys.argv[1]
FA4_path = sys.argv[2]


tagline = "energy    helicity     loop/born     sud/born     (loop-sud)/born     born"


def parse_content(text):
    """
 energy    helicity     loop/born     sud/born     (loop-sud)/born     born     
   10000.000000000000          summed -0.49129850669769404      -0.46230549991272890       -2.8993006784965126E-002   1.1924022810786012E-002
   10000.000000000000                5  -7.5456236960916551E-002 -0.15836708842545213        8.2910851464535584E-002   1.3429027980133972E-004
   10000.000000000000               28 -0.52074985888682279      -0.47464796633383194       -4.6101892552990846E-002   6.9756601414753226E-003
   10000.000000000000               32 -0.28427920886799724      -0.38185958259737202        9.7580373729374767E-002   1.1096997397835725E-003
   10000.000000000000               36 -0.51318365389679499      -0.47464796633445006       -3.8535687562344899E-002   3.7006768611195320E-003
   10000.000000000000      lead hel summed -0.49137062264976467      -0.46244690065798988       -2.8923721991774797E-002   1.1920327022179766E-002
    """

    points = text.split(tagline) 
    events = []
    for point in points:
        helicities = []
        if not point.strip(): continue
        for line in point.split('\n'):
            if not line.strip(): continue
            entries = line.split()
            helicities.append({'energy': float(entries[0]),
                               'helicity': entries[1],
                               'loop/born': float(entries[2]),
                               'sud/born': float(entries[3]),
                               'loop-sud/born': float(entries[4]),
                               'born': float(entries[5])})
        events.append(helicities)

    return events


def merge_contents(def_list, FA4_list):
    """ take sudakov from FA4, loops from def
    """

    content = ''
    for def_helicities, FA4_helicities in zip(def_list, FA4_list):
        content += tagline + '\n'
        for def_h, FA4_h in zip(def_helicities, FA4_helicities):
            if def_h['energy'] != FA4_h['energy']:
                print('Different energies!! %f %f' % (def_h['energy'], FA4_h['energy']))
            if def_h['helicity'] != FA4_h['helicity']:
                print('Different helicities!! %f %f' % (def_h['helicity'], FA4_h['helicity']))
            if def_h['born'] != FA4_h['born']:
                print('Different borns!! %e %e' % (def_h['born'], FA4_h['born']))

            replace_dict = copy.copy(def_h)
            replace_dict['sud/born'] = FA4_h['sud/born']
            replace_dict['loop-sud/born'] = def_h['loop/born'] - FA4_h['sud/born']
            
            content += "%(energy)f  %(helicity)s  %(loop/born)f  %(sud/born)f  %(loop-sud/born)f  %(born)e\n" % replace_dict
    return content



def_P0 = [os.path.split(d)[-1] for d in glob.glob(os.path.join(def_path, 'SubProcesses', 'P0*'))]
FA4_P0 = [os.path.split(d)[-1] for d in glob.glob(os.path.join(FA4_path, 'SubProcesses', 'P0*'))]

print (def_P0)
print (FA4_P0)

if def_P0 != FA4_P0:
    print('Different P0 directories: \n%s\n%s' % (str(def_P0), str(FA4_P0)))
    sys.exit(1)

for P0 in def_P0:
    def_out = os.path.join(def_path, 'SubProcesses', P0, 'Sud_Approx.dat')
    FA4_out = os.path.join(FA4_path, 'SubProcesses', P0, 'Sud_Approx.dat')

    def_content = parse_content(open(def_out).read())
    FA4_content = parse_content(open(FA4_out).read())

    merged = merge_contents(def_content, FA4_content)
    outfile = open(def_path + '_' + P0 + '_Sud_Approx.dat', 'w').write(merged)

    
