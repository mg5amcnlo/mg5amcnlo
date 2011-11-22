#!/usr/bin/python
#
# Collect the decay widths and calculate BRs for all particles, and put in param_card form

import glob
import optparse
import os
import re

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.path.pardir, os.path.pardir))

# Main routine
if __name__ == "__main__":

    # Find available runs
    run_names = glob.glob(os.path.join(root_path, "Events", "*_banner.txt"))
    run_names = [os.path.basename(name) for name in run_names]
    run_names = [re.match("(.*)_banner.txt", name).group(1) for name in run_names]
    usage = "usage: %prog run_name"
    usage +="\n      where run_name in (%s)" % ",".join(run_names)
    parser = optparse.OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 1 or args[0] not in run_names:
        if len(run_names) == 1:
            args = run_names
        else:
            parser.error("Please give a run name")
            exit
    run_name = args[0].strip()

    print "Collecting results for run ", run_name
    
    # Start collecting results
    subproc_dirs = \
           open(os.path.join(root_path, "SubProcesses", "subproc.mg")).read().split('\n')
    
    particle_dict = {}
    for subdir in subproc_dirs[:-1]:
        subdir = os.path.join(root_path, "SubProcesses", subdir)
        leshouche = open(os.path.join(subdir, 'leshouche.inc')).read().split('\n')[0]
        particles = re.search("/([\d,-]+)/", leshouche)
        if not particles:
            continue
        particles = [int(p) for p in particles.group(1).split(',')]
        results = \
             open(os.path.join(subdir, run_name + '_results.dat')).read().split('\n')[0]
        result = float(results.strip().split(' ')[0])
        try:
            particle_dict[particles[0]].append([particles[1:], result])
        except KeyError:
            particle_dict[particles[0]] = [[particles[1:], result]]


    # Open the param_card.dat and insert the calculated decays and BRs
    param_card_file = open(os.path.join(root_path, 'Cards', 'param_card.dat'))
    param_card = param_card_file.read().split('\n')
    param_card_file.close()

    decay_lines = []
    line_number = 0
    # Read and remove all decays from the param_card
    while line_number < len(param_card):
        line = param_card[line_number]
        if line.lower().startswith('decay'):
            # Read decay if particle in particle_dict
            # DECAY  6   1.455100e+00
            line = param_card.pop(line_number)
            line = line.split()
            particle = 0
            if int(line[1]) not in particle_dict:
                try: # If formatting is wrong, don't want this particle
                    particle = int(line[1])
                    width = float(line[2])
                except:
                    particle = 0
            # Read BRs for this decay
            line = param_card[line_number]
            while line.startswith('#') or line.startswith(' '):
                line = param_card.pop(line_number)
                if not particle or line.startswith('#'):
                    line=param_card[line_number]
                    continue
                #    6.668201e-01   3    5  2  -1
                line = line.split()
                try: # Remove BR if formatting is wrong
                    partial_width = float(line[0])*width
                    decay_products = [int(p) for p in line[2:2+int(line[1])]]
                except:
                    line=param_card[line_number]
                    continue
                try:
                    particle_dict[particle].append([decay_products, partial_width])
                except KeyError:
                    particle_dict[particle] = [[decay_products, partial_width]]
                line=param_card[line_number]
            if particle and particle not in particle_dict:
                # No decays given, only total width
                particle_dict[particle] = [[[], width]]
        else: # Not decay
            line_number += 1
    # Clean out possible remaining comments at the end of the card
    while not param_card[-1] or param_card[-1].startswith('#'):
        param_card.pop(-1)

    # Append calculated and read decays to the param_card
    param_card.append("#\n#*************************")
    param_card.append("#      Decay widths      *")
    param_card.append("#*************************")
    for key in sorted(particle_dict.keys()):
        width = sum([r for p,r in particle_dict[key]])
        param_card.append("#\n#      PDG        Width")
        param_card.append("DECAY  %i   %e" % (key, width))
        if not width:
            continue
        if particle_dict[key][0][0]:
            param_card.append("#  BR             NDA  ID1    ID2   ...")
            brs = [[val[1]/width, val[0]] for val in particle_dict[key] if val[1]]
            for val in sorted(brs, reverse=True):
                param_card.append("   %e   %i    %s" % (val[0], len(val[1]),
                                           "  ".join([str(v) for v in val[1]])))
    output_name = os.path.join(root_path, run_name + "_param_card.dat")
    decay_table = open(output_name, 'w')
    decay_table.write("\n".join(param_card) + "\n")
    print "Results written to ", output_name
    
