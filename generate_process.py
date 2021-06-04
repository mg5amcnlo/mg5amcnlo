#!/usr/bin/python3

import sys
import os
import subprocess
import glob
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_sud_approx(infile):
    content = open(infile).read()
    lines = [l for l in content.split('\n') if l.strip()]
    helicities = []
    points = []
    for l in lines:
        values = l.split()

        # check if the line is not a text line, oterwise skip it
        try:
            energy = float(values[0])
            helicity = values[1]
            loop = float(values[2])
            sud = float(values[3])
            diff = float(values[4])
            born = float(values[5])
        except ValueError:
            continue

        point = {'energy': energy,
                 'loop' : loop,
                 'sud' : sud,
                 'diff' : diff,
                 'born' : born}

        # add the point to the corresponding helicity
        try:
            ihel = helicities.index(helicity)
            points[ihel].append(point)
        except ValueError:
            helicities.append(helicity)
            points.append([point])

    return helicities, points


def get_pdir_title(pdir):
    # at the moment, just return the process directory
    title = os.path.split(pdir)[1]
    return title


def get_helicity_label(pdir, h):
    try:
        ih = int(h)
    except ValueError:
        # if it is not an integer, then just return it
        return h
    else:
        sdk_wrap = open(os.path.join(pdir,'ewsudakov_wrapper.f')).read()
        hel_line = [l for l in sdk_wrap.split('\n') if 'DATA (NHEL(I' in l and '%d),I=' % ih in l][0]
        helicities = hel_line.split('/')[1]
        # replace +-1 with +-
        helicities = helicities.replace('-1', '-')
        helicities = helicities.replace(' 1', '+')
        helicities = helicities.replace(' 0', '0')
        # remove the commas
        helicities = helicities.replace(',', '')
        
        return "%s: %s" % (h, helicities)


def plot_sud_approx(pdir):
    sud_approx = os.path.join(pdir, 'Sud_Approx.dat')
    helicities, points = parse_sud_approx(sud_approx)

    fig = plt.figure()
    fig.suptitle(get_pdir_title(pdir))
    ax = fig.add_subplot()
    fig.set_size_inches(7,5)
    for h, plist in zip(helicities, points):
        # discard helicities with just one points
        if len(plist) < 2:
            continue
        # plot individual helicities with thin lines, and
        # summed helicities with thick ones
        if h == 'summed':
            thickness = 1.5
            linestyle = 'solid'
            linecolor = 'black'
        elif h == "lead-hel-summed":
            thickness = 1.5
            linestyle = 'dashed'
            linecolor = 'black'
        else:
            thickness = 0.8
            linestyle = 'solid'
            linecolor = list(mcolors.TABLEAU_COLORS.values())[helicities.index(h)]

        ax.plot([p['energy'] for p in plist], [p['diff'] for p in plist], label = get_helicity_label(pdir,h),
                linestyle=linestyle, color=linecolor, linewidth=thickness)

    ax.set_xscale('log')
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\frac{|V-SDK|}{B}$", loc='top', fontsize=13)
    # set the y range according to the second point
    ymax = max([p[1]['diff'] for p in points if len(p) > 1])
    ymin = min([p[1]['diff'] for p in points if len(p) > 1])
    # add a bit of margin
    if ymax > 0:
        ymax *= 1.2
    else:
        ymax *= 0.8

    if ymin < 0:
        ymin *= 1.2
    else:
        ymin *= 0.8

    ax.set_ylim(ymin, ymax)
    ax.legend()
    fig.savefig("%s.pdf" % os.path.split(pdir)[1])


# let us use a parser to parse arguments
parser = argparse.ArgumentParser('Generate processes and plot Sudakov-Loop for dominant helicities')
parser.add_argument('--process', type=str, help="the process, with the same syntax as MG5_aMC without [QED SDK]")
parser.add_argument('--existing_dir', type=str, help="the process directory, if it has been already generated") 
parser.add_argument('--recompile', action='store_true', help="recompile the directory")
args = parser.parse_args()


if args.process:
    # name for the output dir
    idir = 0
    outdir_name = 'outdir_sdk_%d' 
    while os.path.isdir(outdir_name % idir):
        idir +=1

    outdir = outdir_name % idir


    infile = \
    """import model loop_qcd_qed_sm_Gmu_forSudakov 
    generate %s [QED SDK]
    output %s
    launch -i
    compile FO
    done 
    """
    fout = open('infile.txt', 'w')
    fout.write(infile % (args.process, outdir))
    fout.close()

    p = subprocess.run(['./bin/mg5_aMC infile.txt'], cwd = os.getcwd(), shell = True)

if args.existing_dir:
    outdir = args.existing_dir

    if args.recompile:

        infile = \
        """launch -i %s
        compile FO
        done 
        """
        fout = open('infile.txt', 'w')
        fout.write(infile % (outdir))
        fout.close()
        p = subprocess.run(['./bin/mg5_aMC infile.txt'], cwd = os.getcwd(), shell = True)

pdirs = glob.glob(os.path.join(outdir, 'SubProcesses', 'P0_*'))

for pdir in pdirs:
    plot_sud_approx(pdir)
