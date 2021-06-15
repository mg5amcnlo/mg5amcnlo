#!/usr/bin/python3

import sys
import os
import subprocess
import glob
import math
import argparse
import shutil

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager

font_legend = font_manager.FontProperties(size=6)

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


def get_ymargin(inlist):

    ymax = max(inlist)
    ymin = min(inlist)
    # add a bit of margin
    if ymax > 0:
        ymax *= 1.2
    else:
        ymax *= 0.8

    if ymin < 0:
        ymin *= 1.2
    else:
        ymin *= 0.8

    # add 0 if both max and min are positive/negative
    if ymax * ymin > 0:
        if ymax > 0:
            ymin = 0.
        else:
            ymax = 0.

    return ymin, ymax


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
    sud_approx_r = os.path.join(pdir, 'Sud_Approx_Rij.dat')
    helicities_r, points_r = parse_sud_approx(sud_approx_r)

    sud_approx_s = os.path.join(pdir, 'Sud_Approx_noRij.dat')
    helicities_s, points_s = parse_sud_approx(sud_approx_s)

    fig, axes = plt.subplots(nrows = 4, ncols= 1, sharex = True, gridspec_kw = {'height_ratios' : [1,1,1,1]})
    #fig = plt.figure()
    fig.suptitle(get_pdir_title(pdir))
    fig.set_size_inches(5,7)
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])

    ############ inset, plot born
    ax = axes[0]
    for h, plist_r, plist_s in zip(helicities_r, points_r, points_s):
        # discard helicities with just one points
        if len(plist_r) < 2:
            continue
        # plot individual helicities with thin lines, and
        # summed helicities with thick ones
        if h == 'summed':
            thickness = 1.5
            linecolor = 'black'
        elif h == "lead-hel-summed":
            thickness = 1.5
            linecolor = 'red'
        else:
            thickness = 0.8
            linecolor = list(mcolors.TABLEAU_COLORS.values())[helicities_r.index(h)]

        linestyle = 'solid'
        ax.plot([p['energy'] for p in plist_r], [p['born'] for p in plist_r], label = "%s" % get_helicity_label(pdir,h),
                linestyle=linestyle, color=linecolor, linewidth=thickness)

        ax.set_ylabel('Born [GeV]$^{%d}$' % get_me_dimension(pdir))
        ax.legend(prop = font_legend)


    ############ inset, plot loop and the two sudakov approximations
    ax = axes[1]
    for h, plist_r, plist_s in zip(helicities_r, points_r, points_s):
        # discard helicities with just one points
        if len(plist_r) < 2:
            continue
        # plot individual helicities with thin lines, and
        # summed helicities with thick ones
        if h == 'summed':
            thickness = 1.5
            linecolor = 'black'
            use_label = True
        elif h == "lead-hel-summed":
            thickness = 1.5
            linecolor = 'red'
            use_label = False
        else:
            thickness = 0.8
            linecolor = list(mcolors.TABLEAU_COLORS.values())[helicities_r.index(h)]
            use_label = False

        linestyle = 'solid'
        ax.plot([p['energy'] for p in plist_r], [p['sud'] for p in plist_r], label = "SDK, $r_{ij}$" if use_label else None,
                linestyle=linestyle, color=linecolor, linewidth=thickness)
        linestyle = 'dashed'
        ax.plot([p['energy'] for p in plist_s], [p['sud'] for p in plist_s], label = "SDK, $s$" if use_label else None,
                linestyle=linestyle, color=linecolor, linewidth=thickness)

        linestyle="None"
        ax.plot([p['energy'] for p in plist_s], [p['loop'] for p in plist_s], '.', label = "V" if use_label else None,
                linestyle=linestyle, color=linecolor, linewidth=thickness)
    ax.legend(prop = font_legend)


    ############ inset with rij
    ax = axes[2]
    ax.set_title(r"$r_{ij}$ in logs", loc="right", pad=3)
    ax.set_ylabel(r"$\frac{V-SDK}{B}$, $r_{ij}$")

    helicities = helicities_r
    points = points_r
    linestyle = 'solid'

    for h, plist in zip(helicities, points):
        # discard helicities with just one points
        if len(plist) < 2:
            continue
        # plot individual helicities with thin lines, and
        # summed helicities with thick ones
        if h == 'summed':
            thickness = 1.5
            linecolor = 'black'
        elif h == "lead-hel-summed":
            thickness = 1.5
            linecolor = 'red'
        else:
            thickness = 0.8
            linecolor = list(mcolors.TABLEAU_COLORS.values())[helicities.index(h)]

        ax.plot([p['energy'] for p in plist], [p['diff'] for p in plist], label = get_helicity_label(pdir,h),
                linestyle=linestyle, color=linecolor, linewidth=thickness)

    # set the y range according to the second point
    ymin, ymax = get_ymargin([p[1]['diff'] for p in points if len(p) > 1])

    ax.set_ylim(ymin, ymax)

    ############ inset with s
    ax = axes[3]
    ax.set_title(r"$s$ in logs", loc="right", pad=3)
    ax.set_ylabel(r"$\frac{V-SDK}{B}$, $s$")

    helicities = helicities_s
    points = points_s
    linestyle = 'dashed'

    for h, plist in zip(helicities, points):
        # discard helicities with just one points
        if len(plist) < 2:
            continue
        # plot individual helicities with thin lines, and
        # summed helicities with thick ones
        if h == 'summed':
            thickness = 1.5
            linecolor = 'black'
        elif h == "lead-hel-summed":
            thickness = 1.5
            linecolor = 'red'
        else:
            thickness = 0.8
            linecolor = list(mcolors.TABLEAU_COLORS.values())[helicities.index(h)]

        ax.plot([p['energy'] for p in plist], [p['diff'] for p in plist], label = get_helicity_label(pdir,h),
                linestyle=linestyle, color=linecolor, linewidth=thickness)

    # set the y range according to the second point
    ymin, ymax = get_ymargin([p[1]['diff'] for p in points if len(p) > 1])
    ax.set_ylim(ymin, ymax)

    ########################

    ax.set_xscale('log')
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    fig.savefig("%s.pdf" % os.path.split(pdir)[1])


def plot_sud_approx0(pdir):
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
    ymin, ymax = get_ymargin([p[1]['diff'] for p in points if len(p) > 1])

    ax.set_ylim(ymin, ymax)

    ax.legend()
    fig.savefig("%s.pdf" % os.path.split(pdir)[1])


def set_rij(procdir, rijbool):

    sudfunc = os.path.join(procdir, 'SubProcesses', 'ewsudakov_functions.f') 
    infile = open(sudfunc)
    content = infile.read()
    infile.close()

    outfile = open(sudfunc, 'w')
    for l in content.split('\n'):
        if not "DATA s_to_rij" in l:
            outfile.write(l+'\n')
        else:
            outfile.write('      DATA s_to_rij / .%s. /\n' % str(rijbool))
    outfile.close()


def get_nparticles(pdir):
    nexternal = os.path.join(pdir, 'nexternal.inc') 
    infile = open(nexternal)
    content = infile.read()
    infile.close()
    for l in content.split('\n'):
        if 'PARAMETER (NEXTERNAL' in l:
            l = l.replace(')', '')
            return int(l.split('=')[1]) - 1

def get_me_dimension(pdir):
    """return the dimension of the matrix element, based on the number of external particles
    if n external particles, then dim = 4 - 2**(n-2)
    """
    return 4 - 2**(get_nparticles(pdir)-2)


def run_check_sudakov(pdir, log_suffix=''):
    p = subprocess.run(['make check_sudakov'], cwd = pdir, shell=True)
    p = subprocess.run(['./check_sudakov < ../../check_sudakov_input.txt > check_sudakov%s.log' % log_suffix], cwd = pdir, shell=True)



# let us use a parser to parse arguments
parser = argparse.ArgumentParser('Generate processes and plot Sudakov-Loop for dominant helicities')
parser.add_argument('--process', type=str, help="the process, with the same syntax as MG5_aMC without [QED SDK]")
parser.add_argument('--existing_dir', type=str, help="the process directory, if it has been already generated") 
parser.add_argument('--recompile', action='store_true', help="recompile the directory")
parser.add_argument('--onlyplot', action='store_true', help="skip everything but the plots")
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
    define p = u u~ d d~ g a
    define all = g u d b u~  b~ a ve e- ve~ e+ t t~ z w+ h w-
    generate %s QCD=0 QED=2 [QED SDK]
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


if not args.onlyplot:
    set_rij(outdir, True)

    # copy the Sud_Approx.dat files, these are with Rij activated
    for pdir in pdirs:
        run_check_sudakov(pdir, '_Rij')
        shutil.copyfile(os.path.join(pdir, 'Sud_Approx.dat'), os.path.join(pdir, 'Sud_Approx_Rij.dat'))

    set_rij(outdir, False)
    # copy the Sud_Approx.dat files, these are without Rij activated
    for pdir in pdirs:
        run_check_sudakov(pdir, '_noRij')
        shutil.copyfile(os.path.join(pdir, 'Sud_Approx.dat'), os.path.join(pdir, 'Sud_Approx_noRij.dat'))

for pdir in pdirs:
    plot_sud_approx(pdir)
