#!/usr/bin/env python3
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Mar 2023) for the MG5aMC CUDACPP plugin.
# Integrated with the MadGraph7 project in Feb 2026.

import os, sys
os.chdir( os.path.dirname( __file__ ) ) # go to the tput directory

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from subprocess import Popen

# Adapt plots to screen size
plots_smallscreen=False # desktop
###plots_smallscreen=True # laptop
if plots_smallscreen:
    plots_figsize=(6,3)
    plots_ftitlesize=9
    plots_txtsize=9
    plots_labelsize=9
    plots_legendsize=8
else:
    plots_figsize=(10,5)
    plots_ftitlesize=12
    plots_txtsize=15
    plots_legendsize=12
    plots_labelsize=12

#---------------------------------------

def loadOneTableFile( tablesuff, debug=False ):
    ###debug=True
    filetxt = 'summaryTable_' + tablesuff + '.txt'
    if not os.path.isfile( filetxt ):
        print( 'Unknown file', filetxt )
        return
    print( 'Loading tables in file', filetxt )
    file = open( filetxt )
    tputs = {}
    for line in file.readlines():
        lline = line.split()
        if len( lline ) == 0 : continue
        if lline[0] == "***" : fptype = lline[1][7]
        elif lline[0] == 'On' : metal = lline[5]
        elif line.startswith( '[' ):
            if   'icx 2023' in line : comp = 'icx2023'
            elif 'clang 14' in line : comp = 'clang14'
            elif 'gcc 12.1' in line : comp = 'gcc12.1'
            else : comp = 'unknown'
        elif lline[0].startswith( 'HELINL' ) : helinl = lline[0][7]
        elif lline[0] == "eemumu" : procs = [ '%-7s'%proc for proc in lline ]
        elif lline[0].startswith( 'CPP/' ) :
            avx = lline[0][4:]
            for i, proc in enumerate( procs ) :
                key = ( metal, fptype, helinl, comp, avx, proc )
                tput = lline[ i+1 ]
                if not tput.startswith( '-' ):
                    tputs[ key ] = tput
                    if debug : print( key, tput )
    file.close()
    return tputs

#---------------------------------------

def fptypeDesc( fptype ):
    if fptype == 'd' : return 'double'
    elif fptype == 'f' : return 'float'
    elif fptype == 'm' : return 'mixed'
    else: return 'UNKNOWN FPTYPE'

#---------------------------------------

def axesOneProc( ax, tputs, fptype='d', proc='ggttgg', tputabs=True ):
    metal = list( set( key[0] for key in tputs ) )[0] # assume all results come from the same node
    avxs = ( 'none', 'sse4', 'avx2', '512y', '512z' )
    markers = ( 'o', 'X' ) 
    ###colours = ( 'black', 'blue', 'red' ) 
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    yval0 = None
    desc0 = None
    if proc == 'ggttg' or proc == 'ggttggg' : helinls = [ '0' ]
    else: helinls = [ '0', '1' ]
    for ih, helinl in enumerate( helinls ):
        for ic, comp in enumerate( [ 'gcc12.1', 'clang14', 'icx2023' ] ):
            desc = '%s-inl%s'%( comp, helinl )
            xvals = []
            yvals = []
            for ia, avx in enumerate( avxs ) :
                key = ( metal, fptype, helinl, comp, avx, '%-7s'%proc )
                tput = tputs[key]
                xval = ia+1
                yval = float( tput )
                if yval0 is None : yval0 = yval
                if desc0 is None : desc0 = desc
                if not tputabs : yval = yval / yval0
                xvals.append( xval )
                yvals.append( yval )
                ###print( key, xval, yval )
            xmax = max( xmax, max( xvals ) )
            ymax = max( ymax, max( yvals ) )
            ###p = ax.plot( xvals, yvals, marker=markers[ih], color=colours[ic], label=desc )
            p = ax.plot( xvals, yvals, marker=markers[ih], label=desc )
    # Prepare axis ticks
    xmax = xmax + 1
    ymax = ymax * 1.2
    ax.axis( [xmin, xmax, ymin, ymax] )
    loc = 'upper left'
    ax.legend( loc=loc, fontsize=plots_legendsize )
    ax.set_xticks( xvals, avxs )
    # Prepare axis labels and title
    ax.set_title( '%s %s %s'%( 'Gold6130' if metal == 'Gold' else 'Silver4216', proc, fptypeDesc( fptype ) ) )
    ax.set_xlabel('SIMD level', size=plots_labelsize )
    if tputabs : ax.set_ylabel('Throughput (E6 events per second)', size=plots_labelsize )
    else : ax.set_ylabel('Throughput / Throughput-%s'%desc0, size=plots_labelsize )

#---------------------------------------

def figOneProc( tputs, proc='ggttgg' ):
    metal = list( set( key[0] for key in tputs ) )[0] # assume all results come from the same node
    ###abstputs = [ True, False ]
    abstputs = [ False ]
    fig = plt.figure( figsize = ( 10, 8 ) ) # keep same height with one or two rows (NB units are inches)
    fig.set_tight_layout( True )
    offset = 0
    for abstput in abstputs :
        ax1 = fig.add_subplot( 221 + offset )
        ax2 = fig.add_subplot( 222 + offset )
        axesOneProc( ax1, tputs, 'd', proc, abstput )
        axesOneProc( ax2, tputs, 'f', proc, abstput )
        ymax = 0
        ymax = max( ymax, ax1.get_ylim()[1] )
        ymax = max( ymax, ax2.get_ylim()[1] )
        ax1.set_ylim( [0, ymax] )
        ax2.set_ylim( [0, ymax] )
        offset += 2
    pngpath = 'PLOTS/ol23%s_%s.png'%( 'gold' if metal == 'Gold' else 'silv', proc.strip() )
    fig.set_tight_layout( True )
    fig.savefig( pngpath, format='png', bbox_inches="tight" )
    Popen( ['display', '-geometry', '+50+50', pngpath] )
    ###Popen( ['eog', pngpath] )
    print( 'Plot successfully saved on', pngpath )

#---------------------------------------

if __name__ == '__main__':

    # Test loading
    ###loadOneTableFile( 'ol23silv', debug=True )
    ###loadOneTableFile( 'ol23gold', debug=True )

    # Production plots
    figOneProc( loadOneTableFile( 'ol23silv' ), 'ggttgg' )
    figOneProc( loadOneTableFile( 'ol23gold' ), 'ggttgg' )
    figOneProc( loadOneTableFile( 'ol23silv' ), 'ggttggg' )
    figOneProc( loadOneTableFile( 'ol23gold' ), 'ggttggg' )
