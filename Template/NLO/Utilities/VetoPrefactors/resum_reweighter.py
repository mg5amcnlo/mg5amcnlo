#! /usr/bin/env python

################################################################################
#
# Script intended to reweight events generated using aMC@LO and
# already reweighted to include the virtual corrections (without an
# overall alpha_s/2pi) with the corresponding Resummation
# contributions. No secondary unweighting is performed.
#
################################################################################

import os
import re
import subprocess
import sys
import time
import logging
import glob
import copy
import math
pjoin = os.path.join
proc_path = os.path.abspath(pjoin(os.path.dirname(os.path.realpath( __file__ )),os.pardir,os.pardir))
MGRootPath = pjoin(proc_path,os.pardir)
sys.path.append(MGRootPath)
sys.path.append(pjoin(proc_path,'bin'))
import madgraph.various.lhe_parser as lhe_parser
import models.import_ufo as import_ufo
import banner
import madgraph.various.misc as misc
from madgraph.iolibs.files import cp
import madgraph.various.process_checks as process_checks

logging.basicConfig(level=logging.INFO)
logging.getLogger('madgraph.model').setLevel(logging.CRITICAL)

# ==============================================
# Helper functions
# ==============================================

def setup_channel(channelPath):
    """ Compiles the Resummation fortran steering 
    code and returns the corresponding process."""

    logging.info('Setting up resummation runner in\n   %s'%channelPath)

    try:
        misc.compile(cwd=channelPath, mode='fortran', job_specs = False)
    except:
        logging.error("Could not compile resummation runner in %s/%s"%(channelPath))
        raise
    
    logging.info('  ... Done.')
    
    return subprocess.Popen([os.path.join(channelPath,'resum_driver')], 
          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                                   cwd=channelPath)

def kill_processes(encountered_channels):
    """ Kills the MadLoop processes open """

    for proc in encountered_channels.values():
        try:
            proc[0].stdin.write('y\n')
        except:
            proc[0].kill()

def get_value(Runner, input_str):
    """ This version of get_me_value is simplified for the purpose of this
    class. No compilation is necessary. The CT mode can be specified."""
    Runner.stdin.write(input_str)
    output = Runner.stdout.readline()  
    return output

# ==============================================
# Start program
# ==============================================

# Specify here the path of the file to reweight
if len(sys.argv)>0:
  eventFile=sys.argv[1]
else:
    logging.error("Could not process the arguments. The arguments are: ./resum_reweighter.py <eventFilePath> <outputEvtName>")
    exit()

outSpecified=False
evtFileName = '.'.join(os.path.basename(eventFile).split('.')[:-1])
evtFileExt  = os.path.basename(eventFile).split('.')[-1]
try:
  outPath=pjoin(os.path.dirname(os.path.realpath(eventFile)),sys.argv[2])
  outSpecified=True
except:
  outPath=pjoin(os.path.dirname(os.path.realpath(eventFile)),'%s_rwgt.%s'%(evtFileName,evtFileExt))

i=0
while os.path.exists(outPath):
    i=i+1    
    if outSpecified and i==1:
        logging.warning("The chosen filename %s for the output already exists."%os.path.basename(outPath))
    outPath=pjoin(os.path.dirname(os.path.realpath(eventFile)),
                  '%s_rwgt_%i.%s'%(evtFileName,i,evtFileExt))

try:
    evtBanner = banner.Banner(eventFile)    
    evtFile = lhe_parser.EventFile(eventFile)
except:
    logging.error("Could not read event file %s."%eventFile)

# Detect the total number of event in an efficient way
n_evts = int(subprocess.Popen('grep -rin "<event>" %s | wc -l'%os.path.realpath(eventFile),
                        stdout=subprocess.PIPE, shell=True).communicate()[0])

logging.info("Writing out the reweighted event file on\n   %s"%outPath)
# Write out the output event file
outputEvtFile = open(outPath,'w')
outputEvtFile.write(evtFile.banner)
# List the channel present in the process output
channel_list = [ os.path.basename(chan) for chan in \
                 misc.glob('P*', pjoin(proc_path, 'SubProcesses')) ]

# Now scan over events
# For each encountered channel, store the corresponding process runner
encountered_channels=dict([])

ebeam1=float( evtBanner.get('run_card','ebeam1'))
ebeam2=float( evtBanner.get('run_card','ebeam2'))

t_before = time.time()
i_evt=0
wgt_summed = 0.0
squared_wgt_summed = 0.0
running_spread = 0.0
max_wgt = 0.0
n_percent_monitor=1
for event in evtFile:
    i_evt = i_evt+1
    t_evt_start = time.time()    
    # The channel Folder
    chanFolder=""
    EvtChannelName="VetoPrefactors"
    # The Channel name to setup, if needed
    ChannelNameToSetup=""

    # =======================================
    # Setup the runner if not already present
    # =======================================

    try:
        runner=encountered_channels[EvtChannelName][0]
    except KeyError:
        logging.info('Found new channel %s'%EvtChannelName)
        virtFolderPath=pjoin(os.path.dirname(os.path.realpath(eventFile)),EvtChannelName)
        try:
            runner=setup_channel(virtFolderPath)
        except:
            logging.error("Could not setup resum runner %s"%virtFolderPath)
            kill_processes(encountered_channels)
            exit()
    
            # Save the runners just created for ChannelNameToSetup and possibly related to
            # EvtChannelName as well.
        encountered_channels[EvtChannelName]=(runner,virtFolderPath)

    # =======================================
    # End of virtual runner setup
    # =======================================

    momenta=[[p.E,p.px,p.py,p.pz] for p in event if p.status==-1]
    pdg=[int(p.pid) for p in event if p.status==-1]

    event.parse_reweight()
    born_wgt=event.reweight_data['1001']

# PDG codes of the incoming particles
    f1=str(pdg[0])
    f2=str(pdg[1])
# Bjorken x's of the particles
    x1=str(momenta[0][0]/ebeam1)
    x2=str(momenta[1][0]/ebeam2) 
# Q is set to \sqrt(\hat{s})
    Q =str(2*math.sqrt(momenta[0][0]*momenta[1][0]))
# renormalisation scale used to generate the events
    muMad=str(event.scale)

    input_string=f1+' '+f2+' '+x1+' '+x2+' '+Q+' '+muMad+'\n'
    trial  = 0
    max_trial = 50
    worked = False
    new_wgt = 0.0

    output_string = get_value(runner,input_string)
    mode=float(output_string.split()[0])
    BCorr0=float(output_string.split()[1])
    BCorr1=float(output_string.split()[2])
    BCorrm0=float(output_string.split()[3])
    BCorrm1=float(output_string.split()[4])
    Efull=float(output_string.split()[5])
    EfullNLL=float(output_string.split()[6])
    Hcomp_fact=float(output_string.split()[7])
    Hcomp_factm=float(output_string.split()[8])
    alphah=float(output_string.split()[9])
    alpham=float(output_string.split()[10])
    EfO=float(output_string.split()[11])
    EfONLL=float(output_string.split()[12])
    alpha=float(output_string.split()[13])
    if mode==1:
        new_wgt= (1+alpha*BCorr1+(alphah/(2.0*math.pi))*(event.wgt/born_wgt+Hcomp_fact))*BCorr0*Efull*born_wgt
    if mode==2:
        new_wgt= BCorr0*EfullNLL*born_wgt
    if mode==3:
        new_wgt= (1+BCorrm1+EfO+(alpham/(2.0*math.pi))*(event.wgt/born_wgt+Hcomp_factm))*BCorrm0*born_wgt
    if mode==4:
        new_wgt= BCorrm0*(1+EfONLL)*born_wgt

    worked = True

    if i_evt<=5:
        logging.info('Event %i processed in %.2E seconds.'%(i_evt,time.time()-t_evt_start))
    if i_evt==5:
        logging.info('Further output of timing event by event suppressed.')

    event.wgt=new_wgt
    wgt_summed = wgt_summed + new_wgt
    squared_wgt_summed = squared_wgt_summed + new_wgt**2
    max_wgt = max(max_wgt,abs(new_wgt))
    if (i_evt/float(n_evts))>n_percent_monitor*0.01:
        logging.info('%i%% of %i events processed in %.2E seconds. Cross section: %.4E pb'\
                     %(n_percent_monitor,n_evts,(time.time()-t_before),wgt_summed/i_evt))
        n_percent_monitor=n_percent_monitor+1
    # Only start computing the cumulative spread after the first 10% of events.
    if n_percent_monitor>1:
        running_spread = running_spread + (new_wgt - float(wgt_summed/i_evt))**2
    outputEvtFile.write(str(event))

# Clean up things
outputEvtFile.close()
kill_processes(encountered_channels)

# Finish
logging.info("Successfully reweighted %i events in %.2E seconds in file\n  %s."\
    %(n_evts,(time.time()-t_before),outPath))

# Give statistics on the weight
logging.info("Resummation weight statistics,\n  avg       : %.2E\n  cum. var. : %.2E\n  max       : %.2E"\
    %((wgt_summed/n_evts),math.sqrt(running_spread/n_evts),max_wgt))

# Give statistics on the weight
logging.info("Resummation cross-section obtained\n  %.6E  +/- %.2E"\
    %((wgt_summed/n_evts),math.sqrt(squared_wgt_summed)/n_evts))
