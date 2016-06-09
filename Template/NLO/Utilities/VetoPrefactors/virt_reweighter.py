#! /usr/bin/env python

################################################################################
#
# Script intended to reweight events generated in the aMC@LO with the
# corresponding virtual Matrix Elements. No secondary unweighting is performed.
# It can be run from any aMC@NLO running output which includes MadLoop virtuals
# in the V0_<proc> directories in the P<i>_* subprocesses.
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

# Particle PDG to shell name mapping 
def PDGToShellName(PDG):
    """Returns the shell name of particle identified by PDG"""
    partName=user_model.get('particle_dict')[PDG].get_name()
    # Replace '~' with 'x'
    partName = partName.replace('~', 'x')
    # Replace '+' with 'p'
    partName = partName.replace('+', 'p')
    # Replace '-' with 'm'
    partName = partName.replace('-', 'm')
    # Just to be safe, remove all spaces
    partName = partName.replace(' ', '')
    return partName

# Process identified by list of PDGs to shell name mapping 
def channelToShellName(in_pdgs,out_pdgs):
    return ''.join([PDGToShellName(in_pdg) for in_pdg in in_pdgs])+'_'+\
           ''.join([PDGToShellName(out_pdg) for out_pdg in out_pdgs])

# Alternative names for the process if the channels with different flavors
# for the massless quarks are put together under the denomination u, ux, d, dx
def channelToAlternativeShellNames(in_pdgs,out_pdgs):
    mapping={-3:-1,-4:-2,3:1,4:2,-5:-1,5:1}
    new_in_pdgs=[]
    new_out_pdgs=[]
    
    for pdg in in_pdgs:
        try:
            new_in_pdgs.append(mapping[pdg])
        except KeyError:
            new_in_pdgs.append(pdg)

    for pdg in out_pdgs:
        try:
            new_out_pdgs.append(mapping[pdg])
        except KeyError:
            new_out_pdgs.append(pdg)

    # For now return only one alternative, but could add others
    return [channelToShellName(new_in_pdgs,new_out_pdgs)]

def setup_channel(channelPath):
    """ Copies and compile the MadLoop fortran steering 
    code and returns the corresponding process."""

    logging.info('Setting up virtual runner in\n   %s'%channelPath)

    # Use the presence of the file born_matrix.f to check if this output
    # is a loop_induced one or not.
    if os.path.isfile(os.path.join(channelPath,'born_matrix.f')):
        checkerName = 'StabilityCheckDriver.f'
    else:
        checkerName = 'StabilityCheckDriver_loop_induced.f'                
    # First create the stability check fortran driver executable if not 
    # already present.
    if not os.path.isfile(pjoin(channelPath,'StabilityCheckDriver.f')):
        with open(pjoin(MGRootPath,'Template','loop_material','Checks',
                        'StabilityCheckDriver.f'),'r') as checkerFile:
            with open(pjoin(channelPath,'proc_prefix.txt')) as proc_prefix:
                checkerToWrite = checkerFile.read()%{'proc_prefix':
                                                     proc_prefix.read()}
        checkerFile = open(pjoin(channelPath,'StabilityCheckDriver.f'),'w')
        checkerFile.write(checkerToWrite)
        checkerFile.close()                
    # Append the compilation of the StabilityCheckDriver to the makefile
        with open (pjoin(channelPath,'makefile'),'a') as makefile:
            makefile.write("\nStabilityCheckDriver:  StabilityCheckDriver.o $(PROCESS)\n\t$(FC) $(FFLAGS) -o StabilityCheckDriver StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)")
    try:
        misc.compile(arg=['StabilityCheckDriver'], \
                     cwd=channelPath, mode='fortran', job_specs = False)
    except:
        logging.error("Could not compile %s/%s"%(channelPath,StabilityCheckDriver))
        raise
    
    logging.info('  ... Done.')
    
    return subprocess.Popen([os.path.join(channelPath,'StabilityCheckDriver')], 
          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                                   cwd=channelPath)

def kill_processes(encountered_channels):
    """ Kills the MadLoop processes open """

    for proc in encountered_channels.values():
        try:
            proc[0].stdin.write('y\n')
        except:
            proc[0].kill()

# ==============================================
# Start program
# ==============================================

# Specify here the path of the file to reweight
if len(sys.argv)>0:
  eventFile=sys.argv[1]
else:
    logging.error("Could not process the arguments. The arguments are: ./virt_reweighter.py <eventFilePath> <outputEvtName>")
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

# Start by loading the model of the banner
try:
    model_name=evtBanner.get_detail('model')
    model_name=model_name.split('-')[0]
    # For now use the model 'sm' no matter what since it is only used for the particle naming scheme
    model_name='sm'
    user_model = import_ufo.import_full_model(pjoin(proc_path,os.pardir,'models',model_name))
except:
    logging.error("Could not load the model %s used for generating the event file"\
                 %evtBanner.get_detail('model'))
    exit()

# Detect the total number of event in an efficient way
n_evts = int(subprocess.Popen('grep -rin "<event>" %s | wc -l'%os.path.realpath(eventFile),
                        stdout=subprocess.PIPE, shell=True).communicate()[0])

logging.info("Writing out the reweighted event file on\n   %s"%outPath)
# Write out the output event file
outputEvtFile = open(outPath,'w')
outputEvtFile.write(evtFile.banner)

# List the channel present in the process output
channel_list = [ os.path.basename(chan) for chan in \
                 misc.glob('P*', pjoin(proc_path, 'SubProcesses'))) ]

# Now scan over events
# For each encountered channel, store the corresponding process runner
encountered_channels=dict([])

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
    # The channel name of the particular event
    EvtChannelName=channelToShellName(
        [int(p.pid) for p in event if p.status==-1],
        [int(p.pid) for p in event if p.status==1])
    # The possible alternative channel name it is related to.
    RelatedChannelName=""
    # The Channel name to setup, if needed
    ChannelNameToSetup=""

    # =======================================
    # Setup the runner if not already present
    # =======================================

    try:
        runner=encountered_channels[EvtChannelName][0]
    except KeyError:
        logging.info('Found new channel %s'%EvtChannelName)
        chanFinder=re.compile("^P[0-9]*\S*%s\S*$"%EvtChannelName)
        match_channels=[chan for chan in channel_list if \
                        not chanFinder.match(chan) is None]
        # If not found try to consider u and d quark initial state as
        # a general tag for any massless quark of the corresponding electric charge.
        if len(match_channels)==0:
            channelNames=channelToAlternativeShellNames(
                [int(p.pid) for p in event if p.status==-1],
                [int(p.pid) for p in event if p.status==1])
            for chanName in channelNames:
                chanFinder=re.compile("^P[0-9]*\S*%s\S*$"%chanName)
                match_channels=[chan for chan in channel_list if \
                                    not chanFinder.match(chan) is None]
                if len(match_channels)>1:
                    logging.error("There are %i (more than one!) channels matching "\
                      %len(match_channels)+\
                      "for %s (related to %s found in the event file) in \n => %s"\
                      %(chanName,EvtChannelName,str(pjoin(proc_path, 'SubProcesses'))))
                    kill_processes(encountered_channels)                    
                    exit()
                elif len(match_channels)==1:
                    chanFolder=match_channels[0]
                    RelatedChannelName=chanName
                    logging.info("Relating channel %s (found in event file) to %s"\
                    %(EvtChannelName,chanFolder))
                    break

            if chanFolder=="":
                logging.error("No matching channel for %s in %s."\
                %(EvtChannelName,str(pjoin(proc_path, 'SubProcesses'))))
                kill_processes(encountered_channels)                
                exit()
            else:
                try:
                    runner=encountered_channels[RelatedChannelName][0]
                    encountered_channels[EvtChannelName]=\
                            encountered_channels[RelatedChannelName]
                except:
                    ChannelNameToSetup=RelatedChannelName
 
        elif len(match_channels)>1:
            logging.error("There are %i (more than one!) channels matching for %s in \n => %s."\
                %(len(match_channels),channelName,str(pjoin(proc_path, 'SubProcesses'))))
            kill_processes(encountered_channels)            
            exit()
        else:
            chanFolder=match_channels[0]
            ChannelNameToSetup=EvtChannelName

        if ChannelNameToSetup!="":
            virtFolders = [ os.path.basename(virt) for virt in \
                misc.glob('V*%s*'%ChannelNameToSetup, pjoin(proc_path, 'SubProcesses',
                chanFolder)) ]
            if len(virtFolders)!=1:
                if len(virtFolders)==0:
                    logging.error("No virtual folder V*%s* found in %s."\
                    %(ChannelNameToSetup,str(pjoin(proc_path, 'SubProcesses',chanFolder))))
                else:
                    logging.error("There are %i (more than one!) virtual folders V*%s* in \n => %s."\
                    %(len(virtFolders),ChannelNameToSetup,str(pjoin(proc_path, 'SubProcesses',chanFolder))))
                exit()

            virtFolderPath=pjoin(proc_path, 'SubProcesses',chanFolder,virtFolders[0])
            try:
                runner=setup_channel(virtFolderPath)
            except:
                logging.error("Could not setup virtual runner %s"%virtFolderPath)
                kill_processes(encountered_channels)
                exit()
    
            # Save the runners just created for ChannelNameToSetup and possibly related to
            # EvtChannelName as well.
            encountered_channels[ChannelNameToSetup]=(runner,virtFolderPath)
            if RelatedChannelName != "":
                encountered_channels[EvtChannelName]=\
                   encountered_channels[ChannelNameToSetup]

    # =======================================
    # End of virtual runner setup
    # =======================================

    momenta=[[p.E,p.px,p.py,p.pz] for p in event if p.status==-1]+\
            [[p.E,p.px,p.py,p.pz] for p in event if p.status==1]
   
    trial  = 0
    max_trial = 50
    worked = False
    new_wgt = 0.0
    while not worked and trial<max_trial:
        trial = trial+1
        try:
            new_wgt = process_checks.LoopMatrixElementTimer.get_me_value(
                runner,
                '\n'.join([' '.join(['%.16E'%pi for pi in p]) for p in momenta]),
                -1,
                mu_r=event.scale)
            # The result above was finite/(BORN*AO2PI)
#            new_wgt=((new_wgt*event.aqcd)/(2.0*math.pi))*event.wgt
# do not include alpha_S/2pi
            new_wgt=new_wgt*event.wgt
            worked = True
        except IOError:
            worked = False
            if trial==1:
                logging.warning('The python-fortran cross-talk yielded an IOError. Retrying')
            try:
                runner.kill()
            except:
                pass
            time.sleep(0.5)
            virtPath=encountered_channels[EvtChannelName][1]
            runner = subprocess.Popen([pjoin(virtPath,'StabilityCheckDriver')], 
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, cwd=virtPath)
            encountered_channels[EvtChannelName]=(runner,virtPath)
    
    if not worked:
        logging.error('The python fortran cross-talk failed %i times consecutively. Aborting run'%max_trial)
        kill_processes(encountered_channels)
        exit()

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
logging.info("Virtual weight statistics,\n  avg       : %.2E\n  cum. var. : %.2E\n  max       : %.2E"\
    %((wgt_summed/n_evts),math.sqrt(running_spread/n_evts),max_wgt))

# Give statistics on the weight
logging.info("Virtual cross-section obtained\n  %.6E  +/- %.2E"\
    %((wgt_summed/n_evts),math.sqrt(squared_wgt_summed)/n_evts))
