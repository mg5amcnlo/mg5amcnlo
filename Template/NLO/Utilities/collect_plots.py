from __future__ import absolute_import
from __future__ import print_function
import pickle
import subprocess
import sys

def combine_plots_HwU(jobs,out,normalisation=None):
    """Sums all the plots in the HwU format."""
    command =  [sys.executable]
    command.append('bin/internal/histograms.py')
    for job in jobs:
        if job['dirname'].endswith('.HwU'):
            command.append(job['dirname'])
        else:
            command.append(job['dirname']+'/MADatNLO.HwU')
    command.append("--out="+out)
    command.append("--gnuplot")
    command.append("--band=[]")
    if normalisation:
        command.append("--multiply="+(','.join([str(n) for n in normalisation])))
    command.append("--sum")
    command.append("--keep_all_weights")
    command.append("--no_open")

    p = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        
    while p.poll() is None:
        line = p.stdout.readline()
        if any(t in line for t in ['INFO:','WARNING:','CRITICAL:','ERROR:','KEEP:']):
            print(line[:-1])

with open("SubProcesses/job_status.pkl",'rb') as f:
    jobs_to_collect=pickle.load(f)

combine_plots_HwU(jobs_to_collect,'MADatNLO_combined')
