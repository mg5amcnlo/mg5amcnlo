################################################################################
# Copyright (c) 2009 The MadGraph Development team and Contributors             
#
# This file is a part of the MadGraph 5 project, an application which           
# automatically generates Feynman diagrams and matrix elements for arbitrary    
# high-energy processes in the Standard Model and beyond.                       
#
# It is subject to the MadGraph license which should accompany this             
# distribution.                                                                 
#                                                                               
# For more information, please visit: http://madgraph.phys.ucl.ac.be            
#                                                                               
################################################################################
import subprocess
import logging
import hashlib
import os
import time
import re

logger = logging.getLogger('madgraph.cluster') 

try:
    from madgraph import MadGraph5Error
except:
    from internal import MadGraph5Error
    
class ClusterManagmentError(MadGraph5Error):
    pass



class Cluster(object):
    """Basic Class for all cluster type submission"""
    name = 'mother class'

    def __init__(self, cluster_queue=None):
        """Init the cluster"""
        self.submitted = 0
        self.submitted_ids = []
        self.finish = 0
        self.cluster_queue = cluster_queue

    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """How to make one submission. Return status id on the cluster."""
        raise NotImplemented, 'No implementation of how to submit a job to cluster \'%s\'' % self.name

    def control(self, me_dir=None):
        """Check the status of job associated to directory me_dir. return (idle, run, finish, fail)"""
        if not self.submitted_ids:
            raise NotImplemented, 'No implementation of how to control the job status to cluster \'%s\'' % self.name
        idle, run, fail = 0, 0, 0
        for id in self.submitted_ids[:]:
            status = self.control_one_job(id)
            if status == 'I':
                idle += 1
            elif status == 'R':
                run += 1
            elif status == 'F':
                self.finish +=1
                self.submitted_ids.remove(id)
            else:
                fail += 1

        return idle, run, self.finish, fail

    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        raise NotImplemented, 'No implementation of how to control the job status to cluster \'%s\'' % self.name

    def wait(self, me_dir, fct):
        """Wait that all job are finish"""
        
        while 1: 
            idle, run, finish, fail = self.control(me_dir)
            if fail:
                raise ClusterManagmentError('Some Jobs are in a Hold/... state. Please try to investigate or contact the IT team')
            if idle + run == 0:
                logger.info('All jobs finished')
                break
            fct(idle, run, finish)
            time.sleep(30)
        self.submitted = 0
        self.submitted_ids = []

    def launch_and_wait(self, prog, argument=[], cwd=None, stdout=None, 
                                                         stderr=None, log=None):
        """launch one job on the cluster and wait for it"""
        id = self.submit(prog, argument, cwd, stdout, stderr, log)
        while 1:        
            status = self.control_one_job(id)
            if not status in ['R','I']:
                break
            time.sleep(30)

class CondorCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'condor'

    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """Submit the """
        
        text = """Executable = %(prog)s
                  output = %(stdout)s
                  error = %(stderr)s
                  log = %(log)s
                  %(argument)s
                  Universe = vanilla
                  notification = Error
                  Initialdir = %(cwd)s
                  %(requirement)s
                  queue 1
               """
        
        if self.cluster_queue not in ['None', None]:
            requirement = 'Requirements = %s=?=True' % self.cluster_queue
        else:
            requirement = ''

        if cwd is None:
            cwd = os.getcwd()
        if stdout is None:
            stdout = '/dev/null'
        if stderr is None:
            stderr = '/dev/null'
        if log is None:
            log = '/dev/null'
        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)
        if argument:
            argument = 'Arguments = %s' % ' '.join(argument)
        else:
            argument = ''
        

        dico = {'prog': prog, 'cwd': cwd, 'stdout': stdout, 
                'stderr': stderr,'log': log,'argument': argument,
                'requirement': requirement}

        open('submit_condor','w').write(text % dico)
        a = subprocess.Popen(['condor_submit','submit_condor'], stdout=subprocess.PIPE)
        output = a.stdout.read()
        #Submitting job(s).
        #Logging submit event(s).
        #1 job(s) submitted to cluster 2253622.
        pat = re.compile("submitted to cluster (\d*)",re.MULTILINE)
        try:
            id = pat.search(output).groups()[0]
        except:
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        self.submitted_ids.append(id)
        return id

    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'condor_q '+str(id)+" -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        return status.stdout.readline().strip()
        
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        
        if not self.submitted_ids:
            return 0, 0, 0, 0
        
        cmd = "condor_q " + ' '.join(self.submitted_ids) + " -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            status = line.strip()
            if status == 'I':
                idle += 1
            elif status == 'R':
                run += 1
            else:
                fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail

class PBSCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'pbs'
    idle_tag = ['Q']
    running_tag = ['T','E','R']


    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """Submit the prog to the cluser"""
        
        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = hashlib.md5(me_dir).hexdigest()[-14:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]
        
        text = ""
        if cwd is None:
            cwd = os.getcwd()
        else: 
            text = " cd %s;" % cwd
        if stdout is None:
            stdout = '/dev/null'
        if stderr is None:
            stderr = '/dev/null'
        elif stderr == -2: # -2 is subprocess.STDOUT
            stderr = stdout
        if log is None:
            log = '/dev/null'
        
        text += prog
        if argument:
            text += ' ' + ' '.join(argument)

        a = subprocess.Popen(['qsub','-o', stdout,
                                     '-N', me_dir, 
                                     '-e', stderr,
                                     '-q', self.cluster_queue,
                                     '-V'], stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE, cwd=cwd)
            
        output = a.communicate(text)[0]
        id = output.split('.')[0]
        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        return id

    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'qstat '+str(id)
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        
        for line in status.stdout:
            line = line.strip()
            if 'Unknown' in line:
                return 'F'
            elif line.startswith(str(id)):
                status = line.split()[4]
        if status in self.idle_tag:
            return 'I' 
        elif status in self.running_tag:                
            return 'R' 
        return 'F'
        

        
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        cmd = "qstat"
        status = subprocess.Popen([cmd], stdout=subprocess.PIPE)

        if me_dir.endswith('/'):
           me_dir = me_dir[:-1]    
        me_dir = hashlib.md5(me_dir).hexdigest()[-14:]
        if not me_dir[0].isalpha():
                  me_dir = 'a' + me_dir[1:]

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            if me_dir in line:
                status = line.split()[4]
                if status in self.idle_tag:
                    idle += 1
                elif status in self.running_tag:
                    run += 1
                else:
                    print line
                    fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail

class SGECluster(Cluster):
    """Basic class for dealing with cluster submission"""
    # Class written by Arian Abrahantes.

    name = 'sge'
    idle_tag = ['qw', 'hqw','hRqw','w']
    running_tag = ['r','t','Rr','Rt']

    def def_get_path(self,location):
        """replace string for path issues"""
        location = os.path.realpath(location)
        homePath = os.getenv("HOME")
        if homePath:
            location = location.replace(homePath,'$HOME')
        return location

    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """Submit the prog to the cluser"""

        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = hashlib.md5(me_dir).hexdigest()[-10:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]

        text = ""
        if cwd is None:
           #cwd = os.getcwd()
           cwd = self.def_get_path(os.getcwd())
        else: 
           text = " cd %s;" % cwd
        cwd1 = self.def_get_path(cwd)
        text = " cd %s;" % cwd1
        if stdout is None:
            stdout = '/dev/null'
        else:
            stdout = self.def_get_path(stdout)
        if stderr is None:
            stderr = '/dev/null'
        elif stderr == -2: # -2 is subprocess.STDOUT
            stderr = stdout
        if log is None:
            log = '/dev/null'
        else:
            log = self.def_get_path(log)

        text += prog
        if argument:
            text += ' ' + ' '.join(argument)

        #if anything slips through argument
        #print "!=== inteded change ",text.replace('/srv/nfs','')
        #text = text.replace('/srv/nfs','')
        homePath = os.getenv("HOME")
        if homePath:
            text = text.replace(homePath,'$HOME')

        logger.debug("!=== input  %s" % text)
        logger.debug("!=== output %s" %  stdout)
        logger.debug("!=== error  %s" % stderr)
        logger.debug("!=== logs   %s" % log)

        a = subprocess.Popen(['qsub','-o', stdout,
                                    '-N', me_dir, 
                                    '-e', stderr,
                                    '-q', self.cluster_queue,
                                    '-V'], stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT,
                                    stdin=subprocess.PIPE, cwd=cwd)

        output = a.communicate(text)[0]
        id = output.split(' ')[2]
        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        logger.debug(output)

        return id

    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        #cmd = 'qstat '+str(id)
        cmd = 'qstat '
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        for line in status.stdout:
            #print "!==",line
            #line = line.strip()
            #if 'Unknown' in line:
            #    return 'F'
            #elif line.startswith(str(id)):
            #    status = line.split()[4]
            if str(id) in line:
                status = line.split()[4]
                #print "!=status", status
        if status in self.idle_tag:
            return 'I' 
        elif status in self.running_tag:                
            return 'R' 
        return 'F'

    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        cmd = "qstat "
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

        if me_dir.endswith('/'):
           me_dir = me_dir[:-1]    
        me_dir = hashlib.md5(me_dir).hexdigest()[-10:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            if me_dir in line:
                status = line.split()[4]
                if status in self.idle_tag:
                    idle += 1
                elif status in self.running_tag:
                    run += 1
                else:
                    logger.debug(line)
                    fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail


from_name = {'condor':CondorCluster, 'pbs': PBSCluster, 'sge': SGECluster}


    
