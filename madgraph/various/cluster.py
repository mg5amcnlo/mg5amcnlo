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

class NotImplemented(MadGraph5Error):
    pass

def multiple_try(nb_try=5, sleep=20):

    def deco_retry(f):
        def deco_f_retry(*args, **opt):
            for i in range(nb_try):
                try:
                    return f(*args, **opt)
                except KeyboardInterrupt:
                    raise
                except:
                    time.sleep(sleep * (i+1))
            raise
        return deco_f_retry
    return deco_retry

def check_interupt(error=KeyboardInterrupt):

    def deco_interupt(f):
        def deco_f_interupt(self, *args, **opt):
            try:
                return f(self, *args, **opt)
            except error:
                self.remove(*args, **opt)
                raise error
        return deco_f_interupt
    return deco_interupt

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

    @check_interupt()
    def wait(self, me_dir, fct):
        """Wait that all job are finish"""
        
        while 1: 
            idle, run, finish, fail = self.control(me_dir)
            if fail:
                raise ClusterManagmentError('Some Jobs are in a Hold/... state. Please try to investigate or contact the IT team')
            if idle + run == 0:
                time.sleep(20) #security to ensure that the file are really written on the disk
                logger.info('All jobs finished')
                break
            fct(idle, run, finish)
            time.sleep(30)
        self.submitted = 0
        self.submitted_ids = []

    @check_interupt()
    def launch_and_wait(self, prog, argument=[], cwd=None, stdout=None, 
                                                         stderr=None, log=None):
        """launch one job on the cluster and wait for it"""
        
        special_output = False # tag for concatanate the error with the output.
        if stderr == -2 and stdout: 
            #We are suppose to send the output to stdout
            special_output = True
            stderr = stdout + '.err'
        id = self.submit(prog, argument, cwd, stdout, stderr, log)
        while 1:        
            status = self.control_one_job(id)
            if not status in ['R','I']:
                time.sleep(20) #security to ensure that the file are really written on the disk
                break
            time.sleep(30)
        
        if special_output:
            # combine the stdout and the stderr
            #wait up to 50 s to see if those files exists
            for i in range(5):
                if os.path.exists(stdout):
                    if not os.path.exists(stderr):
                        time.sleep(5)
                    if os.path.exists(stderr):
                        err_text = open(stderr).read()
                        if not err_text:
                            return
                        logger.warning(err_text)                        
                        text = open(stdout).read()
                        open(stdout,'w').write(text + err_text)
                    else:
                        return
                time.sleep(10)
                        
    def remove(self, *args):
        """ """
        logger.warning("""This cluster didn't support job removal, 
    the jobs are still running on the cluster.""")

class CondorCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'condor'

    @multiple_try()
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
                  getenv=True
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
    
    @multiple_try()
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'condor_q '+str(id)+" -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)
        if status.returncode:
            raise ClusterManagmentError, 'condor_q returns error: %s' % \
                                                            status.stderr.read()

        return status.stdout.readline().strip()
    
    @check_interupt()
    @multiple_try()
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        
        if not self.submitted_ids:
            return 0, 0, 0, 0
        
        cmd = "condor_q " + ' '.join(self.submitted_ids) + " -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)

        if status.returncode:
            raise ClusterManagmentError, 'condor_q returns error: %s' % \
                                                            status.stderr.read()
            
            
        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            status = line.strip()
            if status in ['I','U']:
                idle += 1
            elif status == 'R':
                run += 1
            elif status != 'C':
                fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail
    
    @multiple_try()
    def remove(self, *args):
        """Clean the jobson the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "condor_rm %s" % ' '.join(self.submitted_ids)
        
        status = subprocess.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))
        
class PBSCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'pbs'
    idle_tag = ['Q']
    running_tag = ['T','E','R']
    complete_tag = ['C']

    @multiple_try()
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

        command = ['qsub','-o', stdout,
                   '-N', me_dir, 
                   '-e', stderr,
                   '-V']

        if self.cluster_queue and self.cluster_queue != 'None':
            command.extend(['-q', self.cluster_queue])

        a = subprocess.Popen(command, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT,
                                      stdin=subprocess.PIPE, cwd=cwd)
            
        output = a.communicate(text)[0]
        id = output.split('.')[0]
        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        self.submitted_ids.append(id)
        return id

    @multiple_try()
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
        

    @multiple_try()    
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
                elif status in self.complete_tag:
                    continue
                else:
                    fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = subprocess.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


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

    @multiple_try()
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

        command = ['qsub','-o', stdout,
                   '-N', me_dir, 
                   '-e', stderr,
                   '-V']

        if self.cluster_queue and self.cluster_queue != 'None':
            command.extend(['-q', self.cluster_queue])

        a = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE, cwd=cwd)

        output = a.communicate(text)[0]
        id = output.split(' ')[2]
        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        self.submitted_ids.append(id)
        logger.debug(output)

        return id

    @multiple_try()
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

    @multiple_try()
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

    
    
    @multiple_try()
    def remove(self, *args):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = subprocess.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


class LSFCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'lsf'

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """Submit the """
        
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

        command = ['bsub','-o', stdout,
                   '-J', me_dir, 
                   '-e', stderr]

        if self.cluster_queue and self.cluster_queue != 'None':
            command.extend(['-q', self.cluster_queue])

        a = subprocess.Popen(command, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT,
                                      stdin=subprocess.PIPE, cwd=cwd)
            
        output = a.communicate(text)[0]
        #Job <nnnn> is submitted to default queue <normal>.
        try:
            id = output.split('>',1)[0].split('<')[1]
        except:
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        self.submitted_ids.append(id)
        return id        
        
        
    @multiple_try()
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        
        cmd = 'bjobs '+str(id)
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        
        for line in status.stdout:
            line = line.strip().upper()
            if 'JOBID' in line:
                continue
            elif str(id) not in line:
                continue
            status = line.split()[2]
            if status == 'RUN':
                return 'R'
            elif status == 'PEND':
                return 'I'
            elif status == 'DONE':
                return 'F'
            else:
                return 'H'
            return 'F'

    @multiple_try()   
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        
        if not self.submitted_ids:
            return 0, 0, 0, 0
        
        cmd = "bjobs " + ' '.join(self.submitted_ids) 
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            line = line.strip()
            if 'JOBID' in line:
                continue
            splitline = line.split()
            id = splitline[0]
            if id not in self.submitted_ids:
                continue
            status = splitline[2]
            if status == 'RUN':
                run += 1
            elif status == 'PEND':
                idle += 1
            elif status == 'DONE':
                pass
            else:
                fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "bdel %s" % ' '.join(self.submitted_ids)
        status = subprocess.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))

class GECluster(Cluster):
    """Class for dealing with cluster submission on a GE cluster"""
    
    name = 'ge'
    idle_tag = ['qw']
    running_tag = ['r']

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None):
        """Submit the prog to the cluser"""
        
        text = ""
        if cwd is None:
            cwd = os.getcwd()
        else: 
            text = " cd %s; bash " % cwd
        if stdout is None:
            stdout = os.path.join(cwd, "log.%s" % prog.split('/')[-1])
        if stderr is None:
            stderr = os.path.join(cwd, "err.%s" % prog.split('/')[-1])
        elif stderr == -2: # -2 is subprocess.STDOUT
            stderr = stdout
        if log is None:
            log = '/dev/null'

        text += prog
        if argument:
            text += ' ' + ' '.join(argument)
        text += '\n'
        tmp_submit = os.path.join(cwd, 'tmp_submit')
        open(tmp_submit,'w').write(text)

        a = subprocess.Popen(['qsub','-o', stdout,
                                     '-e', stderr,
                                     tmp_submit],
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE, cwd=cwd)

        output = a.communicate()[0]
        #Your job 874511 ("test.sh") has been submitted
        pat = re.compile("Your job (\d*) \(",re.MULTILINE)
        try:
            id = pat.search(output).groups()[0]
        except:
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \
                                                                        % output 
        self.submitted += 1
        self.submitted_ids.append(id)
        return id

    @multiple_try()
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'qstat | grep '+str(id)
        status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        if not status:
            return 'F'
        #874516 0.00000 test.sh    alwall       qw    03/04/2012 22:30:35                                    1
        pat = re.compile("^(\d+)\s+[\d\.]+\s+[\w\d\.]+\s+[\w\d\.]+\s+(\w+)\s")
        stat = ''
        for line in status.stdout.read().split('\n'):
            if not line:
                continue
            line = line.strip()
            try:
                groups = pat.search(line).groups()
            except:
                raise ClusterManagmentError, 'bad syntax for stat: \n\"%s\"' % line
            if groups[0] != id: continue
            stat = groups[1]
        if not stat:
            return 'F'
        if stat in self.idle_tag:
            return 'I' 
        if stat in self.running_tag:                
            return 'R' 
        
    @multiple_try()
    def control(self, me_dir=None):
        """Check the status of job associated to directory me_dir. return (idle, run, finish, fail)"""
        if not self.submitted_ids:
            return 0, 0, 0, 0
        idle, run, fail = 0, 0, 0
        ongoing = []
        for statusflag in ['p', 'r', 'sh']:
            cmd = 'qstat -s %s' % statusflag
            status = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
            #874516 0.00000 test.sh    alwall       qw    03/04/2012 22:30:35                                    1
            pat = re.compile("^(\d+)")
            for line in status.stdout.read().split('\n'):
                line = line.strip()
                try:
                    id = pat.search(line).groups()[0]
                except:
                    pass
                else:
                    if id not in self.submitted_ids:
                        continue
                    ongoing.append(id)
                    if statusflag == 'p':
                        idle += 1
                    if statusflag == 'r':
                        run += 1
                    if statusflag == 'sh':
                        fail += 1

        self.submitted_ids = ongoing

        return idle, run, self.submitted - idle - run - fail, fail

    @multiple_try()
    def remove(self, *args):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = subprocess.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))

from_name = {'condor':CondorCluster, 'pbs': PBSCluster, 'sge': SGECluster, 
             'lsf': LSFCluster, 'ge':GECluster}


