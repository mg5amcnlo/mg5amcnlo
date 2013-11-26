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
import inspect

logger = logging.getLogger('madgraph.cluster') 

try:
    from madgraph import MadGraph5Error
    import madgraph.various.misc as misc
except Exception, error:
    if __debug__:
        print  str(error)
    from internal import MadGraph5Error
    import internal.misc as misc
    
class ClusterManagmentError(MadGraph5Error):
    pass

class NotImplemented(MadGraph5Error):
    pass


multiple_try = misc.multiple_try
pjoin = os.path.join


def check_interupt(error=KeyboardInterrupt):

    def deco_interupt(f):
        def deco_f_interupt(self, *args, **opt):
            try:
                return f(self, *args, **opt)
            except error:
                try:
                    self.remove(*args, **opt)
                except Exception:
                    pass
                raise error
        return deco_f_interupt
    return deco_interupt

def store_input(arg=''):

    def deco_store(f):
        def deco_f_store(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
                input_files=[], output_files=[], required_output=[], nb_submit=0):
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            args = dict([(i, values[i]) for i in args if i != 'self'])
            id = f(self, **args)
            if self.nb_retry > 0:
                self.retry_args[id] = args
            return id
        return deco_f_store
    return deco_store


class Cluster(object):
    """Basic Class for all cluster type submission"""
    name = 'mother class'

    def __init__(self, cluster_queue=None, temp_dir=None):
        """Init the cluster"""
        self.submitted = 0
        self.submitted_ids = []
        self.finish = 0
        self.cluster_queue = cluster_queue
        self.temp_dir = temp_dir
        # attribute to relaunch jobs if they failed to produce expected data
        self.nb_retry = 1
        self.retry_args = {}
        self.cluster_retry_wait = 300
    
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
               log=None, required_output=[], nb_submit=0):
        """How to make one submission. Return status id on the cluster."""
        raise NotImplemented, 'No implementation of how to submit a job to cluster \'%s\'' % self.name

    @store_input()
    def submit2(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
                log=None, input_files=[], output_files=[], required_output=[],nb_submit=0):
        """How to make one submission. Return status id on the cluster.
        NO SHARE DISK"""
        
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
            return self.submit(prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)

        if cwd is None:
            cwd = os.getcwd()
        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)

        temp_file_name = "sub." + os.path.basename(prog)
        text = """#!/bin/bash
        MYTMP=%(tmpdir)s/run$%(job_id)s
        MYPWD=%(cwd)s
        mkdir -p $MYTMP
        cd $MYPWD
        input_files=( %(input_files)s )
        for i in ${input_files[@]}
        do
            cp -rL $i $MYTMP
        done
        cd $MYTMP
        bash ./%(script)s %(arguments)s
        output_files=( %(output_files)s )
        for i in ${output_files[@]}
        do
            cp -r $MYTMP/$i $MYPWD
        done
        rm -rf $MYTMP
        """
        
        dico = {'tmpdir' : self.temp_dir, 'script': os.path.basename(prog),
                'cwd': cwd, 'job_id': self.job_id,
                'input_files': ' '.join(input_files + [prog]),
                'output_files': ' '.join(output_files),
                'arguments': ' '.join(argument)}
        
        # writing a new script for the submission
        new_prog = os.path.join(os.path.dirname(prog), temp_file_name)
        open(new_prog, 'w').write(text % dico)
        misc.Popen(['chmod','+x',new_prog],cwd=cwd)
        
        return self.submit(new_prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)
        

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
                #time.sleep(20) #security to ensure that the file are really written on the disk
                logger.info('All jobs finished')
                break
            fct(idle, run, finish)
            time.sleep(30)
        self.submitted = 0
        self.submitted_ids = []
        
    def check_termination(self, job_id):
        """Check the termination of the jobs with job_id and relaunch it if needed."""
        

        if job_id not in self.retry_args:
            return True

        args = self.retry_args[job_id]
        if 'time_check' in args:
            time_check = args['time_check']
        else:
            time_check = 0

        for path in args['required_output']:
            if args['cwd']:
                path = pjoin(args['cwd'], path)
            if not os.path.exists(path):
                break
        else:
            # all requested output are present
            if time_check > 0:
                logger.info('Job %s Finally found the missing output.' % (job_id))
            del self.retry_args[job_id]
            self.submitted_ids.remove(job_id)
            return 'done'
        
        if time_check == 0:
            logger.debug('''Job %s: missing output:%s''' % (job_id,path))
            args['time_check'] = time.time()
            return 'wait'
        elif self.cluster_retry_wait > time.time() - time_check:    
            return 'wait'

        #jobs failed to be completed even after waiting time!!
        if self.nb_retry < 0:
            logger.critical('''Fail to run correctly job %s.
            with option: %s
            file missing: %s''' % (job_id, args, path))
            raw_input('press enter to continue.')
        elif self.nb_retry == 0:
            logger.critical('''Fail to run correctly job %s.
            with option: %s
            file missing: %s.
            Stopping all runs.''' % (job_id, args, path))
            #self.remove()
        elif args['nb_submit'] >= self.nb_retry:
            logger.critical('''Fail to run correctly job %s.
            with option: %s
            file missing: %s
            Fails %s times
            No resubmition. ''' % (job_id, args, path, args['nb_submit']))
            #self.remove()
        else:
            args['nb_submit'] += 1            
            logger.warning('resubmit job (for the %s times)' % args['nb_submit'])
            del self.retry_args[job_id]
            self.submitted_ids.remove(job_id)
            if 'time_check' in args: 
                del args['time_check']
            self.submit2(**args)
            return 'resubmit'
        return 'done'
            
            
            
    @check_interupt()
    def launch_and_wait(self, prog, argument=[], cwd=None, stdout=None, 
                        stderr=None, log=None, required_output=[], nb_submit=0):
        """launch one job on the cluster and wait for it"""
        
        special_output = False # tag for concatenate the error with the output.
        if stderr == -2 and stdout: 
            #We are suppose to send the output to stdout
            special_output = True
            stderr = stdout + '.err'
        id = store_input(self.submit(prog, argument, cwd, stdout, stderr, log,
                          required_output=required_output))
        
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        args = dict([(i, values[i]) for i in args if i != 'self'])        
        self.retry_args[id] = args
        
        nb_wait=0
        while 1: 
            nb_wait+=1
            status = self.control_one_job(id)
            if not status in ['R','I']:
                status = self.check_termination(id)
                if status in ['wait']:
                    time.sleep(30)
                    continue
                elif status in ['resubmit']:
                    id = self.submitted_ids[0]
                    time.sleep(30)
                    continue
                #really stop!
                time.sleep(30) #security to ensure that the file are really written on the disk
                break
            time.sleep(30)
        
        if required_output:
            status = self.check_termination(id)
            if status == 'wait':
                run += 1
            elif status == 'resubmit':
                idle += 1
        
        
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
                        
    def remove(self, *args, **opts):
        """ """
        logger.warning("""This cluster didn't support job removal, 
    the jobs are still running on the cluster.""")


class CondorCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'condor'
    job_id = 'CONDOR_ID'

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a Condor cluster"""
        
        text = """Executable = %(prog)s
                  output = %(stdout)s
                  error = %(stderr)s
                  log = %(log)s
                  %(argument)s
                  environment = CONDOR_ID=$(Cluster).$(Process)
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
        a = misc.Popen(['condor_submit','submit_condor'], stdout=subprocess.PIPE)
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

    @store_input()
    @multiple_try()
    def submit2(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
                log=None, input_files=[], output_files=[], required_output=[], 
                nb_submit=0):
        """Submit the job on the cluster NO SHARE DISK
           input/output file should be give relative to cwd
        """
        
        text = """Executable = %(prog)s
                  output = %(stdout)s
                  error = %(stderr)s
                  log = %(log)s
                  %(argument)s
                  should_transfer_files = YES
                  when_to_transfer_output = ON_EXIT
                  transfer_input_files = %(input_files)s
                  %(output_files)s
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
        # input/output file treatment
        if input_files:
            input_files = ','.join(input_files)
        else: 
            input_files = ''
        if output_files:
            output_files = 'transfer_output_files = %s' % ','.join(output_files)
        else:
            output_files = ''
        
        

        dico = {'prog': prog, 'cwd': cwd, 'stdout': stdout, 
                'stderr': stderr,'log': log,'argument': argument,
                'requirement': requirement, 'input_files':input_files, 
                'output_files':output_files}

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




    
    @multiple_try(nb_try=10, sleep=10)
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'condor_q '+str(id)+" -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)
        
        error = status.stderr.read()
        if status.returncode or error:
            raise ClusterManagmentError, 'condor_q returns error: %s' % error

        return status.stdout.readline().strip()
    
    @check_interupt()
    @multiple_try(nb_try=10, sleep=10)
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        
        if not self.submitted_ids:
            return 0, 0, 0, 0
        
        packet = 15000
        idle, run, fail = 0, 0, 0
        ongoing = []
        for i in range(1+(len(self.submitted_ids)-1)//packet):
            start = i * packet
            stop = (i+1) * packet
            cmd = "condor_q " + ' '.join(self.submitted_ids[start:stop]) + \
            " -format \'%-2s\  ' \'ClusterId\' " + \
            " -format \'%-2s \\n\' \'ifThenElse(JobStatus==0,\"U\",ifThenElse(JobStatus==1,\"I\",ifThenElse(JobStatus==2,\"R\",ifThenElse(JobStatus==3,\"X\",ifThenElse(JobStatus==4,\"C\",ifThenElse(JobStatus==5,\"H\",ifThenElse(JobStatus==6,\"E\",string(JobStatus))))))))\'"
            
            status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE, 
                                                             stderr=subprocess.PIPE)
            error = status.stderr.read()
            if status.returncode or error:
                raise ClusterManagmentError, 'condor_q returns error: %s' % error
                
            for line in status.stdout:
                id, status = line.strip().split()
                ongoing.append(int(id))
                if status in ['I','U']:
                    idle += 1
                elif status == 'R':
                    run += 1
                elif status != 'C':
                    fail += 1

        for id in list(self.submitted_ids):
            if int(id) not in ongoing:
                status = self.check_termination(id)
                if status == 'wait':
                    run += 1
                elif status == 'resubmit':
                    idle += 1

        return idle, run, self.submitted - (idle+run+fail), fail
    
    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobson the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "condor_rm %s" % ' '.join(self.submitted_ids)
        
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))
        
class PBSCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'pbs'
    job_id = 'PBS_JOBID'
    idle_tag = ['Q']
    running_tag = ['T','E','R']
    complete_tag = ['C']

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a PBS cluster"""
        
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
        
        if not os.path.isabs(prog):
            text += "./%s" % prog
        else:
            text+= prog
        
        if argument:
            text += ' ' + ' '.join(argument)

        command = ['qsub','-o', stdout,
                   '-N', me_dir, 
                   '-e', stderr,
                   '-V']

        if self.cluster_queue and self.cluster_queue != 'None':
            command.extend(['-q', self.cluster_queue])

        a = misc.Popen(command, stdout=subprocess.PIPE, 
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE,
                                  stderr=open(os.devnull,'w'))
        
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
        status = misc.Popen([cmd], stdout=subprocess.PIPE)

        if me_dir.endswith('/'):
           me_dir = me_dir[:-1]    
        me_dir = hashlib.md5(me_dir).hexdigest()[-14:]
        if not me_dir[0].isalpha():
                  me_dir = 'a' + me_dir[1:]
        
        ongoing = []
        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            if me_dir in line:
                ongoing.append(line.split()[0].split('.')[0])
                status = line.split()[4]
                if status in self.idle_tag:
                    idle += 1
                elif status in self.running_tag:
                    run += 1
                elif status in self.complete_tag:
                    if not self.check_termination(line.split()[0].split('.')[0]):
                        idle += 1
                else:
                    fail += 1
            
        for id in list(self.submitted_ids):
            if id not in ongoing:
                status = self.check_termination(id)
                if status == 'wait':
                    run += 1
                elif status == 'resubmit':
                    idle += 1

        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


class SGECluster(Cluster):
    """Basic class for dealing with cluster submission"""
    # Class written by Arian Abrahantes.

    name = 'sge'
    job_id = 'JOB_ID'
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
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to an SGE cluster"""

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
        else:
            stderr = self.def_get_path(stderr)
            
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

        a = misc.Popen(command, stdout=subprocess.PIPE,
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)

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
    def remove(self, *args, **opts):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


class LSFCluster(Cluster):
    """Basic class for dealing with cluster submission"""
    
    name = 'lsf'
    job_id = 'LSB_JOBID'

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit the job prog to an LSF cluster"""
        
        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = hashlib.md5(me_dir).hexdigest()[-14:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]
        
        text = ""
        command = ['bsub', '-J', me_dir]
        if cwd is None:
            cwd = os.getcwd()
        else: 
            text = " cd %s;" % cwd
        if stdout and isinstance(stdout, str):
            command.extend(['-o', stdout])
        if stderr and isinstance(stdout, str):
            command.extend(['-e', stderr])
        elif stderr == -2: # -2 is subprocess.STDOUT
            pass
        if log is None:
            log = '/dev/null'
        
        text += prog
        if argument:
            text += ' ' + ' '.join(argument)
        
        if self.cluster_queue and self.cluster_queue != 'None':
            command.extend(['-q', self.cluster_queue])

        a = misc.Popen(command, stdout=subprocess.PIPE, 
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)

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
                status = self.check_termination(id)
                if status == 'wait':
                    run += 1
                elif status == 'resubmit':
                    idle += 1
            else:
                fail += 1

        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args,**opts):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "bkill %s" % ' '.join(self.submitted_ids)
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))

class GECluster(Cluster):
    """Class for dealing with cluster submission on a GE cluster"""
    
    name = 'ge'
    job_id = 'JOB_ID'
    idle_tag = ['qw']
    running_tag = ['r']

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a GE cluster"""
        
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

        a = misc.Popen(['qsub','-o', stdout,
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
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)
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
            status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE)
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
        for id in list(self.submitted_ids):
            if id not in ongoing:
                self.check_termination(id)
        #self.submitted_ids = ongoing

        return idle, run, self.submitted - idle - run - fail, fail

    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "qdel %s" % ' '.join(self.submitted_ids)
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))

class SLURMCluster(Cluster):
    """Basic class for dealing with cluster submission"""

    name = 'slurm'
    job_id = 'SLURM_JOBID'
    idle_tag = ['Q','PD','S','CF']
    running_tag = ['R']
    complete_tag = ['C']

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a SLURM cluster"""
        
        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = hashlib.md5(me_dir).hexdigest()[-8:]

        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]
        
        if cwd is None:
            cwd = os.getcwd()
        if stdout is None:
            stdout = '/dev/null'
        if stderr is None:
            stderr = '/dev/null'
        elif stderr == -2: # -2 is subprocess.STDOUT
            stderr = stdout
        if log is None:
            log = '/dev/null'
        
        command = ['sbatch','-o', stdout,
                   '-J', me_dir, 
                   '-e', stderr, prog] + argument
                   
        a = misc.Popen(command, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT,
                                      stdin=subprocess.PIPE, cwd=cwd)

        output = a.communicate()
        output_arr = output[0].split(' ')
        id = output_arr[3].rstrip()

        if not id.isdigit():
            raise ClusterManagmentError, 'fail to submit to the cluster: \n%s' \

        self.submitted += 1
        self.submitted_ids.append(id)
        return id

    @multiple_try()
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """
        cmd = 'squeue j'+str(id)
        status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE,
                                  stderr=open(os.devnull,'w'))
        
        for line in status.stdout:
            line = line.strip()
            if 'Invalid' in line:
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
        cmd = "squeue"
        status = misc.Popen([cmd], stdout=subprocess.PIPE)

        if me_dir.endswith('/'):
           me_dir = me_dir[:-1]    
        me_dir = hashlib.md5(me_dir).hexdigest()[-8:]
        if not me_dir[0].isalpha():
                  me_dir = 'a' + me_dir[1:]

        idle, run, fail = 0, 0, 0
        ongoing=[]
        for line in status.stdout:
            if me_dir in line:
                id, _, _,_ , status,_ = line.split(None,5)
                ongoing.append(id)
                if status in self.idle_tag:
                    idle += 1
                elif status in self.running_tag:
                    run += 1
                elif status in self.complete_tag:
                    status = self.check_termination(id)
                    if status == 'wait':
                        run += 1
                    elif status == 'resubmit':
                        idle += 1                    
                else:
                    fail += 1
        
        #control other finished job
        for id in list(self.submitted_ids):
            if id not in ongoing:
                status = self.check_termination(id)
                if status == 'wait':
                    run += 1
                elif status == 'resubmit':
                    idle += 1
                    
        
        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobs on the cluster"""
        
        if not self.submitted_ids:
            return
        cmd = "scancel %s" % ' '.join(self.submitted_ids)
        status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


from_name = {'condor':CondorCluster, 'pbs': PBSCluster, 'sge': SGECluster, 
             'lsf': LSFCluster, 'ge':GECluster, 'slurm': SLURMCluster}


