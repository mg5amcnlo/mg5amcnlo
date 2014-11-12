################################################################################
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors             
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which           
# automatically generates Feynman diagrams and matrix elements for arbitrary    
# high-energy processes in the Standard Model and beyond.                       
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this             
# distribution.                                                                 
#                                                                               
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch            
#                                                                               
################################################################################
import subprocess
import logging
import os
import time
import re
import glob
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

pjoin = os.path.join
   
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

    def __init__(self,*args, **opts):
        """Init the cluster"""

        self.submitted = 0
        self.submitted_ids = []
        self.finish = 0
        if 'cluster_queue' in opts:
            self.cluster_queue = opts['cluster_queue']
        else:
            self.cluster_queue = 'madgraph'
        if 'cluster_temp_path' in opts:
            self.temp_dir = opts['cluster_temp_path']
        else:
            self.temp_dir = None
        self.options = {'cluster_status_update': (600, 30)}
        for key,value in opts.items():
            self.options[key] = value
        self.nb_retry = opts['cluster_nb_retry'] if 'cluster_nb_retry' in opts else 0
        self.cluster_retry_wait = opts['cluster_retry_wait'] if 'cluster_retry_wait' in opts else 300
        self.options = dict(opts)
        self.retry_args = {}
        # controlling jobs in controlled type submision
        self.packet = {}
        self.id_to_packet = {}

    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
               log=None, required_output=[], nb_submit=0):
        """How to make one submission. Return status id on the cluster."""
        raise NotImplemented, 'No implementation of how to submit a job to cluster \'%s\'' % self.name


    @store_input()
    def submit2(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
                log=None, input_files=[], output_files=[], required_output=[],
                nb_submit=0):
        """How to make one submission. Return status id on the cluster.
        NO SHARE DISK"""

        if cwd is None:
            cwd = os.getcwd()
        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)
            
        if not required_output and output_files:
            required_output = output_files
        
        if not hasattr(self, 'temp_dir') or not self.temp_dir or \
            (input_files == [] == output_files):
            return self.submit(prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)
            
        if not input_files and not output_files:
            # not input/output so not using submit2
            return self.submit(prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)

        if cwd is None:
            cwd = os.getcwd()
        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)
        temp_file_name = "sub." + os.path.basename(prog) + '.'.join(argument)

        text = """#!/bin/bash
        MYTMP=%(tmpdir)s/run$%(job_id)s
        MYPWD=%(cwd)s
        mkdir -p $MYTMP
        cd $MYPWD
        input_files=( %(input_files)s )
        for i in ${input_files[@]}
        do
            cp -R -L $i $MYTMP
        done
        cd $MYTMP
        echo '%(arguments)s' > arguments
        chmod +x ./%(script)s
        %(program)s ./%(script)s %(arguments)s
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
                'arguments': ' '.join([str(a) for a in argument]),
                'program': ' ' if '.py' in prog else 'bash'}
        
        # writing a new script for the submission
        new_prog = pjoin(cwd, temp_file_name)
        open(new_prog, 'w').write(text % dico)
        misc.Popen(['chmod','+x',new_prog],cwd=cwd)
        
        return self.submit(new_prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)


    def cluster_submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
                log=None, input_files=[], output_files=[], required_output=[],
                nb_submit=0, packet_member=None):
            """This function wrap the cluster submition with cluster independant
               method should not be overwritten (but for DAG type submission)"""
               
            id = self.submit2(prog, argument, cwd, stdout, stderr, log, input_files, 
                             output_files, required_output, nb_submit)               
               
            
            if not packet_member:
                return id
            else:
                if isinstance(packet_member, Packet):
                    self.id_to_packet[id] = packet_member
                    packet_member.put(id)
                    if packet_member.tag not in self.packet:
                        self.packet[packet_member.tag] = packet_member
                else:
                    if packet_member in self.packet:
                        packet = self.packet[packet_member]
                        packet.put(id)
                        self.id_to_packet[id] = packet
                return id
                
    def control(self, me_dir=None):
        """Check the status of job associated to directory me_dir. return (idle, run, finish, fail)"""
        if not self.submitted_ids:
            raise NotImplemented, 'No implementation of how to control the job status to cluster \'%s\'' % self.name
        idle, run, fail = 0, 0, 0
        for pid in self.submitted_ids[:]:
            status = self.control_one_job(id)
            if status == 'I':
                idle += 1
            elif status == 'R':
                run += 1
            elif status == 'F':
                self.finish +=1
                self.submitted_ids.remove(pid)
            else:
                fail += 1

        return idle, run, self.finish, fail

    def control_one_job(self, pid):
        """ control the status of a single job with it's cluster id """
        raise NotImplemented, 'No implementation of how to control the job status to cluster \'%s\'' % self.name

    @check_interupt()
    def wait(self, me_dir, fct, minimal_job=0, update_first=None):
        """Wait that all job are finish.
        if minimal_job set, then return if idle + run is lower than that number"""
        
        
        mode = 1 # 0 is long waiting/ 1 is short waiting
        nb_iter = 0
        nb_short = 0 
        change_at = 5 # number of iteration from which we wait longer between update.

        if update_first:
            idle, run, finish, fail = self.control(me_dir)
            update_first(idle, run, finish)

        #usefull shortcut for readibility
        longtime, shorttime = self.options['cluster_status_update']
        

        while 1: 
            old_mode = mode
            nb_iter += 1
            idle, run, finish, fail = self.control(me_dir)
            if fail:
                raise ClusterManagmentError('Some Jobs are in a Hold/... state. Please try to investigate or contact the IT team')
            if idle + run == 0:
                #time.sleep(20) #security to ensure that the file are really written on the disk
                logger.info('All jobs finished')
                fct(idle, run, finish)
                break
            if idle + run < minimal_job:
                return
            fct(idle, run, finish)
            #Determine how much we have to wait (mode=0->long time, mode=1->short time)
            if nb_iter < change_at:
                mode = 1
            elif idle < run:
                if old_mode == 0:
                    if nb_short:
                        mode = 0 #we already be back from short to long so stay in long
                    #check if we need to go back to short mode
                    elif idle:
                        if nb_iter > change_at + int(longtime)//shorttime:
                            mode = 0 #stay in long waiting mode
                        else:
                            mode = 1 # pass in short waiting mode
                            nb_short =0
                    else:
                        mode = 1 # pass in short waiting mode
                        nb_short = 0
                elif old_mode == 1:
                    nb_short +=1
                    if nb_short > 3* max(change_at, int(longtime)//shorttime):
                        mode = 0 #go back in slow waiting
            else:
                mode = 0
            
            #if pass from fast(mode=1) to slow(mode=0) make a print statement:
            if old_mode > mode:
                logger.info('''Start to wait %ss between checking status.
Note that you can change this time in the configuration file.
Press ctrl-C to force the update.''' % self.options['cluster_status_update'][0])   
            
            #now Waiting!        
            if mode == 0:
                try:
                    time.sleep(self.options['cluster_status_update'][0])
                except KeyboardInterrupt:
                    logger.info('start to update the status')
                    nb_iter = min(0, change_at -2)
                    nb_short = 0                
            else:
                time.sleep(self.options['cluster_status_update'][1])
                    
                    
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
# check that file exists and is not empty.
            if not (os.path.exists(path) and os.stat(path).st_size != 0) :
                break
        else:
            # all requested output are present
            if time_check > 0:
                logger.info('Job %s Finally found the missing output.' % (job_id))
            del self.retry_args[job_id]
            self.submitted_ids.remove(job_id)
            # check if the job_id is in a packet
            if job_id in self.id_to_packet:
                nb_in_packet = self.id_to_packet[job_id].remove_one()
                if nb_in_packet == 0:
                    # packet done run the associate function
                    packet = self.id_to_packet[job_id]
                    # fully ensure that the packet is finished (thread safe)
                    packet.queue.join()
                    #running the function
                    packet.fct(*packet.args)                    
                del self.id_to_packet[job_id]
                return 'resubmit'
            
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
                        stderr=None, log=None, required_output=[], nb_submit=0,
                        input_files=[], output_files=[]):
        """launch one job on the cluster and wait for it"""
        
        special_output = False # tag for concatenate the error with the output.
        if stderr == -2 and stdout: 
            #We are suppose to send the output to stdout
            special_output = True
            stderr = stdout + '.err'

        id = self.submit2(prog, argument, cwd, stdout, stderr, log,
                          required_output=required_output, input_files=input_files,
                          output_files=output_files)
        
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
            time.sleep(self.options['cluster_status_update'][1])
        
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


class Packet(object):
    """ an object for handling packet of job, it is designed to be thread safe
    """

    def __init__(self, name, fct, args, opts={}):
        import Queue
        import threading
        self.queue = Queue.Queue()
        self.tag = name
        self.fct = fct
        self.args = args
        self.opts = opts
        self.done = threading.Event()

    def put(self, *args, **opts):
        self.queue.put(*args, **opts)

    append = put

    def remove_one(self):
        self.queue.get(True)
        self.queue.task_done()
        return self.queue.qsize()
        
class MultiCore(Cluster):
    """class for dealing with the submission in multiple node"""

    job_id = "$"

    def __init__(self, *args, **opt):
        """Init the cluster """
        
        
        super(MultiCore, self).__init__(self, *args, **opt)
        
        import Queue
        import threading
        import thread
        self.queue = Queue.Queue() # list of job to do
        self.done = Queue.Queue()  # list of job finisned
        self.submitted = Queue.Queue() # one entry by job submitted
        self.stoprequest = threading.Event() #flag to ensure everything to close
        self.demons = []
        self.nb_done =0
        if 'nb_core' in opt:
            self.nb_core = opt['nb_core']
        elif isinstance(args[0],int):
            self.nb_core = args[0]
        else:
            self.nb_core = 1
        self.update_fct = None
        
        self.lock = threading.Event() # allow nice lock of the main thread
        self.pids = [] # allow to clean jobs submit via subprocess
        self.fail_msg = None

        # starting the worker node
        for _ in range(self.nb_core):
            self.start_demon()

        
    def start_demon(self):
        import threading
        t = threading.Thread(target=self.worker)
        t.daemon = True
        t.start()
        self.demons.append(t)


    def worker(self):
        import Queue
        import thread
        while not self.stoprequest.isSet():
            try:
                args = self.queue.get()
                tag, exe, arg, opt = args
                try:
                    # check for executable case
                    if isinstance(exe,str):
                        if os.path.exists(exe) and not exe.startswith('/'):
                            exe = './' + exe
                        if opt['stderr'] == None:
                            opt['stderr'] = subprocess.STDOUT
                        proc = misc.Popen([exe] + arg,  **opt)
                        pid = proc.pid
                        self.pids.append(pid)
                        proc.wait()
                        if proc.returncode not in [0, 143, -15]:
                            fail_msg = 'program %s launch ends with non zero status: %s. Stop all computation' % \
                            (' '.join([exe]+arg), proc.returncode)
                            logger.warning(fail_msg)
                            self.stoprequest.set()
                            self.remove(fail_msg)
                    # handle the case when this is a python function. Note that
                    # this use Thread so they are NO built-in parralelization this is 
                    # going to work on a single core! (but this is fine for IO intensive 
                    # function. for CPU intensive fct this will slow down the computation
                    else:
                        pid = tag
                        self.pids.append(pid)
                        # the function should return 0 if everything is fine
                        # the error message otherwise
                        returncode = exe(*arg, **opt)
                        if returncode != 0:
                            logger.warning("fct %s does not return 0", exe)
                            self.stoprequest.set()
                            self.remove("fct %s does not return 0" % exe)
                except Exception,error:
                    logger.warning(str(error))
                    self.stoprequest.set()
                    self.remove(error)
                    if __debug__:
                        raise

                self.queue.task_done()
                self.done.put(tag)
                #release the mother to print the status on the screen
                try:
                    self.lock.set()
                except thread.error:
                    continue
            except Queue.Empty:
                continue
            
            
            
    
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None,
               log=None, required_output=[], nb_submit=0):
        """submit a job on multicore machine"""
        
        tag = (prog, tuple(argument), cwd, nb_submit)
        if isinstance(prog, str):
            
    
            opt = {'cwd': cwd, 
                   'stdout':stdout,
                   'stderr': stderr}
            self.queue.put((tag, prog, argument, opt))                                                                                                                                
            self.submitted.put(1)
            return tag
        else:
            # python function
            self.queue.put((tag, prog, argument, {}))
            self.submitted.put(1)
            return tag            
        
    def launch_and_wait(self, prog, argument=[], cwd=None, stdout=None, 
                                stderr=None, log=None, **opts):
        """launch one job and wait for it"""    
        if isinstance(stdout, str):
            stdout = open(stdout, 'w')
        if isinstance(stderr, str):
            stdout = open(stderr, 'w')        
        return misc.call([prog] + argument, stdout=stdout, stderr=stderr, cwd=cwd) 

    def remove(self, error=None):
        """Ensure that all thread are killed"""
        
        # ensure the worker to stop
        self.stoprequest.set()
        if error:
            self.fail_msg = error        

        for pid in list(self.pids):
            if isinstance(pid, tuple):
                continue
            out = os.system('CPIDS=$(pgrep -P %(pid)s); kill -15 $CPIDS > /dev/null 2>&1' \
                            % {'pid':pid} )
            out = os.system('kill -15 %(pid)s > /dev/null 2>&1' % {'pid':pid} )            
            if out == 0:
                try:
                    self.pids.remove(pid)
                except:
                    pass
            #out = os.system('kill -9 %s &> /dev/null' % pid)

        time.sleep(1) # waiting if some were submitting at the time of ctrl-c
        for pid in list(self.pids):
            if isinstance(pid, tuple):
                continue
            out = os.system('CPIDS=$(pgrep -P %s); kill -15 $CPIDS > /dev/null 2>&1' % pid )
            out = os.system('kill -15 %(pid)s > /dev/null 2>&1' % {'pid':pid} ) 
            if out == 0:
                try:
                    self.pids.remove(pid)
                except:
                    pass
                    

    def wait(self, me_dir, update_status, update_first=None):
        """Waiting that all the jobs are done. This function also control that
        the submission by packet are handle correctly (i.e. submit the function)"""
        import Queue
        import threading

        last_status = (0, 0, 0)
        sleep_time = 1
        use_lock = True
        first = True
        while True:
            force_one_more_loop = False # some security
                        
            # Loop over the job tagged as done to check if some packet of jobs
            # are finished in case, put the associate function in the queue
            while self.done.qsize():
                try:
                    tag = self.done.get(True, 1)
                    if tuple(tag) in self.id_to_packet:
                        packet = self.id_to_packet[tuple(tag)]
                        remaining = packet.remove_one()
                        if remaining == 0:
                            # fully ensure that the packet is finished (thread safe)
                            packet.queue.join()
                            self.submit(packet.fct, packet.args)
                            force_one_more_loop = True
                    self.nb_done += 1
                    self.done.task_done()
                except Queue.Empty:
                    pass            

            # Get from the various queue the Idle/Done/Running information 
            # Those variable should be thread safe but approximate.
            Idle = self.queue.qsize()
            Done = self.nb_done + self.done.qsize()
            Running = max(0, self.submitted.qsize() - Idle - Done)            
            
            
            
            if Idle + Running <= 0 and not force_one_more_loop:
                update_status(Idle, Running, Done)
                # Going the quit since everything is done
                # Fully Ensure that everything is indeed done.
                self.queue.join()
                break
            
            if (Idle, Running, Done) != last_status:
                if first:
                    update_first(Idle, Running, Done)
                    first = False
                else:
                    update_status(Idle, Running, Done)
                last_status = (Idle, Running, Done)
                    
            # Define how to wait for the next iteration
            if use_lock:
                # simply wait that a worker release the lock
                use_lock = self.lock.wait(300)
                self.lock.clear()
                if not use_lock and Idle > 0:
                    use_lock = True
            else:
                # to be sure that we will never fully lock at the end pass to 
                # a simple time.sleep()
                time.sleep(sleep_time)
                sleep_time = min(sleep_time + 2, 180)
        update_first(Idle, Running, Done)
        
        if self.stoprequest.isSet():
            if isinstance(self.fail_msg, Exception):
                raise self.fail_msg
            else:
                raise Exception, self.fail_msg
        # reset variable for next submission
        try:
            self.lock.clear()
        except Exception:
            pass
        self.done = Queue.Queue()
        self.nb_done = 0
        self.submitted = Queue.Queue()
        self.pids = []
        self.stoprequest.clear()

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
        
        if not required_output and output_files:
            required_output = output_files
        
        if (input_files == [] == output_files):
            return self.submit(prog, argument, cwd, stdout, stderr, log, 
                               required_output=required_output, nb_submit=nb_submit)
        
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
            argument = 'Arguments = %s' % ' '.join([str(a) for a in argument])
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
    
    maximum_submited_jobs = 2500

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a PBS cluster"""
        
        
        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = misc.digest(me_dir)[-14:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]

        if len(self.submitted_ids) > self.maximum_submited_jobs:
            fct = lambda idle, run, finish: logger.info('Waiting for free slot: %s %s %s' % (idle, run, finish))
            me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
            self.wait(me_dir, fct, self.maximum_submited_jobs)

        
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
        if not id.isdigit() or a.returncode !=0:
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
                                  stderr=subprocess.STDOUT)

        for line in status.stdout:
            line = line.strip()
            if 'cannot connect to server' in line or 'cannot read reply' in line:
                raise ClusterManagmentError, 'server disconnected'
            if 'Unknown' in line:
                return 'F'
            elif line.startswith(str(id)):
                jobstatus = line.split()[4]
            else:
                jobstatus=""
                        
        if status.returncode != 0 and status.returncode is not None:
            raise ClusterManagmentError, 'server fails in someway (errorcode %s)' % status.returncode
        if jobstatus in self.idle_tag:
            return 'I' 
        elif jobstatus in self.running_tag:                
            return 'R' 
        return 'F'
        
    
    @multiple_try()    
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        cmd = "qstat"
        status = misc.Popen([cmd], stdout=subprocess.PIPE)

        if me_dir.endswith('/'):
            me_dir = me_dir[:-1]    
        me_dir = misc.digest(me_dir)[-14:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]
        ongoing = []

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            if 'cannot connect to server' in line or 'cannot read reply' in line:
                raise ClusterManagmentError, 'server disconnected'
            if me_dir in line:
                ongoing.append(line.split()[0].split('.')[0])
                status2 = line.split()[4]
                if status2 in self.idle_tag:
                    idle += 1
                elif status2 in self.running_tag:
                    run += 1
                elif status2 in self.complete_tag:
                    if not self.check_termination(line.split()[0].split('.')[0]):
                        idle += 1
                else:
                    fail += 1

        if status.returncode != 0 and status.returncode is not None:
            raise ClusterManagmentError, 'server fails in someway (errorcode %s)' % status.returncode

        for id in list(self.submitted_ids):
            if id not in ongoing:
                status2 = self.check_termination(id)
                if status2 == 'wait':
                    run += 1
                elif status2 == 'resubmit':
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
        me_dir = misc.digest(me_dir)[-10:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]

        if cwd is None:
            #cwd = os.getcwd()
            cwd = self.def_get_path(os.getcwd())
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
        me_dir = misc.digest(me_dir)[-10:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]

        finished = list(self.submitted_ids)

        idle, run, fail = 0, 0, 0
        for line in status.stdout:
            if me_dir in line:
                id,_,_,_,status = line.split()[:5]
                if status in self.idle_tag:
                    idle += 1
                    finished.remove(id)
                elif status in self.running_tag:
                    run += 1
                    finished.remove(id)
                else:
                    logger.debug(line)
                    fail += 1
                    finished.remove(id)

        for id in finished:
            self.check_termination(id)

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
        me_dir = misc.digest(me_dir)[-14:]
        if not me_dir[0].isalpha():
            me_dir = 'a' + me_dir[1:]
        
        text = ""
        command = ['bsub', '-C0', '-J', me_dir]
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

        jobstatus = {}
        for line in status.stdout:
            line = line.strip()
            if 'JOBID' in line:
                continue
            splitline = line.split()
            id = splitline[0]
            if id not in self.submitted_ids:
                continue
            jobstatus[id] = splitline[2]

        idle, run, fail = 0, 0, 0
        for id in self.submitted_ids[:]:
            if id in jobstatus:
                status = jobstatus[id]
            else:
                status = 'MISSING'
            if status == 'RUN':
                run += 1
            elif status == 'PEND':
                idle += 1
            else:
                status = self.check_termination(id)
                if status == 'wait':
                    run += 1
                elif status == 'resubmit':
                    idle += 1                

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
                except Exception:
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

def asyncrone_launch(exe, cwd=None, stdout=None, argument = [], **opt):
    """start a computation and not wait for it to finish.
       this fonction returns a lock which is locked as long as the job is 
       running."""

    mc = MultiCore(1)
    mc.submit(exe, argument, cwd, stdout, **opt)
    mc.need_waiting = True
    return mc.lock


class SLURMCluster(Cluster):
    """Basic class for dealing with cluster submission"""

    name = 'slurm'
    job_id = 'SLURM_JOBID'
    idle_tag = ['Q','PD','S','CF']
    running_tag = ['R', 'CG']
    complete_tag = ['C']

    @multiple_try()
    def submit(self, prog, argument=[], cwd=None, stdout=None, stderr=None, log=None,
               required_output=[], nb_submit=0):
        """Submit a job prog to a SLURM cluster"""
        
        me_dir = os.path.realpath(os.path.join(cwd,prog)).rsplit('/SubProcesses',1)[0]
        me_dir = misc.digest(me_dir)[-8:]

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
        
        command = ['sbatch', '-o', stdout,
                   '-J', me_dir, 
                   '-e', stderr, prog] + argument

        if self.cluster_queue and self.cluster_queue != 'None':
                command.insert(1, '-p')
                command.insert(2, self.cluster_queue)

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
        me_dir = misc.digest(me_dir)[-8:]
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

class HTCaaSCluster(Cluster):
    """Class for dealing with cluster submission on a HTCaaS cluster using GPFS """

    name= 'htcaas'
    job_id = 'HTCAAS_JOBID'

    @store_input()
    @multiple_try()
    def submit2(self, prog, argument=[], cwd=None, stdout=None, stderr=None,
                log=None, input_files=[], output_files=[], required_output=[],
                nb_submit=0):
        """Submit the HTCaaS job on the cluster with NO SHARE DISK
           input/output file should be give relative to cwd
        """
        # To make workspace name(temp)
        if 'ajob' in prog:
            prog_num = prog.rsplit("ajob",1)[1]
        else:
            prog_num = '0'

        cur_usr = os.getenv('USER')

        if cwd is None:
            cwd = os.getcwd()

        cwd_cp = cwd.rsplit("/",2)
        #print 'This is HTCaaS Mode'

        if not stdout is None:
            print "stdout: %s" % stdout

        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)

        if not required_output and output_files:
            required_output = output_files


        if not 'combine' and not 'pythia' in prog :
         cwd_arg = cwd+"/arguments"
         temp = ' '.join([str(a) for a in argument])
         arg_cmd="echo '"+temp+"' > " + cwd_arg
         #print arg_cmd
         #aa = misc.Popen([arg_cmd], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
         command = ['htcaas-mgjob-submit','-d',cwd,'-e',os.path.basename(prog)]
         if argument :
            command.extend(['-a ', '='.join([str(a) for a in argument])])
         print command
         a = misc.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=cwd)
         id = a.stdout.read().strip()

        else:
         cwd_arg = cwd+"/arguments"
         temp = ' '.join([str(a) for a in argument])
         #arg_cmd="echo '"+temp+"' > " + cwd_arg
         #print arg_cmd
         #aa = misc.Popen([arg_cmd], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
         #print os.path.basename(prog)
         temp_file_name = "sub." + os.path.basename(prog)
         text = """#!/bin/bash
         MYPWD=%(cwd)s
         cd $MYPWD
         input_files=(%(input_files)s )
         for i in ${input_files[@]}
         do
          chmod -f +x $i
         done
         /bin/bash %(prog)s %(arguments)s > %(stdout)s
         """
         dico = {'cwd':cwd, 'input_files': ' '.join(input_files + [prog]), 'stdout': stdout, 'prog':prog,
                 'arguments': ' '.join([str(a) for a in argument]),
                 'program': ' ' if '.py' in prog else 'bash'}

         # writing a new script for the submission
         new_prog = pjoin(cwd, temp_file_name)
         open(new_prog, 'w').write(text % dico)
         misc.Popen(['chmod','+x',new_prog],cwd=cwd)
         command = ['htcaas-mgjob-submit','-d',cwd,'-e',temp_file_name]
         a = misc.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=cwd)
         id = a.stdout.read().strip()

        nb_try=0
        nb_limit=5
        if not id.isdigit() :
                print "[ID is not digit]:" + id

        while not id.isdigit() :
            nb_try+=1
            print "[fail_retry]:"+ nb_try
            a=misc.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=cwd)
            id = a.stdout.read().strip()
            if nb_try > nb_limit :
              raise ClusterManagementError, 'fail to submit to the HTCaaS cluster: \n %s' % id
              break

        self.submitted += 1
        self.submitted_ids.append(id)

        return id

    @multiple_try(nb_try=10, sleep=10)
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """

        if id == 0 :
         status_out ='C'
        else  :
         cmd = 'htcaas-job-status -m '+str(id)+ " -s | grep Status "
         status = misc.Popen([cmd], shell=True,stdout=subprocess.PIPE,
                                                         stderr=subprocess.PIPE)
         error = status.stderr.read()
         if status.returncode or error:
             raise ClusterManagmentError, 'htcaas-job-submit returns error: %s' % error
         status_out= status.stdout.read().strip()
         status_out= status_out.split(":",1)[1]
         if status_out == 'waiting':
              status_out='I'
         elif status_out == 'preparing' or status_out == 'running':
              status_out = 'R'
         elif status_out != 'done':
              status_out = 'F'
         elif status_out == 'done':
              status_out = 'C'

        return status_out

    @multiple_try(nb_try=15, sleep=1)
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
        #print "HTCaaS2 Control"
        if not self.submitted_ids:
            return 0, 0, 0, 0

        ongoing = []
        idle, run, fail = 0, 0, 0

        if id == 0 :
         return 0 , 0, 0, 0
        else :
         for i in range(len(self.submitted_ids)):
            ongoing.append(int(self.submitted_ids[i]))
            cmd = "htcaas-job-status -m " + self.submitted_ids[i] + " -s | grep Status "
            status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            status_out= status.stdout.read().strip()
            status_out= status_out.split(":",1)[1]
            if status_out == 'waiting':
                idle += 1
            elif status_out == 'preparing':
                run += 1
            elif status_out == 'running':
                run += 1
            elif status_out != 'done':
                fail += 1

            if status_out != 'done':
                print "["+ self.submitted_ids[i] + "] " + status_out
        '''
        for i in range(len(self.submitted_ids)):
            if int(self.submitted_ids[i]) not in ongoing:
               status = self.check_termination(int(self.submitted_ids[i]))
               if status = 'waiting':
                    idle += 1
               elif status == 'resubmit':
                    idle += 1
               elif status == 'failed':
                    fail += 1
        '''

        return idle, run, self.submitted - (idle+run+fail), fail

    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobson the cluster"""

        if not self.submitted_ids:
            return
        for i in range(len(self.submitted_ids)):
         cmd = "htcaas-job-cancel -m %s" % ' '.join(self.submitted_ids[i])
         status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))

 
class HTCaaS2Cluster(Cluster):
    """Class for dealing with cluster submission on a HTCaaS cluster"""    
      
    name= 'htcaas2'
    job_id = 'HTCAAS2_JOBID' 
      
    @store_input()
    @multiple_try() 
    def submit2(self, prog, argument=[], cwd=None, stdout=None, stderr=None, 
                log=None, input_files=[], output_files=[], required_output=[], 
                nb_submit=0):
        """Submit the job on the cluster NO SHARE DISK
           input/output file should be give relative to cwd
        """
        # To make workspace name(temp) 
        if 'ajob' in prog:          
            prog_num = prog.rsplit("ajob",1)[1]
        elif 'run_combine' in prog: 
            prog_num = '0'
        else:
            prog_num = prog

        cur_usr = os.getenv('USER')

        import uuid
        dir = str(uuid.uuid4().hex)
        #dir = str(int(time()))        
        prog_dir = '_run%s'% prog_num
        prog_dir = dir+prog_dir

        if cwd is None:
            cwd = os.getcwd()
            
        cwd_cp = cwd.rsplit("/",2)      

        if stdout is None:
            stdout='/dev/null'

        if not os.path.exists(prog):
            prog = os.path.join(cwd, prog)
            
        if not required_output and output_files:
            required_output = output_files

        if '/' in argument :
            temp_file_name = "sub." + os.path.basename(prog) 
        else :
            temp_file_name = "sub." + os.path.basename(prog) + '.'.join(argument)
                        
        
        if 'combine' in prog or 'pythia' in prog :
         text = """#!/bin/bash
        MYPWD=%(cwd)s
        cd $MYPWD
        script=%(script)s
        input_files=(%(input_files)s )
        if [ $# -ge 1 ]; then
           arg1=$1
        else
           arg1=''
        fi
        args=' %(arguments)s'
        for i in ${input_files[@]}; do
          if [[ "$i" == *$script* ]]; then
                script=$i
          fi
          chmod -f +x $i
        done
        /bin/bash ${script}  ${args} > %(stdout)s
        """

        elif 'shower' in prog :
         text = """#!/bin/bash
        MYPWD=%(cwd)s
        cd $MYPWD
        args=' %(arguments)s'
        input_files=( %(input_files)s )
        for i in ${input_files[@]}
        do
          chmod -f +x $i
        done
        /bin/bash %(script)s   ${args} > $MYPWD/done
         """

        else :
         text = """#!/bin/bash
        MYPWD=%(cwd)s
        #mkdir -p $MYTMP
        cd $MYPWD
        input_files=( %(input_files)s )
        for i in ${input_files[@]}
        do
         if [[ $i != */*/* ]]; then
            i=$PWD"/"$i
         fi
         echo $i
         if [ -d $i ]; then
            htcaas-file-put -l $i -r /pwork01/%(cur_usr)s/MG5_workspace/%(prog_dir)s/ -i %(cur_usr)s
         else
           htcaas-file-put -f $i -r /pwork01/%(cur_usr)s/MG5_workspace/%(prog_dir)s/ -i %(cur_usr)s
         fi
        done
         """

        dico = {'cur_usr' : cur_usr, 'script': os.path.basename(prog),
                'cwd': cwd, 'job_id': self.job_id, 'prog_dir': prog_dir,
                'input_files': ' '.join(input_files + [prog]),
                'output_files': ' '.join(output_files), 'stdout': stdout,
                'arguments': ' '.join([str(a) for a in argument]),
                'program': ' ' if '.py' in prog else 'bash'}

        # writing a new script for the submission
        new_prog = pjoin(cwd, temp_file_name)
        open(new_prog, 'w').write(text % dico)
        misc.Popen(['chmod','+x',new_prog],cwd=cwd)

       # print temp_file_name
        cmd1='/bin/bash '+ cwd+'/'+temp_file_name
        status1 = misc.Popen([cmd1], shell=True, stdout=subprocess.PIPE,
                                                         stderr=subprocess.PIPE)
        #print '%s' % status1.stdout.read()


        if not 'combine' in prog and not 'shower' in prog and not 'pythia' in prog:

         cmd3 = """htcaas-mgjob-submit -d /pwork01/%(cur_usr)s/MG5_workspace/%(prog_dir)s/ -e %(script)s %(arguments)s""" 
         dico3 = {'cur_usr' : cur_usr, 'script': os.path.basename(prog), 
                  'arguments': ' ' if not argument else "-a " + '='.join([str(a) for a in argument]) ,
                  'prog_dir': prog_dir }
         status3 = misc.Popen([cmd3 % dico3], shell=True, stdout=subprocess.PIPE,
                                                         stderr=subprocess.PIPE)
         id = status3.stdout.read().strip()
         ## exception
         nb_try=0
         nb_limit=5
         while not id.isdigit() :
            nb_try+=1
            a=misc.Popen( [cmd3 % dico3], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=cwd)
            id = a.stdout.read().strip()
            if nb_try > nb_limit :
              raise ClusterManagmentError, 'Fail to submit to the HTCaaS cluster: \n %s' % id
              break

         temp_file_name2 = "sub." +id
         text2 = """#!/bin/bash
         MYPWD=%(cwd)s
         output_files=( %(output_files)s )
         result=done
         if [ ! -e ${MYPWD}/done.%(job_id)s ]; then 
                 for i in ${output_files[@]}
                 do
                        htcaas-file-get -l ${MYPWD}/$i -r /pwork01/%(cur_usr)s/MG5_workspace/%(prog_dir)s/$i -i %(cur_usr)s
                    chmod -Rf 777 ${MYPWD}/$i
                 done
                 for i in ${output_files[@]}; do 
                    if [[ -e ${MYPWD}/$i ]]; then
                        result=done        
                    else 
                        result=running
                        echo $result
                        exit 0
                    fi 
                 done
                 echo $result
                 touch ${MYPWD}/done.%(job_id)s
         else
                  for i in ${output_files[@]}; do
                    if [ -e ${MYPWD}/$i ]; then
                        result=done
                    else
                        rm -f ${MYPWD}/done.%(job_id)s
                        result=running
                        echo $result
                        exit 0
                    fi
                 done         
                echo $result

         fi        
         
         """
         dico2 = {'cur_usr' : cur_usr, 'script': os.path.basename(prog),
                'cwd': cwd, 'job_id': self.job_id, 'prog_dir': prog_dir,
                'output_files': ' '.join(output_files), 'job_id': id,
                'program': ' ' if '.py' in prog else 'bash'}

         homePath = os.getenv("HOME")
         outPath = homePath +"/MG5"

         new_prog2 = pjoin(outPath, temp_file_name2)
         open(new_prog2, 'w').write(text2 % dico2)
         misc.Popen(['chmod','+x',new_prog2],cwd=cwd)

         
         self.submitted += 1
         self.submitted_ids.append(id)

        elif 'combine' in prog or 'shower' in prog or 'pythia' in prog:
          if '/dev/null' in stdout :
              stdout=''
        
          temp_file_shower = "sub.out"
          text_shower = """#!/bin/bash
          MYPWD=%(cwd)s
          result=done
          output_files=(%(output_files)s)
          for i in ${output_files[@]}; do
            if [ -e $MYPWD/$i -o  -e $i ]; then
                result=done
            else 
                result=running
                echo $result
                exit 0
            fi
          done
          echo $result
          """
          dico_shower = { 'cwd': cwd, 'output_files': ' '.join([stdout]+output_files),
                'program': ' ' if '.py' in prog else 'bash'}
          homePath = os.getenv("HOME")
          outPath = homePath +"/MG5"
          new_prog_shower = pjoin(outPath, temp_file_shower)
          open(new_prog_shower, 'w').write(text_shower % dico_shower)
          misc.Popen(['chmod','+x',new_prog_shower],cwd=cwd) 
          
          id='-1'
          self.submitted += 1
          self.submitted_ids.append(id) 

        else :  
         id='-2'
         self.submitted += 1
         self.submitted_ids.append(id)

        return id
   
    @multiple_try(nb_try=10, sleep=10)
    def control_one_job(self, id):
        """ control the status of a single job with it's cluster id """   
     
        homePath = os.getenv("HOME")
        outPath = homePath +"/MG5"

 
        if id == '0' or id=='-2' :
           status_out ='done' 
        elif id == '-1' :
           cmd='/bin/bash ' +outPath+'/sub.out'
           status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
           status_out=status.stdout.read().strip()
           print "["+id+"]" + status_out
           if status_out == 'waiting': 
                status_out='wait'
           elif status_out == 'preparing' or status_out == 'running':
                status_out = 'R'
           elif status_out != 'done':
                status_out = 'F'
           elif status_out == 'done':
                status_out = 'C'
                
           print "["+id+"]" + status_out
        else  :
         cmd = 'htcaas-job-status -m '+str(id)+ " -s | grep Status "  
         status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)
         error = status.stderr.read()
         if status.returncode or error:
             raise ClusterManagmentError, 'htcaas-job-submit returns error: %s' % error        
         status_out= status.stdout.read().strip()
         status_out= status_out.split(":",1)[1]
         print "["+id+"]" + status_out
         if status_out == 'waiting':
              status_out='wait'
         elif status_out == 'preparing' or status_out == 'running':
              status_out = 'R'
         elif status_out == 'failed' :
              args = self.retry_args[id]
              id_temp = self.submit2(**args)
              del self.retry_args[id]
              self.submitted_ids.remove(id)
              status_out = 'I'
         elif status_out != 'done':
              status_out = 'F'
         elif status_out == 'done':
              status_out = 'C'

        return status_out

   
    @check_interupt()
    @multiple_try(nb_try=15, sleep=10)
    def control(self, me_dir):
        """ control the status of a single job with it's cluster id """
 
        if not self.submitted_ids:
            return 0, 0, 0, 0 

        ongoing = []
        idle, run, fail = 0, 0, 0      

        homePath = os.getenv("HOME")
        outPath = homePath +"/MG5"

        for i in range(len(self.submitted_ids)):
           ongoing.append(self.submitted_ids[i])
           if self.submitted_ids[i] == '-2' :
              return 0,0,0,0
           if self.submitted_ids[i] == '0' :
                      # ongoing.append('0')
                        status_out='done'            
           elif self.submitted_ids[i] == '-1' :
              cmd='/bin/bash ' +outPath+'/sub.out'
              status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE) 
              status_out=status.stdout.read().strip()
              if status_out == 'waiting':
                idle += 1
              elif status_out == 'preparing':
                run += 1
              elif status_out == 'running':
                run += 1
              elif status_out != 'done':
                fail += 1
           else : 
              args = self.retry_args[str(self.submitted_ids[i])]
              if 'required_output'in args and not args['required_output']:
                 args['required_output'] = args['output_files']
                 self.retry_args[str(self.submitted_ids[i])] = args

              cmd = "htcaas-job-status -m " + self.submitted_ids[i] + " -s | grep Status "
              status = misc.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
              status_out= status.stdout.read().strip()
              status_out= status_out.split(":",1)[1]
              if status_out == 'waiting':
                idle += 1
              elif status_out == 'preparing':
                run += 1
              elif status_out == 'running':
                run += 1
              elif status_out == 'failed' or status_out == 'canceled': 
                id = self.submit2(**args)
                #self.submitted_ids[i]=id
                del self.retry_args[self.submitted_ids[i]]
                self.submitted_ids.remove(self.submitted_ids[i])
                self.submitted-=1
                idle += 1
              elif status_out != 'done':
                fail += 1
              if status_out == 'done': 
                 cmd2='/bin/bash '+ outPath+'/sub.'+self.submitted_ids[i]
                 status2 = misc.Popen([cmd2], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                 aa= status2.stdout.read().strip()
                 #result= self.check_termination(str(self.submitted_ids[i]))
                 #print result
                 #if not result : 
                 #if not self.check_termination(str(self.submitted_ids[i])):
                #        print "not_self" + self.submitted_ids[i]
                #        idle += 1
                 #else :
                 for path in args['required_output']:
                       if args['cwd']:
                           path = pjoin(args['cwd'], path)
                       # check that file exists and is not empty.
                       temp1=os.path.exists(path)
                       temp2=os.stat(path).st_size
                       if not (os.path.exists(path) and os.stat(path).st_size != 0) :
                            status2 = misc.Popen([cmd2], shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                            aa= status2.stdout.read().strip()
                            if aa == 'done':
                                self.submitted_ids[i] = '0' 
                            elif aa == 'running':
                                run += 1
                       else :
                                   self.submitted_ids[i]='0' 

        
        for i in range(len(self.submitted_ids)):
             if str(self.submitted_ids[i]) not in ongoing:
                status2= self.check_termination(str(self.submitted_ids[i]))
                if status2 == 'wait':
                   run += 1
                elif status2 == 'resubmit':
                   idle += 1

        return idle, run, self.submitted - (idle+run+fail), fail
 
    @multiple_try()
    def remove(self, *args, **opts):
        """Clean the jobson the cluster"""
        
        if not self.submitted_ids:
            return
        for i in range(len(self.submitted_ids)):
         cmd = "htcaas-job-cancel -m %s" % ' '.join(self.submitted_ids[i])        
         status = misc.Popen([cmd], shell=True, stdout=open(os.devnull,'w'))


from_name = {'condor':CondorCluster, 'pbs': PBSCluster, 'sge': SGECluster, 
             'lsf': LSFCluster, 'ge':GECluster, 'slurm': SLURMCluster, 
             'htcaas':HTCaaSCluster, 'htcaas2':HTCaaS2Cluster}


