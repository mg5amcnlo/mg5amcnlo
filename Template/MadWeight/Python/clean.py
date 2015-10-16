#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##       email:  olivier.mattelaer@uclouvain.be                         ##
##   author: Artoisenet Pierre (CP3)                                    ##
##       email:  pierre.artoisenet@uclouvain.be                         ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:26/08/08                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##      +Clean_run                                                      ##
##      +Clean_event                                                    ##
##      +Class Clean                                                    ##
##      |     +  init                                                   ##
##      |     |    + protect                                            ##
##      |     |    + protect_ext                                        ##
##      |     |    + protect_all_but                                    ##
##      |     +  supress_file                                           ##
##      |     +  supress_dir                                            ##
##                                                                      ##
##########################################################################
##
## BEGIN INCLUDE
##
import os

#1 #########################################################################
def Clean_run(run_name):

    clean=Clean()
    clean.protect(['CVS','madevent','madweight','main_condor','schedular'])
    clean.protect_ext(['f','o','py','inc','ps','sym','cmd'])
    os.chdir('./SubProcesses')
    dir_list=[]
    ls=os.listdir('.')
    for element in ls:
        if os.path.isdir(element) and (element[0]=='P' or element[:4]=='MW_P'):
            status,mess=clean.suppress_dir(os.path.join(element,run_name))
            if not status:
                print 'supress ',element,' failed:' 
                print mess

    os.chdir(os.pardir)
    
#1 #########################################################################
def Clean_event(run_name,mode='short'):
    """ supress one complete MW_run (except Event file) """

    clean=Clean('s')
    if mode=='control-only':
        clean.protect_all_but(['start','stop','done','finish_level'])
    elif mode=='short':
        clean.protect_all_but(['events.lhe','start','stop','done'])
    else:
        clean.protect_all_but(['finish_level','all_iter.dat','ftn25','ftn26','results.dat','events.lhe','start','stop','done'])
    clean.protect_ext(['f','o','py','inc','mg','ps','sym'])
    os.chdir('./SubProcesses')
    dir_list=[]
    ls=os.listdir('.')
    for element in ls:
        if os.path.isdir(element) and element[0]=='P':
            status,mess=clean.suppress_dir(os.path.join(element,run_name))
            #if not status:
            #    print 'supress ',element,' failed:' 
            #    print mess
#            print '\n\n\n'
#            print mess
    os.chdir(os.pardir)

#1 #########################################################################
def Clean_weight(run_name,mode='short'):
    """ supress one complete MW_run (except Event file) """

    clean=Clean('s')
    if mode=='short':
        clean.protect_all_but(['start','stop','done'])
    else:
        clean.protect_all_but(['vegas_value.out','details.out','weigths.out','events.lhe','start','stop','done'])
    clean.protect_ext(['f','o','py','inc','mg','ps','sym','.dat'])
    os.chdir('./SubProcesses')
    dir_list=[]
    ls=os.listdir('.')
    for element in ls:
        if os.path.isdir(element) and len(element)>4 and element[:4]=='MW_P':
            status,mess=clean.suppress_dir(os.path.join(element,run_name))
            #if not status:
            #    print 'supress ',element,' failed:' 
            #    print mess
#            print '\n\n\n'
#            print mess
    os.chdir(os.pardir)
    

#1 #########################################################################
class Clean:
    """ supress one file or directory (check the content) sume rules of preservation can be defined
        supported option:
          f : (force)   remove rules of preservation
          s : (silent)  remove print at each delete
          n : (nothing) preserve all file/dir (no suppression) but write the schedulle suppression without n option
    """
##########################################################################
##      +Class Clean                                                    ##
##      |     +  init                                                   ##
##      |     |    + protect                                            ##
##      |     |    + protect_ext                                        ##
##      |     |    + protect_all_but                                    ##
##      |     +  supress_file                                           ##
##      |     +  supress_dir                                            ##
##########################################################################
    
    #2 #########################################################################
    def __init__(self,opt=''):

        self.protected=[]
        self.ext_protected=[]
        self.protect_all=0
        self.opt=opt

    #3 #########################################################################
    def protect(self,name):
        if type(name)==list:
            self.protected+=name
        else:
            self.protected.append(name)
            
    #3 #########################################################################
    def protect_ext(self,name):
        if type(name)==list:
            self.ext_protected+=name
        else:
            self.ext_protected.append(name)
            
    #3 #########################################################################
    def protect_all_but(self,list=[]):
        self.protect_all=1
        self.authorized=list

    


    #2 #########################################################################
    def suppress_file(self,file,opt=''):

        if opt=='':
            opt=self.opt

        if 'f' not in opt:
            suppres,mess=self.check_suppresion(file)

        if suppres and os.path.isfile(file):
            if 'n' in opt:
                print 'schedulle deleting file',file
                return 1,mess
            try:
                os.remove(file)
            except:
                pass
            if 's' not in opt:
                print 'delete file',file
            return 1,''
        elif suppres:
            return 0, 'not a file'+file
        else:
            return suppres,mess

    #3 #########################################################################
    def check_suppresion(self,pos):
        
        name=pos.split(os.sep)[-1]
        if name in self.protected:
            return 0,'error: file protected '+str(name)
        ext=name.split('.')[-1]
        if ext in self.ext_protected:
            return 0,'error: extension protected '+pos
        if self.protect_all and name not in self.authorized:
            return 0,'error: not in authorized_file'
        else:                                                     
            return 1, ''
        
    #2 #########################################################################
    def suppress_dir(self,pos,opt=''):

        if opt=='':
            opt=self.opt
        
        supress=1
        message=''
        try:
            content=os.listdir(pos)
        except:
            return 0, 'no directory'
                    
        if 'f' not in opt:
            supress,message=self.check_suppresion(pos)

            for element in content:
                if os.path.isfile(os.path.join(pos,element)):
                    value,mess=self.suppress_file(os.path.join(pos,element),opt)
                elif os.path.isdir(os.path.join(pos,element)):
                    value,mess=self.suppress_dir(os.path.join(pos,element),opt)
                else:
                    value=0
                    mess="unknow data type "+str(element)+str(os.path.isdir(element))
                supress*=value
                if mess:
                    message+=mess+'\n'

        if supress:
            if 'n' not in opt:
                os.rmdir(pos)
                if 's' not in opt:
                    print 'delete dir',pos
            else:
                print 'schedulle removing directory',pos
                
            return 1,''
        else:
            return 0,message

#2 #########################################################################
if (__name__=='__main__'):

    from MW_param import go_to_main_dir

    go_to_main_dir()
    #Clean_event('fermi')
    #Clean_run('fermi')
    print 'no cleaning by default'
