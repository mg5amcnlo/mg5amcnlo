#!/usr/bin/env python
import os,re,sys
sys.path+=[i*'../'+'./Source/MadWeight_File/Python' for i in range(0,5)]
import MW_param


#import MW_param

class move:
    def __init__(self):
        self.initpos=os.getcwd()
        self.old=self.initpos

    def to_SubProcesses(self):
        
        old_pos=os.getcwd()+'/'
        self.old=old_pos
        if old_pos.count('/SubProcesses/')>1:
            new_pos=pos.split('/SubProcesses/')[0]+'/SubProcesses/'
        elif old_pos.count('/SubProcesses/')==0:
            new_pos=old_pos+'/SubProcesses/'
        else:
            new_pos='/SubProcesses/'.join(old_pos.split('/SubProcesses/')[:-1])+'/SubProcesses/'
            
        if new_pos!=old_pos:
            self.old=old_pos
            os.chdir(new_pos)
            
    def to_init(self):
        self.old=os.getcwd()
        os.chdir(self.initpos)

    def store(self):
        self.store=os.getcwd()

    def to_store(self):
        self.old=os.getcwd()
        os.chdir(self.store)

    def to_last_pos(self):
        pos,self.old=self.old,os.getcwd()
        os.chdir(pos)

    def to_main(self):
        old=os.getcwd()
        try:
            MW_param.go_to_main_dir()
        except:
            self.to_SubProcesses()
            os.chdir('..')
        self.old=old

class status:
    def __init__(self):
        self.def_pos=move()
        ff=open('start','w')
        ff.writelines('start')
        ff.close()

    def to_end(self):
        self.def_pos.store()
        self.def_pos.to_init()
        ff=open('stop','w')
        ff.writelines('stop')
        ff.close()
        self.def_pos.to_store()

#1 #########################################################################
class compute_acc:

    #2 #####################################################################
    def __init__(self,MWparam='',run_name='',cardnb='',auto=0):
        """compute all the term """

        self.def_pos=move()
        if run_name=='':
            self.run_name,self.cardnb=self.find_position_information()
        else:
            self.run_name,self.cardnb=run_name,cardnb
        
        if not MWparam:
            self.def_pos.to_main()
            import madweight #need to include these library in order to have the correct sys.path defined
            MWparam=MW_param.MW_info('MadWeight_card.dat')            
            self.def_pos.to_last_pos()
        self.MWparam=MWparam

        if auto==1:
            self.main()
    #2 #####################################################################
    def find_position_information(self):
        pos=os.getcwd()
        pos=pos.split('/')
        try:
            run_name=pos[-2]
            cardnb=int(pos[-1].split('_')[-1])
            return run_name,cardnb
        except:
            return 0,0
        
    #2 #####################################################################
    def main(self,run_name="",card_nb=""):
        """ launch all the acceptance term must be launch from SubProcesses """
        
        if not run_name:
            run_name=self.run_name
            card_nb=self.cardnb

        #ensure that we are at the correct position (or go to that position)
        self.def_pos.store()
        self.def_pos.to_SubProcesses()
        
        self.combine(run_name,card_nb)
        self.run_pythia(run_name,card_nb)
        self.run_pgs(run_name,card_nb)
        self.apply_cut(run_name,card_nb)

        #ensure that we go back to initial directory position (if move)
        self.def_pos.to_store()

    #2 #####################################################################
    def combine(self,run_name,card_nb):
        """create combine event"""
            
        ff=open('acc_input_'+str(card_nb),'w')
        ff.writelines(str(card_nb)+'\n')
        ff.writelines(str(run_name)+'\n')
        ff.close()
        os.system('../bin/combine_events < acc_input_'+str(card_nb))
        #treat result
        import mod_file
        mod_file.mod_file('../Events/'+run_name+'/unwgt_event_'+str(card_nb)+'.lhe',{'S-REGEXP_              nan+0.00000000000E\+00+re.I':''})
        #find the number of Events:
        events_pattern=re.compile(r'''Number of Events\s*\:\s*(?P<event_nb>\d+)\s*''')
        try:
            self.generated_events=events_pattern.search(file('../Events/'+run_name+'/unwgt_event_'+str(card_nb)+'.lhe').read()).group('event_nb')
        except:
            print 'not found'

        os.system('../bin/put_banner.py ../Events/'+run_name+'/unwgt_event_'+str(card_nb)+'.lhe')

        
    #2 #####################################################################
    def run_pythia(self,run_name,card_nb):
        """run pythia"""
        
        ff=open('acc_input_'+str(card_nb),'w')
        ff.writelines('pythia_'+str(card_nb)+'.hep\n')
        ff.writelines('../Events/'+run_name+'/unwgt_event_'+str(card_nb)+'.lhe\n')
        ff.close()
        os.system('export PDG_MASS_TBL=../../MW_pythia-pgs/src/mass_width_2004.mc;../../MW_pythia-pgs/src/pythia < acc_input_'+str(card_nb))

    #2 #####################################################################
    def run_pgs(self,run_name,card_nb):
        """run pgs"""
        
        ff=open('acc_input_'+str(card_nb),'w')
        ff.writelines('pythia_'+str(card_nb)+'.hep\n')
        ff.writelines('pgs_'+str(card_nb)+'.lhco\n')
        ff.close()
        os.system('export PDG_MASS_TBL=../../MW_pythia-pgs/src/mass_width_2004.mc;../../MW_pythia-pgs/src/pgs < acc_input_'+str(card_nb))

    #2 #####################################################################
    def apply_cut(self,run_name,card_nb):
        """ take the pgs file and apply the cut to find efficiency """
        import verif_event

        self.def_pos.to_main()
        select=verif_event.Lhco_filter(self.MWparam.MW_listdir[0],self.MWparam,auto=0,write_mode=0)
        self.event_pass_cut=select.verif_event('./SubProcesses/pgs_'+str(card_nb)+'.lhco')
        
        ff=open('Events/'+run_name+'/cut_efficiency.out','a')
        ff.writelines('\t'.join([str(val) for val in [card_nb,self.generated_events,self.event_pass_cut]])+'\n')


if __name__=='__main__':
    def_pos=move()
    def_status=status()
    acc_run=compute_acc()
    def_pos.to_SubProcesses()
    acc_run.main()
    def_status.to_end()

