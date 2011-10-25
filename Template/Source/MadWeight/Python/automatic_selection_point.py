#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##       email:  olivier.mattelaer@uclouvain.be                         ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:14/03/09                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##     + video_code                                                     ##
##     |    + init                                                      ##
##     |    + loop                                                      ##
##     |    + control_integration_precision                             ##
##     |    + prepare_next_grid                                         ##
##     |    + compute_weight                                            ##
##     |    + charge_new_events                                         ##
##     |    + store_output_files                                        ##
##                                                                      ##
##     + nextrunschedular                                               ##
##     |    C Errorschedular                                            ##
##     |    + init                                                      ##
##     |    + reinit                                                    ##
##     |    +    readmapfile                                            ##
##     |    +    readlikfile                                            ##
##     |    + select_area                                               ##
##     |    +    find_min                                               ##
##     |    +    find_point_under_limit                                 ##
##     |    +    extend_to_neighbour                                    ##
##     |    +    define_as_neighbour                                    ##
##     |    +       card_at_position                                    ##
##     |    +       position_to_card                                    ##
##     |    +       define_grid                                         ##
##     |    +    redefine_mapping_file                                  ##
##     |    + update_grid                                               ##
##     |    +    update_grid_by_axis                                    ##
##     |    +    find_new_card_from_position                            ##
##     |    +    define_new_phase_space_point                           ##
##     + main:                                                          ##
##          launch main program                                         ##
##########################################################################
#Extension
import time
import sys
import os
import re
sys.path.append('./Source/MadWeight/Python')
sys.path.append('../Source/MadWeight/Python')
import MW_param
from verif_event import verif_event
import madweight
import plot
#end patch

class video_code:
    """ the object controlling the video code """

######## run parameter ############################################################################
    events_by_step=100                                                                           ##
    max_events=500                                                                               ##
    cut_value=6                                                                                  ##
    max_param_for_refine_grid=40                                                                 ##
    max_param_for_refine_grid_by_axis=9                                                          ##
    step=0                                                                                       ##
    starting_point=0                                                                             ##
    start_refine=0                                                                               ##
###################################################################################################
    
    class ERRORVIDEO(Exception): pass
    
    def __init__(self,MWparam):
        self.nevents=int(MWparam.nb_event)
        self.MWparam=MWparam
        #self.check=self.Check(self.MWparam,self)
        #check the status of the run at the starting points
        #self.check.check_step(self.step)
        #self.check.check_events_by_step(self.events_by_step)
        #self.check.check_nevents(self.nevents)

    def loop(self):
        """ main program """

#        self.MWparam.n_join=3
        self.compute_weight(self.nevents,control=0)
#        self.MWparam.n_join=30
#        self.control_integration_precision()
        while self.nevents<self.max_events:
            if self.nevents>self.start_refine:
                self.MWparam.refine=1
            self.create_output()
            new_grid=self.prepare_next_grid()
            self.store_output()
#            self.update_events_by_step(self.nevents)
            if new_grid:
                self.compute_weight(self.nevents,events_by_step=self.events_by_step)
            else:
                self.MWparam.startevent=int(self.nevents)
                self.charge_new_events(self.nevents,self.nevents+self.events_by_step)
                self.compute_weight(self.nevents)

#     class Check:
#        """ class of routine to control that the video code was on correct shape """
#
#        def __init__(self,MWparam,mother):
#            self.MWparam=MWparam
#            self.mother=mother
#            
#        def check_step(self,value):
#
#            file_list=os.listdir('./Cards/')
#            file_list=[name for name in file_list if len(name)>18]
#            file_list=[name[19:] for name in file_list if name[:18]=="mapping_card_step_"]
#            value_list=[int(name.split('_')[0]) for name in file_list]
#            expected_value=max([0]+value_list)
#            self.compare(value,expected_value,'step')

    def control_integration_precision(self,mult_fact=10):
        """ launch a refine with more points """

#        option=[' ','-refine=0.01']                    #prepare the option
#        self.MWparam.set_run_opt(option)
#        madweight.Launch_all_SubProcess(self.MWparam) # launch the refine
#        self.MWparam.n_join=1
        normal_run_with_refine=0
        if hasattr(self.MWparam,'refine') and self.MWparam.refine:
            normal_run_with_refine=1
            self.MWparam.refine=0
        
        option=[' ','refine=0.01']                    #prepare the option
        self.MWparam.set_run_opt(option)
        self.MWparam['mw_run']['6']*=mult_fact               # boost the number of points
        madweight.Launch_all_SubProcess(self.MWparam) # launch the refine
        self.MWparam['mw_run']['6']/=mult_fact               # restore the number of points 

        if normal_run_with_refine:
            self.MWparam.refine=1
        
    def create_output(self):

        value=self.MWparam.startevent
        self.MWparam.startevent=0
        self.MWparam.norm_with_cross=1
        option=[' ','-8']                    #prepare the option
        self.MWparam.set_run_opt(option)
        madweight.Launch_all_SubProcess(self.MWparam) # launch the refine
        self.MWparam.norm_with_cross=1
        plot.Likelihood(mw_param=self.MWparam,auto=1)      # create the updated output file
        self.MWparam.startevent=value
            
    def prepare_next_grid(self):
        """ analyze the precedent run and prepare the nex one """

        print 'old actif card',len(self.MWparam.actif_param)
        #create the object for the analysisof the old run /preparing next run
        self.analrun=next_run_schedular('./Cards/mapping_card.dat','./Events/'+MWparam.name+'/likelihood_value.out',MWparam)
        #select the interesting area for the next pahse
        self.analrun.select_area(self.cut_value)
        #check if we need to add new card
        new_grid=self.analrun.update_grid(self.max_param_for_refine_grid,self.max_param_for_refine_grid_by_axis) 
        print 'now ',len(self.MWparam.actif_param)
#       adding point by hand!       
#        if self.step==22:
#            new_point= [[0,512.5],[12.5,512.5],[25,512.5],[37.5,512.5],[50,512.5],[62.5,512.5]]
#            self.analrun.define_new_phase_space_point(new_point)
#            new_grid=1
        
        self.MWparam.update_nb_card()
        
        return new_grid

    
    def compute_weight(self,nevents,events_by_step='',control=1):
        """ compute the weight for the actif card """

        if events_by_step=='':
            events_by_step=nevents
            update_startevent=0
        else:
            #we have to restart from zero
            self.MWparam.startevent=0
            update_startevent=1
            
        #put the grid at the same level as the rest
        self.MWparam['mw_run']['2']=0
        self.MWparam.nb_event=0
        print 'compute weight number of step:',nevents//events_by_step
        for i in range(0,nevents//events_by_step):
                self.MWparam['mw_run']['2']+=events_by_step
                self.MWparam.nb_event+=events_by_step
                print 'start at',self.MWparam.startevent,'stop at',self.MWparam['mw_run']['2']
                if control:
                    #launch the new one (by relaunch)
                    option=[' ','-8']
                    self.MWparam.set_run_opt(option)
                    madweight.Launch_all_SubProcess(self.MWparam)
                self.MWparam.norm_with_cross=1
                option=[' ','relaunch']
                self.MWparam.set_run_opt(option)
                madweight.Launch_all_SubProcess(self.MWparam)
                #check the precision
                self.control_integration_precision()
                if update_startevent:
                    self.MWparam.startevent+=events_by_step

#    def update_events_by_step(self,nevents):
#        """ autorize different step for the number of events """
#        if nevents<=200:
#            self.events_by_step=20
#        elif nevents<=1000: 
#            self.events_by_step=100
#        else:
#            self.events_by_step=200
        


    def charge_new_events(self,oldnb,newnb):
        """ create the new event directory """

        check=self.MWparam.find_existing_events()

        print 'creating events: old',oldnb,'check',check,'new',newnb
        print 'diff',newnb-oldnb

        #control to avoid a multi identical creation
        if check<oldnb:
            raise self.ERRORVIDEO, 'wrong number of events'
        elif check<newnb:
            oldnb=check 
        else:
            print 'events already generated'
            self.MWparam.nb_event=newnb
            self.nevents=newnb
            self.MWparam['mw_run']['2']=str(newnb)
            return
      
        self.MWparam['mw_run']['22']=1
        self.MWparam['mw_run']['21']=oldnb
        self.MWparam['mw_run']['2']=newnb-oldnb
        self.MWparam.nb_event=newnb-oldnb
        verif_event(MWparam)
        option=[' ','-5']
        MWparam.set_run_opt(option)
        madweight.Launch_all_SubProcess(MWparam)
        self.MWparam['mw_run']['22']=0
        self.MWparam.nb_event=newnb
        self.nevents=newnb
        self.MWparam['mw_run']['2']=newnb
        self.MWparam.nb_event=newnb

    def store_output(self):

        os.system('cp Cards/mapping_card.dat Events/'+self.MWparam.name+'/mapping_card_step_'+str(self.step)+"_events_"+str(self.nevents)+'.out')

        os.system('cp Events/'+self.MWparam.name+'/likelihood_value.out Events/'+self.MWparam.name+\
                  '/likelihood_value_step_'+str(self.step)+"_events_"+str(self.nevents)+'.out')
                
        text="""
        {
        TTree* points=new TTree("points","Points");
        points->ReadFile("likelihood_value_step_"""+str(self.step)+"""_events_"""+str(self.nevents)+""".out","val:mQ:mlsp:value:error");
        points->SetMarkerStyle(2);
        points->Draw("value:mQ:mlsp","");
        

        points->SetMarkerColor(3);
        points->Draw("value:mQ:mlsp","value>"""+str(self.analrun.minimum+self.cut_value)+"""","same");
        points->SetMarkerColor(4);
        points->Draw("value:mQ:mlsp","value<"""+str(self.analrun.minimum+self.cut_value)+"""","same");}
        """
        
        ff=open('Events/'+self.MWparam.name+'/plot_step_'+str(self.step)+'.C','w')
        ff.writelines(text)
        ff.close()
        self.step+=1
        

#1 #########################################################################################
class next_run_schedular:
    class ERRORSCHEDULAR(Exception): pass
    
    def __init__(self,file1,file2,MWparam):

        self.mapfile=file1
        self.mapping=self.readmapfile(file1)
        self.likfile=file2
        self.likelihood=self.readlikfile(file2)
        self.MWparam=MWparam

    def reinit(self):
        self.mapping=self.readmapfile(self.mapfile)
        self.likelihood=self.readlikfile(self.likfile)
        self.define_grid()
        self.MWparam.update_nb_card()
        
    def readmapfile(self,map_file):
        """ read the mapping file and return a dictionary with keys: card number
        content: list of parameter value.
        only actif card are read"""
        
        out={}
        for line in open(map_file):
            splitline=line.split()
            if len(splitline)>1 and splitline[0][0]!='#' and splitline[-1]=='1':
                out[int(splitline[0])]=[float(data) for data in splitline[1:-1]]    
        return out

    def readlikfile(self,lik_file):
        """ read the likelihood file and return a dictionary with keys: card number
        content: likelihood value
        only actif card are read """

        out={}
        for line in open(lik_file):
            splitline=line.split()
            if len(splitline)>1 and splitline[0][0]!='#':
                out[int(splitline[0])]=float(splitline[-2])
                
        return out


    def select_area(self,gap=5):
        """ all routine for area selection gap:max differrence for the upper bound area """

        self.gap=gap
        self.minimum=self.find_min()
        print 'minimum', self.minimum
        print 'cut',self.minimum+gap
        self.find_point_under_limit(self.minimum+gap)
        self.extend_to_neighbour()
        self.redefine_mapping_file()
        self.reinit()


    def find_min(self):
        """ find the minimum_value in the likelihood_plot  return that value"""
        return min(self.likelihood.values())

    def find_point_under_limit(self,cut):
        """ select all the point where the likelihood is below the cut
            define self.under_limit a list containing the card number passing this selection
        """
        print 'underlimit',len([1 for nb_card,value in self.likelihood.items() \
                                             if value<=cut])
        self.under_limit=[nb_card for nb_card,value in self.likelihood.items() \
                          if value<=cut]
        return self.under_limit

    def extend_to_neighbour(self):
        """
        define self.extend_area containing with the neigbourgh
        definer self.border
        """
        self.border=[]
        
        for card in self.under_limit:
            position=self.position_to_card(card)
            for i in range(0,len(position)):
                self.define_as_neighbour(position[:i]+[position[i]+1]+position[i+1:],i)
                self.define_as_neighbour(position[:i]+[position[i]-1]+position[i+1:],i)

    def define_as_neighbour(self,position,level=-1):
        """
        check if a card exist for that position+check if this card is not in self.under limit
        then put this card in self.border
        """

        card_nb=self.card_at_position(position)

#        if card_nb==-1:
#            print 'reach border of the initial point -> no border',position
        
        if card_nb not in self.under_limit+self.border and card_nb!=-1:
            self.border.append(card_nb)

        #check for diagonal
        for i in range(level+1,len(position)):
            self.define_as_neighbour(position[:i]+[position[i]+1]+position[i+1:],i)
            self.define_as_neighbour(position[:i]+[position[i]-1]+position[i+1:],i)
            

        
        
    def card_at_position(self,pos_list):
        """
        return card_nb for the position return -1 if no card defined at this position
        this routine use self.grid initialize with self.define_grid(). this object will be initialize at the first call
        """
        
        try:
            return self.grid[str(pos_list)]
        except KeyError:
            return -1
        except AttributeError:
            self.define_grid()
            return self.card_at_position(pos_list)


    def position_to_card(self,card_nb):
        """
        return list of position for the card_nb
        this routine use self.grid initialize with self.define_grid(). this object will be initialize at the first call
        """
        try:
            return self.card_to_grid_position[card_nb]
        except KeyError:
            return -1
        except AttributeError:
            self.define_grid()
            return self.position_to_card(card_nb)

    def define_grid(self):
        """
        define self.card_to_grid_position: dict {card_nb:[list of coordinate on the grid]}
        self.grid: dict{[list of coordinate on the grid]:card_nb}\
        """

        self.grid={}
        self.card_to_grid_position={}
        
        #collect information and put in suitable form (all_value by direction)
        data=[]
        for param_list in self.mapping.values():
            for i in range(0,len(param_list)):
                try:
                    data[i]
                except:
                    data.append([])
                if param_list[i] not in data[i]:
                    data[i].append(param_list[i])
                    
        #order each direction
        for i in range(0,len(data)):
            data[i].sort()

        #pass on each card to assignate position
        for card,param_list in self.mapping.items():
            position=[]
            for i in range(0,len(param_list)):
                for j in range(0,len(data[i])):
                    if data[i][j]==param_list[i]:
                        position.append(j)
                        break
            assert len(position)==len(param_list),'some unknow values found'
            self.grid[str(position)]=card
            self.card_to_grid_position[card]=position

        #store for help
        self.position_to_value=data

    def redefine_mapping_file(self):
        """
        redefine mapping file for the desactivation of pointless param_card
        """
        
        for i in range(0,len(self.position_to_value)):
            print self.position_to_value[i]
        print 'underlimit',self.under_limit
        print 'border',self.border
        os.system('cp Cards/mapping_card.dat Cards/old_mapping.dat')

        ff=open('./Cards/new_mapping_file','w')
        for line in file('./Cards/mapping_card.dat'):
            splitline=line.split()
            if int(splitline[0]) in self.border+self.under_limit:
                ff.writelines('\t'.join(splitline[:-1])+'\t1\n')
            else:
                ff.writelines('\t'.join(splitline[:-1])+'\t0\n')
        ff.close()
        os.system('cp Cards/new_mapping_file Cards/mapping_card.dat')

    def update_grid(self,cut,cut_on_axis=0):

        if len(self.under_limit+self.border)>cut:
            out=self.update_grid_by_axis(cut_on_axis) #check for update on a axis by axis 
            return out

        new_point=[]
        for nbcard in self.under_limit+self.border:
            position=self.position_to_card(nbcard)
            point_list=self.find_newcard_from_position(position)
            for data in point_list:
                new_point.append(data)
        self.define_new_phase_space_point(new_point)
        
        return 1

    def update_grid_by_axis(self,cut,axis=-1):
        """ check if one specific axis need an update
            cut: maximum number of card for launching refine
            axis: which axis under study (-1==all)
        """
        print 'test axis by axis'
        if axis==-1:
            out=0
            for axis in range(0,len(self.position_to_value)):
                out+=self.update_grid_by_axis(cut,axis)
                return out%1
                
        position_on_axis=[]
        for nbcard in self.under_limit:
            position=self.position_to_card(nbcard)
            position_on_axis.append(position[axis])

        if len(position_on_axis)>cut:
            return 0 # too much of card on this axis to do an update

        new_point=[]
        for nbcard in self.under_limit+self.border:
            position=self.position_to_card(nbcard)
            point_list=self.find_newcard_from_position(position,start=axis,stop=axis)
            for data in point_list:
                new_point.append(data)
        self.define_new_phase_space_point(new_point)
        
        return 1
                                                                                        
        


    def find_newcard_from_position(self,position,compare_point='',start=0,stop=-1):
        if start==0 :
            # remove old output
            self.newcard_output=[]
            compare_point=position

        if stop==-1:
            stop=len(position)-1

        for i in range(start,stop+1):
            new_pos=list(compare_point)
            new_pos[i]+=1
            card_nb=self.card_at_position(new_pos)
            if card_nb==-1:
                continue
            self.find_newcard_from_position(position,new_pos,i+1)
            card=self.card_at_position(new_pos)
            if card in self.under_limit+self.border:
                new_value=[]
                for i in range(0,len(position)):
                    middle_point=(self.position_to_value[i][position[i]]+self.position_to_value[i][new_pos[i]])/2
                    new_value.append(middle_point)
                self.newcard_output.append(new_value)

        return self.newcard_output


    def define_new_phase_space_point(self,new_point):
        """ define the new card """

        self.MWparam['mw_parameter']['1'] = 2
        self.MWparam['mw_parameter']['2'] = 1
        
        for i in range(1,len(new_point[-1])+1):
            self.MWparam['mw_parameter'][str(10*i+3)]=[]
            for j in range(0,len(new_point)):
                self.MWparam['mw_parameter'][str(10*i+3)].append(str(new_point[j][i-1]))

        import create_param
        print 'before update' ,MWparam.nb_card,
        create_param.Param_card(run_name=MWparam)
        MWparam.update_nb_card()
        print 'after update' ,MWparam.nb_card
        #        Cards.create_include_file(MWparam)
        #        update_cuts_status(MWparam)






if __name__=='__main__':
    os.system('./bin/madweight.py -89')
    MW_param.go_to_main_dir()
    MWparam=MW_param.MW_info('MadWeight_card.dat')
    driver=video_code(MWparam)
    driver.loop()
    
    



    
            
