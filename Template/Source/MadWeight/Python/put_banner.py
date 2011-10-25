#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MG/ME/MW                               ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author:                                                            ##
##     Mattelaer Olivier // MadGraph Team                               ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:02/06/09                                                ##
##                                                                      ##
##########################################################################
##   How to launch?                                                     ##
##   --------------                                                     ##
##                                                                      ##
##   for MadEvent: ./bin/put_banner.py event_file                       ##
##     (in python file)banner=ME_Banner(event_file)                     ##
##                     banner.write()                                   ##
##   for MadWeight:                                                     ##
##     (in python file)banner=ME_Banner(out_file)                       ##
##                     banner.write()                                   ##
##########################################################################
##   Structure                                                          ##
##   ---------                                                          ##
##                                                                      ##
##   mother class: Banner                                               ##
##   child class : ME_Banner and MW_Banner                              ##
##  Routine details ######################################################
##                                                                      ##
##---Banner:------------------------------------------------------------##
##     - init:                                                          ##
##     |    initialization of global variable:                          ##
##     |      version_info   : dict containing label->file position     ##
##     |                       containing the version number associate  ##
##     |                       to the label                             ##
##     |      card_info      : dict containing label-> file position    ##
##     |                       of the Card associate the label          ##
##     |      full_banner_txt: void text will be filled by all the      ##
##     |                       information to write in the banner       ##
##     |                                                                ##
##     +put_header:                                                     ##
##     |   put in variable full_banner_txt the content of the file      ##
##     |   defined in position self.header_file                         ##
##     |                                                                ##
##     +put_version_info:                                               ##
##     |   put in variable full_banner_txt the version information      ##
##     |                                                                ##
##     +put_info_card:                                                  ##
##     |   put in variable full_banner_txt the copy of the card betwenn ##
##     |   xml tag. defined in self.card_info                           ##
##     |   if self.mod_TAG(filepos) is defined the information between  ##
##     |   the tag is the output of that function                       ##
##     |                                                                ##
##     +put_events:                                                     ##
##     |   put in variable full_banner_txt the content of the file      ##
##     |   self.input_file                                              ##
##     |                                                                ##
##     +write_file:                                                     ##
##     |   write in file at location self.output_file the content of    ##
##     |   the variable self.full_banner_txt                            ##
##                                                                      ##
##---ME_Banner:---------------------------------------------------------##
##     derived from Class Banner                                        ##
##     - init(event_file):                                              ##
##     |    Banner initialization and definition of some variable       ##
##     |      self.input_file: position of file to add banner           ##
##     |                       (=event_file)                            ##
##     |      self.output_file: position to write the output            ##
##     |                       (=event_file)                            ##
##     |      self.header_file: position of the header file             ##
##     |                       (=./Events/banner_header.txt)            ##
##     |                                                                ##
##     + write:                                                         ##
##     |   main schedular to write the banner                           ##
##     |                                                                ##
##     + check_grid_pack:                                               ##
##     |   check if the grid_pack option is active and if it is update  ##
##     |   the variable self.card_info in order to add this card in the ##
##     |   banner                                                       ##
##     |                                                                ##
##     + mod_MGRUnGrid(filepos):                                        ##
##     |   modify the seed of the run_card (if needed) and retrun the   ##
##     |   modify (or original) run_card.dat                            ##
##                                                                      ##
##---MW_Banner:---------------------------------------------------------##
##     derived from Class Banner                                        ##
##     -init(output_file)                                               ##
##     |    use Banner init and define interesting input/output/header  ##   
##     |    update the global version and card for MW run               ##
##     |                                                                ##
##     + write:                                                         ##
##     |   main schedular to write the banner                           ##
##                                                                      ##
##########################################################################



import os,re,sys
sys.path+=['../'*i+'./Source/MadWeight/Python' for i in range(1,6)]
import MW_param






#1 #######################################################################
class Banner:

    #2 #######################################################################
    def __init__(self):
        """ initialization of global variable:                                  ##
        ##     |      version_info   : dict containing label->file position     ##
        ##     |                       containing the version number associate  ##
        ##     |                       to the label                             ##
        ##     |      card_info      : dict containing label-> file position    ##
        ##     |                       of the Card associate the label          ##
        ##     |      full_banner_txt: void text will be filled by all the      ##
        ##     |                       information to write in the banner       ##
        """
        
        self.version_info={'MG/ME version':'./MGMEVersion.txt',
                           'madgraph version':'./SubProcesses/MGVersion.txt',
                           'template version':'./TemplateVersion.txt',
                           'helas version':'./Source/DHELAS/HELASVersion.txt',
                           'model version':'./Source/MODEL/ModelVersion.txt'}
        
        self.card_info={'MGProcCard':'./Cards/proc_card_mg5.dat',
                        'slha':'./Cards/param_card.dat',
                        'MGRunCard':'./Cards/run_card.dat'}
        
        self.full_banner_txt=''
        
    #2 #######################################################################
    def put_header(self):
        """    |   put in variable full_banner_txt the content of the file      ##
        ##     |   defined in position self.header_file                         ##
        """
        self.full_banner_txt+=file(self.header_file,'rU').read()

    #2 #######################################################################
    def put_version_info(self):
        """ find the version number and create the text_file """
        
        self.full_banner_txt+="<MGVersion>\n"
        for key,pos in self.version_info.items():
            try:
                self.full_banner_txt+="# "+key+' '*(25-len(key))+':'+file(pos,'rU').read()
            except IOError:
                self.full_banner_txt+="# "+key+' '*(25-len(key))+':'
            if self.full_banner_txt[-1]!='\n':
                self.full_banner_txt+='\n'
        self.full_banner_txt+="</MGVersion>\n"

    #2 #######################################################################        
    def put_info_card(self):
        """    |   put in variable full_banner_txt the copy of the card betwenn ##
        ##     |   xml tag. defined in self.card_info                           ##
        ##     |   if self.mod_TAG(filepos) is defined the information between  ##
        ##     |   the tag is the output of that function                       ##
        """
        
        for key,pos in self.card_info.items():
            self.full_banner_txt+="<"+key+">\n"
            #check if a special routine has to be apply on this key (def by mod_KEY)
            try:
                self.full_banner_txt+=eval('self.mod_'+key+'(\"'+pos+'\")')
            except:
                self.full_banner_txt+=file(pos,'rU').read()
            self.full_banner_txt+="</"+key+">\n"

    #2 #######################################################################
    def put_events(self):
        """    |   put in variable full_banner_txt the content of the file      ##
        ##     |   self.input_file                                              ##
        """
        
        self.full_banner_txt+=file(self.input_file,'rU').read()
        if self.full_banner_txt[-1]!='\n':
            self.full_banner_txt+='\n'
        self.full_banner_txt+="</LesHouchesEvents>\n"

    #2 #######################################################################
    def write_file(self):
        """    |   write in file at location self.output_file the content of    ##
        ##     |   the variable self.full_banner_txt                            ##
        ##     |   during this step the empty line are removed                  ##
        """
        suppress_empty_line=re.compile(r'''\n\s*\n''')
        self.full_banner_txt=suppress_empty_line.sub('\n',self.full_banner_txt)
        output=file(self.output_file,'w')
        output.writelines(self.full_banner_txt)
        output.close()



#1 #######################################################################            
class ME_Banner(Banner):

    #2 #######################################################################
    def __init__(self,event_file,initdir=''):
        """    |    Banner initialization and definition of some variable       ##
        ##     |      self.input_file: position of file to add banner           ##
        ##     |                       (=event_file)                            ##
        ##     |      self.output_file: position to write the output            ##
        ##     |                       (=event_file)                            ##
        ##     |      self.header_file: position of the header file             ##
        ##     |                       (=./Events/banner_header.txt)            ##
        """
        
        Banner.__init__(self)
        if os.path.isfile(event_file):
            self.input_file=event_file
        elif os.path.isfile('./Events/'+event_file):
            self.input_file='./Events/'+event_file
        elif os.path.isfile(initdir+'/'+event_file):
            self.input_file=initdir+'/'+event_file
        else:
            raise Exception('no file '+event_file+' know')
        self.output_file=self.input_file
        self.header_file='./Events/banner_header.txt'

    #2 #######################################################################        
    def write(self):
        """   |   main schedular to write the banner                           ##
        """
        
        self.put_header()          #put default header
        self.check_grid_pack()     #check if gridpack is actif and if we need to write this file
        self.put_version_info()    #define the version of all the part
        self.put_info_card()       #define all the text linked to the card
        self.put_events()          #define the text linked to the event
        self.write_file()          #write the file

    #2 #######################################################################
    def check_grid_pack(self):
        """    |   check if the grid_pack option is active and if it is update  ##
        ##     |   the variable self.card_info in order to add this card in the ##
        ##     |   banner                                                       ##
        """

        pattern=re.compile(r'''(?P<value>.true.|.false.)\s*=\s*gridpack''')
        value=pattern.search(open('./Cards/run_card.dat').read()).group('value')

        if value=='.true.':
            self.card_info['MGGridCard']='./Cards/grid_card.dat'

    #2 #######################################################################        
    def mod_MGRunCard(self,inputfile):
        """    |   modify the seed of the run_card (if needed) and retrun the   ##
        ##     |   modify (or original) run_card.dat                            ##
        """

        text=file(inputfile).read()
        if os.path.isfile('./SubProcesses/randinit'):
            pattern_random=re.compile(r'''r=\s*(?P<val>\d*)''')
            print file('./SubProcesses/randinit').read()
            seed=pattern_random.search(file('./SubProcesses/randinit').read()).group('val')
            pattern_seed=re.compile(r'''0\s*=\s*iseed''')
            text=pattern_seed.sub(' '+seed+' = iseed',text)#

        return text


#1 #######################################################################    
class MW_Banner(Banner):

    #2 #######################################################################    
    def __init__(self,event_file):
        """    |    use Banner init and define interesting input/output/header  ##
        ##     |    update the global version and card for MW run               ##
        """
        
        Banner.__init__(self)
        self.input_file=''
        self.output_file=event_file
        self.header_file='./Source/MadWeight/MW_banner_header.txt'

        #adding information for version
        self.version_info['MW version']='./Source/MadWeight/MW_TemplateVersion.txt'
        self.version_info['Transfer_Function version']='./Source/MadWeight/transfer_function/Transfer_FctVersion.txt'

        #adding card information
        self.card_info['MWCard']='./Cards/MadWeight_card.dat'
        self.card_info['TransferCard']='./Cards/transfer_card.dat'

    #2 #######################################################################        
    def write(self):
        """   |   main schedular to write the banner                           ##
        """
        
        self.put_header()          #put default header
        self.put_version_info()    #define the version of all the part
        self.put_info_card()       #define all the text linked to the card
        self.write_file()          #write the file

############################################################################
if '__main__'==__name__:
    """ if this script is launched by hand do the old put banner of MG_ME """

    def_pos=MW_param.move()
    def_pos.to_main()
    opt=sys.argv
    if len(opt)==1:
        filename=raw_input('Enter file with events (in directory Events)')
    else:
        filename=opt[1]
    
    me_banner=ME_Banner(filename,initdir=def_pos.initpos)
    me_banner.write()
