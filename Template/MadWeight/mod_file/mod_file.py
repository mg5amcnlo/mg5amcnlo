#!/usr/bin/env python

#Extension
import string
import os
import sys
import time
import re

############################################################################
##                          PARAMETER
############################################################################

def fuse_f77_files(list_input,output):

    Pattern=re.compile(r'''^\s*subroutine\s*(?P<name>\w*)\s*''',re.I)
    comment_line='C*****************************************************\
               \nC**                 NEXT FILE                      **c\
               \nC*****************************************************'
         
    
#SECURITY POINT
    if type(list_input)!=list:
        print 'subroutine need at least two files for fuse'
        return 0
    
    if output in list_input:
        print 'output file cann\'t have the same name than an input file'
        return 0

#initialisation
    gg=open(output, 'w')
    subroutine_name=[]
    routine_already_exist=False
    
    for input in list_input:

        ff=open(input,'r')
        #read  file and add non-existent subroutine
        while 1:
            line=ff.readline()
            if line=="":
                break
            if Pattern.search(line):
                #detect new subroutine -> update write permision
                if Pattern.search(line).groups('name') in subroutine_name:
                    routine_already_exist=True
                else:
                    routine_already_exist=False
                    subroutine_name.append(Pattern.search(line).groups('name'))
            #write subroutine if not already exist
            if routine_already_exist:
                continue
            else:
                gg.writelines(line)

        #add comment between different file
        gg.writelines(comment_line)
        #close file
        ff.close()

    gg.close()
    return 1


def mod_matrix(input_file,output_file,new_rel_pos):

    ##Pattern to look at
    Pattern=[]
    Pattern.append(re.compile(r'''\s*(DATA)\s*multi_channel/.\w{4,5}./''',re.I))
    Pattern.append(re.compile(r'''\s*(include)\s*"(\D*)"''',re.I))
    Pattern.append(re.compile(r'''\s*(SAVE)\s*''',re.I))   
    Pattern.append(re.compile(r'''c*\s*(WRITE)\s*''',re.I))
    Pattern.append(re.compile(r'''\s*(DO)\s*J\s*=\s*\d\s*[,]\s*ISUM_HEL\s*''',re.I))
    

    ff=open(input_file,'r')
    gg=open(output_file,'w')


    while 1:
        j=0
        line=ff.readline()
        if line=="":
            break
        for i in range(0,len(Pattern)):
            if(Pattern[i].search(line)):
                j=j+1
                value=Pattern[i].search(line).groups()
                if(i==0):
                    line=mod_data(line)
                    break
                elif(i==1):
                    line=mod_include(line,value,new_rel_pos)
                    break
                elif(i==2):
                    line=""
                    break
                elif(i==3):
                    line=""
                    break
                elif(i==4):
                    line=mod_randomHel(line)
                    break

        gg.write(line)
    gg.close()
#########################################################################"

def rem_open_file(input_file,output_file):

    file_in=open(input_file,'r')
    file_out=open(output_file,'w')
    mode=1
    Pattern=re.compile(r'''^\s*subroutine\s*(\w*_\w*)\s*\(''',re.I)
    while 1:
        line=file_in.readline()
        if line=="":
            break
        if mode and not(Pattern.search(line)):
            file_out.writelines(line)
        elif (not mode and not(Pattern.search(line))):
            file_out.writelines("c"+line)
        elif(mode and Pattern.search(line)):
            value=Pattern.search(line).groups()[0]
            if value=="open_file_mdl":
                mode=0
                file_out.writelines("c"+line)
            else:
               file_out.writelines(line)
        else:
            mode=1
            print(line)
        
    return

#########################################################################"
def mod_model_make(pos_file):

    os.system('cp '+pos_file+' '+pos_file+'_bak')
    mod_file=open(pos_file,'r')

    pattern=re.compile(r'''MODEL\s*=\s*printout.o\s*couplings.o''')
    text=''
    while 1:
        line=mod_file.readline()
        if line=="":
            break   
        if pattern.search(line):
            text+=line[:-1]+' lha_reading.o \n'
        else:
            text+=line
    mod_file.close()
    mod_file=open(pos_file,'w')
    mod_file.writelines(text)
    mod_file.close()    
    
#########################################################################"

def mod_photon_flux(input_file,output_file):

    file_in=open(input_file,'r')
    file_out=open(output_file,'w')

    Pattern=re.compile(r'''\bphi\b''',re.I)  

    while 1:
        line=file_in.readline()
        if line=="":
            break
        if Pattern.search(line):
            new_line=Pattern.split(line)
            print_line=new_line[0]
            for i in range(1,len(new_line)):
                print_line+="phi_f"+new_line[i]                
            file_out.writelines(print_line)           
        else:
            file_out.writelines(line)

    file_in.close()
    file_out.close()
    print 'no need of mod_photon_flux'
    sys.exit()
    return

#########################################################################"

def mod_pdfwrap(input_file,output_file):

    file_in=open(input_file,'r')
    file_out=open(output_file,'w')

    Pattern=re.compile(r'''../alfas.inc''',re.I)
    replace="../../parameters/alfas.inc"
    
    while 1:
        line=file_in.readline()
        if line=="":
            break
        if Pattern.search(line):
            new_line=Pattern.split(line)
            print_line=new_line[0]
            for i in range(1,len(new_line)):
                print_line+=replace+new_line[i]
            file_out.writelines(print_line)
        else:
            file_out.writelines(line)

    file_in.close()
    file_out.close() 
#########################################################################"
def mod_run_card(input_file,output_file):

    file_in=open(input_file,'r')
    file_out=open(output_file,'w')
    
    Pattern_min=re.compile(r'''^\s*[\d.d]*\s*=\s*\w*\s*!\s*min([\s]|[imum\s])''',re.I)   
    Pattern_max=re.compile(r'''^\s*[\d.d]*\s*=\s*\w*\s*!\s*max([\s]|[imum\s])''',re.I)
    Pattern_dist=re.compile(r'''^\s*[\d.d]*\s*=\s*\w*\s*!\s*dist([\s]|[ance\s])''',re.I)

    while 1:
        line=file_in.readline()
        if line=="":
            break    
        if Pattern_min.search(line):
            line_split=line.split('=',1)
#            print line_split,Pattern_min.search(line)
            print_line=" 0   ="+line_split[1]
            file_out.writelines(print_line)            
        elif Pattern_max.search(line):
            line_split=line.split('=',1)
#            print line_split,Pattern_min.search(line)            
            print_line=" 1d2   ="+line_split[1]
            file_out.writelines(print_line)            
        elif Pattern_dist.search(line):
            line_split=line.split('=',1)
#            print line_split,Pattern_min.search(line)           
            print_line=" 0   ="+line_split[1]
            file_out.writelines(print_line)            
        else:
            file_out.writelines(line)

#########################################################################"
def mod_auto_dsig(input_file,output_file,new_rel_pos):
    
    file_in=open(input_file,'r')
    file_out=open(output_file,'w')

    ##part concerning the Transfert weight to add in file
    transfert_text="""
         call transfer_fct(PP(0,1),TWGT)
         If (twgt .lt. 10d0 .and. twgt .ge. 0d0) then
           dsig=dsig*twgt
         else
             write(*,*) \"Error in transfer function twgt=\",twgt
             dsig = 0d0
         endif
    """
  
    ##Pattern to look at
    Pattern=[]
    Pattern.append(re.compile(r'''\s*(include)\s*"(\D*)"''',re.I))
    Pattern.append(re.compile(r'''DOUBLE\s+PRECISION\s+[\w,]*rwgt''',re.I))   
    Pattern.append(re.compile(r'''\brwgt\b''',re.I))
    Pattern.append(re.compile(r'''\*\s*conv\s*\*''',re.I))
    Pattern.append(re.compile(r'''\s*\call\s+unwgt''',re.I))

    while 1:
        line=file_in.readline()
        if (line==""):
            break
        for i in range(0,len(Pattern)):
            if(Pattern[i].search(line)):
                value=Pattern[i].search(line).groups()
                if(i==0):
                    line=mod_include(line,value,new_rel_pos)
                    break
                if(i==1):
                    line=line.replace('rwgt','twgt')
                    break
                if(i==2):
                    line="c"+line
                    break
                if(i==3):
                    line=Pattern[i].sub('*',line)
                    #no break this line can still be modified
                if(i==4):
                    line=transfert_text
                    break
        file_out.write(line)
    file_out.close()
    file_in.close()
                   
##
## FUNCTION
        

def mod_data(line):
    return """      DATA multi_channel/.false./ \n"""

def mod_include(line,value,new_rel_pos):
    return "      include \""+new_rel_pos+'/'+value[1]+"\"\n" 

def mod_randomHel(line):
    return line+"      write(*,*) \"problem: random helicities\"\n"



############################################################################
##             TEST/Main_program
###########################################################################
if(__name__=="__main__"):
#    input_file="/nfs/cms/mass5/USERS/o/omatt/top_mass/MADGRAPH/HWW/SubProcesses/P_gg_e+mu-vevmx/matrix.f"
#    output_file="/nfs/cms/mass5/USERS/o/omatt/top_mass/pamo_HWW/SubProcess/P_gg_H_E+mu-vevmx/src/matrix.f"
   # output_file="./matrix.f"
#    new_rel_pos="../../../parameters/"
#    mod_matrix(input_file,output_file,new_rel_pos)
    input_file="""/nfs/cms/mass5/USERS/o/omatt/top_mass/MADGRAPH/test3_1neut/SubProcesses/P_gg_bbxudxe-vex/auto_dsig.f"""
    output_file="""/nfs/cms/mass5/USERS/o/omatt/top_mass/MadWeigth/TEST_OLI/SubProcesses/P_gg_bbxudxe-vex/src/auto_dsig.f"""
    mod_auto_dsig(input_file,output_file,"../../../parameters")
