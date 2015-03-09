c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

c *************************************************************************
c **                                                                     **
c **                    LHA format reading routines                      **
c **                                                                     **
c *************************************************************************


c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_islatin -> islatin=true if letter is a latin letter
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine LHA_islatin(letter,islatin)
      implicit none

      logical islatin
      character letter
      integer i

      islatin=.false.
      i=ichar(letter)
      if(i.ge.65.and.i.le. 90) islatin=.true.
      if(i.ge.97.and.i.le.122) islatin=.true.

      end

c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_isnum -> isnum=true if letter is a number
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine LHA_isnum(letter,isnum)
      implicit none

      logical isnum
      character letter
      character*10 ref
      integer i

      isnum=.false.
      ref='1234567890'

      do i=1,10
        if(letter .eq. ref(i:i)) isnum=.true.
      end do

      end

c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_firststring -> first is the first "word" of string
c ++ Warning: string is returned with first REMOVED!
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine LHA_firststring(first,string)

      implicit none
      character*(*) string
      character*(*) first
      
      if(len_trim(string).le.0) return
      
      do while(string(1:1) .eq. ' ' .or. string(1:1) .eq. CHAR(9)) 
        string=string(2:len(string))
      end do
      if (index(string,' ').gt.1) then
         first=string(1:index(string,' ')-1)
         string=string(index(string,' '):len(string))
      else 
         first=string
      end if

      end


      subroutine LHA_case_trap(name)
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_case_trap -> change string to lower case
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      implicit none
      
      character*20 name
      integer i,k

      do i=1,20
         k=ichar(name(i:i))
         if(k.ge.65.and.k.le.90) then  !upper case A-Z
            k=ichar(name(i:i))+32
            name(i:i)=char(k)
         endif
      enddo

      return
      end

c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_blockread -> read a LHA line and return parameter name (evntually found in 
c ++ a ref file) and value
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine LHA_blockread(blockname,buff,par,val,found)

      implicit none
      character*132 buff,buffer,curr_ref,curr_buff
      character*20 blockname,val,par,temp,first_ref,first_line
      logical fopened
      integer ref_file
      logical islast,isnum,found
      character*20 temp_val


c     *********************************************************************
c     Try to find a correspondance in ident_card
c
      ref_file = 20
      call LHA_open_file(ref_file,'ident_card.dat',fopened)
      if(.not. fopened) goto 99 ! If the file does not exist -> no matter, use default!
        
      islast=.false.
      found=.false.
      do while(.not. found)!run over reference file
      

        ! read a line
        read(ref_file,'(a132)',end=98,err=98) buffer
        
        ! Seek a corresponding blockname
        call LHA_firststring(temp,buffer)
        call LHA_case_trap(temp)
        
        if(temp .eq. blockname) then
             ! Seek for a corresponding LHA code
             curr_ref=buffer
             curr_buff=buff
             first_ref=''
             first_line=''
             
             do while((.not. islast).and.(first_ref .eq. first_line))
                 call LHA_firststring(first_ref,curr_ref)
                 call LHA_firststring(first_line,curr_buff)
                 call LHA_islatin(first_ref(1:1),islast)
                 if (islast) then
                   par=first_ref
                   val=first_line ! If found set param name & value
                   found=.true.
                 end if
             end do
        end if
                     
      end do
98    close(ref_file)
99    return    
      end


c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c ++
c ++ LHA_loadcard -> Open a LHA file and load all model param in a table
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine LHA_loadcard(param_name,npara,param,value)

      implicit none

      integer maxpara
      parameter (maxpara=1000)
      character*20 param(maxpara),value(maxpara),val,par
      character*20 blockname
      integer npara
      logical fopened,found
      integer iunit,GL,logfile
      character*20 ctemp
      character*132 buff
      character*20 tag
      character*132 temp
      character*(*) param_name
      data iunit/21/
      data logfile/22/

      logical WriteParamLog
      common/IOcontrol/WriteParamLog

      GL=0
      npara=1

      param(1)=' '
      value(1)=' '

      ! Try to open param-card file
      call LHA_open_file(iunit,param_name,fopened)
      if(.not.fopened) then
         write(*,*) 'Error: Could not open file',param_name
         write(*,*) 'Exiting'
         stop
      endif
      
      ! Try to open log file
      if (WriteParamLog) then
        open (unit = logfile, file = "param.log")
      endif
      
      ! Scan the data file
      do while(.true.)  
      
         read(iunit,'(a132)',end=99,err=99) buff
         
         if(buff .ne. '' .and. buff(1:1) .ne.'#') then ! Skip comments and empty lines

             tag=buff(1:5)
             call LHA_case_trap(tag) ! Select decay/block tag
             if(tag .eq. 'block') then ! If we are in a block, get the blockname
                 temp=buff(7:132)
                 call LHA_firststring(blockname,temp)
                 call LHA_case_trap(blockname)
             else if (tag .eq. 'decay') then ! If we are in a decay, directly try to get back the correct name/value pair
                 blockname='decay'
                 temp=buff(7:132)
                 call LHA_blockread(blockname,temp,par,val,found)
                 if(found) GL=1
             else if ((tag .eq. 'qnumbers').or.(blockname.eq.'')) then! if qnumbers or empty tag do nothing
                 blockname=''
             else ! If we are in valid block, try to get back a name/value pair
                 call LHA_blockread(blockname,buff,par,val,found)
                 if(found) GL=1
             end if

             !if LHA_blockread has been called, record name and value

             if(GL .eq. 1) then
                  value(npara)=val
                  ctemp=par
                  call LHA_case_trap(ctemp)
                  param(npara)=ctemp
                  npara=npara+1
                  GL=0
                  if (WriteParamLog) then
                    write (logfile,*) 'Parameter ',ctemp,
     &                                  ' has been read with value ',val
                  endif
             endif

         endif
      enddo
      
      npara=npara-1
c      close(iunit)
      if (WriteParamLog) then
        close(logfile)
      endif

 99   return
 
      end



      subroutine LHA_get_real(npara,param,value,name,var,def_value_num)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value
c----------------------------------------------------------------------------------
      implicit none

c
c     parameters
c
      integer maxpara
      parameter (maxpara=1000)
c
c     arguments
c
      integer npara
      character*20 param(maxpara),value(maxpara)
      character*(*)  name
      real*8 var,def_value_num
      character*20 c_param,c_name,ctemp
      character*19 def_value
c
c     local
c
      logical found
      integer i
c
c     start
c
      i=1
      found=.false.
      do while(.not.found.and.i.le.npara)
         ctemp=param(i)
         call LHA_firststring(c_param,ctemp)
         ctemp=name
         call LHA_firststring(c_name,ctemp)
         call LHA_case_trap(c_name)
         call LHA_case_trap(c_param)
         found = (c_param .eq. c_name)
         if (found) then
             read(value(i),*) var
         end if
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",
     &def_value_num
         var=def_value_num
      endif
      return

      end
c

      subroutine LHA_open_file(lun,filename,fopened)
c***********************************************************************
c     opens file input-card.dat in current directory or above
c***********************************************************************
      implicit none
c
c     Arguments
c
      integer lun
      logical fopened
      character*(*) filename
      character*90  tempname
      integer fine
      integer dirup,i

c-----
c     Begin Code
c-----
c
c     first check that we will end in the main directory
c
      open(unit=lun,file=filename,status='old',ERR=20)
c      write(*,*) 'read model file ',filename
      fopened=.true.
      return
      
20    tempname=filename
      fine=index(tempname,' ')
      if(fine.eq.0) fine=len(tempname)
      tempname=tempname(1:fine)
c
c     if I have to read a card
c
      if(index(filename,"_card").gt.0) then
        tempname='./Cards/'//tempname
      endif

      fopened=.false.
      do i=0,5
        open(unit=lun,file=tempname,status='old',ERR=30)
        fopened=.true.
c        write(*,*) 'read model file ',tempname
        exit
30      tempname='../'//tempname
        if (i.eq.5)then
           write(*,*) 'Warning: file ',filename,
     &                           ' not found in the parent directories!(lha_read)'
           stop
        endif
      enddo

      return
      end

