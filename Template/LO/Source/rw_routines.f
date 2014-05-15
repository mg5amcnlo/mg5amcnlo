      subroutine load_gridpack_para(npara,param,value)
c----------------------------------------------------------------------
c Read the params from the run_card.dat file
c----------------------------------------------------------------------
      implicit none
c
c     arguments
c
      character*20 param(*),value(*)
      integer npara
c
c     local
c
      logical fopened,done
      integer iunit
      character*20 ctemp
      integer k,i,l1,l2,iproc
      character*132 buff
      data iunit/21/
c
c     global
c
      integer ngroup
      common/to_group/ngroup
c
c----------
c     start
c----------
      npara=0
      param(1)=' '
      value(1)=' '
c
c     open file
c
      call open_file(iunit,'grid_card.dat',fopened)
      if(fopened) then
c
c     first look for process-specific parameters
c
      done=.false.
      do while(.not.done)
         read(iunit,'(a132)',end=30,err=30) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").gt.0) then
            l1=index(buff,"@")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            read(buff(l1+1:l2),*,err=21) iproc
            if(iproc.ne.ngroup) cycle

            l1=index(buff,"=")
            l2=index(buff,"@")
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c
 21          cycle
         endif
       enddo
 30   rewind(iunit)
c
c     read in values
c
      done=.false.
      do while(.not.done)
         read(iunit,'(a132)',end=99,err=99) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").le.0) then
            l1=index(buff,"=")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
c             write (*,*) param(npara),l1,l2
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c             write(*,*) "New param:",param(npara)," = ", value(npara)
c
         endif
      enddo
 99   close(iunit)
      endif

      return
      end


      subroutine load_para(npara,param,value)
c----------------------------------------------------------------------
c Read the params from the run_card.dat file
c---------------------------------------------------------------------- 
      implicit none
c
c     arguments
c     
      character*20 param(*),value(*)
      integer npara
c
c     local
c
      logical fopened,done
      integer iunit
      character*20 ctemp
      integer k,i,l1,l2,iproc
      character*132 buff
      data iunit/21/
c
c     global
c
      integer ngroup
      common/to_group/ngroup
c
c----------
c     start
c----------
c
c     read the run_card.dat
c
      npara=0
      param(1)=' '
      value(1)=' '
c
c     open file
c
      call open_file(iunit,'run_card.dat',fopened)
      if(.not.fopened) then
         write(*,*) 'Error: File run_card.dat not found'
         stop
      else
c
c     first look for process-specific parameters
c
      done=.false.
      do while(.not.done)  
         read(iunit,'(a132)',end=20,err=20) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").gt.0) then
            l1=index(buff,"@")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            read(buff(l1+1:l2),*,err=11) iproc
            if(iproc.ne.ngroup) cycle

            l1=index(buff,"=")
            l2=index(buff,"@")
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c
 11          cycle
         endif
      enddo
 20   rewind(iunit)
c
c     read in values
c
      done=.false.
      do while(.not.done)  
         read(iunit,'(a132)',end=96,err=96) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").le.0) then
            l1=index(buff,"=")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c
         endif
      enddo
 96   close(iunit)
      endif
c
c     open file
c
c
c     tjs modified 11-16-07 to include grid_card.dat
c
      call open_file(iunit,'grid_card.dat',fopened)
      if(fopened) then
c
c     first look for process-specific parameters
c
      done=.false.
      do while(.not.done)  
         read(iunit,'(a132)',end=30,err=30) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").gt.0) then
            l1=index(buff,"@")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            read(buff(l1+1:l2),*,err=21) iproc
            if(iproc.ne.ngroup) cycle

            l1=index(buff,"=")
            l2=index(buff,"@")
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c
 21          cycle
         endif
       enddo
 30   rewind(iunit)
c
c     read in values
c
      done=.false.
      do while(.not.done)  
         read(iunit,'(a132)',end=99,err=99) buff
         if(buff(1:1).ne.'#' .and. index(buff,"=").gt.0
     $        .and. index(buff,"@").le.0) then
            l1=index(buff,"=")
            l2=index(buff,"!")
            if(l2.eq.0) l2=l1+20  !maybe there is no comment...
            if(l2-l1.lt.0) cycle
            npara=npara+1
c
             value(npara)=buff(1:l1-1)
             ctemp=value(npara)
             call case_trap2(ctemp)
             value(npara)=ctemp
c
             param(npara)=" "//buff(l1+1:l2-1)
c             write (*,*) param(npara),l1,l2
             ctemp=param(npara)
             call case_trap2(ctemp)
             param(npara)=ctemp
c             write(*,*) "New param:",param(npara)," = ", value(npara)
c
         endif
      enddo
 99   close(iunit)
      endif

      return
      end



      subroutine get_real(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none

c
c     arguments
c
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      real*8 var,def_value
      character*20 c_param,c_name
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
          call firststring(c_param,param(i))
          call firststring(c_name,name)
         found = (c_param .eq. c_name)
         if (found) read(value(i),*) var
c         if (found) write (*,*) name,var
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
c

      subroutine get_integer(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none
c
c     arguments
c
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      integer var,def_value
      character*20 c_param,c_name
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
         call firststring(c_param,param(i))
          call firststring(c_name,name)
         found = (c_param .eq. c_name)
         if (found) read(value(i),*) var
c         if (found) write (*,*) name,var
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
c
      subroutine get_int8(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none
c
c     arguments
c
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      integer def_value
      integer*8 var
      character*20 c_param,c_name
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
         call firststring(c_param,param(i))
          call firststring(c_name,name)
         found = (c_param .eq. c_name)
         if (found) read(value(i),*) var
c         if (found) write (*,*) name,var
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
c
      subroutine get_string(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none

c
c     arguments
c
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      character*(*)  var,def_value
      character*20 c_param,c_name
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
         call firststring(c_param,param(i))
          call firststring(c_name,name)
         found = (c_param .eq. c_name)
         if (found) read(value(i),*) var
c         if (found) write (*,*) name,var
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
c
      subroutine get_logical(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none

c
c     arguments
c
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      logical  var,def_value
      character*20 c_param,c_name
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
         call firststring(c_param,param(i))
          call firststring(c_name,name)
         found = (c_param .eq. c_name)
         if (found) read(value(i),*) var
c         if (found) write (*,*) name,var
         i=i+1
      enddo
      if (.not.found) then
         write (*,*) "Warning: parameter ",name," not found"
         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
c



      subroutine case_trap2(name)
c**********************************************************    
c change the string to lowercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*20 name
c
c     LOCAL
c
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
c ++ firststring -> return the first "word" of string 
c ++ & remove whitespaces around
c ++ Needed to correct a bug in "get_" routines
c ++ Michel Herquet - CP3 - 05-04-2006
c ++
c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine firststring(first,string)

      implicit none
      character*(*) string
      character*20 first
      character*20 temp

      temp=string
      do while(temp(1:1) .eq. ' ') 
	temp=temp(2:len(temp))
      end do
      first=temp(1:index(temp,' ')-1)

      end
