c----------------------------------------------------------------------
C  couplings.f 
c----------------------------------------------------------------------
c  This files takes the inputs of the standard model from a Les Houches 
c  file (param_card.dat) and calculates all the couplings that end up
c  in the Feynman rules, i.e., in the HELAS routines.
c   
c  With the convention of the New Web Generation MadEvent in this
c  part no non-trivial computation is made. The SM model secondary
c  parameters have been all calculated by the SM calculator, SMCalc
c  which produced param_card.dat.
c
c  The only exception to the above rule is for alpha_S. In the case
c  where a pdf is used in the initial state, then the values as(MZ)
c  set in the les houches card is superseeded.
c  Scales and pdf's are set in the run_card.dat.
c
c This file contains the following routines:
c 
c- madgraph original call to set the parameters
c- lh_readin is called from here.
c  It talks to the lh_readin through the common block values.
c      subroutine setpara
c
c-routine to read the LH parameters in
c      subroutine lh_readin
c
c-to set
c      subroutine set_it(npara,ivalue,value,name,id,
c      subroutine case_trap(string,length)
c      subroutine no_spaces(buff,nchars)
c---------------------------------------------------------------------- 


      subroutine setpara(param_name,readlha)
c***********************************************************************
c This subroutine sets up the HELAS couplings of the STANDARD MODEL.
c***********************************************************************
      implicit none
c
c local
c
      character*(*) param_name
      logical readlha
      integer i
      real*8 dum
c
c     common file with the couplings
c
      include 'coupl.inc'
c
c     local
c
      double precision axialf,scalarf
      double precision  v,tau,series_t,series_p,tau_t,tau_w,series_a
      double precision  ee, ee2, ez, ey, sw, cw, sc2, sin2w, wm
      double precision  gwne, gwud, lambda, lam4, xt, rew, rqcd
      double precision  alphas, alfa, alfaw, mfrun
      external          alphas, alfa, alfaw, mfrun
      logical firsttime
      data firsttime /.true./

c
c     Common to lh_readin and printout
c
      double precision  alpha, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,gfermi,alfas,   
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
c
c constants
c
      double complex  ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
      double precision  Zero, One, Two, Three, Four, Half, Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
      parameter( Three = 3.0d0, Four = 4.0d0, Half = 0.5d0 )
      parameter( Rt2   = 1.414213562d0 )
      double precision  Pi, Fourpi
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Fourpi = Four * Pi )

      save scalarf,axialf,series_t,series_p,v
c------------------------------------------
c Start calculating the couplings for HELAS
c------------------------------------------
c
      if(readlha) then 
         call lh_readin(param_name)
         G = DSQRT(4d0*PI*ALFAS) ! use setting of the param_card.dat @ NLO
c     
c     Strong coupling
c
c     As a rule we first check if a pdf has been chosen in the    
c     run_card.dat (which has been already read at this stage).
c     If there pdfs in the initial state, then the alpha_s(MZ) used
c     is set to the corresponding value.  

c
c auxiliary local values
c
      wm = sqrt(zmass**2/Two+
     $     sqrt(zmass**4/Four-Pi/Rt2*alpha/gfermi*zmass**2))
      sin2w  = One-(wm/zmass)**2
      cw  = sqrt( One - sin2w )
      ee2 = alpha * Fourpi
      sw  = sqrt( sin2w )
      ee  = sqrt( ee2 )
      ez  = ee/(sw*cw)
      ey  = ee*(sw/cw)
      sc2 = sin2w*( One - sin2w )
      v   = Two*wm*sw/ee   ! the wmass is used to calculate v
      lambda = hmass**2 / (Two * v**2)
c
c vector boson couplings
c
      gw   = ee/sw
      gwwa = ee
      gwwz = ee*cw/sw
c
c gauge & higgs boson coupling constants
c
      gwwh  = dcmplx( ee2/sin2w*Half*v, Zero )
      gzzh  = dcmplx( ee2/sc2*Half*v, Zero )
      ghhh  = dcmplx( -hmass**2/v*Three, Zero )
      gwwhh = dcmplx( ee2/sin2w*Half, Zero )
      gzzhh = dcmplx( ee2/sc2*Half, Zero)
      ghhhh = ghhh/v
c
c fermion-fermion-vector couplings
c
      gal(1) = dcmplx(  ee          , Zero )
      gal(2) = dcmplx(  ee          , Zero )
      gau(1) = dcmplx( -ee*Two/Three, Zero )
      gau(2) = dcmplx( -ee*Two/Three, Zero )
      gad(1) = dcmplx(  ee/Three    , Zero )
      gad(2) = dcmplx(  ee/Three    , Zero )

      gwf(1) = dcmplx( -ee/sqrt(Two*sin2w), Zero )
      gwf(2) = dcmplx(  Zero              , Zero )

      gzn(1) = dcmplx( -ez*Half                     , Zero )
      gzn(2) = dcmplx(  Zero                        , Zero )
      gzl(1) = dcmplx( -ez*(-Half + sin2w)          , Zero )
      gzl(2) = dcmplx( -ey                          , Zero )
      gzu(1) = dcmplx( -ez*( Half - sin2w*Two/Three), Zero )
      gzu(2) = dcmplx(  ey*Two/Three                , Zero )
      gzd(1) = dcmplx( -ez*(-Half + sin2w/Three)    , Zero )
      gzd(2) = dcmplx( -ey/Three                    , Zero )
c
c fermion-fermion-Higgs couplings (complex) hff(2)
c and also pseudo-scalar.
c
c NOTE: the running mass is evaluated @ the same order 
c nloop of alpha_s set by the PDF choice
c 

      if(mtMS.gt.1d0) then
         ghtop(1)  = dcmplx( -mtMS/v, Zero )
         gh3top(1) = dcmplx( Zero, -mtMS/v )
      else
         ghtop(1)  = dcmplx( Zero,Zero)
         gh3top(1) = dcmplx( Zero,Zero)
      endif
      ghtop(2)  = ghtop(1)
      gh3top(2) = -gh3top(1)

      if(mbMS.gt.1d0) then
         ghbot(1)  = dcmplx( -mbMS/v, Zero )
         gh3bot(1) = dcmplx( Zero, -mbMS/v )
      else
         ghbot(1)  = dcmplx( Zero, Zero )
         gh3bot(1) = dcmplx( Zero, Zero )
      endif
      ghbot(2)  = ghbot(1)
      gh3bot(2) = -gh3bot(1)
      
      if(mcMS.gt.1d0) then
         ghcha(1)  = dcmplx( -mcMS/v, Zero )
         gh3cha(1) = dcmplx( Zero, -mcMS/v )
      else
         ghcha(1)  = dcmplx( Zero, Zero )
         gh3cha(1) = dcmplx( Zero, Zero )
      endif
      ghcha(2)  = ghcha(1)
      gh3cha(2) = -gh3cha(1)

      ghtau(1)  = dcmplx( -mtaMS/v, Zero )
      ghtau(2)  = ghtau(1)
      gh3tau(1) = dcmplx( Zero, -mtaMS/v )
      gh3tau(2) = -gh3tau(1)
c
c     CKM matrix: 
c     symmetric 3x3 matrix, Vud=Vcs, Vus=Vcd Vcb=Vub=0
c
c     >>>>>>>>>>>>>>>***** NOTE****<<<<<<<<<<<<<<<<<<<<<<<<<
c     these couplings matter only when interaction_CKM.dat
c     is used to generate all the diagrams with off-diagonal
c     couplings. The default of MadEvent is a diagonal
c     CKM matrix.

	  Vus=DSQRT(1d0-Vud**2)
      do i=1,2
         gwfc(i) = gwf(i)*Vud
         gwfs(i) = gwf(i)*Vus
         gwfm(i) =-gwf(i)*Vus
      enddo

c---------------------------------------------------------
c Set Photon Width to Zero, used by symmetry optimization
c---------------------------------------------------------

      awidth = 0d0
            
c---------------------------------
c Higgs effective couplings (couplings of Higgs's directly to gluons)
c---------------------------------

c Coupling to gluons

c Higgs coupling:

        tau = hmass**2/(4d0*tmass**2)
	series_t = 1d0 + tau*7d0/30d0 + tau**2*2d0/21d0 + tau**3*26d0/525d0
	series_p = 1d0 + tau/3d0      + tau**2*8d0/45d0 + tau**3*4d0/35d0

        scalarf = 1d0
        axialf  = 0d0

c Coupling to photons

	tau_t = hmass**2/(4d0*tmass**2)
	tau_w = hmass**2/(4d0*wm**2)
        series_a = 1d0 + tau_w*66d0/235d0 +tau_w**2*228d0/1645d0
     &             +tau_w**3*696d0/8225d0+tau_w**4*5248d0/90475d0
     &             +tau_w**5*1280d0/29939d0+tau_w**6*54528d0/1646645d0
     &             -tau_t*56d0/705d0-tau_t**2*32d0/987d0

c Higgs coupling:
        if (hmass.le.150)then	
       	   gphph(1) = dcmplx( -ee2/4d0/PI/(PI*V)*(47d0/18d0)*series_a, Zero)
        else
	   gphph(1) = dcmplx( Zero, Zero )
           if (firsttime) then
              write (*,*) 'Higgs mass larger than 150 GeV,'
              write (*,*) '    coupling to photons set to zero.'
              firsttime=.false.
           endif
	endif
	gphph(2) = dcmplx( Zero, Zero )
      endif

      GG(1) = -G
      GG(2) = -G     

      gh(1) = dcmplx( scalarf*g**2/4d0/PI/(3d0*PI*V)*series_t, Zero)
      gh(2) = dcmplx( axialf *g**2/4d0/PI/(2d0*PI*V)*series_p, Zero)

c Pseudo-scalar Higgs coupling:
      ga(1) = dcmplx( Zero                  , Zero)
      ga(2) = dcmplx( g**2/4d0/PI/(2d0*PI*V)*series_p, Zero)

c   Couplings for the 4-vertices VVVS
      gh4(1) = G*gh(1)
      gh4(2) = G*gh(2)
      ga4(1) = G*ga(1)
      ga4(2) = G*ga(2)
	
c----------------------------
c end subroutine coupsm
c----------------------------

      return
      end
 
      subroutine lh_readin(param_name)
c----------------------------------------------------------------------
c Read the parameters from the lh file
c
c 1. Input values for the EW sector 
c 2. Higgs mass and width
c 3. Fermion masses (pole and MSbar) and widths
c---------------------------------------------------------------------- 
      implicit none
c
c     parameters
c
      integer maxpara
      parameter (maxpara=100)
      double precision  Two, Four, Rt2, Pi
      parameter( Two = 2.0d0, Four = 4.0d0 )
      parameter( Rt2   = 1.414213562d0 )
      parameter( Pi = 3.14159265358979323846d0 )
c
c     local
c
      character*(*) param_name
      integer npara,l1,l2,id
	  character*132 buff
	  real*8 real_value 
	  real*8 value(maxpara)
	  integer ivalue(maxpara),n
	  character*20 name(maxpara),bn
	  logical block_found,done,fopened	  
      integer iunit,i,name_length,idum
c
c	block info
c      
      character*20 block_name
c
c   Common
c
	  include 'coupl.inc'
c
c     Common to lh_readin and printout
c
      double precision  alpha, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,gfermi,alfas,   
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
c
c----------
c     start
c----------
c
c     open file
c
      iunit=14
      call open_file_mdl(iunit,param_name,fopened)
	   done=.false.
 
       n=0
	   do while(.not.done)
	   block_found=.false.
c
c looks for the blocks or for decay
c
      do while(.not.block_found)
       read(iunit,'(a132)',end=99,err=99) buff
c--	change to lower case
	   call case_trap(buff,20)
       if(buff(1:5).eq.'block') then
         if(index(buff,"#").ne.0) l1=index(buff,"#")-1 
         block_name=buff(6:min(l1,26)) 
         call no_spaces(block_name,name_length)
c        write(*,*) block_name(1:name_length)
         block_found=.true.        
	   elseif(buff(1:5).eq.'decay') then
               n=n+1
               l1=60
               if(index(buff,"#").ne.0) l1=index(buff,"#")-1 ! ignore comments       
               read(buff(6:l1),*) ivalue(n),value(n)
               name(n)="decay" 
       endif		   
      end do
c
c      
      if(block_found) then
 	  do while(.true.)
          read(iunit,'(a132)',end=99,err=99) buff
     	  call case_trap(buff,20)
          if(buff(1:1).eq.'b'.or.buff(1:1).eq.'d') then
          	backspace iunit
          	exit
          endif	         	
            if(buff(1:1).ne.'#') then  !if it not a comment
              n=n+1	       
              l1=60
              if(index(buff,"#").ne.0) l1=index(buff,"#")-1 ! ignore comments       
c
c  WARNING:... not all blocks have the same sintax!! You need to change it
c  depending on the block you are reading
c
             if(block_name(1:5).eq."mgckm") then
                read(buff(1:l1),*) ivalue(n),idum,value(n)
              else
                read(buff(1:l1),*) ivalue(n),value(n)
              endif  
              name(n)=block_name(1:name_length)
c              write(*,"(1x,i2,2x,e16.8,1x,a)") 
c     &        ivalue(n),value(n),name(n)
           	  endif
      end do ! do while in the block
      else
	  done=.true.
	  endif
	  end do ! do while the entire file
	  

99	continue      
	
	   bn="sminputs"
       call set_it(n,ivalue,value,name,1,bn,alpha,128.9d0)
       alpha=1d0/alpha
       call set_it(n,ivalue,value,name,2,bn,gfermi,0.1166d-4)
       call set_it(n,ivalue,value,name,3,bn,alfas,0.119d0)
       call set_it(n,ivalue,value,name,4,bn,zmass,91.188d0)
       call set_it(n,ivalue,value,name,6,bn,tmass,174.3d0)
       call set_it(n,ivalue,value,name,7,bn,lmass,1.777d0)
       bn="mgyukawa"
       call set_it(n,ivalue,value,name,4,bn,mcMS,1.25d0)
       call set_it(n,ivalue,value,name,5,bn,mbMS,4.2d0)
       call set_it(n,ivalue,value,name,6,bn,mtMS,174d0)
       call set_it(n,ivalue,value,name,15,bn,mtaMS,1.777d0)
       bn="mgckm"
       call set_it(n,ivalue,value,name,1,bn,vud,1d0)
       bn="mass"
       call set_it(n,ivalue,value,name,4,bn,cmass,1.4d0)
       call set_it(n,ivalue,value,name,5,bn,bmass,4.7d0)
       call set_it(n,ivalue,value,name,6,bn,tmass,tmass*1d0)
       call set_it(n,ivalue,value,name,15,bn,lmass,lmass*1d0)
       call set_it(n,ivalue,value,name,25,bn,hmass,120d0)
       call set_it(n,ivalue,value,name,23,bn,zmass,zmass*1d0)
       call set_it(n,ivalue,value,name,24,bn,wmass,sqrt(zmass**2/Two+
     $     sqrt(zmass**4/Four-Pi/Rt2*alpha/gfermi*zmass**2)))
       bn="decay"
       call set_it(n,ivalue,value,name,6,bn,twidth,1.5083d0)
       call set_it(n,ivalue,value,name,25,bn,hwidth,0.0037d0)
       call set_it(n,ivalue,value,name,23,bn,zwidth,2.441d0)
       call set_it(n,ivalue,value,name,24,bn,wwidth,2.0476d0)
     
      return
      end

      
      subroutine set_it(npara,ivalue,value,name,id,
     &                  block_name,var,def_value)
c----------------------------------------------------------------------------------
c     finds the parameter value  in block_name and associate var to it.
c     If it is not found a default is given.
c----------------------------------------------------------------------------------
      implicit none

c
c     parameters
c
      integer maxpara
      parameter (maxpara=100)
c
c     arguments
c
      integer npara,ivalue(maxpara),id
      character*20  block_name,name(maxpara)
      real*8 var,def_value,value(maxpara)
c
c     local
c
      logical found
      integer i
c
c     start
c
	  found=.false.
      do i=1,npara
         found = (id.eq.ivalue(i)).and.(name(i).eq.block_name)
 	               if(found) then
         	var=value(i)
            exit
          endif	
      enddo
      
      if (.not.found) then
c         write (*,*) "Warning: parameter not found"
c         write (*,*) "         setting it to default value ",def_value
         var=def_value
      endif
      return

      end
      
      
      subroutine case_trap(string,length)
c**********************************************************    
c change string to lowercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) string
      integer length
c
c     LOCAL
c
      integer i,k

      do i=1,length
         k=ichar(string(i:i))
         if(k.ge.65.and.k.le.90) then  !upper case A-Z
            k=ichar(string(i:i))+32   
            string(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine no_spaces(buff,nchars)
c**********************************************************************
c     Given buff a buffer of words separated by spaces
c     returns it where all space are moved to the right
c     returns also the length of the single word.
c     maxlength is the length of the buffer
c**********************************************************************
      implicit none
c
c     Constants
c
      integer    maxline
      parameter (maxline=20)
      character*1 null
      parameter  (null=' ')
c
c     Arguments
c
      character*(maxline) buff
      integer nchars,maxlength
c
c     Local
c
      integer i,j
      character*(maxline) temp
c-----
c  Begin Code
c-----
      nchars=0
c      write (*,*) "buff=",buff(1:maxlength)
      do i=1,maxline
         if(buff(i:i).ne.null) then
            nchars=nchars+1
            temp(nchars:nchars)=buff(i:i)
         endif
c         write(*,*) i,":",buff(1:maxlength),":",temp(1:nchars),":"
      enddo
      buff=temp      
      end

      subroutine open_file_mdl(lun,filename,fopened)
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
c  Begin Code
c-----
c
c     first check that we will end in the main directory
c
      tempname=filename
      fine=index(tempname,' ')
      if(fine.eq.0) fine=len(tempname)
      tempname=tempname(1:fine)
c
c         if I have to read a card
c
      if(index(filename,"_card").gt.0) then
         tempname='./Cards/'//tempname
      endif


      fopened=.false.
      do i=0,5
         open(unit=lun,file=tempname,status='old',ERR=30)
         fopened=.true.
         write(*,*) 'read model file',tempname
         exit
 30      tempname='../'//tempname
         if (i.eq.5)then
            write(*,*) 'Warning: file ',tempname,' is not correct'
            stop
         endif
      enddo


      return
      end
 
