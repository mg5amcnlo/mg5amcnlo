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


      subroutine setpara
c***********************************************************************
c This subroutine sets up the HELAS couplings of the STANDARD MODEL.
c***********************************************************************
      implicit none
c
c local
c
      integer i
      real*8 dum
c
c     common file with the couplings
c
      include 'coupl.inc'
      include 'decay.inc'
      include 'calc_values.inc'
c
c     local
c
      double precision  v
      double precision  ee, ee2, ez, ey, sw, cw, sc2
      double precision  gwne, gwud, lambda, lam4, xt, rew, rqcd
      double precision  alphas, alfa, alfaw, mfrun
      external          alphas, alfa, alfaw, mfrun
c
c     Common to lh_readin and printout
c
      double precision  alpha, sin2w, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,sin2w,gfermi,alfas,   
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
c
c constants
c
      double complex  ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
c     alfas and run
c
      include 'alfas.inc'
      include 'run.inc'
c
c     Auxiliary functions
c
      REAL*8 XX,Y,Z
      REAL*8 HFF,BETA,HEAVY

      BETA(XX) = DSQRT(One-Four*XX)
      HFF(XX,Y)  = One/(Two*Four*PI)*XX*(BETA(Y))**3
      HEAVY(XX,Y,Z)= ( One - Half*(y**2+z**2) - Half*(y**2-z**2)**2
     &                 + Three*y*z*(XX**2 - One)/(XX**2 + One)        )
     &               * sqrt( (One-y**2-z**2)**2 - Four * y**2 * z**2 )
c
c------------------------------------------
c Start calculating the couplings for HELAS
c------------------------------------------
c
      call lh_readin
c     
c     Strong coupling
c
c     As a rule we first check if a pdf has been chosen in the    
c     run_card.dat (which has been already read at this stage).
c     If there pdfs in the initial state, then the alpha_s(MZ) used
c     is set to the corresponding value.  

      if(scale.le.1d0)    scale=zmass   
   
      if(lpp(1).ne.0.or.lpp(2).ne.0) then    
          if(asmz .le.0.01d0) asmz =0.118d0   
      else 
	      asmz=alfas   !value read from the param_card.dat
c          nloop=2 
      endif      
      if(nloop.eq.0)      nloop=1   

      G = DSQRT(4d0*PI*ALPHAS(SCALE)) ! use setting of the param_card.dat @ NLO
      GG(1) = -G
      GG(2) = -G     
c
c auxiliary local values
c
      cw  = sqrt( One - sin2w )
      ee2 = alpha * Fourpi
      sw  = sqrt( sin2w )
      ee  = sqrt( ee2 )
      ez  = ee/(sw*cw)
      ey  = ee*(sw/cw)
      sc2 = sin2w*( One - sin2w )
      v   = Two*wmass*sw/ee   ! the wmass is used to calculate v
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
c
c NOTE: the running mass is evaluated @ the same order 
c nloop of alpha_s set by the PDF choice
c 

      if(mtMS.gt.1d0) then
         ghtop(1) = dcmplx( -mtMS/v, Zero )
      else
         ghtop(1) = dcmplx( Zero,Zero)
      endif
      ghtop(2) = ghtop(1)

      if(mbMS.gt.1d0) then
         ghbot(1) = dcmplx( -mbMS/v, Zero )
      else
         ghbot(1) = dcmplx( Zero, Zero )
      endif
      ghbot(2) = ghbot(1)
      
      if(mcMS.gt.1d0) then
         ghcha(1) = dcmplx( -mcMS/v, Zero )
      else
         ghcha(1) = dcmplx( Zero, Zero )
      endif
      ghcha(2) = ghcha(1)

      ghtau(1) = dcmplx( -mtaMS/v, Zero )
      ghtau(2) = ghtau(1)
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
c     
c Z boson partial widths
c     
      decz = zmass / ( 24.0d0 * Pi )
      w_z_nn = decz * ( gzn(1)**2 + gzn(2)**2 )
      w_z_ll = decz * ( gzl(1)**2 + gzl(2)**2 )
      decz = decz * Three 
      w_z_uu = decz * ( gzu(1)**2 + gzu(2)**2 )
      w_z_dd = decz * ( gzd(1)**2 + gzd(2)**2 )
      dum = dble( (gzl(2)+gzl(1))/(gzl(2)-gzl(1)) )
      w_z_tau = w_z_ll * heavy( dum, lmass/zmass, lmass/zmass )
      dum = dble( (gzu(2)+gzu(1))/(gzu(2)-gzu(1)) )
      w_z_cc = w_z_uu *  heavy( dum, cmass/zmass, cmass/zmass )
      dum = dble( (gzd(2)+gzd(1))/(gzd(2)-gzd(1)) )
      w_z_bb = w_z_dd *  heavy( dum, bmass/zmass, bmass/zmass )
      
         zwidth =   Three*w_z_nn + Two*w_z_ll + w_z_tau
     &        + Two*w_z_dd + w_z_uu + w_z_cc + w_z_bb

c
c W boson partial widths
c     
      decw = wmass / ( 24.0d0 * Pi )
      w_w_nl = decw * ( gwf(1)**2 + gwf(2)**2 )
      dum = dble( (gwf(2)+gwf(1))/(gwf(2)-gwf(1)) )
      w_w_tau = w_w_nl * heavy( dum, lmass/wmass, Zero )
      w_w_ud = w_w_nl * Three 
      w_w_cs = w_w_ud * heavy( dum, cmass/wmass, Zero )
      
      wwidth = Two*w_w_nl + w_w_tau + w_w_ud + w_w_cs

c
c top quark width
c
      call topwid(tmass,wmass,bmass,wwidth,gw,twidth) 

c
c tau width
c
      lwidth  = 2.36d-12 !tau width, PDG value
c
c     LO withds of the Higgs into tt~,bb~,tau tau~.
c
      if(hmass.gt.2d0*tmass) then
         w_h_tt =3d0*cdabs(ghtop(1)**2)*hff(hmass,(tmass/hmass)**2)
      else
         w_h_tt =0d0
      endif

      w_h_bb =3d0*cdabs(ghbot(1)**2)*hff(hmass,(bmass/hmass)**2)
      w_h_tau=cdabs(ghtau(1)**2)*hff(hmass,(lmass/hmass)**2)
      w_h_bb =3d0*cdabs(ghbot(1)**2)*hff(hmass,(bmass/hmass)**2)
      w_h_cc =3d0*cdabs(ghcha(1)**2)*hff(hmass,(cmass/hmass)**2)
      w_h_tau=cdabs(ghtau(1)**2)*hff(hmass,(lmass/hmass)**2)

      
      if(hmass.gt.2d0*wmass) then         
         w_h_ww=gfermi/8d0/pi/rt2*hmass**3*
     &        dsqrt(1-4d0*(wmass/hmass)**2)*
     &        (12d0*(wmass/hmass)**4 -4d0*(wmass/hmass)**2+1d0)         
         w_h_WLWL=gw**2/64d0/pi*hmass**3/wmass**2* !longitudinal W's
     &        dsqrt(1-4d0*(wmass/hmass)**2)*
     &        (1-2d0*(wmass/hmass)**2)**2 
      else
         w_h_ww=0d0
         w_h_WLWL=0d0
      endif
      
      if(hmass.gt.2d0*zmass) then
         w_h_zz=gfermi/16d0/pi/rt2*hmass**3*
     &        dsqrt(1-4d0*(zmass/hmass)**2)*
     &        (12d0*(zmass/hmass)**4 -4d0*(zmass/hmass)**2+1d0)
         w_h_ZLZL=gw**2/128d0/pi*hmass**3/wmass**2* !longitudinal Z's
     &        dsqrt(1-4d0*(zmass/hmass)**2)*
     &        (1-2d0*(zmass/hmass)**2)**2         
      else
         w_h_zz=0d0
         w_h_zLzL=0d0
      endif
            
c--------------------------
c start interface to HDECAY
c--------------------------
c do not change things here unless you exactly know what you are doing
c
      ihiggs = 0
      nnlo   = 0
      ipole  = 0

      ionsh   = 0
      ionwz   = 0
      iofsusy = 1

      hdals  = asmz
c-- do not set masses to zero here
      ams    = 0.190d0    !strange pole mass

      if(cmass.gt.0d0) then 
         amc    = cmass
      else
         amc    = 1.42d0        !charm   pole mass      
      endif

      if(bmass.gt.0d0) then 
         amb    = bmass
      else
         amb    = 4.7d0          !bottom   pole mass      
      endif

      if(tmass.gt.0d0) then 
         amt    = tmass
      else
         amt    = 174.3d0        !top pole mass      
      endif

      if(lmass.gt.0d0) then 
         almass    = lmass
      else
         almass    = 1.777d0     !tau  mass      
      endif

      ammuon = 0.105658389d0 !muon mass

      alph   = 137.0359895D0 !alpha_EM
      gf     = gfermi    

      if(wwidth.gt.0d0) then
         gamw   = wwidth
      else
         gamw=2.12d0
      endif
      
      if(zwidth.gt.0d0) then
         gamz   = zwidth
      else
         gamz=2.495d0
      endif
      
      amz    = zmass
      amw    = wmass

      hdmhbeg = hmass
      hdmhend = hmass

c 
c     this calculates the branching ratios of the Higgs
c     at the best of our knowledge 
c
      call hdecay     
      hwidth = SMWDTH

c------------------------
c end interface to HDECAY
c------------------------


c----------------------------
c end subroutine coupsm
c----------------------------


      return
      end
 

      subroutine lh_readin
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
      parameter (maxpara=1000)
c
c     local
c
      integer npara,l1,l2
      character*132 buff
      real*8 real_value 
      real*8 value(maxpara)
      integer ivalue(maxpara),n
      character*20 name(maxpara),bn,dumstring
      logical block_found,done,fopened	  
      integer i,name_length,idum

c
c	block info
c      
      character*20 block_name
c
c   Common
c
      include 'coupl.inc'
      include 'decay.inc'
c
c     Common to lh_readin and printout
c
      double precision  alpha, sin2w, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,sin2w,gfermi,alfas,   
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
c
c----------
c     start
c----------
c
        n=0
       rewind(lunp)
       done=.false.

       do while(.not.done)
          block_found=.false.
c
c looks for the blocks or for decay
c
      do while(.not.block_found)
       read(lunp,'(a132)',end=99,err=99) buff
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
               l1=30
               if(index(buff,"#").ne.0) l1=index(buff,"#")-1 ! ignore comments       
               read(buff(6:l1),*) ivalue(n),value(n)
               name(n)="decay" 
       endif		   
      end do
c
c     
 
      if(block_found) then
 	  do while(.true.)
          read(lunp,'(a132)',end=99,err=99) buff
     	  call case_trap(buff,20)
          if(buff(1:1).eq.'b'.or.buff(1:1).eq.'d') then
          	backspace lunp
          	exit
          endif	         	
            if(buff(1:1).ne.'#') then  !if it not a comment
              n=n+1	       
              l1=30
              if(index(buff,"#").ne.0) l1=index(buff,"#")-1 ! ignore comments       
c
c  WARNING:... not all blocks have the same sintax!! You need to change it
c  depending on the block you are reading
c
          
             if(block_name(1:5).eq."mgckm") then
                read(buff(1:l1),*) ivalue(n),idum,value(n)
              elseif (block_name(1:6).eq."spinfo".or.
     &block_name(1:6).eq."dcinfo") then
                read(buff(1:l1),*) ivalue(n),dumstring
              else
                read(buff(1:l1),*) ivalue(n),value(n)
              endif  
              name(n)=block_name(1:name_length)
c             write(*,"(1x,i2,2x,e16.8,1x,a)") 
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
	   bn="mgsmparam"
       call set_it(n,ivalue,value,name,1,bn,sin2w,0.2312d0)
       call set_it(n,ivalue,value,name,2,bn,wmass,80.419d0)
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
       call set_it(n,ivalue,value,name,24,bn,wmass,wmass*1d0)

c       bn="decay"
c       call set_it(n,ivalue,value,name,6,bn,twidth,1.5083d0)
c       call set_it(n,ivalue,value,name,25,bn,hwidth,0.0037d0)
c       call set_it(n,ivalue,value,name,23,bn,zwidth,2.441d0)
c       call set_it(n,ivalue,value,name,24,bn,wwidth,2.0476d0)

       write(*,*)
       write(*,*) ' >>> Widths in param_card.dat are ignored <<<'
       write(*,*)

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

      
      SUBROUTINE TOPWID(RMT,RMW,RMB,RGW,GW,RGT)
c*************************************************************************
c     THE TOTAL WEAK DECAY WIDTH OF THE TOP QUARK, INCLUDING
c     THE EFFECTS OF BOTTOM MASS AND, IF IGW=1,  A FINITE W WIDTH.
c     From James Stirling 6-10-94
c
c     RMT=TOP MASS
c     RMW=W   MASS
c     RMB=B   MASS
c     RGW=W   WIDTH
c     GW =WEAK COUPLING
c
c     RGT=TOP WIDTH
c
c*************************************************************************
      IMPLICIT COMPLEX*16(A-H,O-Z)
      REAL*8 RMT,RMB,RMW,XW,XB,RGW,RGT,GW
*
      PI=4.*DATAN(1.D0)
      XGW=dcmplx(GW/2d0/dsqrt(2d0))
*
      XB=RMB/RMT
      XW=RMW/RMT
*
      RM=XB**2
      OM=1.+RM-DCMPLX(RMW**2,RMW*RGW)/RMT**2
      Y1=OM+CDSQRT(OM*OM-4.*RM)
      Y0=OM-CDSQRT(OM*OM-4.*RM)
      Z1=2.
      Z0=2.*CDSQRT(RM)
*
      D0=(-Y0**8+3.*Y0**7*RM+3.*Y0**7-8.*Y0**6*RM-12.*Y0**5*RM**
     . 2-12.*Y0**5*RM+96.*Y0**4*RM**2-48.*Y0**3*RM**3-48.*Y0**3*
     . RM**2-128.*Y0**2*RM**3+192.*Y0*RM**4+192.*Y0*RM**3-256.*
     . RM**4)/(24.*Y0**4*(Y1-Y0))
      D1=(-Y1**8+3.*Y1**7*RM+3.*Y1**7-8.*Y1**6*RM-12.*Y1**5*RM**
     . 2-12.*Y1**5*RM+96.*Y1**4*RM**2-48.*Y1**3*RM**3-48.*Y1**3*
     . RM**2-128.*Y1**2*RM**3+192.*Y1*RM**4+192.*Y1*RM**3-256.*
     . RM**4)/(24.*Y1**4*(Y1-Y0))
      A4=(32.*RM**4*(Y1-Y0))/(3.*Y1*Y0*(Y1-Y0))
      A3=(8.*RM**3*(-3.*Y1**2*Y0*RM-3.*Y1**2*Y0+4.*Y1**2*RM+3.*
     . Y1*Y0**2*RM+3.*Y1*Y0**2-4.*Y0**2*RM))/(3.*Y1**2*Y0**2*(Y1
     . -Y0))
      A2=(8.*RM**3*(2.*Y1**3*Y0**2-3.*Y1**3*Y0*RM-3.*Y1**3*Y0+4.
     . *Y1**3*RM-2.*Y1**2*Y0**3+3.*Y1*Y0**3*RM+3.*Y1*Y0**3-4.*Y0
     . **3*RM))/(3.*Y1**3*Y0**3*(Y1-Y0))
      A1=(2.*RM**2*(3.*Y1**4*Y0**3*RM+3.*Y1**4*Y0**3+8.*Y1**4*Y0
     . **2*RM-12.*Y1**4*Y0*RM**2-12.*Y1**4*Y0*RM+16.*Y1**4*RM**2
     . -3.*Y1**3*Y0**4*RM-3.*Y1**3*Y0**4-8.*Y1**2*Y0**4*RM+12.*
     . Y1*Y0**4*RM**2+12.*Y1*Y0**4*RM-16.*Y0**4*RM**2))/(3.*Y1**
     . 4*Y0**4*(Y1-Y0))
      B0=(Y1**3-3.*Y1**2*RM-3.*Y1**2+8.*Y1*RM-Y0**3+3.*Y0**2*RM+
     . 3.*Y0**2-8.*Y0*RM)/(24.*(Y1-Y0))
      B1=(Y1+Y0-3.*RM-3.)/24.
      B2=1./24.
*
      RINT=D0*CDLOG((Z1-Y0)/(Z0-Y0))
     .    -D1*CDLOG((Y1-Z1)/(Y1-Z0))
     .    -A4/3.*(1./Z1**3-1./Z0**3)
     .    -A3/2.*(1./Z1**2-1./Z0**2)
     .    -A2   *(1./Z1   -1./Z0   )
     .    +A1*CDLOG(Z1/Z0)
     .    +B0   *(Z1   -Z0   )
     .    +B1/2.*(Z1**2-Z0**2)
     .    +B2/3.*(Z1**3-Z0**3)
*
      XGW4=XGW**4
*
* TOTAL WIDTH INCLUDES FLAVOUR & COLOUR FACTORS
      RGT=RMT**3/(RMW*RGW)*XGW4/(8.*PI**3)*DIMAG(RINT)
      RGT=9.*RGT
      RETURN
      END
