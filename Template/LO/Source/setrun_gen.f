      subroutine setrun
c----------------------------------------------------------------------
c     Sets the run parameters reading them from the run_card.dat
c
c 1. PDF set
c 2. Collider parameters
c 3. cuts
c---------------------------------------------------------------------- 
      implicit none
c
c     include
c
      include 'genps.inc'
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
c
c     local
c
      integer npara
      character*20 param(maxpara),value(maxpara)
      character*20 ctemp
      integer k,i,l1,l2
      character*132 buff
      real*8 sf1,sf2
      integer lp1,lp2
      real*8 eb1,eb2
      real*8 pb1,pb2
C
C     input cuts
C
      include 'cuts.inc'
c
c----------
c     start
c----------
c
c     read the run_card.dat
c
      call load_para(npara,param,value)

c*********************************************************************
c     Jet measure cuts                                               *
c*********************************************************************

      call get_real   (npara,param,value," xqcut ",xqcut,0d0)

c************************************************************************     
c    Collider energy and type                                           *
c************************************************************************     
c     lpp  = -1 (antiproton), 0 (no pdf), 1 (proton)
c     lpp  =  2 (proton emitting a photon without breaking)
c     lpp  =  3 (electron emitting a photon)
c     ebeam= energy of each beam in GeV

      call get_integer(npara,param,value," lpp1 "   ,lp1,1  )
      call get_integer(npara,param,value," lpp2 "   ,lp2,1  )
      call get_real   (npara,param,value," ebeam1 " ,eb1,7d3)
      call get_real   (npara,param,value," ebeam2 " ,eb2,7d3)
     
      lpp(1)=lp1
      lpp(2)=lp2
      ebeam(1)=eb1
      ebeam(2)=eb2

c************************************************************************     
c    Collider pdf                                                       *
c************************************************************************     

      call get_string (npara,param,value," pdlabel ",pdlabel,'cteq6l1')
c
c     if lhapdf is used the following number identifies the set
c
      if(pdlabel.eq.'''lhapdf''')
     $     call get_integer(npara,param,value," lhaid  ",lhaid,10042)

      call pdfwrap

      return
 99   write(*,*) 'error in reading'
      return
      end

