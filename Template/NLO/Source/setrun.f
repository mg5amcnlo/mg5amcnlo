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
c     parameters
c
      integer maxpara
      parameter (maxpara=1000)
c
c     local
c     
      integer npara
      character*20 param(maxpara),value(maxpara)
c
c     include
c
      include 'genps.inc'
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc'

      double precision D
      common/to_dj/D
c
c     local
c
      character*20 ctemp
      integer k,i,l1,l2
      character*132 buff
C
C     input cuts
C
      include 'cuts.inc'
C
C     BEAM POLARIZATION
C
      REAL*8 POL(2)
      common/to_polarization/ POL
      data POL/1d0,1d0/
c
c     Les Houches init block (for the <init> info)
c
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
c
      include 'nexternal.inc'
      include 'leshouche_decl.inc'
      logical gridrun,gridpack
      integer          iseed
      common /to_seed/ iseed
      integer nevents
      character*7 event_norm
      common /event_normalisation/event_norm
      integer iappl
      common /for_applgrid/ iappl
c
c----------
c     start
c----------
      include 'run_card.inc'
      
c MZ add the possibility to have shower_MC input lowercase
      call to_upper(shower_MC)
C
      call read_leshouche_info(idup_d,mothup_d,icolup_d)

c merging cuts
      xqcut=0d0
      xmtc=0d0
      d=1d0
      
c*********************************************************************
c     Random Number Seed                                             *
c*********************************************************************

        gridrun=.false.
        gridpack=.false.

c************************************************************************     
c     Renormalization and factorization scales                          *
c************************************************************************     
c

c For backward compatibility
      scale = muR_ref_fixed
      q2fact(1) = muF1_ref_fixed**2      ! fact scale**2 for pdf1
      q2fact(2) = muF2_ref_fixed**2      ! fact scale**2 for pdf2     
      scalefact=muR_over_ref
      ellissextonfact=QES_over_ref

c check that the event normalization input is reasoble
      call case_trap2(event_norm)
      if (event_norm(1:7).ne.'average' .and. event_norm(1:3).ne.'sum'
     $     .and. event_norm(1:5).ne.'unity')then
         write (*,*) 'Do not understand the event_norm parameter'/
     &        /' in the run_card.dat. Possible options are'/
     &        /' "average", "sum" or "unity". Current input is: ',
     &        event_norm
         open(unit=26,file='../../error',status='unknown')
         write (26,*) 'Do not understand the event_norm parameter'/
     &        /' in the run_card.dat. Possible options are'/
     &        /' "average", "sum" or "unity". Current input is: ',
     &        event_norm
         
         stop 1
      endif

c info for reweight

      if (ickkw.ne.0 .and. ickkw.ne.4 .and. ickkw.ne.3) then
         write (*,*) 'ickkw parameter not known. ickkw=',ickkw
         stop
      endif
c$$$      ickkw=0
      chcluster=.false.
      ktscheme=1

c !!! Default behavior changed (MH, Aug. 07) !!!
c If no pdf, read the param_card and use the value from there and
c order of alfas running = 2

      if(lpp(1).ne.0.or.lpp(2).ne.0) then
          write(*,*) 'A PDF is used, so alpha_s(MZ) is going to be modified'
          call setpara('param_card.dat')
          asmz=G**2/(16d0*atan(1d0))
          write(*,*) 'Old value of alpha_s from param_card: ',asmz
          call pdfwrap
          write(*,*) 'New value of alpha_s from PDF ',pdlabel,':',asmz
      else
          call setpara('param_card.dat',.true.)
          asmz=G**2/(16d0*atan(1d0))
          nloop=2
          pdlabel='none'
          write(*,*) 'No PDF is used, alpha_s(MZ) from param_card is used'
          write(*,*) 'Value of alpha_s from param_card: ',asmz
          write(*,*) 'The default order of alpha_s running is fixed to ',nloop
      endif
c !!! end of modification !!!

C       Fill common block for Les Houches init info
      do i=1,2
        if(lpp(i).eq.1.or.lpp(i).eq.2) then
          idbmup(i)=2212
        elseif(lpp(i).eq.-1.or.lpp(i).eq.-2) then
          idbmup(i)=-2212
        elseif(lpp(i).eq.3) then
          idbmup(i)=11
        elseif(lpp(i).eq.-3) then
          idbmup(i)=-11
        elseif(lpp(i).eq.0) then
          idbmup(i)=idup_d(1,i,1)
        else
          idbmup(i)=lpp(i)
        endif
        ebmup(i)=ebeam(i)
      enddo
      call get_pdfup(pdlabel,pdfgup,pdfsup,lhaid)

      return
 99   write(*,*) 'error in reading'
      return
      end

C-------------------------------------------------
C   GET_PDFUP
C   Convert MadEvent pdf name to LHAPDF number
C-------------------------------------------------

      subroutine get_pdfup(pdfin,pdfgup,pdfsup,lhaid)
      implicit none

      character*(*) pdfin
      integer mpdf
      integer npdfs,i,pdfgup(2),pdfsup(2),lhaid

      parameter (npdfs=16)
      character*7 pdflabs(npdfs)
      data pdflabs/
     $   'none',
     $   'mrs02nl',
     $   'mrs02nn',
     $   'cteq4_m',
     $   'cteq4_l',
     $   'cteq4_d',
     $   'cteq5_m',
     $   'cteq5_d',
     $   'cteq5_l',
     $   'cteq5m1',
     $   'cteq6_m',
     $   'cteq6_l',
     $   'cteq6l1',     
     $   'nn23lo',
     $   'nn23lo1',
     $   'nn23nlo'/
      integer numspdf(npdfs)
      data numspdf/
     $   00000,
     $   20250,
     $   20270,
     $   19150,
     $   19170,
     $   19160,
     $   19050,
     $   19060,
     $   19070,
     $   19051,
     $   10000,
     $   10041,
     $   10042,
     $   200200,
     $   200400,
     $   244600/


      if(pdfin.eq."lhapdf") then
        write(*,*)'using LHAPDF'
        do i=1,2
           pdfgup(i)=-1
           pdfsup(i)=lhaid
        enddo
        return
      endif

      
      mpdf=-1
      do i=1,npdfs
        if(pdfin(1:len_trim(pdfin)) .eq. pdflabs(i))then
          mpdf=numspdf(i)
        endif
      enddo

      if(mpdf.eq.-1) then
        write(*,*)'ERROR: pdf ',pdfin,' not implemented in get_pdfup.'
        write(*,*)'known pdfs are'
        write(*,*) pdflabs
        open(unit=26,file='../../error',status='unknown')
        write(26,*)'ERROR: pdf ',pdfin,' not implemented in get_pdfup.'
        write(26,*)'known pdfs are'
        write(26,*) pdflabs
        stop 1
c$$$        write(*,*)'using ',pdflabs(12)
c$$$        mpdf=numspdf(12)
      endif

      do i=1,2
        pdfgup(i)=-1
        pdfsup(i)=mpdf
      enddo

      return
      end
