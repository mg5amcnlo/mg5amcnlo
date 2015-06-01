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
      include 'run_config.inc'
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc'

      double precision D
      common/to_dj/D
c
c     PARAM_CARD
c
      character*30 param_card_name
      common/to_param_card_name/param_card_name
c
c     local
c     
      integer npara
      character*20 param(maxpara),value(maxpara)
      character*20 ctemp
      integer k,i,l1,l2
      character*132 buff
      real*8 sf1,sf2
      real*8 pb1,pb2
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
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
      data pdfwgt/.false./
c
c
c
      logical gridrun,gridpack
      integer*8          iseed
      common /to_seed/ iseed
c
c----------
c     start
c----------
c
c     read the run_card.dat
c
      include 'run_card.inc'

c     if no matching ensure that no pdfreweight are done
      if (ickkw.eq.0) pdfwgt = .false.

      q2fact(1) = sf1**2      ! fact scale**2 for pdf1
      q2fact(2) = sf2**2      ! fact scale**2 for pdf2     

      if(pb1.ne.0d0.and.lpp(1).eq.0) pol(1)=sign(1+abs(pb1)/100d0,pb1)
      if(pb2.ne.0d0.and.lpp(2).eq.0) pol(2)=sign(1+abs(pb2)/100d0,pb2)

      if(pb1.ne.0.or.pb2.ne.0) write(*,*) 'Setting beam polarization ',
     $     sign((abs(pol(1))-1)*100,pol(1)),
     $     sign((abs(pol(2))-1)*100,pol(2))

c !!! Default behavior changed (MH, Aug. 07) !!!
c If no pdf, read the param_card and use the value from there and
c order of alfas running = 2

      if(lpp(1).ne.0.or.lpp(2).ne.0) then
          write(*,*) 'A PDF is used, so alpha_s(MZ) is going to be modified'
          call setpara(param_card_name)
          asmz=G**2/(16d0*atan(1d0))
          write(*,*) 'Old value of alpha_s from param_card: ',asmz
          call pdfwrap
          write(*,*) 'New value of alpha_s from PDF ',pdlabel,':',asmz
      else
          call setpara(param_card_name)
          asmz=G**2/(16d0*atan(1d0))
          nloop=2
          pdlabel='none'
          write(*,*) 'No PDF is used, alpha_s(MZ) from param_card is used'
          write(*,*) 'Value of alpha_s from param_card: ',asmz
          write(*,*) 'The default order of alpha_s running is fixed to ',nloop
      endif
c !!! end of modification !!!

C     If use_syst, ensure that all variational parameters are 1
c           In principle this should be always the case since the
c           banner.py is expected to correct such wrong run_card.
      if(use_syst)then
         if(scalefact.ne.1)then
            write(*,*) 'Warning: use_syst=T, setting scalefact to 1'
            scalefact=1
         endif
         if(alpsfact.ne.1)then
            write(*,*) 'Warning: use_syst=T, setting alpsfact to 1'
            alpsfact=1
         endif
      endif

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
          idbmup(i)=idup(i,1,1)
        else
          idbmup(i)=lpp(i)
        endif
      enddo
      ebmup(1)=ebeam(1)
      ebmup(2)=ebeam(2)
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
     $   246800,
     $   247000,
     $   244600/


      if(pdfin.eq."lhapdf") then
        write(*,*)'using LHAPDF'
        do i=1,2
           pdfgup(i)=0
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
        write(*,*)'pdf ',pdfin,' not implemented in get_pdfup.'
        write(*,*)'known pdfs are'
        write(*,*) pdflabs
        write(*,*)'using ',pdflabs(12)
        mpdf=numspdf(12)
      endif

      do i=1,2
        pdfgup(i)=0
        pdfsup(i)=mpdf
      enddo

      return
      end
